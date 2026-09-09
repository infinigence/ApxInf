// Standalone operator gate. Compile with nvcc -O3 -arch=sm_101 -lcublas.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#define MARLIN_NAMESPACE_NAME apxinf_marlin_probe
#include "../../crates/apxinf-cuda/kernels/marlin/csrc/moe/marlin_moe_wna16/kernel.h"
#include "../../crates/apxinf-cuda/kernels/marlin/csrc/moe/marlin_moe_wna16/marlin_template.h"
#include "../../crates/apxinf-cuda/kernels/marlin/csrc/quantization/gptq_marlin/awq_marlin_repack.cu"
#define CUDA(x) do { auto e=(x); if(e != cudaSuccess) { fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1); } } while(0)
template<class T> T* upload(const std::vector<T>& v) { T* p; CUDA(cudaMalloc(&p,v.size()*sizeof(T))); CUDA(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice)); return p; }
int main() {
  for (int K : {768,2048}) {
    const int N=K==768?2048:1536;
    const int E=128, M=8192;
    std::vector<int> offsets(E+1);int total=0;
    for(int e=0;e<E;++e){int size=(e<32?8:e<64?32:e<96?80:136);total+=size;offsets[e+1]=total;}
    if(total!=M)return 4;
    std::vector<half> a(M*K), dense(E*K*N);
    std::vector<int32_t> q(E*K*N/8), z(E*(K/128)*N/8), tiles;
    std::vector<half> s(E*(K/128)*N);
    uint32_t rng=42;
    auto next=[&]() { rng=rng*1664525+1013904223; return rng; };
    for(auto& x:a) x=__float2half((int(next()%2001)-1000)/1000.f);
    for(auto& x:q) x=next();
    for(auto& x:z) x=next();
    for(auto& x:s) x=__float2half((1+next()%100)/1000.f);
    for(int e=0;e<E;++e) {
      for(int r=offsets[e];r<offsets[e+1];r+=32) { tiles.push_back(e);tiles.push_back(r);tiles.push_back(offsets[e+1]); }
      for(int k=0;k<K;++k) for(int n=0;n<N;++n) {
        int nib=(n%8)/2+(n%2)*4;
        int qi=(e*K+k)*(N/8)+n/8, zi=(e*(K/128)+k/128)*(N/8)+n/8;
        float val=float((uint32_t(q[qi])>>(4*nib))&15)-float((uint32_t(z[zi])>>(4*nib))&15);
        dense[(e*K+k)*N+n]=__float2half(val*__half2float(s[(e*(K/128)+k/128)*N+n]));
      }
    }
    auto da=upload(a);auto dq=upload(q);auto dz=upload(z);auto ds=upload(s);auto dt=upload(tiles);auto db=upload(dense);
    std::vector<half> sp(s.size());
    std::vector<int32_t> zp(z.size(),0), sorted, expert_ids;
    for(int i=0;i<int(s.size());++i)sp[i]=s[(i/64)*64+(i%64)/8+8*(i%8)];
    for(int i=0;i<int(z.size());++i)for(int nib=0;nib<8;++nib){
      int j=i*8+((nib&3)<<1)+(nib>>2);
      int source=(j/64)*64+(j%64)/8+8*(j%8);
      int source_nib=(source%8)/2+(source%2)*4;
      zp[i]|=((uint32_t(z[source/8])>>(source_nib*4))&15)<<(nib*4);
    }
    for(int e=0;e<E;++e)for(int r=offsets[e];r<offsets[e+1];r+=32){
      expert_ids.push_back(e);for(int i=0;i<32;++i)sorted.push_back(r+i<offsets[e+1]?r+i:M);
    }
    auto dsp=upload(sp);auto dzp=upload(zp);auto dsorted=upload(sorted);auto de=upload(expert_ids);
    std::vector<int> count={int(sorted.size())};auto dc=upload(count);
    int32_t* packed;CUDA(cudaMalloc(&packed,q.size()*4));
    for(int e=0;e<E;++e)apxinf_marlin_probe::awq_marlin_repack_kernel<256,4><<<14,256,8192>>>(reinterpret_cast<uint32_t*>(dq)+e*K*N/8,reinterpret_cast<uint32_t*>(packed)+e*K*N/8,K,N);
    CUDA(cudaGetLastError());CUDA(cudaDeviceSynchronize());
    int *locks;int4*tmp;CUDA(cudaMalloc(&locks,1024*1024));CUDA(cudaMemset(locks,0,1024*1024));CUDA(cudaMalloc(&tmp,32*1024*1024));
    half *out,*ref;CUDA(cudaMalloc(&out,M*N*2));CUDA(cudaMalloc(&ref,M*N*2));
    cublasHandle_t handle;cublasCreate(&handle);float one=1,zero=0;
    for(int e=0;e<E;++e) if(cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,N,offsets[e+1]-offsets[e],K,&one,db+e*K*N,CUDA_R_16F,N,da+offsets[e]*K,CUDA_R_16F,K,&zero,ref+offsets[e]*N,CUDA_R_16F,N,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT)!=CUBLAS_STATUS_SUCCESS) return 2;
    for(int mode=0;mode<5;++mode) {
    auto f4=apxinf_marlin_probe::Marlin<half,vllm::kU4.id(),vllm::kFloat16.id(),64,2,8,4,false,4,8,false>;
    auto f2=apxinf_marlin_probe::Marlin<half,vllm::kU4.id(),vllm::kFloat16.id(),64,2,8,4,false,2,8,false>;
    auto fn=mode==0 ? apxinf_marlin_probe::Marlin<half,vllm::kU4.id(),vllm::kFloat16.id(),128,2,16,4,false,4,8,false> : mode==1 ? f4 : f2;
    int shared=mode==0?53248:mode==1?36864:mode==2?20480:mode==3?24576:32768;
    int threads=mode==0?128:64,grid=mode==0?42:mode==1?84:mode==2?112:mode==3?84:56;
    CUDA(cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
    cudaFuncAttributes attr;int resident;CUDA(cudaFuncGetAttributes(&attr,fn));CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,fn,threads,shared));
    auto launch=[&](){fn<<<grid,threads,shared>>>(reinterpret_cast<int4*>(da),reinterpret_cast<int4*>(packed),reinterpret_cast<int4*>(out),tmp,nullptr,reinterpret_cast<int4*>(dsp),nullptr,reinterpret_cast<int4*>(dzp),nullptr,dsorted,de,dc,nullptr,1,false,false,K/128,M,N,K,locks,false,false,true,shared);};
    launch();
    CUDA(cudaGetLastError());CUDA(cudaDeviceSynchronize());
    cudaEvent_t begin,end;CUDA(cudaEventCreate(&begin));CUDA(cudaEventCreate(&end));
    CUDA(cudaEventRecord(begin));
    for(int repeat=0;repeat<20;++repeat)launch();
    CUDA(cudaEventRecord(end));CUDA(cudaEventSynchronize(end));float ms;CUDA(cudaEventElapsedTime(&ms,begin,end));
    printf("mode=%d shared=%d grid=%d regs=%d resident=%d operator_ms=%g ",mode,shared,grid,attr.numRegs,resident,ms/20);CUDA(cudaEventDestroy(begin));CUDA(cudaEventDestroy(end));
    std::vector<half> got(M*N),expected(M*N);
    CUDA(cudaMemcpy(got.data(),out,M*N*2,cudaMemcpyDeviceToHost));CUDA(cudaMemcpy(expected.data(),ref,M*N*2,cudaMemcpyDeviceToHost));
    float maxerr=0;int bad=0;
    for(int i=0;i<M*N;++i) { float x=__half2float(got[i]), y=__half2float(expected[i]);maxerr=fmaxf(maxerr,fabsf(x-y));if(!std::isfinite(x)||fabsf(x-y)>0.02f+0.01f*fabsf(y)) ++bad; }
    printf("K=%d N=%d max_error=%g bad=%d\n",K,N,maxerr,bad);if(bad)return 3;
    }
    for(void* p:{(void*)dsp,(void*)dzp,(void*)dsorted,(void*)de,(void*)dc,(void*)packed,(void*)locks,(void*)tmp,(void*)da,(void*)dq,(void*)dz,(void*)ds,(void*)dt,(void*)db,(void*)out,(void*)ref})CUDA(cudaFree(p));cublasDestroy(handle);
  }
}
