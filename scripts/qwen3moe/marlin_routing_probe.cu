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
#include <cstring>
#define MARLIN_NAMESPACE_NAME apxinf_marlin_probe
#include "../../crates/apxinf-cuda/kernels/marlin/csrc/moe/marlin_moe_wna16/kernel.h"
#include "../../crates/apxinf-cuda/kernels/marlin/csrc/moe/marlin_moe_wna16/marlin_template.h"
#include "../../crates/apxinf-cuda/kernels/marlin/csrc/quantization/gptq_marlin/awq_marlin_repack.cu"
#include <marlin_routing_probe.generated.cuh>
#define CUDA(x) do { auto e=(x); if(e != cudaSuccess) { fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1); } } while(0)
template<class T> T* upload(const std::vector<T>& v) { T* p; CUDA(cudaMalloc(&p,v.size()*sizeof(T))); CUDA(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice)); return p; }
template<class T> struct Mapped {
  T* host; T* device;
  explicit Mapped(size_t count) {
    CUDA(cudaHostAlloc(&host,count*sizeof(T),cudaHostAllocMapped));
    CUDA(cudaHostGetDevicePointer(&device,host,0));
  }
  ~Mapped() { CUDA(cudaFreeHost(host)); }
};
int main() {
  for (int M : {1024,8192}) for (int layout : {0,1,2}) for (int K : {768,2048}) {
    const int N=K==768?2048:1536;
    const int E=128, topk=K==2048?8:1;
    std::vector<int> offsets(E+1);int total=0;
    for(int e=0;e<E;++e){int size=layout==0?M/E:layout==1?(e<32?8:e<64?32:e<96?80:136)/(8192/M):(e<96?0:M/32);total+=size;offsets[e+1]=total;}
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
      for(int r=offsets[e];r<offsets[e+1];r+=64) { tiles.push_back(e);tiles.push_back(r);tiles.push_back(offsets[e+1]); }
      for(int k=0;k<K;++k) for(int n=0;n<N;++n) {
        int nib=(n%8)/2+(n%2)*4;
        int qi=(e*K+k)*(N/8)+n/8, zi=(e*(K/128)+k/128)*(N/8)+n/8;
        float val=float((uint32_t(q[qi])>>(4*nib))&15)-float((uint32_t(z[zi])>>(4*nib))&15);
        dense[(e*K+k)*N+n]=__float2half(val*__half2float(s[(e*(K/128)+k/128)*N+n]));
      }
    }
    std::vector<half> gathered(a.size());
    for(int row=0;row<M;++row)memcpy(gathered.data()+row*K,a.data()+(row/topk)*K,K*sizeof(half));
    auto dga=upload(gathered);auto da=upload(a);auto dq=upload(q);auto dz=upload(z);auto ds=upload(s);auto dt=upload(tiles);auto db=upload(dense);
    std::vector<half> sp(s.size());
    std::vector<int32_t> zp(z.size(),0), sorted, expert_ids;
    for(int i=0;i<int(s.size());++i)sp[i]=s[(i/64)*64+(i%64)/8+8*(i%8)];
    for(int i=0;i<int(z.size());++i)for(int nib=0;nib<8;++nib){
      int j=i*8+((nib&3)<<1)+(nib>>2);
      int source=(j/64)*64+(j%64)/8+8*(j%8);
      int source_nib=(source%8)/2+(source%2)*4;
      zp[i]|=((uint32_t(z[source/8])>>(source_nib*4))&15)<<(nib*4);
    }
    for(int e=0;e<E;++e)for(int r=offsets[e];r<offsets[e+1];r+=64){
      expert_ids.push_back(e);for(int i=0;i<64;++i)sorted.push_back(r+i<offsets[e+1]?r+i:M);
    }
    Mapped<half> msp(sp.size());Mapped<int32_t> mzp(zp.size());
    memcpy(msp.host,sp.data(),sp.size()*sizeof(half));memcpy(mzp.host,zp.data(),zp.size()*4);
    auto dsp=msp.device;auto dzp=mzp.device;auto dsorted=upload(sorted);auto de=upload(expert_ids);
    std::vector<int> count={int(sorted.size())};auto dc=upload(count);
    Mapped<int32_t> mq(q.size());auto packed=mq.device;
    for(int e=0;e<E;++e)apxinf_marlin_probe::awq_marlin_repack_kernel<256,4><<<14,256,8192>>>(reinterpret_cast<uint32_t*>(dq)+e*K*N/8,reinterpret_cast<uint32_t*>(packed)+e*K*N/8,K,N);
    CUDA(cudaGetLastError());CUDA(cudaDeviceSynchronize());
    int *locks;int4*tmp;CUDA(cudaMalloc(&locks,1024*1024));CUDA(cudaMemset(locks,0,1024*1024));CUDA(cudaMalloc(&tmp,32*1024*1024));
    half *out,*ref;CUDA(cudaMalloc(&out,M*N*2));CUDA(cudaMalloc(&ref,M*N*2));
    cublasHandle_t handle;cublasCreate(&handle);float one=1,zero=0;
    for(int e=0;e<E;++e) if(offsets[e+1]>offsets[e] && cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,N,offsets[e+1]-offsets[e],K,&one,db+e*K*N,CUDA_R_16F,N,dga+offsets[e]*K,CUDA_R_16F,K,&zero,ref+offsets[e]*N,CUDA_R_16F,N,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT)!=CUBLAS_STATUS_SUCCESS) return 2;
    std::vector<half> baseline(M*N);
    for(int mode=0;mode<5;++mode) {
    auto ordinary=apxinf_marlin_probe::Marlin<half,vllm::kU4.id(),vllm::kFloat16.id(),128,4,8,4,false,4,8,false>;
    auto constant=apxinf_marlin_probe::Marlin<half,vllm::kU4.id(),vllm::kFloat16.id(),128,4,8,4,false,4,8,false,false,true>;
    decltype(ordinary) variants[]={ordinary,constant,
      apxinf_marlin_probe::MarlinRouting<half,vllm::kU4.id(),vllm::kFloat16.id(),128,4,8,4,false,4,8,false,false,true,1>,
      apxinf_marlin_probe::MarlinRouting<half,vllm::kU4.id(),vllm::kFloat16.id(),128,4,8,4,false,4,8,false,false,true,2>,
      apxinf_marlin_probe::MarlinRouting<half,vllm::kU4.id(),vllm::kFloat16.id(),128,4,8,4,false,4,8,false,false,true,3>};
    auto fn=variants[mode];
    int shared=65536,threads=128,grid=28;
    CUDA(cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
    cudaFuncAttributes attr;int resident;CUDA(cudaFuncGetAttributes(&attr,fn));CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,fn,threads,shared));
    auto launch=[&](){fn<<<grid,threads,shared>>>(reinterpret_cast<int4*>(da),reinterpret_cast<int4*>(packed),reinterpret_cast<int4*>(out),tmp,nullptr,reinterpret_cast<int4*>(dsp),nullptr,reinterpret_cast<int4*>(dzp),nullptr,dsorted,de,dc,nullptr,topk,false,false,K/128,M/topk,N,K,locks,false,false,true,shared);};
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
    int byte_bad=0;
    if(mode==0)baseline=got;
    else for(size_t i=0;i<got.size();++i)byte_bad+=memcmp(&got[i],&baseline[i],sizeof(half))!=0;
    printf("M=%d layout=%d topk=%d K=%d N=%d max_error=%g bad=%d bit_mismatches=%d mapped_weights=1\n",M,layout,topk,K,N,maxerr,bad,byte_bad);if(bad||byte_bad)return 3;
    }
    for(void* p:{(void*)dsorted,(void*)de,(void*)dc,(void*)locks,(void*)tmp,(void*)dga,(void*)da,(void*)dq,(void*)dz,(void*)ds,(void*)dt,(void*)db,(void*)out,(void*)ref})CUDA(cudaFree(p));cublasDestroy(handle);
  }
}
