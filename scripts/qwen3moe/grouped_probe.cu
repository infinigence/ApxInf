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
#include "../../crates/apxinf-cuda/kernels/custom/grouped_gemm.cuh"
#define CUDA(x) do { auto e=(x); if(e != cudaSuccess) { fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1); } } while(0)
template<class T> T* upload(const std::vector<T>& v) { T* p; CUDA(cudaMalloc(&p,v.size()*sizeof(T))); CUDA(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice)); return p; }
int main() {
  for (int K : {128,768,2048}) for (int N : {136,1536,2048}) {
    const int E=3, M=151;
    const int offsets[]={0,1,66,151};
    std::vector<__nv_bfloat16> a(M*K), dense(E*K*N);
    std::vector<int32_t> q(E*K*N/8), z(E*(K/128)*N/8), tiles;
    std::vector<half> s(E*(K/128)*N);
    uint32_t rng=42;
    auto next=[&]() { rng=rng*1664525+1013904223; return rng; };
    for(auto& x:a) x=__float2bfloat16((int(next()%2001)-1000)/1000.f);
    for(auto& x:q) x=next();
    for(auto& x:z) x=next();
    for(auto& x:s) x=__float2half((1+next()%100)/1000.f);
    for(int e=0;e<E;++e) {
      for(int r=offsets[e];r<offsets[e+1];r+=64) { tiles.push_back(e);tiles.push_back(r);tiles.push_back(offsets[e+1]); }
      for(int k=0;k<K;++k) for(int n=0;n<N;++n) {
        int nib=(n%8)/2+(n%2)*4;
        int qi=(e*K+k)*(N/8)+n/8, zi=(e*(K/128)+k/128)*(N/8)+n/8;
        float val=float((uint32_t(q[qi])>>(4*nib))&15)-float((uint32_t(z[zi])>>(4*nib))&15);
        dense[(e*K+k)*N+n]=__float2bfloat16(val*__half2float(s[(e*(K/128)+k/128)*N+n]));
      }
    }
    auto da=upload(a);auto dq=upload(q);auto dz=upload(z);auto ds=upload(s);auto dt=upload(tiles);auto db=upload(dense);
    __nv_bfloat16 *out,*ref;CUDA(cudaMalloc(&out,M*N*2));CUDA(cudaMalloc(&ref,M*N*2));
    cublasHandle_t handle;cublasCreate(&handle);float one=1,zero=0;
    for(int e=0;e<E;++e) if(cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,N,offsets[e+1]-offsets[e],K,&one,db+e*K*N,CUDA_R_16BF,N,da+offsets[e]*K,CUDA_R_16BF,K,&zero,ref+offsets[e]*N,CUDA_R_16BF,N,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT)!=CUBLAS_STATUS_SUCCESS) return 2;
    w4a16_grouped_bf16_kernel<<<dim3(tiles.size()/3,(N+127)/128),128>>>(da,dq,dz,ds,dt,out,K,N,128,K*N/8,(K/128)*N/8,(K/128)*N);
    CUDA(cudaGetLastError());CUDA(cudaDeviceSynchronize());
    cudaEvent_t begin,end;CUDA(cudaEventCreate(&begin));CUDA(cudaEventCreate(&end));
    CUDA(cudaEventRecord(begin));
    for(int repeat=0;repeat<20;++repeat)
      w4a16_grouped_bf16_kernel<<<dim3(tiles.size()/3,(N+127)/128),128>>>(da,dq,dz,ds,dt,out,K,N,128,K*N/8,(K/128)*N/8,(K/128)*N);
    CUDA(cudaEventRecord(end));CUDA(cudaEventSynchronize(end));float ms;CUDA(cudaEventElapsedTime(&ms,begin,end));
    printf("operator_ms=%g ",ms/20);CUDA(cudaEventDestroy(begin));CUDA(cudaEventDestroy(end));
    std::vector<__nv_bfloat16> got(M*N),expected(M*N);
    CUDA(cudaMemcpy(got.data(),out,M*N*2,cudaMemcpyDeviceToHost));CUDA(cudaMemcpy(expected.data(),ref,M*N*2,cudaMemcpyDeviceToHost));
    float maxerr=0;int bad=0;
    for(int i=0;i<M*N;++i) { float x=__bfloat162float(got[i]), y=__bfloat162float(expected[i]);maxerr=fmaxf(maxerr,fabsf(x-y));if(!std::isfinite(x)||fabsf(x-y)>0.02f+0.01f*fabsf(y)) ++bad; }
    printf("K=%d N=%d max_error=%g bad=%d\n",K,N,maxerr,bad);if(bad)return 3;
    for(void* p:{(void*)da,(void*)dq,(void*)dz,(void*)ds,(void*)dt,(void*)db,(void*)out,(void*)ref})CUDA(cudaFree(p));cublasDestroy(handle);
  }
}
