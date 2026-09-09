// Actual decode residual/RMSNorm -> blocked GEMV dependency, eager and captured.
// nvcc -std=c++17 -O3 -arch=sm_101 scripts/qwen3moe/pdl_gemv_probe.cu -o pdl_gemv_probe
// No model runtime enables these PDL specializations until this gate is measured.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include "../../crates/apxinf-cuda/kernels/custom/reduction.cuh"
#include "../../crates/apxinf-cuda/kernels/custom/quantization.cuh"
#include "../../crates/apxinf-cuda/kernels/custom/w4a16_blocked.cuh"
#include "../../crates/apxinf-cuda/kernels/custom/w4a16_magic.cuh"
#include "../../crates/apxinf-cuda/kernels/custom/decode_epilogues.cuh"

#define CHECK(x) do { auto e=(x); if(e!=cudaSuccess) { \
  fprintf(stderr,"%s:%d %s: %s\n",__FILE__,__LINE__,#x,cudaGetErrorString(e));std::exit(1); \
} } while(0)
template<class T> T* upload(const std::vector<T>& v) {
  T* p;CHECK(cudaMalloc(&p,v.size()*sizeof(T)));
  CHECK(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return p;
}

int main() {
  cudaStream_t stream;CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
  struct Geometry {int k,n,splits;};
  for(auto shape:{Geometry{2048,5120,4},Geometry{4096,2048,8},Geometry{2048,5120,3}}) {
    const int k=shape.k,n=shape.n,splits=shape.splits,pc=n/8;
    const int rp=((k+splits-1)/splits+7)/8*8,count=8,pairs=32;
    std::vector<int32_t> q(int64_t(k)*pc),z(int64_t(k/128)*pc);
    std::vector<half> scales(int64_t(k/128)*n);
    std::vector<__nv_bfloat16> residual(k),weight(k);
    std::vector<float> delta(k*count);
    uint32_t seed=42;auto next=[&](){return seed=seed*1664525+1013904223;};
    for(auto& v:q)v=next();for(auto& v:z)v=next();
    for(auto& v:scales)v=__float2half((1+next()%100)/1000.f);
    for(auto& v:weight)v=__float2bfloat16(0.5f+(next()%1000)/1000.f);
    for(auto& v:residual)v=__float2bfloat16((int(next()%2001)-1000)/1000.f);
    for(auto& v:delta)v=(int(next()%2001)-1000)/100000.f;
    auto dq=upload(q),dz=upload(z);auto ds=upload(scales);
    auto dx=upload(residual),dw=upload(weight);auto dp=upload(delta);
    __nv_bfloat16* normed;float* out;int32_t* packed;
    CHECK(cudaMalloc(&normed,k*2));CHECK(cudaMalloc(&out,int64_t(splits)*n*4));
    CHECK(cudaMalloc(&packed,q.size()*4));
    w4a16_blocked_repack_kernel<<<256,256,0,stream>>>(dq,reinterpret_cast<int4*>(packed),k,pc,1);
    CHECK(cudaStreamSynchronize(stream));
    cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed=1;
    auto enqueue=[&](bool pdl) {
      cudaLaunchConfig_t cfg{};cfg.stream=stream;cfg.attrs=pdl?&attr:nullptr;cfg.numAttrs=pdl?1:0;
      cfg.gridDim=dim3(1);cfg.blockDim=dim3(1024);cfg.dynamicSmemBytes=k*4;
      auto norm=pdl?partial_residual_rms_bf16_kernel<true,true>:partial_residual_rms_bf16_kernel<true,false>;
      CHECK(cudaLaunchKernelEx(&cfg,norm,dx,dp,dw,normed,uint32_t(k),uint32_t(1),uint32_t(count),1e-6f));
      cfg.gridDim=dim3(n/256,splits);cfg.blockDim=dim3(256);cfg.dynamicSmemBytes=(rp+8*256)*4;
      auto gemv=pdl?w4a16_gemv_magic_kernel<true,true>:w4a16_gemv_magic_kernel<true,false>;
      CHECK(cudaLaunchKernelEx(&cfg,gemv,normed,int64_t(0),packed,dz,ds,
          static_cast<const int32_t*>(nullptr),int64_t(k)*pc,int64_t(k/128)*pc,
          int64_t(k/128)*n,static_cast<const float*>(nullptr),out,k,pc,128,rp));
    };
    for(bool capture:{false,true}) {
      std::vector<float> reference;
      for(bool pdl:{false,true}) {
        cudaGraph_t graph{};cudaGraphExec_t executable{};
        if(capture) {
          CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
          for(int i=0;i<pairs;++i)enqueue(pdl);
          CHECK(cudaStreamEndCapture(stream,&graph));
          CHECK(cudaGraphInstantiate(&executable,graph,0));
        }
        auto run=[&](){if(capture)CHECK(cudaGraphLaunch(executable,stream));else for(int i=0;i<pairs;++i)enqueue(pdl);};
        CHECK(cudaMemcpyAsync(dx,residual.data(),k*2,cudaMemcpyHostToDevice,stream));
        run();CHECK(cudaStreamSynchronize(stream));
        std::vector<float> got(int64_t(splits)*n);
        CHECK(cudaMemcpy(got.data(),out,got.size()*4,cudaMemcpyDeviceToHost));
        for(float v:got)if(!std::isfinite(v)){fprintf(stderr,"nonfinite GEMV partial\n");return 3;}
        if(!pdl)reference=got;
        else if(std::memcmp(reference.data(),got.data(),got.size()*4)) {
          fprintf(stderr,"PDL changed GEMV partials: K=%d N=%d splits=%d capture=%d\n",k,n,splits,capture);
          return 2;
        }
        cudaEvent_t begin,end;CHECK(cudaEventCreate(&begin));CHECK(cudaEventCreate(&end));
        CHECK(cudaEventRecord(begin,stream));for(int repeat=0;repeat<20;++repeat)run();
        CHECK(cudaEventRecord(end,stream));CHECK(cudaEventSynchronize(end));
        float ms;CHECK(cudaEventElapsedTime(&ms,begin,end));
        printf("K=%d N=%d splits=%d capture=%d pdl=%d pair_us=%.4f bit_equal=%d\n",
            k,n,splits,capture,pdl,ms*1000/(pairs*20),pdl?1:-1);
        CHECK(cudaEventDestroy(begin));CHECK(cudaEventDestroy(end));
        if(capture){CHECK(cudaGraphExecDestroy(executable));CHECK(cudaGraphDestroy(graph));}
      }
    }
    for(void* p:{(void*)dq,(void*)dz,(void*)ds,(void*)dx,(void*)dw,(void*)dp,
        (void*)normed,(void*)out,(void*)packed})CHECK(cudaFree(p));
  }
  CHECK(cudaStreamDestroy(stream));return 0;
}
