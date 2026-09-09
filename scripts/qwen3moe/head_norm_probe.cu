// Check and time RMSNorm -> BF16 high/low input representation for the head.
// Uses the production kernel; no head weights are modified by this operation.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "../../crates/apxinf-cuda/kernels/custom/math.cuh"
#include "../../crates/apxinf-cuda/kernels/custom/reduction.cuh"
#include "../../crates/apxinf-cuda/kernels/custom/normalization.cuh"
#define CHECK(x) do { auto e=(x); if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));exit(1);} } while(0)

__global__ void split_normalized(const float* input, __nv_bfloat16* output, int cols, int rows) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<rows*cols) {
    int row=i/cols, col=i%cols;
    auto high=__float2bfloat16(input[i]);
    output[2*row*cols+col]=high;
    output[(2*row+1)*cols+col]=__float2bfloat16(input[i]-__bfloat162float(high));
  }
}

int main() {
  cudaStream_t stream; CHECK(cudaStreamCreate(&stream));
  for(int rows:{1,3,17}) for(int cols:{257,2048,4096}) {
    const int count=rows*cols;
    std::vector<__nv_bfloat16> input(count),weight(cols);
    uint32_t seed=93;
    auto random=[&](){seed=1664525*seed+1013904223;return float(int(seed>>16)-32768)/16384.f;};
    for(auto&v:input)v=__float2bfloat16(random());
    for(auto&v:weight)v=__float2bfloat16(1+random()/4);
    __nv_bfloat16 *dx,*dw,*original,*composed,*fused; float* exact;
    CHECK(cudaMalloc(&dx,count*2)); CHECK(cudaMalloc(&dw,cols*2));
    CHECK(cudaMalloc(&original,count*2)); CHECK(cudaMalloc(&exact,count*4));
    CHECK(cudaMalloc(&composed,count*4)); CHECK(cudaMalloc(&fused,count*4));
    CHECK(cudaMemcpy(dx,input.data(),count*2,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dw,weight.data(),cols*2,cudaMemcpyHostToDevice));
    rms_norm_bf16_kernel<><<<rows,256,cols*4,stream>>>(dx,dw,original,cols,rows,1e-6f);
    float timings[2];
    for(int mode=0;mode<2;++mode) {
      auto launch=[&](){
        if(mode)rms_norm_bf16_kernel<__nv_bfloat16,__nv_bfloat16,true><<<rows,256,cols*4,stream>>>(dx,dw,fused,cols,rows,1e-6f);
        else {
          rms_norm_bf16_kernel<__nv_bfloat16,float><<<rows,256,cols*4,stream>>>(dx,dw,exact,cols,rows,1e-6f);
          split_normalized<<<(count+255)/256,256,0,stream>>>(exact,composed,cols,rows);
        }
      };
      launch(); CHECK(cudaStreamSynchronize(stream));
      cudaGraph_t graph; cudaGraphExec_t exec;
      CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));launch();CHECK(cudaStreamEndCapture(stream,&graph));
      CHECK(cudaGraphInstantiate(&exec,graph,0));
      for(int i=0;i<20;++i)CHECK(cudaGraphLaunch(exec,stream));
      cudaEvent_t start,end;CHECK(cudaEventCreate(&start));CHECK(cudaEventCreate(&end));
      CHECK(cudaEventRecord(start,stream));for(int i=0;i<200;++i)CHECK(cudaGraphLaunch(exec,stream));CHECK(cudaEventRecord(end,stream));CHECK(cudaEventSynchronize(end));
      CHECK(cudaEventElapsedTime(&timings[mode],start,end));timings[mode]*=5;
      CHECK(cudaEventDestroy(start));CHECK(cudaEventDestroy(end));CHECK(cudaGraphExecDestroy(exec));CHECK(cudaGraphDestroy(graph));
    }
    std::vector<__nv_bfloat16> a(2*count),b(2*count),base(count);
    CHECK(cudaMemcpy(a.data(),composed,count*4,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(b.data(),fused,count*4,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(base.data(),original,count*2,cudaMemcpyDeviceToHost));
    if(memcmp(a.data(),b.data(),count*4)){fprintf(stderr,"composed/fused mismatch\n");return 2;}
    double max_error=0,base_error=0;
    for(int row=0;row<rows;++row) {
      double squares=0;for(int c=0;c<cols;++c){double x=__bfloat162float(input[row*cols+c]);squares+=x*x;}
      double inv=1/std::sqrt(squares/cols+1e-6);
      for(int c=0;c<cols;++c) {
        auto hi=b[2*row*cols+c],lo=b[(2*row+1)*cols+c];
        if(__bfloat16_as_ushort(hi)!=__bfloat16_as_ushort(base[row*cols+c]))return 3;
        double expected=__bfloat162float(input[row*cols+c])*inv*__bfloat162float(weight[c]);
        double got=double(__bfloat162float(hi))+__bfloat162float(lo);
        max_error=std::max(max_error,std::abs(got-expected));
        base_error=std::max(base_error,std::abs(__bfloat162float(hi)-expected));
        if(!std::isfinite(got)||std::abs(got-expected)>1e-5*std::max(1.,std::abs(expected)))return 4;
      }
    }
    printf("rows=%d cols=%d composed_us=%g fused_us=%g byte_checks=pass fp64_error=%g original_bf16_error=%g\n",rows,cols,timings[0],timings[1],max_error,base_error);
    for(void*p:{(void*)dx,(void*)dw,(void*)original,(void*)exact,(void*)composed,(void*)fused})CHECK(cudaFree(p));
  }
  CHECK(cudaStreamDestroy(stream)); return 0;
}
