// Preserve BF16 weights; represent one FP32 input with BF16 high/low rows.
// Compare a two-row cuBLAS GEMM plus FP32 sum with the one-row BF16 head.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CHECK(x) do { auto e=(x); if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));exit(1);} } while(0)
#define BLAS(x) do { auto e=(x); if(e!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"%s: cuBLAS %d\n",#x,int(e));exit(1);} } while(0)
__global__ void split_input(const float* input,__nv_bfloat16* output,int k) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<k) {
    auto hi=__float2bfloat16(input[i]);
    output[i]=hi;output[k+i]=__float2bfloat16(input[i]-__bfloat162float(hi));
  }
}
__global__ void sum_rows(const float* rows,float* output,int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n)output[i]=rows[i]+rows[n+i];
}
int main() {
  cudaStream_t stream;CHECK(cudaStreamCreate(&stream));
  cublasHandle_t handle;BLAS(cublasCreate(&handle));BLAS(cublasSetStream(handle,stream));
  for(int n:{128,513,151936}) {
    const int k=2048;
    std::vector<float> input(k);
    std::vector<__nv_bfloat16> weights(size_t(n)*k), expected_split(2*k);
    uint32_t seed=19;
    auto random=[&](){seed=1664525*seed+1013904223;return seed;};
    for(auto& x:input)x=float(int(random()%60001)-30000)/12347.f;
    for(auto& x:weights)x=__float2bfloat16(float(int(random()%2001)-1000)/2048.f);
    for(int i=0;i<k;++i){expected_split[i]=__float2bfloat16(input[i]);expected_split[k+i]=__float2bfloat16(input[i]-__bfloat162float(expected_split[i]));}
    float *dx,*rows,*out;__nv_bfloat16 *parts,*dw;
    CHECK(cudaMalloc(&dx,k*4));CHECK(cudaMalloc(&parts,2*k*2));
    CHECK(cudaMalloc(&dw,weights.size()*2));CHECK(cudaMalloc(&rows,2*n*4));CHECK(cudaMalloc(&out,n*4));
    CHECK(cudaMemcpy(dx,input.data(),k*4,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dw,weights.data(),weights.size()*2,cudaMemcpyHostToDevice));
    auto gemm=[&](int m){float one=1,zero=0;BLAS(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,n,m,k,&one,dw,CUDA_R_16BF,k,parts,CUDA_R_16BF,k,&zero,rows,CUDA_R_32F,n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));};
    split_input<<<8,256,0,stream>>>(dx,parts,k);CHECK(cudaStreamSynchronize(stream));
    std::vector<__nv_bfloat16> got_split(2*k);CHECK(cudaMemcpy(got_split.data(),parts,2*k*2,cudaMemcpyDeviceToHost));
    for(int i=0;i<2*k;++i)if(__bfloat16_as_ushort(got_split[i])!=__bfloat16_as_ushort(expected_split[i]))return 2;
    for(int mode=0;mode<2;++mode) {
      auto launch=[&](){
        if(mode)split_input<<<8,256,0,stream>>>(dx,parts,k);
        gemm(mode?2:1);
        if(mode)sum_rows<<<(n+255)/256,256,0,stream>>>(rows,out,n);
      };
      launch();CHECK(cudaStreamSynchronize(stream));
      cudaGraph_t graph;cudaGraphExec_t executable;
      CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));launch();CHECK(cudaStreamEndCapture(stream,&graph));CHECK(cudaGraphInstantiate(&executable,graph,0));
      cudaEvent_t start,end;CHECK(cudaEventCreate(&start));CHECK(cudaEventCreate(&end));
      CHECK(cudaEventRecord(start,stream));for(int i=0;i<40;++i)CHECK(cudaGraphLaunch(executable,stream));CHECK(cudaEventRecord(end,stream));CHECK(cudaEventSynchronize(end));
      float ms;CHECK(cudaEventElapsedTime(&ms,start,end));
      std::vector<float> got(n);CHECK(cudaMemcpy(got.data(),mode?out:rows,n*4,cudaMemcpyDeviceToHost));
      for(auto x:got)if(!std::isfinite(x))return 3;
      double original_error=0,represented_error=0;
      for(int col=0;col<n;col+=std::max(n/67,1)) {
        double original=0,represented=0;
        for(int j=0;j<k;++j) {
          double w=__bfloat162float(weights[size_t(col)*k+j]);
          double h=__bfloat162float(expected_split[j]);
          if(mode)h+=__bfloat162float(expected_split[k+j]);
          original+=w*input[j];represented+=w*h;
        }
        original_error=std::max(original_error,std::abs(got[col]-original));
        represented_error=std::max(represented_error,std::abs(got[col]-represented));
        if(std::abs(got[col]-represented)>2e-4+2e-5*std::abs(represented))return 4;
      }
      printf("N=%d mode=%d graph_ms=%g error_vs_fp32_input=%g arithmetic_error=%g finite=1 split_bytes_equal=1\n",n,mode,ms/40,original_error,represented_error);
      CHECK(cudaEventDestroy(start));CHECK(cudaEventDestroy(end));CHECK(cudaGraphExecDestroy(executable));CHECK(cudaGraphDestroy(graph));
    }
    for(void* p:{(void*)dx,(void*)parts,(void*)dw,(void*)rows,(void*)out})CHECK(cudaFree(p));
  }
  BLAS(cublasDestroy(handle));CHECK(cudaStreamDestroy(stream));return 0;
}
