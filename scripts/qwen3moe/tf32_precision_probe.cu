// Precision/performance feasibility only: vendored CUTLASS 3xTF32 GEMM vs
// cuBLAS FP32 and sampled independent FP64 dots. Weights are already dense
// device FP32 here; this timing excludes AWQ dequantization and routing.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"

#define CHECK(x) do { auto e=(x); if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));exit(1);} } while(0)
#define BLAS(x) do { auto e=(x); if(e!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"%s: cuBLAS %d\n",#x,int(e));exit(1);} } while(0)
#define CUT(x) do { auto e=(x); if(e!=cutlass::Status::kSuccess){fprintf(stderr,"%s: CUTLASS %d\n",#x,int(e));exit(1);} } while(0)
using Gemm = cutlass::gemm::device::Gemm<
    float,cutlass::layout::RowMajor,float,cutlass::layout::ColumnMajor,
    float,cutlass::layout::RowMajor,float,cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,cutlass::gemm::GemmShape<64,64,32>,
    cutlass::gemm::GemmShape<32,32,32>,cutlass::gemm::GemmShape<16,8,8>,
    cutlass::epilogue::thread::LinearCombination<float,4,float,float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,3,4,4,
    false,cutlass::arch::OpMultiplyAddFastF32>;

int main() {
  cudaStream_t stream;CHECK(cudaStreamCreate(&stream));
  cublasHandle_t handle;BLAS(cublasCreate(&handle));BLAS(cublasSetStream(handle,stream));
  BLAS(cublasSetMathMode(handle,CUBLAS_PEDANTIC_MATH));
  struct Shape { int m,n,k; };
  for(Shape s:std::vector<Shape>{{32,1536,2048},{128,1536,2048},{128,2048,768},
                                {1024,5120,2048},{1024,2048,4096},{1024,128,2048}}) {
    int m=s.m,n=s.n,k=s.k;uint32_t seed=51;
    auto random=[&](){seed=1664525*seed+1013904223;return seed;};
    std::vector<float> a(size_t(m)*k),b(size_t(n)*k);
    for(auto&x:a)x=float(int(random()%60001)-30000)/27563.f;
    for(int j=0;j<n;++j)for(int g=0;g<k/128;++g) {
      float scale=__half2float(__float2half(.001f+float(random()%1000)/16384.f));
      for(int i=0;i<128;++i)b[size_t(j)*k+g*128+i]=float(int(random()%31)-15)*scale;
    }
    float *da,*db,*dc,*dr;CHECK(cudaMalloc(&da,a.size()*4));CHECK(cudaMalloc(&db,b.size()*4));
    CHECK(cudaMalloc(&dc,size_t(m)*n*4));CHECK(cudaMalloc(&dr,size_t(m)*n*4));
    CHECK(cudaMemcpy(da,a.data(),a.size()*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(db,b.data(),b.size()*4,cudaMemcpyHostToDevice));
    Gemm op;Gemm::Arguments args({m,n,k},{da,k},{db,k},{dc,n},{dc,n},{1,0});
    CUT(op.can_implement(args));size_t bytes=op.get_workspace_size(args);void* workspace=nullptr;
    if(bytes)CHECK(cudaMalloc(&workspace,bytes));CUT(op.initialize(args,workspace,stream));
    float timings[2];
    for(int mode=0;mode<2;++mode) {
      auto launch=[&](){
        if(mode)CUT(op(stream));
        else {float one=1,zero=0;BLAS(cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,n,m,k,&one,db,k,da,k,&zero,dr,n));}
      };
      launch();CHECK(cudaStreamSynchronize(stream));
      cudaGraph_t graph;cudaGraphExec_t exec;CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));launch();
      CHECK(cudaStreamEndCapture(stream,&graph));CHECK(cudaGraphInstantiate(&exec,graph,0));
      for(int i=0;i<5;++i)CHECK(cudaGraphLaunch(exec,stream));
      cudaEvent_t start,end;CHECK(cudaEventCreate(&start));CHECK(cudaEventCreate(&end));
      CHECK(cudaEventRecord(start,stream));for(int i=0;i<30;++i)CHECK(cudaGraphLaunch(exec,stream));
      CHECK(cudaEventRecord(end,stream));CHECK(cudaEventSynchronize(end));CHECK(cudaEventElapsedTime(&timings[mode],start,end));timings[mode]/=30;
      CHECK(cudaEventDestroy(start));CHECK(cudaEventDestroy(end));CHECK(cudaGraphExecDestroy(exec));CHECK(cudaGraphDestroy(graph));
    }
    std::vector<float> got(size_t(m)*n),ref(got.size());
    CHECK(cudaMemcpy(got.data(),dc,got.size()*4,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(ref.data(),dr,ref.size()*4,cudaMemcpyDeviceToHost));
    double delta=0,oracle_error=0,baseline_error=0;
    for(size_t i=0;i<got.size();++i) {
      if(!std::isfinite(got[i])||!std::isfinite(ref[i]))return 2;
      delta=std::max(delta,std::abs(double(got[i])-ref[i]));
    }
    for(int row=0;row<m;row+=std::max(m/7,1))for(int col=0;col<n;col+=std::max(n/13,1)) {
      double expected=0;for(int i=0;i<k;++i)expected+=double(a[size_t(row)*k+i])*b[size_t(col)*k+i];
      double error=std::abs(got[size_t(row)*n+col]-expected);
      oracle_error=std::max(oracle_error,error);baseline_error=std::max(baseline_error,std::abs(ref[size_t(row)*n+col]-expected));
      if(error>2e-4+2e-5*std::abs(expected)){fprintf(stderr,"oracle failure %d %d %g %.12g\n",row,col,got[size_t(row)*n+col],expected);return 3;}
    }
    printf("M=%d N=%d K=%d fp32_ms=%g tf32x3_ms=%g max_delta=%g fp64_error=%g baseline_fp64_error=%g oracle=pass\n",m,n,k,timings[0],timings[1],delta,oracle_error,baseline_error);
    for(void*p:{(void*)da,(void*)db,(void*)dc,(void*)dr,workspace})if(p)CHECK(cudaFree(p));
  }
  BLAS(cublasDestroy(handle));CHECK(cudaStreamDestroy(stream));return 0;
}
