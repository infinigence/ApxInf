// Precision feasibility: scaled FP16 high/low products vs cuBLAS FP32.
// Includes activation splitting; weights are already split on device before
// timing. Excludes AWQ dequantization and routing; not runtime acceptance.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CHECK(x) do { auto e=(x); if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));exit(1);} } while(0)
#define BLAS(x) do { auto e=(x); if(e!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"%s: cuBLAS %d\n",#x,int(e));exit(1);} } while(0)
__global__ void split_scaled_f16(const float* input,half* parts,int count) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<count){half high=__float2half(input[i]);parts[i]=high;parts[count+i]=__float2half((input[i]-__half2float(high))*4096.f);}
}

int main(int argc,char** argv) {
  const bool verify_only=argc==2 && !std::strcmp(argv[1],"--verify-only");
  if(argc>1 && !verify_only)return 2;
  cudaStream_t stream;CHECK(cudaStreamCreate(&stream));
  cublasHandle_t handle;BLAS(cublasCreate(&handle));BLAS(cublasSetStream(handle,stream));
  BLAS(cublasSetMathMode(handle,CUBLAS_DEFAULT_MATH));
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
    half *ap,*bp;CHECK(cudaMalloc(&ap,a.size()*4));CHECK(cudaMalloc(&bp,b.size()*4));
    split_scaled_f16<<<(b.size()+255)/256,256,0,stream>>>(db,bp,int(b.size()));
    CHECK(cudaStreamSynchronize(stream));
    float timings[3]={};
    std::vector<float> outputs[3];
    for(int mode=0;mode<3;++mode) {
      auto launch=[&](){
        if(mode) {
          split_scaled_f16<<<(a.size()+255)/256,256,0,stream>>>(da,ap,int(a.size()));
          auto product=[&](const half* pa,const half* pb,float alpha,float beta){BLAS(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,n,m,k,&alpha,pb,CUDA_R_16F,k,pa,CUDA_R_16F,k,&beta,dc,CUDA_R_32F,n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));};
          product(ap,bp,1.f,0.f);
          product(ap,bp+b.size(),1.f/4096.f,1.f);
          product(ap+a.size(),bp,1.f/4096.f,1.f);
          if(mode==2)product(ap+a.size(),bp+b.size(),1.f/(4096.f*4096.f),1.f);
        } else {
          float one=1,zero=0;BLAS(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,n,m,k,&one,db,CUDA_R_32F,k,da,CUDA_R_32F,k,&zero,dr,CUDA_R_32F,n,CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT));
        }
      };
      launch();CHECK(cudaStreamSynchronize(stream));
      if(!verify_only) {
      cudaGraph_t graph;cudaGraphExec_t exec;CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));launch();
      CHECK(cudaStreamEndCapture(stream,&graph));CHECK(cudaGraphInstantiate(&exec,graph,0));
      for(int i=0;i<5;++i)CHECK(cudaGraphLaunch(exec,stream));
      cudaEvent_t start,end;CHECK(cudaEventCreate(&start));CHECK(cudaEventCreate(&end));
      CHECK(cudaEventRecord(start,stream));for(int i=0;i<30;++i)CHECK(cudaGraphLaunch(exec,stream));
      CHECK(cudaEventRecord(end,stream));CHECK(cudaEventSynchronize(end));CHECK(cudaEventElapsedTime(&timings[mode],start,end));timings[mode]/=30;
      CHECK(cudaEventDestroy(start));CHECK(cudaEventDestroy(end));CHECK(cudaGraphExecDestroy(exec));CHECK(cudaGraphDestroy(graph));
      }
      outputs[mode].resize(size_t(m)*n);CHECK(cudaMemcpy(outputs[mode].data(),mode?dc:dr,outputs[mode].size()*4,cudaMemcpyDeviceToHost));
    }
    for(int mode=1;mode<3;++mode) {
      const auto& got=outputs[mode];const auto& ref=outputs[0];double delta=0,oracle_error=0;
      for(size_t i=0;i<got.size();++i){if(!std::isfinite(got[i])||!std::isfinite(ref[i]))return 2;delta=std::max(delta,std::abs(double(got[i])-ref[i]));}
      for(int row=0;row<m;row+=std::max(m/7,1))for(int col=0;col<n;col+=std::max(n/13,1)) {
        double expected=0;for(int i=0;i<k;++i)expected+=double(a[size_t(row)*k+i])*b[size_t(col)*k+i];
        double error=std::abs(got[size_t(row)*n+col]-expected);oracle_error=std::max(oracle_error,error);
        if(error>2e-4+2e-5*std::abs(expected)){fprintf(stderr,"oracle failure mode=%d row=%d col=%d got=%g expected=%.12g\n",mode,row,col,got[size_t(row)*n+col],expected);return 3;}
      }
      printf("M=%d N=%d K=%d products=%d timing_valid=%d fp32_ms=%g paired_f16_ms=%g max_delta=%g fp64_error=%g oracle=pass\n",m,n,k,mode+2,!verify_only,timings[0],timings[mode],delta,oracle_error);
    }
    for(void*p:{(void*)da,(void*)db,(void*)dc,(void*)dr,(void*)ap,(void*)bp})CHECK(cudaFree(p));
  }
  BLAS(cublasDestroy(handle));CHECK(cudaStreamDestroy(stream));return 0;
}
