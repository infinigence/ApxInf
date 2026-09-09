// BF16 checkpoint head inputs, FP32 output: independent host dot-product oracle.
// nvcc -O3 -arch=sm_101 bf16_f32_output_probe.cu -lcublas -o bf16_f32_output_probe
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#define CHECK(x) do { auto e=(x); if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));return 1;} } while(0)
#define BLAS(x) do { auto e=(x); if(e!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"%s: cuBLAS %d\n",#x,int(e));return 1;} } while(0)
int main() {
    cublasHandle_t handle; BLAS(cublasCreate(&handle));
    for(int m:{1,5}) for(int n:{128,513,151936}) {
        const int k=2048;
        std::vector<__nv_bfloat16> a(m*k),w(size_t(n)*k);
        // Exactly representable dyadic BF16 inputs make the host sum exact.
        for(size_t i=0;i<a.size();++i)a[i]=__float2bfloat16(float(int(i*19%127)-63)/128.f);
        for(size_t i=0;i<w.size();++i)w[i]=__float2bfloat16(float(int(i*29%131)-65)/128.f);
        __nv_bfloat16 *da,*dw,*db;float* df;
        CHECK(cudaMalloc(&da,a.size()*2));CHECK(cudaMalloc(&dw,w.size()*2));
        CHECK(cudaMalloc(&db,m*n*2));CHECK(cudaMalloc(&df,m*n*4));
        CHECK(cudaMemcpy(da,a.data(),a.size()*2,cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dw,w.data(),w.size()*2,cudaMemcpyHostToDevice));
        float alpha=1,beta=0;
        BLAS(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,n,m,k,&alpha,
            dw,CUDA_R_16BF,k,da,CUDA_R_16BF,k,&beta,df,CUDA_R_32F,n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
        BLAS(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,n,m,k,&alpha,
            dw,CUDA_R_16BF,k,da,CUDA_R_16BF,k,&beta,db,CUDA_R_16BF,n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
        CHECK(cudaDeviceSynchronize());
        std::vector<float> got(m*n);std::vector<__nv_bfloat16> rounded(m*n);
        CHECK(cudaMemcpy(got.data(),df,m*n*4,cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(rounded.data(),db,m*n*2,cudaMemcpyDeviceToHost));
        float max_error=0,max_bf16_error=0;int non_bf16=0;
        for(int row=0;row<m;++row)for(int col=0;col<n;col+=std::max(n/67,1)) {
            double expected=0;for(int inner=0;inner<k;++inner)
                expected+=double(__bfloat162float(a[row*k+inner]))*__bfloat162float(w[size_t(col)*k+inner]);
            float value=got[row*n+col];
            if(!std::isfinite(value) || std::abs(double(value)-expected)>1e-5){fprintf(stderr,"FP32 output mismatch M=%d N=%d row=%d col=%d got=%g ref=%.12g\n",m,n,row,col,value,expected);return 2;}
            max_error=std::max(max_error,float(std::abs(value-expected)));
            max_bf16_error=std::max(max_bf16_error,float(std::abs(__bfloat162float(rounded[row*n+col])-expected)));
        }
        for(int i=0;i<m*n;++i)non_bf16+=got[i]!=__bfloat162float(__float2bfloat16(got[i]));
        if(!non_bf16){fprintf(stderr,"output was unexpectedly rounded to BF16\n");return 3;}
        printf("M=%d N=%d K=%d fp32_max_error=%g bf16_max_error=%g non_bf16_outputs=%d\n",m,n,k,max_error,max_bf16_error,non_bf16);
        CHECK(cudaFree(da));CHECK(cudaFree(dw));CHECK(cudaFree(db));CHECK(cudaFree(df));
    }
    BLAS(cublasDestroy(handle));return 0;
}
