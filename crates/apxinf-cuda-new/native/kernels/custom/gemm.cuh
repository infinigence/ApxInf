#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
namespace apxinf::cuda::custom {
__device__ inline float load(const void* p, int type, int64_t i) {
 if(type==4)return float(((const int8_t*)p)[i]);
 if(type==5)return float(((const int32_t*)p)[i]);
 if(type==0) return ((const float*)p)[i];
 if(type==1) return __half2float(((const half*)p)[i]);
 if(type==2) return __bfloat162float(((const __nv_bfloat16*)p)[i]);
 return float(((const __nv_fp8_e4m3*)p)[i]);
}
__device__ inline void save(void* p,int type,int64_t i,float x) {
 if(type==0) ((float*)p)[i]=x;
 else if(type==1) ((half*)p)[i]=__float2half_rn(x);
 else if(type==2) ((__nv_bfloat16*)p)[i]=__float2bfloat16_rn(x);
 else ((__nv_fp8_e4m3*)p)[i]=__nv_fp8_e4m3(x);
}
static __global__ void unpack(const void* src,void* dst,int out_type,int type,int64_t rows,int64_t cols,int layout) {
 for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<rows*cols;i+=int64_t(gridDim.x)*blockDim.x) {
  int64_t r=i/cols,c=i%cols,j=i;
  if(layout==1) j=c*rows+r;
  if(layout==2) { int64_t h=cols/2; j=r*cols+(c%h/256)*512+(c>=h?256:0)+c%256; }
  save(dst,out_type,i,load(src,type,j));
 }
}
static __global__ void pack_gate_up(const void* src,void* dst,int type,int64_t rows,int64_t cols){
 for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<rows*cols;i+=int64_t(gridDim.x)*blockDim.x){
  int64_t r=i/cols,c=i%cols,h=cols/2;
  int64_t j=r*cols+(c%h/256)*512+(c>=h?256:0)+c%256;
  save(dst,type,j,load(src,type,i));
 }
}
__device__ inline float gelu(float x) {return 0.5f*x*(1.f+tanhf(0.7978845608028654f*(x+0.044715f*x*x*x)));}
static __global__ void finish(const void* projection,int projection_type,void* output,int out_type,
 const void* bias,int bias_type,const float* as,const float* bs,
 int64_t m,int64_t n,int semantic,int scale_mode,float alpha,float output_scale) {
 int64_t width=semantic==2?n/2:n;
 for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<m*width;i+=int64_t(gridDim.x)*blockDim.x) {
  int64_t r=i/width,c=i%width;
  float factor=scale_mode==1?as[r]*bs[c]:1.f;
  float x=load(projection,projection_type,r*n+c)*factor*alpha;
  if(semantic==2) {
   float up=load(projection,projection_type,r*n+c+width)*alpha;
   if(scale_mode==1) up*=as[r]*bs[c+width];
   x=gelu(x)*up;
  } else {
   if(bias) x+=load(bias,bias_type,c);
   if(semantic==1) x=gelu(x);
  }
  save(output,out_type,i,x/output_scale);
 }
}
inline cudaError_t unpack_gemm(const void* src,void* dst,int out_type,int type,int64_t r,int64_t c,int layout,cudaStream_t stream) {
 unpack<<<int(std::min<int64_t>((r*c+255)/256,4096)),256,0,stream>>>(src,dst,out_type,type,r,c,layout); return cudaGetLastError();
}
} // namespace
