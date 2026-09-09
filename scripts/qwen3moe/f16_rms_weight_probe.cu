// BF16 activation kernels with checkpoint FP16 RMS weights.
// nvcc -O3 -std=c++17 -arch=sm_101 scripts/qwen3moe/f16_rms_weight_probe.cu -o f16_rms_weight_probe
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include "../../crates/apxinf-cuda/kernels/custom/math.cuh"
#include "../../crates/apxinf-cuda/kernels/custom/reduction.cuh"
#include "../../crates/apxinf-cuda/kernels/custom/normalization.cuh"
#include "../../crates/apxinf-cuda/kernels/custom/fused.cuh"
#define main original_residual_probe_main
#include "decode_residual_probe.cu"
#undef main

int main() {
 for(int rows:{1,17,128})for(int cols:{2048,4096}) {
  const int count=8;uint32_t seed=48;auto random=[&](){seed=seed*1664525+1013904223;return float(int(seed>>16)-32768)/32768.f;};
  std::vector<__nv_bfloat16>x(rows*cols),delta(x.size()),bw(cols);std::vector<half>fw(cols),y(rows*count*cols);std::vector<float>parts(rows*count*cols),router(rows*count);std::vector<int32_t>ids(router.size());
  for(auto&v:x)v=__float2bfloat16(random());for(auto&v:delta)v=__float2bfloat16(random());
  for(int i=0;i<cols;++i){bw[i]=__float2bfloat16(1+random());fw[i]=__float2half(__bfloat162float(bw[i]));}
  for(auto&v:y)v=__float2half(random());for(auto&v:parts)v=random()/8;
  for(int i=0;i<int(ids.size());++i){ids[i]=int(ids.size())-1-i;router[i]=(1+random())/8;}
  auto dx=upload(x),rx=upload(x),dd=upload(delta),db=upload(bw);auto df=upload(fw),dy=upload(y);auto dp=upload(parts),dr=upload(router);auto di=upload(ids);
  __nv_bfloat16*out,*ref;CHECK(cudaMalloc(&out,x.size()*2));CHECK(cudaMalloc(&ref,x.size()*2));
  for(int op=0;op<6;++op) {
   CHECK(cudaMemcpy(dx,x.data(),x.size()*2,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(rx,x.data(),x.size()*2,cudaMemcpyHostToDevice));
   if(op==0){
    rms_norm_bf16_kernel<half><<<rows,256,cols*4>>>(dx,df,out,cols,rows,1e-6f);
    rms_norm_bf16_kernel<__nv_bfloat16><<<rows,256,cols*4>>>(rx,db,ref,cols,rows,1e-6f);
   }else if(op==1){
    rms_norm_add_bf16_kernel<half><<<rows,256,cols*4>>>(dx,dd,df,out,cols,rows,1e-6f);
    reference_residual_norm<<<rows,256,cols*4>>>(rx,dd,db,ref,cols,rows,1e-6f);
   }else if(op==2){
    partial_residual_rms_bf16_kernel<false,false,half><<<rows,256,cols*4>>>(dx,dp,df,out,cols,rows,count,1e-6f);
    partial_residual_rms_bf16_kernel<false><<<rows,256,cols*4>>>(rx,dp,db,ref,cols,rows,count,1e-6f);
   }else if(op==3){
    partial_residual_rms_bf16_kernel<true,false,half><<<rows,1024,cols*4>>>(dx,dp,df,out,cols,rows,count,1e-6f);
    partial_residual_rms_bf16_kernel<true><<<rows,1024,cols*4>>>(rx,dp,db,ref,cols,rows,count,1e-6f);
   }else if(op==4){
    routed_residual_rms_bf16_kernel<true,half><<<rows,256,cols*4>>>(dx,reinterpret_cast<__nv_bfloat16*>(dy),di,dr,df,out,cols,rows,count,1e-6f);
    routed_residual_rms_bf16_kernel<true><<<rows,256,cols*4>>>(rx,reinterpret_cast<__nv_bfloat16*>(dy),di,dr,db,ref,cols,rows,count,1e-6f);
   }else{
    routed_residual_rms_vector_kernel<true,half><<<rows,256,cols*4>>>(dx,reinterpret_cast<__nv_bfloat16*>(dy),di,dr,df,out,cols,rows,count,1e-6f);
    routed_residual_rms_vector_kernel<true><<<rows,256,cols*4>>>(rx,reinterpret_cast<__nv_bfloat16*>(dy),di,dr,db,ref,cols,rows,count,1e-6f);
   }
   CHECK(cudaDeviceSynchronize());std::vector<__nv_bfloat16>a(x.size()),b(x.size()),ax(x.size()),bx(x.size());
   CHECK(cudaMemcpy(a.data(),out,a.size()*2,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(b.data(),ref,b.size()*2,cudaMemcpyDeviceToHost));
   CHECK(cudaMemcpy(ax.data(),dx,ax.size()*2,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(bx.data(),rx,bx.size()*2,cudaMemcpyDeviceToHost));
   if(memcmp(a.data(),b.data(),a.size()*2)||memcmp(ax.data(),bx.data(),ax.size()*2)){fprintf(stderr,"format-preservation mismatch rows=%d cols=%d op=%d\n",rows,cols,op);return 2;}
  }
  // Non-BF16-exact weights must retain their FP16 bits. Check against host RMS.
  for(int i=0;i<cols;++i)fw[i]=__float2half(.5f+(i%997)/997.f);
  CHECK(cudaMemcpy(df,fw.data(),cols*2,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(dx,x.data(),x.size()*2,cudaMemcpyHostToDevice));
  rms_norm_bf16_kernel<half><<<rows,256,cols*4>>>(dx,df,out,cols,rows,1e-6f);CHECK(cudaDeviceSynchronize());
  std::vector<__nv_bfloat16>got(x.size());CHECK(cudaMemcpy(got.data(),out,got.size()*2,cudaMemcpyDeviceToHost));
  double maximum=0;
  for(int row=0;row<rows;++row){double sum=0;for(int i=0;i<cols;++i){double v=__bfloat162float(x[row*cols+i]);sum+=v*v;}double inv=1/std::sqrt(sum/cols+1e-6);
   for(int i=0;i<cols;++i){double expected=__bfloat162float(x[row*cols+i])*inv*__half2float(fw[i]);double value=__bfloat162float(got[row*cols+i]);double error=std::abs(value-expected);maximum=std::max(maximum,error);
    double tolerance=(expected==0?0:std::ldexp(1.,std::ilogb(std::abs(expected))-8))+5e-6*std::max(1.,std::abs(expected));
    if(!std::isfinite(value)||error>tolerance){fprintf(stderr,"FP16 RMS oracle mismatch row=%d col=%d got=%g expected=%.12g tolerance=%g\n",row,i,value,expected,tolerance);return 3;}
   }
  }
  printf("rows=%d cols=%d six_bit_checks=pass fp16_oracle_max_error=%g\n",rows,cols,maximum);
  for(void*p:{(void*)dx,(void*)rx,(void*)dd,(void*)db,(void*)df,(void*)dy,(void*)dp,(void*)dr,(void*)di,(void*)out,(void*)ref})CHECK(cudaFree(p));
 }
 return 0;
}
