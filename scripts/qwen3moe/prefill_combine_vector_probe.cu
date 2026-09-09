// Production vector combine versus the original scalar fused composition.
// nvcc -O3 -arch=sm_101 scripts/qwen3moe/prefill_combine_vector_probe.cu -o prefill_combine_vector_probe
#define main residual_probe_main
#include "decode_residual_probe.cu"
#undef main

int main() {
 for(int rows:{1,17,128,1024})for(int cols:{2048,4096})for(bool f16:{false,true}) {
  const int topk=8;uint32_t seed=62;auto random=[&](){seed=seed*1664525+1013904223;return float(int(seed>>16)-32768)/32768.f;};
  std::vector<__nv_bfloat16>x(rows*cols),y(rows*topk*cols),weight(cols);std::vector<int32_t>ids(rows*topk);std::vector<float>router(ids.size());
  for(auto&p:x)p=__float2bfloat16(random());for(auto&p:weight)p=__float2bfloat16(1+random());
  for(int i=0;i<rows*topk;++i){ids[i]=(i*13+rows*topk-1)%(rows*topk);router[i]=(1+random())/topk;if(rows==17&&i%17==0)ids[i]=-1;}
  std::vector<half> yh(y.size());for(size_t i=0;i<y.size();++i){yh[i]=__float2half(random());y[i]=__float2bfloat16(__half2float(yh[i]));}
  auto fy=upload(yh);auto dx=upload(x),rx=upload(x),dy=upload(y),dw=upload(weight);auto di=upload(ids);auto dr=upload(router);__nv_bfloat16 *out,*ref;
  CHECK(cudaMalloc(&out,x.size()*2));CHECK(cudaMalloc(&ref,x.size()*2));
  auto input=f16?reinterpret_cast<__nv_bfloat16*>(fy):dy;
  auto launch=[&](bool vector){
   if(vector){
    if(f16)routed_residual_rms_vector_kernel<true><<<rows,256,cols*4>>>(dx,input,di,dr,dw,out,cols,rows,topk,1e-6f);
    else routed_residual_rms_vector_kernel<false><<<rows,256,cols*4>>>(dx,input,di,dr,dw,out,cols,rows,topk,1e-6f);
   }else{
    if(f16)routed_residual_rms_bf16_kernel<true><<<rows,256,cols*4>>>(rx,input,di,dr,dw,ref,cols,rows,topk,1e-6f);
    else routed_residual_rms_bf16_kernel<false><<<rows,256,cols*4>>>(rx,input,di,dr,dw,ref,cols,rows,topk,1e-6f);
   }
  };
  launch(false);launch(true);CHECK(cudaDeviceSynchronize());
  std::vector<__nv_bfloat16>got(x.size()),expected(x.size()),gx(x.size()),ex(x.size());
  CHECK(cudaMemcpy(got.data(),out,x.size()*2,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(expected.data(),ref,x.size()*2,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(gx.data(),dx,x.size()*2,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(ex.data(),rx,x.size()*2,cudaMemcpyDeviceToHost));
  int bad=0;for(size_t i=0;i<x.size();++i){bad+=memcmp(&got[i],&expected[i],2)!=0;bad+=memcmp(&gx[i],&ex[i],2)!=0;}
  printf("rows=%d cols=%d f16=%d bit_mismatches=%d\n",rows,cols,f16,bad);if(bad)return 2;
  if(rows==1024&&cols==2048&&f16)for(bool vector:{false,true,false,true}) {
    CHECK(cudaMemcpy(vector?dx:rx,x.data(),x.size()*2,cudaMemcpyHostToDevice));
    for(int i=0;i<10;++i)launch(vector);
    cudaEvent_t begin,end;CHECK(cudaEventCreate(&begin));CHECK(cudaEventCreate(&end));
    CHECK(cudaEventRecord(begin));for(int i=0;i<100;++i)launch(vector);CHECK(cudaEventRecord(end));CHECK(cudaEventSynchronize(end));
    float ms;CHECK(cudaEventElapsedTime(&ms,begin,end));printf("vector=%d us=%.3f\n",vector,ms*10);
    CHECK(cudaEventDestroy(begin));CHECK(cudaEventDestroy(end));
  }
  for(void*p:{(void*)fy,(void*)dx,(void*)rx,(void*)dy,(void*)dw,(void*)di,(void*)dr,(void*)out,(void*)ref})CHECK(cudaFree(p));
 }
 return 0;
}
