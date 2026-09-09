#define main residual_probe_main
#include "decode_residual_probe.cu"
#undef main
__global__ void reference_combine(
    const __nv_bfloat16* __restrict__ y, const int32_t* __restrict__ slot_rows,
    const float* __restrict__ weight, __nv_bfloat16* __restrict__ output,
    int tokens, int k, int cols) {
  const int64_t total = static_cast<int64_t>(tokens) * cols;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total; index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int m = static_cast<int>(index / cols);
    const int c = static_cast<int>(index - static_cast<int64_t>(m) * cols);
    float sum = 0.0f;
    for (int s = 0; s < k; ++s) {
      const int64_t slot = static_cast<int64_t>(m) * k + s;
      sum += weight[slot] *
             __bfloat162float(y[static_cast<int64_t>(slot_rows[slot]) * cols + c]);
    }
    output[index] = __float2bfloat16(sum);
  }
}
int main(){
 const int cols=2048,topk=8;
 for(int rows:{1,17,128,1024}){
  uint32_t seed=57;auto random=[&](){seed=seed*1664525+1013904223;return float(int(seed>>16)-32768)/32768.f;};
  std::vector<__nv_bfloat16>x(rows*cols),y(rows*topk*cols),weight(cols);std::vector<int32_t>ids(rows*topk);std::vector<float>router(ids.size());
  for(auto&p:x)p=__float2bfloat16(random());for(auto&p:y)p=__float2bfloat16(random());for(auto&p:weight)p=__float2bfloat16(1+random());
  for(int i=0;i<rows*topk;++i){ids[i]=rows*topk-1-i;router[i]=(1+random())/topk;}
  std::vector<half> yh(y.size());for(size_t i=0;i<y.size();++i){yh[i]=__float2half(random());y[i]=__float2bfloat16(__half2float(yh[i]));}
  auto fy=upload(yh);
  auto dx=upload(x),rx=upload(x),dy=upload(y),dw=upload(weight);auto di=upload(ids);auto dr=upload(router);__nv_bfloat16*out,*delta,*ref;
  CHECK(cudaMalloc(&out,x.size()*2));CHECK(cudaMalloc(&ref,x.size()*2));CHECK(cudaMalloc(&delta,x.size()*2));
  reference_combine<<<256,256>>>(dy,di,dr,delta,rows,topk,cols);reference_residual_norm<<<rows,256,cols*4>>>(rx,delta,dw,ref,cols,rows,1e-6f);
  routed_residual_rms_bf16_kernel<true><<<rows,256,cols*4>>>(dx,reinterpret_cast<__nv_bfloat16*>(fy),di,dr,dw,out,cols,rows,topk,1e-6f);CHECK(cudaDeviceSynchronize());
  std::vector<__nv_bfloat16>got(x.size()),expected(x.size()),gx(x.size()),ex(x.size());
  CHECK(cudaMemcpy(got.data(),out,x.size()*2,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(expected.data(),ref,x.size()*2,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(gx.data(),dx,x.size()*2,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(ex.data(),rx,x.size()*2,cudaMemcpyDeviceToHost));
  int bad=0;for(size_t i=0;i<x.size();++i){bad+=memcmp(&got[i],&expected[i],2)!=0;bad+=memcmp(&gx[i],&ex[i],2)!=0;}
  printf("rows=%d bit_mismatches=%d\n",rows,bad);if(bad)return 2;
  for(void*p:{(void*)fy,(void*)dx,(void*)rx,(void*)dy,(void*)dw,(void*)di,(void*)dr,(void*)out,(void*)delta,(void*)ref})CHECK(cudaFree(p));
 }
 return 0;
}
