// Isolated experiment, not a production/acceptance path. Times cold device
// weight reads and checks both the existing GEMV and an independent FP64 dot.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <cmath>
#include <algorithm>
#define MARLIN_NAMESPACE_NAME apxinf_decode_pair
#include "../../crates/apxinf-cuda/kernels/marlin/csrc/moe/marlin_moe_wna16/kernel.h"
#include "../../crates/apxinf-cuda/kernels/marlin/csrc/quantization/gptq_marlin/dequant.h"
#include "../../crates/apxinf-cuda/kernels/custom/reduction.cuh"
#include "../../crates/apxinf-cuda/kernels/custom/quantization.cuh"
#include "../../crates/apxinf-cuda/kernels/custom/w4a16_blocked.cuh"
#include "../../crates/apxinf-cuda/kernels/custom/w4a16_pair.cuh"
#include "gemv_integer_mma_probe.cuh"
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));exit(1);}}while(0)
template<class T>T* upload(const std::vector<T>&x){T*p;CHECK(cudaMalloc(&p,x.size()*sizeof(T)));CHECK(cudaMemcpy(p,x.data(),x.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
__global__ void evict_integer_probe(uint32_t*p,uint32_t salt){for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<(64<<20)/4;i+=gridDim.x*blockDim.x)p[i]^=salt;}
int main(){
 struct Case {int k,n,slots,splits;bool separate,scaled;};
 uint32_t* cache;CHECK(cudaMalloc(&cache,64<<20));CHECK(cudaMemset(cache,0,64<<20));
 for(auto c:{Case{2048,5120,1,4,false,false},Case{4096,2048,1,8,false,false},Case{2048,1536,8,2,false,false},Case{768,2048,8,1,true,true},Case{2048,5120,1,3,false,false}}){
  int experts=c.slots==1?1:128,pc=c.n/8,rp=((c.k+c.splits-1)/c.splits+7)/8*8;
  int64_t sq=int64_t(c.k)*pc,sz=(c.k/128)*pc,ss=(c.k/128)*c.n;
  std::vector<int32_t>q(experts*sq),z(experts*sz),ids(c.slots);std::vector<half>s(experts*ss);
  std::vector<__nv_bfloat16>x(c.k*c.slots);std::vector<float>sc(c.slots);
  uint32_t seed=17;auto next=[&](){seed=seed*1664525+1013904223;return seed;};
  for(auto&v:q)v=next();for(auto&v:z)v=next();for(auto&v:s)v=__float2half((1+next()%100)/1000.f);
  for(auto&v:x)v=__float2bfloat16(float(int(next()%2001)-1000)/1000.f);
  for(int i=0;i<c.slots;++i){ids[i]=c.slots==1?0:i*17;sc[i]=(1+i)/36.f;}
  auto dq=upload(q),dz=upload(z),di=upload(ids);auto ds=upload(s);auto dx=upload(x);auto dw=upload(sc);
  int4*packed;CHECK(cudaMalloc(&packed,q.size()*4));
  w4a16_blocked_repack_kernel<<<1024,256>>>(dq,packed,c.k,pc,experts);CHECK(cudaDeviceSynchronize());
  float *ref,*out;int64_t count=int64_t(c.n)*c.slots*c.splits;CHECK(cudaMalloc(&ref,count*4));CHECK(cudaMalloc(&out,count*4));
  auto original=[&](){w4a16_gemv_pair_kernel<true,false,128,true><<<dim3(c.n/256,c.splits,c.slots),256,(rp+8*256)*4>>>(dx,c.separate?c.k:0,reinterpret_cast<int32_t*>(packed),dz,ds,di,sq,sz,ss,c.scaled?dw:nullptr,ref,c.k,pc,128,rp);};
  auto candidate=[&](){gemv_integer_mma_probe_kernel<<<dim3(c.n/128,c.splits,c.slots),256>>>(dx,c.separate?c.k:0,reinterpret_cast<int32_t*>(packed),dz,ds,di,sq,sz,ss,c.scaled?dw:nullptr,out,c.k,c.n,rp);};
  cudaFuncAttributes attr;CHECK(cudaFuncGetAttributes(&attr,gemv_integer_mma_probe_kernel));
  original();candidate();CHECK(cudaDeviceSynchronize());std::vector<float>a(count),b(count);
  CHECK(cudaMemcpy(a.data(),ref,count*4,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(b.data(),out,count*4,cudaMemcpyDeviceToHost));
  int different=0;double max_delta=0,oracle_error=0;
  for(int64_t i=0;i<count;++i){if(!std::isfinite(a[i])||!std::isfinite(b[i]))return 2;different+=memcmp(&a[i],&b[i],4)!=0;max_delta=std::max(max_delta,std::abs(double(a[i])-b[i]));}
  const int shifts[8]={0,4,1,5,2,6,3,7};
  for(int slot=0;slot<c.slots;++slot)for(int split=0;split<c.splits;++split)for(int col=0;col<c.n;col+=std::max(c.n/37,1)) {
   double expected=0;int e=ids[slot],nibble=shifts[col%8],j=col/8;
   for(int k=split*rp;k<std::min(c.k,(split+1)*rp);++k){int g=k/128;int qw=(q[e*sq+int64_t(k)*pc+j]>>(4*nibble))&15;int qz=(z[e*sz+int64_t(g)*pc+j]>>(4*nibble))&15;expected+=double(__bfloat162float(x[(c.separate?slot*c.k:0)+k]))*(qw-qz)*__half2float(s[e*ss+int64_t(g)*c.n+col]);}
   if(c.scaled)expected*=sc[slot];int64_t index=(int64_t(slot)*c.splits+split)*c.n+col;
   double error=std::abs(b[index]-expected);oracle_error=std::max(oracle_error,error);
   if(error>2e-4+2e-5*std::abs(expected)){fprintf(stderr,"oracle failure slot=%d split=%d col=%d got=%g expected=%.12g\n",slot,split,col,b[index],expected);return 3;}
  }
  cudaEvent_t start,end;CHECK(cudaEventCreate(&start));CHECK(cudaEventCreate(&end));float times[2]={};
  for(int mode=0;mode<2;++mode)for(int repeat=0;repeat<30;++repeat){evict_integer_probe<<<1024,256>>>(cache,repeat+1);CHECK(cudaEventRecord(start));if(mode)candidate();else original();CHECK(cudaEventRecord(end));CHECK(cudaEventSynchronize(end));float ms;CHECK(cudaEventElapsedTime(&ms,start,end));times[mode]+=ms/30;}
  printf("K=%d N=%d slots=%d splits=%d registers=%d shared=%zu paired_ms=%g integer_mma_ms=%g differing_values=%d max_delta=%g fp64_error=%g oracle=pass\n",c.k,c.n,c.slots,c.splits,attr.numRegs,attr.sharedSizeBytes,times[0],times[1],different,max_delta,oracle_error);
  CHECK(cudaEventDestroy(start));CHECK(cudaEventDestroy(end));for(void*p:{(void*)dq,(void*)dz,(void*)di,(void*)ds,(void*)dx,(void*)dw,(void*)packed,(void*)ref,(void*)out})CHECK(cudaFree(p));
 }
 CHECK(cudaFree(cache));return 0;
}
