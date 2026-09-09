#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include "../../crates/apxinf-cuda/kernels/custom/reduction.cuh"
#include "../../crates/apxinf-cuda/kernels/custom/quantization.cuh"
#include "../../crates/apxinf-cuda/kernels/custom/w4a16_blocked.cuh"
#include "../../crates/apxinf-cuda/kernels/custom/w4a16_magic.cuh"
#include "gemv_pair_probe.cuh"
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));exit(1);}}while(0)
template<class T>T* upload(const std::vector<T>&x){T*p;CHECK(cudaMalloc(&p,x.size()*sizeof(T)));CHECK(cudaMemcpy(p,x.data(),x.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
__global__ void evict(uint32_t*p,int count,uint32_t salt){for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=gridDim.x*blockDim.x)p[i]^=salt;}
int main(){
 struct Case {int k,n,slots,splits;bool separate,scaled;int group=128;};
 uint32_t* cache;CHECK(cudaMalloc(&cache,64<<20));CHECK(cudaMemset(cache,0,64<<20));
 for(auto c:{Case{2048,5120,1,4,false,false},Case{4096,2048,1,8,false,false},Case{2048,1536,8,2,false,false},Case{768,2048,8,1,true,true},Case{2048,5120,1,3,false,false},Case{2048,1536,8,3,false,false,64}}){
  int e=c.slots==1?1:128,pc=c.n/8,rp=((c.k+c.splits-1)/c.splits+7)/8*8;int64_t sq=int64_t(c.k)*pc,sz=(c.k/c.group)*pc,ss=(c.k/c.group)*c.n;
  std::vector<int32_t>q(e*sq),z(e*sz),ids(c.slots);std::vector<half>s(e*ss);std::vector<__nv_bfloat16>x(c.k*c.slots);std::vector<float>sc(c.slots);
  uint32_t seed=17;auto next=[&](){seed=seed*1664525+1013904223;return seed;};for(auto&v:q)v=next();for(auto&v:z)v=next();for(auto&v:s)v=__float2half((1+next()%100)/1000.f);for(auto&v:x)v=__float2bfloat16(float(int(next()%2001)-1000)/1000.f);for(int i=0;i<c.slots;++i){ids[i]=c.slots==1?0:i*17;sc[i]=(1+i)/36.f;}
  auto dq=upload(q),dz=upload(z),di=upload(ids);auto ds=upload(s);auto dx=upload(x);auto dw=upload(sc);int4*packed;CHECK(cudaMalloc(&packed,q.size()*4));
  w4a16_blocked_repack_kernel<<<1024,256>>>(dq,packed,c.k,pc,e);CHECK(cudaDeviceSynchronize());
  std::vector<int32_t>pq(q.size());CHECK(cudaMemcpy(pq.data(),packed,q.size()*4,cudaMemcpyDeviceToHost));
  for(int expert=0;expert<e;++expert)for(int row=0;row<c.k;++row)for(int col=0;col<pc;++col){int64_t to=expert*sq+((int64_t(col/32)*(c.k/4)+row/4)*32+col%32)*4+row%4;if(pq[to]!=q[expert*sq+int64_t(row)*pc+col])return 3;}
  float *ref,*out;int64_t count=int64_t(c.n)*c.slots*c.splits;CHECK(cudaMalloc(&ref,count*4));CHECK(cudaMalloc(&out,count*4));
  dim3 grid(c.n/256,c.splits,c.slots);size_t shared=(rp+8*256)*4;
  auto base=w4a16_gemv_magic_kernel<true>;
  decltype(base) variants[]={base,w4a16_gemv_pair_kernel<true>,(c.group==128?w4a16_gemv_pair_kernel<true,false,128,false>:w4a16_gemv_pair_kernel<true,false,0,false>),(c.group==128?w4a16_gemv_pair_kernel<true,false,128,true>:w4a16_gemv_pair_kernel<true,false,0,true>),w4a16_gemv_pair_kernel<true,false,0,true>};
  auto launch=[&](int mode){variants[mode]<<<grid,256,shared>>>(dx,c.separate?c.k:0,reinterpret_cast<int32_t*>(packed),dz,ds,di,sq,sz,ss,c.scaled?dw:nullptr,mode?out:ref,c.k,pc,c.group,rp);};
  launch(0);CHECK(cudaDeviceSynchronize());std::vector<float>a(count),b(count);CHECK(cudaMemcpy(a.data(),ref,count*4,cudaMemcpyDeviceToHost));
  cudaEvent_t start,end;CHECK(cudaEventCreate(&start));CHECK(cudaEventCreate(&end));
  for(int mode=0;mode<5;++mode) {
    launch(mode);CHECK(cudaDeviceSynchronize());CHECK(cudaMemcpy(b.data(),mode?out:ref,count*4,cudaMemcpyDeviceToHost));
    int bad=0;for(int64_t i=0;i<count;++i)bad+=memcmp(&a[i],&b[i],4)!=0;
    cudaFuncAttributes attr;CHECK(cudaFuncGetAttributes(&attr,variants[mode]));float ms=0;
    for(int repeat=0;repeat<40;++repeat){evict<<<1024,256>>>(cache,(64<<20)/4,repeat+1);CHECK(cudaEventRecord(start));launch(mode);CHECK(cudaEventRecord(end));CHECK(cudaEventSynchronize(end));float elapsed;CHECK(cudaEventElapsedTime(&elapsed,start,end));ms+=elapsed/40;}
    printf("K=%d N=%d slots=%d splits=%d group=%d mode=%d registers=%d operator_ms=%g bit_mismatches=%d\n",c.k,c.n,c.slots,c.splits,c.group,mode,attr.numRegs,ms,bad);if(bad)return 2;
  }
  CHECK(cudaEventDestroy(start));CHECK(cudaEventDestroy(end));for(void*p:{(void*)dq,(void*)dz,(void*)di,(void*)ds,(void*)dx,(void*)dw,(void*)packed,(void*)ref,(void*)out})CHECK(cudaFree(p));
 }
 CHECK(cudaFree(cache));return 0;
}
