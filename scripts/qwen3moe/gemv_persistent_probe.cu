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
#include "gemv_schedule_probe.cuh"
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));exit(1);}}while(0)
template<class T>T* upload(const std::vector<T>&x){T*p;CHECK(cudaMalloc(&p,x.size()*sizeof(T)));CHECK(cudaMemcpy(p,x.data(),x.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
__global__ void evict(uint32_t*p,int count,uint32_t salt){for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=gridDim.x*blockDim.x)p[i]^=salt;}
int main(){
 struct Case {int k,n,slots,splits;bool separate,scaled;};
 uint32_t* cache;CHECK(cudaMalloc(&cache,64<<20));CHECK(cudaMemset(cache,0,64<<20));
 for(auto c:{Case{2048,5120,1,4,false,false},Case{4096,2048,1,8,false,false},Case{2048,1536,8,2,false,false},Case{768,2048,8,1,true,true},Case{2048,5120,1,3,false,false},Case{768,768,17,3,true,true}}){
  int e=c.slots==1?1:128,pc=c.n/8,rp=((c.k+c.splits-1)/c.splits+7)/8*8;int64_t sq=int64_t(c.k)*pc,sz=(c.k/128)*pc,ss=(c.k/128)*c.n;
  std::vector<int32_t>q(e*sq),z(e*sz),ids(c.slots);std::vector<half>s(e*ss);std::vector<__nv_bfloat16>x(c.k*c.slots);std::vector<float>sc(c.slots);
  uint32_t seed=17;auto next=[&](){seed=seed*1664525+1013904223;return seed;};for(auto&v:q)v=next();for(auto&v:z)v=next();for(auto&v:s)v=__float2half((1+next()%100)/1000.f);for(auto&v:x)v=__float2bfloat16(float(int(next()%2001)-1000)/1000.f);for(int i=0;i<c.slots;++i){ids[i]=c.slots==1?0:(i*17)%e;sc[i]=(1+i)/36.f;}
  auto dq=upload(q),dz=upload(z),di=upload(ids);auto ds=upload(s);auto dx=upload(x);auto dw=upload(sc);int4*packed;CHECK(cudaMalloc(&packed,q.size()*4));
  w4a16_blocked_repack_kernel<<<1024,256>>>(dq,packed,c.k,pc,e);CHECK(cudaDeviceSynchronize());
  std::vector<int32_t>pq(q.size());CHECK(cudaMemcpy(pq.data(),packed,q.size()*4,cudaMemcpyDeviceToHost));
  for(int expert=0;expert<e;++expert)for(int row=0;row<c.k;++row)for(int col=0;col<pc;++col){int64_t to=expert*sq+((int64_t(col/32)*(c.k/4)+row/4)*32+col%32)*4+row%4;if(pq[to]!=q[expert*sq+int64_t(row)*pc+col])return 3;}
  float *ref,*out;int64_t count=int64_t(c.n)*c.slots*c.splits;CHECK(cudaMalloc(&ref,count*4));CHECK(cudaMalloc(&out,count*4));
  dim3 grid(c.n/256,c.splits,c.slots);size_t shared=(rp+8*256)*4;
  auto launch=[&](int mode){
    auto ids_ptr=c.slots==1?nullptr:di;
    if(mode==0) w4a16_gemv_magic_kernel<true><<<grid,256,shared>>>(dx,c.separate?c.k:0,reinterpret_cast<int32_t*>(packed),dz,ds,ids_ptr,sq,sz,ss,c.scaled?dw:nullptr,ref,c.k,pc,128,rp);
    else {
      int blocks=mode==1?56:mode==2?112:28;
      w4a16_gemv_schedule_kernel<true,false,true,16,2><<<blocks,256,shared>>>(dx,c.separate?c.k:0,reinterpret_cast<int32_t*>(packed),dz,ds,ids_ptr,sq,sz,ss,c.scaled?dw:nullptr,out,c.k,pc,128,rp,c.splits,c.slots);
    }
  };
  cudaFuncAttributes attr;int resident;
  CHECK(cudaFuncGetAttributes(&attr,w4a16_gemv_schedule_kernel<true,false,true,16,2>));
  CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,w4a16_gemv_schedule_kernel<true,false,true,16,2>,256,shared));
  launch(0);CHECK(cudaDeviceSynchronize());std::vector<float>a(count),b(count);CHECK(cudaMemcpy(a.data(),ref,count*4,cudaMemcpyDeviceToHost));
  cudaEvent_t start,end;CHECK(cudaEventCreate(&start));CHECK(cudaEventCreate(&end));float times[4]={};
  for(int mode=0;mode<4;++mode) {
    launch(mode);CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(b.data(),mode?out:ref,count*4,cudaMemcpyDeviceToHost));int bad=0;for(int64_t i=0;i<count;++i)bad+=memcmp(&a[i],&b[i],4)!=0;
    for(int repeat=0;repeat<40;++repeat){evict<<<1024,256>>>(cache,(64<<20)/4,repeat+1);CHECK(cudaEventRecord(start));launch(mode);CHECK(cudaEventRecord(end));CHECK(cudaEventSynchronize(end));float ms;CHECK(cudaEventElapsedTime(&ms,start,end));times[mode]+=ms/40;}
    printf("K=%d N=%d slots=%d splits=%d mode=%d grid=%d operator_ms=%g bit_mismatches=%d registers=%d resident_ctas=%d\n",c.k,c.n,c.slots,c.splits,mode,mode==0?int(grid.x*grid.y*grid.z):mode==1?56:mode==2?112:28,times[mode],bad,attr.numRegs,resident);if(bad)return 2;
  }
  CHECK(cudaEventDestroy(start));CHECK(cudaEventDestroy(end));for(void*p:{(void*)dq,(void*)dz,(void*)di,(void*)ds,(void*)dx,(void*)dw,(void*)packed,(void*)ref,(void*)out})CHECK(cudaFree(p));
 }
 CHECK(cudaFree(cache));return 0;
}
