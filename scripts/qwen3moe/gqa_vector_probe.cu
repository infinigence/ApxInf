// Independent CPU FP64 softmax oracle for the GQA decode candidate.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "gqa_vector_probe.cuh"
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));exit(1);}}while(0)
template<class T>T* upload(const std::vector<T>& x){T*p;CHECK(cudaMalloc(&p,x.size()*sizeof(T)));CHECK(cudaMemcpy(p,x.data(),x.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
__global__ void evict_gqa(uint32_t* p,int salt) {
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<(64<<20)/4;i+=gridDim.x*blockDim.x)p[i]^=salt;
}
int main(int argc,char** argv){
  const int tile=argc>2?atoi(argv[2]):32;
  if(tile!=0 && tile!=1 && tile!=32 && tile!=64 && tile!=128)return 2;
  auto fn=tile==1?gqa_decode_vector_bf16_kernel<false>:tile==64?gqa_decode_mma_vector_bf16_kernel<64>:tile==128?gqa_decode_mma_vector_bf16_kernel<128>:gqa_decode_mma_vector_bf16_kernel<32>;
  const int shared=tile==1?0:8*136*2+tile*136*2+tile*128*2+8*(tile+4)*4+2*8*(tile+8)*2+8*132*4;
  if(tile==128)CHECK(cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
  const int capacity=8192,heads=32,kvheads=4,splits=argc>1?atoi(argv[1]):16;
  if(splits<1 || splits>128)return 2;
  std::vector<__nv_bfloat16> q(heads*128),k(kvheads*capacity*128),v(k.size());
  uint32_t seed=123;auto random=[&](){seed=seed*1664525+1013904223;return __float2bfloat16(float(int(seed>>16)-32768)/32768.f);};
  for(auto&x:q)x=random();for(auto&x:k)x=random();for(auto&x:v)x=random();
  auto dq=upload(q),dk=upload(k),dv=upload(v);__nv_bfloat16*out;float*partial;uint32_t*pos;
  CHECK(cudaMalloc(&out,heads*128*2));CHECK(cudaMalloc(&partial,splits*heads*130*4));CHECK(cudaMalloc(&pos,4));
  cudaStream_t stream;CHECK(cudaStreamCreate(&stream));
  uint32_t* eviction;CHECK(cudaMalloc(&eviction,64<<20));CHECK(cudaMemset(eviction,0,64<<20));
  for(int length:{1,31,32,33,127,256,512,513,1024,1025,2048,2049,4096,4097,8192}){
    uint32_t position=length-1;CHECK(cudaMemcpy(pos,&position,4,cudaMemcpyHostToDevice));
    auto launch=[&](){if(tile)fn<<<dim3(kvheads,splits),256,shared,stream>>>(dq,dk,dv,partial,pos,capacity,splits,kvheads,1.f/sqrtf(128));else gqa_decode_partial_bf16_kernel<<<dim3(kvheads,splits),256,0,stream>>>(dq,dk,dv,partial,pos,capacity,splits,kvheads,1.f/sqrtf(128));gqa_decode_combine_bf16_kernel<<<heads,128,0,stream>>>(partial,out,heads,splits);};
    launch();CHECK(cudaDeviceSynchronize());
    cudaGraph_t graph;cudaGraphExec_t executable;CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));launch();CHECK(cudaStreamEndCapture(stream,&graph));CHECK(cudaGraphInstantiate(&executable,graph,0));
    cudaEvent_t begin,end;CHECK(cudaEventCreate(&begin));CHECK(cudaEventCreate(&end));float ms=0;
    for(int i=0;i<40;++i) {
      evict_gqa<<<1024,256,0,stream>>>(eviction,i+1);
      CHECK(cudaEventRecord(begin,stream));CHECK(cudaGraphLaunch(executable,stream));
      CHECK(cudaEventRecord(end,stream));CHECK(cudaEventSynchronize(end));float elapsed;
      CHECK(cudaEventElapsedTime(&elapsed,begin,end));ms+=elapsed/40;
    }
    std::vector<__nv_bfloat16> got(heads*128);CHECK(cudaMemcpy(got.data(),out,got.size()*2,cudaMemcpyDeviceToHost));
    if(tile==1) {
      gqa_decode_partial_bf16_kernel<<<dim3(kvheads,splits),256,0,stream>>>(dq,dk,dv,partial,pos,capacity,splits,kvheads,1.f/sqrtf(128));
      gqa_decode_combine_bf16_kernel<<<heads,128,0,stream>>>(partial,out,heads,splits);CHECK(cudaStreamSynchronize(stream));
      std::vector<__nv_bfloat16> original(heads*128);CHECK(cudaMemcpy(original.data(),out,original.size()*2,cudaMemcpyDeviceToHost));
      int mismatches=0;for(size_t i=0;i<got.size();++i)mismatches+=__bfloat16_as_ushort(got[i])!=__bfloat16_as_ushort(original[i]);
      printf("bit_length=%d mismatches=%d\n",length,mismatches);if(mismatches)return 3;
    }
    double maxerror=0;int bad=0;
    for(int h=0;h<heads;++h){
      std::vector<double> score(length);double maximum=-INFINITY,total=0;
      for(int t=0;t<length;++t){double dot=0;for(int d=0;d<128;++d)dot+=double(__bfloat162float(q[h*128+d]))*__bfloat162float(k[((h/8)*capacity+t)*128+d]);score[t]=dot/sqrt(128.);maximum=fmax(maximum,score[t]);}
      for(auto&x:score){x=exp(x-maximum);total+=x;}
      for(int d=0;d<128;++d){double ref=0;for(int t=0;t<length;++t)ref+=score[t]*__bfloat162float(v[((h/8)*capacity+t)*128+d]);ref/=total;double x=__bfloat162float(got[h*128+d]),error=fabs(x-ref);maxerror=fmax(maxerror,error);if(!std::isfinite(x)||error>.003+.01*fabs(ref))++bad;}
    }
    printf("length=%d ms=%g max_error=%g bad=%d\n",length,ms,maxerror,bad);if(bad)return 2;CHECK(cudaEventDestroy(begin));CHECK(cudaEventDestroy(end));CHECK(cudaGraphExecDestroy(executable));CHECK(cudaGraphDestroy(graph));
  }
  CHECK(cudaFree(eviction));CHECK(cudaStreamDestroy(stream));CHECK(cudaFree(dq));CHECK(cudaFree(dk));CHECK(cudaFree(dv));CHECK(cudaFree(out));CHECK(cudaFree(partial));CHECK(cudaFree(pos));
}
