#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <cstring>
#include "../../crates/apxinf-cuda/kernels/custom/decode_epilogues.cuh"
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));exit(1);}}while(0)
int main(){
 float*table;CHECK(cudaMalloc(&table,65536*4));silu_bf16_table_kernel<<<256,256>>>(table);
 for(int mode=0;mode<2;++mode){
  int rows=mode?8192:65536*8,inter=mode?768:1;size_t count=size_t(rows)*inter;
  std::vector<half>gu(count*2);uint32_t rng=73;auto random=[&](){rng=rng*1664525+1013904223;return float(int(rng>>16)-32768)/4096.f;};
  for(int r=0;r<rows;++r)for(int j=0;j<inter;++j){
   half g;
   if(mode)g=__float2half(random());else {uint16_t bits=r/8;memcpy(&g,&bits,2);}
   gu[(size_t(r)*2)*inter+j]=g;gu[(size_t(r)*2+1)*inter+j]=__float2half(random());
  }
  half *input,*out,*ref;CHECK(cudaMalloc(&input,count*4));CHECK(cudaMalloc(&out,count*2));CHECK(cudaMalloc(&ref,count*2));CHECK(cudaMemcpy(input,gu.data(),count*4,cudaMemcpyHostToDevice));
  auto original=[&](){silu_mul_rows_f16_rounded_kernel<<<256,256>>>(input,ref,rows,inter);};
  auto lookup=[&](){silu_mul_rows_f16_lut_kernel<<<256,256>>>(input,out,table,rows,inter);};
  original();lookup();CHECK(cudaDeviceSynchronize());
  std::vector<half>a(count),b(count);CHECK(cudaMemcpy(a.data(),out,count*2,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(b.data(),ref,count*2,cudaMemcpyDeviceToHost));
  int bad=0;for(size_t i=0;i<count;++i){int r=i/inter,j=i%inter;if(!std::isfinite(__half2float(gu[size_t(r)*2*inter+j])))continue;if(memcmp(&a[i],&b[i],2))++bad;}
  printf("mode=%d bit_mismatches=%d ",mode,bad);if(bad)return 2;
  cudaEvent_t start,end;CHECK(cudaEventCreate(&start));CHECK(cudaEventCreate(&end));
  for(int which=0;which<2;++which){CHECK(cudaEventRecord(start));for(int r=0;r<100;++r){if(which)lookup();else original();}CHECK(cudaEventRecord(end));CHECK(cudaEventSynchronize(end));float ms;CHECK(cudaEventElapsedTime(&ms,start,end));printf("kernel%d_us=%g ",which,ms*10);}
  printf("\n");CHECK(cudaEventDestroy(start));CHECK(cudaEventDestroy(end));CHECK(cudaFree(input));CHECK(cudaFree(out));CHECK(cudaFree(ref));
 }
 CHECK(cudaFree(table));return 0;
}
