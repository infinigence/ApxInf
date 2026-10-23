#pragma once
#include "../../include/apxinf_cuda/reduction.h"

namespace apxinf::reduction::kernels {
__global__ void indexed_mean(apxinf_indexed_mean_spec_t s,apxinf_indexed_mean_bindings_t b) {
 __shared__ float partial[128];
 for(int64_t c=blockIdx.x;c<s.cols;c+=gridDim.x) {
  float sum=0;
  for(int64_t r=threadIdx.x;r<s.rows;r+=128) {
   uint32_t index=b.indices[r*s.cols+c];
   if(index<s.input_elements) sum=__fadd_rn(sum,b.input[index]);
  }
  partial[threadIdx.x]=sum;__syncthreads();
  for(int offset=64;offset;offset/=2) {
   if(threadIdx.x<offset)partial[threadIdx.x]=__fadd_rn(partial[threadIdx.x],partial[threadIdx.x+offset]);
   __syncthreads();
  }
  if(!threadIdx.x)b.output[c]=__fdiv_rn(partial[0],float(s.rows));
  __syncthreads();
 }
}
}  // namespace apxinf::reduction::kernels
