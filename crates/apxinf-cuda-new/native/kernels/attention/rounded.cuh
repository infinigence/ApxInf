#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace apxinf::attention::rounded_kernels {
template<class T> __device__ float read(T x) { return static_cast<float>(x); }
template<class T> __device__ T cast(float x) { return static_cast<T>(x); }

template<bool Maximum>
__device__ float reduce(float x, float* scratch) {
  for (int d=16; d; d/=2) {
    const float y = __shfl_down_sync(0xffffffff, x, d);
    x = Maximum ? fmaxf(x,y) : x+y;
  }
  if ((threadIdx.x&31)==0) scratch[threadIdx.x/32]=x;
  __syncthreads();
  if (threadIdx.x<32) {
    x = threadIdx.x < blockDim.x/32 ? scratch[threadIdx.x]
                                  : Maximum ? -CUDART_INF_F : 0.0f;
    for (int d=16; d; d/=2) {
      const float y = __shfl_down_sync(0xffffffff, x, d);
      x = Maximum ? fmaxf(x,y) : x+y;
    }
    if (threadIdx.x==0) scratch[0]=x;
  }
  __syncthreads();
  return scratch[0];
}

template<class T>
__global__ void bias_softmax(T* scores, const T* bias, int keys) {
  __shared__ float scratch[32];
  const int64_t offset = static_cast<int64_t>(blockIdx.x)*keys;
  float maximum = -CUDART_INF_F;
  for (int64_t k=threadIdx.x; k<keys; k+=blockDim.x) {
    const T value = cast<T>(__fadd_rn(read(scores[offset+k]), read(bias[offset+k])));
    scores[offset+k] = value;
    maximum = fmaxf(maximum, read(value));
  }
  maximum = reduce<true>(maximum, scratch);
  float sum=0.0f;
  for (int64_t k=threadIdx.x; k<keys; k+=blockDim.x)
    sum += expf(read(scores[offset+k])-maximum);
  __syncthreads();
  sum = reduce<false>(sum, scratch);
  for (int64_t k=threadIdx.x; k<keys; k+=blockDim.x)
    scores[offset+k] = cast<T>(expf(read(scores[offset+k])-maximum)/sum);
}

// One warp per materialized row. Lane-strided accumulation followed by
// butterfly reduction preserves the reference FP32 softmax rounding before
// casting probabilities back to the input dtype. No block-wide scratch.
template<class T>
__global__ void bias_softmax_warp(T* scores, const T* bias, int keys, int rows) {
  const int64_t row=static_cast<int64_t>(blockIdx.x)*4+threadIdx.y;
  if (row>=rows) return;
  const int lane=threadIdx.x;
  const int64_t base=static_cast<int64_t>(row)*keys;
  float values[64];
  const int iterations=(keys+31)/32;
  float maximum=-CUDART_INF_F;
  for (int i=0;i<iterations;++i) {
    const int k=i*32+lane;
    values[i]=k<keys ? read(cast<T>(__fadd_rn(read(scores[base+k]),read(bias[base+k])))) : -CUDART_INF_F;
    maximum=fmaxf(maximum,values[i]);
  }
  for (int delta=16;delta;delta/=2)
    maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,delta));
  float sum=0.f;
  for (int i=0;i<iterations;++i) {
    values[i]=expf(values[i]-maximum);
    sum=__fadd_rn(sum,values[i]);
  }
  for (int delta=16;delta;delta/=2)
    sum=__fadd_rn(sum,__shfl_xor_sync(0xffffffff,sum,delta));
  for (int i=0;i<iterations;++i) {
    const int k=i*32+lane;
    if (k<keys) scores[base+k]=cast<T>(values[i]/sum);
  }
}

template<class T>
__global__ void unpack_heads(const T* input, T* output, int64_t count,
                             int queries, int heads, int dim) {
  for (int64_t i=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
       i<count; i+=static_cast<int64_t>(gridDim.x)*blockDim.x) {
    const int c=i%dim;
    const int h=(i/dim)%heads;
    const int q=(i/dim/heads)%queries;
    const int64_t b=i/dim/heads/queries;
    output[i]=input[((b*heads+h)*queries+q)*dim+c];
  }
}
} // namespace apxinf::attention::rounded_kernels
