// Copyright 2026 ApxInf contributors.
//
// The sampling family's greedy selection.
//
// Moved verbatim out of kernels/custom/model_ops.cu, which was named after a
// call site rather than a mathematical family (doc/cuda-new-design.md section
// 1.3). Kernel and host launch are unchanged apart from the namespace.

#include "sampling_argmax.h"

#include <cuda_bf16.h>

namespace apxinf::cuda_new::sampling_argmax {
namespace {

__global__ void argmax_kernel(const __nv_bfloat16* __restrict__ logits,
                              int32_t* __restrict__ index, int count) {
  __shared__ float best_value[32];
  __shared__ int best_index[32];

  float local_value = -INFINITY;
  int local_index = 0;
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    const float value = __bfloat162float(logits[i]);
    if (value > local_value) {
      local_value = value;
      local_index = i;
    }
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    const float other_value =
        __shfl_down_sync(0xFFFFFFFFu, local_value, offset);
    const int other_index = __shfl_down_sync(0xFFFFFFFFu, local_index, offset);
    if (other_value > local_value) {
      local_value = other_value;
      local_index = other_index;
    }
  }
  const int warp = threadIdx.x >> 5;
  if ((threadIdx.x & 31) == 0) {
    best_value[warp] = local_value;
    best_index[warp] = local_index;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const int warps = (blockDim.x + 31) / 32;
    float value = best_value[0];
    int chosen = best_index[0];
    for (int i = 1; i < warps; ++i) {
      if (best_value[i] > value) {
        value = best_value[i];
        chosen = best_index[i];
      }
    }
    *index = chosen;
  }
}


}  // namespace

int argmax_bf16(const void* logits, int32_t* index, int count,
                cudaStream_t stream) {
  if (count <= 0) return -1;
  argmax_kernel<<<1, 1024, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(logits), index, count);
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

}  // namespace apxinf::cuda_new::sampling_argmax
