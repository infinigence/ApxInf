#pragma once

#include <cuda_runtime.h>

namespace apxinf::norm::kernels {

__device__ __forceinline__ float warp_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1)
    value += __shfl_down_sync(0xffffffff, value, offset);
  return value;
}

// One block per row; `scratch` must hold blockDim.x / 32 floats. The final
// barrier publishes scratch[0], but does not protect the reads that follow it.
// Callers must consume the result and synchronize all block threads before
// reusing the same scratch storage.
__device__ __forceinline__ float block_sum_parallel_unsafe(float value,
                                                            float* scratch) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = blockDim.x >> 5;
  value = warp_sum(value);
  if (lane == 0) scratch[warp] = value;
  __syncthreads();
  if (warp == 0) {
    value = lane < warps ? scratch[lane] : 0.0f;
    value = warp_sum(value);
    if (lane == 0) scratch[0] = value;
  }
  __syncthreads();
  return scratch[0];
}

}  // namespace apxinf::norm::kernels
