// Copyright 2026 ApxInf contributors.
//
// The `(1 + weight)` RMSNorm of the norm family.
//
// This translation unit exists only so the kernel can be moved here byte for
// byte. It was written for the Qwen3.8 MLP and used to live in
// kernels/custom/mlp_ops.cu behind the separate `apxinf_new_rms_norm_bf16`
// ABI; the code below -- `kWarpSize`, `warp_sum`, `rms_norm_kernel` and the
// host launch -- is that code unchanged apart from the enclosing namespace.
//
// It is deliberately NOT folded into kernels/custom/norm.cuh. Those kernels
// reduce through `block_sum_parallel_unsafe` with a fixed 256-thread block
// and 32 floats of static shared memory; this one reduces with its own
// `warp_sum` tree over a width-derived block of up to 1024 threads and
// dynamic shared memory. Both are correct f32 reductions, but they sum in
// different orders, so rebuilding this on the shared helpers would move the
// Qwen3.8 output bits. Keeping the file separate keeps that impossible by
// construction.
//
// Single implementation, no persisted selection, so per
// doc/adding-new-kernels.md section 6 it carries no recipe.

#include "norm_offset.h"

#include <cuda_bf16.h>

namespace apxinf::cuda_new::norm_offset {
namespace {

constexpr int kWarpSize = 32;

__device__ __forceinline__ float warp_sum(float value) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xFFFFFFFFu, value, offset);
  }
  return value;
}

// One block per row. The reduction is in f32 regardless of the BF16 storage:
// summing 5120 squares in BF16 loses the small terms entirely.
__global__ void rms_norm_kernel(const __nv_bfloat16* __restrict__ input,
                                const __nv_bfloat16* __restrict__ weight,
                                __nv_bfloat16* __restrict__ output, int rows,
                                int width, float epsilon) {
  extern __shared__ float partials[];
  const int row = blockIdx.x;
  if (row >= rows) return;
  const long long base = (long long)row * width;

  float sum = 0.0f;
  for (int index = threadIdx.x; index < width; index += blockDim.x) {
    const float value = __bfloat162float(input[base + index]);
    sum += value * value;
  }
  sum = warp_sum(sum);
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  if (lane == 0) partials[warp] = sum;
  __syncthreads();

  if (threadIdx.x < kWarpSize) {
    const int warps = (blockDim.x + kWarpSize - 1) / kWarpSize;
    float total = threadIdx.x < warps ? partials[threadIdx.x] : 0.0f;
    total = warp_sum(total);
    if (threadIdx.x == 0) partials[0] = total;
  }
  __syncthreads();

  const float scale = rsqrtf(partials[0] / static_cast<float>(width) + epsilon);
  for (int index = threadIdx.x; index < width; index += blockDim.x) {
    const float value = __bfloat162float(input[base + index]) * scale *
                        (1.0f + __bfloat162float(weight[index]));
    output[base + index] = __float2bfloat16(value);
  }
}

}  // namespace

cudaError_t rms_norm_bf16(const void* input, const void* weight, void* output,
                          int rows, int width, float epsilon,
                          cudaStream_t stream) {
  if (rows <= 0 || width <= 0) return cudaErrorInvalidValue;
  const int threads = width >= 1024 ? 1024 : ((width + 31) / 32) * 32;
  const int warps = (threads + kWarpSize - 1) / kWarpSize;
  rms_norm_kernel<<<rows, threads, warps * sizeof(float), stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<__nv_bfloat16*>(output), rows, width, epsilon);
  // Returns the real launch status. This call also *consumes* the error, so
  // the caller must use this value rather than its own cudaGetLastError().
  return cudaGetLastError();
}

}  // namespace apxinf::cuda_new::norm_offset
