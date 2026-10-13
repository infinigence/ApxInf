// Copyright 2026 ApxInf contributors.
//
// The norm family's per-head RMSNorm, for Qwen3.5's q_norm and k_norm.
//
// Moved verbatim out of kernels/custom/attn_ops.cu, which was named after a
// call site rather than a mathematical family (doc/cuda-new-design.md section
// 1.3). Kernel and host launch are unchanged apart from the namespace.
//
// This is the norm family's third reduction tree, after norm.cuh's
// block_sum_parallel_unsafe and norm_offset.cu's warp tree over dynamic shared
// memory: a warp shuffle into 32 floats of static shared memory, then a serial
// sum by thread 0. Rebuilding it on either of the others would reorder the
// summation and move Qwen3.5's q/k bits, so it keeps its own translation unit
// for the same reason norm_offset.cu does.

#include "norm_head.h"

#include <cuda_bf16.h>

namespace apxinf::cuda_new::norm_head {
namespace {

// Per-head RMSNorm without a gate, for q_norm and k_norm.
__global__ void head_rms_norm_kernel(__nv_bfloat16* __restrict__ data,
                                     const __nv_bfloat16* __restrict__ weight,
                                     int rows, int head_dim, float epsilon) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const long long base = (long long)row * head_dim;

  float sum = 0.0f;
  for (int index = threadIdx.x; index < head_dim; index += blockDim.x) {
    const float value = __bfloat162float(data[base + index]);
    sum += value * value;
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xFFFFFFFFu, sum, offset);
  }
  __shared__ float partial[32];
  const int warp = threadIdx.x >> 5;
  if ((threadIdx.x & 31) == 0) partial[warp] = sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    const int warps = (blockDim.x + 31) / 32;
    float total = 0.0f;
    for (int index = 0; index < warps; ++index) total += partial[index];
    partial[0] = total;
  }
  __syncthreads();

  const float scale = rsqrtf(partial[0] / static_cast<float>(head_dim) + epsilon);
  for (int index = threadIdx.x; index < head_dim; index += blockDim.x) {
    data[base + index] = __float2bfloat16(
        __bfloat162float(data[base + index]) * scale *
        (1.0f + __bfloat162float(weight[index])));
  }
}

}  // namespace

int head_rms_norm(void* data, const void* weight, int rows, int head_dim,
                  float epsilon, cudaStream_t stream) {
  if (rows <= 0 || head_dim <= 0) return -1;
  const int threads = head_dim >= 256 ? 256 : ((head_dim + 31) / 32) * 32;
  head_rms_norm_kernel<<<rows, threads, 0, stream>>>(
      static_cast<__nv_bfloat16*>(data),
      static_cast<const __nv_bfloat16*>(weight), rows, head_dim, epsilon);
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

}  // namespace apxinf::cuda_new::norm_head
