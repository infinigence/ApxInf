// Copyright 2026 ApxInf contributors.
// Fixed implementation for this operator family; ported without numerical changes.
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace {
#include "adaln_bf16.cuh"
#include "gated_rms_bf16.cuh"
#include "rms_plus1_bf16.cuh"
__global__ void adaln_gate_residual_rms_bf16_kernel(
    const __nv_bfloat16* proj, const __nv_bfloat16* residual,
    const __nv_bfloat16* gate, const __nv_bfloat16* weight,
    const __nv_bfloat16* scale, const __nv_bfloat16* shift,
    __nv_bfloat16* hidden, __nv_bfloat16* out, int cols, float eps) {
  const int row = blockIdx.x;
  extern __shared__ float la_adaln[];
  const int64_t base = static_cast<int64_t>(row) * cols;
  float partial = 0.0f;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    const float multiplier = __bfloat162float(
        __float2bfloat16(1.0f + __bfloat162float(gate[i])));
    const float projected = __bfloat162float(
        __float2bfloat16(__bfloat162float(proj[base + i]) * multiplier));
    const __nv_bfloat16 rounded =
        __float2bfloat16(__bfloat162float(residual[base + i]) + projected);
    hidden[base + i] = rounded;
    const float v = __bfloat162float(rounded);
    la_adaln[i] = v;
    partial += v * v;
  }
  __shared__ float warp_sums[32];
  for (int offset = 16; offset > 0; offset >>= 1)
    partial += __shfl_xor_sync(0xffffffff, partial, offset);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) warp_sums[warp] = partial;
  __syncthreads();
  if (warp == 0) {
    float v = (lane < (blockDim.x + 31) / 32) ? warp_sums[lane] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1)
      v += __shfl_xor_sync(0xffffffff, v, offset);
    if (lane == 0) warp_sums[0] = v;
  }
  __syncthreads();
  float mean = warp_sums[0] / cols;
  if (gridDim.x >= 16 && cols > 128 && cols % 4 == 0)
    mean = rms_vector_square_mean_bf16(hidden + base, cols);
  const float rms = rsqrtf(__fadd_rn(mean, eps));
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    const float normed = __bfloat162float(__float2bfloat16(
        la_adaln[i] * rms * __bfloat162float(weight[i])));
    const float multiplier = __bfloat162float(
        __float2bfloat16(1.0f + __bfloat162float(scale[i])));
    const float scaled = __bfloat162float(__float2bfloat16(normed * multiplier));
    out[base + i] = __float2bfloat16(scaled + __bfloat162float(shift[i]));
  }
}

}  // namespace
