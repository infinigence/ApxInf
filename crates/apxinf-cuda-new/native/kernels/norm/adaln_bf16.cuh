// Copyright 2026 ApxInf contributors.
// AdaLN kernels moved mechanically from the historical aggregate.
#pragma once

#include "../custom/rms_reduction.cuh"

// adaLN normalization: out = bf16(bf16(rms(x)*w) * bf16(1 + scale) + shift).
__global__ void adaln_rms_norm_bf16_kernel(
    const __nv_bfloat16* x, const __nv_bfloat16* weight,
    const __nv_bfloat16* scale, const __nv_bfloat16* shift,
    __nv_bfloat16* out, int cols, float eps) {
  const int row = blockIdx.x;
  extern __shared__ float la_adaln[];
  const int64_t base = static_cast<int64_t>(row) * cols;
  float partial = 0.0f;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    const float v = __bfloat162float(x[base + i]);
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
    mean = rms_vector_square_mean_bf16(x + base, cols);
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

// adaLN residual: out = bf16(residual + bf16(proj * bf16(1 + gate))).
__global__ void adaln_gate_residual_bf16_kernel(
    const __nv_bfloat16* proj, const __nv_bfloat16* residual,
    const __nv_bfloat16* gate, __nv_bfloat16* out, int64_t count, int cols) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    const int col = static_cast<int>(index % cols);
    const float multiplier = __bfloat162float(
        __float2bfloat16(1.0f + __bfloat162float(gate[col])));
    const float projected = __bfloat162float(
        __float2bfloat16(__bfloat162float(proj[index]) * multiplier));
    out[index] = __float2bfloat16(__bfloat162float(residual[index]) + projected);
  }
}
