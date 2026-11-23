// Copyright 2026 ApxInf contributors.
// RMSNorm +1 kernels moved mechanically from the historical aggregate.
#pragma once

#include "../custom/rms_reduction.cuh"

// RMSNorm with zero-init (1 + w) weights: out = rms(x) * (1 + w), fp32 compute.
__global__ void rms_norm_plus1_bf16_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* weight, __nv_bfloat16* output,
    int cols, float eps) {
  const int row = blockIdx.x;
  extern __shared__ float la_plus1[];
  const int64_t base = static_cast<int64_t>(row) * cols;
  float partial = 0.0f;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    const float v = __bfloat162float(input[base + i]);
    la_plus1[i] = v;
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
  // Preserve the reference mean-reduction and epsilon rounding boundaries.
  // Division followed by a fused add can cross a BF16 rounding boundary.
  float mean = __fmul_rn(warp_sums[0], __fdiv_rn(1.0f, static_cast<float>(cols)));
  if (gridDim.x >= 16 && cols > 128 && cols % 4 == 0)
    mean = rms_vector_square_mean_bf16(input + base, cols);
  const float rms = rsqrtf(__fadd_rn(mean, eps));
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    output[base + i] = __float2bfloat16(
        __fmul_rn(__fmul_rn(la_plus1[i], rms), __fadd_rn(1.0f, __bfloat162float(weight[i]))));
  }
}

// Identical four-accumulator/warp summation order; input is the existing
// shared FP32 conversion of the explicitly BF16-rounded sum.
__device__ __forceinline__ float rms_vector_square_mean_f32_staged(
    const float* input, int cols) {
  __shared__ float mean;
  if (threadIdx.x < 32) {
    float sums[4] = {0, 0, 0, 0};
    for (int col = threadIdx.x * 4; col < cols; col += 128) {
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float value = input[col + j];
        sums[j] = __fadd_rn(sums[j], __fmul_rn(value, value));
      }
    }
    float sum = ((sums[0] + sums[1]) + sums[2]) + sums[3];
    for (int offset = 16; offset > 0; offset >>= 1)
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (threadIdx.x == 0)
      mean = __fmul_rn(sum, __fdiv_rn(1.f, static_cast<float>(cols)));
  }
  __syncthreads();
  return mean;
}

// The residual add fused into the norm that always follows it.
//
// x = a + b; out = rms_norm_plus1(x). Both halves of every text block end this
// way -- sixty-four pairs per decoded token -- and at decode each is one
// 2560-element row, a couple of microseconds of work behind a kernel's fixed
// cost. Fusing removes one launch and one round trip through the sum per pair.
//
// The sum still goes to memory in BF16. The default path retains the
// separate norm's reduction and its vector-mean reload. The prefill template
// reads that same BF16-rounded value from the existing shared stage while
// keeping the vector mean's four-accumulator order. Both keep the explicit
// (1 + weight) and BF16 output rounding boundaries.
template <bool STAGED_MEAN = false>
__global__ void add_rms_norm_plus1_bf16_kernel(
    const __nv_bfloat16* a, const __nv_bfloat16* b,
    const __nv_bfloat16* weight, __nv_bfloat16* sum_out, __nv_bfloat16* output,
    int cols, float eps) {
  const int row = blockIdx.x;
  extern __shared__ float la_fused_plus1[];
  const int64_t base = static_cast<int64_t>(row) * cols;
  // The add it replaces was a vec8 kernel spread over the whole tensor; this
  // one has a block per row, so a scalar add here is a step backwards and was
  // measured as one -- 17.8us per pair against the 15.9us of the two separate
  // kernels. Eight bf16 per instruction restores it.
  //
  // The rounded sum goes into shared on the way past, so the reduction below
  // never reads it back from global. la_fused_plus1[i] holds exactly what the
  // separate norm would have loaded, and the reduction still walks i in the
  // same strided order, so the sum of squares is unchanged.
  const bool wide =
      cols % 8 == 0 &&
      ((reinterpret_cast<uintptr_t>(a + base) |
        reinterpret_cast<uintptr_t>(b + base) |
        reinterpret_cast<uintptr_t>(sum_out + base)) &
       15) == 0;
  if (wide) {
    const float4* a4 = reinterpret_cast<const float4*>(a + base);
    const float4* b4 = reinterpret_cast<const float4*>(b + base);
    float4* s4 = reinterpret_cast<float4*>(sum_out + base);
    const int vec_count = cols / 8;
    for (int v = threadIdx.x; v < vec_count; v += blockDim.x) {
      float4 pa = a4[v];
      float4 pb = b4[v];
      float4 po;
      const __nv_bfloat16* av = reinterpret_cast<const __nv_bfloat16*>(&pa);
      const __nv_bfloat16* bv = reinterpret_cast<const __nv_bfloat16*>(&pb);
      __nv_bfloat16* ov = reinterpret_cast<__nv_bfloat16*>(&po);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const __nv_bfloat16 t = __float2bfloat16(
            __bfloat162float(av[j]) + __bfloat162float(bv[j]));
        ov[j] = t;
        la_fused_plus1[v * 8 + j] = __bfloat162float(t);
      }
      s4[v] = po;
    }
  } else {
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
      const __nv_bfloat16 t = __float2bfloat16(
          __bfloat162float(a[base + i]) + __bfloat162float(b[base + i]));
      sum_out[base + i] = t;
      la_fused_plus1[i] = __bfloat162float(t);
    }
  }
  // Publish the explicitly BF16-rounded sum before either reduction reads it.
  __syncthreads();
  float mean;
  if constexpr (STAGED_MEAN) {
    // Prefill shape: the vector mean below overwrites the generic reduction.
    // Read identical rounded values from shared with its original FP32 order.
    mean = rms_vector_square_mean_f32_staged(la_fused_plus1, cols);
  } else {
    float partial = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
      const float v = la_fused_plus1[i];
      partial += v * v;
    }
    __shared__ float fused_warp_sums[32];
    for (int offset = 16; offset > 0; offset >>= 1)
      partial += __shfl_xor_sync(0xffffffff, partial, offset);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) fused_warp_sums[warp] = partial;
    __syncthreads();
    if (warp == 0) {
      float v = (lane < (blockDim.x + 31) / 32) ? fused_warp_sums[lane] : 0.0f;
      for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, offset);
      if (lane == 0) fused_warp_sums[0] = v;
    }
    __syncthreads();
    mean =
        __fmul_rn(fused_warp_sums[0], __fdiv_rn(1.0f, static_cast<float>(cols)));
    if (gridDim.x >= 16 && cols > 128 && cols % 4 == 0)
      mean = rms_vector_square_mean_bf16(sum_out + base, cols);
  }
  const float rms = rsqrtf(__fadd_rn(mean, eps));
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    output[base + i] = __float2bfloat16(__fmul_rn(
        __fmul_rn(la_fused_plus1[i], rms),
        __fadd_rn(1.0f, __bfloat162float(weight[i]))));
  }
}
