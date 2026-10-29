// Copyright 2026 ApxInf contributors.
// Fixed BF16 implementation for this operator family; ported without numerical changes.
#include "../custom/reduction.cuh"
#ifdef APXINF_GEMM_CUTLASS_SM87_W8A8
#include "../cutlass/ops/gemm/gemm_i8_bf16_sm80.h"
#endif

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {

constexpr int kBlockSize = 256;

__device__ __forceinline__ float gelu_tanh(float value) {
  constexpr float kAlpha = 0.7978845608028654f;
  return 0.5f * value *
         (1.0f + tanhf(kAlpha * (value + 0.044715f * value * value * value)));
}

struct alignas(8) Bf16x4 {
  __nv_bfloat162 low;
  __nv_bfloat162 high;
};

struct alignas(8) W8Bf16x4 {
  __nv_bfloat162 low;
  __nv_bfloat162 high;
};

struct alignas(16) Bf16Pairx8 {
  Bf16x4 first;
  Bf16x4 second;
};

// ── Elementwise / selection primitives ───────────────────────────────────
// Copied verbatim from the legacy kernels/custom/elementwise.cuh and
// kernels/custom/activation.cuh.

__global__ void scatter_rows_bf16_kernel(
    const __nv_bfloat16* source, const uint32_t* rows,
    __nv_bfloat16* output, int64_t count, int cols, bool add) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    const int source_row = static_cast<int>(index / cols);
    const int column = static_cast<int>(index % cols);
    const int64_t output_index =
        static_cast<int64_t>(rows[source_row]) * cols + column;
    if (add) {
      output[output_index] = __float2bfloat16(
          __bfloat162float(output[output_index]) + __bfloat162float(source[index]));
    } else {
      output[output_index] = source[index];
    }
  }
}

__global__ void bias_qkv_in_place_bf16_packed4_kernel(
    __nv_bfloat16* query, __nv_bfloat16* key, __nv_bfloat16* value,
    const __nv_bfloat16* query_bias, const __nv_bfloat16* key_bias,
    const __nv_bfloat16* value_bias, int64_t group_count,
    int groups_per_row) {
  int64_t group = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; group < group_count; group += stride) {
    const int bias_group = static_cast<int>(group % groups_per_row);
    const __nv_bfloat162* qb = reinterpret_cast<const __nv_bfloat162*>(query_bias) + 2 * bias_group;
    const __nv_bfloat162* kb = reinterpret_cast<const __nv_bfloat162*>(key_bias) + 2 * bias_group;
    const __nv_bfloat162* vb = reinterpret_cast<const __nv_bfloat162*>(value_bias) + 2 * bias_group;
    __nv_bfloat162* q = reinterpret_cast<__nv_bfloat162*>(query) + 2 * group;
    __nv_bfloat162* k = reinterpret_cast<__nv_bfloat162*>(key) + 2 * group;
    __nv_bfloat162* v = reinterpret_cast<__nv_bfloat162*>(value) + 2 * group;
    const float2 q0 = __bfloat1622float2(q[0]);
    const float2 q1 = __bfloat1622float2(q[1]);
    const float2 k0 = __bfloat1622float2(k[0]);
    const float2 k1 = __bfloat1622float2(k[1]);
    const float2 v0 = __bfloat1622float2(v[0]);
    const float2 v1 = __bfloat1622float2(v[1]);
    const float2 qb0 = __bfloat1622float2(qb[0]);
    const float2 qb1 = __bfloat1622float2(qb[1]);
    const float2 kb0 = __bfloat1622float2(kb[0]);
    const float2 kb1 = __bfloat1622float2(kb[1]);
    const float2 vb0 = __bfloat1622float2(vb[0]);
    const float2 vb1 = __bfloat1622float2(vb[1]);
    q[0] = __floats2bfloat162_rn(q0.x + qb0.x, q0.y + qb0.y);
    q[1] = __floats2bfloat162_rn(q1.x + qb1.x, q1.y + qb1.y);
    k[0] = __floats2bfloat162_rn(k0.x + kb0.x, k0.y + kb0.y);
    k[1] = __floats2bfloat162_rn(k1.x + kb1.x, k1.y + kb1.y);
    v[0] = __floats2bfloat162_rn(v0.x + vb0.x, v0.y + vb0.y);
    v[1] = __floats2bfloat162_rn(v1.x + vb1.x, v1.y + vb1.y);
  }
}

}  // namespace

// ── Elementwise / selection launchers ────────────────────────────────────
// Copied verbatim from the legacy static_bf16_adapter.cu, renamed with the
// family-specific elementwise ABI.
