// Copyright 2026 ApxInf contributors.
// GR00T-family fused activation primitives, ported verbatim from the legacy
// apxinf-cuda adapters (core_kernels_adapter.cu, custom_kernels.cu,
// static_bf16_adapter.cu) and kernels/custom/activation.cuh. Symbols carry the
// apxinf_gr00t_ prefix because the legacy crate links into the same binary and
// its own launchers are already present. The copied kernels live in an
// anonymous namespace so their names cannot collide with the identically named
// kernels that cuda-new already builds in pointwise.cu / mlp_ops.cu.

#include "../../include/apxinf_cuda/gr00t.h"
#include "../../kernels/custom/reduction.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {

constexpr int kGr00tBlockSize = 256;

__device__ __forceinline__ float gelu_tanh(float value) {
  constexpr float kAlpha = 0.7978845608028654f;
  return 0.5f * value *
         (1.0f + tanhf(kAlpha * (value + 0.044715f * value * value * value)));
}

struct alignas(8) Bf16x4 {
  __nv_bfloat162 low;
  __nv_bfloat162 high;
};

struct alignas(8) Gr00tW8Bf16x4 {
  __nv_bfloat162 low;
  __nv_bfloat162 high;
};

struct alignas(16) Bf16Pairx8 {
  Bf16x4 first;
  Bf16x4 second;
};

// Separate-input variant used when gate/up projections are not packed.  The
// intermediate SiLU value is explicitly rounded to BF16 so this remains
// bit-compatible with the former `silu` then `mul` kernel sequence.
__global__ void silu_mul_separate_bf16_kernel(
    const __nv_bfloat16* gate, const __nv_bfloat16* up,
    __nv_bfloat16* output, uint32_t count) {
  const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count) return;
  const float g = __bfloat162float(gate[gid]);
  const __nv_bfloat16 activated = __float2bfloat16(g / (1.0f + expf(-g)));
  output[gid] = __float2bfloat16(
      __bfloat162float(activated) * __bfloat162float(up[gid]));
}

__global__ void silu_mul_separate_bf16_packed4_kernel(
    const Bf16x4* gate, const Bf16x4* up, Bf16x4* output,
    uint32_t quad_count) {
  uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t stride = blockDim.x * gridDim.x;
  for (; index < quad_count; index += stride) {
    const Bf16x4 g = gate[index];
    const Bf16x4 u = up[index];
    float values[4] = {
        __bfloat162float(g.low.x), __bfloat162float(g.low.y),
        __bfloat162float(g.high.x), __bfloat162float(g.high.y)};
    const float ups[4] = {
        __bfloat162float(u.low.x), __bfloat162float(u.low.y),
        __bfloat162float(u.high.x), __bfloat162float(u.high.y)};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const __nv_bfloat16 activated =
          __float2bfloat16(values[i] / (1.0f + expf(-values[i])));
      values[i] = __bfloat162float(activated) * ups[i];
    }
    output[index] = Bf16x4{
        __floats2bfloat162_rn(values[0], values[1]),
        __floats2bfloat162_rn(values[2], values[3])};
  }
}

// Preserve the existing three-stage numerical contract while avoiding the
// two BF16 intermediates: SiLU rounds to BF16, multiplication rounds to BF16,
// then the calibrated value converts to saturating E4M3.
__global__ void silu_mul_quant_bf16_e4m3_kernel(
    const __nv_bfloat16* gate, const __nv_bfloat16* up,
    __nv_fp8_e4m3* output, int64_t count, float inverse_scale) {
  int64_t pair = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t pair_count = count / 2;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; pair < pair_count; pair += stride) {
    const int64_t index = pair * 2;
    const __nv_bfloat162 gate2 =
        reinterpret_cast<const __nv_bfloat162*>(gate)[pair];
    const __nv_bfloat162 up2 =
        reinterpret_cast<const __nv_bfloat162*>(up)[pair];
    const float gx = __bfloat162float(gate2.x);
    const float gy = __bfloat162float(gate2.y);
    const __nv_bfloat16 sx =
        __float2bfloat16(gx / (1.0f + expf(-gx)));
    const __nv_bfloat16 sy =
        __float2bfloat16(gy / (1.0f + expf(-gy)));
    const __nv_bfloat16 px = __float2bfloat16(
        __bfloat162float(sx) * __bfloat162float(up2.x));
    const __nv_bfloat16 py = __float2bfloat16(
        __bfloat162float(sy) * __bfloat162float(up2.y));
    reinterpret_cast<__nv_fp8x2_e4m3*>(output)[pair] =
        __nv_fp8x2_e4m3(make_float2(
            __bfloat162float(px) * inverse_scale,
            __bfloat162float(py) * inverse_scale));
  }
  if (count % 2 != 0 && pair == pair_count) {
    const float g = __bfloat162float(gate[count - 1]);
    const __nv_bfloat16 s = __float2bfloat16(g / (1.0f + expf(-g)));
    const __nv_bfloat16 product = __float2bfloat16(
        __bfloat162float(s) * __bfloat162float(up[count - 1]));
    output[count - 1] = static_cast<__nv_fp8_e4m3>(
        __bfloat162float(product) * inverse_scale);
  }
}

__global__ void bias_gelu_quant_bf16_e4m3_packed4_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* bias,
    __nv_fp8_e4m3* output, int64_t quad_count, int cols,
    float inverse_scale) {
  int64_t quad_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  const int quads_per_row = cols / 4;
  const Bf16x4* input4 = reinterpret_cast<const Bf16x4*>(input);
  const Bf16x4* bias4 = reinterpret_cast<const Bf16x4*>(bias);
  for (; quad_index < quad_count; quad_index += stride) {
    const Bf16x4 x = input4[quad_index];
    const Bf16x4 b = bias4[quad_index % quads_per_row];
    const float y0 = __bfloat162float(__float2bfloat16(gelu_tanh(
        __bfloat162float(x.low.x) + __bfloat162float(b.low.x))));
    const float y1 = __bfloat162float(__float2bfloat16(gelu_tanh(
        __bfloat162float(x.low.y) + __bfloat162float(b.low.y))));
    const float y2 = __bfloat162float(__float2bfloat16(gelu_tanh(
        __bfloat162float(x.high.x) + __bfloat162float(b.high.x))));
    const float y3 = __bfloat162float(__float2bfloat16(gelu_tanh(
        __bfloat162float(x.high.y) + __bfloat162float(b.high.y))));
    const __nv_fp8x2_e4m3 first(make_float2(
        y0 * inverse_scale, y1 * inverse_scale));
    const __nv_fp8x2_e4m3 second(make_float2(
        y2 * inverse_scale, y3 * inverse_scale));
    reinterpret_cast<uint32_t*>(output)[quad_index] =
        static_cast<uint32_t>(first.__x) |
        (static_cast<uint32_t>(second.__x) << 16);
  }
}

__device__ __forceinline__ Bf16x4 bias_gelu_bf16_quad(
    const Bf16x4 input, const Bf16x4 bias) {
  const float v0 = gelu_tanh(
      __bfloat162float(input.low.x) + __bfloat162float(bias.low.x));
  const float v1 = gelu_tanh(
      __bfloat162float(input.low.y) + __bfloat162float(bias.low.y));
  const float v2 = gelu_tanh(
      __bfloat162float(input.high.x) + __bfloat162float(bias.high.x));
  const float v3 = gelu_tanh(
      __bfloat162float(input.high.y) + __bfloat162float(bias.high.y));
  return Bf16x4{
      __floats2bfloat162_rn(v0, v1), __floats2bfloat162_rn(v2, v3)};
}

__global__ void bias_gelu_bf16_packed8_kernel(
    const Bf16Pairx8* input, const Bf16Pairx8* bias, Bf16Pairx8* output,
    int64_t octet_count, int octets_per_row) {
  int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < octet_count; index += stride) {
    const Bf16Pairx8 value = input[index];
    const Bf16Pairx8 row_bias = bias[index % octets_per_row];
    output[index] = Bf16Pairx8{
        bias_gelu_bf16_quad(value.first, row_bias.first),
        bias_gelu_bf16_quad(value.second, row_bias.second)};
  }
}

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

// ── Norm kernels (BF16-input variants) ───────────────────────────────────
// cuda-new's own norm kernels take F16 input for the E4M3-output forms; the
// gr00t path needs BF16-input versions, so these are copied verbatim from the
// legacy normalization.cuh. `block_sum_parallel_unsafe` is cuda-new's own
// reduction helper (reduction.cuh), identical to the legacy one.

__global__ void gr00t_adaptive_layer_norm_bf16_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* modulation,
    __nv_bfloat16* output, uint32_t rows, uint32_t cols, float eps) {
  __shared__ float scratch[16];
  const uint32_t row = blockIdx.x;
  if (row >= rows) return;

  float sum = 0.0f;
  for (uint32_t col = threadIdx.x; col < cols; col += blockDim.x)
    sum += __bfloat162float(input[(uint64_t)row * cols + col]);
  const float mean = block_sum_parallel_unsafe(sum, scratch) / cols;

  float variance_sum = 0.0f;
  for (uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
    const float centered =
        __bfloat162float(input[(uint64_t)row * cols + col]) - mean;
    variance_sum += centered * centered;
  }
  __syncthreads();
  const float inverse_std =
      rsqrtf(block_sum_parallel_unsafe(variance_sum, scratch) / cols + eps);

  for (uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
    const uint64_t index = (uint64_t)row * cols + col;
    const float normalized =
        (__bfloat162float(input[index]) - mean) * inverse_std;
    const float scale = __bfloat162float(modulation[col]);
    const float shift = __bfloat162float(modulation[cols + col]);
    output[index] = __float2bfloat16(normalized * (1.0f + scale) + shift);
  }
}

__global__ void gr00t_adaptive_layer_norm_quant_bf16_e4m3_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* modulation,
    __nv_bfloat16* output, __nv_fp8_e4m3* quantized, uint32_t rows,
    uint32_t cols, float eps, float inverse_scale) {
  __shared__ float scratch[16];
  const uint32_t row = blockIdx.x;
  if (row >= rows) return;

  float sum = 0.0f;
  for (uint32_t col = threadIdx.x; col < cols; col += blockDim.x)
    sum += __bfloat162float(input[(uint64_t)row * cols + col]);
  const float mean = block_sum_parallel_unsafe(sum, scratch) / cols;

  float variance_sum = 0.0f;
  for (uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
    const float centered =
        __bfloat162float(input[(uint64_t)row * cols + col]) - mean;
    variance_sum += centered * centered;
  }
  __syncthreads();
  const float inverse_std =
      rsqrtf(block_sum_parallel_unsafe(variance_sum, scratch) / cols + eps);

  for (uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
    const uint64_t index = (uint64_t)row * cols + col;
    const float normalized =
        (__bfloat162float(input[index]) - mean) * inverse_std;
    const float scale = __bfloat162float(modulation[col]);
    const float shift = __bfloat162float(modulation[cols + col]);
    const __nv_bfloat16 rounded =
        __float2bfloat16(normalized * (1.0f + scale) + shift);
    output[index] = rounded;
    float value = __bfloat162float(rounded) * inverse_scale;
    value = fminf(448.0f, fmaxf(-448.0f, value));
    quantized[index] = static_cast<__nv_fp8_e4m3>(value);
  }
}

__global__ void gr00t_rms_norm_quant_bf16_e4m3_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    __nv_fp8_e4m3* output, int rows, int cols, float eps,
    float inverse_scale) {
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  float square_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float value =
        __bfloat162float(input[static_cast<int64_t>(row) * cols + col]);
    square_sum += value * value;
  }
  const float inverse_rms =
      rsqrtf(block_sum_parallel_unsafe(square_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    float value = __bfloat162float(input[index]) * inverse_rms *
                  __bfloat162float(weight[col]) * inverse_scale;
    value = fminf(448.0f, fmaxf(-448.0f, value));
    output[index] = static_cast<__nv_fp8_e4m3>(value);
  }
}

__global__ void gr00t_layer_norm_quant_bf16_e4m3_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    const __nv_bfloat16* bias, __nv_fp8_e4m3* output, int rows, int cols,
    float eps, float inverse_scale) {
  extern __shared__ float x_buf[];
  __shared__ float scratch[16];
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int offset = row * cols;

  float sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float value = __bfloat162float(input[offset + col]);
    x_buf[col] = value;
    sum += value;
  }
  const float mean = block_sum_parallel_unsafe(sum, scratch) / cols;

  float variance_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float centered = x_buf[col] - mean;
    variance_sum += centered * centered;
  }
  __syncthreads();
  const float inverse_std =
      rsqrtf(block_sum_parallel_unsafe(variance_sum, scratch) / cols + eps);

  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    float value = (x_buf[col] - mean) * inverse_std;
    value = value * __bfloat162float(weight[col]) +
            __bfloat162float(bias[col]);
    value = __bfloat162float(__float2bfloat16(value));
    value = fminf(448.0f, fmaxf(-448.0f, value * inverse_scale));
    output[offset + col] = static_cast<__nv_fp8_e4m3>(value);
  }
}


// ── Fused residual / layer-norm kernels (from legacy fused.cuh) ─────────
__global__ void bias_residual_bf16_packed4_kernel(
    const Bf16x4* projection, const Bf16x4* bias,
    const Bf16x4* residual, Bf16x4* output,
    int64_t packed_count, int packed_cols) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < packed_count; index += stride) {
    const Bf16x4 projected = projection[index];
    const Bf16x4 skipped = residual[index];
    const Bf16x4 bias_value = bias[index % packed_cols];
    const float2 projection_low = __bfloat1622float2(projected.low);
    const float2 projection_high = __bfloat1622float2(projected.high);
    const float2 residual_low = __bfloat1622float2(skipped.low);
    const float2 residual_high = __bfloat1622float2(skipped.high);
    const float2 bias_low = __bfloat1622float2(bias_value.low);
    const float2 bias_high = __bfloat1622float2(bias_value.high);
    output[index] = Bf16x4{
        __floats2bfloat162_rn(
            projection_low.x + residual_low.x + bias_low.x,
            projection_low.y + residual_low.y + bias_low.y),
        __floats2bfloat162_rn(
            projection_high.x + residual_high.x + bias_high.x,
            projection_high.y + residual_high.y + bias_high.y)};
  }
}

__global__ void bias_then_residual_bf16_packed4_kernel(
    const Bf16x4* projection, const Bf16x4* bias,
    const Bf16x4* residual, Bf16x4* output,
    int64_t packed_count, int packed_cols) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < packed_count; index += stride) {
    const Bf16x4 projected = projection[index];
    float2 projection_low = __bfloat1622float2(projected.low);
    float2 projection_high = __bfloat1622float2(projected.high);
    if (bias != nullptr) {
      const Bf16x4 bias_value = bias[index % packed_cols];
      const float2 bias_low = __bfloat1622float2(bias_value.low);
      const float2 bias_high = __bfloat1622float2(bias_value.high);
      projection_low.x += bias_low.x;
      projection_low.y += bias_low.y;
      projection_high.x += bias_high.x;
      projection_high.y += bias_high.y;
    }
    const Bf16x4 biased{
        __floats2bfloat162_rn(projection_low.x, projection_low.y),
        __floats2bfloat162_rn(projection_high.x, projection_high.y)};
    float2 biased_low = __bfloat1622float2(biased.low);
    float2 biased_high = __bfloat1622float2(biased.high);
    const Bf16x4 skipped = residual[index];
    const float2 residual_low = __bfloat1622float2(skipped.low);
    const float2 residual_high = __bfloat1622float2(skipped.high);
    biased_low.x += residual_low.x;
    biased_low.y += residual_low.y;
    biased_high.x += residual_high.x;
    biased_high.y += residual_high.y;
    output[index] = Bf16x4{
        __floats2bfloat162_rn(biased_low.x, biased_low.y),
        __floats2bfloat162_rn(biased_high.x, biased_high.y)};
  }
}

__global__ void bias_then_residual_bf16_kernel(
    const __nv_bfloat16* projection, const __nv_bfloat16* bias,
    const __nv_bfloat16* residual, __nv_bfloat16* output,
    int64_t count, int cols) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    float value = __bfloat162float(projection[index]);
    if (bias != nullptr) value += __bfloat162float(bias[index % cols]);
    const __nv_bfloat16 biased = __float2bfloat16(value);
    output[index] = __float2bfloat16(
        __bfloat162float(biased) + __bfloat162float(residual[index]));
  }
}

__global__ void bias_residual_layer_norm_quant_bf16_e4m3_kernel(
    const __nv_bfloat16* projection, const __nv_bfloat16* projection_bias,
    const __nv_bfloat16* residual, const __nv_bfloat16* norm_weight,
    const __nv_bfloat16* norm_bias, __nv_bfloat16* hidden,
    __nv_fp8_e4m3* normalized, int rows, int cols, float eps,
    float inverse_scale) {
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  float sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    float value = __bfloat162float(projection[index]) +
                  __bfloat162float(residual[index]);
    if (projection_bias != nullptr) value += __bfloat162float(projection_bias[col]);
    const __nv_bfloat16 rounded = __float2bfloat16(value);
    hidden[index] = rounded;
    sum += __bfloat162float(rounded);
  }
  const float mean = block_sum_parallel_unsafe(sum, scratch) / cols;
  float variance_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float centered =
        __bfloat162float(hidden[static_cast<int64_t>(row) * cols + col]) - mean;
    variance_sum += centered * centered;
  }
  // Finish reading the previous reduction before reusing scratch.
  __syncthreads();
  const float inverse_std =
      rsqrtf(block_sum_parallel_unsafe(variance_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    float value =
        (__bfloat162float(hidden[index]) - mean) * inverse_std *
            __bfloat162float(norm_weight[col]) +
        __bfloat162float(norm_bias[col]);
    // Match bias_residual_layer_norm_bf16 followed by quantize_bf16_e4m3.
    value = __bfloat162float(__float2bfloat16(value));
    value = fminf(448.0f, fmaxf(-448.0f, value * inverse_scale));
    normalized[index] = static_cast<__nv_fp8_e4m3>(value);
  }
}
__global__ void
bias_then_residual_adaptive_layer_norm_bf16_cached_1536_kernel(
    const __nv_bfloat16* projection, const __nv_bfloat16* projection_bias,
    const __nv_bfloat16* residual, const __nv_bfloat16* modulation,
    __nv_bfloat16* hidden, __nv_bfloat16* normalized, int rows, float eps) {
  __shared__ float scratch[16];
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int64_t base = static_cast<int64_t>(row) * 1536;
  float cache[6];
#pragma unroll
  for (int i = 0; i < 6; ++i) {
    const int col = threadIdx.x + i * static_cast<int>(blockDim.x);
    const __nv_bfloat16 biased = __float2bfloat16(
        __bfloat162float(projection[base + col]) +
        __bfloat162float(projection_bias[col]));
    const __nv_bfloat16 rounded = __float2bfloat16(
        __bfloat162float(biased) + __bfloat162float(residual[base + col]));
    hidden[base + col] = rounded;
    cache[i] = __bfloat162float(rounded);
  }
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < 6; ++i) sum += cache[i];
  const float mean = block_sum_parallel_unsafe(sum, scratch) / 1536.0f;
  float variance_sum = 0.0f;
#pragma unroll
  for (int i = 0; i < 6; ++i) {
    const float centered = cache[i] - mean;
    variance_sum += centered * centered;
  }
  __syncthreads();
  const float inverse_std =
      rsqrtf(block_sum_parallel_unsafe(variance_sum, scratch) / 1536.0f + eps);
#pragma unroll
  for (int i = 0; i < 6; ++i) {
    const int col = threadIdx.x + i * static_cast<int>(blockDim.x);
    const float value = (cache[i] - mean) * inverse_std;
    const float scale = __bfloat162float(modulation[col]);
    const float shift = __bfloat162float(modulation[1536 + col]);
    normalized[base + col] =
        __float2bfloat16(value * (1.0f + scale) + shift);
  }
}

// Exact-shape opt-in for the GR00T DiT attention-output boundary. Unlike the
// generic fused residual path, this retains the legacy bias kernel's BF16
// boundary before the residual add. Each of the 256 threads owns the same six
// columns, in the same order, as layer_norm_bf16_kernel at width 1536.
__global__ void bias_then_residual_layer_norm_bf16_cached_1536_kernel(
    const __nv_bfloat16* projection, const __nv_bfloat16* projection_bias,
    const __nv_bfloat16* residual, const __nv_bfloat16* norm_weight,
    const __nv_bfloat16* norm_bias, __nv_bfloat16* hidden,
    __nv_bfloat16* normalized, int rows, float eps) {
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int64_t base = static_cast<int64_t>(row) * 1536;
  float cache[6];
#pragma unroll
  for (int i = 0; i < 6; ++i) {
    const int col = threadIdx.x + i * static_cast<int>(blockDim.x);
    const __nv_bfloat16 biased = __float2bfloat16(
        __bfloat162float(projection[base + col]) +
        __bfloat162float(projection_bias[col]));
    const __nv_bfloat16 rounded = __float2bfloat16(
        __bfloat162float(biased) + __bfloat162float(residual[base + col]));
    hidden[base + col] = rounded;
    cache[i] = __bfloat162float(rounded);
  }
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < 6; ++i) sum += cache[i];
  const float mean = block_sum_parallel_unsafe(sum, scratch) / 1536.0f;
  float variance_sum = 0.0f;
#pragma unroll
  for (int i = 0; i < 6; ++i) {
    const float centered = cache[i] - mean;
    variance_sum += centered * centered;
  }
  __syncthreads();
  const float inverse_std =
      rsqrtf(block_sum_parallel_unsafe(variance_sum, scratch) / 1536.0f + eps);
#pragma unroll
  for (int i = 0; i < 6; ++i) {
    const int col = threadIdx.x + i * static_cast<int>(blockDim.x);
    normalized[base + col] = __float2bfloat16(
        (cache[i] - mean) * inverse_std *
            __bfloat162float(norm_weight[col]) +
        __bfloat162float(norm_bias[col]));
  }
}

__global__ void bias_residual_layer_norm_bf16_cached_1024_kernel(
    const __nv_bfloat16* projection, const __nv_bfloat16* projection_bias,
    const __nv_bfloat16* residual, const __nv_bfloat16* norm_weight,
    const __nv_bfloat16* norm_bias, __nv_bfloat16* hidden,
    __nv_bfloat16* normalized, int rows, float eps) {
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  const int64_t base = static_cast<int64_t>(row) * 1024;
  float cache[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int col = threadIdx.x + i * static_cast<int>(blockDim.x);
    float value = __bfloat162float(projection[base + col]) +
                  __bfloat162float(residual[base + col]);
    if (projection_bias != nullptr)
      value += __bfloat162float(projection_bias[col]);
    const __nv_bfloat16 rounded = __float2bfloat16(value);
    hidden[base + col] = rounded;
    cache[i] = __bfloat162float(rounded);
  }
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) sum += cache[i];
  const float mean = block_sum_parallel_unsafe(sum, scratch) / 1024.0f;
  float variance_sum = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float centered = cache[i] - mean;
    variance_sum += centered * centered;
  }
  __syncthreads();
  const float inverse_std =
      rsqrtf(block_sum_parallel_unsafe(variance_sum, scratch) / 1024.0f + eps);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int col = threadIdx.x + i * static_cast<int>(blockDim.x);
    const float value = (cache[i] - mean) * inverse_std *
                            __bfloat162float(norm_weight[col]) +
                        __bfloat162float(norm_bias[col]);
    normalized[base + col] = __float2bfloat16(value);
  }
}


// ── RoPE kernels (from legacy rope.cuh) ──────────────────────────────────
__device__ __forceinline__ uint32_t mrope_axis_for_pair(
    uint32_t pair_idx, uint32_t sec_h, uint32_t sec_w)
{
    uint32_t rem = pair_idx % 3;
    if (rem == 1 && pair_idx < sec_h * 3) return 1;
    if (rem == 2 && pair_idx < sec_w * 3) return 2;
    return 0;
}

__global__ void build_vision_rotation_table_f32_kernel(
    const uint32_t* pos_ids, float2* rotation_table,
    uint32_t head_dim, uint32_t seq_len, float theta)
{
    const uint64_t linear = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t half = head_dim / 2;
    const uint64_t total = (uint64_t)seq_len * half;
    if (linear >= total) return;

    const uint32_t pair_idx = (uint32_t)(linear % half);
    const uint32_t seq_idx = (uint32_t)(linear / half);
    const uint32_t axis = pair_idx < half / 2 ? 0u : 1u;
    const uint32_t pair_in_axis =
        pair_idx < half / 2 ? pair_idx : pair_idx - half / 2;
    const uint32_t pos = pos_ids[seq_idx * 2 + axis];
    const float frequency =
        1.0f / powf(theta, 2.0f * static_cast<float>(pair_in_axis) /
                               static_cast<float>(half));
    float sine, cosine;
    sincosf(static_cast<float>(pos) * frequency, &sine, &cosine);
    rotation_table[linear] = make_float2(cosine, sine);
}

__global__ void qk_rms_norm_mrope_bf16_kernel(
    const __nv_bfloat16* query_input, const __nv_bfloat16* query_weight,
    __nv_bfloat16* query_output, const __nv_bfloat16* key_input,
    const __nv_bfloat16* key_weight, __nv_bfloat16* key_output,
    uint32_t head_dim, uint32_t query_heads, uint32_t key_heads,
    uint32_t seq_len, float eps, float theta, const uint32_t* pos_ids,
    uint32_t sec_h, uint32_t sec_w) {
  extern __shared__ __nv_bfloat16 rounded[];
  __shared__ float warp_sums[32];
  __shared__ float block_square_sum;
  const uint32_t combined_head = blockIdx.y;
  const bool is_query = combined_head < query_heads;
  const uint32_t heads = is_query ? query_heads : key_heads;
  const uint32_t head = is_query ? combined_head : combined_head - query_heads;
  const uint32_t seq = blockIdx.x;
  if (seq >= seq_len || head >= heads) return;
  const __nv_bfloat16* input = is_query ? query_input : key_input;
  const __nv_bfloat16* weight = is_query ? query_weight : key_weight;
  __nv_bfloat16* output = is_query ? query_output : key_output;
  const int64_t base =
      (static_cast<int64_t>(seq) * heads + head) * head_dim;

  float square_sum = 0.0f;
  for (uint32_t col = threadIdx.x; col < head_dim; col += blockDim.x) {
    const float value = __bfloat162float(input[base + col]);
    square_sum += value * value;
  }
  // Keep the reduction tree byte-for-byte equivalent to the public BF16
  // RMSNorm kernel. The legacy boundary is observable because its BF16
  // output is consumed by mRoPE.
  for (int offset = 16; offset > 0; offset >>= 1)
    square_sum += __shfl_xor_sync(0xffffffff, square_sum, offset);
  const uint32_t warp = threadIdx.x / 32;
  const uint32_t lane = threadIdx.x % 32;
  if (lane == 0) warp_sums[warp] = square_sum;
  __syncthreads();
  if (warp == 0) {
    float value =
        threadIdx.x < (blockDim.x + 31) / 32 ? warp_sums[threadIdx.x] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1)
      value += __shfl_xor_sync(0xffffffff, value, offset);
    if (lane == 0) block_square_sum = value;
  }
  __syncthreads();
  const float inverse_rms =
      rsqrtf(block_square_sum / static_cast<float>(head_dim) + eps);
  for (uint32_t col = threadIdx.x; col < head_dim; col += blockDim.x) {
    rounded[col] = __float2bfloat16(
        __bfloat162float(input[base + col]) * inverse_rms *
        __bfloat162float(weight[col]));
  }
  __syncthreads();

  const uint32_t pair = threadIdx.x;
  if (pair >= head_dim / 2) return;
  const uint32_t axis = mrope_axis_for_pair(pair, sec_h, sec_w);
  const uint32_t pos = pos_ids[seq * 3 + axis];
  const float freq =
      1.0f / powf(theta, 2.0f * static_cast<float>(pair) /
                            static_cast<float>(head_dim));
  const float angle = static_cast<float>(pos) * freq;
  const float cos_value = cosf(angle);
  const float sin_value = sinf(angle);
  const float first = __bfloat162float(rounded[pair]);
  const float second = __bfloat162float(rounded[head_dim / 2 + pair]);
  output[base + pair] =
      __float2bfloat16(first * cos_value - second * sin_value);
  output[base + head_dim / 2 + pair] =
      __float2bfloat16(first * sin_value + second * cos_value);
}

__global__ void qkv_split_bias_vision_rope_precomputed_bf16_vec2_kernel(
    const __nv_bfloat16* qkv, const __nv_bfloat16* bias,
    __nv_bfloat16* q_out, __nv_bfloat16* k_out, __nv_bfloat16* v_out,
    uint32_t head_dim, uint32_t n_heads, uint32_t seq_len,
    const float2* rotation_table) {
  const uint64_t group =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint32_t half = head_dim / 2;
  const uint32_t groups_per_head = half / 2;
  const uint64_t total =
      static_cast<uint64_t>(seq_len) * n_heads * groups_per_head;
  if (group >= total) return;

  const uint32_t group_in_head = static_cast<uint32_t>(group % groups_per_head);
  const uint64_t token_head = group / groups_per_head;
  const uint32_t head = static_cast<uint32_t>(token_head % n_heads);
  const uint32_t row = static_cast<uint32_t>(token_head / n_heads);
  const uint32_t width = n_heads * head_dim;
  const uint64_t qkv_row = static_cast<uint64_t>(row) * 3 * width;
  const uint32_t head_col = head * head_dim;
  const uint64_t out_base = token_head * head_dim;

#pragma unroll
  for (uint32_t lane = 0; lane < 2; ++lane) {
    const uint32_t pair = group_in_head * 2 + lane;
    const uint32_t second = pair + half;
    const __nv_bfloat16 q0_bf16 = __float2bfloat16(
        __bfloat162float(qkv[qkv_row + head_col + pair]) +
        __bfloat162float(bias[head_col + pair]));
    const __nv_bfloat16 q1_bf16 = __float2bfloat16(
        __bfloat162float(qkv[qkv_row + head_col + second]) +
        __bfloat162float(bias[head_col + second]));
    const __nv_bfloat16 k0_bf16 = __float2bfloat16(
        __bfloat162float(qkv[qkv_row + width + head_col + pair]) +
        __bfloat162float(bias[width + head_col + pair]));
    const __nv_bfloat16 k1_bf16 = __float2bfloat16(
        __bfloat162float(qkv[qkv_row + width + head_col + second]) +
        __bfloat162float(bias[width + head_col + second]));
    const float2 rotation =
        rotation_table[static_cast<size_t>(row) * half + pair];
    const float q0 = __bfloat162float(q0_bf16);
    const float q1 = __bfloat162float(q1_bf16);
    const float k0 = __bfloat162float(k0_bf16);
    const float k1 = __bfloat162float(k1_bf16);
    q_out[out_base + pair] =
        __float2bfloat16(q0 * rotation.x - q1 * rotation.y);
    q_out[out_base + second] =
        __float2bfloat16(q0 * rotation.y + q1 * rotation.x);
    k_out[out_base + pair] =
        __float2bfloat16(k0 * rotation.x - k1 * rotation.y);
    k_out[out_base + second] =
        __float2bfloat16(k0 * rotation.y + k1 * rotation.x);
    v_out[out_base + pair] = __float2bfloat16(
        __bfloat162float(qkv[qkv_row + 2 * width + head_col + pair]) +
        __bfloat162float(bias[2 * width + head_col + pair]));
    v_out[out_base + second] = __float2bfloat16(
        __bfloat162float(qkv[qkv_row + 2 * width + head_col + second]) +
        __bfloat162float(bias[2 * width + head_col + second]));
  }
}

__global__ void rope_mrope_bf16_kernel(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    uint32_t head_dim, uint32_t n_heads, uint32_t seq_len,
    float theta, const uint32_t* pos_ids,
    uint32_t sec_h, uint32_t sec_w)
{
    uint32_t pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t head_idx = blockIdx.y;
    uint32_t seq_idx  = blockIdx.z;
    if (pair_idx >= head_dim / 2) return;

    uint32_t axis = mrope_axis_for_pair(pair_idx, sec_h, sec_w);
    uint32_t pos  = pos_ids[seq_idx * 3 + axis];

    float freq    = 1.0f / powf(theta, 2.0f * (float)pair_idx / (float)head_dim);
    float angle   = (float)pos * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    uint32_t base = seq_idx * n_heads * head_dim + head_idx * head_dim;
    uint32_t half = head_dim / 2;
    uint32_t idx0 = base + pair_idx;
    uint32_t idx1 = base + half + pair_idx;
    float x0 = __bfloat162float(input[idx0]);
    float x1 = __bfloat162float(input[idx1]);
    output[idx0] = __float2bfloat16(x0 * cos_val - x1 * sin_val);
    output[idx1] = __float2bfloat16(x0 * sin_val + x1 * cos_val);
}

__global__ void rope_vision_2d_pair_bf16_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* k,
    __nv_bfloat16* q_out, __nv_bfloat16* k_out,
    uint32_t head_dim, uint32_t n_heads, uint32_t seq_len,
    float theta, const uint32_t* pos_ids)
{
    uint64_t linear = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t half = head_dim / 2;
    uint64_t total = (uint64_t)seq_len * n_heads * half;
    if (linear >= total) return;

    uint32_t pair_idx = (uint32_t)(linear % half);
    uint64_t token_head = linear / half;
    uint32_t head_idx = (uint32_t)(token_head % n_heads);
    uint32_t seq_idx = (uint32_t)(token_head / n_heads);
    uint32_t axis = pair_idx < half / 2 ? 0u : 1u;
    uint32_t pair_in_axis = pair_idx < half / 2 ? pair_idx : pair_idx - half / 2;
    uint32_t pos = pos_ids[seq_idx * 2 + axis];

    float freq = 1.0f / powf(theta, 2.0f * (float)pair_in_axis / (float)half);
    float angle = (float)pos * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    uint64_t base = ((uint64_t)seq_idx * n_heads + head_idx) * head_dim;
    uint64_t idx0 = base + pair_idx;
    uint64_t idx1 = base + half + pair_idx;

    float q0 = __bfloat162float(q[idx0]);
    float q1 = __bfloat162float(q[idx1]);
    q_out[idx0] = __float2bfloat16(q0 * cos_val - q1 * sin_val);
    q_out[idx1] = __float2bfloat16(q0 * sin_val + q1 * cos_val);

    float k0 = __bfloat162float(k[idx0]);
    float k1 = __bfloat162float(k[idx1]);
    k_out[idx0] = __float2bfloat16(k0 * cos_val - k1 * sin_val);
    k_out[idx1] = __float2bfloat16(k0 * sin_val + k1 * cos_val);
}

template <bool kPrecomputed>
__global__ void qkv_split_bias_vision_rope_bf16_kernel(
    const __nv_bfloat16* qkv, const __nv_bfloat16* bias,
    __nv_bfloat16* q_out, __nv_bfloat16* k_out, __nv_bfloat16* v_out,
    uint32_t head_dim, uint32_t n_heads, uint32_t seq_len,
    float theta, const uint32_t* pos_ids, const float2* rotation_table)
{
    uint64_t linear = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t half = head_dim / 2;
    const uint64_t total = (uint64_t)seq_len * n_heads * half;
    if (linear >= total) return;

    const uint32_t pair_idx = (uint32_t)(linear % half);
    const uint64_t token_head = linear / half;
    const uint32_t head_idx = (uint32_t)(token_head % n_heads);
    const uint32_t seq_idx = (uint32_t)(token_head / n_heads);
    const uint32_t width = n_heads * head_dim;
    const uint64_t qkv_row = (uint64_t)seq_idx * 3 * width;
    const uint32_t head_col = head_idx * head_dim;
    const uint32_t second = pair_idx + half;
    const uint64_t out_base = ((uint64_t)seq_idx * n_heads + head_idx) * head_dim;

    // Match split_qkv_bias_bf16's intermediate output rounding exactly.
    const __nv_bfloat16 q0_bf16 = __float2bfloat16(
        __bfloat162float(qkv[qkv_row + head_col + pair_idx]) +
        __bfloat162float(bias[head_col + pair_idx]));
    const __nv_bfloat16 q1_bf16 = __float2bfloat16(
        __bfloat162float(qkv[qkv_row + head_col + second]) +
        __bfloat162float(bias[head_col + second]));
    const __nv_bfloat16 k0_bf16 = __float2bfloat16(
        __bfloat162float(qkv[qkv_row + width + head_col + pair_idx]) +
        __bfloat162float(bias[width + head_col + pair_idx]));
    const __nv_bfloat16 k1_bf16 = __float2bfloat16(
        __bfloat162float(qkv[qkv_row + width + head_col + second]) +
        __bfloat162float(bias[width + head_col + second]));

    float sine, cosine;
    if constexpr (kPrecomputed) {
        const float2 rotation =
            rotation_table[static_cast<size_t>(seq_idx) * half + pair_idx];
        cosine = rotation.x;
        sine = rotation.y;
    } else {
        const uint32_t axis = pair_idx < half / 2 ? 0u : 1u;
        const uint32_t pair_in_axis =
            pair_idx < half / 2 ? pair_idx : pair_idx - half / 2;
        const uint32_t pos = pos_ids[seq_idx * 2 + axis];
        const float frequency =
            1.0f / powf(theta, 2.0f * static_cast<float>(pair_in_axis) /
                                   static_cast<float>(half));
        sincosf(static_cast<float>(pos) * frequency, &sine, &cosine);
    }
    const float q0 = __bfloat162float(q0_bf16);
    const float q1 = __bfloat162float(q1_bf16);
    const float k0 = __bfloat162float(k0_bf16);
    const float k1 = __bfloat162float(k1_bf16);
    q_out[out_base + pair_idx] = __float2bfloat16(q0 * cosine - q1 * sine);
    q_out[out_base + second] = __float2bfloat16(q0 * sine + q1 * cosine);
    k_out[out_base + pair_idx] = __float2bfloat16(k0 * cosine - k1 * sine);
    k_out[out_base + second] = __float2bfloat16(k0 * sine + k1 * cosine);

    v_out[out_base + pair_idx] = __float2bfloat16(
        __bfloat162float(qkv[qkv_row + 2 * width + head_col + pair_idx]) +
        __bfloat162float(bias[2 * width + head_col + pair_idx]));
    v_out[out_base + second] = __float2bfloat16(
        __bfloat162float(qkv[qkv_row + 2 * width + head_col + second]) +
        __bfloat162float(bias[2 * width + head_col + second]));
}


// Split a rank-2 packed [tokens, 3*hidden] BF16 projection (Q then K then V,
// each hidden = heads * head_dim) into three contiguous [tokens, heads,
// head_dim] tensors. Equivalent to the legacy FA2 strided-QKV reader, done
// with an explicit copy so the generic attention op can consume it.
__global__ void gr00t_split_strided_qkv_bf16_kernel(
    const __nv_bfloat16* qkv, __nv_bfloat16* q, __nv_bfloat16* k,
    __nv_bfloat16* v, int tokens, int hidden) {
  const int64_t total = static_cast<int64_t>(tokens) * hidden;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total; index += stride) {
    const int64_t token = index / hidden;
    const int col = static_cast<int>(index % hidden);
    const __nv_bfloat16* row = qkv + token * 3 * hidden;
    q[index] = row[col];
    k[index] = row[hidden + col];
    v[index] = row[2 * hidden + col];
  }
}


// ── Quantization kernels (from legacy quantization.cuh / concat_quantization.cuh)

struct alignas(16) Gr00tBf16Pack8 {
  __nv_bfloat16 values[8];
};
struct alignas(8) Gr00tFp8Pack8 {
  __nv_fp8_e4m3 values[8];
};

__global__ void gr00t_quantize_bf16_e4m3_kernel(
    const __nv_bfloat16* input, __nv_fp8_e4m3* output, int64_t count,
    float inverse_scale) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    float value = fminf(448.0f, fmaxf(-448.0f,
        __bfloat162float(input[index]) * inverse_scale));
    output[index] = static_cast<__nv_fp8_e4m3>(value);
  }
}

__global__ void gr00t_quantize_bf16_e4m3_packed8_kernel(
    const Gr00tBf16Pack8* input, Gr00tFp8Pack8* output, int64_t vector_count,
    float inverse_scale) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < vector_count; index += stride) {
    const Gr00tBf16Pack8 values = input[index];
    Gr00tFp8Pack8 quantized;
#pragma unroll
    for (int item = 0; item < 8; ++item) {
      float value = __bfloat162float(values.values[item]) * inverse_scale;
      value = fminf(448.0f, fmaxf(-448.0f, value));
      quantized.values[item] = static_cast<__nv_fp8_e4m3>(value);
    }
    output[index] = quantized;
  }
}

// Copy `first` then `second` into `output`, quantizing every element to E4M3
// with the same inverse scale. Row concatenation with a fused epilogue.
__global__ void gr00t_concat_rows_quantize_bf16_e4m3_kernel(
    const __nv_bfloat16* first, const __nv_bfloat16* second,
    __nv_fp8_e4m3* output, int64_t first_count, int64_t total_count,
    float inverse_scale) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < total_count; index += stride) {
    const __nv_bfloat16 source =
        index < first_count ? first[index] : second[index - first_count];
    float value = __bfloat162float(source) * inverse_scale;
    value = fminf(448.0f, fmaxf(-448.0f, value));
    output[index] = static_cast<__nv_fp8_e4m3>(value);
  }
}


// Dequantize an INT32 accumulator with per-row and per-column F32 scales.
// Byte-identical to the legacy dequantize_int32_bf16_kernel.
__global__ void gr00t_dequantize_int32_bf16_kernel(
    const int32_t* accumulators, const float* row_scales,
    const float* column_scales, __nv_bfloat16* output, int rows, int cols) {
  const int row = blockIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows && col < cols) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    output[index] = __float2bfloat16(
        static_cast<float>(accumulators[index]) * row_scales[row] *
        column_scales[col]);
  }
}

// ── W8A8 (INT8) quantization kernels (from legacy w8a8_adapter.cu / quantization.cuh) ──
__global__ void gr00t_quantize_rows_bf16_int8_vec4_kernel(
    const __nv_bfloat16* input, int8_t* output, float* scales,
    int rows, int cols) {
  extern __shared__ Gr00tW8Bf16x4 rounded_vec4[];
  __shared__ float scratch[16];
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int quads = cols / 4;
  const int64_t base = static_cast<int64_t>(row) * quads;
  const auto* input4 = reinterpret_cast<const Gr00tW8Bf16x4*>(input);
  float maximum = 0.0f;
  for (int quad = threadIdx.x; quad < quads; quad += blockDim.x) {
    const Gr00tW8Bf16x4 value = input4[base + quad];
    rounded_vec4[quad] = value;
    maximum = fmaxf(maximum, fabsf(__bfloat162float(value.low.x)));
    maximum = fmaxf(maximum, fabsf(__bfloat162float(value.low.y)));
    maximum = fmaxf(maximum, fabsf(__bfloat162float(value.high.x)));
    maximum = fmaxf(maximum, fabsf(__bfloat162float(value.high.y)));
  }
  const float scale =
      fmaxf(block_max_parallel_unsafe(maximum, scratch) / 127.0f, 1.0e-12f);
  if (threadIdx.x == 0) scales[row] = scale;
  __syncthreads();
  auto* output4 = reinterpret_cast<uint32_t*>(output) + base;
  for (int quad = threadIdx.x; quad < quads; quad += blockDim.x) {
    const Gr00tW8Bf16x4 value = rounded_vec4[quad];
    const float values[4] = {
        __bfloat162float(value.low.x), __bfloat162float(value.low.y),
        __bfloat162float(value.high.x), __bfloat162float(value.high.y)};
    uint32_t packed = 0;
#pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
      const float quantized = roundf(values[lane] / scale);
      const int8_t byte = static_cast<int8_t>(
          fminf(127.0f, fmaxf(-128.0f, quantized)));
      packed |= static_cast<uint32_t>(static_cast<uint8_t>(byte)) << (8 * lane);
    }
    output4[quad] = packed;
  }
}

__global__ void gr00t_bias_gelu_quantize_rows_bf16_int8_vec4_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* bias,
    int8_t* output, float* scales, int rows, int cols) {
  extern __shared__ Gr00tW8Bf16x4 rounded_vec4[];
  __shared__ float scratch[16];
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int quads = cols / 4;
  const int64_t base = static_cast<int64_t>(row) * quads;
  const auto* input4 = reinterpret_cast<const Gr00tW8Bf16x4*>(input);
  const auto* bias4 = reinterpret_cast<const Gr00tW8Bf16x4*>(bias);
  float maximum = 0.0f;
  for (int quad = threadIdx.x; quad < quads; quad += blockDim.x) {
    const Gr00tW8Bf16x4 x = input4[base + quad];
    const Gr00tW8Bf16x4 b = bias4[quad];
    const float values[4] = {
        gelu_tanh(__bfloat162float(x.low.x) + __bfloat162float(b.low.x)),
        gelu_tanh(__bfloat162float(x.low.y) + __bfloat162float(b.low.y)),
        gelu_tanh(__bfloat162float(x.high.x) + __bfloat162float(b.high.x)),
        gelu_tanh(__bfloat162float(x.high.y) + __bfloat162float(b.high.y)),
    };
    const Gr00tW8Bf16x4 activated{
        __floats2bfloat162_rn(values[0], values[1]),
        __floats2bfloat162_rn(values[2], values[3])};
    rounded_vec4[quad] = activated;
    maximum = fmaxf(maximum, fabsf(__bfloat162float(activated.low.x)));
    maximum = fmaxf(maximum, fabsf(__bfloat162float(activated.low.y)));
    maximum = fmaxf(maximum, fabsf(__bfloat162float(activated.high.x)));
    maximum = fmaxf(maximum, fabsf(__bfloat162float(activated.high.y)));
  }
  const float scale =
      fmaxf(block_max_parallel_unsafe(maximum, scratch) / 127.0f, 1.0e-12f);
  if (threadIdx.x == 0) scales[row] = scale;
  __syncthreads();
  auto* output4 = reinterpret_cast<uint32_t*>(output) + base;
  for (int quad = threadIdx.x; quad < quads; quad += blockDim.x) {
    const Gr00tW8Bf16x4 value = rounded_vec4[quad];
    const float values[4] = {
        __bfloat162float(value.low.x), __bfloat162float(value.low.y),
        __bfloat162float(value.high.x), __bfloat162float(value.high.y)};
    uint32_t packed = 0;
#pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
      const float quantized = roundf(values[lane] / scale);
      const int8_t byte = static_cast<int8_t>(
          fminf(127.0f, fmaxf(-128.0f, quantized)));
      packed |= static_cast<uint32_t>(static_cast<uint8_t>(byte)) << (8 * lane);
    }
    output4[quad] = packed;
  }
}

__global__ void gr00t_silu_mul_quantize_rows_bf16_int8_vec4_kernel(
    const __nv_bfloat16* gate, const __nv_bfloat16* up,
    int8_t* output, float* scales, int rows, int cols) {
  extern __shared__ Gr00tW8Bf16x4 rounded_vec4[];
  __shared__ float scratch[16];
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int quads = cols / 4;
  const int64_t base = static_cast<int64_t>(row) * quads;
  const auto* gate4 = reinterpret_cast<const Gr00tW8Bf16x4*>(gate);
  const auto* up4 = reinterpret_cast<const Gr00tW8Bf16x4*>(up);
  float maximum = 0.0f;
  for (int quad = threadIdx.x; quad < quads; quad += blockDim.x) {
    const Gr00tW8Bf16x4 g = gate4[base + quad];
    const Gr00tW8Bf16x4 u = up4[base + quad];
    const float gate_values[4] = {
        __bfloat162float(g.low.x), __bfloat162float(g.low.y),
        __bfloat162float(g.high.x), __bfloat162float(g.high.y)};
    const float up_values[4] = {
        __bfloat162float(u.low.x), __bfloat162float(u.low.y),
        __bfloat162float(u.high.x), __bfloat162float(u.high.y)};
    __nv_bfloat16 values[4];
#pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
      const __nv_bfloat16 silu = __float2bfloat16(
          gate_values[lane] / (1.0f + expf(-gate_values[lane])));
      values[lane] = __float2bfloat16(
          __bfloat162float(silu) * up_values[lane]);
      maximum = fmaxf(maximum, fabsf(__bfloat162float(values[lane])));
    }
    rounded_vec4[quad] = Gr00tW8Bf16x4{
        __floats2bfloat162_rn(__bfloat162float(values[0]),
                             __bfloat162float(values[1])),
        __floats2bfloat162_rn(__bfloat162float(values[2]),
                             __bfloat162float(values[3]))};
  }
  const float scale =
      fmaxf(block_max_parallel_unsafe(maximum, scratch) / 127.0f, 1.0e-12f);
  if (threadIdx.x == 0) scales[row] = scale;
  __syncthreads();
  auto* output4 = reinterpret_cast<uint32_t*>(output) + base;
  for (int quad = threadIdx.x; quad < quads; quad += blockDim.x) {
    const Gr00tW8Bf16x4 value = rounded_vec4[quad];
    const float values[4] = {
        __bfloat162float(value.low.x), __bfloat162float(value.low.y),
        __bfloat162float(value.high.x), __bfloat162float(value.high.y)};
    uint32_t packed = 0;
#pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
      const float quantized = roundf(values[lane] / scale);
      const int8_t byte = static_cast<int8_t>(
          fminf(127.0f, fmaxf(-128.0f, quantized)));
      packed |= static_cast<uint32_t>(static_cast<uint8_t>(byte)) << (8 * lane);
    }
    output4[quad] = packed;
  }
}

__global__ void gr00t_bias_gelu_quantize_rows_bf16_int8_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* bias,
    int8_t* output, float* scales, int rows, int cols) {
  extern __shared__ __nv_bfloat16 rounded[];
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int64_t base = static_cast<int64_t>(row) * cols;
  float maximum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float value = __bfloat162float(input[base + col]) +
                        __bfloat162float(bias[col]);
    const __nv_bfloat16 activated = __float2bfloat16(gelu_tanh(value));
    rounded[col] = activated;
    maximum = fmaxf(maximum, fabsf(__bfloat162float(activated)));
  }
  const float scale =
      fmaxf(block_max_parallel_unsafe(maximum, scratch) / 127.0f, 1.0e-12f);
  if (threadIdx.x == 0) scales[row] = scale;
  __syncthreads();
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float quantized = roundf(__bfloat162float(rounded[col]) / scale);
    output[base + col] =
        static_cast<int8_t>(fminf(127.0f, fmaxf(-128.0f, quantized)));
  }
}

__global__ void gr00t_layer_norm_quantize_rows_bf16_int8_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    const __nv_bfloat16* bias, __nv_bfloat16* output,
    int8_t* quantized, float* scales, int rows, int cols, float eps) {
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int64_t base = static_cast<int64_t>(row) * cols;
  float sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x)
    sum += __bfloat162float(input[base + col]);
  const float mean = block_sum_parallel_unsafe(sum, scratch) / cols;
  float variance_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float centered = __bfloat162float(input[base + col]) - mean;
    variance_sum += centered * centered;
  }
  __syncthreads();
  const float inverse_std =
      rsqrtf(block_sum_parallel_unsafe(variance_sum, scratch) / cols + eps);
  float maximum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = base + col;
    const float value =
        (__bfloat162float(input[index]) - mean) * inverse_std *
            __bfloat162float(weight[col]) +
        __bfloat162float(bias[col]);
    const __nv_bfloat16 rounded = __float2bfloat16(value);
    output[index] = rounded;
    maximum = fmaxf(maximum, fabsf(__bfloat162float(rounded)));
  }
  __syncthreads();
  const float scale =
      fmaxf(block_max_parallel_unsafe(maximum, scratch) / 127.0f, 1.0e-12f);
  if (threadIdx.x == 0) scales[row] = scale;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = base + col;
    const float value = roundf(__bfloat162float(output[index]) / scale);
    quantized[index] =
        static_cast<int8_t>(fminf(127.0f, fmaxf(-128.0f, value)));
  }
}

__global__ void gr00t_silu_mul_quantize_rows_bf16_int8_kernel(
    const __nv_bfloat16* gate, const __nv_bfloat16* up,
    int8_t* output, float* scales, int rows, int cols) {
  extern __shared__ __nv_bfloat16 rounded[];
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  const int64_t base = static_cast<int64_t>(row) * cols;
  float maximum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float x = __bfloat162float(gate[base + col]);
    const float y = __bfloat162float(up[base + col]);
    // Match the existing two-stage SiLU + multiply contract exactly: the
    // SiLU result is rounded to BF16 before it is multiplied by `up`, and the
    // product is rounded to BF16 again before row-wise quantization.
    const __nv_bfloat16 silu = __float2bfloat16(x / (1.0f + expf(-x)));
    const __nv_bfloat16 value =
        __float2bfloat16(__bfloat162float(silu) * y);
    rounded[col] = value;
    maximum = fmaxf(maximum, fabsf(__bfloat162float(value)));
  }
  const float scale =
      fmaxf(block_max_parallel_unsafe(maximum, scratch) / 127.0f, 1.0e-12f);
  if (threadIdx.x == 0) scales[row] = scale;
  __syncthreads();
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float quantized = roundf(__bfloat162float(rounded[col]) / scale);
    output[base + col] =
        static_cast<int8_t>(fminf(127.0f, fmaxf(-128.0f, quantized)));
  }
}

__global__ void gr00t_quantize_rows_bf16_int8_kernel(
    const __nv_bfloat16* input, int8_t* output, float* scales,
    int rows, int cols) {
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  float maximum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    maximum = fmaxf(
        maximum,
        fabsf(__bfloat162float(input[static_cast<int64_t>(row) * cols + col])));
  }
  const float scale =
      fmaxf(block_max_parallel_unsafe(maximum, scratch) / 127.0f, 1.0e-12f);
  if (threadIdx.x == 0) scales[row] = scale;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    const float quantized = roundf(__bfloat162float(input[index]) / scale);
    output[index] = static_cast<int8_t>(fminf(127.0f, fmaxf(-128.0f, quantized)));
  }
}

__global__ void gr00t_adaptive_layer_norm_quantize_rows_bf16_int8_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* modulation,
    __nv_bfloat16* output, int8_t* quantized, float* scales,
    int rows, int cols, float eps) {
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int64_t base = static_cast<int64_t>(row) * cols;
  float sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x)
    sum += __bfloat162float(input[base + col]);
  const float mean = block_sum_parallel_unsafe(sum, scratch) / cols;
  float variance_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float centered = __bfloat162float(input[base + col]) - mean;
    variance_sum += centered * centered;
  }
  // Finish reading the previous reduction before reusing scratch.
  __syncthreads();
  const float inverse_std =
      rsqrtf(block_sum_parallel_unsafe(variance_sum, scratch) / cols + eps);
  float maximum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float normalized =
        (__bfloat162float(input[base + col]) - mean) * inverse_std;
    const float scale = __bfloat162float(modulation[col]);
    const float shift = __bfloat162float(modulation[cols + col]);
    const __nv_bfloat16 rounded =
        __float2bfloat16(normalized * (1.0f + scale) + shift);
    output[base + col] = rounded;
    maximum = fmaxf(maximum, fabsf(__bfloat162float(rounded)));
  }
  // Finish reading the previous reduction before reusing scratch.
  __syncthreads();
  const float row_scale =
      fmaxf(block_max_parallel_unsafe(maximum, scratch) / 127.0f, 1.0e-12f);
  if (threadIdx.x == 0) scales[row] = row_scale;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float value = roundf(__bfloat162float(output[base + col]) / row_scale);
    quantized[base + col] =
        static_cast<int8_t>(fminf(127.0f, fmaxf(-128.0f, value)));
  }
}

}  // namespace

extern "C" cudaError_t apxinf_gr00t_silu_mul_separate_bf16(
    const void* gate, const void* up, void* output, uint32_t count,
    cudaStream_t stream) {
  const bool packed =
      (count & 3U) == 0 &&
      (reinterpret_cast<uintptr_t>(gate) & 7U) == 0 &&
      (reinterpret_cast<uintptr_t>(up) & 7U) == 0 &&
      (reinterpret_cast<uintptr_t>(output) & 7U) == 0;
  if (packed) {
    const uint32_t quad_count = count / 4;
    int blocks = static_cast<int>(
        (quad_count + kGr00tBlockSize - 1) / kGr00tBlockSize);
    blocks = blocks > 512 ? 512 : blocks;
    silu_mul_separate_bf16_packed4_kernel<<<blocks, kGr00tBlockSize, 0,
        stream>>>(
        (const Bf16x4*)gate, (const Bf16x4*)up, (Bf16x4*)output, quad_count);
    return cudaGetLastError();
  }
  dim3 grid((count + kGr00tBlockSize - 1) / kGr00tBlockSize, 1, 1);
  dim3 block(kGr00tBlockSize, 1, 1);
  silu_mul_separate_bf16_kernel<<<grid, block, 0, stream>>>(
      (const __nv_bfloat16*)gate, (const __nv_bfloat16*)up,
      (__nv_bfloat16*)output, count);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_gr00t_silu_mul_quant_bf16_e4m3(
    const void* gate, const void* up, void* output, int64_t count,
    float scale, cudaStream_t stream) {
  if (gate == nullptr || up == nullptr || output == nullptr || count <= 0 ||
      !(scale > 0.0f)) return cudaErrorInvalidValue;
  constexpr int threads = 256;
  const int64_t pairs = (count + 1) / 2;
  int blocks = static_cast<int>((pairs + threads - 1) / threads);
  blocks = blocks > 1024 ? 1024 : blocks;
  silu_mul_quant_bf16_e4m3_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(gate),
      static_cast<const __nv_bfloat16*>(up),
      static_cast<__nv_fp8_e4m3*>(output), count, 1.0f / scale);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_gr00t_bias_gelu_quant_bf16_e4m3(
    const void* input, const void* bias, void* output, int rows, int cols,
    float scale, cudaStream_t stream) {
  if (!input || !bias || !output || rows <= 0 || cols <= 0 || cols % 4 != 0 ||
      !(scale > 0.0f) || reinterpret_cast<uintptr_t>(input) % alignof(Bf16x4) != 0 ||
      reinterpret_cast<uintptr_t>(bias) % alignof(Bf16x4) != 0 ||
      reinterpret_cast<uintptr_t>(output) % alignof(uint32_t) != 0)
    return cudaErrorInvalidValue;
  constexpr int threads = 256;
  const int64_t quad_count = static_cast<int64_t>(rows) * cols / 4;
  int blocks = static_cast<int>((quad_count + threads - 1) / threads);
  blocks = blocks > 1024 ? 1024 : blocks;
  bias_gelu_quant_bf16_e4m3_packed4_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(bias),
      static_cast<__nv_fp8_e4m3*>(output), quad_count, cols, 1.0f / scale);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_gr00t_bias_gelu_bf16_packed8(
    const void* input, const void* bias, void* output, int rows, int cols,
    cudaStream_t stream) {
  if (input == nullptr || bias == nullptr || output == nullptr || rows <= 0 ||
      cols <= 0 || cols % 8 != 0 ||
      reinterpret_cast<uintptr_t>(input) % alignof(Bf16Pairx8) != 0 ||
      reinterpret_cast<uintptr_t>(bias) % alignof(Bf16Pairx8) != 0 ||
      reinterpret_cast<uintptr_t>(output) % alignof(Bf16Pairx8) != 0)
    return cudaErrorInvalidValue;
  const int64_t octet_count = static_cast<int64_t>(rows) * cols / 8;
  const int threads = rows >= 512 ? 256 : 128;
  const int blocks = static_cast<int>((octet_count + threads - 1) / threads);
  bias_gelu_bf16_packed8_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const Bf16Pairx8*>(input), static_cast<const Bf16Pairx8*>(bias),
      static_cast<Bf16Pairx8*>(output), octet_count, cols / 8);
  return cudaGetLastError();
}

// ── Elementwise / selection launchers ────────────────────────────────────
// Copied verbatim from the legacy static_bf16_adapter.cu, renamed with the
// apxinf_gr00t_ prefix.

extern "C" cudaError_t apxinf_gr00t_scatter_rows_bf16(
    const void* source, const void* rows, void* output,
    int row_count, int cols, int add, cudaStream_t stream) {
  if (source == nullptr || rows == nullptr || output == nullptr ||
      row_count <= 0 || cols <= 0 || (add != 0 && add != 1))
    return cudaErrorInvalidValue;
  const int64_t count = static_cast<int64_t>(row_count) * cols;
  const int blocks = static_cast<int>((count + kGr00tBlockSize - 1) / kGr00tBlockSize);
  scatter_rows_bf16_kernel<<<blocks, kGr00tBlockSize, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(source),
      static_cast<const uint32_t*>(rows),
      static_cast<__nv_bfloat16*>(output), count, cols, add != 0);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_gr00t_bias_qkv_in_place_bf16(
    void* query, void* key, void* value, const void* query_bias,
    const void* key_bias, const void* value_bias, int rows, int cols,
    cudaStream_t stream) {
  if (!query || !key || !value || !query_bias || !key_bias || !value_bias ||
      rows <= 0 || cols <= 0 || cols % 4 != 0)
    return cudaErrorInvalidValue;
  constexpr int threads = 256;
  const int64_t groups = static_cast<int64_t>(rows) * cols / 4;
  int blocks = static_cast<int>((groups + threads - 1) / threads);
  blocks = blocks > 1024 ? 1024 : blocks;
  bias_qkv_in_place_bf16_packed4_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<__nv_bfloat16*>(query),
      static_cast<__nv_bfloat16*>(key),
      static_cast<__nv_bfloat16*>(value),
      static_cast<const __nv_bfloat16*>(query_bias),
      static_cast<const __nv_bfloat16*>(key_bias),
      static_cast<const __nv_bfloat16*>(value_bias), groups, cols / 4);
  return cudaGetLastError();
}

// ── Norm launchers ───────────────────────────────────────────────────────

extern "C" cudaError_t apxinf_gr00t_adaptive_layer_norm_bf16(
    const void* input, const void* modulation, void* output, uint32_t rows,
    uint32_t cols, float eps, cudaStream_t stream) {
  if (rows == 0 || cols == 0 || !(eps > 0.0f)) return cudaErrorInvalidValue;
  gr00t_adaptive_layer_norm_bf16_kernel<<<rows, kGr00tBlockSize, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(modulation),
      static_cast<__nv_bfloat16*>(output), rows, cols, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_gr00t_adaptive_layer_norm_quant_bf16_e4m3(
    const void* input, const void* modulation, void* output, void* quantized,
    uint32_t rows, uint32_t cols, float eps, float scale, cudaStream_t stream) {
  if (rows == 0 || cols == 0 || !(eps > 0.0f) || !(scale > 0.0f))
    return cudaErrorInvalidValue;
  gr00t_adaptive_layer_norm_quant_bf16_e4m3_kernel<<<rows, kGr00tBlockSize, 0,
                                                     stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(modulation),
      static_cast<__nv_bfloat16*>(output),
      static_cast<__nv_fp8_e4m3*>(quantized), rows, cols, eps, 1.0f / scale);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_gr00t_rms_norm_quant_bf16_e4m3(
    const void* input, const void* weight, void* output, int rows, int cols,
    float eps, float scale, cudaStream_t stream) {
  if (input == nullptr || weight == nullptr || output == nullptr || rows <= 0 ||
      cols <= 0 || !std::isfinite(scale) || scale <= 0.0f)
    return cudaErrorInvalidValue;
  gr00t_rms_norm_quant_bf16_e4m3_kernel<<<rows, kGr00tBlockSize, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<__nv_fp8_e4m3*>(output), rows, cols, eps, 1.0f / scale);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_gr00t_layer_norm_quant_bf16_e4m3(
    const void* input, const void* weight, const void* bias, void* output,
    int rows, int cols, float eps, float scale, cudaStream_t stream) {
  if (input == nullptr || weight == nullptr || bias == nullptr ||
      output == nullptr || rows <= 0 || cols <= 0 || !std::isfinite(scale) ||
      scale <= 0.0f)
    return cudaErrorInvalidValue;
  const size_t shared_bytes = static_cast<size_t>(cols) * sizeof(float);
  gr00t_layer_norm_quant_bf16_e4m3_kernel<<<rows, kGr00tBlockSize,
                                            shared_bytes, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<const __nv_bfloat16*>(bias),
      static_cast<__nv_fp8_e4m3*>(output), rows, cols, eps, 1.0f / scale);
  return cudaGetLastError();
}

// ── Fused residual launchers ─────────────────────────────────────────────

extern "C" cudaError_t apxinf_gr00t_bias_then_residual_bf16(
    const void* projection, const void* bias, const void* residual,
    void* output, int rows, int cols, cudaStream_t stream) {
  const int64_t count = static_cast<int64_t>(rows) * cols;
  const int blocks = static_cast<int>((count + kGr00tBlockSize - 1) / kGr00tBlockSize);
  bias_then_residual_bf16_kernel<<<blocks, kGr00tBlockSize, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(projection),
      static_cast<const __nv_bfloat16*>(bias),
      static_cast<const __nv_bfloat16*>(residual),
      static_cast<__nv_bfloat16*>(output), count, cols);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_gr00t_bias_residual_bf16_packed4(
    const void* projection, const void* bias, const void* residual,
    void* output, int rows, int cols, cudaStream_t stream) {
  if (projection == nullptr || bias == nullptr || residual == nullptr ||
      output == nullptr || rows <= 0 || cols <= 0 || (cols & 3) != 0)
    return cudaErrorInvalidValue;
  const uintptr_t pointers = reinterpret_cast<uintptr_t>(projection) |
      reinterpret_cast<uintptr_t>(residual) |
      reinterpret_cast<uintptr_t>(output) |
      reinterpret_cast<uintptr_t>(bias);
  if ((pointers & (alignof(Bf16x4) - 1)) != 0) return cudaErrorInvalidValue;
  const int64_t packed_count = static_cast<int64_t>(rows) * (cols / 4);
  const int blocks =
      static_cast<int>((packed_count + kGr00tBlockSize - 1) / kGr00tBlockSize);
  bias_residual_bf16_packed4_kernel<<<blocks, kGr00tBlockSize, 0, stream>>>(
      static_cast<const Bf16x4*>(projection), static_cast<const Bf16x4*>(bias),
      static_cast<const Bf16x4*>(residual), static_cast<Bf16x4*>(output),
      packed_count, cols / 4);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_gr00t_bias_then_residual_bf16_packed4(
    const void* projection, const void* bias, const void* residual,
    void* output, int rows, int cols, cudaStream_t stream) {
  if (projection == nullptr || residual == nullptr || output == nullptr ||
      rows <= 0 || cols <= 0 || (cols & 3) != 0)
    return cudaErrorInvalidValue;
  const uintptr_t pointers = reinterpret_cast<uintptr_t>(projection) |
      reinterpret_cast<uintptr_t>(residual) |
      reinterpret_cast<uintptr_t>(output) |
      reinterpret_cast<uintptr_t>(bias);
  if ((pointers & (alignof(Bf16x4) - 1)) != 0) return cudaErrorInvalidValue;
  const int64_t packed_count = static_cast<int64_t>(rows) * (cols / 4);
  const int blocks =
      static_cast<int>((packed_count + kGr00tBlockSize - 1) / kGr00tBlockSize);
  bias_then_residual_bf16_packed4_kernel<<<blocks, kGr00tBlockSize, 0, stream>>>(
      static_cast<const Bf16x4*>(projection), static_cast<const Bf16x4*>(bias),
      static_cast<const Bf16x4*>(residual), static_cast<Bf16x4*>(output),
      packed_count, cols / 4);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_gr00t_bias_residual_layer_norm_bf16_cached_1024(
    const void* projection, const void* projection_bias, const void* residual,
    const void* norm_weight, const void* norm_bias, void* hidden,
    void* normalized, int rows, int cols, float eps, cudaStream_t stream) {
  if (projection == nullptr || residual == nullptr || norm_weight == nullptr ||
      norm_bias == nullptr || hidden == nullptr || normalized == nullptr ||
      rows <= 0 || cols != 1024)
    return cudaErrorInvalidValue;
  bias_residual_layer_norm_bf16_cached_1024_kernel<<<rows, kGr00tBlockSize, 0,
                                                     stream>>>(
      static_cast<const __nv_bfloat16*>(projection),
      static_cast<const __nv_bfloat16*>(projection_bias),
      static_cast<const __nv_bfloat16*>(residual),
      static_cast<const __nv_bfloat16*>(norm_weight),
      static_cast<const __nv_bfloat16*>(norm_bias),
      static_cast<__nv_bfloat16*>(hidden),
      static_cast<__nv_bfloat16*>(normalized), rows, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_gr00t_bias_then_residual_layer_norm_bf16_cached_1536(
    const void* projection, const void* projection_bias, const void* residual,
    const void* norm_weight, const void* norm_bias, void* hidden,
    void* normalized, int rows, int cols, float eps, cudaStream_t stream) {
  if (projection == nullptr || projection_bias == nullptr ||
      residual == nullptr || norm_weight == nullptr || norm_bias == nullptr ||
      hidden == nullptr || normalized == nullptr || rows <= 0 || cols != 1536)
    return cudaErrorInvalidValue;
  bias_then_residual_layer_norm_bf16_cached_1536_kernel<<<rows, kGr00tBlockSize,
                                                          0, stream>>>(
      static_cast<const __nv_bfloat16*>(projection),
      static_cast<const __nv_bfloat16*>(projection_bias),
      static_cast<const __nv_bfloat16*>(residual),
      static_cast<const __nv_bfloat16*>(norm_weight),
      static_cast<const __nv_bfloat16*>(norm_bias),
      static_cast<__nv_bfloat16*>(hidden),
      static_cast<__nv_bfloat16*>(normalized), rows, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t
apxinf_gr00t_bias_then_residual_adaptive_layer_norm_bf16_cached_1536(
    const void* projection, const void* projection_bias, const void* residual,
    const void* modulation, void* hidden, void* normalized, int rows, int cols,
    float eps, cudaStream_t stream) {
  if (projection == nullptr || projection_bias == nullptr ||
      residual == nullptr || modulation == nullptr || hidden == nullptr ||
      normalized == nullptr || rows <= 0 || cols != 1536)
    return cudaErrorInvalidValue;
  bias_then_residual_adaptive_layer_norm_bf16_cached_1536_kernel
      <<<rows, kGr00tBlockSize, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(projection),
          static_cast<const __nv_bfloat16*>(projection_bias),
          static_cast<const __nv_bfloat16*>(residual),
          static_cast<const __nv_bfloat16*>(modulation),
          static_cast<__nv_bfloat16*>(hidden),
          static_cast<__nv_bfloat16*>(normalized), rows, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_gr00t_bias_residual_layer_norm_quant_bf16_e4m3(
    const void* projection, const void* projection_bias, const void* residual,
    const void* norm_weight, const void* norm_bias, void* hidden,
    void* normalized, int rows, int cols, float eps, float scale,
    cudaStream_t stream) {
  if (projection == nullptr || residual == nullptr || norm_weight == nullptr ||
      norm_bias == nullptr || hidden == nullptr || normalized == nullptr ||
      rows <= 0 || cols <= 0 || !std::isfinite(scale) || scale <= 0.0f)
    return cudaErrorInvalidValue;
  bias_residual_layer_norm_quant_bf16_e4m3_kernel<<<rows, kGr00tBlockSize, 0,
                                                     stream>>>(
      static_cast<const __nv_bfloat16*>(projection),
      static_cast<const __nv_bfloat16*>(projection_bias),
      static_cast<const __nv_bfloat16*>(residual),
      static_cast<const __nv_bfloat16*>(norm_weight),
      static_cast<const __nv_bfloat16*>(norm_bias),
      static_cast<__nv_bfloat16*>(hidden),
      static_cast<__nv_fp8_e4m3*>(normalized), rows, cols, eps, 1.0f / scale);
  return cudaGetLastError();
}

// ── RoPE launchers ───────────────────────────────────────────────────────

extern "C" cudaError_t apxinf_gr00t_rope_mrope_bf16(
    const void* input, void* output, uint32_t head_dim, uint32_t n_heads,
    uint32_t seq_len, float theta, const void* pos_ids, uint32_t sec_h,
    uint32_t sec_w, cudaStream_t stream) {
  dim3 grid((head_dim / 2 + kGr00tBlockSize - 1) / kGr00tBlockSize, n_heads,
            seq_len);
  dim3 block(kGr00tBlockSize, 1, 1);
  rope_mrope_bf16_kernel<<<grid, block, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<__nv_bfloat16*>(output), head_dim, n_heads, seq_len, theta,
      static_cast<const uint32_t*>(pos_ids), sec_h, sec_w);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_gr00t_build_vision_rotation_table_f32(
    const void* pos_ids, void* rotation_table, uint32_t head_dim,
    uint32_t seq_len, float theta, cudaStream_t stream) {
  if (pos_ids == nullptr || rotation_table == nullptr || head_dim == 0 ||
      (head_dim % 2) != 0 || seq_len == 0 || !(theta > 0.0f))
    return cudaErrorInvalidValue;
  const uint64_t total = static_cast<uint64_t>(seq_len) * (head_dim / 2);
  dim3 grid(static_cast<uint32_t>((total + kGr00tBlockSize - 1) /
                                  kGr00tBlockSize),
            1, 1);
  dim3 block(kGr00tBlockSize, 1, 1);
  build_vision_rotation_table_f32_kernel<<<grid, block, 0, stream>>>(
      static_cast<const uint32_t*>(pos_ids),
      static_cast<float2*>(rotation_table), head_dim, seq_len, theta);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_gr00t_rope_vision_2d_pair_bf16(
    const void* q, const void* k, void* q_out, void* k_out, uint32_t head_dim,
    uint32_t n_heads, uint32_t seq_len, float theta, const void* pos_ids,
    cudaStream_t stream) {
  uint64_t total = static_cast<uint64_t>(seq_len) * n_heads * (head_dim / 2);
  dim3 grid(static_cast<uint32_t>((total + kGr00tBlockSize - 1) /
                                  kGr00tBlockSize),
            1, 1);
  dim3 block(kGr00tBlockSize, 1, 1);
  rope_vision_2d_pair_bf16_kernel<<<grid, block, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(q),
      static_cast<const __nv_bfloat16*>(k),
      static_cast<__nv_bfloat16*>(q_out), static_cast<__nv_bfloat16*>(k_out),
      head_dim, n_heads, seq_len, theta,
      static_cast<const uint32_t*>(pos_ids));
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_gr00t_qk_rms_norm_mrope_bf16_with_threads(
    const void* query_input, const void* query_weight, void* query_output,
    const void* key_input, const void* key_weight, void* key_output,
    uint32_t head_dim, uint32_t query_heads, uint32_t key_heads,
    uint32_t seq_len, float eps, float theta, const void* pos_ids,
    uint32_t sec_h, uint32_t sec_w, uint32_t block_threads,
    cudaStream_t stream) {
  if ((block_threads != 128 && block_threads != 256) ||
      query_input == nullptr || query_weight == nullptr ||
      query_output == nullptr || key_input == nullptr ||
      key_weight == nullptr || key_output == nullptr || pos_ids == nullptr ||
      head_dim == 0 || (head_dim % 2) != 0 || query_heads == 0 ||
      key_heads == 0 || seq_len == 0 || head_dim / 2 > block_threads ||
      !(eps > 0.0f) || !(theta > 0.0f))
    return cudaErrorInvalidValue;
  dim3 grid(seq_len, query_heads + key_heads, 1);
  dim3 block(block_threads, 1, 1);
  qk_rms_norm_mrope_bf16_kernel<<<grid, block,
                                  head_dim * sizeof(__nv_bfloat16), stream>>>(
      static_cast<const __nv_bfloat16*>(query_input),
      static_cast<const __nv_bfloat16*>(query_weight),
      static_cast<__nv_bfloat16*>(query_output),
      static_cast<const __nv_bfloat16*>(key_input),
      static_cast<const __nv_bfloat16*>(key_weight),
      static_cast<__nv_bfloat16*>(key_output), head_dim, query_heads,
      key_heads, seq_len, eps, theta, static_cast<const uint32_t*>(pos_ids),
      sec_h, sec_w);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_gr00t_qkv_split_bias_vision_rope_bf16(
    const void* qkv, const void* bias, void* q_out, void* k_out, void* v_out,
    uint32_t head_dim, uint32_t n_heads, uint32_t seq_len, float theta,
    const void* pos_ids, cudaStream_t stream) {
  uint64_t total = static_cast<uint64_t>(seq_len) * n_heads * (head_dim / 2);
  dim3 grid(static_cast<uint32_t>((total + kGr00tBlockSize - 1) /
                                  kGr00tBlockSize),
            1, 1);
  dim3 block(kGr00tBlockSize, 1, 1);
  qkv_split_bias_vision_rope_bf16_kernel<false><<<grid, block, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(qkv),
      static_cast<const __nv_bfloat16*>(bias),
      static_cast<__nv_bfloat16*>(q_out), static_cast<__nv_bfloat16*>(k_out),
      static_cast<__nv_bfloat16*>(v_out), head_dim, n_heads, seq_len, theta,
      static_cast<const uint32_t*>(pos_ids), nullptr);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_gr00t_qkv_split_bias_vision_rope_precomputed_bf16(
    const void* qkv, const void* bias, void* q_out, void* k_out, void* v_out,
    uint32_t head_dim, uint32_t n_heads, uint32_t seq_len,
    const void* rotation_table, cudaStream_t stream) {
  if (qkv == nullptr || bias == nullptr || q_out == nullptr ||
      k_out == nullptr || v_out == nullptr || rotation_table == nullptr ||
      head_dim == 0 || (head_dim % 2) != 0 || n_heads == 0 || seq_len == 0)
    return cudaErrorInvalidValue;
  uint64_t total = static_cast<uint64_t>(seq_len) * n_heads * (head_dim / 2);
  dim3 grid(static_cast<uint32_t>((total + kGr00tBlockSize - 1) /
                                  kGr00tBlockSize),
            1, 1);
  dim3 block(kGr00tBlockSize, 1, 1);
  qkv_split_bias_vision_rope_bf16_kernel<true><<<grid, block, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(qkv),
      static_cast<const __nv_bfloat16*>(bias),
      static_cast<__nv_bfloat16*>(q_out), static_cast<__nv_bfloat16*>(k_out),
      static_cast<__nv_bfloat16*>(v_out), head_dim, n_heads, seq_len, 0.0f,
      nullptr, static_cast<const float2*>(rotation_table));
  return cudaGetLastError();
}

extern "C" cudaError_t
apxinf_gr00t_qkv_split_bias_vision_rope_precomputed_vec2_bf16(
    const void* qkv, const void* bias, void* q_out, void* k_out, void* v_out,
    uint32_t head_dim, uint32_t n_heads, uint32_t seq_len,
    const void* rotation_table, cudaStream_t stream) {
  if (qkv == nullptr || bias == nullptr || q_out == nullptr ||
      k_out == nullptr || v_out == nullptr || rotation_table == nullptr ||
      head_dim == 0 || head_dim % 4 != 0 || n_heads == 0 || seq_len == 0)
    return cudaErrorInvalidValue;
  const uint64_t total =
      static_cast<uint64_t>(seq_len) * n_heads * (head_dim / 4);
  constexpr uint32_t threads = 128;
  const dim3 grid(static_cast<uint32_t>((total + threads - 1) / threads), 1, 1);
  qkv_split_bias_vision_rope_precomputed_bf16_vec2_kernel<<<grid, threads, 0,
                                                            stream>>>(
      static_cast<const __nv_bfloat16*>(qkv),
      static_cast<const __nv_bfloat16*>(bias),
      static_cast<__nv_bfloat16*>(q_out), static_cast<__nv_bfloat16*>(k_out),
      static_cast<__nv_bfloat16*>(v_out), head_dim, n_heads, seq_len,
      static_cast<const float2*>(rotation_table));
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_gr00t_split_strided_qkv_bf16(
    const void* qkv, void* q, void* k, void* v, int tokens, int hidden,
    cudaStream_t stream) {
  if (qkv == nullptr || q == nullptr || k == nullptr || v == nullptr ||
      tokens <= 0 || hidden <= 0)
    return cudaErrorInvalidValue;
  const int64_t total = static_cast<int64_t>(tokens) * hidden;
  const int blocks =
      static_cast<int>((total + kGr00tBlockSize - 1) / kGr00tBlockSize);
  gr00t_split_strided_qkv_bf16_kernel<<<blocks, kGr00tBlockSize, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(qkv), static_cast<__nv_bfloat16*>(q),
      static_cast<__nv_bfloat16*>(k), static_cast<__nv_bfloat16*>(v), tokens,
      hidden);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_gr00t_quantize_bf16_e4m3_packed8(
    const void* input, void* output, int64_t count, float scale,
    cudaStream_t stream) {
  if (input == nullptr || output == nullptr || count <= 0 || !(scale > 0.0f))
    return cudaErrorInvalidValue;
  constexpr int threads = 256;
  const bool aligned =
      (reinterpret_cast<uintptr_t>(input) % alignof(Gr00tBf16Pack8) == 0) &&
      (reinterpret_cast<uintptr_t>(output) % alignof(Gr00tFp8Pack8) == 0);
  if (!aligned || count < 8) {
    int blocks = static_cast<int>((count + threads - 1) / threads);
    blocks = blocks > 4096 ? 4096 : blocks;
    gr00t_quantize_bf16_e4m3_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input),
        static_cast<__nv_fp8_e4m3*>(output), count, 1.0f / scale);
    return cudaGetLastError();
  }
  const int64_t vector_count = count / 8;
  int blocks = static_cast<int>((vector_count + threads - 1) / threads);
  blocks = blocks > 4096 ? 4096 : blocks;
  gr00t_quantize_bf16_e4m3_packed8_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const Gr00tBf16Pack8*>(input),
      static_cast<Gr00tFp8Pack8*>(output), vector_count, 1.0f / scale);
  const int64_t tail = count - vector_count * 8;
  if (tail != 0) {
    gr00t_quantize_bf16_e4m3_kernel<<<1, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input) + vector_count * 8,
        static_cast<__nv_fp8_e4m3*>(output) + vector_count * 8, tail,
        1.0f / scale);
  }
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_gr00t_concat_rows_quantize_bf16_e4m3(
    const void* first, const void* second, void* output, int first_rows,
    int second_rows, int cols, float scale, cudaStream_t stream) {
  if (first == nullptr || second == nullptr || output == nullptr ||
      first_rows <= 0 || second_rows <= 0 || cols <= 0 || !(scale > 0.0f))
    return cudaErrorInvalidValue;
  const int64_t first_count = static_cast<int64_t>(first_rows) * cols;
  const int64_t total_count =
      static_cast<int64_t>(first_rows + second_rows) * cols;
  int blocks = static_cast<int>((total_count + 255) / 256);
  blocks = blocks > 4096 ? 4096 : blocks;
  gr00t_concat_rows_quantize_bf16_e4m3_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(first),
      static_cast<const __nv_bfloat16*>(second),
      static_cast<__nv_fp8_e4m3*>(output), first_count, total_count,
      1.0f / scale);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_gr00t_dequantize_int32_bf16(
    const void* accumulators, const void* row_scales,
    const void* column_scales, void* output, int rows, int cols,
    cudaStream_t stream) {
  if (accumulators == nullptr || row_scales == nullptr ||
      column_scales == nullptr || output == nullptr || rows <= 0 || cols <= 0)
    return cudaErrorInvalidValue;
  const dim3 grid((cols + kGr00tBlockSize - 1) / kGr00tBlockSize, rows);
  gr00t_dequantize_int32_bf16_kernel<<<grid, kGr00tBlockSize, 0, stream>>>(
      static_cast<const int32_t*>(accumulators),
      static_cast<const float*>(row_scales),
      static_cast<const float*>(column_scales),
      static_cast<__nv_bfloat16*>(output), rows, cols);
  return cudaGetLastError();
}

// ── W8A8 (INT8) quantization launchers ───────────────────────────────────
// Copied from the legacy w8a8_adapter.cu. These produce the dynamically
// row-quantized INT8 activations the W8A8 GEMM consumes.

extern "C" cudaError_t apxinf_gr00t_quantize_rows_bf16_int8(
    const void* input, void* output, void* scales, int rows, int cols,
    cudaStream_t stream) {
  if (input == nullptr || output == nullptr || scales == nullptr ||
      rows <= 0 || cols <= 0)
    return cudaErrorInvalidValue;
  gr00t_quantize_rows_bf16_int8_kernel<<<rows, kGr00tBlockSize, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input), static_cast<int8_t*>(output),
      static_cast<float*>(scales), rows, cols);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_gr00t_quantize_rows_bf16_int8_packed4(
    const void* input, void* output, void* scales, int rows, int cols,
    cudaStream_t stream) {
  if (input == nullptr || output == nullptr || scales == nullptr ||
      rows <= 0 || cols <= 0 || cols % 4 != 0)
    return cudaErrorInvalidValue;
  gr00t_quantize_rows_bf16_int8_vec4_kernel<<<
      rows, 512, static_cast<size_t>(cols) * sizeof(__nv_bfloat16), stream>>>(
      static_cast<const __nv_bfloat16*>(input), static_cast<int8_t*>(output),
      static_cast<float*>(scales), rows, cols);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_gr00t_bias_gelu_quantize_rows_bf16_int8(
    const void* input, const void* bias, void* output, void* scales, int rows,
    int cols, cudaStream_t stream) {
  if (input == nullptr || bias == nullptr || output == nullptr ||
      scales == nullptr || rows <= 0 || cols <= 0)
    return cudaErrorInvalidValue;
  if (cols % 4 == 0) {
    gr00t_bias_gelu_quantize_rows_bf16_int8_vec4_kernel<<<
        rows, 512, static_cast<size_t>(cols) * sizeof(__nv_bfloat16), stream>>>(
        static_cast<const __nv_bfloat16*>(input),
        static_cast<const __nv_bfloat16*>(bias), static_cast<int8_t*>(output),
        static_cast<float*>(scales), rows, cols);
  } else {
    gr00t_bias_gelu_quantize_rows_bf16_int8_kernel<<<
        rows, kGr00tBlockSize,
        static_cast<size_t>(cols) * sizeof(__nv_bfloat16), stream>>>(
        static_cast<const __nv_bfloat16*>(input),
        static_cast<const __nv_bfloat16*>(bias), static_cast<int8_t*>(output),
        static_cast<float*>(scales), rows, cols);
  }
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_gr00t_adaptive_layer_norm_quantize_rows_bf16_int8(
    const void* input, const void* modulation, void* output, void* quantized,
    void* scales, int rows, int cols, float eps, cudaStream_t stream) {
  if (input == nullptr || modulation == nullptr || output == nullptr ||
      quantized == nullptr || scales == nullptr || rows <= 0 || cols <= 0 ||
      !(eps > 0.0f))
    return cudaErrorInvalidValue;
  gr00t_adaptive_layer_norm_quantize_rows_bf16_int8_kernel<<<
      rows, kGr00tBlockSize, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(modulation),
      static_cast<__nv_bfloat16*>(output), static_cast<int8_t*>(quantized),
      static_cast<float*>(scales), rows, cols, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_gr00t_layer_norm_quantize_rows_bf16_int8(
    const void* input, const void* weight, const void* bias, void* output,
    void* quantized, void* scales, int rows, int cols, float eps,
    cudaStream_t stream) {
  if (input == nullptr || weight == nullptr || bias == nullptr ||
      output == nullptr || quantized == nullptr || scales == nullptr ||
      rows <= 0 || cols <= 0 || !(eps > 0.0f))
    return cudaErrorInvalidValue;
  gr00t_layer_norm_quantize_rows_bf16_int8_kernel<<<rows, kGr00tBlockSize, 0,
                                                    stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<const __nv_bfloat16*>(bias),
      static_cast<__nv_bfloat16*>(output), static_cast<int8_t*>(quantized),
      static_cast<float*>(scales), rows, cols, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_gr00t_silu_mul_quantize_rows_bf16_int8(
    const void* gate, const void* up, void* output, void* scales, int rows,
    int cols, cudaStream_t stream) {
  if (gate == nullptr || up == nullptr || output == nullptr ||
      scales == nullptr || rows <= 0 || cols <= 0)
    return cudaErrorInvalidValue;
  gr00t_silu_mul_quantize_rows_bf16_int8_kernel<<<
      rows, kGr00tBlockSize,
      static_cast<size_t>(cols) * sizeof(__nv_bfloat16), stream>>>(
      static_cast<const __nv_bfloat16*>(gate),
      static_cast<const __nv_bfloat16*>(up), static_cast<int8_t*>(output),
      static_cast<float*>(scales), rows, cols);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_gr00t_silu_mul_quantize_rows_bf16_int8_packed4(
    const void* gate, const void* up, void* output, void* scales, int rows,
    int cols, cudaStream_t stream) {
  if (gate == nullptr || up == nullptr || output == nullptr ||
      scales == nullptr || rows <= 0 || cols <= 0 || cols % 4 != 0 ||
      (reinterpret_cast<uintptr_t>(gate) & 7) != 0 ||
      (reinterpret_cast<uintptr_t>(up) & 7) != 0 ||
      (reinterpret_cast<uintptr_t>(output) & 3) != 0)
    return cudaErrorInvalidValue;
  gr00t_silu_mul_quantize_rows_bf16_int8_vec4_kernel<<<
      rows, 512, static_cast<size_t>(cols) * sizeof(__nv_bfloat16), stream>>>(
      static_cast<const __nv_bfloat16*>(gate),
      static_cast<const __nv_bfloat16*>(up), static_cast<int8_t*>(output),
      static_cast<float*>(scales), rows, cols);
  return cudaGetLastError();
}
