#pragma once

struct alignas(8) Bf16x4 {
  __nv_bfloat162 low;
  __nv_bfloat162 high;
};

// Copyright 2026 apxinf contributors.
// Pure CUDA operators grouped by physical operation; launch policy lives under adapters/.

// ── SiLU ──────────────────────────────────────────────────────────────────

__global__ void silu_f32_kernel(
    const float* input, float* output, uint32_t count)
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    float x = input[gid];
    output[gid] = x / (1.0f + expf(-x));
}



// ── SiLU (bf16) ──────────────────────────────────────────────────────────

__global__ void silu_bf16_kernel(
    const __nv_bfloat16* input, __nv_bfloat16* output, uint32_t count)
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    float x = __bfloat162float(input[gid]);
    float y = x / (1.0f + expf(-x));
    output[gid] = __float2bfloat16(y);
}



// ── Fused SiLU + Mul (bf16) — reads gate and up from a single packed ────
// buffer, writes silu(gate) * up. Used by the fused Gate/Up GEMM path:
// one GEMM produces [1, 2*inter], this kernel reads both halves and
// writes [1, inter]. Replaces separate silu + mul kernels (−1 launch).

__global__ void silu_mul_bf16_kernel(
    const __nv_bfloat16* gate_up, __nv_bfloat16* output, uint32_t inter)
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= inter) return;
    float g = __bfloat162float(gate_up[gid]);
    float u = __bfloat162float(gate_up[gid + inter]);
    float s = g / (1.0f + expf(-g));
    output[gid] = __float2bfloat16(s * u);
}



// ── GELU (tanh approximation, bf16) — Qwen3-VL vision MLP ────────────────
//
// PyTorch's `gelu_pytorch_tanh` (a.k.a. hidden_act="gelu_pytorch_tanh"):
//     y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

__global__ void gelu_tanh_bf16_kernel(
    const __nv_bfloat16* input, __nv_bfloat16* output, uint32_t count)
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    float x = __bfloat162float(input[gid]);
    // sqrt(2/pi) ~= 0.7978845608028654
    const float kBeta  = 0.7978845608028654f;
    const float kAlpha = 0.044715f;
    float inner = kBeta * (x + kAlpha * x * x * x);
    float y = 0.5f * x * (1.0f + tanhf(inner));
    output[gid] = __float2bfloat16(y);
}




__global__ void bias_gelu_quant_f16_e4m3_kernel(
    const half* input, const half* bias, __nv_fp8_e4m3* output,
    int64_t count, int cols, float inverse_scale) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    float x = __half2float(input[index]) + __half2float(bias[index % cols]);
    float gelu = gelu_tanh(x);
    gelu = fminf(448.0f, fmaxf(-448.0f, gelu * inverse_scale));
    output[index] = static_cast<__nv_fp8_e4m3>(gelu);
  }
}

__global__ void bias_silu_quant_f16_e4m3_kernel(
    const half* input, const half* bias, __nv_fp8_e4m3* output,
    int64_t count, int cols, float inverse_scale) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    float value = __half2float(input[index]);
    if (bias != nullptr) value += __half2float(bias[index % cols]);
    value = value / (1.0f + expf(-value));
    value = fminf(448.0f, fmaxf(-448.0f, value * inverse_scale));
    output[index] = static_cast<__nv_fp8_e4m3>(value);
  }
}

__global__ void bias_silu_f16_kernel(
    const half* input, const half* bias, half* output, int64_t count, int cols) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    float value = __half2float(input[index]);
    if (bias != nullptr) value += __half2float(bias[index % cols]);
    output[index] = __float2half(value / (1.0f + expf(-value)));
  }
}

__global__ void geglu_quant_f16_e4m3_kernel(
    const half* gate_up, __nv_fp8_e4m3* output, int rows, int inner,
    float inverse_scale) {
  const int pairs_per_row = inner / 2;
  const int pair_count = rows * pairs_per_row;
  int pair_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for (; pair_index < pair_count; pair_index += stride) {
    const int row = pair_index / pairs_per_row;
    const int col_pair = pair_index - row * pairs_per_row;
    const half* row_input = gate_up + static_cast<int64_t>(row) * 2 * inner;
    float2 gate = __half22float2(reinterpret_cast<const half2*>(row_input)[col_pair]);
    float2 up = __half22float2(
        reinterpret_cast<const half2*>(row_input + inner)[col_pair]);
    float2 value;
    value.x = gelu_tanh(gate.x) * up.x * inverse_scale;
    value.y = gelu_tanh(gate.y) * up.y * inverse_scale;
    value.x = fminf(448.0f, fmaxf(-448.0f, value.x));
    value.y = fminf(448.0f, fmaxf(-448.0f, value.y));
    reinterpret_cast<__nv_fp8x2_e4m3*>(output + static_cast<int64_t>(row) * inner)
        [col_pair] = static_cast<__nv_fp8x2_e4m3>(value);
  }
}

struct alignas(4) Fp8x4 {
  __nv_fp8x2_e4m3 low;
  __nv_fp8x2_e4m3 high;
};

struct alignas(8) Fp8x8 {
  __nv_fp8x2_e4m3 pair0;
  __nv_fp8x2_e4m3 pair1;
  __nv_fp8x2_e4m3 pair2;
  __nv_fp8x2_e4m3 pair3;
};

// Eight output values per thread. This increases independent arithmetic per
// thread on the language path and further amortizes address/loop work. Each
// value keeps the packed4 kernel's exact GeGLU, clamp, and fp8x2 conversion
// sequence; packed4 remains the alignment/divisibility fallback.
__global__ void geglu_quant_f16_e4m3_packed8_kernel(
    const half* gate_up, __nv_fp8_e4m3* output, int rows, int inner,
    float inverse_scale) {
  const int groups_per_row = inner / 8;
  const int group_count = rows * groups_per_row;
  int group_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for (; group_index < group_count; group_index += stride) {
    const int row = group_index / groups_per_row;
    const int group_col = group_index - row * groups_per_row;
    const int pair_col = group_col * 4;
    const half* row_input = gate_up + static_cast<int64_t>(row) * 2 * inner;
    const half2* gate2 = reinterpret_cast<const half2*>(row_input);
    const half2* up2 = reinterpret_cast<const half2*>(row_input + inner);
    const float2 gate0 = __half22float2(gate2[pair_col]);
    const float2 gate1 = __half22float2(gate2[pair_col + 1]);
    const float2 gate2_value = __half22float2(gate2[pair_col + 2]);
    const float2 gate3 = __half22float2(gate2[pair_col + 3]);
    const float2 up0 = __half22float2(up2[pair_col]);
    const float2 up1 = __half22float2(up2[pair_col + 1]);
    const float2 up2_value = __half22float2(up2[pair_col + 2]);
    const float2 up3 = __half22float2(up2[pair_col + 3]);
    float2 value0;
    float2 value1;
    float2 value2;
    float2 value3;
    value0.x = gelu_tanh(gate0.x) * up0.x * inverse_scale;
    value0.y = gelu_tanh(gate0.y) * up0.y * inverse_scale;
    value1.x = gelu_tanh(gate1.x) * up1.x * inverse_scale;
    value1.y = gelu_tanh(gate1.y) * up1.y * inverse_scale;
    value2.x = gelu_tanh(gate2_value.x) * up2_value.x * inverse_scale;
    value2.y = gelu_tanh(gate2_value.y) * up2_value.y * inverse_scale;
    value3.x = gelu_tanh(gate3.x) * up3.x * inverse_scale;
    value3.y = gelu_tanh(gate3.y) * up3.y * inverse_scale;
    value0.x = fminf(448.0f, fmaxf(-448.0f, value0.x));
    value0.y = fminf(448.0f, fmaxf(-448.0f, value0.y));
    value1.x = fminf(448.0f, fmaxf(-448.0f, value1.x));
    value1.y = fminf(448.0f, fmaxf(-448.0f, value1.y));
    value2.x = fminf(448.0f, fmaxf(-448.0f, value2.x));
    value2.y = fminf(448.0f, fmaxf(-448.0f, value2.y));
    value3.x = fminf(448.0f, fmaxf(-448.0f, value3.x));
    value3.y = fminf(448.0f, fmaxf(-448.0f, value3.y));
    reinterpret_cast<Fp8x8*>(output + static_cast<int64_t>(row) * inner)
        [group_col] = {static_cast<__nv_fp8x2_e4m3>(value0),
                       static_cast<__nv_fp8x2_e4m3>(value1),
                       static_cast<__nv_fp8x2_e4m3>(value2),
                       static_cast<__nv_fp8x2_e4m3>(value3)};
  }
}

// Four output values per thread. Retain the canonical half2 loads, float2
// GeGLU arithmetic, clamp order, and fp8x2 conversions while halving address,
// loop, and scheduling work relative to the packed2 fallback above.
__global__ void geglu_quant_f16_e4m3_packed4_kernel(
    const half* gate_up, __nv_fp8_e4m3* output, int rows, int inner,
    float inverse_scale) {
  const int groups_per_row = inner / 4;
  const int group_count = rows * groups_per_row;
  int group_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for (; group_index < group_count; group_index += stride) {
    const int row = group_index / groups_per_row;
    const int group_col = group_index - row * groups_per_row;
    const int pair_col = group_col * 2;
    const half* row_input = gate_up + static_cast<int64_t>(row) * 2 * inner;
    const half2* gate2 = reinterpret_cast<const half2*>(row_input);
    const half2* up2 = reinterpret_cast<const half2*>(row_input + inner);
    const float2 gate_low = __half22float2(gate2[pair_col]);
    const float2 gate_high = __half22float2(gate2[pair_col + 1]);
    const float2 up_low = __half22float2(up2[pair_col]);
    const float2 up_high = __half22float2(up2[pair_col + 1]);
    float2 value_low;
    float2 value_high;
    value_low.x = gelu_tanh(gate_low.x) * up_low.x * inverse_scale;
    value_low.y = gelu_tanh(gate_low.y) * up_low.y * inverse_scale;
    value_high.x = gelu_tanh(gate_high.x) * up_high.x * inverse_scale;
    value_high.y = gelu_tanh(gate_high.y) * up_high.y * inverse_scale;
    value_low.x = fminf(448.0f, fmaxf(-448.0f, value_low.x));
    value_low.y = fminf(448.0f, fmaxf(-448.0f, value_low.y));
    value_high.x = fminf(448.0f, fmaxf(-448.0f, value_high.x));
    value_high.y = fminf(448.0f, fmaxf(-448.0f, value_high.y));
    reinterpret_cast<Fp8x4*>(output + static_cast<int64_t>(row) * inner)
        [group_col] = {static_cast<__nv_fp8x2_e4m3>(value_low),
                       static_cast<__nv_fp8x2_e4m3>(value_high)};
  }
}


__global__ void bias_activation_bf16_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* bias,
    __nv_bfloat16* output, int64_t count, int cols, int activation) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    float value = __bfloat162float(input[index]);
    if (bias != nullptr) value += __bfloat162float(bias[index % cols]);
    if (activation == 1) value = gelu_tanh(value);
    if (activation == 2) value = value / (1.0f + expf(-value));
    if (activation == 3) value = fmaxf(value, 0.0f);
    output[index] = __float2bfloat16(value);
  }
}

__global__ void bias_activation_bf16_packed2_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* bias,
    __nv_bfloat16* output, int64_t pair_count, int cols, int activation) {
  int64_t pair_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  const int pairs_per_row = cols / 2;
  const __nv_bfloat162* input2 =
      reinterpret_cast<const __nv_bfloat162*>(input);
  const __nv_bfloat162* bias2 =
      reinterpret_cast<const __nv_bfloat162*>(bias);
  __nv_bfloat162* output2 = reinterpret_cast<__nv_bfloat162*>(output);
  for (; pair_index < pair_count; pair_index += stride) {
    const __nv_bfloat162 packed_input = input2[pair_index];
    float first = __bfloat162float(packed_input.x);
    float second = __bfloat162float(packed_input.y);
    if (bias != nullptr) {
      const __nv_bfloat162 packed_bias = bias2[pair_index % pairs_per_row];
      first += __bfloat162float(packed_bias.x);
      second += __bfloat162float(packed_bias.y);
    }
    if (activation == 1) {
      first = gelu_tanh(first);
      second = gelu_tanh(second);
    }
    if (activation == 2) {
      first = first / (1.0f + expf(-first));
      second = second / (1.0f + expf(-second));
    }
    if (activation == 3) {
      first = fmaxf(first, 0.0f);
      second = fmaxf(second, 0.0f);
    }
    output2[pair_index] = __floats2bfloat162_rn(first, second);
  }
}

__global__ void bias_activation_bf16_packed4_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* bias,
    __nv_bfloat16* output, int64_t quad_count, int cols, int activation) {
  int64_t quad_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  const int quads_per_row = cols / 4;
  const Bf16x4* input4 = reinterpret_cast<const Bf16x4*>(input);
  const Bf16x4* bias4 = reinterpret_cast<const Bf16x4*>(bias);
  Bf16x4* output4 = reinterpret_cast<Bf16x4*>(output);
  for (; quad_index < quad_count; quad_index += stride) {
    const Bf16x4 packed_input = input4[quad_index];
    float values[4] = {
        __bfloat162float(packed_input.low.x),
        __bfloat162float(packed_input.low.y),
        __bfloat162float(packed_input.high.x),
        __bfloat162float(packed_input.high.y),
    };
    if (bias != nullptr) {
      const Bf16x4 packed_bias = bias4[quad_index % quads_per_row];
      values[0] += __bfloat162float(packed_bias.low.x);
      values[1] += __bfloat162float(packed_bias.low.y);
      values[2] += __bfloat162float(packed_bias.high.x);
      values[3] += __bfloat162float(packed_bias.high.y);
    }
    if (activation == 1) {
#pragma unroll
      for (int i = 0; i < 4; ++i) values[i] = gelu_tanh(values[i]);
    }
    if (activation == 2) {
#pragma unroll
      for (int i = 0; i < 4; ++i)
        values[i] = values[i] / (1.0f + expf(-values[i]));
    }
    if (activation == 3) {
#pragma unroll
      for (int i = 0; i < 4; ++i) values[i] = fmaxf(values[i], 0.0f);
    }
    output4[quad_index] = Bf16x4{
        __floats2bfloat162_rn(values[0], values[1]),
        __floats2bfloat162_rn(values[2], values[3]),
    };
  }
}

__global__ void geglu_bf16_kernel(
    const __nv_bfloat16* gate_up, __nv_bfloat16* output,
    int rows, int inner) {
  const int64_t count = static_cast<int64_t>(rows) * inner;
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    const int row = static_cast<int>(index / inner);
    const int col = static_cast<int>(index % inner);
    const __nv_bfloat16* row_input =
        gate_up + static_cast<int64_t>(row) * 2 * inner;
    output[index] = __float2bfloat16(
        gelu_tanh(__bfloat162float(row_input[col])) *
        __bfloat162float(row_input[inner + col]));
  }
}

__global__ void geglu_bf16_packed2_kernel(
    const __nv_bfloat16* gate_up, __nv_bfloat16* output,
    int rows, int inner) {
  const int pairs_per_row = inner / 2;
  const int pair_count = rows * pairs_per_row;
  int pair_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for (; pair_index < pair_count; pair_index += stride) {
    const int row = pair_index / pairs_per_row;
    const int col_pair = pair_index - row * pairs_per_row;
    const __nv_bfloat16* row_input =
        gate_up + static_cast<int64_t>(row) * 2 * inner;
    const __nv_bfloat162 gate =
        reinterpret_cast<const __nv_bfloat162*>(row_input)[col_pair];
    const __nv_bfloat162 up =
        reinterpret_cast<const __nv_bfloat162*>(row_input + inner)[col_pair];
    reinterpret_cast<__nv_bfloat162*>(
        output + static_cast<int64_t>(row) * inner)[col_pair] =
        __floats2bfloat162_rn(
            gelu_tanh(__bfloat162float(gate.x)) * __bfloat162float(up.x),
            gelu_tanh(__bfloat162float(gate.y)) * __bfloat162float(up.y));
  }
}

__global__ void geglu_bf16_packed4_kernel(
    const __nv_bfloat16* gate_up, __nv_bfloat16* output,
    int rows, int inner) {
  const int quads_per_row = inner / 4;
  const int quad_count = rows * quads_per_row;
  int quad_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for (; quad_index < quad_count; quad_index += stride) {
    const int row = quad_index / quads_per_row;
    const int col_quad = quad_index - row * quads_per_row;
    const __nv_bfloat16* row_input =
        gate_up + static_cast<int64_t>(row) * 2 * inner;
    const Bf16x4 gate =
        reinterpret_cast<const Bf16x4*>(row_input)[col_quad];
    const Bf16x4 up =
        reinterpret_cast<const Bf16x4*>(row_input + inner)[col_quad];
    reinterpret_cast<Bf16x4*>(
        output + static_cast<int64_t>(row) * inner)[col_quad] = Bf16x4{
        __floats2bfloat162_rn(
            gelu_tanh(__bfloat162float(gate.low.x)) *
                __bfloat162float(up.low.x),
            gelu_tanh(__bfloat162float(gate.low.y)) *
                __bfloat162float(up.low.y)),
        __floats2bfloat162_rn(
            gelu_tanh(__bfloat162float(gate.high.x)) *
                __bfloat162float(up.high.x),
            gelu_tanh(__bfloat162float(gate.high.y)) *
                __bfloat162float(up.high.y)),
    };
  }
}
