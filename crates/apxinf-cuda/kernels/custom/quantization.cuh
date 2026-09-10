#pragma once

// Copyright 2026 apxinf contributors.
// Pure CUDA operators grouped by physical operation; launch policy lives under adapters/.

struct alignas(16) Bf16Pack8 {
  __nv_bfloat16 values[8];
};

struct alignas(8) Bf16Pack4 {
  __nv_bfloat16 values[4];
};

struct alignas(8) Fp8Pack8 {
  __nv_fp8_e4m3 values[8];
};

struct alignas(4) Fp8Pack4 {
  __nv_fp8_e4m3 values[4];
};

__global__ void quantize_f16_e4m3_kernel(
    const half* input, __nv_fp8_e4m3* output, int64_t count,
    float inverse_scale) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    float value = fminf(448.0f, fmaxf(-448.0f,
        __half2float(input[index]) * inverse_scale));
    output[index] = static_cast<__nv_fp8_e4m3>(value);
  }
}

__global__ void quantize_bf16_e4m3_kernel(
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

// Four BF16 values per thread with two bfloat162 loads and one uint32 store.
// The scalar kernel remains the fallback for an unaligned tail.
__global__ void quantize_bf16_e4m3_packed4_kernel(
    const __nv_bfloat16* input, __nv_fp8_e4m3* output, int64_t vector_count,
    float inverse_scale) {
  int64_t index =
      (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
  const int64_t stride =
      static_cast<int64_t>(blockDim.x) * gridDim.x * 4;
  const __nv_bfloat162* input2 =
      reinterpret_cast<const __nv_bfloat162*>(input);
  for (; index < vector_count; index += stride) {
    const __nv_bfloat162 first = input2[index / 2];
    const __nv_bfloat162 second = input2[index / 2 + 1];
    const __nv_fp8x2_e4m3 first_fp8(make_float2(
        __bfloat162float(first.x) * inverse_scale,
        __bfloat162float(first.y) * inverse_scale));
    const __nv_fp8x2_e4m3 second_fp8(make_float2(
        __bfloat162float(second.x) * inverse_scale,
        __bfloat162float(second.y) * inverse_scale));
    reinterpret_cast<uint32_t*>(output)[index / 4] =
        static_cast<uint32_t>(first_fp8.__x) |
        (static_cast<uint32_t>(second_fp8.__x) << 16);
  }
}

__global__ void quantize_rows_bf16_e4m3_kernel(
    const __nv_bfloat16* input, __nv_fp8_e4m3* output, float* scales,
    int rows, int input_cols, int output_cols) {
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  float maximum = 0.0f;
  for (int col = threadIdx.x; col < input_cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * input_cols + col;
    maximum = fmaxf(maximum, fabsf(__bfloat162float(input[index])));
  }
  const float scale = fmaxf(block_max(maximum, scratch) / 448.0f, 1.0e-12f);
  if (threadIdx.x == 0) scales[row] = scale;
  for (int col = threadIdx.x; col < output_cols; col += blockDim.x) {
    const int64_t output_index = static_cast<int64_t>(row) * output_cols + col;
    if (col < input_cols) {
      const int64_t input_index = static_cast<int64_t>(row) * input_cols + col;
      const float value = fminf(448.0f, fmaxf(
          -448.0f, __bfloat162float(input[input_index]) / scale));
      output[output_index] = static_cast<__nv_fp8_e4m3>(value);
    } else {
      output[output_index] = static_cast<__nv_fp8_e4m3>(0.0f);
    }
  }
}

__global__ void quantize_rows_bf16_e4m3_vec8_kernel(
    const __nv_bfloat16* input, __nv_fp8_e4m3* output, float* scales,
    int rows, int input_cols, int output_cols) {
  constexpr int kRowsPerBlock = 8;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * kRowsPerBlock + warp;
  if (row >= rows) return;

  const int input_vectors = input_cols / 8;
  const int output_vectors = output_cols / 8;
  const int64_t input_offset = static_cast<int64_t>(row) * input_cols;
  const int64_t output_offset = static_cast<int64_t>(row) * output_cols;
  float maximum = 0.0f;
  for (int vector = lane; vector < input_vectors; vector += 32) {
    const Bf16Pack8 values =
        *reinterpret_cast<const Bf16Pack8*>(input + input_offset + vector * 8);
#pragma unroll
    for (int item = 0; item < 8; ++item) {
      maximum = fmaxf(maximum, fabsf(__bfloat162float(values.values[item])));
    }
  }
  const float scale = fmaxf(warp_max(maximum) / 448.0f, 1.0e-12f);
  const float inverse_scale = 1.0f / scale;
  if (lane == 0) scales[row] = scale;

  for (int vector = lane; vector < output_vectors; vector += 32) {
    Fp8Pack8 quantized{};
    if (vector < input_vectors) {
      const Bf16Pack8 values = *reinterpret_cast<const Bf16Pack8*>(
          input + input_offset + vector * 8);
#pragma unroll
      for (int item = 0; item < 8; ++item) {
        float value = __bfloat162float(values.values[item]) * inverse_scale;
        value = fminf(448.0f, fmaxf(-448.0f, value));
        quantized.values[item] = static_cast<__nv_fp8_e4m3>(value);
      }
    }
    *reinterpret_cast<Fp8Pack8*>(output + output_offset + vector * 8) = quantized;
  }
}

// One warp owns one row so the normalization reduction, row amax, and E4M3
// conversion stay in a single launch. Padding columns are written as zero and
// can be consumed directly by an aligned rowwise GEMM.
__global__ void rms_norm_quantize_rows_bf16_e4m3_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    __nv_fp8_e4m3* output, float* scales, int rows, int input_cols,
    int output_cols, float eps) {
  constexpr int kWarpsPerBlock = 8;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * kWarpsPerBlock + warp;
  if (row >= rows) return;

  const int64_t input_offset = static_cast<int64_t>(row) * input_cols;
  const int64_t output_offset = static_cast<int64_t>(row) * output_cols;
  float square_sum = 0.0f;
  for (int col = lane; col < input_cols; col += 32) {
    const float value = __bfloat162float(input[input_offset + col]);
    square_sum += value * value;
  }
  const float inverse_rms =
      rsqrtf(warp_sum_all(square_sum) / static_cast<float>(input_cols) + eps);

  float maximum = 0.0f;
  for (int col = lane; col < input_cols; col += 32) {
    const float value = __bfloat162float(input[input_offset + col]) *
                        inverse_rms * __bfloat162float(weight[col]);
    maximum = fmaxf(maximum, fabsf(value));
  }
  const float scale = fmaxf(warp_max(maximum) / 448.0f, 1.0e-12f);
  if (lane == 0) scales[row] = scale;

  for (int col = lane; col < output_cols; col += 32) {
    float value = 0.0f;
    if (col < input_cols) {
      value = __bfloat162float(input[input_offset + col]) * inverse_rms *
              __bfloat162float(weight[col]) / scale;
      value = fminf(448.0f, fmaxf(-448.0f, value));
    }
    output[output_offset + col] = static_cast<__nv_fp8_e4m3>(value);
  }
}

__global__ void rms_norm_quantize_rows_bf16_e4m3_vec8_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    __nv_fp8_e4m3* output, float* scales, int rows, int input_cols,
    int output_cols, float eps) {
  constexpr int kRowsPerBlock = 8;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * kRowsPerBlock + warp;
  if (row >= rows) return;

  const int input_vectors = input_cols / 8;
  const int output_vectors = output_cols / 8;
  const int64_t input_offset = static_cast<int64_t>(row) * input_cols;
  const int64_t output_offset = static_cast<int64_t>(row) * output_cols;
  float square_sum = 0.0f;
  for (int vector = lane; vector < input_vectors; vector += 32) {
    const Bf16Pack8 values =
        *reinterpret_cast<const Bf16Pack8*>(input + input_offset + vector * 8);
#pragma unroll
    for (int item = 0; item < 8; ++item) {
      const float value = __bfloat162float(values.values[item]);
      square_sum += value * value;
    }
  }
  const float inverse_rms = rsqrtf(
      warp_sum_all(square_sum) / static_cast<float>(input_cols) + eps);

  float maximum = 0.0f;
  for (int vector = lane; vector < input_vectors; vector += 32) {
    const Bf16Pack8 values =
        *reinterpret_cast<const Bf16Pack8*>(input + input_offset + vector * 8);
    const Bf16Pack8 weights =
        *reinterpret_cast<const Bf16Pack8*>(weight + vector * 8);
#pragma unroll
    for (int item = 0; item < 8; ++item) {
      const float value = __bfloat162float(values.values[item]) * inverse_rms *
                          __bfloat162float(weights.values[item]);
      maximum = fmaxf(maximum, fabsf(value));
    }
  }
  const float scale = fmaxf(warp_max(maximum) / 448.0f, 1.0e-12f);
  const float inverse_scale = 1.0f / scale;
  if (lane == 0) scales[row] = scale;

  for (int vector = lane; vector < output_vectors; vector += 32) {
    Fp8Pack8 quantized{};
    if (vector < input_vectors) {
      const Bf16Pack8 values = *reinterpret_cast<const Bf16Pack8*>(
          input + input_offset + vector * 8);
      const Bf16Pack8 weights =
          *reinterpret_cast<const Bf16Pack8*>(weight + vector * 8);
#pragma unroll
      for (int item = 0; item < 8; ++item) {
        float value = __bfloat162float(values.values[item]) * inverse_rms *
                      __bfloat162float(weights.values[item]) * inverse_scale;
        value = fminf(448.0f, fmaxf(-448.0f, value));
        quantized.values[item] = static_cast<__nv_fp8_e4m3>(value);
      }
    }
    *reinterpret_cast<Fp8Pack8*>(output + output_offset + vector * 8) = quantized;
  }
}

// Fuses bias, SiLU gating, per-row amax, and E4M3 conversion. The input may
// have trailing physical padding after the logical gate/up columns, and the
// output can independently pad the down-projection K dimension.
__global__ void swiglu_quantize_rows_bf16_e4m3_kernel(
    const __nv_bfloat16* gate_up, const __nv_bfloat16* bias,
    __nv_fp8_e4m3* output, float* scales, int rows, int input_cols,
    int inner, int output_cols) {
  __shared__ float scratch[8];
  extern __shared__ float activated[];
  const int row = blockIdx.x;
  if (row >= rows) return;

  const int64_t input_offset = static_cast<int64_t>(row) * input_cols;
  const int64_t output_offset = static_cast<int64_t>(row) * output_cols;
  float maximum = 0.0f;
  for (int col = threadIdx.x; col < inner; col += blockDim.x) {
    float gate = __bfloat162float(gate_up[input_offset + col]);
    float up = __bfloat162float(gate_up[input_offset + inner + col]);
    if (bias != nullptr) {
      gate += __bfloat162float(bias[col]);
      up += __bfloat162float(bias[inner + col]);
    }
    const float value = (gate / (1.0f + expf(-gate))) * up;
    activated[col] = value;
    maximum = fmaxf(maximum, fabsf(value));
  }
  const float scale = fmaxf(block_max(maximum, scratch) / 448.0f, 1.0e-12f);
  if (threadIdx.x == 0) scales[row] = scale;

  for (int col = threadIdx.x; col < output_cols; col += blockDim.x) {
    float value = 0.0f;
    if (col < inner) {
      value = activated[col] / scale;
      value = fminf(448.0f, fmaxf(-448.0f, value));
    }
    output[output_offset + col] = static_cast<__nv_fp8_e4m3>(value);
  }
}

__global__ void swiglu_quantize_rows_bf16_e4m3_vec8_kernel(
    const __nv_bfloat16* gate_up, const __nv_bfloat16* bias,
    __nv_fp8_e4m3* output, float* scales, int rows, int input_cols,
    int inner, int output_cols) {
  __shared__ float scratch[8];
  extern __shared__ float activated[];
  const int row = blockIdx.x;
  if (row >= rows) return;

  const int inner_vectors = inner / 8;
  const int output_vectors = output_cols / 8;
  const int64_t input_offset = static_cast<int64_t>(row) * input_cols;
  const int64_t output_offset = static_cast<int64_t>(row) * output_cols;
  float maximum = 0.0f;
  for (int vector = threadIdx.x; vector < inner_vectors;
       vector += blockDim.x) {
    Bf16Pack8 gates = *reinterpret_cast<const Bf16Pack8*>(
        gate_up + input_offset + vector * 8);
    Bf16Pack8 ups = *reinterpret_cast<const Bf16Pack8*>(
        gate_up + input_offset + inner + vector * 8);
    Bf16Pack8 gate_bias{};
    Bf16Pack8 up_bias{};
    if (bias != nullptr) {
      gate_bias = *reinterpret_cast<const Bf16Pack8*>(bias + vector * 8);
      up_bias = *reinterpret_cast<const Bf16Pack8*>(bias + inner + vector * 8);
    }
#pragma unroll
    for (int item = 0; item < 8; ++item) {
      float gate = __bfloat162float(gates.values[item]);
      float up = __bfloat162float(ups.values[item]);
      if (bias != nullptr) {
        gate += __bfloat162float(gate_bias.values[item]);
        up += __bfloat162float(up_bias.values[item]);
      }
      const float value = (gate / (1.0f + __expf(-gate))) * up;
      activated[vector * 8 + item] = value;
      maximum = fmaxf(maximum, fabsf(value));
    }
  }
  const float scale = fmaxf(block_max(maximum, scratch) / 448.0f, 1.0e-12f);
  const float inverse_scale = 1.0f / scale;
  if (threadIdx.x == 0) scales[row] = scale;

  for (int vector = threadIdx.x; vector < output_vectors;
       vector += blockDim.x) {
    Fp8Pack8 quantized{};
    if (vector < inner_vectors) {
#pragma unroll
      for (int item = 0; item < 8; ++item) {
        float value = activated[vector * 8 + item] * inverse_scale;
        value = fminf(448.0f, fmaxf(-448.0f, value));
        quantized.values[item] = static_cast<__nv_fp8_e4m3>(value);
      }
    }
    *reinterpret_cast<Fp8Pack8*>(output + output_offset + vector * 8) = quantized;
  }
}

__global__ void swiglu_quantize_rows_bf16_e4m3_vec4_kernel(
    const __nv_bfloat16* gate_up, const __nv_bfloat16* bias,
    __nv_fp8_e4m3* output, float* scales, int rows, int input_cols,
    int inner, int output_cols) {
  __shared__ float scratch[8];
  extern __shared__ float activated[];
  const int row = blockIdx.x;
  if (row >= rows) return;

  const int inner_vectors = inner / 4;
  const int output_vectors = output_cols / 4;
  const int64_t input_offset = static_cast<int64_t>(row) * input_cols;
  const int64_t output_offset = static_cast<int64_t>(row) * output_cols;
  float maximum = 0.0f;
  for (int vector = threadIdx.x; vector < inner_vectors;
       vector += blockDim.x) {
    Bf16Pack4 gates = *reinterpret_cast<const Bf16Pack4*>(
        gate_up + input_offset + vector * 4);
    Bf16Pack4 ups = *reinterpret_cast<const Bf16Pack4*>(
        gate_up + input_offset + inner + vector * 4);
    Bf16Pack4 gate_bias{};
    Bf16Pack4 up_bias{};
    if (bias != nullptr) {
      gate_bias = *reinterpret_cast<const Bf16Pack4*>(bias + vector * 4);
      up_bias = *reinterpret_cast<const Bf16Pack4*>(bias + inner + vector * 4);
    }
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      float gate = __bfloat162float(gates.values[item]);
      float up = __bfloat162float(ups.values[item]);
      if (bias != nullptr) {
        gate += __bfloat162float(gate_bias.values[item]);
        up += __bfloat162float(up_bias.values[item]);
      }
      const float value = (gate / (1.0f + __expf(-gate))) * up;
      activated[vector * 4 + item] = value;
      maximum = fmaxf(maximum, fabsf(value));
    }
  }
  const float scale = fmaxf(block_max(maximum, scratch) / 448.0f, 1.0e-12f);
  const float inverse_scale = 1.0f / scale;
  if (threadIdx.x == 0) scales[row] = scale;

  for (int vector = threadIdx.x; vector < output_vectors;
       vector += blockDim.x) {
    Fp8Pack4 quantized{};
    if (vector < inner_vectors) {
#pragma unroll
      for (int item = 0; item < 4; ++item) {
        float value = activated[vector * 4 + item] * inverse_scale;
        value = fminf(448.0f, fmaxf(-448.0f, value));
        quantized.values[item] = static_cast<__nv_fp8_e4m3>(value);
      }
    }
    *reinterpret_cast<Fp8Pack4*>(output + output_offset + vector * 4) =
        quantized;
  }
}

__global__ void slice_columns_bf16_kernel(
    const __nv_bfloat16* input, __nv_bfloat16* output, int rows,
    int input_cols, int output_cols) {
  const int64_t count = static_cast<int64_t>(rows) * output_cols;
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    const int row = static_cast<int>(index / output_cols);
    const int col = static_cast<int>(index % output_cols);
    output[index] = input[static_cast<int64_t>(row) * input_cols + col];
  }
}

__global__ void cast_f16_bf16_kernel(
    const half* input, __nv_bfloat16* output, int64_t count) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    output[index] = __float2bfloat16(__half2float(input[index]));
  }
}

// Four values per thread with one half2 pair per load and one uint32 store.
// The scalar kernel above remains the fallback for unaligned buffers and the
// final 0..3 values.
__global__ void quantize_f16_e4m3_packed4_kernel(
    const half* input, __nv_fp8_e4m3* output, int64_t vector_count,
    float inverse_scale) {
  int64_t index =
      (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
  const int64_t stride =
      static_cast<int64_t>(blockDim.x) * gridDim.x * 4;
  const half2* input2 = reinterpret_cast<const half2*>(input);
  for (; index < vector_count; index += stride) {
    const half2 first = input2[index / 2];
    const half2 second = input2[index / 2 + 1];
    __nv_fp8_e4m3 values[4];
    values[0] = static_cast<__nv_fp8_e4m3>(fminf(
        448.0f, fmaxf(-448.0f, __half2float(first.x) * inverse_scale)));
    values[1] = static_cast<__nv_fp8_e4m3>(fminf(
        448.0f, fmaxf(-448.0f, __half2float(first.y) * inverse_scale)));
    values[2] = static_cast<__nv_fp8_e4m3>(fminf(
        448.0f, fmaxf(-448.0f, __half2float(second.x) * inverse_scale)));
    values[3] = static_cast<__nv_fp8_e4m3>(fminf(
        448.0f, fmaxf(-448.0f, __half2float(second.y) * inverse_scale)));
    reinterpret_cast<uint32_t*>(output)[index / 4] =
        *reinterpret_cast<const uint32_t*>(values);
  }
}

__global__ void dequantize_e4m3_f16_kernel(
    const __nv_fp8_e4m3* input, half* output, int64_t count, float scale) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    output[index] = __float2half_rn(static_cast<float>(input[index]) * scale);
  }
}


__global__ void quantize_rows_bf16_int8_kernel(
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
  const float scale = fmaxf(block_max(maximum, scratch) / 127.0f, 1.0e-12f);
  if (threadIdx.x == 0) scales[row] = scale;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    const float quantized = roundf(__bfloat162float(input[index]) / scale);
    output[index] = static_cast<int8_t>(fminf(127.0f, fmaxf(-128.0f, quantized)));
  }
}

// Produce adaptive LayerNorm's rounded BF16 output and the exact dynamic
// per-row INT8 representation consumed by W8A8 GEMMs in one kernel.
__global__ void adaptive_layer_norm_quantize_rows_bf16_int8_kernel(
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
  const float mean = block_sum(sum, scratch) / cols;
  float variance_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float centered = __bfloat162float(input[base + col]) - mean;
    variance_sum += centered * centered;
  }
  const float inverse_std = rsqrtf(block_sum(variance_sum, scratch) / cols + eps);
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
  const float row_scale = fmaxf(block_max(maximum, scratch) / 127.0f, 1.0e-12f);
  if (threadIdx.x == 0) scales[row] = row_scale;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float value = roundf(__bfloat162float(output[base + col]) / row_scale);
    quantized[base + col] =
        static_cast<int8_t>(fminf(127.0f, fmaxf(-128.0f, value)));
  }
}

__global__ void silu_mul_quantize_rows_bf16_int8_kernel(
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
  const float scale = fmaxf(block_max(maximum, scratch) / 127.0f, 1.0e-12f);
  if (threadIdx.x == 0) scales[row] = scale;
  __syncthreads();
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float quantized = roundf(__bfloat162float(rounded[col]) / scale);
    output[base + col] =
        static_cast<int8_t>(fminf(127.0f, fmaxf(-128.0f, quantized)));
  }
}

__global__ void dequantize_int32_bf16_kernel(
    const int32_t* accumulators, const float* row_scales,
    const float* column_scales, __nv_bfloat16* output,
    int rows, int cols) {
  const int row = blockIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows && col < cols) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    output[index] = __float2bfloat16(
        static_cast<float>(accumulators[index]) * row_scales[row] *
        column_scales[col]);
  }
}
