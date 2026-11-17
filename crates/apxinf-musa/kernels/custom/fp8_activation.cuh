#pragma once

#include "math.cuh"

__device__ __forceinline__ __mt_fp8_e4m3 fp8_activation_value(float value,
                                                               float inverse_scale) {
  return __mt_fp8_e4m3(
      fminf(448.0f, fmaxf(-448.0f, value * inverse_scale)));
}

union __align__(16) Fp8ActivationInputPack {
  uint4 bits;
  __mt_bfloat16 values[8];
};

union __align__(8) Fp8ActivationOutputPack {
  uint64_t bits;
  __mt_fp8_e4m3 values[8];
};

template <bool HAS_BIAS>
__global__ void bias_gelu_quant_bf16_e4m3_vector8_kernel(
    const __mt_bfloat16* input, const __mt_bfloat16* bias,
    __mt_fp8_e4m3* output, int rows, int cols, float inverse_scale) {
  const int width = cols / 8;
  const int stride = blockDim.x * gridDim.x;
  for (int row = blockIdx.y; row < rows; row += gridDim.y) {
    const int64_t base = static_cast<int64_t>(row) * cols;
    for (int lane = blockIdx.x * blockDim.x + threadIdx.x; lane < width;
         lane += stride) {
      const int64_t column = static_cast<int64_t>(lane) * 8;
      Fp8ActivationInputPack input_raw, bias_raw;
      Fp8ActivationOutputPack output_raw;
      input_raw.bits = *reinterpret_cast<const uint4*>(input + base + column);
      if (HAS_BIAS) {
        bias_raw.bits = *reinterpret_cast<const uint4*>(bias + column);
      }
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        float value = __bfloat162float(input_raw.values[i]);
        if (HAS_BIAS) value += __bfloat162float(bias_raw.values[i]);
        output_raw.values[i] = fp8_activation_value(gelu_tanh(value), inverse_scale);
      }
      *reinterpret_cast<uint64_t*>(output + base + column) = output_raw.bits;
    }
  }
}

__global__ void silu_mul_quant_bf16_e4m3_vector8_kernel(
    const __mt_bfloat16* gate, const __mt_bfloat16* up,
    __mt_fp8_e4m3* output, int64_t count, float inverse_scale) {
  const int64_t start =
      (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 8;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x * 8;
  for (int64_t index = start; index < count; index += stride) {
    Fp8ActivationInputPack gate_raw, up_raw;
    Fp8ActivationOutputPack output_raw;
    gate_raw.bits = *reinterpret_cast<const uint4*>(gate + index);
    up_raw.bits = *reinterpret_cast<const uint4*>(up + index);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float gate_value = __bfloat162float(gate_raw.values[i]);
      const float value = (gate_value / (1.0f + expf(-gate_value))) *
                          __bfloat162float(up_raw.values[i]);
      output_raw.values[i] = fp8_activation_value(value, inverse_scale);
    }
    *reinterpret_cast<uint64_t*>(output + index) = output_raw.bits;
  }
}

__global__ void bias_gelu_quant_bf16_e4m3_kernel(
    const __mt_bfloat16* input, const __mt_bfloat16* bias,
    __mt_fp8_e4m3* output, size_t count, int cols, float inverse_scale) {
  const size_t start = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) * 8;
  const size_t stride = size_t(gridDim.x) * blockDim.x * 8;
  for (size_t offset = start; offset < count; offset += stride) {
    const size_t limit = min(size_t(8), count - offset);
    for (size_t lane = 0; lane < limit; ++lane) {
      const size_t index = offset + lane;
      const float value = gelu_tanh(float(input[index]) +
                                    float(bias[index % size_t(cols)]));
      output[index] = fp8_activation_value(value, inverse_scale);
    }
  }
}

__global__ void silu_mul_quant_bf16_e4m3_kernel(
    const __mt_bfloat16* gate, const __mt_bfloat16* up,
    __mt_fp8_e4m3* output, size_t count, float inverse_scale) {
  const size_t start = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) * 8;
  const size_t stride = size_t(gridDim.x) * blockDim.x * 8;
  for (size_t offset = start; offset < count; offset += stride) {
    const size_t limit = min(size_t(8), count - offset);
    for (size_t lane = 0; lane < limit; ++lane) {
      const size_t index = offset + lane;
      const float gate_value = float(gate[index]);
      const float value = (gate_value / (1.0f + expf(-gate_value))) *
                          float(up[index]);
      output[index] = fp8_activation_value(value, inverse_scale);
    }
  }
}
