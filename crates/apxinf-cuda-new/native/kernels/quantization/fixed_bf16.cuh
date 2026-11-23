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

// ── Quantization kernels (from legacy quantization.cuh / concat_quantization.cuh)

struct alignas(16) Bf16Pack8 {
  __nv_bfloat16 values[8];
};
struct alignas(8) Fp8Pack8 {
  __nv_fp8_e4m3 values[8];
};

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

__global__ void quantize_bf16_e4m3_packed8_kernel(
    const Bf16Pack8* input, Fp8Pack8* output, int64_t vector_count,
    float inverse_scale) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < vector_count; index += stride) {
    const Bf16Pack8 values = input[index];
    Fp8Pack8 quantized;
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
__global__ void concat_rows_quantize_bf16_e4m3_kernel(
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
__global__ void dequantize_int32_bf16_kernel(
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
__global__ void quantize_rows_bf16_int8_vec4_kernel(
    const __nv_bfloat16* input, int8_t* output, float* scales,
    int rows, int cols) {
  extern __shared__ W8Bf16x4 rounded_vec4[];
  __shared__ float scratch[16];
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int quads = cols / 4;
  const int64_t base = static_cast<int64_t>(row) * quads;
  const auto* input4 = reinterpret_cast<const W8Bf16x4*>(input);
  float maximum = 0.0f;
  for (int quad = threadIdx.x; quad < quads; quad += blockDim.x) {
    const W8Bf16x4 value = input4[base + quad];
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
    const W8Bf16x4 value = rounded_vec4[quad];
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

__global__ void bias_gelu_quantize_rows_bf16_int8_vec4_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* bias,
    int8_t* output, float* scales, int rows, int cols) {
  extern __shared__ W8Bf16x4 rounded_vec4[];
  __shared__ float scratch[16];
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int quads = cols / 4;
  const int64_t base = static_cast<int64_t>(row) * quads;
  const auto* input4 = reinterpret_cast<const W8Bf16x4*>(input);
  const auto* bias4 = reinterpret_cast<const W8Bf16x4*>(bias);
  float maximum = 0.0f;
  for (int quad = threadIdx.x; quad < quads; quad += blockDim.x) {
    const W8Bf16x4 x = input4[base + quad];
    const W8Bf16x4 b = bias4[quad];
    const float values[4] = {
        gelu_tanh(__bfloat162float(x.low.x) + __bfloat162float(b.low.x)),
        gelu_tanh(__bfloat162float(x.low.y) + __bfloat162float(b.low.y)),
        gelu_tanh(__bfloat162float(x.high.x) + __bfloat162float(b.high.x)),
        gelu_tanh(__bfloat162float(x.high.y) + __bfloat162float(b.high.y)),
    };
    const W8Bf16x4 activated{
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
    const W8Bf16x4 value = rounded_vec4[quad];
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

__global__ void silu_mul_quantize_rows_bf16_int8_vec4_kernel(
    const __nv_bfloat16* gate, const __nv_bfloat16* up,
    int8_t* output, float* scales, int rows, int cols) {
  extern __shared__ W8Bf16x4 rounded_vec4[];
  __shared__ float scratch[16];
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int quads = cols / 4;
  const int64_t base = static_cast<int64_t>(row) * quads;
  const auto* gate4 = reinterpret_cast<const W8Bf16x4*>(gate);
  const auto* up4 = reinterpret_cast<const W8Bf16x4*>(up);
  float maximum = 0.0f;
  for (int quad = threadIdx.x; quad < quads; quad += blockDim.x) {
    const W8Bf16x4 g = gate4[base + quad];
    const W8Bf16x4 u = up4[base + quad];
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
    rounded_vec4[quad] = W8Bf16x4{
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
    const W8Bf16x4 value = rounded_vec4[quad];
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

__global__ void bias_gelu_quantize_rows_bf16_int8_kernel(
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
  const float scale =
      fmaxf(block_max_parallel_unsafe(maximum, scratch) / 127.0f, 1.0e-12f);
  if (threadIdx.x == 0) scales[row] = scale;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    const float quantized = roundf(__bfloat162float(input[index]) / scale);
    output[index] = static_cast<int8_t>(fminf(127.0f, fmaxf(-128.0f, quantized)));
  }
}

}  // namespace
