// Copyright 2026 ApxInf contributors.
// Fixed BF16 implementation for this operator family; ported without numerical changes.
#include "../../include/apxinf_cuda/transformer_bf16.h"
#include "../../kernels/custom/reduction.cuh"
#ifdef APXINF_GEMM_CUTLASS_SM87_W8A8
#include "../../kernels/cutlass/ops/gemm/gemm_i8_bf16_sm80.h"
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

__global__ void layer_norm_quantize_rows_bf16_int8_kernel(
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

extern "C" cudaError_t apxinf_transformer_bf16_quantize_bf16_e4m3_packed8(
    const void* input, void* output, int64_t count, float scale,
    cudaStream_t stream) {
  if (input == nullptr || output == nullptr || count <= 0 || !(scale > 0.0f))
    return cudaErrorInvalidValue;
  constexpr int threads = 256;
  const bool aligned =
      (reinterpret_cast<uintptr_t>(input) % alignof(Bf16Pack8) == 0) &&
      (reinterpret_cast<uintptr_t>(output) % alignof(Fp8Pack8) == 0);
  if (!aligned || count < 8) {
    int blocks = static_cast<int>((count + threads - 1) / threads);
    blocks = blocks > 4096 ? 4096 : blocks;
    quantize_bf16_e4m3_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input),
        static_cast<__nv_fp8_e4m3*>(output), count, 1.0f / scale);
    return cudaGetLastError();
  }
  const int64_t vector_count = count / 8;
  int blocks = static_cast<int>((vector_count + threads - 1) / threads);
  blocks = blocks > 4096 ? 4096 : blocks;
  quantize_bf16_e4m3_packed8_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const Bf16Pack8*>(input),
      static_cast<Fp8Pack8*>(output), vector_count, 1.0f / scale);
  const int64_t tail = count - vector_count * 8;
  if (tail != 0) {
    quantize_bf16_e4m3_kernel<<<1, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input) + vector_count * 8,
        static_cast<__nv_fp8_e4m3*>(output) + vector_count * 8, tail,
        1.0f / scale);
  }
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_transformer_bf16_concat_rows_quantize_bf16_e4m3(
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
  concat_rows_quantize_bf16_e4m3_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(first),
      static_cast<const __nv_bfloat16*>(second),
      static_cast<__nv_fp8_e4m3*>(output), first_count, total_count,
      1.0f / scale);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_transformer_bf16_dequantize_int32_bf16(
    const void* accumulators, const void* row_scales,
    const void* column_scales, void* output, int rows, int cols,
    cudaStream_t stream) {
  if (accumulators == nullptr || row_scales == nullptr ||
      column_scales == nullptr || output == nullptr || rows <= 0 || cols <= 0)
    return cudaErrorInvalidValue;
  const dim3 grid((cols + kBlockSize - 1) / kBlockSize, rows);
  dequantize_int32_bf16_kernel<<<grid, kBlockSize, 0, stream>>>(
      static_cast<const int32_t*>(accumulators),
      static_cast<const float*>(row_scales),
      static_cast<const float*>(column_scales),
      static_cast<__nv_bfloat16*>(output), rows, cols);
  return cudaGetLastError();
}

// ── W8A8 (INT8) quantization launchers ───────────────────────────────────
// Copied from the legacy w8a8_adapter.cu. These produce the dynamically
// row-quantized INT8 activations the W8A8 GEMM consumes.

extern "C" cudaError_t apxinf_transformer_bf16_quantize_rows_bf16_int8(
    const void* input, void* output, void* scales, int rows, int cols,
    cudaStream_t stream) {
  if (input == nullptr || output == nullptr || scales == nullptr ||
      rows <= 0 || cols <= 0)
    return cudaErrorInvalidValue;
  quantize_rows_bf16_int8_kernel<<<rows, kBlockSize, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input), static_cast<int8_t*>(output),
      static_cast<float*>(scales), rows, cols);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_transformer_bf16_quantize_rows_bf16_int8_packed4(
    const void* input, void* output, void* scales, int rows, int cols,
    cudaStream_t stream) {
  if (input == nullptr || output == nullptr || scales == nullptr ||
      rows <= 0 || cols <= 0 || cols % 4 != 0)
    return cudaErrorInvalidValue;
  quantize_rows_bf16_int8_vec4_kernel<<<
      rows, 512, static_cast<size_t>(cols) * sizeof(__nv_bfloat16), stream>>>(
      static_cast<const __nv_bfloat16*>(input), static_cast<int8_t*>(output),
      static_cast<float*>(scales), rows, cols);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_transformer_bf16_bias_gelu_quantize_rows_bf16_int8(
    const void* input, const void* bias, void* output, void* scales, int rows,
    int cols, cudaStream_t stream) {
  if (input == nullptr || bias == nullptr || output == nullptr ||
      scales == nullptr || rows <= 0 || cols <= 0)
    return cudaErrorInvalidValue;
  if (cols % 4 == 0) {
    bias_gelu_quantize_rows_bf16_int8_vec4_kernel<<<
        rows, 512, static_cast<size_t>(cols) * sizeof(__nv_bfloat16), stream>>>(
        static_cast<const __nv_bfloat16*>(input),
        static_cast<const __nv_bfloat16*>(bias), static_cast<int8_t*>(output),
        static_cast<float*>(scales), rows, cols);
  } else {
    bias_gelu_quantize_rows_bf16_int8_kernel<<<
        rows, kBlockSize,
        static_cast<size_t>(cols) * sizeof(__nv_bfloat16), stream>>>(
        static_cast<const __nv_bfloat16*>(input),
        static_cast<const __nv_bfloat16*>(bias), static_cast<int8_t*>(output),
        static_cast<float*>(scales), rows, cols);
  }
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_transformer_bf16_adaptive_layer_norm_quantize_rows_bf16_int8(
    const void* input, const void* modulation, void* output, void* quantized,
    void* scales, int rows, int cols, float eps, cudaStream_t stream) {
  if (input == nullptr || modulation == nullptr || output == nullptr ||
      quantized == nullptr || scales == nullptr || rows <= 0 || cols <= 0 ||
      !(eps > 0.0f))
    return cudaErrorInvalidValue;
  adaptive_layer_norm_quantize_rows_bf16_int8_kernel<<<
      rows, kBlockSize, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(modulation),
      static_cast<__nv_bfloat16*>(output), static_cast<int8_t*>(quantized),
      static_cast<float*>(scales), rows, cols, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_transformer_bf16_layer_norm_quantize_rows_bf16_int8(
    const void* input, const void* weight, const void* bias, void* output,
    void* quantized, void* scales, int rows, int cols, float eps,
    cudaStream_t stream) {
  if (input == nullptr || weight == nullptr || bias == nullptr ||
      output == nullptr || quantized == nullptr || scales == nullptr ||
      rows <= 0 || cols <= 0 || !(eps > 0.0f))
    return cudaErrorInvalidValue;
  layer_norm_quantize_rows_bf16_int8_kernel<<<rows, kBlockSize, 0,
                                                    stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<const __nv_bfloat16*>(bias),
      static_cast<__nv_bfloat16*>(output), static_cast<int8_t*>(quantized),
      static_cast<float*>(scales), rows, cols, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_transformer_bf16_silu_mul_quantize_rows_bf16_int8(
    const void* gate, const void* up, void* output, void* scales, int rows,
    int cols, cudaStream_t stream) {
  if (gate == nullptr || up == nullptr || output == nullptr ||
      scales == nullptr || rows <= 0 || cols <= 0)
    return cudaErrorInvalidValue;
  silu_mul_quantize_rows_bf16_int8_kernel<<<
      rows, kBlockSize,
      static_cast<size_t>(cols) * sizeof(__nv_bfloat16), stream>>>(
      static_cast<const __nv_bfloat16*>(gate),
      static_cast<const __nv_bfloat16*>(up), static_cast<int8_t*>(output),
      static_cast<float*>(scales), rows, cols);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_transformer_bf16_silu_mul_quantize_rows_bf16_int8_packed4(
    const void* gate, const void* up, void* output, void* scales, int rows,
    int cols, cudaStream_t stream) {
  if (gate == nullptr || up == nullptr || output == nullptr ||
      scales == nullptr || rows <= 0 || cols <= 0 || cols % 4 != 0 ||
      (reinterpret_cast<uintptr_t>(gate) & 7) != 0 ||
      (reinterpret_cast<uintptr_t>(up) & 7) != 0 ||
      (reinterpret_cast<uintptr_t>(output) & 3) != 0)
    return cudaErrorInvalidValue;
  silu_mul_quantize_rows_bf16_int8_vec4_kernel<<<
      rows, 512, static_cast<size_t>(cols) * sizeof(__nv_bfloat16), stream>>>(
      static_cast<const __nv_bfloat16*>(gate),
      static_cast<const __nv_bfloat16*>(up), static_cast<int8_t*>(output),
      static_cast<float*>(scales), rows, cols);
  return cudaGetLastError();
}
