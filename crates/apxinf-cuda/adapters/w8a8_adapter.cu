// Copyright 2026 apxinf contributors.
// Stable C ABI and CUDA launch policy for dynamic W8A8 operators.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace {
#include "../kernels/custom/math.cuh"
#include "../kernels/custom/reduction.cuh"
#include "../kernels/custom/quantization.cuh"

struct alignas(8) W8A8Bf16x4 {
  __nv_bfloat162 low;
  __nv_bfloat162 high;
};

__global__ void quantize_rows_bf16_int8_vec4_kernel(
    const __nv_bfloat16* input, int8_t* output, float* scales,
    int rows, int cols) {
  extern __shared__ W8A8Bf16x4 rounded_vec4[];
  __shared__ float scratch[16];
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int quads = cols / 4;
  const int64_t base = static_cast<int64_t>(row) * quads;
  const auto* input4 = reinterpret_cast<const W8A8Bf16x4*>(input);
  float maximum = 0.0f;
  for (int quad = threadIdx.x; quad < quads; quad += blockDim.x) {
    const W8A8Bf16x4 value = input4[base + quad];
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
    const W8A8Bf16x4 value = rounded_vec4[quad];
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
  extern __shared__ W8A8Bf16x4 rounded_vec4[];
  __shared__ float scratch[16];
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int quads = cols / 4;
  const int64_t base = static_cast<int64_t>(row) * quads;
  const auto* input4 = reinterpret_cast<const W8A8Bf16x4*>(input);
  const auto* bias4 = reinterpret_cast<const W8A8Bf16x4*>(bias);
  float maximum = 0.0f;
  for (int quad = threadIdx.x; quad < quads; quad += blockDim.x) {
    const W8A8Bf16x4 x = input4[base + quad];
    const W8A8Bf16x4 b = bias4[quad];
    const float values[4] = {
        gelu_tanh(__bfloat162float(x.low.x) + __bfloat162float(b.low.x)),
        gelu_tanh(__bfloat162float(x.low.y) + __bfloat162float(b.low.y)),
        gelu_tanh(__bfloat162float(x.high.x) + __bfloat162float(b.high.x)),
        gelu_tanh(__bfloat162float(x.high.y) + __bfloat162float(b.high.y)),
    };
    const W8A8Bf16x4 activated{
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
    const W8A8Bf16x4 value = rounded_vec4[quad];
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

// Preserve the scalar SiLU-mul contract lane by lane while using aligned
// 8-byte loads and packed INT8 stores. SiLU is rounded to BF16 before the
// multiply, and the product is rounded again before absmax and quantization.
__global__ void silu_mul_quantize_rows_bf16_int8_vec4_kernel(
    const __nv_bfloat16* gate, const __nv_bfloat16* up,
    int8_t* output, float* scales, int rows, int cols) {
  extern __shared__ W8A8Bf16x4 rounded_vec4[];
  __shared__ float scratch[16];
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int quads = cols / 4;
  const int64_t base = static_cast<int64_t>(row) * quads;
  const auto* gate4 = reinterpret_cast<const W8A8Bf16x4*>(gate);
  const auto* up4 = reinterpret_cast<const W8A8Bf16x4*>(up);
  float maximum = 0.0f;
  for (int quad = threadIdx.x; quad < quads; quad += blockDim.x) {
    const W8A8Bf16x4 g = gate4[base + quad];
    const W8A8Bf16x4 u = up4[base + quad];
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
    rounded_vec4[quad] = W8A8Bf16x4{
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
    const W8A8Bf16x4 value = rounded_vec4[quad];
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
}  // namespace

extern "C" cudaError_t apxinf_static_quantize_rows_bf16_int8(
    const void* input, void* output, void* scales,
    int rows, int cols, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || scales == nullptr ||
      rows <= 0 || cols <= 0) {
    return cudaErrorInvalidValue;
  }
  quantize_rows_bf16_int8_kernel<<<rows, kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<int8_t*>(output), static_cast<float*>(scales), rows, cols);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_quantize_rows_bf16_int8_packed4(
    const void* input, void* output, void* scales,
    int rows, int cols, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || scales == nullptr ||
      rows <= 0 || cols <= 0 || cols % 4 != 0) {
    return cudaErrorInvalidValue;
  }
  quantize_rows_bf16_int8_vec4_kernel<<<
      rows, 512, static_cast<size_t>(cols) * sizeof(__nv_bfloat16), stream>>>(
      static_cast<const __nv_bfloat16*>(input), static_cast<int8_t*>(output),
      static_cast<float*>(scales), rows, cols);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_bias_gelu_quantize_rows_bf16_int8(
    const void* input, const void* bias, void* output, void* scales,
    int rows, int cols, cudaStream_t stream) {
  if (input == nullptr || bias == nullptr || output == nullptr ||
      scales == nullptr || rows <= 0 || cols <= 0) {
    return cudaErrorInvalidValue;
  }
  if (cols % 4 == 0) {
    bias_gelu_quantize_rows_bf16_int8_vec4_kernel<<<
        rows, 512, static_cast<size_t>(cols) * sizeof(__nv_bfloat16), stream>>>(
        static_cast<const __nv_bfloat16*>(input),
        static_cast<const __nv_bfloat16*>(bias), static_cast<int8_t*>(output),
        static_cast<float*>(scales), rows, cols);
  } else {
    bias_gelu_quantize_rows_bf16_int8_kernel<<<
        rows, kThreads, static_cast<size_t>(cols) * sizeof(__nv_bfloat16), stream>>>(
        static_cast<const __nv_bfloat16*>(input),
        static_cast<const __nv_bfloat16*>(bias), static_cast<int8_t*>(output),
        static_cast<float*>(scales), rows, cols);
  }
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_adaptive_layer_norm_quantize_rows_bf16_int8(
    const void* input, const void* modulation, void* output,
    void* quantized, void* scales, int rows, int cols, float eps,
    cudaStream_t stream) {
  if (input == nullptr || modulation == nullptr || output == nullptr ||
      quantized == nullptr || scales == nullptr || rows <= 0 || cols <= 0 ||
      !(eps > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  adaptive_layer_norm_quantize_rows_bf16_int8_kernel<<<
      rows, kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(modulation),
      static_cast<__nv_bfloat16*>(output), static_cast<int8_t*>(quantized),
      static_cast<float*>(scales), rows, cols, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_layer_norm_quantize_rows_bf16_int8(
    const void* input, const void* weight, const void* bias, void* output,
    void* quantized, void* scales, int rows, int cols, float eps,
    cudaStream_t stream) {
  if (input == nullptr || weight == nullptr || bias == nullptr ||
      output == nullptr || quantized == nullptr || scales == nullptr ||
      rows <= 0 || cols <= 0 || !(eps > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  layer_norm_quantize_rows_bf16_int8_kernel<<<
      rows, kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<const __nv_bfloat16*>(bias),
      static_cast<__nv_bfloat16*>(output), static_cast<int8_t*>(quantized),
      static_cast<float*>(scales), rows, cols, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_silu_mul_quantize_rows_bf16_int8(
    const void* gate, const void* up, void* output, void* scales,
    int rows, int cols, cudaStream_t stream) {
  if (gate == nullptr || up == nullptr || output == nullptr || scales == nullptr ||
      rows <= 0 || cols <= 0) {
    return cudaErrorInvalidValue;
  }
  silu_mul_quantize_rows_bf16_int8_kernel<<<
      rows, kThreads, static_cast<size_t>(cols) * sizeof(__nv_bfloat16), stream>>>(
      static_cast<const __nv_bfloat16*>(gate),
      static_cast<const __nv_bfloat16*>(up), static_cast<int8_t*>(output),
      static_cast<float*>(scales), rows, cols);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_silu_mul_quantize_rows_bf16_int8_packed4(
    const void* gate, const void* up, void* output, void* scales,
    int rows, int cols, cudaStream_t stream) {
  if (gate == nullptr || up == nullptr || output == nullptr || scales == nullptr ||
      rows <= 0 || cols <= 0 || cols % 4 != 0 ||
      (reinterpret_cast<uintptr_t>(gate) & 7) != 0 ||
      (reinterpret_cast<uintptr_t>(up) & 7) != 0 ||
      (reinterpret_cast<uintptr_t>(output) & 3) != 0) {
    return cudaErrorInvalidValue;
  }
  silu_mul_quantize_rows_bf16_int8_vec4_kernel<<<
      rows, 512, static_cast<size_t>(cols) * sizeof(__nv_bfloat16), stream>>>(
      static_cast<const __nv_bfloat16*>(gate),
      static_cast<const __nv_bfloat16*>(up), static_cast<int8_t*>(output),
      static_cast<float*>(scales), rows, cols);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_dequantize_int32_bf16(
    const void* accumulators, const void* row_scales,
    const void* column_scales, void* output,
    int rows, int cols, cudaStream_t stream) {
  if (accumulators == nullptr || row_scales == nullptr ||
      column_scales == nullptr || output == nullptr || rows <= 0 || cols <= 0) {
    return cudaErrorInvalidValue;
  }
  const dim3 grid((cols + kThreads - 1) / kThreads, rows);
  dequantize_int32_bf16_kernel<<<grid, kThreads, 0, stream>>>(
      static_cast<const int32_t*>(accumulators),
      static_cast<const float*>(row_scales),
      static_cast<const float*>(column_scales),
      static_cast<__nv_bfloat16*>(output), rows, cols);
  return cudaGetLastError();
}
