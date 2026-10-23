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

// Exact-shape opt-in for the DiT attention-output boundary. Unlike the
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


}  // namespace

// ── Fused residual launchers ─────────────────────────────────────────────

extern "C" cudaError_t apxinf_transformer_bf16_bias_then_residual_bf16(
    const void* projection, const void* bias, const void* residual,
    void* output, int rows, int cols, cudaStream_t stream) {
  const int64_t count = static_cast<int64_t>(rows) * cols;
  const int blocks = static_cast<int>((count + kBlockSize - 1) / kBlockSize);
  bias_then_residual_bf16_kernel<<<blocks, kBlockSize, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(projection),
      static_cast<const __nv_bfloat16*>(bias),
      static_cast<const __nv_bfloat16*>(residual),
      static_cast<__nv_bfloat16*>(output), count, cols);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_transformer_bf16_bias_residual_bf16_packed4(
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
      static_cast<int>((packed_count + kBlockSize - 1) / kBlockSize);
  bias_residual_bf16_packed4_kernel<<<blocks, kBlockSize, 0, stream>>>(
      static_cast<const Bf16x4*>(projection), static_cast<const Bf16x4*>(bias),
      static_cast<const Bf16x4*>(residual), static_cast<Bf16x4*>(output),
      packed_count, cols / 4);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_transformer_bf16_bias_then_residual_bf16_packed4(
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
      static_cast<int>((packed_count + kBlockSize - 1) / kBlockSize);
  bias_then_residual_bf16_packed4_kernel<<<blocks, kBlockSize, 0, stream>>>(
      static_cast<const Bf16x4*>(projection), static_cast<const Bf16x4*>(bias),
      static_cast<const Bf16x4*>(residual), static_cast<Bf16x4*>(output),
      packed_count, cols / 4);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_transformer_bf16_bias_residual_layer_norm_bf16_cached_1024(
    const void* projection, const void* projection_bias, const void* residual,
    const void* norm_weight, const void* norm_bias, void* hidden,
    void* normalized, int rows, int cols, float eps, cudaStream_t stream) {
  if (projection == nullptr || residual == nullptr || norm_weight == nullptr ||
      norm_bias == nullptr || hidden == nullptr || normalized == nullptr ||
      rows <= 0 || cols != 1024)
    return cudaErrorInvalidValue;
  bias_residual_layer_norm_bf16_cached_1024_kernel<<<rows, kBlockSize, 0,
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

extern "C" cudaError_t apxinf_transformer_bf16_bias_then_residual_layer_norm_bf16_cached_1536(
    const void* projection, const void* projection_bias, const void* residual,
    const void* norm_weight, const void* norm_bias, void* hidden,
    void* normalized, int rows, int cols, float eps, cudaStream_t stream) {
  if (projection == nullptr || projection_bias == nullptr ||
      residual == nullptr || norm_weight == nullptr || norm_bias == nullptr ||
      hidden == nullptr || normalized == nullptr || rows <= 0 || cols != 1536)
    return cudaErrorInvalidValue;
  bias_then_residual_layer_norm_bf16_cached_1536_kernel<<<rows, kBlockSize,
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
apxinf_transformer_bf16_bias_then_residual_adaptive_layer_norm_bf16_cached_1536(
    const void* projection, const void* projection_bias, const void* residual,
    const void* modulation, void* hidden, void* normalized, int rows, int cols,
    float eps, cudaStream_t stream) {
  if (projection == nullptr || projection_bias == nullptr ||
      residual == nullptr || modulation == nullptr || hidden == nullptr ||
      normalized == nullptr || rows <= 0 || cols != 1536)
    return cudaErrorInvalidValue;
  bias_then_residual_adaptive_layer_norm_bf16_cached_1536_kernel
      <<<rows, kBlockSize, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(projection),
          static_cast<const __nv_bfloat16*>(projection_bias),
          static_cast<const __nv_bfloat16*>(residual),
          static_cast<const __nv_bfloat16*>(modulation),
          static_cast<__nv_bfloat16*>(hidden),
          static_cast<__nv_bfloat16*>(normalized), rows, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_transformer_bf16_bias_residual_layer_norm_quant_bf16_e4m3(
    const void* projection, const void* projection_bias, const void* residual,
    const void* norm_weight, const void* norm_bias, void* hidden,
    void* normalized, int rows, int cols, float eps, float scale,
    cudaStream_t stream) {
  if (projection == nullptr || residual == nullptr || norm_weight == nullptr ||
      norm_bias == nullptr || hidden == nullptr || normalized == nullptr ||
      rows <= 0 || cols <= 0 || !std::isfinite(scale) || scale <= 0.0f)
    return cudaErrorInvalidValue;
  bias_residual_layer_norm_quant_bf16_e4m3_kernel<<<rows, kBlockSize, 0,
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
