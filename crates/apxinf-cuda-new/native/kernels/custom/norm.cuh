#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace apxinf::norm::kernels {

template <class T>
__device__ float to_float(T value);

template <>
__device__ inline float to_float(__half value) {
  return __half2float(value);
}

template <>
__device__ inline float to_float(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <class T>
__device__ T from_float(float value);

template <>
__device__ inline __half from_float(float value) {
  return __float2half(value);
}

template <>
__device__ inline __nv_bfloat16 from_float(float value) {
  return __float2bfloat16(value);
}

__device__ __forceinline__ float warp_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1)
    value += __shfl_down_sync(0xffffffff, value, offset);
  return value;
}

// One block per row; `scratch` must hold blockDim.x / 32 floats. The final
// barrier publishes scratch[0], but does not protect the reads that follow it.
// Callers must consume the result and synchronize all block threads before
// reusing the same scratch storage.
__device__ __forceinline__ float block_sum_parallel_unsafe(float value,
                                                            float* scratch) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = blockDim.x >> 5;
  value = warp_sum(value);
  if (lane == 0) scratch[warp] = value;
  __syncthreads();
  if (warp == 0) {
    value = lane < warps ? scratch[lane] : 0.0f;
    value = warp_sum(value);
    if (lane == 0) scratch[0] = value;
  }
  __syncthreads();
  return scratch[0];
}

// Every kernel below takes (rows, cols) in that order.  The legacy backend
// carries two same-named overloads that differ only by `int` versus
// `uint32_t`, with the row and column arguments swapped between them; the port
// deliberately keeps a single signature so that mistake cannot recur.

template <class T>
__global__ void rms_norm(const T* input, const T* weight, T* output, int rows,
                         int cols, float eps) {
  __shared__ float scratch[32];
  const int row = blockIdx.x;
  if (row >= rows) return;
  float square_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float value = to_float(input[static_cast<int64_t>(row) * cols + col]);
    square_sum += value * value;
  }
  const float inverse_rms =
      rsqrtf(block_sum_parallel_unsafe(square_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    output[index] =
        from_float<T>(to_float(input[index]) * inverse_rms * to_float(weight[col]));
  }
}

template <class T>
__global__ void layer_norm(const T* input, const T* weight, const T* bias,
                           T* output, int rows, int cols, float eps) {
  __shared__ float scratch[32];
  const int row = blockIdx.x;
  if (row >= rows) return;
  float sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x)
    sum += to_float(input[static_cast<int64_t>(row) * cols + col]);
  const float mean = block_sum_parallel_unsafe(sum, scratch) / cols;
  float variance_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float centered =
        to_float(input[static_cast<int64_t>(row) * cols + col]) - mean;
    variance_sum += centered * centered;
  }
  // Finish reading the mean before the variance reduction overwrites scratch.
  __syncthreads();
  const float inverse_std = rsqrtf(
      block_sum_parallel_unsafe(variance_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    output[index] = from_float<T>(
        (to_float(input[index]) - mean) * inverse_std * to_float(weight[col]) +
        to_float(bias[col]));
  }
}

// norm_style is [2 * cols]: scale in [0, cols), shift in [cols, 2 * cols).
template <class T>
__global__ void ada_rms_norm(const T* input, const T* norm_style, T* output,
                             int rows, int cols, float eps) {
  __shared__ float scratch[32];
  const int row = blockIdx.x;
  if (row >= rows) return;
  float square_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float value = to_float(input[static_cast<int64_t>(row) * cols + col]);
    square_sum += value * value;
  }
  const float inverse_rms =
      rsqrtf(block_sum_parallel_unsafe(square_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    const float normalized = to_float(input[index]) * inverse_rms;
    output[index] =
        from_float<T>(normalized * (1.0f + to_float(norm_style[col])) +
                      to_float(norm_style[cols + col]));
  }
}

template <class T>
__global__ void bias_residual(const T* projection, const T* bias,
                              const T* residual, T* output, int64_t count,
                              int cols) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    float value = to_float(projection[index]) + to_float(residual[index]);
    if (bias != nullptr) value += to_float(bias[index % cols]);
    output[index] = from_float<T>(value);
  }
}

// Preserve the observable result of two BF16 kernels (bias, then residual) in
// one launch. The explicit T round-trip is semantically required: folding all
// three operands into one float expression produces different BF16 values at
// rounding boundaries. The L3 contract only registers this kernel for BF16.
template <class T>
__global__ void bias_then_residual(const T* projection, const T* bias,
                                   const T* residual, T* output,
                                   int64_t count, int cols) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    float value = to_float(projection[index]);
    if (bias != nullptr) value += to_float(bias[index % cols]);
    const T biased = from_float<T>(value);
    output[index] =
        from_float<T>(to_float(biased) + to_float(residual[index]));
  }
}

// The combined value is rounded to T in `hidden` and read back before the
// reduction.  The legacy kernel does the same; keeping the round-trip is what
// makes this port bit-comparable against the old backend.
template <class T>
__global__ void bias_residual_rms_norm(const T* projection, const T* bias,
                                       const T* residual, const T* weight,
                                       T* hidden, T* normalized, int rows,
                                       int cols, float eps) {
  __shared__ float scratch[32];
  const int row = blockIdx.x;
  if (row >= rows) return;
  float square_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    float value = to_float(projection[index]) + to_float(residual[index]);
    if (bias != nullptr) value += to_float(bias[col]);
    const T rounded = from_float<T>(value);
    hidden[index] = rounded;
    const float stored = to_float(rounded);
    square_sum += stored * stored;
  }
  const float inverse_rms =
      rsqrtf(block_sum_parallel_unsafe(square_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    normalized[index] = from_float<T>(to_float(hidden[index]) * inverse_rms *
                                      to_float(weight[col]));
  }
}

template <class T>
__global__ void bias_residual_layer_norm(const T* projection,
                                         const T* projection_bias,
                                         const T* residual,
                                         const T* norm_weight,
                                         const T* norm_bias, T* hidden,
                                         T* normalized, int rows, int cols,
                                         float eps) {
  __shared__ float scratch[32];
  const int row = blockIdx.x;
  if (row >= rows) return;
  float sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    float value = to_float(projection[index]) + to_float(residual[index]);
    if (projection_bias != nullptr) value += to_float(projection_bias[col]);
    const T rounded = from_float<T>(value);
    hidden[index] = rounded;
    sum += to_float(rounded);
  }
  const float mean = block_sum_parallel_unsafe(sum, scratch) / cols;
  float variance_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float centered =
        to_float(hidden[static_cast<int64_t>(row) * cols + col]) - mean;
    variance_sum += centered * centered;
  }
  // Finish reading the mean before the variance reduction overwrites scratch.
  __syncthreads();
  const float inverse_std = rsqrtf(
      block_sum_parallel_unsafe(variance_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    normalized[index] = from_float<T>(
        (to_float(hidden[index]) - mean) * inverse_std *
            to_float(norm_weight[col]) +
        to_float(norm_bias[col]));
  }
}

// gate_style is [3 * cols]; the gate is the third segment.
template <class T>
__global__ void ada_gate_residual(const T* projection, const T* residual,
                                  const T* gate_style, T* output,
                                  int64_t count, int cols) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    const int col = static_cast<int>(index % cols);
    output[index] = from_float<T>(
        to_float(residual[index]) +
        to_float(projection[index]) * to_float(gate_style[2 * cols + col]));
  }
}

// Reads the gate from this layer's gate_style and normalizes with the next
// layer's norm_style; see the semantic table in norm_types.h.
template <class T>
__global__ void ada_gate_residual_rms_norm(const T* projection,
                                           const T* residual,
                                           const T* gate_style,
                                           const T* norm_style, T* hidden,
                                           T* normalized, int rows, int cols,
                                           float eps) {
  __shared__ float scratch[32];
  const int row = blockIdx.x;
  if (row >= rows) return;
  float square_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    const T rounded = from_float<T>(
        to_float(residual[index]) +
        to_float(projection[index]) * to_float(gate_style[2 * cols + col]));
    hidden[index] = rounded;
    const float value = to_float(rounded);
    square_sum += value * value;
  }
  const float inverse_rms =
      rsqrtf(block_sum_parallel_unsafe(square_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    const float value = to_float(hidden[index]) * inverse_rms;
    normalized[index] =
        from_float<T>(value * (1.0f + to_float(norm_style[col])) +
                      to_float(norm_style[cols + col]));
  }
}

// FP8-output variants preserve the legacy PI0.5 contract: normalization is
// accumulated in FP32 and quantized directly, without materializing and then
// re-reading a full F16 normalized tensor.
__global__ void rms_norm_quant_f16_e4m3(
    const half* input, const half* weight, __nv_fp8_e4m3* output,
    int rows, int cols, float eps, float inverse_scale) {
  __shared__ float scratch[32];
  const int row = blockIdx.x;
  float square_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float value = __half2float(input[row * cols + col]);
    square_sum += value * value;
  }
  const float inverse_rms =
      rsqrtf(block_sum_parallel_unsafe(square_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    float value = __half2float(input[row * cols + col]) * inverse_rms *
                  __half2float(weight[col]);
    value = fminf(448.0f, fmaxf(-448.0f, value * inverse_scale));
    output[row * cols + col] = static_cast<__nv_fp8_e4m3>(value);
  }
}

__global__ void layer_norm_quant_f16_e4m3(
    const half* input, const half* weight, const half* bias,
    __nv_fp8_e4m3* output, int rows, int cols, float eps,
    float inverse_scale) {
  __shared__ float scratch[32];
  const int row = blockIdx.x;
  float sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x)
    sum += __half2float(input[row * cols + col]);
  const float mean = block_sum_parallel_unsafe(sum, scratch) / cols;
  float variance_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float centered = __half2float(input[row * cols + col]) - mean;
    variance_sum += centered * centered;
  }
  __syncthreads();
  const float inverse_std = rsqrtf(
      block_sum_parallel_unsafe(variance_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    float value = (__half2float(input[row * cols + col]) - mean) * inverse_std;
    value = value * __half2float(weight[col]) + __half2float(bias[col]);
    value = fminf(448.0f, fmaxf(-448.0f, value * inverse_scale));
    output[row * cols + col] = static_cast<__nv_fp8_e4m3>(value);
  }
}

__global__ void ada_rms_norm_quant_f16_e4m3(
    const half* input, const half* style, __nv_fp8_e4m3* output,
    int rows, int cols, float eps, float inverse_scale) {
  __shared__ float scratch[32];
  const int row = blockIdx.x;
  float square_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float value = __half2float(input[row * cols + col]);
    square_sum += value * value;
  }
  const float inverse_rms =
      rsqrtf(block_sum_parallel_unsafe(square_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float normalized =
        __half2float(input[row * cols + col]) * inverse_rms;
    float value = normalized * (1.0f + __half2float(style[col])) +
                  __half2float(style[cols + col]);
    value = fminf(448.0f, fmaxf(-448.0f, value * inverse_scale));
    output[row * cols + col] = static_cast<__nv_fp8_e4m3>(value);
  }
}

__global__ void bias_residual_rms_norm_quant_f16_e4m3(
    const half* projection, const half* bias, const half* residual,
    const half* weight, half* hidden, __nv_fp8_e4m3* normalized,
    int rows, int cols, float eps, float inverse_scale) {
  __shared__ float scratch[32];
  const int row = blockIdx.x;
  float square_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    float value =
        __half2float(projection[index]) + __half2float(residual[index]);
    if (bias != nullptr) value += __half2float(bias[col]);
    const half rounded = __float2half(value);
    hidden[index] = rounded;
    value = __half2float(rounded);
    square_sum += value * value;
  }
  const float inverse_rms =
      rsqrtf(block_sum_parallel_unsafe(square_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    float value = __half2float(hidden[index]) * inverse_rms *
                  __half2float(weight[col]);
    value = fminf(448.0f, fmaxf(-448.0f, value * inverse_scale));
    normalized[index] = static_cast<__nv_fp8_e4m3>(value);
  }
}

__global__ void bias_residual_layer_norm_quant_f16_e4m3(
    const half* projection, const half* projection_bias, const half* residual,
    const half* norm_weight, const half* norm_bias, half* hidden,
    __nv_fp8_e4m3* normalized, int rows, int cols, float eps,
    float inverse_scale) {
  __shared__ float scratch[32];
  const int row = blockIdx.x;
  float sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    float value =
        __half2float(projection[index]) + __half2float(residual[index]);
    if (projection_bias != nullptr)
      value += __half2float(projection_bias[col]);
    const half rounded = __float2half(value);
    hidden[index] = rounded;
    sum += __half2float(rounded);
  }
  const float mean = block_sum_parallel_unsafe(sum, scratch) / cols;
  float variance_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float centered =
        __half2float(hidden[static_cast<int64_t>(row) * cols + col]) - mean;
    variance_sum += centered * centered;
  }
  __syncthreads();
  const float inverse_std = rsqrtf(
      block_sum_parallel_unsafe(variance_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    float value = (__half2float(hidden[index]) - mean) * inverse_std;
    value = value * __half2float(norm_weight[col]) +
            __half2float(norm_bias[col]);
    value = fminf(448.0f, fmaxf(-448.0f, value * inverse_scale));
    normalized[index] = static_cast<__nv_fp8_e4m3>(value);
  }
}

__global__ void ada_gate_residual_rms_norm_quant_f16_e4m3(
    const half* projection, const half* residual, const half* gate_style,
    const half* norm_style, half* hidden, __nv_fp8_e4m3* normalized,
    int rows, int cols, float eps, float inverse_scale) {
  __shared__ float scratch[32];
  const int row = blockIdx.x;
  float square_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    const float gate = __half2float(gate_style[2 * cols + col]);
    const half rounded = __float2half(
        __half2float(residual[index]) + __half2float(projection[index]) * gate);
    hidden[index] = rounded;
    const float value = __half2float(rounded);
    square_sum += value * value;
  }
  const float inverse_rms =
      rsqrtf(block_sum_parallel_unsafe(square_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    float value = __half2float(hidden[index]) * inverse_rms;
    value = value * (1.0f + __half2float(norm_style[col])) +
            __half2float(norm_style[cols + col]);
    value = fminf(448.0f, fmaxf(-448.0f, value * inverse_scale));
    normalized[index] = static_cast<__nv_fp8_e4m3>(value);
  }
}

// Exact [10, 1024] action specialization from the legacy production path.
// It retains the same 256-lane reduction tree while caching rounded hidden
// values and storing four E4M3 outputs per thread.
__global__ void ada_gate_residual_rms_norm_quant_f16_e4m3_10x1024(
    const half* projection, const half* residual, const half* gate_style,
    const half* norm_style, half* hidden, __nv_fp8_e4m3* normalized,
    float eps, float inverse_scale) {
  constexpr int cols = 1024;
  __shared__ float scratch[32];
  __shared__ half cached[cols];
  const int row = blockIdx.x;
  float square_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    const float gate = __half2float(gate_style[2 * cols + col]);
    const half rounded = __float2half(
        __half2float(residual[index]) + __half2float(projection[index]) * gate);
    hidden[index] = rounded;
    cached[col] = rounded;
    const float value = __half2float(rounded);
    square_sum += value * value;
  }
  const float inverse_rms =
      rsqrtf(block_sum_parallel_unsafe(square_sum, scratch) / cols + eps);

  union Half4 {
    uint2 packed;
    half values[4];
  };
  union Bytes4 {
    uint32_t packed;
    uint8_t values[4];
  };
  const int col = threadIdx.x * 4;
  Half4 h;
  Half4 scale;
  Half4 shift;
  h.packed = *reinterpret_cast<const uint2*>(cached + col);
  scale.packed = *reinterpret_cast<const uint2*>(norm_style + col);
  shift.packed = *reinterpret_cast<const uint2*>(norm_style + cols + col);
  Bytes4 output;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float value = __half2float(h.values[i]) * inverse_rms;
    value = value * (1.0f + __half2float(scale.values[i])) +
            __half2float(shift.values[i]);
    value = fminf(448.0f, fmaxf(-448.0f, value * inverse_scale));
    const __nv_fp8_e4m3 quantized = static_cast<__nv_fp8_e4m3>(value);
    output.values[i] = *reinterpret_cast<const uint8_t*>(&quantized);
  }
  *reinterpret_cast<uint32_t*>(normalized + row * cols + col) = output.packed;
}

}  // namespace apxinf::norm::kernels
