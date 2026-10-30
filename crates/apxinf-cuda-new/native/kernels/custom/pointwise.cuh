#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace apxinf::pointwise::kernels {

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

template <>
__device__ inline float to_float(float value) { return value; }

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

template <>
__device__ inline float from_float(float value) { return value; }

template <class T>
__device__ __forceinline__ float rounded(float value) {
  return to_float(from_float<T>(value));
}

// Tensor-expression tanh GELU. Each assignment is a materialization boundary,
// including 0.5*x, rather than a single fused float expression.
template <class T>
__device__ __forceinline__ float gelu_rounded(float x) {
  // PyTorch scalar pow(x, 3) specializes to scalar_t x*x*x: the
  // square is rounded before the second multiplication for F16/BF16.
  const float square = rounded<T>(__fmul_rn(x, x));
  const float cube = rounded<T>(__fmul_rn(square, x));
  const float scaled_cube = rounded<T>(__fmul_rn(0.044715f, cube));
  const float sum = rounded<T>(__fadd_rn(x, scaled_cube));
  const float inner = rounded<T>(__fmul_rn(0.7978845608028654f, sum));
  const float activation = rounded<T>(tanhf(inner));
  const float factor = rounded<T>(__fadd_rn(1.0f, activation));
  const float half_x = rounded<T>(__fmul_rn(0.5f, x));
  return __fmul_rn(half_x, factor);
}

__device__ __forceinline__ float gelu_tanh(float value) {
  constexpr float kAlpha = 0.7978845608028654f;
  return 0.5f * value *
         (1.0f + tanhf(kAlpha * (value + 0.044715f * value * value * value)));
}

__device__ __forceinline__ float silu(float value) {
  return value / (1.0f + expf(-value));
}

// Matches the stable pointwise activation ABI encoding.
enum Activation : int {
  kActivationNone = 0,
  kActivationGelu = 1,
  kActivationSilu = 2,
  kActivationGeluExact = 3,
  kActivationRelu = 4,
  kActivationGeluRounded = 5,
  kActivationSwishRounded = 6,
};

__device__ __forceinline__ float apply_activation(float value, int activation) {
  if (activation == kActivationGelu) return gelu_tanh(value);
  if (activation == kActivationSilu) return silu(value);
  if (activation == kActivationGeluExact) {
    return 0.5f * value * (1.0f + erff(value * 0.7071067811865475f));
  }
  if (activation == kActivationRelu) return fmaxf(value, 0.0f);
  return value;
}

// gate_up is [rows, 2 * inner]; the gate half is gelu'd and multiplied by the
// up half, producing [rows, inner].
template <class T>
__global__ void geglu(const T* gate_up, T* output, int rows, int inner) {
  const int64_t count = static_cast<int64_t>(rows) * inner;
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    const int64_t row = index / inner;
    const int col = static_cast<int>(index % inner);
    const T* row_input = gate_up + row * 2 * inner;
    output[index] = from_float<T>(gelu_tanh(to_float(row_input[col])) *
                                  to_float(row_input[inner + col]));
  }
}

// Optional per-column bias followed by an optional activation.  A null bias
// with kActivationNone is a plain copy and is rejected by the contract.
template <class T>
__global__ void bias_activation(const T* input, const T* bias, T* output,
                                int64_t count, int cols, int activation) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    float value = to_float(input[index]);
    if (bias != nullptr) value += to_float(bias[index % cols]);
    if (activation == kActivationSwishRounded) {
      value = rounded<T>(value);
      const float sigmoid = rounded<T>(1.0f / (1.0f + expf(-value)));
      output[index] = from_float<T>(__fmul_rn(value, sigmoid));
    } else {
      output[index] = from_float<T>(activation == kActivationGeluRounded
          ? gelu_rounded<T>(rounded<T>(value)) : apply_activation(value, activation));
    }
  }
}

// Semantic encodings follow pointwise_types.h. Each operation rounds only at
// its output; composing operations preserves the caller's rounding boundaries.
template <class T>
__global__ void binary_scalar(const T* input, const T* secondary, T* output,
                              int64_t count, int cols, int semantic, float scalar) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    const float x = to_float(input[index]);
    float value;
    if (semantic == 12) value = __fmul_rn(x, scalar);
    else if (semantic == 13) value = __fadd_rn(x, scalar);
    else if (semantic == 14) value = __fdiv_rn(x, scalar);
    else if (semantic == 15) value = expf(x);
    else if (semantic == 16) value = sinf(x);
    else if (semantic == 17) value = cosf(x);
    else {
      const int64_t j = (semantic == 9 || semantic == 10) ? index % cols
                         : semantic == 11 ? index / cols : index;
      const float y = to_float(secondary[j]);
      value = (semantic == 6 || semantic == 9) ? __fadd_rn(x, y)
               : semantic == 7 ? __fsub_rn(x, y) : __fmul_rn(x, y);
    }
    output[index] = from_float<T>(value);
  }
}

// One explicit-Euler flow-matching step: state + dt * velocity.
template <class T>
__global__ void euler_update(const T* state, const T* velocity, T* output,
                             int64_t count, float dt) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    output[index] = from_float<T>(to_float(state[index]) +
                                  dt * to_float(velocity[index]));
  }
}

}  // namespace apxinf::pointwise::kernels
