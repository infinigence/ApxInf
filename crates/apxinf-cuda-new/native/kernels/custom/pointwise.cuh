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
};

__device__ __forceinline__ float apply_activation(float value, int activation) {
  if (activation == kActivationGelu) return gelu_tanh(value);
  if (activation == kActivationSilu) return silu(value);
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
    output[index] = from_float<T>(apply_activation(value, activation));
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
