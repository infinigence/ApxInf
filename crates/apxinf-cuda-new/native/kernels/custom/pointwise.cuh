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

__device__ __forceinline__ float mish(float value) {
  const float softplus = value > 20.0f ? value : log1pf(expf(value));
  return value * tanhf(softplus);
}

// Matches the stable pointwise activation ABI encoding.
enum Activation : int {
  kActivationNone = 0,
  kActivationGelu = 1,
  kActivationSilu = 2,
  kActivationGeluExact = 3,
  kActivationRelu = 4,
  kActivationMish = 5,
  kActivationTanh = 6,
  kActivationLeakyRelu = 7,
  kActivationSigmoid = 8,
  kActivationLeakyReluOnePercent = 9,
};

__device__ __forceinline__ float apply_activation(float value, int activation,
                                                  float negative_slope) {
  if (activation == kActivationGelu) return gelu_tanh(value);
  if (activation == kActivationSilu) return silu(value);
  if (activation == kActivationGeluExact) {
    return 0.5f * value * (1.0f + erff(value * 0.7071067811865475f));
  }
  if (activation == kActivationRelu) return fmaxf(value, 0.0f);
  if (activation == kActivationMish) return mish(value);
  if (activation == kActivationTanh) return tanhf(value);
  if (activation == kActivationLeakyRelu) {
    return value >= 0.0f ? value : negative_slope * value;
  }
  if (activation == kActivationSigmoid) return 1.0f / (1.0f + expf(-value));
  if (activation == kActivationLeakyReluOnePercent) {
    return value >= 0.0f ? value : 0.01f * value;
  }
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
//
// `round_bias` rounds `input + bias` to the tensor dtype before the
// activation.  A producer that defers its bias and is read again by an
// activation needs that intermediate rounding to match the value it would
// have written itself.
template <class T>
__global__ void bias_activation(const T* input, const T* bias, T* output,
                                int64_t count, int cols, int activation,
                                int round_bias, float negative_slope) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    float value = to_float(input[index]);
    if (bias != nullptr) value += to_float(bias[index % cols]);
    if (round_bias != 0) value = to_float(from_float<T>(value));
    output[index] = from_float<T>(apply_activation(value, activation,
                                                   negative_slope));
  }
}

// bias holds [2 * cols]: scale then shift.
template <class T>
__global__ void affine(const T* input, const T* bias, T* output, int64_t count,
                       int cols) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    const int col = static_cast<int>(index % cols);
    output[index] = from_float<T>(to_float(input[index]) *
                                      to_float(bias[col]) +
                                  to_float(bias[cols + col]));
  }
}

// output = input * sigmoid(secondary), same width.
template <class T>
__global__ void glu_separate(const T* input, const T* secondary, T* output,
                             int64_t count) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    const float gate = to_float(secondary[index]);
    output[index] =
        from_float<T>(to_float(input[index]) / (1.0f + expf(-gate)));
  }
}

// input is [rows, 2 * cols]: the a half then the b half.  The optional
// secondary is added element-wise to the a/b halves before the gate.
template <class T>
__global__ void tanh_sigmoid_gate(const T* input, const T* secondary,
                                  T* output, int64_t count, int cols) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    const int64_t row = index / cols;
    const int col = static_cast<int>(index % cols);
    const int64_t source = row * 2 * cols;
    float t = to_float(input[source + col]);
    float s = to_float(input[source + cols + col]);
    if (secondary != nullptr) {
      t += to_float(secondary[source + col]);
      s += to_float(secondary[source + cols + col]);
    }
    output[index] = from_float<T>(tanhf(t) / (1.0f + expf(-s)));
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
