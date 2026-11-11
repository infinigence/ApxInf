#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace apxinf::tts::kernels {

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
__device__ inline float to_float(float value) {
  return value;
}

template <class T>
__device__ T from_float(float value);

template <>
__device__ inline __half from_float<__half>(float value) {
  return __float2half(value);
}

template <>
__device__ inline __nv_bfloat16 from_float<__nv_bfloat16>(float value) {
  return __float2bfloat16(value);
}

template <>
__device__ inline float from_float<float>(float value) {
  return value;
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

__device__ __forceinline__ float leaky_relu(float value) {
  return value >= 0.0f ? value : 0.1f * value;
}

__device__ __forceinline__ float leaky_relu_one_percent(float value) {
  return value >= 0.0f ? value : 0.01f * value;
}

enum Activation : int {
  kActivationNone = 0,
  kActivationGelu = 1,
  kActivationSilu = 2,
  kActivationGeluExact = 3,
  kActivationMish = 4,
  kActivationLeakyRelu = 5,
  kActivationTanh = 6,
  kActivationSigmoid = 7,
  kActivationRelu = 8,
  kActivationLeakyReluOnePercent = 9,
};

__device__ __forceinline__ float apply_activation(float value, int activation) {
  if (activation == kActivationGelu) return gelu_tanh(value);
  if (activation == kActivationSilu) return silu(value);
  if (activation == kActivationGeluExact) {
    return value * 0.5f * (1.0f + erff(value * 0.7071067811865475244f));
  }
  if (activation == kActivationMish) return mish(value);
  if (activation == kActivationLeakyRelu) return leaky_relu(value);
  if (activation == kActivationLeakyReluOnePercent) {
    return leaky_relu_one_percent(value);
  }
  if (activation == kActivationTanh) return tanhf(value);
  if (activation == kActivationSigmoid) return 1.0f / (1.0f + expf(-value));
  if (activation == kActivationRelu) return fmaxf(value, 0.0f);
  return value;
}

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

/* bias holds [2 * cols]: scale then shift. */
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

/* input [rows, 2 * cols] -> output [rows, cols]. */
template <class T>
__global__ void glu(const T* input, T* output, int64_t count, int cols) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    const int64_t row = index / cols;
    const int col = static_cast<int>(index % cols);
    const T* row_input = input + row * 2 * cols;
    const float first = to_float(row_input[col]);
    const float second = to_float(row_input[cols + col]);
    output[index] = from_float<T>(first / (1.0f + expf(-second)));
  }
}

/* output = input * sigmoid(secondary), same width. */
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

/* Global response normalization. One block owns the whole [rows, cols]
 * activation (the speech models run batch size one). style holds gamma then
 * beta. Each intermediate is rounded to the input dtype, matching the
 * materialized reference tensor graph. */
template <class T>
__global__ void grn(const T* input, const T* bias, T* output, int64_t rows,
                    int cols) {
  extern __shared__ float column_norm[];
  float local_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    float lane_sum[4];
#pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
      float partial[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      for (int64_t base = lane; base < rows; base += 16) {
#pragma unroll
        for (int part = 0; part < 4; ++part) {
          const int64_t row = base + part * 4;
          if (row < rows) {
            const float value = to_float(input[row * cols + col]);
            partial[part] = __fadd_rn(partial[part], __fmul_rn(value, value));
          }
        }
      }
      lane_sum[lane] = __fadd_rn(
          __fadd_rn(__fadd_rn(partial[0], partial[1]), partial[2]), partial[3]);
    }
    const float sum = __fadd_rn(__fadd_rn(lane_sum[0], lane_sum[2]),
                                __fadd_rn(lane_sum[1], lane_sum[3]));
    const float norm = to_float(from_float<T>(sqrtf(sum)));
    column_norm[col] = norm;
    local_sum += norm;
  }

  __shared__ float shared_total[32];
  float warp_sum = local_sum;
  for (int offset = 16; offset > 0; offset >>= 1) {
    warp_sum += __shfl_down_sync(0xFFFFFFFFu, warp_sum, offset);
  }
  const int warp = threadIdx.x >> 5;
  const int lanes = (blockDim.x + 31) >> 5;
  if ((threadIdx.x & 31) == 0) shared_total[warp] = warp_sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = 0.0f;
    for (int i = 0; i < lanes; ++i) total += shared_total[i];
    shared_total[0] = total;
  }
  __syncthreads();
  const float mean = to_float(from_float<T>(to_float(from_float<T>(
      shared_total[0] / static_cast<float>(cols))) + 1e-6f));

  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float scale = to_float(from_float<T>(column_norm[col] / mean));
    const float gamma = to_float(bias[col]);
    const float beta = to_float(bias[cols + col]);
    for (int64_t row = 0; row < rows; ++row) {
      const int64_t index = row * cols + col;
      const float value = to_float(input[index]);
      const float normalized = to_float(from_float<T>(value * scale));
      const float weighted = to_float(from_float<T>(gamma * normalized));
      const float shifted = to_float(from_float<T>(weighted + beta));
      output[index] = from_float<T>(shifted + value);
    }
  }
}

struct ColumnMoments {
  float mean, m2, count;
};

__device__ inline ColumnMoments combine_column_moments(
    ColumnMoments a, ColumnMoments b) {
  if (a.count == 0.0f) return b;
  if (b.count == 0.0f) return a;
  const float delta = b.mean - a.mean;
  const float count = a.count + b.count;
  const float ratio = b.count / count;
  return {a.mean + delta * ratio,
          a.m2 + b.m2 + delta * delta * a.count * ratio, count};
}

__device__ inline ColumnMoments warp_column_moments(ColumnMoments value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    const ColumnMoments other = {
        __shfl_down_sync(0xFFFFFFFFu, value.mean, offset),
        __shfl_down_sync(0xFFFFFFFFu, value.m2, offset),
        __shfl_down_sync(0xFFFFFFFFu, value.count, offset)};
    value = combine_column_moments(value, other);
  }
  return value;
}

/* FP32 moments followed by an explicit input-dtype statistics boundary. */
template <class T>
__global__ void column_norm_input_statistics(
    const T* input, const T* bias, T* output, int64_t rows, int cols, float eps) {
  __shared__ ColumnMoments partial[16];
  __shared__ float scale, shift;
  const int col = blockIdx.x;
  ColumnMoments value = {0.0f, 0.0f, 0.0f};
  int64_t count = 0;
  for (int64_t row = threadIdx.x; row < rows; row += blockDim.x) {
    const float x = to_float(input[row * cols + col]);
    const float delta = x - value.mean;
    ++count;
    const float mean = value.mean + delta / static_cast<float>(count);
    value = {mean, value.m2 + delta * (x - mean), static_cast<float>(count)};
  }
  value = warp_column_moments(value);
  if ((threadIdx.x & 31) == 0) partial[threadIdx.x / 32] = value;
  __syncthreads();
  if (blockDim.x > 32) {
    value = threadIdx.x < blockDim.x / 32
        ? partial[threadIdx.x] : ColumnMoments{0.0f, 0.0f, 0.0f};
    if (threadIdx.x < 32) value = warp_column_moments(value);
  }
  if (threadIdx.x == 0) {
    const float mean = to_float(from_float<T>(value.mean));
    const float rounded_eps = to_float(from_float<T>(eps));
    const float rstd = to_float(from_float<T>(
        rsqrtf(value.m2 / value.count + rounded_eps)));
    scale = rstd * to_float(bias[col]);
    shift = -scale * mean + to_float(bias[cols + col]);
  }
  __syncthreads();
  for (int64_t row = threadIdx.x; row < rows; row += blockDim.x) {
    const int64_t index = row * cols + col;
    output[index] = from_float<T>(scale * to_float(input[index]) + shift);
  }
}

template <class T>
__device__ __forceinline__ float rope_to_float(T value);
template <>
__device__ __forceinline__ float rope_to_float<float>(float value) { return value; }
template <>
__device__ __forceinline__ float rope_to_float(__half value) { return __half2float(value); }
template <>
__device__ __forceinline__ float rope_to_float(__nv_bfloat16 value) { return __bfloat162float(value); }

template <class T>
__device__ __forceinline__ T rope_from_float(float value);
template <>
__device__ __forceinline__ float rope_from_float<float>(float value) { return value; }
template <>
__device__ __forceinline__ __half rope_from_float<__half>(float value) { return __float2half(value); }
template <>
__device__ __forceinline__ __nv_bfloat16 rope_from_float<__nv_bfloat16>(float value) { return __float2bfloat16(value); }

template <class T>
__global__ void rope_apply_kernel(const T* __restrict__ input,
                                  const T* __restrict__ cos_values,
                                  const T* __restrict__ sin_values,
                                  T* __restrict__ output, int64_t pairs,
                                  int64_t time, int64_t heads, int64_t head_dim) {
  const int64_t index = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (index >= pairs) return;
  const int64_t pair = index % (head_dim / 2);
  const int64_t row = index / (head_dim / 2);
  const int64_t head = row % heads;
  const int64_t position = (row / heads) % time;
  const int64_t batch = (row / heads) / time;
  const int64_t element = 2 * pair;
  const int64_t first = ((batch * time + position) * heads + head) * head_dim + element;
  const int64_t second = first + 1;
  const T c = cos_values[position * head_dim + element];
  const T s = sin_values[position * head_dim + element];
  const T x1 = input[first];
  const T x2 = input[second];
  // Round products independently; CUDA may otherwise contract them into HFMA.
  const T x1c = rope_from_float<T>(__fmul_rn(rope_to_float(x1), rope_to_float(c)));
  const T x2s = rope_from_float<T>(__fmul_rn(rope_to_float(x2), rope_to_float(s)));
  const T x2c = rope_from_float<T>(__fmul_rn(rope_to_float(x2), rope_to_float(c)));
  const T x1s = rope_from_float<T>(__fmul_rn(rope_to_float(x1), rope_to_float(s)));
  output[first] = rope_from_float<T>(__fsub_rn(rope_to_float(x1c), rope_to_float(x2s)));
  output[second] = rope_from_float<T>(__fadd_rn(rope_to_float(x2c), rope_to_float(x1s)));
}

template <class T>
__global__ void mean_rows_kernel(const T* __restrict__ input,
                                 T* __restrict__ output,
                                 int rows, int cols) {
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    float sum = 0.0f;
    for (int row = 0; row < rows; ++row) {
      sum += rope_to_float<T>(input[(long long)row * cols + col]);
    }
    output[col] = rope_from_float<T>(sum / (float)rows);
  }
}

template <class T>
__global__ void argmax_rows_kernel(const T* __restrict__ logits,
                                   int32_t* __restrict__ index, int rows,
                                   int cols) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const T* values = logits + (long long)row * cols;
  float local_value = -INFINITY;
  int local_index = 0;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float value = rope_to_float<T>(values[col]);
    if (value > local_value || (value == local_value && col < local_index)) {
      local_value = value;
      local_index = col;
    }
  }
  __shared__ float best_value[32];
  __shared__ int best_index[32];
  for (int offset = 16; offset > 0; offset >>= 1) {
    const float other_value =
        __shfl_down_sync(0xFFFFFFFFu, local_value, offset);
    const int other_index = __shfl_down_sync(0xFFFFFFFFu, local_index, offset);
    if (other_value > local_value || (other_value == local_value && other_index < local_index)) {
      local_value = other_value;
      local_index = other_index;
    }
  }
  const int warp = threadIdx.x >> 5;
  if ((threadIdx.x & 31) == 0) {
    best_value[warp] = local_value;
    best_index[warp] = local_index;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const int warps = (blockDim.x + 31) >> 5;
    float value = best_value[0];
    int selected = best_index[0];
    for (int i = 1; i < warps; ++i) {
      if (best_value[i] > value || (best_value[i] == value && best_index[i] < selected)) {
        value = best_value[i];
        selected = best_index[i];
      }
    }
    index[row] = selected;
  }
}

/* Bias-before-cast projection epilogue: the GEMM writes an F32 projection and
 * the bias is added before the single F16 rounding, matching Torch's
 * F.linear -> aten::addmm accumulation contract. */
__global__ void project_bias_f32_to_f16_kernel(
    const float* __restrict__ projection, const __half* __restrict__ bias,
    __half* __restrict__ output, int64_t count, int cols) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    output[index] = __float2half(
        projection[index] + __half2float(bias[index % cols]));
  }
}

__global__ void cache_write_token_kernel(const uint16_t* input, uint16_t* cache,
    const uint32_t* meta, int64_t capacity, int64_t elements) {
  const uint32_t row = meta[1];
  if (row >= capacity) return;
  for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
       i < elements; i += int64_t(blockDim.x) * gridDim.x)
    cache[int64_t(row) * elements + i] = input[i];
}

/* output[row, col] = table[ids[row], col]. */
template <class T>
__global__ void gather_rows_kernel(const T* __restrict__ table,
                                   const int32_t* __restrict__ ids,
                                   T* __restrict__ output, int64_t rows,
                                   int cols) {
  const int64_t count = rows * int64_t(cols);
  for (int64_t index = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
       index < count; index += int64_t(blockDim.x) * gridDim.x) {
    const int64_t row = index / cols;
    const int64_t column = index % cols;
    output[index] = table[int64_t(ids[row]) * cols + column];
  }
}

template <class T>
__global__ void sinusoidal_embedding_kernel(
    const T* times, T* output, int64_t rows, int64_t width,
    float scale, float log_step) {
  const int64_t index = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
  if (index >= rows * width) return;
  const int64_t row = index / width;
  const int64_t column = index % width;
  const int64_t half_width = width / 2;
  const T scaled = rope_from_float<T>(rope_to_float(times[row]) * scale);
  const float frequency = expf(-float(column % half_width) * log_step);
  const float angle = rope_to_float(scaled) * frequency;
  output[index] = rope_from_float<T>(column < half_width ? sinf(angle) : cosf(angle));
}

}  // namespace apxinf::tts::kernels
