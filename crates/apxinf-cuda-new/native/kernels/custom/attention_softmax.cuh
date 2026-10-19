#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdint>

namespace apxinf::attention::kernels {

// Head-major BF16 QK and P, with one warp per row. The lane-strided local
// sum followed by a butterfly reduction preserves eager FP32 softmax rounding
// before the explicit BF16 probability boundary. No intermediate is on host.
template <int Columns>
__global__ void attention_input_softmax(__nv_bfloat16* scores, const float* bias,
    int batch, int heads, int queries, int keys, bool broadcast_batch,
    bool causal, int query_start) {
  constexpr int width = Columns < 32 ? Columns : 32;
  constexpr int items = Columns / width;
  const int lane = threadIdx.x;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  if (row >= static_cast<int64_t>(batch) * heads * queries) return;
  const int query = row % queries;
  const int64_t bias_row = broadcast_batch ? row % (static_cast<int64_t>(heads) * queries) : row;
  float values[items];
  float maximum = -CUDART_INF_F;
#pragma unroll
  for (int i = 0; i < items; ++i) {
    const int key = lane + i * width;
    float value = -CUDART_INF_F;
    if (key < keys && (!causal || key <= query_start + query)) {
      value = __bfloat162float(__float2bfloat16(
          __bfloat162float(scores[row * keys + key]) + bias[bias_row * keys + key]));
    }
    values[i] = value;
    maximum = fmaxf(maximum, value);
  }
  const unsigned active = __activemask();
  for (int delta = width / 2; delta; delta /= 2)
    maximum = fmaxf(maximum, __shfl_xor_sync(active, maximum, delta, width));
  float sum = 0.0F;
#pragma unroll
  for (int i = 0; i < items; ++i) {
    values[i] = maximum == -CUDART_INF_F ? 0.0F : expf(values[i] - maximum);
    sum += values[i];
  }
  for (int delta = width / 2; delta; delta /= 2)
    sum += __shfl_xor_sync(active, sum, delta, width);
#pragma unroll
  for (int i = 0; i < items; ++i) {
    const int key = lane + i * width;
    if (key < keys)
      scores[row * keys + key] = __float2bfloat16(sum == 0.0F ? 0.0F : values[i] / sum);
  }
}

inline cudaError_t launch_attention_input_softmax(void* scores, const float* bias,
    int batch, int heads, int queries, int keys, bool broadcast_batch,
    bool causal, int query_start, cudaStream_t stream) {
  const int64_t rows = static_cast<int64_t>(batch) * heads * queries;
#define APXINF_INPUT_SOFTMAX(C) \
  case C: { \
    constexpr int width = C < 32 ? C : 32; \
    constexpr int rows_per_block = 128 / width; \
    attention_input_softmax<C><<<static_cast<unsigned>((rows + rows_per_block - 1) / rows_per_block), dim3(width, rows_per_block), 0, stream>>>( \
        static_cast<__nv_bfloat16*>(scores), bias, batch, heads, queries, keys, broadcast_batch, causal, query_start); \
    break; \
  }
  int columns = 1;
  while (columns < keys) columns *= 2;
  switch (columns) {
    APXINF_INPUT_SOFTMAX(1)
    APXINF_INPUT_SOFTMAX(2)
    APXINF_INPUT_SOFTMAX(4)
    APXINF_INPUT_SOFTMAX(8)
    APXINF_INPUT_SOFTMAX(16)
    APXINF_INPUT_SOFTMAX(32)
    APXINF_INPUT_SOFTMAX(64)
    APXINF_INPUT_SOFTMAX(128)
    APXINF_INPUT_SOFTMAX(256)
    APXINF_INPUT_SOFTMAX(512)
    APXINF_INPUT_SOFTMAX(1024)
    default: return cudaErrorInvalidValue;
  }
#undef APXINF_INPUT_SOFTMAX
  return cudaGetLastError();
}

}  // namespace apxinf::attention::kernels
