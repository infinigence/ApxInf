// Copyright 2026 ApxInf contributors.

#include "model_ops.h"

#include <cuda_bf16.h>

namespace apxinf::cuda_new::model_ops {
namespace {

__global__ void embedding_gather_kernel(const __nv_bfloat16* __restrict__ table,
                                        const int32_t* __restrict__ ids,
                                        __nv_bfloat16* __restrict__ output,
                                        int tokens, int hidden, int vocab) {
  const long long index = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  const long long total = (long long)tokens * hidden;
  if (index >= total) return;
  const int token = static_cast<int>(index / hidden);
  const int element = static_cast<int>(index % hidden);
  int id = ids[token];
  // Clamp rather than fault: an out-of-range id is a caller bug, but reading
  // arbitrary device memory would turn it into a silent wrong answer.
  if (id < 0 || id >= vocab) id = 0;
  output[index] = table[(long long)id * hidden + element];
}

__global__ void argmax_kernel(const __nv_bfloat16* __restrict__ logits,
                              int32_t* __restrict__ index, int count) {
  __shared__ float best_value[32];
  __shared__ int best_index[32];

  float local_value = -INFINITY;
  int local_index = 0;
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    const float value = __bfloat162float(logits[i]);
    if (value > local_value) {
      local_value = value;
      local_index = i;
    }
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    const float other_value =
        __shfl_down_sync(0xFFFFFFFFu, local_value, offset);
    const int other_index = __shfl_down_sync(0xFFFFFFFFu, local_index, offset);
    if (other_value > local_value) {
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
    const int warps = (blockDim.x + 31) / 32;
    float value = best_value[0];
    int chosen = best_index[0];
    for (int i = 1; i < warps; ++i) {
      if (best_value[i] > value) {
        value = best_value[i];
        chosen = best_index[i];
      }
    }
    *index = chosen;
  }
}

__global__ void vision_position_kernel(
    const __nv_bfloat16* __restrict__ table,
    __nv_bfloat16* __restrict__ embeddings,
    int32_t* __restrict__ position_ids, int source_side, int height, int width,
    int hidden, int merge_size) {
  const int token = static_cast<int>(blockIdx.x);
  const int merge_area = merge_size * merge_size;
  const int merged_index = token / merge_area;
  const int inner_index = token % merge_area;
  const int merged_width = width / merge_size;
  const int row = (merged_index / merged_width) * merge_size +
                  inner_index / merge_size;
  const int column = (merged_index % merged_width) * merge_size +
                     inner_index % merge_size;

  if (threadIdx.x == 0) {
    position_ids[2LL * token] = row;
    position_ids[2LL * token + 1] = column;
  }

  const float row_position =
      height > 1 ? static_cast<float>(row) * static_cast<float>(source_side - 1) /
                       static_cast<float>(height - 1)
                 : 0.0f;
  const float column_position =
      width > 1 ? static_cast<float>(column) *
                          static_cast<float>(source_side - 1) /
                      static_cast<float>(width - 1)
                : 0.0f;
  const int row0 = static_cast<int>(floorf(row_position));
  const int column0 = static_cast<int>(floorf(column_position));
  const int row1 = min(row0 + 1, source_side - 1);
  const int column1 = min(column0 + 1, source_side - 1);
  const float row_weight = row_position - static_cast<float>(row0);
  const float column_weight = column_position - static_cast<float>(column0);
  const float row_weight0 = 1.0f - row_weight;
  const float column_weight0 = 1.0f - column_weight;

  for (int channel = threadIdx.x; channel < hidden; channel += blockDim.x) {
    const long long offset00 =
        (static_cast<long long>(row0) * source_side + column0) * hidden +
        channel;
    const long long offset01 =
        (static_cast<long long>(row0) * source_side + column1) * hidden +
        channel;
    const long long offset10 =
        (static_cast<long long>(row1) * source_side + column0) * hidden +
        channel;
    const long long offset11 =
        (static_cast<long long>(row1) * source_side + column1) * hidden +
        channel;
    const float value =
        row_weight0 * column_weight0 * __bfloat162float(table[offset00]) +
        row_weight0 * column_weight * __bfloat162float(table[offset01]) +
        row_weight * column_weight0 * __bfloat162float(table[offset10]) +
        row_weight * column_weight * __bfloat162float(table[offset11]);
    embeddings[static_cast<long long>(token) * hidden + channel] =
        __float2bfloat16(value);
  }
}

}  // namespace

int embedding_gather(const void* table, const int32_t* ids, void* output,
                     int tokens, int hidden, int vocab, cudaStream_t stream) {
  if (tokens <= 0 || hidden <= 0 || vocab <= 0) return -1;
  const long long total = (long long)tokens * hidden;
  const int threads = 256;
  const long long blocks = (total + threads - 1) / threads;
  embedding_gather_kernel<<<static_cast<int>(blocks), threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(table), ids,
      static_cast<__nv_bfloat16*>(output), tokens, hidden, vocab);
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

int argmax_bf16(const void* logits, int32_t* index, int count,
                cudaStream_t stream) {
  if (count <= 0) return -1;
  argmax_kernel<<<1, 1024, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(logits), index, count);
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

int vision_position_bf16(const void* table, void* embeddings,
                         int32_t* position_ids, int source_side, int height,
                         int width, int hidden, int merge_size,
                         cudaStream_t stream) {
  const int tokens = height * width;
  vision_position_kernel<<<tokens, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(table),
      static_cast<__nv_bfloat16*>(embeddings), position_ids, source_side,
      height, width, hidden, merge_size);
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

}  // namespace apxinf::cuda_new::model_ops
