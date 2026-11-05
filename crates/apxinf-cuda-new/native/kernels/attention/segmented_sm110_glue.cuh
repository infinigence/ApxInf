#pragma once

__global__ void duplicate_q_vec8(const uint4* source, uint4* destination) {
  constexpr int vectors = kQueryWords / 8;
  int index = int(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < vectors) {
    uint4 value = source[index];
    destination[index] = value;
    destination[index + vectors] = value;
  }
}

__global__ void repack_lse(const float* native_lse, float* dense_lse) {
  int index = int(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= kBatch * kMergeRows) return;
  int batch = index / kMergeRows;
  int row = index % kMergeRows;
  int query_head = row / kQueryTokens;
  int token = row % kQueryTokens;
  constexpr int rounded_query = 256;
  int kv_head = query_head / 4;
  int lane = query_head % 4;
  dense_lse[index] = native_lse[
      batch * kKvHeads * rounded_query + kv_head * rounded_query +
      token * 4 + lane];
}

union Bf16x8 {
  uint4 vector;
  uint16_t words[8];
};

__global__ void merge_partial_vec8(const uint4* partial, const float* lse,
                                   uint4* output) {
  constexpr int vectors_per_row = kDim / 8;
  constexpr int vectors_per_batch = kQueryWords / 8;
  int warp = (int(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
  if (warp >= kMergeRows) return;
  int lane = threadIdx.x & 31;
  int token = warp / kQueryHeads;
  int head = warp % kQueryHeads;
  int lse_index = head * kQueryTokens + token;
  float weight0 = 0.0f, weight1 = 0.0f, sum = 0.0f;
  if (lane == 0) {
    float first = lse[lse_index];
    float second = lse[kMergeRows + lse_index];
    float maximum = fmaxf(first, second);
    weight0 = expf(first - maximum);
    weight1 = expf(second - maximum);
    sum = weight0 + weight1;
  }
  weight0 = __shfl_sync(0xffffffff, weight0, 0);
  weight1 = __shfl_sync(0xffffffff, weight1, 0);
  sum = __shfl_sync(0xffffffff, sum, 0);
  int vector = warp * vectors_per_row + lane;
  Bf16x8 first{}, second{}, result{};
  first.vector = partial[vector];
  second.vector = partial[vectors_per_batch + vector];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    float a = __bfloat162float(__ushort_as_bfloat16(first.words[i]));
    float b = __bfloat162float(__ushort_as_bfloat16(second.words[i]));
    result.words[i] = __bfloat16_as_ushort(
        __float2bfloat16_rn((weight0 * a + weight1 * b) / sum));
  }
  output[vector] = result.vector;
}
