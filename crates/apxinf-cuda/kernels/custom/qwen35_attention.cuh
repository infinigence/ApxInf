#pragma once

__global__ void qwen35_attention_prepare_bf16_kernel(
    const __nv_bfloat16* q_projection, const __nv_bfloat16* k_projection,
    const __nv_bfloat16* v_projection, const __nv_bfloat16* q_norm_weight,
    const __nv_bfloat16* k_norm_weight, __nv_bfloat16* query,
    __nv_bfloat16* key, __nv_bfloat16* value, __nv_bfloat16* gate,
    const uint32_t* rope_positions) {
  constexpr int kQHeads = 24;
  constexpr int kKvHeads = 4;
  constexpr int kDim = 256;
  constexpr int kRotary = 64;
  constexpr float kTheta = 10000000.0f;
  constexpr float kEpsilon = 1.0e-6f;
  __shared__ float scratch[8];
  __shared__ float normalized[kDim];
  const int block = blockIdx.x;
  const int dimension = threadIdx.x;
  const bool is_query = block < kQHeads;
  const int head = is_query ? block : block - kQHeads;
  const __nv_bfloat16* source = is_query
      ? q_projection + static_cast<int64_t>(head) * 2 * kDim
      : k_projection + static_cast<int64_t>(head) * kDim;
  const __nv_bfloat16* norm_weight =
      is_query ? q_norm_weight : k_norm_weight;
  const float raw = __bfloat162float(source[dimension]);
  const float square_sum = block_sum(raw * raw, scratch);
  const float inverse = rsqrtf(square_sum / kDim + kEpsilon);
  normalized[dimension] =
      raw * inverse * (1.0f + __bfloat162float(norm_weight[dimension]));
  __syncthreads();
  float prepared = normalized[dimension];
  if (dimension < kRotary) {
    const int frequency = dimension & 31;
    const int axis = frequency % 3;
    const float angle = static_cast<float>(rope_positions[axis]) *
        powf(kTheta, -2.0f * static_cast<float>(frequency) / kRotary);
    float sine, cosine;
    sincosf(angle, &sine, &cosine);
    const int partner = dimension < 32 ? dimension + 32 : dimension - 32;
    const float rotated = dimension < 32
        ? -normalized[partner] : normalized[partner];
    prepared = prepared * cosine + rotated * sine;
  }
  if (is_query) {
    query[static_cast<int64_t>(head) * kDim + dimension] =
        __float2bfloat16(prepared);
    gate[static_cast<int64_t>(head) * kDim + dimension] =
        source[kDim + dimension];
  } else {
    key[static_cast<int64_t>(head) * kDim + dimension] =
        __float2bfloat16(prepared);
    value[static_cast<int64_t>(head) * kDim + dimension] =
        v_projection[static_cast<int64_t>(head) * kDim + dimension];
  }
}

__global__ void qwen35_attention_prepare_m8_bf16_kernel(
    const __nv_bfloat16* q_projection, const __nv_bfloat16* k_projection,
    const __nv_bfloat16* v_projection, const __nv_bfloat16* q_norm_weight,
    const __nv_bfloat16* k_norm_weight, __nv_bfloat16* query,
    __nv_bfloat16* key, __nv_bfloat16* value, __nv_bfloat16* gate,
    const uint32_t* rope_positions, int tokens) {
  constexpr int kQHeads = 24;
  constexpr int kKvHeads = 4;
  constexpr int kBlocksPerToken = kQHeads + kKvHeads;
  constexpr int kDim = 256;
  constexpr int kRotary = 64;
  constexpr int kQWidth = kQHeads * kDim;
  constexpr int kKvWidth = kKvHeads * kDim;
  constexpr float kTheta = 10000000.0f;
  constexpr float kEpsilon = 1.0e-6f;
  __shared__ float scratch[8];
  __shared__ float normalized[kDim];
  const int token = blockIdx.x / kBlocksPerToken;
  const int block = blockIdx.x % kBlocksPerToken;
  if (token >= tokens) return;
  const int dimension = threadIdx.x;
  const bool is_query = block < kQHeads;
  const int head = is_query ? block : block - kQHeads;
  const __nv_bfloat16* source = is_query
      ? q_projection + static_cast<int64_t>(token) * 2 * kQWidth +
            static_cast<int64_t>(head) * 2 * kDim
      : k_projection + static_cast<int64_t>(token) * kKvWidth +
            static_cast<int64_t>(head) * kDim;
  const __nv_bfloat16* norm_weight =
      is_query ? q_norm_weight : k_norm_weight;
  const float raw = __bfloat162float(source[dimension]);
  const float square_sum = block_sum(raw * raw, scratch);
  const float inverse = rsqrtf(square_sum / kDim + kEpsilon);
  normalized[dimension] =
      raw * inverse * (1.0f + __bfloat162float(norm_weight[dimension]));
  __syncthreads();
  float prepared = normalized[dimension];
  if (dimension < kRotary) {
    const int frequency = dimension & 31;
    const int axis = frequency % 3;
    const float angle = static_cast<float>(rope_positions[token * 3 + axis]) *
        powf(kTheta, -2.0f * static_cast<float>(frequency) / kRotary);
    float sine, cosine;
    sincosf(angle, &sine, &cosine);
    const int partner = dimension < 32 ? dimension + 32 : dimension - 32;
    const float rotated = dimension < 32
        ? -normalized[partner] : normalized[partner];
    prepared = prepared * cosine + rotated * sine;
  }
  if (is_query) {
    const int64_t offset = static_cast<int64_t>(token) * kQWidth +
        static_cast<int64_t>(head) * kDim + dimension;
    query[offset] = __float2bfloat16(prepared);
    gate[offset] = source[kDim + dimension];
  } else {
    const int64_t offset = static_cast<int64_t>(token) * kKvWidth +
        static_cast<int64_t>(head) * kDim + dimension;
    key[offset] = __float2bfloat16(prepared);
    value[offset] = v_projection[
        static_cast<int64_t>(token) * kKvWidth +
        static_cast<int64_t>(head) * kDim + dimension];
  }
}

__global__ void qwen35_attention_gate_bf16_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* gate,
    __nv_bfloat16* output, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    const float value = __bfloat162float(input[index]);
    const float gate_value = __bfloat162float(gate[index]);
    output[index] = __float2bfloat16(
        value / (1.0f + expf(-gate_value)));
  }
}

// Long-context decode candidate.  The incumbent assigns one CTA to each Q
// head, which exposes only 24 CTAs on an RTX 4090.  This stage partitions the
// valid KV interval across CTAs and writes numerically stable online-softmax
// partials.  A second kernel merges those partials in FP32.  The extra GMEM
// traffic is bounded by heads * splits * (head_dim + 2) floats and is small
// compared with the long-context KV reads it enables the full GPU to service.
template <int kWarps = 8>
__global__ void qwen35_attention_flash_split_cta_bf16_kernel(
    const __nv_bfloat16* query, const __nv_bfloat16* key_cache,
    const __nv_bfloat16* value_cache, float* partial_max,
    float* partial_sum, float* partial_accumulator, int split_count,
    int bucket_kv_len, int max_seq_len, float scale,
    const uint32_t* position, int tokens) {
  constexpr int kQueryHeads = 24;
  constexpr int kKvHeads = 4;
  constexpr int kHeadDim = 256;
  constexpr int kElementsPerThread = kHeadDim / 32;
  const int token_index = blockIdx.z;
  if (token_index >= tokens) return;
  const int query_head = blockIdx.x;
  const int split = blockIdx.y;
  const int thread = threadIdx.x;
  const int warp = thread / 32;
  const int lane = thread & 31;
  const int kv_head = query_head / (kQueryHeads / kKvHeads);
  const int valid_len = min(static_cast<int>(position[token_index]) + 1,
                            bucket_kv_len);
  const int span = (valid_len + split_count - 1) / split_count;
  const int begin = min(split * span, valid_len);
  const int end = min(begin + span, valid_len);
  const int partial = (token_index * kQueryHeads + query_head) * split_count + split;

  if (begin >= end) {
    if (thread == 0) {
      partial_max[partial] = -INFINITY;
      partial_sum[partial] = 0.0f;
    }
    if (thread < kHeadDim)
      partial_accumulator[static_cast<int64_t>(partial) * kHeadDim + thread] =
          0.0f;
    return;
  }

  __shared__ float query_shared[kHeadDim];
  if (warp == 0) {
#pragma unroll
    for (int item = 0; item < kElementsPerThread; ++item) {
      const int dimension = item * 32 + lane;
      query_shared[dimension] = __bfloat162float(
          query[(static_cast<int64_t>(token_index) * kQueryHeads + query_head) *
                    kHeadDim + dimension]);
    }
  }
  __syncthreads();

  float query_register[kElementsPerThread];
  float accumulator[kElementsPerThread];
#pragma unroll
  for (int item = 0; item < kElementsPerThread; ++item) {
    query_register[item] = query_shared[item * 32 + lane];
    accumulator[item] = 0.0f;
  }
  const __nv_bfloat16* key_base =
      key_cache + static_cast<int64_t>(kv_head) * max_seq_len * kHeadDim;
  const __nv_bfloat16* value_base =
      value_cache + static_cast<int64_t>(kv_head) * max_seq_len * kHeadDim;
  float maximum = -INFINITY;
  float sum = 0.0f;
  for (int token = begin + warp; token < end; token += kWarps) {
    float dot = 0.0f;
#pragma unroll
    for (int item = 0; item < kElementsPerThread; ++item) {
      const int dimension = item * 32 + lane;
      dot += query_register[item] * __bfloat162float(
          key_base[static_cast<int64_t>(token) * kHeadDim + dimension]);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      dot += __shfl_xor_sync(0xffffffff, dot, offset);
    dot *= scale;
    const float next_maximum = fmaxf(maximum, dot);
    const float probability = expf(dot - next_maximum);
    const float previous_scale = expf(maximum - next_maximum);
    sum = sum * previous_scale + probability;
#pragma unroll
    for (int item = 0; item < kElementsPerThread; ++item) {
      const int dimension = item * 32 + lane;
      accumulator[item] = accumulator[item] * previous_scale +
          probability * __bfloat162float(
              value_base[static_cast<int64_t>(token) * kHeadDim + dimension]);
    }
    maximum = next_maximum;
  }

  __shared__ float warp_maximum[kWarps];
  __shared__ float warp_sum[kWarps];
  __shared__ float warp_accumulator[kWarps][kHeadDim];
  if (lane == 0) {
    warp_maximum[warp] = maximum;
    warp_sum[warp] = sum;
  }
#pragma unroll
  for (int item = 0; item < kElementsPerThread; ++item)
    warp_accumulator[warp][item * 32 + lane] = accumulator[item];
  __syncthreads();

  if (warp == 0) {
    float merged_maximum = -INFINITY;
#pragma unroll
    for (int source = 0; source < kWarps; ++source)
      merged_maximum = fmaxf(merged_maximum, warp_maximum[source]);
    float merged_sum = 0.0f;
    float merged_accumulator[kElementsPerThread];
#pragma unroll
    for (int item = 0; item < kElementsPerThread; ++item)
      merged_accumulator[item] = 0.0f;
#pragma unroll
    for (int source = 0; source < kWarps; ++source) {
      const float factor = expf(warp_maximum[source] - merged_maximum);
      merged_sum += warp_sum[source] * factor;
#pragma unroll
      for (int item = 0; item < kElementsPerThread; ++item)
        merged_accumulator[item] +=
            warp_accumulator[source][item * 32 + lane] * factor;
    }
    if (lane == 0) {
      partial_max[partial] = merged_maximum;
      partial_sum[partial] = merged_sum;
    }
#pragma unroll
    for (int item = 0; item < kElementsPerThread; ++item)
      partial_accumulator[static_cast<int64_t>(partial) * kHeadDim +
                          item * 32 + lane] = merged_accumulator[item];
  }
}

__global__ void qwen35_attention_flash_split_cta_reduce_bf16_kernel(
    const float* partial_max, const float* partial_sum,
    const float* partial_accumulator, __nv_bfloat16* output,
    int split_count, int tokens) {
  constexpr int kQueryHeads = 24;
  const int token_index = blockIdx.y;
  if (token_index >= tokens) return;
  constexpr int kHeadDim = 256;
  constexpr int kMaxSplits = 16;
  const int query_head = blockIdx.x;
  const int dimension = threadIdx.x;
  const int query_slot = token_index * kQueryHeads + query_head;
  __shared__ float factor[kMaxSplits];
  __shared__ float denominator;
  if (dimension == 0) {
    float maximum = -INFINITY;
    for (int split = 0; split < split_count; ++split)
      maximum = fmaxf(maximum,
          partial_max[query_slot * split_count + split]);
    float sum = 0.0f;
    for (int split = 0; split < split_count; ++split) {
      const int partial = query_slot * split_count + split;
      factor[split] = expf(partial_max[partial] - maximum);
      sum += partial_sum[partial] * factor[split];
    }
    denominator = sum;
  }
  __syncthreads();
  float accumulator = 0.0f;
  for (int split = 0; split < split_count; ++split) {
    const int partial = query_slot * split_count + split;
    accumulator +=
        partial_accumulator[static_cast<int64_t>(partial) * kHeadDim +
                            dimension] * factor[split];
  }
  output[static_cast<int64_t>(query_slot) * kHeadDim + dimension] =
      __float2bfloat16(denominator > 0.0f ? accumulator / denominator : 0.0f);
}
