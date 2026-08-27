#pragma once

// Decode-only packed QKV prelude for Qwen2.5-Omni-3B.  This preserves the
// incumbent seam exactly: projection+bias is first rounded to BF16, then Q/K
// TMRoPE reads those rounded values, while V publishes the rounded value.
// Q is written in the attention consumer layout and K/V directly into their
// persistent head-major cache slot.
__global__ void qwen25_omni_qkv_bias_tmrope_kv_write_bf16_kernel(
    const __nv_bfloat16* packed_qkv, const __nv_bfloat16* bias,
    __nv_bfloat16* query, __nv_bfloat16* key_cache,
    __nv_bfloat16* value_cache, float theta, const uint32_t* positions,
    const uint32_t* cache_position) {
  constexpr int kQueryHeads = 16;
  constexpr int kKvHeads = 2;
  constexpr int kHeadDim = 128;
  constexpr int kQueryWidth = kQueryHeads * kHeadDim;
  constexpr int kKvWidth = kKvHeads * kHeadDim;
  constexpr int kFusedWidth = kQueryWidth + 2 * kKvWidth;
  constexpr int kMaxSeqLen = 32768;
  constexpr int kSectionT = 16;
  constexpr int kSectionH = 24;

  const int projection_head = blockIdx.x;
  const int dimension = threadIdx.x;
  if (projection_head >= kQueryHeads + 2 * kKvHeads ||
      dimension >= kHeadDim) {
    return;
  }

  if (projection_head >= kQueryHeads + kKvHeads) {
    const int head = projection_head - kQueryHeads - kKvHeads;
    const int source = kQueryWidth + kKvWidth + head * kHeadDim + dimension;
    const __nv_bfloat16 rounded = __float2bfloat16(
        __bfloat162float(packed_qkv[source]) + __bfloat162float(bias[source]));
    const int64_t destination =
        (static_cast<int64_t>(head) * kMaxSeqLen + *cache_position) *
            kHeadDim +
        dimension;
    value_cache[destination] = rounded;
    return;
  }

  const bool is_query = projection_head < kQueryHeads;
  const int head = is_query ? projection_head : projection_head - kQueryHeads;
  const int source_base =
      (is_query ? 0 : kQueryWidth) + head * kHeadDim;
  const int half = kHeadDim / 2;
  const int paired_dimension = dimension < half ? dimension + half
                                                 : dimension - half;
  const __nv_bfloat16 value_rounded = __float2bfloat16(
      __bfloat162float(packed_qkv[source_base + dimension]) +
      __bfloat162float(bias[source_base + dimension]));
  const __nv_bfloat16 pair_rounded = __float2bfloat16(
      __bfloat162float(packed_qkv[source_base + paired_dimension]) +
      __bfloat162float(bias[source_base + paired_dimension]));
  const int axis = dimension < 2 * kSectionT
      ? 0
      : (dimension < 2 * (kSectionT + kSectionH) ? 1 : 2);
  const int pair = dimension % half;
  const float frequency =
      1.0f / powf(theta, 2.0f * static_cast<float>(pair) / kHeadDim);
  const float angle = static_cast<float>(positions[axis]) * frequency;
  const float cosine = cosf(angle);
  const float sine = sinf(angle);
  const float value = __bfloat162float(value_rounded);
  const float rotated = dimension < half ? -__bfloat162float(pair_rounded)
                                         : __bfloat162float(pair_rounded);
  const __nv_bfloat16 output =
      __float2bfloat16(value * cosine + rotated * sine);

  if (is_query) {
    query[head * kHeadDim + dimension] = output;
  } else {
    const int64_t destination =
        (static_cast<int64_t>(head) * kMaxSeqLen + *cache_position) *
            kHeadDim +
        dimension;
    key_cache[destination] = output;
  }
}

// Prefill counterpart of the decode QKV prelude. The packed projection is
// already BF16. This kernel preserves the incumbent projection+bias BF16 seam,
// writes rotated Q directly in its attention layout, and publishes rotated K
// plus rounded V directly into the persistent head-major cache.
__global__ void qwen25_omni_prefill_qkv_bias_tmrope_kv_write_bf16_kernel(
    const __nv_bfloat16* packed_qkv, const __nv_bfloat16* bias,
    __nv_bfloat16* query, __nv_bfloat16* key_cache,
    __nv_bfloat16* value_cache, int rows, int start_position,
    const uint32_t* positions) {
  constexpr int kQueryHeads = 16;
  constexpr int kKvHeads = 2;
  constexpr int kHeadDim = 128;
  constexpr int kHalf = kHeadDim / 2;
  constexpr int kQueryWidth = kQueryHeads * kHeadDim;
  constexpr int kKvWidth = kKvHeads * kHeadDim;
  constexpr int kFusedWidth = kQueryWidth + 2 * kKvWidth;
  constexpr int kMaxSeqLen = 32768;
  constexpr int kSectionT = 16;
  constexpr int kSectionH = 24;
  constexpr float kTheta = 1000000.0f;

  const int projection_head = blockIdx.x;
  const int row = blockIdx.y;
  const int pair = threadIdx.x;
  if (projection_head >= kQueryHeads + 2 * kKvHeads || row >= rows ||
      pair >= kHalf) {
    return;
  }

  const int row_base = row * kFusedWidth;
  if (projection_head >= kQueryHeads + kKvHeads) {
    const int head = projection_head - kQueryHeads - kKvHeads;
    const int source_base = row_base + kQueryWidth + kKvWidth + head * kHeadDim;
    const int64_t destination_base =
        (static_cast<int64_t>(head) * kMaxSeqLen + start_position + row) *
        kHeadDim;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int dimension = pair + half * kHalf;
      value_cache[destination_base + dimension] = __float2bfloat16(
          __bfloat162float(packed_qkv[source_base + dimension]) +
          __bfloat162float(bias[kQueryWidth + kKvWidth +
                                head * kHeadDim + dimension]));
    }
    return;
  }

  const bool is_query = projection_head < kQueryHeads;
  const int head = is_query ? projection_head : projection_head - kQueryHeads;
  const int projection_offset =
      (is_query ? 0 : kQueryWidth) + head * kHeadDim;
  const int source_base = row_base + projection_offset;
  const __nv_bfloat16 first_rounded = __float2bfloat16(
      __bfloat162float(packed_qkv[source_base + pair]) +
      __bfloat162float(bias[projection_offset + pair]));
  const __nv_bfloat16 second_rounded = __float2bfloat16(
      __bfloat162float(packed_qkv[source_base + pair + kHalf]) +
      __bfloat162float(bias[projection_offset + pair + kHalf]));
  const float frequency =
      1.0f / powf(kTheta, 2.0f * static_cast<float>(pair) / kHeadDim);

#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const int dimension = pair + half * kHalf;
    const int axis = dimension < 2 * kSectionT
        ? 0
        : (dimension < 2 * (kSectionT + kSectionH) ? 1 : 2);
    const float angle =
        static_cast<float>(positions[row * 3 + axis]) * frequency;
    const float cosine = cosf(angle);
    const float sine = sinf(angle);
    const float value = __bfloat162float(
        half == 0 ? first_rounded : second_rounded);
    const float rotated = half == 0 ? -__bfloat162float(second_rounded)
                                    : __bfloat162float(first_rounded);
    const __nv_bfloat16 output =
        __float2bfloat16(value * cosine + rotated * sine);
    if (is_query) {
      query[(static_cast<int64_t>(row) * kQueryHeads + head) * kHeadDim +
            dimension] = output;
    } else {
      const int64_t destination =
          (static_cast<int64_t>(head) * kMaxSeqLen + start_position + row) *
              kHeadDim +
          dimension;
      key_cache[destination] = output;
    }
  }
}

// Long-context decode candidate.  The incumbent assigns one CTA to each Q
// head, which exposes only 24 CTAs on an RTX 4090.  This stage partitions the
// valid KV interval across CTAs and writes numerically stable online-softmax
// partials.  A second kernel merges those partials in FP32.  The extra GMEM
// traffic is bounded by heads * splits * (head_dim + 2) floats and is small
// compared with the long-context KV reads it enables the full GPU to service.
template <int kQueryHeads, int kKvHeads, int kHeadDim, int kWarps = 8>
__global__ void attention_flash_split_cta_bf16_kernel(
    const __nv_bfloat16* query, const __nv_bfloat16* key_cache,
    const __nv_bfloat16* value_cache, float* partial_max,
    float* partial_sum, float* partial_accumulator, int split_count,
    int bucket_kv_len, int max_seq_len, float scale,
    const uint32_t* position) {
  constexpr int kElementsPerThread = kHeadDim / 32;
  const int query_head = blockIdx.x;
  const int split = blockIdx.y;
  const int thread = threadIdx.x;
  const int warp = thread / 32;
  const int lane = thread & 31;
  const int kv_head = query_head / (kQueryHeads / kKvHeads);
  const int valid_len = min(static_cast<int>(*position) + 1, bucket_kv_len);
  const int span = (valid_len + split_count - 1) / split_count;
  const int begin = min(split * span, valid_len);
  const int end = min(begin + span, valid_len);
  const int partial = query_head * split_count + split;

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
          query[static_cast<int64_t>(query_head) * kHeadDim + dimension]);
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

// GQA ownership probe.  One CTA owns adjacent query heads that share a KV
// head, so each K/V element is loaded once and used by both online-softmax
// states.  Every query head retains its own token order, maxima, sums and
// accumulator; only the read ownership changes.
template <int kQueryHeads, int kKvHeads, int kHeadDim,
          int kGroupedQueryHeads = 2, int kWarps = 8>
__global__ void attention_flash_grouped_split_cta_bf16_kernel(
    const __nv_bfloat16* query, const __nv_bfloat16* key_cache,
    const __nv_bfloat16* value_cache, float* partial_max,
    float* partial_sum, float* partial_accumulator, int split_count,
    int bucket_kv_len, int max_seq_len, float scale,
    const uint32_t* position) {
  static_assert(kQueryHeads % kKvHeads == 0,
                "query heads must divide across KV heads");
  constexpr int kQueryHeadsPerKv = kQueryHeads / kKvHeads;
  static_assert(kQueryHeadsPerKv % kGroupedQueryHeads == 0,
                "grouped query heads must remain inside one KV owner");
  constexpr int kElementsPerThread = kHeadDim / 32;
  const int query_group = blockIdx.x;
  const int query_head_base = query_group * kGroupedQueryHeads;
  const int split = blockIdx.y;
  const int thread = threadIdx.x;
  const int warp = thread / 32;
  const int lane = thread & 31;
  const int kv_head = query_head_base / kQueryHeadsPerKv;
  const int valid_len = min(static_cast<int>(*position) + 1, bucket_kv_len);
  const int span = (valid_len + split_count - 1) / split_count;
  const int begin = min(split * span, valid_len);
  const int end = min(begin + span, valid_len);

  if (begin >= end) {
#pragma unroll
    for (int local_head = 0; local_head < kGroupedQueryHeads; ++local_head) {
      const int query_head = query_head_base + local_head;
      const int partial = query_head * split_count + split;
      if (thread == 0) {
        partial_max[partial] = -INFINITY;
        partial_sum[partial] = 0.0f;
      }
      if (thread < kHeadDim)
        partial_accumulator[static_cast<int64_t>(partial) * kHeadDim +
                            thread] = 0.0f;
    }
    return;
  }

  __shared__ float query_shared[kGroupedQueryHeads][kHeadDim];
  for (int index = thread; index < kGroupedQueryHeads * kHeadDim;
       index += blockDim.x) {
    const int local_head = index / kHeadDim;
    const int dimension = index - local_head * kHeadDim;
    query_shared[local_head][dimension] = __bfloat162float(
        query[static_cast<int64_t>(query_head_base + local_head) * kHeadDim +
              dimension]);
  }
  __syncthreads();

  float query_register[kGroupedQueryHeads][kElementsPerThread];
  float accumulator[kGroupedQueryHeads][kElementsPerThread];
  float maximum[kGroupedQueryHeads];
  float sum[kGroupedQueryHeads];
#pragma unroll
  for (int local_head = 0; local_head < kGroupedQueryHeads; ++local_head) {
    maximum[local_head] = -INFINITY;
    sum[local_head] = 0.0f;
#pragma unroll
    for (int item = 0; item < kElementsPerThread; ++item) {
      query_register[local_head][item] =
          query_shared[local_head][item * 32 + lane];
      accumulator[local_head][item] = 0.0f;
    }
  }
  const __nv_bfloat16* key_base =
      key_cache + static_cast<int64_t>(kv_head) * max_seq_len * kHeadDim;
  const __nv_bfloat16* value_base =
      value_cache + static_cast<int64_t>(kv_head) * max_seq_len * kHeadDim;
  for (int token = begin + warp; token < end; token += kWarps) {
    float dot[kGroupedQueryHeads] = {};
#pragma unroll
    for (int item = 0; item < kElementsPerThread; ++item) {
      const int dimension = item * 32 + lane;
      const float key_value = __bfloat162float(
          key_base[static_cast<int64_t>(token) * kHeadDim + dimension]);
#pragma unroll
      for (int local_head = 0; local_head < kGroupedQueryHeads; ++local_head)
        dot[local_head] += query_register[local_head][item] * key_value;
    }
#pragma unroll
    for (int local_head = 0; local_head < kGroupedQueryHeads; ++local_head) {
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1)
        dot[local_head] +=
            __shfl_xor_sync(0xffffffff, dot[local_head], offset);
      dot[local_head] *= scale;
    }
    float probability[kGroupedQueryHeads];
    float previous_scale[kGroupedQueryHeads];
#pragma unroll
    for (int local_head = 0; local_head < kGroupedQueryHeads; ++local_head) {
      const float next_maximum = fmaxf(maximum[local_head], dot[local_head]);
      probability[local_head] = expf(dot[local_head] - next_maximum);
      previous_scale[local_head] =
          expf(maximum[local_head] - next_maximum);
      sum[local_head] = sum[local_head] * previous_scale[local_head] +
          probability[local_head];
      maximum[local_head] = next_maximum;
    }
#pragma unroll
    for (int item = 0; item < kElementsPerThread; ++item) {
      const int dimension = item * 32 + lane;
      const float value = __bfloat162float(
          value_base[static_cast<int64_t>(token) * kHeadDim + dimension]);
#pragma unroll
      for (int local_head = 0; local_head < kGroupedQueryHeads; ++local_head)
        accumulator[local_head][item] =
            accumulator[local_head][item] * previous_scale[local_head] +
            probability[local_head] * value;
    }
  }

  __shared__ float warp_maximum[kGroupedQueryHeads][kWarps];
  __shared__ float warp_sum[kGroupedQueryHeads][kWarps];
  __shared__ float
      warp_accumulator[kGroupedQueryHeads][kWarps][kHeadDim];
#pragma unroll
  for (int local_head = 0; local_head < kGroupedQueryHeads; ++local_head) {
    if (lane == 0) {
      warp_maximum[local_head][warp] = maximum[local_head];
      warp_sum[local_head][warp] = sum[local_head];
    }
#pragma unroll
    for (int item = 0; item < kElementsPerThread; ++item)
      warp_accumulator[local_head][warp][item * 32 + lane] =
          accumulator[local_head][item];
  }
  __syncthreads();

  if (warp < kGroupedQueryHeads) {
    const int local_head = warp;
    const int query_head = query_head_base + local_head;
    const int partial = query_head * split_count + split;
    float merged_maximum = -INFINITY;
#pragma unroll
    for (int source = 0; source < kWarps; ++source)
      merged_maximum =
          fmaxf(merged_maximum, warp_maximum[local_head][source]);
    float merged_sum = 0.0f;
    float merged_accumulator[kElementsPerThread];
#pragma unroll
    for (int item = 0; item < kElementsPerThread; ++item)
      merged_accumulator[item] = 0.0f;
#pragma unroll
    for (int source = 0; source < kWarps; ++source) {
      const float factor =
          expf(warp_maximum[local_head][source] - merged_maximum);
      merged_sum += warp_sum[local_head][source] * factor;
#pragma unroll
      for (int item = 0; item < kElementsPerThread; ++item)
        merged_accumulator[item] +=
            warp_accumulator[local_head][source][item * 32 + lane] * factor;
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

template <int kHeadDim, int kMaxSplits = 16>
__global__ void attention_flash_split_cta_reduce_bf16_kernel(
    const float* partial_max, const float* partial_sum,
    const float* partial_accumulator, __nv_bfloat16* output,
    int split_count) {
  const int query_head = blockIdx.x;
  const int dimension = threadIdx.x;
  __shared__ float factor[kMaxSplits];
  __shared__ float denominator;
  if (dimension == 0) {
    float maximum = -INFINITY;
    for (int split = 0; split < split_count; ++split)
      maximum = fmaxf(maximum,
          partial_max[query_head * split_count + split]);
    float sum = 0.0f;
    for (int split = 0; split < split_count; ++split) {
      const int partial = query_head * split_count + split;
      factor[split] = expf(partial_max[partial] - maximum);
      sum += partial_sum[partial] * factor[split];
    }
    denominator = sum;
  }
  __syncthreads();
  if (dimension >= kHeadDim) return;
  float accumulator = 0.0f;
  for (int split = 0; split < split_count; ++split) {
    const int partial = query_head * split_count + split;
    accumulator +=
        partial_accumulator[static_cast<int64_t>(partial) * kHeadDim +
                            dimension] * factor[split];
  }
  output[static_cast<int64_t>(query_head) * kHeadDim + dimension] =
      __float2bfloat16(denominator > 0.0f ? accumulator / denominator : 0.0f);
}
