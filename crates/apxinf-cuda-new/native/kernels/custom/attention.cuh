#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace apxinf::attention::kernels {

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
__device__ inline float from_float(float value) { return value; }

template <>
__device__ inline __half from_float(float value) {
  return __float2half(value);
}

template <>
__device__ inline __nv_bfloat16 from_float(float value) {
  return __float2bfloat16(value);
}

// Both attention_scores_softmax and attention_values derive the row length
// from this helper, so `decode_meta` is read by two separate graph nodes and
// correctness needs both reads to agree. The values read is strictly later in
// stream order and can only observe a length >= the softmax read; the softmax
// kernel zeroed the tail weights, so a longer values loop is benign. This
// holds only under the KvCacheDecodeMeta::update contract (synchronize before
// updating) plus a monotonically non-decreasing valid length — a decrease
// published mid-replay would truncate the value accumulation.
__device__ inline int runtime_valid_keys(int key_tokens, bool causal,
                                         int query_start, int query_token,
                                         const uint32_t* decode_meta) {
  uint32_t available = static_cast<uint32_t>(key_tokens);
  uint32_t start = static_cast<uint32_t>(query_start);
  if (decode_meta != nullptr) {
    available = min(available, decode_meta[0]);
    start = decode_meta[1];
  }
  if (!causal) return static_cast<int>(available);
  const uint64_t causal_end = static_cast<uint64_t>(start) + query_token + 1;
  return static_cast<int>(min(static_cast<uint64_t>(available), causal_end));
}

// Correctness fallback. One thread owns one softmax row, keeping the
// implementation shape-general while optimized providers handle throughput.
template <class T>
__global__ void attention_scores_softmax(
    const T* query, const T* key, float* probabilities, int query_tokens,
    int key_tokens, int key_stride, int query_heads, int kv_heads,
    int head_dim, int batch_size, bool causal, int query_start,
    const uint32_t* decode_meta, float scale) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t rows =
      static_cast<int64_t>(batch_size) * query_tokens * query_heads;
  if (row >= rows) return;

  const int q_head = row % query_heads;
  const int query_token = (row / query_heads) % query_tokens;
  const int batch = row / (static_cast<int64_t>(query_heads) * query_tokens);
  const int kv_head = q_head / (query_heads / kv_heads);
  const int valid_keys = runtime_valid_keys(
      key_tokens, causal, query_start, query_token, decode_meta);
  const T* q = query +
      ((static_cast<int64_t>(batch) * query_tokens + query_token) * query_heads +
       q_head) * head_dim;
  float* scores = probabilities + row * key_tokens;

  float maximum = -CUDART_INF_F;
  for (int key_token = 0; key_token < valid_keys; ++key_token) {
    const T* k = key +
        ((static_cast<int64_t>(batch) * key_stride + key_token) * kv_heads +
         kv_head) * head_dim;
    float dot = 0.0F;
    for (int dimension = 0; dimension < head_dim; ++dimension) {
      dot += to_float(q[dimension]) * to_float(k[dimension]);
    }
    scores[key_token] = dot * scale;
    maximum = fmaxf(maximum, scores[key_token]);
  }
  float sum = 0.0F;
  for (int key_token = 0; key_token < valid_keys; ++key_token) {
    const float value = expf(scores[key_token] - maximum);
    scores[key_token] = value;
    sum += value;
  }
  for (int key_token = 0; key_token < valid_keys; ++key_token) {
    scores[key_token] /= sum;
  }
  for (int key_token = valid_keys; key_token < key_tokens; ++key_token) {
    scores[key_token] = 0.0F;
  }
}

template <class T>
__global__ void attention_scores_softmax_additive(
    const T* query, const T* key, float* probabilities, int query_tokens,
    int key_tokens, int key_stride, int query_heads, int kv_heads,
    int head_dim, int batch_size, bool causal, int query_start,
    const uint32_t* decode_meta, float scale, const float* bias,
    bool bias_broadcast_batch) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t rows =
      static_cast<int64_t>(batch_size) * query_tokens * query_heads;
  if (row >= rows) return;

  const int q_head = row % query_heads;
  const int query_token = (row / query_heads) % query_tokens;
  const int batch = row / (static_cast<int64_t>(query_heads) * query_tokens);
  const int kv_head = q_head / (query_heads / kv_heads);
  const int valid_keys = runtime_valid_keys(
      key_tokens, causal, query_start, query_token, decode_meta);
  const T* q = query +
      ((static_cast<int64_t>(batch) * query_tokens + query_token) * query_heads +
       q_head) * head_dim;
  float* scores = probabilities + row * key_tokens;

  float maximum = -CUDART_INF_F;
  for (int key_token = 0; key_token < valid_keys; ++key_token) {
    const T* k = key +
        ((static_cast<int64_t>(batch) * key_stride + key_token) * kv_heads +
         kv_head) * head_dim;
    float dot = 0.0F;
    for (int dimension = 0; dimension < head_dim; ++dimension) {
      dot += to_float(q[dimension]) * to_float(k[dimension]);
    }
    float score = dot * scale;

    if (bias != nullptr) {
      const int bias_batch = bias_broadcast_batch ? 0 : batch;
      score += bias[((static_cast<int64_t>(bias_batch) * query_heads + q_head) *
                     query_tokens + query_token) * key_tokens + key_token];

    }
    scores[key_token] = score;
    maximum = fmaxf(maximum, scores[key_token]);
  }
  float sum = 0.0F;
  for (int key_token = 0; key_token < valid_keys; ++key_token) {
    const float value = maximum == -CUDART_INF_F ? 0.0F : expf(scores[key_token] - maximum);
    scores[key_token] = value;
    sum += value;
  }
  for (int key_token = 0; key_token < valid_keys; ++key_token) {
    scores[key_token] = sum == 0.0F ? 0.0F : scores[key_token] / sum;

  }
  for (int key_token = valid_keys; key_token < key_tokens; ++key_token) {
    scores[key_token] = 0.0F;
  }
}

template <class T>
__global__ void attention_values(const float* probabilities, const T* value,
                                 T* output, int query_tokens, int key_tokens,
                                 int key_stride, int query_heads, int kv_heads,
                                 int head_dim, int batch_size, bool causal,
                                 int query_start,
                                 const uint32_t* decode_meta) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t elements = static_cast<int64_t>(batch_size) * query_tokens *
                           query_heads * head_dim;
  if (index >= elements) return;

  const int dimension = index % head_dim;
  const int q_head = (index / head_dim) % query_heads;
  const int query_token =
      (index / (static_cast<int64_t>(head_dim) * query_heads)) % query_tokens;
  const int batch = index /
      (static_cast<int64_t>(head_dim) * query_heads * query_tokens);
  const int kv_head = q_head / (query_heads / kv_heads);
  const int valid_keys = runtime_valid_keys(
      key_tokens, causal, query_start, query_token, decode_meta);
  const int64_t row =
      (static_cast<int64_t>(batch) * query_tokens + query_token) * query_heads +
      q_head;
  const float* weights = probabilities + row * key_tokens;
  float result = 0.0F;
  for (int key_token = 0; key_token < valid_keys; ++key_token) {
    const T* v = value +
        ((static_cast<int64_t>(batch) * key_stride + key_token) * kv_heads +
         kv_head) * head_dim;
    result += weights[key_token] * to_float(v[dimension]);
  }
  output[index] = from_float<T>(result);
}

template <class T>
cudaError_t launch_attention(const void* query, const void* key,
                             const void* value, void* output,
                             float* probabilities, int batch,
                             int query_tokens, int key_tokens, int key_stride,
                             int query_heads, int kv_heads, int head_dim,
                             bool causal, int query_start,
                             const uint32_t* decode_meta, float scale,
                             cudaStream_t stream, const float* bias = nullptr,
                             bool bias_broadcast_batch = false) {
  constexpr int threads = 128;
  const int64_t rows =
      static_cast<int64_t>(batch) * query_tokens * query_heads;
  if (bias == nullptr) {
  attention_scores_softmax<T><<<
      static_cast<unsigned>((rows + threads - 1) / threads), threads, 0,
      stream>>>(static_cast<const T*>(query), static_cast<const T*>(key),
                probabilities, query_tokens, key_tokens, key_stride,
                query_heads, kv_heads, head_dim, batch, causal, query_start,
                decode_meta, scale);
  } else {
  attention_scores_softmax_additive<T><<<
      static_cast<unsigned>((rows + threads - 1) / threads), threads, 0,
      stream>>>(static_cast<const T*>(query), static_cast<const T*>(key),
                probabilities, query_tokens, key_tokens, key_stride,
                query_heads, kv_heads, head_dim, batch, causal, query_start,
                decode_meta, scale, bias, bias_broadcast_batch);
  }
  auto status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  const int64_t elements = rows * head_dim;
  attention_values<T><<<
      static_cast<unsigned>((elements + threads - 1) / threads), threads, 0,
      stream>>>(probabilities, static_cast<const T*>(value),
                static_cast<T*>(output), query_tokens, key_tokens, key_stride,
                query_heads, kv_heads, head_dim, batch, causal, query_start,
                decode_meta);
  return cudaGetLastError();
}

// All 128 threads participate; the final barrier protects scratch reuse.
template <bool Maximum>
__device__ __forceinline__ float attention_block_reduce(float value, float* scratch) {
  for (int offset = 16; offset > 0; offset /= 2) {
    const float other = __shfl_down_sync(0xffffffff, value, offset);
    value = Maximum ? fmaxf(value, other) : value + other;
  }
  if (threadIdx.x % 32 == 0) scratch[threadIdx.x / 32] = value;
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = scratch[0];
    for (int warp = 1; warp < 4; ++warp)
      total = Maximum ? fmaxf(total, scratch[warp]) : total + scratch[warp];
    scratch[0] = total;
  }
  __syncthreads();
  const float result = scratch[0];
  __syncthreads();
  return result;
}

// One block owns a row. The cooperative variant distributes each QK dot over
// a warp and reduces softmax across the block; the baseline uses serial dots
// and softmax. Global scratch is prepared before capture. Shared memory has a
// fixed size, so arbitrary positive head dimensions remain supported.
template <class T, bool Cooperative>
__global__ void attention_parallel_kernel(
    const T* query, const T* key, const T* value, T* output,
    float* probabilities, int query_tokens, int key_tokens, int query_heads,
    int kv_heads, int head_dim, float scale, bool causal, int query_start,
    const float* bias, bool bias_broadcast_batch) {
  const int64_t row = blockIdx.x;
  const int q_head = row % query_heads;
  const int query_token = (row / query_heads) % query_tokens;
  const int batch = row / (static_cast<int64_t>(query_heads) * query_tokens);
  const int kv_head = q_head / (query_heads / kv_heads);
  const int valid_keys = causal ? min(key_tokens, query_start + query_token + 1) : key_tokens;
  const T* q = query + row * head_dim;
  float* scores = probabilities + row * key_tokens;
  const int lane = threadIdx.x % 32;
  const int first_key = Cooperative ? threadIdx.x / 32 : threadIdx.x;
  const int key_step = Cooperative ? blockDim.x / 32 : blockDim.x;
  for (int token = first_key; token < key_tokens; token += key_step) {
    float score = -CUDART_INF_F;
    if (token < valid_keys) {
      const T* k = key + ((static_cast<int64_t>(batch) * key_tokens + token) * kv_heads + kv_head) * head_dim;
      float dot = 0.0F;
      for (int d = Cooperative ? lane : 0; d < head_dim; d += Cooperative ? 32 : 1)
        dot += to_float(q[d]) * to_float(k[d]);
      if constexpr (Cooperative) {
        for (int offset = 16; offset > 0; offset /= 2)
          dot += __shfl_down_sync(0xffffffff, dot, offset);
      }
      score = dot * scale;

      if (bias != nullptr) {
        const int bias_batch = bias_broadcast_batch ? 0 : batch;
        score += bias[((static_cast<int64_t>(bias_batch) * query_heads + q_head) * query_tokens + query_token) * key_tokens + token];

      }
    }
    if (!Cooperative || lane == 0) scores[token] = score;
  }
  __syncthreads();
  if constexpr (Cooperative) {
    __shared__ float reduction[4];
    float local_max = -CUDART_INF_F;
    for (int token = threadIdx.x; token < valid_keys; token += blockDim.x)
      local_max = fmaxf(local_max, scores[token]);
    const float maximum = attention_block_reduce<true>(local_max, reduction);
    float local_sum = 0.0F;
    for (int token = threadIdx.x; token < key_tokens; token += blockDim.x) {
      const float p = maximum == -CUDART_INF_F ? 0.0F : expf(scores[token] - maximum);
      scores[token] = p;
      local_sum += p;
    }
    const float sum = attention_block_reduce<false>(local_sum, reduction);
    for (int token = threadIdx.x; token < key_tokens; token += blockDim.x) {
      float p = sum == 0.0F ? 0.0F : scores[token] / sum;

      scores[token] = p;
    }
  } else if (threadIdx.x == 0) {
    float maximum = -CUDART_INF_F;
    for (int token = 0; token < valid_keys; ++token) maximum = fmaxf(maximum, scores[token]);
    float sum = 0.0F;
    for (int token = 0; token < key_tokens; ++token) {
      const float p = maximum == -CUDART_INF_F ? 0.0F : expf(scores[token] - maximum);
      scores[token] = p;
      sum += p;
    }
    for (int token = 0; token < key_tokens; ++token) {
      float p = sum == 0.0F ? 0.0F : scores[token] / sum;

      scores[token] = p;
    }
  }
  __syncthreads();
  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float result = 0.0F;
    for (int token = 0; token < valid_keys; ++token) {
      const T* v = value + ((static_cast<int64_t>(batch) * key_tokens + token) * kv_heads + kv_head) * head_dim;
      result += scores[token] * to_float(v[d]);
    }
    output[row * head_dim + d] = from_float<T>(result);
  }
}

template <class T, bool Cooperative = false>
cudaError_t launch_parallel_attention(const void* query, const void* key,
    const void* value, void* output, float* probabilities, int batch,
    int query_tokens, int key_tokens, int query_heads, int kv_heads,
    int head_dim, float scale, bool causal, int query_start, const float* bias,
    bool bias_broadcast_batch, cudaStream_t stream) {
  const uint64_t rows = static_cast<uint64_t>(batch) * query_tokens * query_heads;
  attention_parallel_kernel<T, Cooperative><<<static_cast<unsigned>(rows), 128, 0, stream>>>(
      static_cast<const T*>(query), static_cast<const T*>(key),
      static_cast<const T*>(value), static_cast<T*>(output), probabilities,
      query_tokens, key_tokens, query_heads, kv_heads, head_dim, scale, causal,
      query_start, bias, bias_broadcast_batch);
  return cudaGetLastError();
}

// One CUDA block computes one (segment, query, head) output row. This is the
// shape-general L0 fallback for the segmented L3 semantic; offsets remain a
// device input and all segments are processed by one launch.
template <class T>
__global__ void segmented_attention_kernel(
    const T* query, const T* key, const T* value, const uint32_t* offsets,
    T* output, int total_tokens, int segments, int max_segment_tokens,
    int heads, int head_dim, float scale) {
  extern __shared__ float scores[];
  const int segment = blockIdx.z;
  const int local_query = blockIdx.x;
  const int head = blockIdx.y;
  if (segment >= segments) return;
  const int begin = static_cast<int>(offsets[segment]);
  const int end = static_cast<int>(offsets[segment + 1]);
  if (begin < 0 || end < begin || end > total_tokens ||
      end - begin > max_segment_tokens)
    return;
  const int tokens = end - begin;
  if (local_query >= tokens) return;
  const int global_query = begin + local_query;
  const T* q = query +
      (static_cast<int64_t>(global_query) * heads + head) * head_dim;

  for (int key_token = threadIdx.x; key_token < tokens;
       key_token += blockDim.x) {
    const T* k = key +
        (static_cast<int64_t>(begin + key_token) * heads + head) * head_dim;
    float dot = 0.0F;
    for (int dimension = 0; dimension < head_dim; ++dimension) {
      dot += to_float(q[dimension]) * to_float(k[dimension]);
    }
    scores[key_token] = dot * scale;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float maximum = -CUDART_INF_F;
    for (int key_token = 0; key_token < tokens; ++key_token)
      maximum = fmaxf(maximum, scores[key_token]);
    float sum = 0.0F;
    for (int key_token = 0; key_token < tokens; ++key_token) {
      scores[key_token] = expf(scores[key_token] - maximum);
      sum += scores[key_token];
    }
    for (int key_token = 0; key_token < tokens; ++key_token)
      scores[key_token] /= sum;
  }
  __syncthreads();
  for (int dimension = threadIdx.x; dimension < head_dim;
       dimension += blockDim.x) {
    float result = 0.0F;
    for (int key_token = 0; key_token < tokens; ++key_token) {
      const T* v = value +
          (static_cast<int64_t>(begin + key_token) * heads + head) * head_dim;
      result += scores[key_token] * to_float(v[dimension]);
    }
    output[(static_cast<int64_t>(global_query) * heads + head) * head_dim +
           dimension] = from_float<T>(result);
  }
}

template <class T>
cudaError_t launch_segmented_attention(
    const void* query, const void* key, const void* value,
    const void* offsets, void* output, int segments, int max_segment_tokens,
    int total_tokens, int heads, int head_dim, float scale,
    cudaStream_t stream) {
  constexpr int threads = 128;
  segmented_attention_kernel<T><<<
      dim3(max_segment_tokens, heads, segments), threads,
      static_cast<size_t>(max_segment_tokens) * sizeof(float), stream>>>(
      static_cast<const T*>(query), static_cast<const T*>(key),
      static_cast<const T*>(value), static_cast<const uint32_t*>(offsets),
      static_cast<T*>(output), total_tokens, segments, max_segment_tokens,
      heads, head_dim, scale);
  return cudaGetLastError();
}

}  // namespace apxinf::attention::kernels
