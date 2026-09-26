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

  if (valid_keys == 0) {
    for (int key_token = 0; key_token < key_tokens; ++key_token) {
      scores[key_token] = 0.0F;
    }
    return;
  }

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
                             cudaStream_t stream) {
  constexpr int threads = 128;
  const int64_t rows =
      static_cast<int64_t>(batch) * query_tokens * query_heads;
  attention_scores_softmax<T><<<
      static_cast<unsigned>((rows + threads - 1) / threads), threads, 0,
      stream>>>(static_cast<const T*>(query), static_cast<const T*>(key),
                probabilities, query_tokens, key_tokens, key_stride,
                query_heads, kv_heads, head_dim, batch, causal, query_start,
                decode_meta, scale);
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
