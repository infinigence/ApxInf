// Copyright 2026 apxinf contributors.
// Stable C ABI and CUDA launch adapter for custom static-inference operators.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {
#include "../kernels/custom/math.cuh"
#include "../kernels/custom/reduction.cuh"
#include "../kernels/custom/quantization.cuh"
#include "../kernels/custom/w4a16.cuh"
#include "../kernels/custom/w8a16.cuh"
#include "../kernels/custom/qwen35_common.cuh"
#include "../kernels/custom/qwen35_gdn.cuh"
#include "../kernels/custom/qwen35_attention.cuh"
#include "../kernels/custom/preprocess.cuh"
#include "../kernels/custom/attention.cuh"
#include "../kernels/custom/normalization.cuh"
#include "../kernels/custom/activation.cuh"
#include "../kernels/custom/embedding.cuh"
#include "../kernels/custom/elementwise.cuh"
#include "../kernels/custom/fused.cuh"
#include "../kernels/custom/cache.cuh"
}  // namespace

extern "C" cudaError_t apxinf_static_evict_l2(
    void* buffer, size_t bytes, uint32_t seed, cudaStream_t stream) {
  if (buffer == nullptr || bytes < sizeof(uint32_t) ||
      bytes % sizeof(uint32_t) != 0) {
    return cudaErrorInvalidValue;
  }
  constexpr int threads = 256;
  int blocks = static_cast<int>((bytes / sizeof(uint32_t) + threads - 1) /
                                threads);
  blocks = blocks > 4096 ? 4096 : blocks;
  l2_cache_evict_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<volatile uint32_t*>(buffer), bytes / sizeof(uint32_t), seed);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_qwen35_rmsnorm_offset_bf16(
    const void* input, const void* weight, void* output, int rows, int columns,
    float epsilon, cudaStream_t stream) {
  if (input == nullptr || weight == nullptr || output == nullptr ||
      rows < 1 || rows > 8 || columns != 5120 || !(epsilon > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  qwen35_rmsnorm_offset_bf16_kernel<<<rows, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<__nv_bfloat16*>(output), columns, epsilon);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_qwen35_residual_add_rmsnorm_offset_bf16(
    void* residual, const void* delta, const void* weight, void* output,
    int rows, int columns, float epsilon, cudaStream_t stream) {
  if (residual == nullptr || delta == nullptr || weight == nullptr ||
      output == nullptr || rows < 1 || rows > 64 || columns != 5120 ||
      !(epsilon > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  const size_t shared_bytes = static_cast<size_t>(columns) * sizeof(float);
  qwen35_residual_add_rmsnorm_offset_bf16_kernel<<<
      rows, 256, shared_bytes, stream>>>(
      static_cast<__nv_bfloat16*>(residual),
      static_cast<const __nv_bfloat16*>(delta),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<__nv_bfloat16*>(output), columns, epsilon);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_qwen35_attention_prepare_bf16(
    const void* q_projection, const void* k_projection,
    const void* v_projection, const void* q_norm_weight,
    const void* k_norm_weight, void* query, void* key, void* value,
    void* gate, const void* position, cudaStream_t stream) {
  if (q_projection == nullptr || k_projection == nullptr ||
      v_projection == nullptr || q_norm_weight == nullptr ||
      k_norm_weight == nullptr || query == nullptr || key == nullptr ||
      value == nullptr || gate == nullptr || position == nullptr) {
    return cudaErrorInvalidValue;
  }
  qwen35_attention_prepare_bf16_kernel<<<28, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(q_projection),
      static_cast<const __nv_bfloat16*>(k_projection),
      static_cast<const __nv_bfloat16*>(v_projection),
      static_cast<const __nv_bfloat16*>(q_norm_weight),
      static_cast<const __nv_bfloat16*>(k_norm_weight),
      static_cast<__nv_bfloat16*>(query),
      static_cast<__nv_bfloat16*>(key),
      static_cast<__nv_bfloat16*>(value),
      static_cast<__nv_bfloat16*>(gate),
      static_cast<const uint32_t*>(position));
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_qwen35_attention_prepare_m8_bf16(
    const void* q_projection, const void* k_projection,
    const void* v_projection, const void* q_norm_weight,
    const void* k_norm_weight, void* query, void* key, void* value,
    void* gate, const void* positions, int tokens, cudaStream_t stream) {
  if (q_projection == nullptr || k_projection == nullptr ||
      v_projection == nullptr || q_norm_weight == nullptr ||
      k_norm_weight == nullptr || query == nullptr || key == nullptr ||
      value == nullptr || gate == nullptr || positions == nullptr ||
      tokens < 1 || tokens > 8) {
    return cudaErrorInvalidValue;
  }
  qwen35_attention_prepare_m8_bf16_kernel<<<tokens * 28, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(q_projection),
      static_cast<const __nv_bfloat16*>(k_projection),
      static_cast<const __nv_bfloat16*>(v_projection),
      static_cast<const __nv_bfloat16*>(q_norm_weight),
      static_cast<const __nv_bfloat16*>(k_norm_weight),
      static_cast<__nv_bfloat16*>(query),
      static_cast<__nv_bfloat16*>(key),
      static_cast<__nv_bfloat16*>(value),
      static_cast<__nv_bfloat16*>(gate),
      static_cast<const uint32_t*>(positions), tokens);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_qwen35_attention_gate_bf16(
    const void* input, const void* gate, void* output, int count,
    cudaStream_t stream) {
  if (input == nullptr || gate == nullptr || output == nullptr ||
      count < 6144 || count > 8 * 6144 || count % 6144 != 0)
    return cudaErrorInvalidValue;
  qwen35_attention_gate_bf16_kernel<<<(count + 255) / 256, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(gate),
      static_cast<__nv_bfloat16*>(output), count);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_qwen35_kv_quantize_bf16_e4m3(
    const void* input, void* output, void* scales, uint32_t rows,
    uint32_t head_dim, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || scales == nullptr ||
      rows == 0 || head_dim == 0 || head_dim > 1024) {
    return cudaErrorInvalidValue;
  }
  kv_cache_quantize_bf16_e4m3_kernel<<<rows, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<__nv_fp8_e4m3*>(output), static_cast<float*>(scales),
      rows, head_dim);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_qwen35_kv_append_bf16_e4m3(
    void* cache, void* scales, const void* new_data, uint32_t n_kv_heads,
    uint32_t head_dim, uint32_t max_seq_len, uint32_t seq_len,
    uint32_t append_len, cudaStream_t stream) {
  if (cache == nullptr || scales == nullptr || new_data == nullptr ||
      n_kv_heads == 0 || head_dim == 0 || head_dim > 1024 ||
      append_len == 0 || seq_len > max_seq_len ||
      append_len > max_seq_len - seq_len) {
    return cudaErrorInvalidValue;
  }
  dim3 grid(n_kv_heads, append_len, 1);
  kv_cache_append_bf16_e4m3_kernel<<<grid, 256, 0, stream>>>(
      static_cast<__nv_fp8_e4m3*>(cache), static_cast<float*>(scales),
      static_cast<const __nv_bfloat16*>(new_data), n_kv_heads, head_dim,
      max_seq_len, seq_len, append_len);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_qwen35_kv_append_decode_bf16_e4m3(
    void* cache, void* scales, const void* new_data, uint32_t n_kv_heads,
    uint32_t head_dim, uint32_t max_seq_len, const void* position,
    cudaStream_t stream) {
  if (cache == nullptr || scales == nullptr || new_data == nullptr ||
      position == nullptr || n_kv_heads == 0 || head_dim == 0 ||
      head_dim > 1024 || max_seq_len == 0) {
    return cudaErrorInvalidValue;
  }
  kv_cache_append_decode_bf16_e4m3_kernel<<<n_kv_heads, 256, 0, stream>>>(
      static_cast<__nv_fp8_e4m3*>(cache), static_cast<float*>(scales),
      static_cast<const __nv_bfloat16*>(new_data), n_kv_heads, head_dim,
      max_seq_len, static_cast<const uint32_t*>(position));
  return cudaGetLastError();
}

extern "C" cudaError_t
apxinf_static_qwen35_attention_flash_split_cta_e4m3(
    const void* query, const void* key_cache, const void* key_scales,
    const void* value_cache, const void* value_scales, void* partial_max,
    void* partial_sum, void* partial_accumulator, void* output,
    int split_count, int bucket_kv_len, int max_seq_len, float scale,
    const void* position, cudaStream_t stream) {
  if (query == nullptr || key_cache == nullptr || key_scales == nullptr ||
      value_cache == nullptr || value_scales == nullptr ||
      partial_max == nullptr || partial_sum == nullptr ||
      partial_accumulator == nullptr || output == nullptr ||
      position == nullptr || split_count < 2 || split_count > 16 ||
      (split_count & (split_count - 1)) != 0 || bucket_kv_len <= 0 ||
      bucket_kv_len > max_seq_len || !(scale > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  dim3 stage_grid(24, split_count, 1);
  qwen35_attention_flash_split_cta_e4m3_kernel<<<
      stage_grid, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(query),
      static_cast<const __nv_fp8_e4m3*>(key_cache),
      static_cast<const float*>(key_scales),
      static_cast<const __nv_fp8_e4m3*>(value_cache),
      static_cast<const float*>(value_scales),
      static_cast<float*>(partial_max), static_cast<float*>(partial_sum),
      static_cast<float*>(partial_accumulator), split_count, bucket_kv_len,
      max_seq_len, scale, static_cast<const uint32_t*>(position), 1);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  dim3 reduce_grid(24, 1, 1);
  qwen35_attention_flash_split_cta_reduce_bf16_kernel<<<
      reduce_grid, 256, 0, stream>>>(
      static_cast<const float*>(partial_max),
      static_cast<const float*>(partial_sum),
      static_cast<const float*>(partial_accumulator),
      static_cast<__nv_bfloat16*>(output), split_count, 1);
  return cudaGetLastError();
}

extern "C" cudaError_t
apxinf_static_qwen35_attention_flash_split_cta_m8_e4m3(
    const void* query, const void* key_cache, const void* key_scales,
    const void* value_cache, const void* value_scales, void* partial_max,
    void* partial_sum, void* partial_accumulator, void* output,
    int split_count, int bucket_kv_len, int max_seq_len, float scale,
    const void* positions, int tokens, cudaStream_t stream) {
  if (query == nullptr || key_cache == nullptr || key_scales == nullptr ||
      value_cache == nullptr || value_scales == nullptr ||
      partial_max == nullptr || partial_sum == nullptr ||
      partial_accumulator == nullptr || output == nullptr ||
      positions == nullptr || tokens < 1 || tokens > 8 ||
      split_count < 2 || split_count > 16 ||
      (split_count & (split_count - 1)) != 0 || bucket_kv_len <= 0 ||
      bucket_kv_len > max_seq_len || !(scale > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  dim3 stage_grid(24, split_count, tokens);
  qwen35_attention_flash_split_cta_e4m3_kernel<<<
      stage_grid, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(query),
      static_cast<const __nv_fp8_e4m3*>(key_cache),
      static_cast<const float*>(key_scales),
      static_cast<const __nv_fp8_e4m3*>(value_cache),
      static_cast<const float*>(value_scales),
      static_cast<float*>(partial_max), static_cast<float*>(partial_sum),
      static_cast<float*>(partial_accumulator), split_count, bucket_kv_len,
      max_seq_len, scale, static_cast<const uint32_t*>(positions), tokens);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  dim3 reduce_grid(24, tokens, 1);
  qwen35_attention_flash_split_cta_reduce_bf16_kernel<<<
      reduce_grid, 256, 0, stream>>>(
      static_cast<const float*>(partial_max),
      static_cast<const float*>(partial_sum),
      static_cast<const float*>(partial_accumulator),
      static_cast<__nv_bfloat16*>(output), split_count, tokens);
  return cudaGetLastError();
}

extern "C" cudaError_t
apxinf_static_qwen35_attention_flash_split_cta_m8_shared_q4_e4m3(
    const void* query, const void* key_cache, const void* key_scales,
    const void* value_cache, const void* value_scales, void* partial_max,
    void* partial_sum, void* partial_accumulator, void* output,
    int split_count, int bucket_kv_len, int max_seq_len, float scale,
    const void* positions, int tokens, cudaStream_t stream) {
  if (query == nullptr || key_cache == nullptr || key_scales == nullptr ||
      value_cache == nullptr || value_scales == nullptr ||
      partial_max == nullptr || partial_sum == nullptr ||
      partial_accumulator == nullptr || output == nullptr ||
      positions == nullptr || tokens != 8 || split_count != 16 ||
      bucket_kv_len <= 0 || bucket_kv_len > max_seq_len ||
      !(scale > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  dim3 stage_grid(24, split_count, 2);
  qwen35_attention_flash_split_cta_m8_shared_q4_e4m3_kernel<<<
      stage_grid, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(query),
      static_cast<const __nv_fp8_e4m3*>(key_cache),
      static_cast<const float*>(key_scales),
      static_cast<const __nv_fp8_e4m3*>(value_cache),
      static_cast<const float*>(value_scales),
      static_cast<float*>(partial_max), static_cast<float*>(partial_sum),
      static_cast<float*>(partial_accumulator), split_count, bucket_kv_len,
      max_seq_len, scale, static_cast<const uint32_t*>(positions));
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  dim3 reduce_grid(24, tokens, 1);
  qwen35_attention_flash_split_cta_reduce_bf16_kernel<<<
      reduce_grid, 256, 0, stream>>>(
      static_cast<const float*>(partial_max),
      static_cast<const float*>(partial_sum),
      static_cast<const float*>(partial_accumulator),
      static_cast<__nv_bfloat16*>(output), split_count, tokens);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_qwen35_attention_flash_split_cta_bf16(
    const void* query, const void* key_cache, const void* value_cache,
    void* partial_max, void* partial_sum, void* partial_accumulator,
    void* output, int split_count, int bucket_kv_len, int max_seq_len,
    float scale, const void* position, cudaStream_t stream) {
  if (query == nullptr || key_cache == nullptr || value_cache == nullptr ||
      partial_max == nullptr || partial_sum == nullptr ||
      partial_accumulator == nullptr || output == nullptr ||
      position == nullptr || split_count < 2 || split_count > 16 ||
      (split_count & (split_count - 1)) != 0 || bucket_kv_len <= 0 ||
      bucket_kv_len > max_seq_len || !(scale > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  dim3 stage_grid(24, split_count, 1);
  qwen35_attention_flash_split_cta_bf16_kernel<<<
      stage_grid, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(query),
      static_cast<const __nv_bfloat16*>(key_cache),
      static_cast<const __nv_bfloat16*>(value_cache),
      static_cast<float*>(partial_max), static_cast<float*>(partial_sum),
      static_cast<float*>(partial_accumulator), split_count, bucket_kv_len,
      max_seq_len, scale, static_cast<const uint32_t*>(position), 1);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  dim3 reduce_grid(24, 1, 1);
  qwen35_attention_flash_split_cta_reduce_bf16_kernel<<<
      reduce_grid, 256, 0, stream>>>(
      static_cast<const float*>(partial_max),
      static_cast<const float*>(partial_sum),
      static_cast<const float*>(partial_accumulator),
      static_cast<__nv_bfloat16*>(output), split_count, 1);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_qwen35_attention_flash_split_cta_m8_bf16(
    const void* query, const void* key_cache, const void* value_cache,
    void* partial_max, void* partial_sum, void* partial_accumulator,
    void* output, int split_count, int bucket_kv_len, int max_seq_len,
    float scale, const void* positions, int tokens, cudaStream_t stream) {
  if (query == nullptr || key_cache == nullptr || value_cache == nullptr ||
      partial_max == nullptr || partial_sum == nullptr ||
      partial_accumulator == nullptr || output == nullptr ||
      positions == nullptr || tokens < 1 || tokens > 8 ||
      split_count < 2 || split_count > 16 ||
      (split_count & (split_count - 1)) != 0 || bucket_kv_len <= 0 ||
      bucket_kv_len > max_seq_len || !(scale > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  dim3 stage_grid(24, split_count, tokens);
  qwen35_attention_flash_split_cta_bf16_kernel<<<
      stage_grid, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(query),
      static_cast<const __nv_bfloat16*>(key_cache),
      static_cast<const __nv_bfloat16*>(value_cache),
      static_cast<float*>(partial_max), static_cast<float*>(partial_sum),
      static_cast<float*>(partial_accumulator), split_count, bucket_kv_len,
      max_seq_len, scale, static_cast<const uint32_t*>(positions), tokens);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  dim3 reduce_grid(24, tokens, 1);
  qwen35_attention_flash_split_cta_reduce_bf16_kernel<<<
      reduce_grid, 256, 0, stream>>>(
      static_cast<const float*>(partial_max),
      static_cast<const float*>(partial_sum),
      static_cast<const float*>(partial_accumulator),
      static_cast<__nv_bfloat16*>(output), split_count, tokens);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_w8a16_gemv_bf16(
    const void* activation, const void* weight, const void* scales,
    void* output, int input_dim, int output_dim, cudaStream_t stream) {
  if (activation == nullptr || weight == nullptr || scales == nullptr ||
      output == nullptr || input_dim <= 0 || output_dim <= 0 ||
      input_dim % 8 != 0 || output_dim % 8 != 0) {
    return cudaErrorInvalidValue;
  }
  constexpr int kRowsPerBlock = 8;
  const size_t shared_bytes =
      static_cast<size_t>(input_dim) * sizeof(__nv_bfloat16) +
      kRowsPerBlock * sizeof(float);
  w8a16_gemv_bf16_kernel<<<
      output_dim / kRowsPerBlock, 256, shared_bytes, stream>>>(
      static_cast<const __nv_bfloat16*>(activation),
      static_cast<const int8_t*>(weight), static_cast<const float*>(scales),
      static_cast<__nv_bfloat16*>(output), input_dim);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_qwen35_gdn_recurrent_bf16(
    const void* query, const void* key, const void* value, const void* g,
    const void* beta, void* recurrent_state, void* output, int heads,
    int key_dim, int value_dim, cudaStream_t stream) {
  if (query == nullptr || key == nullptr || value == nullptr || g == nullptr ||
      beta == nullptr || recurrent_state == nullptr || output == nullptr ||
      heads != 48 || key_dim != 128 || value_dim != 128) {
    return cudaErrorInvalidValue;
  }
  qwen35_gdn_recurrent_bf16_kernel<<<heads, 128, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(query),
      static_cast<const __nv_bfloat16*>(key),
      static_cast<const __nv_bfloat16*>(value), static_cast<const float*>(g),
      static_cast<const float*>(beta), static_cast<float*>(recurrent_state),
      static_cast<__nv_bfloat16*>(output));
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_qwen35_gdn_conv4_silu_bf16(
    const void* input, const void* weight, void* conv_state, void* output,
    int channels, int kernel_size, cudaStream_t stream) {
  if (input == nullptr || weight == nullptr || conv_state == nullptr ||
      output == nullptr || channels != 10240 || kernel_size != 4) {
    return cudaErrorInvalidValue;
  }
  const int blocks = (channels + 255) / 256;
  qwen35_gdn_conv4_silu_bf16_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<__nv_bfloat16*>(conv_state),
      static_cast<__nv_bfloat16*>(output), channels);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_qwen35_gdn_prepare_bf16(
    const void* convolved_qkv, const void* a, const void* b,
    const void* a_log, const void* dt_bias, void* query, void* key,
    void* value, void* g, void* beta, cudaStream_t stream) {
  if (convolved_qkv == nullptr || a == nullptr || b == nullptr ||
      a_log == nullptr || dt_bias == nullptr || query == nullptr ||
      key == nullptr || value == nullptr || g == nullptr || beta == nullptr) {
    return cudaErrorInvalidValue;
  }
  qwen35_gdn_prepare_bf16_kernel<<<48, 128, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(convolved_qkv),
      static_cast<const __nv_bfloat16*>(a),
      static_cast<const __nv_bfloat16*>(b),
      static_cast<const __nv_bfloat16*>(a_log),
      static_cast<const __nv_bfloat16*>(dt_bias),
      static_cast<__nv_bfloat16*>(query),
      static_cast<__nv_bfloat16*>(key),
      static_cast<__nv_bfloat16*>(value), static_cast<float*>(g),
      static_cast<float*>(beta));
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_qwen35_gdn_gated_rmsnorm_bf16(
    const void* input, const void* gate, const void* weight, void* output,
    int heads, int dimension, float epsilon, cudaStream_t stream) {
  if (input == nullptr || gate == nullptr || weight == nullptr ||
      output == nullptr || heads != 48 || dimension != 128 ||
      !(epsilon > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  qwen35_gdn_gated_rmsnorm_bf16_kernel<<<heads, dimension, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(gate),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<__nv_bfloat16*>(output), epsilon);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_qwen35_gdn_conv4_prepare_bf16(
    const void* projected_qkv, const void* conv_weight, void* conv_state,
    const void* projected_ab, const void* a_log, const void* dt_bias,
    void* a_output, void* b_output, void* query, void* key, void* value,
    void* g, void* beta, cudaStream_t stream) {
  if (projected_qkv == nullptr || conv_weight == nullptr ||
      conv_state == nullptr || projected_ab == nullptr || a_log == nullptr ||
      dt_bias == nullptr || a_output == nullptr || b_output == nullptr ||
      query == nullptr || key == nullptr || value == nullptr || g == nullptr ||
      beta == nullptr) {
    return cudaErrorInvalidValue;
  }
  qwen35_gdn_conv4_prepare_bf16_kernel<<<40, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(projected_qkv),
      static_cast<const __nv_bfloat16*>(conv_weight),
      static_cast<__nv_bfloat16*>(conv_state),
      static_cast<const __nv_bfloat16*>(projected_ab),
      static_cast<const __nv_bfloat16*>(a_log),
      static_cast<const __nv_bfloat16*>(dt_bias),
      static_cast<__nv_bfloat16*>(a_output),
      static_cast<__nv_bfloat16*>(b_output),
      static_cast<__nv_bfloat16*>(query),
      static_cast<__nv_bfloat16*>(key),
      static_cast<__nv_bfloat16*>(value), static_cast<float*>(g),
      static_cast<float*>(beta));
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_qwen35_gdn_conv4_prepare_m8_bf16(
    const void* projected_qkv, const void* conv_weight, void* conv_state,
    const void* projected_ab, const void* a_log, const void* dt_bias,
    void* a_output, void* b_output, void* query, void* key, void* value,
    void* g, void* beta, int tokens, cudaStream_t stream) {
  if (projected_qkv == nullptr || conv_weight == nullptr ||
      conv_state == nullptr || projected_ab == nullptr || a_log == nullptr ||
      dt_bias == nullptr || a_output == nullptr || b_output == nullptr ||
      query == nullptr || key == nullptr || value == nullptr || g == nullptr ||
      beta == nullptr || tokens < 1 || tokens > 8) {
    return cudaErrorInvalidValue;
  }
  qwen35_gdn_conv4_prepare_m8_bf16_kernel<<<40, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(projected_qkv),
      static_cast<const __nv_bfloat16*>(conv_weight),
      static_cast<__nv_bfloat16*>(conv_state),
      static_cast<const __nv_bfloat16*>(projected_ab),
      static_cast<const __nv_bfloat16*>(a_log),
      static_cast<const __nv_bfloat16*>(dt_bias),
      static_cast<__nv_bfloat16*>(a_output),
      static_cast<__nv_bfloat16*>(b_output),
      static_cast<__nv_bfloat16*>(query),
      static_cast<__nv_bfloat16*>(key),
      static_cast<__nv_bfloat16*>(value), static_cast<float*>(g),
      static_cast<float*>(beta), tokens);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_qwen35_gdn_recurrent_m8_bf16(
    const void* query, const void* key, const void* value, const void* g,
    const void* beta, void* recurrent_state, void* output, int tokens,
    cudaStream_t stream) {
  if (query == nullptr || key == nullptr || value == nullptr || g == nullptr ||
      beta == nullptr || recurrent_state == nullptr || output == nullptr ||
      tokens < 1 || tokens > 8) {
    return cudaErrorInvalidValue;
  }
  qwen35_gdn_recurrent_m8_bf16_kernel<<<48, 128, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(query),
      static_cast<const __nv_bfloat16*>(key),
      static_cast<const __nv_bfloat16*>(value), static_cast<const float*>(g),
      static_cast<const float*>(beta), static_cast<float*>(recurrent_state),
      static_cast<__nv_bfloat16*>(output), tokens);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_qwen35_gdn_recurrent_m8_hybrid_bf16(
    const void* query, const void* key, const void* value, const void* g,
    const void* beta, void* recurrent_state, void* output, int tokens,
    cudaStream_t stream) {
  if (query == nullptr || key == nullptr || value == nullptr || g == nullptr ||
      beta == nullptr || recurrent_state == nullptr || output == nullptr ||
      tokens < 1 || tokens > 8) {
    return cudaErrorInvalidValue;
  }
  qwen35_gdn_recurrent_m8_hybrid_bf16_kernel<<<48, 128, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(query),
      static_cast<const __nv_bfloat16*>(key),
      static_cast<const __nv_bfloat16*>(value), static_cast<const float*>(g),
      static_cast<const float*>(beta), static_cast<float*>(recurrent_state),
      static_cast<__nv_bfloat16*>(output), tokens);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_qwen35_gdn_recurrent_m8_hybrid_pairnorm_bf16(
    const void* query, const void* key, const void* value, const void* g,
    const void* beta, void* recurrent_state, void* output, int tokens,
    cudaStream_t stream) {
  if (query == nullptr || key == nullptr || value == nullptr || g == nullptr ||
      beta == nullptr || recurrent_state == nullptr || output == nullptr ||
      tokens < 1 || tokens > 8) {
    return cudaErrorInvalidValue;
  }
  qwen35_gdn_recurrent_m8_hybrid_pairnorm_bf16_kernel<<<48, 128, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(query),
      static_cast<const __nv_bfloat16*>(key),
      static_cast<const __nv_bfloat16*>(value), static_cast<const float*>(g),
      static_cast<const float*>(beta), static_cast<float*>(recurrent_state),
      static_cast<__nv_bfloat16*>(output), tokens);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_qwen35_gdn_gated_rmsnorm_m8_bf16(
    const void* input, const void* gate, const void* weight, void* output,
    float epsilon, int tokens, cudaStream_t stream) {
  if (input == nullptr || gate == nullptr || weight == nullptr ||
      output == nullptr || !(epsilon > 0.0f) || tokens < 1 || tokens > 8) {
    return cudaErrorInvalidValue;
  }
  qwen35_gdn_gated_rmsnorm_m8_bf16_kernel<<<
      tokens * 48, 128, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(gate),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<__nv_bfloat16*>(output), epsilon, tokens);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_w4a16_gemv_bf16(
    const void* activation, const void* packed, const void* scales,
    const void* packed_zero_points, void* output, int input_dim,
    int output_dim, int group_size, cudaStream_t stream) {
  if (activation == nullptr || packed == nullptr || scales == nullptr ||
      packed_zero_points == nullptr || output == nullptr || input_dim <= 0 ||
      output_dim <= 0 || group_size != 32 || input_dim % 32 != 0 ||
      output_dim % 8 != 0) {
    return cudaErrorInvalidValue;
  }
  constexpr int kThreads = 256;
  constexpr int kRowsPerBlock = 8;
  const int blocks = (output_dim + kRowsPerBlock - 1) / kRowsPerBlock;
  w4a16_gemv_bf16_kernel<<<blocks, kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(activation),
      static_cast<const int32_t*>(packed),
      static_cast<const __nv_bfloat16*>(scales),
      static_cast<const int32_t*>(packed_zero_points),
      static_cast<__nv_bfloat16*>(output), input_dim);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_w4a16_gemv_bf16_staged(
    const void* activation, const void* packed, const void* scales,
    const void* packed_zero_points, void* output, int input_dim,
    int output_dim, int group_size, cudaStream_t stream) {
  if (activation == nullptr || packed == nullptr || scales == nullptr ||
      packed_zero_points == nullptr || output == nullptr || input_dim <= 0 ||
      output_dim <= 0 || group_size != 32 || input_dim % 32 != 0 ||
      output_dim % 8 != 0) {
    return cudaErrorInvalidValue;
  }
  constexpr int kThreads = 256;
  constexpr int kRowsPerBlock = 8;
  const int groups = input_dim / group_size;
  const size_t shared_bytes =
      static_cast<size_t>(input_dim) * sizeof(__nv_bfloat16) +
      static_cast<size_t>(kRowsPerBlock * groups) * sizeof(__nv_bfloat16) +
      static_cast<size_t>(groups) * sizeof(int32_t);
  const int blocks = output_dim / kRowsPerBlock;
  w4a16_gemv_bf16_staged_kernel<<<
      blocks, kThreads, shared_bytes, stream>>>(
      static_cast<const __nv_bfloat16*>(activation),
      static_cast<const int32_t*>(packed),
      static_cast<const __nv_bfloat16*>(scales),
      static_cast<const int32_t*>(packed_zero_points),
      static_cast<__nv_bfloat16*>(output), input_dim);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_w4a16_gemm_m8_bf16(
    const void* activation, const void* packed, const void* scales,
    const void* packed_zero_points, void* output, int tokens, int input_dim,
    int output_dim, int group_size, cudaStream_t stream) {
  if (activation == nullptr || packed == nullptr || scales == nullptr ||
      packed_zero_points == nullptr || output == nullptr || tokens < 1 ||
      tokens > 8 || input_dim <= 0 || output_dim <= 0 || group_size != 32 ||
      input_dim % 32 != 0 || output_dim % 8 != 0) {
    return cudaErrorInvalidValue;
  }
  constexpr int kThreads = 256;
  constexpr int kRowsPerBlock = 8;
  const int blocks = output_dim / kRowsPerBlock;
  w4a16_gemm_m8_bf16_kernel<<<blocks, kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(activation),
      static_cast<const int32_t*>(packed),
      static_cast<const __nv_bfloat16*>(scales),
      static_cast<const int32_t*>(packed_zero_points),
      static_cast<__nv_bfloat16*>(output), tokens, input_dim, output_dim);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_quantize_f16_e4m3(
    const void* input, void* output, int64_t count, float scale,
    cudaStream_t stream) {
  if (input == nullptr || output == nullptr || count <= 0 || !(scale > 0.0f))
    return cudaErrorInvalidValue;
  constexpr int threads = 256;
  const float inverse_scale = 1.0f / scale;
  const bool aligned =
      (reinterpret_cast<uintptr_t>(input) & 3U) == 0 &&
      (reinterpret_cast<uintptr_t>(output) & 3U) == 0;
  int64_t vector_count = aligned ? count & ~int64_t{3} : 0;
  if (vector_count != 0) {
    const int64_t groups = vector_count / 4;
    int blocks = static_cast<int>((groups + threads - 1) / threads);
    blocks = blocks > 1024 ? 1024 : blocks;
    quantize_f16_e4m3_packed4_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const half*>(input),
        static_cast<__nv_fp8_e4m3*>(output), vector_count, inverse_scale);
  }
  const int64_t tail = count - vector_count;
  if (tail != 0) {
    int blocks = static_cast<int>((tail + threads - 1) / threads);
    blocks = blocks > 1024 ? 1024 : blocks;
    quantize_f16_e4m3_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const half*>(input) + vector_count,
        static_cast<__nv_fp8_e4m3*>(output) + vector_count,
        tail, inverse_scale);
  }
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_dequantize_e4m3_f16(
    const void* input, void* output, int64_t count, float scale,
    cudaStream_t stream) {
  if (input == nullptr || output == nullptr || count <= 0 ||
      !(scale > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 4096 ? 4096 : blocks;
  dequantize_e4m3_f16_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const __nv_fp8_e4m3*>(input),
      static_cast<half*>(output), count, scale);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_rgb_u8_to_patches_e4m3(
    const void* images, void* patches, int views, int image_size,
    int patch_size, int layout, float scale, cudaStream_t stream) {
  if (images == nullptr || patches == nullptr || views <= 0 ||
      image_size <= 0 || patch_size <= 0 || image_size % patch_size != 0 ||
      (layout != 0 && layout != 1) || !(scale > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  const int patches_per_side = image_size / patch_size;
  const int64_t count = static_cast<int64_t>(views) * patches_per_side *
                        patches_per_side * 3 * patch_size * patch_size;
  constexpr int threads = 256;
  int blocks = static_cast<int>((count + threads - 1) / threads);
  blocks = blocks > 1024 ? 1024 : blocks;
  if (layout == 0) {
    rgb_u8_to_patches_e4m3_kernel<true><<<blocks, threads, 0, stream>>>(
        static_cast<const uint8_t*>(images),
        static_cast<__nv_fp8_e4m3*>(patches), views, image_size, patch_size,
        1.0f / scale);
  } else {
    rgb_u8_to_patches_e4m3_kernel<false><<<blocks, threads, 0, stream>>>(
        static_cast<const uint8_t*>(images),
        static_cast<__nv_fp8_e4m3*>(patches), views, image_size, patch_size,
        1.0f / scale);
  }
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_mqa_flash_f16(
    const void* q, const void* prefix_k, const void* prefix_v,
    const void* suffix_k, const void* suffix_v, void* output,
    int suffix_tokens, int heads, int head_dim, int prefix_tokens,
    cudaStream_t stream) {
  if (suffix_tokens <= 0 || heads <= 0 || head_dim <= 0 || head_dim > 256 ||
      prefix_tokens < 0) return cudaErrorInvalidValue;
  int threads = 256;
  int warps = threads / 32;
  size_t shared_bytes =
      static_cast<size_t>(prefix_tokens + suffix_tokens + warps) * sizeof(float);
  mqa_flash_f16_kernel<<<dim3(suffix_tokens, heads), threads, shared_bytes, stream>>>(
      static_cast<const half*>(q), static_cast<const half*>(prefix_k),
      static_cast<const half*>(prefix_v), static_cast<const half*>(suffix_k),
      static_cast<const half*>(suffix_v), static_cast<half*>(output),
      suffix_tokens, heads, head_dim, prefix_tokens);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_rms_norm_quant_f16_e4m3(
    const void* input, const void* weight, void* output, int rows, int cols,
    float eps, float scale, cudaStream_t stream) {
  if (rows <= 0 || cols <= 0 || !(scale > 0.0f)) return cudaErrorInvalidValue;
  rms_norm_quant_f16_e4m3_kernel<<<rows, 256, 0, stream>>>(
      static_cast<const half*>(input), static_cast<const half*>(weight),
      static_cast<__nv_fp8_e4m3*>(output), rows, cols, eps, 1.0f / scale);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_layer_norm_quant_f16_e4m3(
    const void* input, const void* weight, const void* bias, void* output,
    int rows, int cols, float eps, float scale, cudaStream_t stream) {
  if (rows <= 0 || cols <= 0 || !(scale > 0.0f)) return cudaErrorInvalidValue;
  layer_norm_quant_f16_e4m3_kernel<<<rows, 256, 0, stream>>>(
      static_cast<const half*>(input), static_cast<const half*>(weight),
      static_cast<const half*>(bias), static_cast<__nv_fp8_e4m3*>(output),
      rows, cols, eps, 1.0f / scale);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_bias_gelu_quant_f16_e4m3(
    const void* input, const void* bias, void* output, int rows, int cols,
    float scale, cudaStream_t stream) {
  if (rows <= 0 || cols <= 0 || !(scale > 0.0f)) return cudaErrorInvalidValue;
  int64_t count = static_cast<int64_t>(rows) * cols;
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 1024 ? 1024 : blocks;
  bias_gelu_quant_f16_e4m3_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const half*>(input), static_cast<const half*>(bias),
      static_cast<__nv_fp8_e4m3*>(output), count, cols, 1.0f / scale);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_bias_silu_quant_f16_e4m3(
    const void* input, const void* bias, void* output, int rows, int cols,
    float scale, cudaStream_t stream) {
  if (rows <= 0 || cols <= 0 || !(scale > 0.0f)) return cudaErrorInvalidValue;
  int64_t count = static_cast<int64_t>(rows) * cols;
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 1024 ? 1024 : blocks;
  bias_silu_quant_f16_e4m3_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const half*>(input), static_cast<const half*>(bias),
      static_cast<__nv_fp8_e4m3*>(output), count, cols, 1.0f / scale);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_bias_silu_f16(
    const void* input, const void* bias, void* output, int rows, int cols,
    cudaStream_t stream) {
  if (rows <= 0 || cols <= 0) return cudaErrorInvalidValue;
  int64_t count = static_cast<int64_t>(rows) * cols;
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 1024 ? 1024 : blocks;
  bias_silu_f16_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const half*>(input), static_cast<const half*>(bias),
      static_cast<half*>(output), count, cols);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_bias_f16(
    const void* input, const void* bias, void* output, int rows, int cols,
    cudaStream_t stream) {
  if (rows <= 0 || cols <= 0) return cudaErrorInvalidValue;
  int64_t count = static_cast<int64_t>(rows) * cols;
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 1024 ? 1024 : blocks;
  bias_f16_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const half*>(input), static_cast<const half*>(bias),
      static_cast<half*>(output), count, cols);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_embedding_f16(
    const void* table, const void* ids, void* output, int tokens,
    int width, int vocab_size, cudaStream_t stream) {
  if (tokens <= 0 || width <= 0 || vocab_size <= 0) return cudaErrorInvalidValue;
  int64_t count = static_cast<int64_t>(tokens) * width;
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 1024 ? 1024 : blocks;
  embedding_f16_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const half*>(table), static_cast<const uint32_t*>(ids),
      static_cast<half*>(output), tokens, width, vocab_size);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_concat_rows_f16(
    const void* first, const void* second, void* output, int first_rows,
    int second_rows, int cols, cudaStream_t stream) {
  if (first_rows <= 0 || second_rows <= 0 || cols <= 0) return cudaErrorInvalidValue;
  int64_t first_count = static_cast<int64_t>(first_rows) * cols;
  int64_t total_count = static_cast<int64_t>(first_rows + second_rows) * cols;
  int blocks = static_cast<int>((total_count + 255) / 256);
  blocks = blocks > 1024 ? 1024 : blocks;
  concat_rows_f16_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const half*>(first), static_cast<const half*>(second),
      static_cast<half*>(output), first_count, total_count);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_euler_update_f16(
    const void* state, const void* velocity, void* output, int64_t count,
    float dt, cudaStream_t stream) {
  if (count <= 0) return cudaErrorInvalidValue;
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 1024 ? 1024 : blocks;
  euler_update_f16_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const half*>(state), static_cast<const half*>(velocity),
      static_cast<half*>(output), count, dt);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_geglu_quant_f16_e4m3(
    const void* gate_up, void* output, int rows, int inner, float scale,
    cudaStream_t stream) {
  if (rows <= 0 || inner <= 0 || !(scale > 0.0f)) return cudaErrorInvalidValue;
  if ((inner & 1) != 0) return cudaErrorInvalidValue;
  const bool packed8 = (inner & 7) == 0 &&
      (reinterpret_cast<uintptr_t>(gate_up) & 7U) == 0 &&
      (reinterpret_cast<uintptr_t>(output) & 7U) == 0;
  if (packed8) {
    int group_count = rows * (inner / 8);
    int blocks = (group_count + 255) / 256;
    blocks = blocks > 1024 ? 1024 : blocks;
    geglu_quant_f16_e4m3_packed8_kernel<<<blocks, 256, 0, stream>>>(
        static_cast<const half*>(gate_up),
        static_cast<__nv_fp8_e4m3*>(output), rows, inner, 1.0f / scale);
    return cudaGetLastError();
  }
  const bool packed4 = (inner & 3) == 0 &&
      (reinterpret_cast<uintptr_t>(gate_up) & 3U) == 0 &&
      (reinterpret_cast<uintptr_t>(output) & 3U) == 0;
  if (packed4) {
    int group_count = rows * (inner / 4);
    int blocks = (group_count + 255) / 256;
    blocks = blocks > 1024 ? 1024 : blocks;
    geglu_quant_f16_e4m3_packed4_kernel<<<blocks, 256, 0, stream>>>(
        static_cast<const half*>(gate_up),
        static_cast<__nv_fp8_e4m3*>(output), rows, inner, 1.0f / scale);
    return cudaGetLastError();
  }
  int pair_count = rows * (inner / 2);
  int blocks = (pair_count + 255) / 256;
  blocks = blocks > 1024 ? 1024 : blocks;
  geglu_quant_f16_e4m3_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const half*>(gate_up), static_cast<__nv_fp8_e4m3*>(output),
      rows, inner, 1.0f / scale);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_bias_residual_f16(
    const void* projection, const void* bias, const void* residual, void* output,
    int rows, int cols, cudaStream_t stream) {
  if (rows <= 0 || cols <= 0) return cudaErrorInvalidValue;
  int64_t count = static_cast<int64_t>(rows) * cols;
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 1024 ? 1024 : blocks;
  bias_residual_f16_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const half*>(projection), static_cast<const half*>(bias),
      static_cast<const half*>(residual), static_cast<half*>(output), count, cols);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_bias_residual_rms_norm_quant_f16_e4m3(
    const void* projection, const void* bias, const void* residual,
    const void* weight, void* hidden, void* normalized, int rows, int cols,
    float eps, float scale, cudaStream_t stream) {
  if (rows <= 0 || cols <= 0 || !(scale > 0.0f)) return cudaErrorInvalidValue;
  bias_residual_rms_norm_quant_f16_e4m3_kernel<<<rows, 256, 0, stream>>>(
      static_cast<const half*>(projection), static_cast<const half*>(bias),
      static_cast<const half*>(residual), static_cast<const half*>(weight),
      static_cast<half*>(hidden), static_cast<__nv_fp8_e4m3*>(normalized),
      rows, cols, eps, 1.0f / scale);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_bias_residual_layer_norm_quant_f16_e4m3(
    const void* projection, const void* projection_bias, const void* residual,
    const void* norm_weight, const void* norm_bias, void* hidden,
    void* normalized, int rows, int cols, float eps, float scale,
    cudaStream_t stream) {
  if (rows <= 0 || cols <= 0 || !(scale > 0.0f)) return cudaErrorInvalidValue;
  bias_residual_layer_norm_quant_f16_e4m3_kernel<<<rows, 256, 0, stream>>>(
      static_cast<const half*>(projection), static_cast<const half*>(projection_bias),
      static_cast<const half*>(residual), static_cast<const half*>(norm_weight),
      static_cast<const half*>(norm_bias), static_cast<half*>(hidden),
      static_cast<__nv_fp8_e4m3*>(normalized), rows, cols, eps, 1.0f / scale);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_ada_rms_norm_quant_f16_e4m3(
    const void* input, const void* style, void* output, int rows, int cols,
    float eps, float scale, cudaStream_t stream) {
  if (rows <= 0 || cols <= 0 || !(scale > 0.0f)) return cudaErrorInvalidValue;
  ada_rms_norm_quant_f16_e4m3_kernel<<<rows, 256, 0, stream>>>(
      static_cast<const half*>(input), static_cast<const half*>(style),
      static_cast<__nv_fp8_e4m3*>(output), rows, cols, eps, 1.0f / scale);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_ada_gate_residual_f16(
    const void* projection, const void* residual, const void* style,
    void* output, int rows, int cols, cudaStream_t stream) {
  if (rows <= 0 || cols <= 0) return cudaErrorInvalidValue;
  int64_t count = static_cast<int64_t>(rows) * cols;
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 1024 ? 1024 : blocks;
  ada_gate_residual_f16_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const half*>(projection), static_cast<const half*>(residual),
      static_cast<const half*>(style), static_cast<half*>(output), rows, cols);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_ada_gate_residual_rms_norm_quant_f16_e4m3(
    const void* projection, const void* residual, const void* gate_style,
    const void* norm_style, void* hidden, void* normalized, int rows, int cols,
    float eps, float scale, cudaStream_t stream) {
  if (rows <= 0 || cols <= 0 || !(scale > 0.0f)) return cudaErrorInvalidValue;
  ada_gate_residual_rms_norm_quant_f16_e4m3_kernel<<<rows, 256, 0, stream>>>(
      static_cast<const half*>(projection), static_cast<const half*>(residual),
      static_cast<const half*>(gate_style), static_cast<const half*>(norm_style),
      static_cast<half*>(hidden), static_cast<__nv_fp8_e4m3*>(normalized),
      rows, cols, eps, 1.0f / scale);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_qkv_rope_f16(
    const void* qkv, const void* bias, void* q, void* k, void* v, int tokens, int q_heads,
    int kv_heads, int head_dim, float theta, int position_offset,
    int kv_output_offset, cudaStream_t stream) {
  if (tokens <= 0 || q_heads <= 0 || kv_heads <= 0 || head_dim <= 0 ||
      head_dim > 256 || (head_dim & 1) != 0) return cudaErrorInvalidValue;
  qkv_rope_f16_kernel<<<dim3(tokens, q_heads + 2 * kv_heads), head_dim / 2, 0, stream>>>(
      static_cast<const half*>(qkv), static_cast<const half*>(bias),
      static_cast<half*>(q), static_cast<half*>(k),
      static_cast<half*>(v), tokens, q_heads, kv_heads, head_dim, theta,
      position_offset, kv_output_offset);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_qkv_split_bias_f16(
    const void* qkv, const void* bias, void* q, void* k, void* v,
    int tokens, int projection_width, cudaStream_t stream) {
  if (tokens <= 0 || projection_width <= 0) return cudaErrorInvalidValue;
  qkv_split_bias_f16_kernel<<<tokens, 256, 0, stream>>>(
      static_cast<const half*>(qkv), static_cast<const half*>(bias),
      static_cast<half*>(q), static_cast<half*>(k), static_cast<half*>(v),
      tokens, projection_width);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_mha_flash_f16(
    const void* q, const void* k, const void* v, void* output,
    int tokens_per_batch, int batches, int heads, int head_dim, cudaStream_t stream) {
  if (tokens_per_batch <= 0 || batches <= 0 || heads <= 0 ||
      head_dim <= 0 || head_dim > 256)
    return cudaErrorInvalidValue;
  constexpr int threads = 256;
  size_t shared_bytes = static_cast<size_t>(tokens_per_batch + threads / 32) * sizeof(float);
  mha_flash_f16_kernel<<<dim3(tokens_per_batch, heads, batches), threads, shared_bytes, stream>>>(
      static_cast<const half*>(q), static_cast<const half*>(k),
      static_cast<const half*>(v), static_cast<half*>(output),
      tokens_per_batch, heads, head_dim);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_bias_position_f16(
    const void* projection, const void* bias, const void* position,
    void* output, int rows, int cols, int tokens_per_view, cudaStream_t stream) {
  if (rows <= 0 || cols <= 0 || tokens_per_view <= 0 ||
      rows % tokens_per_view != 0) return cudaErrorInvalidValue;
  int64_t count = static_cast<int64_t>(rows) * cols;
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 1024 ? 1024 : blocks;
  bias_position_f16_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const half*>(projection), static_cast<const half*>(bias),
      static_cast<const half*>(position), static_cast<half*>(output),
      count, cols, tokens_per_view);
  return cudaGetLastError();
}
