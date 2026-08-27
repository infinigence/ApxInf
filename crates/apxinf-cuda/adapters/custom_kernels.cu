// Copyright 2026 apxinf contributors.
// Stable C ABI and CUDA launch adapter for custom static-inference operators.

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace {
#include "../kernels/custom/math.cuh"
#include "../kernels/custom/reduction.cuh"
#include "../kernels/custom/quantization.cuh"
#include "../kernels/custom/preprocess.cuh"
#include "../kernels/custom/attention.cuh"
#include "../kernels/custom/qwen25_omni_attention.cuh"
#include "../kernels/custom/qwen25_omni_fused.cuh"
#include "../kernels/custom/qwen25_omni_vision.cuh"
#include "../kernels/custom/normalization.cuh"
#include "../kernels/custom/activation.cuh"
#include "../kernels/custom/embedding.cuh"
#include "../kernels/custom/elementwise.cuh"
#include "../kernels/custom/fused.cuh"
#include "../kernels/custom/cache.cuh"
}  // namespace

namespace {

// Resolve the validated action Ada packed8 route before CUDA graph capture.
// Auto enables it only for the exact supported shape.
const int kActionAdaPacked8Mode = [] {
    const char* value = std::getenv("APXINF_PI05_ACTION_ADA_PACKED8");
    if (value == nullptr || std::strcmp(value, "auto") == 0) {
      return 2;
    }
    if (std::strcmp(value, "0") == 0 || std::strcmp(value, "off") == 0) {
      return 0;
    }
    if (std::strcmp(value, "1") == 0 || std::strcmp(value, "on") == 0) {
      return 1;
    }
    return -1;
  }();

}  // namespace

extern "C" cudaError_t apxinf_im2col1d_bf16(
    const void* input, void* output, int frames, int channels, int kernel,
    int stride, int padding, int output_frames, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || frames <= 0 || channels <= 0 ||
      kernel <= 0 || stride <= 0 || padding < 0 || output_frames <= 0) {
    return cudaErrorInvalidValue;
  }
  int64_t count = static_cast<int64_t>(output_frames) * channels * kernel;
  constexpr int threads = 256;
  int blocks = static_cast<int>((count + threads - 1) / threads);
  im2col1d_bf16_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<__nv_bfloat16*>(output), frames, channels, kernel, stride,
      padding, output_frames);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_avg_pool1d_bf16(
    const void* input, void* output, int frames, int channels, int kernel,
    int stride, int output_frames, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || frames <= 0 || channels <= 0 ||
      kernel <= 0 || stride <= 0 || output_frames <= 0) {
    return cudaErrorInvalidValue;
  }
  int64_t count = static_cast<int64_t>(output_frames) * channels;
  constexpr int threads = 256;
  int blocks = static_cast<int>((count + threads - 1) / threads);
  avg_pool1d_bf16_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<__nv_bfloat16*>(output), frames, channels, kernel, stride,
      output_frames);
  return cudaGetLastError();
}

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

extern "C" cudaError_t
apxinf_static_qwen25_omni_attention_flash_w32_bf16(
    const void* query, const void* key_cache, const void* value_cache,
    void* output, int query_heads, int kv_heads, int head_dim,
    int bucket_kv_len, int max_seq_len, float scale, const void* position,
    cudaStream_t stream) {
  if (query == nullptr || key_cache == nullptr || value_cache == nullptr ||
      output == nullptr || position == nullptr || query_heads != 16 ||
      kv_heads != 2 || head_dim != 128 || bucket_kv_len != 32768 ||
      max_seq_len != 32768 || !(scale > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  flash_attn_decode_bf16_splitk_kernel<128, 32><<<16, 1024, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(query),
      static_cast<const __nv_bfloat16*>(key_cache),
      static_cast<const __nv_bfloat16*>(value_cache),
      static_cast<__nv_bfloat16*>(output), query_heads, kv_heads,
      bucket_kv_len, max_seq_len, scale,
      static_cast<const uint32_t*>(position));
  return cudaGetLastError();
}

extern "C" cudaError_t
apxinf_static_qwen25_omni_qkv_bias_tmrope_kv_write_bf16(
    const void* packed_qkv, const void* bias, void* query, void* key_cache,
    void* value_cache, float theta, const void* positions,
    const void* cache_position, cudaStream_t stream) {
  if (packed_qkv == nullptr || bias == nullptr || query == nullptr ||
      key_cache == nullptr || value_cache == nullptr || positions == nullptr ||
      cache_position == nullptr || theta != 1000000.0f) {
    return cudaErrorInvalidValue;
  }
  qwen25_omni_qkv_bias_tmrope_kv_write_bf16_kernel<<<20, 128, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(packed_qkv),
      static_cast<const __nv_bfloat16*>(bias),
      static_cast<__nv_bfloat16*>(query),
      static_cast<__nv_bfloat16*>(key_cache),
      static_cast<__nv_bfloat16*>(value_cache), theta,
      static_cast<const uint32_t*>(positions),
      static_cast<const uint32_t*>(cache_position));
  return cudaGetLastError();
}

extern "C" cudaError_t
apxinf_static_qwen25_omni_prefill_qkv_bias_tmrope_kv_write_bf16(
    const void* packed_qkv, const void* bias, void* query, void* key_cache,
    void* value_cache, int rows, int start_position, const void* positions,
    cudaStream_t stream) {
  if (packed_qkv == nullptr || bias == nullptr || query == nullptr ||
      key_cache == nullptr || value_cache == nullptr || positions == nullptr ||
      rows != 1024 || start_position < 0 ||
      start_position > 32768 - rows) {
    return cudaErrorInvalidValue;
  }
  dim3 grid(20, rows, 1);
  qwen25_omni_prefill_qkv_bias_tmrope_kv_write_bf16_kernel<<<
      grid, 64, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(packed_qkv),
      static_cast<const __nv_bfloat16*>(bias),
      static_cast<__nv_bfloat16*>(query),
      static_cast<__nv_bfloat16*>(key_cache),
      static_cast<__nv_bfloat16*>(value_cache), rows, start_position,
      static_cast<const uint32_t*>(positions));
  return cudaGetLastError();
}

extern "C" cudaError_t
apxinf_static_qwen25_omni_residual_rmsnorm_pack8_bf16(
    void* residual, const void* delta, const void* weight, void* output,
    int rows, int columns, float eps, cudaStream_t stream) {
  const uintptr_t alignment = reinterpret_cast<uintptr_t>(residual) |
      reinterpret_cast<uintptr_t>(delta) |
      reinterpret_cast<uintptr_t>(weight) |
      reinterpret_cast<uintptr_t>(output);
  if (residual == nullptr || delta == nullptr || weight == nullptr ||
      output == nullptr || rows != 1 || columns != 2048 ||
      eps != 1.0e-6f || (alignment & 15U) != 0) {
    return cudaErrorInvalidValue;
  }
  qwen25_omni_residual_rmsnorm_pack8_bf16_kernel<<<
      1, 256, 2048 * sizeof(float), stream>>>(
      static_cast<__nv_bfloat16*>(residual),
      static_cast<const __nv_bfloat16*>(delta),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<__nv_bfloat16*>(output), eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_qwen25_omni_vision_qkv_bias_rope_bf16(
    const void* query, const void* key, const void* value,
    const void* query_bias, const void* key_bias, const void* value_bias,
    void* query_output, void* key_output, void* value_output,
    int sequence, int heads, int head_dim, float theta,
    const void* positions, cudaStream_t stream) {
  if (query == nullptr || key == nullptr || value == nullptr ||
      query_bias == nullptr || key_bias == nullptr || value_bias == nullptr ||
      query_output == nullptr || key_output == nullptr || value_output == nullptr ||
      positions == nullptr || sequence <= 0 || sequence > 65535 ||
      heads != 16 || head_dim != 80 || theta != 10000.0f) {
    return cudaErrorInvalidValue;
  }
  constexpr int kHidden = 16 * 80;
  constexpr int kThreads = 256;
  dim3 grid((kHidden + kThreads - 1) / kThreads, sequence, 3);
  qwen25_omni_vision_qkv_bias_rope_bf16_kernel<16, 80, false, false><<<
      grid, kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(query),
      static_cast<const __nv_bfloat16*>(key),
      static_cast<const __nv_bfloat16*>(value),
      static_cast<const __nv_bfloat16*>(query_bias),
      static_cast<const __nv_bfloat16*>(key_bias),
      static_cast<const __nv_bfloat16*>(value_bias),
      static_cast<__nv_bfloat16*>(query_output),
      static_cast<__nv_bfloat16*>(key_output),
      static_cast<__nv_bfloat16*>(value_output), sequence, theta,
      static_cast<const uint32_t*>(positions), nullptr);
  return cudaGetLastError();
}

extern "C" cudaError_t
apxinf_static_qwen25_omni_vision_grouped_qkv_bias_rope_bf16(
    const void* query, const void* key, const void* value,
    const void* query_bias, const void* key_bias, const void* value_bias,
    void* query_output, void* key_output, void* value_output,
    int sequence, int heads, int head_dim, float theta,
    const void* positions, const void* group_indices, cudaStream_t stream) {
  if (query == nullptr || key == nullptr || value == nullptr ||
      query_bias == nullptr || key_bias == nullptr || value_bias == nullptr ||
      query_output == nullptr || key_output == nullptr || value_output == nullptr ||
      positions == nullptr || group_indices == nullptr || sequence <= 0 ||
      sequence > 65535 || heads != 16 || head_dim != 80 || theta != 10000.0f) {
    return cudaErrorInvalidValue;
  }
  constexpr int kHidden = 16 * 80;
  constexpr int kThreads = 256;
  dim3 grid((kHidden + kThreads - 1) / kThreads, sequence, 3);
  qwen25_omni_vision_qkv_bias_rope_bf16_kernel<16, 80, true, false><<<
      grid, kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(query),
      static_cast<const __nv_bfloat16*>(key),
      static_cast<const __nv_bfloat16*>(value),
      static_cast<const __nv_bfloat16*>(query_bias),
      static_cast<const __nv_bfloat16*>(key_bias),
      static_cast<const __nv_bfloat16*>(value_bias),
      static_cast<__nv_bfloat16*>(query_output),
      static_cast<__nv_bfloat16*>(key_output),
      static_cast<__nv_bfloat16*>(value_output), sequence, theta,
      static_cast<const uint32_t*>(positions),
      static_cast<const uint32_t*>(group_indices));
  return cudaGetLastError();
}

extern "C" cudaError_t
apxinf_static_qwen25_omni_vision_packed_qkv_bias_rope_bf16(
    const void* packed_qkv, const void* query_bias, const void* key_bias,
    const void* value_bias, void* query_output, void* key_output,
    void* value_output, int sequence, int heads, int head_dim, float theta,
    const void* positions, cudaStream_t stream) {
  if (packed_qkv == nullptr || query_bias == nullptr || key_bias == nullptr ||
      value_bias == nullptr || query_output == nullptr || key_output == nullptr ||
      value_output == nullptr || positions == nullptr || sequence <= 0 ||
      sequence > 65535 || heads != 16 || head_dim != 80 || theta != 10000.0f) {
    return cudaErrorInvalidValue;
  }
  constexpr int kHidden = 16 * 80;
  constexpr int kThreads = 256;
  dim3 grid((kHidden + kThreads - 1) / kThreads, sequence, 3);
  const auto* input = static_cast<const __nv_bfloat16*>(packed_qkv);
  qwen25_omni_vision_qkv_bias_rope_bf16_kernel<16, 80, false, true><<<
      grid, kThreads, 0, stream>>>(
      input, input, input,
      static_cast<const __nv_bfloat16*>(query_bias),
      static_cast<const __nv_bfloat16*>(key_bias),
      static_cast<const __nv_bfloat16*>(value_bias),
      static_cast<__nv_bfloat16*>(query_output),
      static_cast<__nv_bfloat16*>(key_output),
      static_cast<__nv_bfloat16*>(value_output), sequence, theta,
      static_cast<const uint32_t*>(positions), nullptr);
  return cudaGetLastError();
}

extern "C" cudaError_t
apxinf_static_qwen25_omni_vision_packed_grouped_qkv_bias_rope_bf16(
    const void* packed_qkv, const void* query_bias, const void* key_bias,
    const void* value_bias, void* query_output, void* key_output,
    void* value_output, int sequence, int heads, int head_dim, float theta,
    const void* positions, const void* group_indices, cudaStream_t stream) {
  if (packed_qkv == nullptr || query_bias == nullptr || key_bias == nullptr ||
      value_bias == nullptr || query_output == nullptr || key_output == nullptr ||
      value_output == nullptr || positions == nullptr || group_indices == nullptr ||
      sequence <= 0 || sequence > 65535 || heads != 16 || head_dim != 80 ||
      theta != 10000.0f) {
    return cudaErrorInvalidValue;
  }
  constexpr int kHidden = 16 * 80;
  constexpr int kThreads = 256;
  dim3 grid((kHidden + kThreads - 1) / kThreads, sequence, 3);
  const auto* input = static_cast<const __nv_bfloat16*>(packed_qkv);
  qwen25_omni_vision_qkv_bias_rope_bf16_kernel<16, 80, true, true><<<
      grid, kThreads, 0, stream>>>(
      input, input, input,
      static_cast<const __nv_bfloat16*>(query_bias),
      static_cast<const __nv_bfloat16*>(key_bias),
      static_cast<const __nv_bfloat16*>(value_bias),
      static_cast<__nv_bfloat16*>(query_output),
      static_cast<__nv_bfloat16*>(key_output),
      static_cast<__nv_bfloat16*>(value_output), sequence, theta,
      static_cast<const uint32_t*>(positions),
      static_cast<const uint32_t*>(group_indices));
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_qwen25_omni_vision_bias_residual_exact_bf16(
    const void* projection, const void* bias, const void* residual,
    void* output, int sequence, int hidden, cudaStream_t stream) {
  if (projection == nullptr || bias == nullptr || residual == nullptr ||
      output == nullptr || sequence <= 0 || sequence > 65535 || hidden != 1280) {
    return cudaErrorInvalidValue;
  }
  constexpr int kHidden = 1280;
  constexpr int kThreads = 256;
  dim3 grid((kHidden + kThreads - 1) / kThreads, sequence, 1);
  qwen25_omni_vision_bias_residual_exact_bf16_kernel<kHidden><<<
      grid, kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(projection),
      static_cast<const __nv_bfloat16*>(bias),
      static_cast<const __nv_bfloat16*>(residual),
      static_cast<__nv_bfloat16*>(output), sequence);
  return cudaGetLastError();
}

extern "C" cudaError_t
apxinf_static_qwen25_omni_vision_gate_up_bias_silu_mul_exact_bf16(
    const void* gate, const void* gate_bias, const void* up,
    const void* up_bias, void* output, int sequence, int intermediate,
    cudaStream_t stream) {
  if (gate == nullptr || gate_bias == nullptr || up == nullptr ||
      up_bias == nullptr || output == nullptr || sequence <= 0 ||
      sequence > 65535 || intermediate != 3420) {
    return cudaErrorInvalidValue;
  }
  constexpr int kIntermediate = 3420;
  constexpr int kThreads = 256;
  dim3 grid((kIntermediate + kThreads - 1) / kThreads, sequence, 1);
  qwen25_omni_vision_gate_up_bias_silu_mul_exact_bf16_kernel<kIntermediate, false><<<
      grid, kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(gate),
      static_cast<const __nv_bfloat16*>(gate_bias),
      static_cast<const __nv_bfloat16*>(up),
      static_cast<const __nv_bfloat16*>(up_bias),
      static_cast<__nv_bfloat16*>(output), sequence);
  return cudaGetLastError();
}

extern "C" cudaError_t
apxinf_static_qwen25_omni_vision_packed_gate_up_bias_silu_mul_exact_bf16(
    const void* packed_gate_up, const void* gate_bias,
    const void* up_bias, void* output, int sequence, int intermediate,
    cudaStream_t stream) {
  if (packed_gate_up == nullptr || gate_bias == nullptr ||
      up_bias == nullptr || output == nullptr || sequence <= 0 ||
      sequence > 65535 || intermediate != 3420) {
    return cudaErrorInvalidValue;
  }
  constexpr int kIntermediate = 3420;
  constexpr int kThreads = 256;
  dim3 grid((kIntermediate + kThreads - 1) / kThreads, sequence, 1);
  const auto* packed = static_cast<const __nv_bfloat16*>(packed_gate_up);
  qwen25_omni_vision_gate_up_bias_silu_mul_exact_bf16_kernel<
      kIntermediate, true><<<grid, kThreads, 0, stream>>>(
      packed, static_cast<const __nv_bfloat16*>(gate_bias),
      packed, static_cast<const __nv_bfloat16*>(up_bias),
      static_cast<__nv_bfloat16*>(output), sequence);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_qwen25_omni_attention_flash_split_cta_bf16(
    const void* query, const void* key_cache, const void* value_cache,
    void* partial_max, void* partial_sum, void* partial_accumulator,
    void* output, int split_count, int bucket_kv_len, int max_seq_len,
    float scale, const void* position, cudaStream_t stream) {
  if (query == nullptr || key_cache == nullptr || value_cache == nullptr ||
      partial_max == nullptr || partial_sum == nullptr ||
      partial_accumulator == nullptr || output == nullptr ||
      position == nullptr || split_count < 2 || split_count > 40 ||
      split_count % 8 != 0 || bucket_kv_len <= 0 ||
      bucket_kv_len > max_seq_len || !(scale > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  dim3 stage_grid(16, split_count, 1);
  attention_flash_split_cta_bf16_kernel<16, 2, 128, 8><<<
      stage_grid, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(query),
      static_cast<const __nv_bfloat16*>(key_cache),
      static_cast<const __nv_bfloat16*>(value_cache),
      static_cast<float*>(partial_max), static_cast<float*>(partial_sum),
      static_cast<float*>(partial_accumulator), split_count, bucket_kv_len,
      max_seq_len, scale, static_cast<const uint32_t*>(position));
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  attention_flash_split_cta_reduce_bf16_kernel<128, 40><<<
      16, 128, 0, stream>>>(
      static_cast<const float*>(partial_max),
      static_cast<const float*>(partial_sum),
      static_cast<const float*>(partial_accumulator),
      static_cast<__nv_bfloat16*>(output), split_count);
  return cudaGetLastError();
}

extern "C" cudaError_t
apxinf_static_qwen25_omni_attention_flash_grouped2_split_cta_bf16(
    const void* query, const void* key_cache, const void* value_cache,
    void* partial_max, void* partial_sum, void* partial_accumulator,
    void* output, int split_count, int bucket_kv_len, int max_seq_len,
    float scale, const void* position, cudaStream_t stream) {
  if (query == nullptr || key_cache == nullptr || value_cache == nullptr ||
      partial_max == nullptr || partial_sum == nullptr ||
      partial_accumulator == nullptr || output == nullptr ||
      position == nullptr || split_count != 48 || bucket_kv_len <= 0 ||
      bucket_kv_len > max_seq_len || !(scale > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  dim3 stage_grid(8, split_count, 1);
  attention_flash_grouped_split_cta_bf16_kernel<16, 2, 128, 2, 8><<<
      stage_grid, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(query),
      static_cast<const __nv_bfloat16*>(key_cache),
      static_cast<const __nv_bfloat16*>(value_cache),
      static_cast<float*>(partial_max), static_cast<float*>(partial_sum),
      static_cast<float*>(partial_accumulator), split_count, bucket_kv_len,
      max_seq_len, scale, static_cast<const uint32_t*>(position));
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  attention_flash_split_cta_reduce_bf16_kernel<128, 48><<<
      16, 128, 0, stream>>>(
      static_cast<const float*>(partial_max),
      static_cast<const float*>(partial_sum),
      static_cast<const float*>(partial_accumulator),
      static_cast<__nv_bfloat16*>(output), split_count);
  return cudaGetLastError();
}

extern "C" cudaError_t
apxinf_static_qwen25_omni_attention_flash_grouped4_split_cta_bf16(
    const void* query, const void* key_cache, const void* value_cache,
    void* partial_max, void* partial_sum, void* partial_accumulator,
    void* output, int split_count, int bucket_kv_len, int max_seq_len,
    float scale, const void* position, cudaStream_t stream) {
  if (query == nullptr || key_cache == nullptr || value_cache == nullptr ||
      partial_max == nullptr || partial_sum == nullptr ||
      partial_accumulator == nullptr || output == nullptr ||
      position == nullptr || split_count != 64 || bucket_kv_len <= 0 ||
      bucket_kv_len > max_seq_len || !(scale > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  dim3 stage_grid(4, split_count, 1);
  attention_flash_grouped_split_cta_bf16_kernel<16, 2, 128, 4, 8><<<
      stage_grid, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(query),
      static_cast<const __nv_bfloat16*>(key_cache),
      static_cast<const __nv_bfloat16*>(value_cache),
      static_cast<float*>(partial_max), static_cast<float*>(partial_sum),
      static_cast<float*>(partial_accumulator), split_count, bucket_kv_len,
      max_seq_len, scale, static_cast<const uint32_t*>(position));
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  attention_flash_split_cta_reduce_bf16_kernel<128, 64><<<
      16, 128, 0, stream>>>(
      static_cast<const float*>(partial_max),
      static_cast<const float*>(partial_sum),
      static_cast<const float*>(partial_accumulator),
      static_cast<__nv_bfloat16*>(output), split_count);
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
  const int packed8_mode = kActionAdaPacked8Mode;
  if (packed8_mode < 0) return cudaErrorInvalidValue;
  const bool packed8_exact_shape = rows == 10 && cols == 1024;
  if (packed8_mode == 1 && !packed8_exact_shape) return cudaErrorInvalidValue;
  if (packed8_mode != 0 && packed8_exact_shape) {
    if (!std::isfinite(scale) ||
        projection == nullptr || residual == nullptr || gate_style == nullptr ||
        norm_style == nullptr || hidden == nullptr || normalized == nullptr) {
      return cudaErrorInvalidValue;
    }
    ada_gate_residual_rms_norm_quant_f16_e4m3_packed8_kernel
        <<<rows, 256, 0, stream>>>(
            static_cast<const half*>(projection),
            static_cast<const half*>(residual),
            static_cast<const half*>(gate_style),
            static_cast<const half*>(norm_style), static_cast<half*>(hidden),
            static_cast<__nv_fp8_e4m3*>(normalized), eps, 1.0f / scale);
    return cudaGetLastError();
  }
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
