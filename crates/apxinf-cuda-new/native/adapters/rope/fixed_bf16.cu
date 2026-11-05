#include "../../kernels/rope/fixed_bf16.cuh"
#include "../../framework/runtime_internal.h"
#include "../../include/apxinf_cuda/types.h"

#include <cstdlib>

extern "C" cudaError_t apxinf_rope_build_vision_rotation_table_f32(
    const void* pos_ids, void* rotation_table, uint32_t head_dim,
    uint32_t seq_len, float theta, cudaStream_t stream) {
  if (pos_ids == nullptr || rotation_table == nullptr || head_dim == 0 ||
      (head_dim % 2) != 0 || seq_len == 0 || !(theta > 0.0f))
    return cudaErrorInvalidValue;
  const uint64_t total = static_cast<uint64_t>(seq_len) * (head_dim / 2);
  dim3 grid(static_cast<uint32_t>((total + kBlockSize - 1) /
                                  kBlockSize),
            1, 1);
  dim3 block(kBlockSize, 1, 1);
  build_vision_rotation_table_f32_kernel<<<grid, block, 0, stream>>>(
      static_cast<const uint32_t*>(pos_ids),
      static_cast<float2*>(rotation_table), head_dim, seq_len, theta);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_rope_vision_2d_pair_bf16(
    const void* q, const void* k, void* q_out, void* k_out, uint32_t head_dim,
    uint32_t n_heads, uint32_t seq_len, float theta, const void* pos_ids,
    cudaStream_t stream) {
  uint64_t total = static_cast<uint64_t>(seq_len) * n_heads * (head_dim / 2);
  dim3 grid(static_cast<uint32_t>((total + kBlockSize - 1) /
                                  kBlockSize),
            1, 1);
  dim3 block(kBlockSize, 1, 1);
  rope_vision_2d_pair_bf16_kernel<<<grid, block, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(q),
      static_cast<const __nv_bfloat16*>(k),
      static_cast<__nv_bfloat16*>(q_out), static_cast<__nv_bfloat16*>(k_out),
      head_dim, n_heads, seq_len, theta,
      static_cast<const uint32_t*>(pos_ids));
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_rope_qk_rms_norm_mrope_bf16(
    apxinf_runtime_t runtime, const void* query_input,
    const void* query_weight, void* query_output,
    const void* key_input, const void* key_weight, void* key_output,
    uint32_t head_dim, uint32_t query_heads, uint32_t key_heads,
    uint32_t seq_len, float eps, float theta, const void* pos_ids,
    uint32_t sec_h, uint32_t sec_w, cudaStream_t stream) {
  if (runtime == nullptr || query_input == nullptr || query_weight == nullptr ||
      query_output == nullptr || key_input == nullptr ||
      key_weight == nullptr || key_output == nullptr || pos_ids == nullptr ||
      head_dim == 0 || (head_dim % 2) != 0 || query_heads == 0 || key_heads == 0 ||
      seq_len == 0 || !(eps > 0.0f) || !(theta > 0.0f))
    return cudaErrorInvalidValue;
  const int sm = runtime->compute_major * 10 + runtime->compute_minor;
  const bool measured_arch_shape =
      (sm == 110 && (seq_len == 90 || seq_len == 156)) ||
      (sm == 87 && seq_len == 156);
  const bool legacy_threads =
      (sm == 110 &&
       std::getenv("APXINF_GR00T_LEGACY_QK_MROPE_THREADS") != nullptr) ||
      (sm == 87 &&
       std::getenv("APXINF_GR00T_BF16_LEGACY_QK_MROPE_THREADS") != nullptr);
  const uint32_t block_threads =
      !legacy_threads && measured_arch_shape && head_dim == 128 &&
              query_heads == 16 && key_heads == 8
          ? 128
          : 256;
  if (head_dim / 2 > block_threads) return cudaErrorInvalidValue;
  dim3 grid(seq_len, query_heads + key_heads, 1);
  dim3 block(block_threads, 1, 1);
  qk_rms_norm_mrope_bf16_kernel<<<grid, block,
                                  head_dim * sizeof(__nv_bfloat16), stream>>>(
      static_cast<const __nv_bfloat16*>(query_input),
      static_cast<const __nv_bfloat16*>(query_weight),
      static_cast<__nv_bfloat16*>(query_output),
      static_cast<const __nv_bfloat16*>(key_input),
      static_cast<const __nv_bfloat16*>(key_weight),
      static_cast<__nv_bfloat16*>(key_output), head_dim, query_heads,
      key_heads, seq_len, eps, theta, static_cast<const uint32_t*>(pos_ids),
      sec_h, sec_w);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_rope_qkv_split_bias_vision_rope_bf16(
    const void* qkv, const void* bias, void* q_out, void* k_out, void* v_out,
    uint32_t head_dim, uint32_t n_heads, uint32_t seq_len, float theta,
    const void* pos_ids, cudaStream_t stream) {
  uint64_t total = static_cast<uint64_t>(seq_len) * n_heads * (head_dim / 2);
  dim3 grid(static_cast<uint32_t>((total + kBlockSize - 1) /
                                  kBlockSize),
            1, 1);
  dim3 block(kBlockSize, 1, 1);
  qkv_split_bias_vision_rope_bf16_kernel<false><<<grid, block, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(qkv),
      static_cast<const __nv_bfloat16*>(bias),
      static_cast<__nv_bfloat16*>(q_out), static_cast<__nv_bfloat16*>(k_out),
      static_cast<__nv_bfloat16*>(v_out), head_dim, n_heads, seq_len, theta,
      static_cast<const uint32_t*>(pos_ids), nullptr);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_rope_qkv_split_bias_vision_rope_precomputed_bf16(
    const void* qkv, const void* bias, void* q_out, void* k_out, void* v_out,
    uint32_t head_dim, uint32_t n_heads, uint32_t seq_len,
    const void* rotation_table, cudaStream_t stream) {
  if (qkv == nullptr || bias == nullptr || q_out == nullptr ||
      k_out == nullptr || v_out == nullptr || rotation_table == nullptr ||
      head_dim == 0 || (head_dim % 2) != 0 || n_heads == 0 || seq_len == 0)
    return cudaErrorInvalidValue;
  uint64_t total = static_cast<uint64_t>(seq_len) * n_heads * (head_dim / 2);
  dim3 grid(static_cast<uint32_t>((total + kBlockSize - 1) /
                                  kBlockSize),
            1, 1);
  dim3 block(kBlockSize, 1, 1);
  qkv_split_bias_vision_rope_bf16_kernel<true><<<grid, block, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(qkv),
      static_cast<const __nv_bfloat16*>(bias),
      static_cast<__nv_bfloat16*>(q_out), static_cast<__nv_bfloat16*>(k_out),
      static_cast<__nv_bfloat16*>(v_out), head_dim, n_heads, seq_len, 0.0f,
      nullptr, static_cast<const float2*>(rotation_table));
  return cudaGetLastError();
}

extern "C" cudaError_t
apxinf_rope_qkv_split_bias_vision_rope_precomputed_vec2_bf16(
    const void* qkv, const void* bias, void* q_out, void* k_out, void* v_out,
    uint32_t head_dim, uint32_t n_heads, uint32_t seq_len,
    const void* rotation_table, cudaStream_t stream) {
  if (qkv == nullptr || bias == nullptr || q_out == nullptr ||
      k_out == nullptr || v_out == nullptr || rotation_table == nullptr ||
      head_dim == 0 || head_dim % 4 != 0 || n_heads == 0 || seq_len == 0)
    return cudaErrorInvalidValue;
  const uint64_t total =
      static_cast<uint64_t>(seq_len) * n_heads * (head_dim / 4);
  constexpr uint32_t threads = 128;
  const dim3 grid(static_cast<uint32_t>((total + threads - 1) / threads), 1, 1);
  qkv_split_bias_vision_rope_precomputed_bf16_vec2_kernel<<<grid, threads, 0,
                                                            stream>>>(
      static_cast<const __nv_bfloat16*>(qkv),
      static_cast<const __nv_bfloat16*>(bias),
      static_cast<__nv_bfloat16*>(q_out), static_cast<__nv_bfloat16*>(k_out),
      static_cast<__nv_bfloat16*>(v_out), head_dim, n_heads, seq_len,
      static_cast<const float2*>(rotation_table));
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_rope_split_strided_qkv_bf16(
    const void* qkv, void* q, void* k, void* v, int tokens, int hidden,
    cudaStream_t stream) {
  if (qkv == nullptr || q == nullptr || k == nullptr || v == nullptr ||
      tokens <= 0 || hidden <= 0)
    return cudaErrorInvalidValue;
  const int64_t total = static_cast<int64_t>(tokens) * hidden;
  const int blocks =
      static_cast<int>((total + kBlockSize - 1) / kBlockSize);
  split_strided_qkv_bf16_kernel<<<blocks, kBlockSize, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(qkv), static_cast<__nv_bfloat16*>(q),
      static_cast<__nv_bfloat16*>(k), static_cast<__nv_bfloat16*>(v), tokens,
      hidden);
  return cudaGetLastError();
}
