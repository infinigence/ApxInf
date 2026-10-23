#pragma once

#include <cuda_runtime.h>

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* LLM decode fast-path primitives for the llama / qwen3-vl allocation-free
   decode graph. Copied verbatim from the legacy apxinf-cuda adapters with the
   apxinf_llmdec_ prefix, because the legacy crate links into the same binary
   and the legacy symbols are already present. Launch policy matches the
   legacy core_kernels_adapter.cu entries byte for byte. */

cudaError_t apxinf_llmdec_silu_f32(
    const void* input, void* output, uint32_t count, void* stream);

cudaError_t apxinf_llmdec_silu_bf16(
    const void* input, void* output, uint32_t count, void* stream);

cudaError_t apxinf_llmdec_silu_mul_bf16(
    const void* gate_up, void* output, uint32_t inter, void* stream);

cudaError_t apxinf_llmdec_add_f32(
    const void* a, const void* b, void* output, uint32_t count, void* stream);

cudaError_t apxinf_llmdec_add_bf16(
    const void* a, const void* b, void* output, uint32_t count, void* stream);

cudaError_t apxinf_llmdec_mul_f32(
    const void* a, const void* b, void* output, uint32_t count, void* stream);

cudaError_t apxinf_llmdec_mul_bf16(
    const void* a, const void* b, void* output, uint32_t count, void* stream);

cudaError_t apxinf_llmdec_scale_f32(
    const void* input, void* output, uint32_t count, float scale, void* stream);

cudaError_t apxinf_llmdec_scale_bf16(
    const void* input, void* output, uint32_t count, float scale, void* stream);

cudaError_t apxinf_llmdec_embedding_f32(
    const void* table, const void* ids, void* output,
    uint32_t embed_dim, uint32_t seq_len, void* stream);

cudaError_t apxinf_llmdec_embedding_bf16(
    const void* table, const void* ids, void* output,
    uint32_t embed_dim, uint32_t seq_len, void* stream);

cudaError_t apxinf_llmdec_rms_norm_f32(
    const void* input, const void* weight, void* output,
    uint32_t cols, uint32_t rows, float eps, void* stream);

cudaError_t apxinf_llmdec_rms_norm_bf16(
    const void* input, const void* weight, void* output,
    uint32_t cols, uint32_t rows, float eps, void* stream);

cudaError_t apxinf_llmdec_rms_norm_add_bf16(
    void* x_inout, const void* delta, const void* weight, void* output,
    uint32_t cols, uint32_t rows, float eps, void* stream);

cudaError_t apxinf_llmdec_attention_softmax_decode_f32(
    const void* scores, void* output,
    uint32_t cols, uint32_t n_heads, const void* pos_ptr, void* stream);

cudaError_t apxinf_llmdec_attention_softmax_f32(
    const void* scores, void* output,
    uint32_t cols, uint32_t rows, uint32_t kv_offset,
    uint32_t n_heads, void* stream);

cudaError_t apxinf_llmdec_attention_softmax_bf16(
    const void* scores, void* output,
    uint32_t cols, uint32_t rows, uint32_t kv_offset,
    uint32_t n_heads, void* stream);

cudaError_t apxinf_llmdec_flash_attn_decode_bf16(
    const void* q, const void* k_cache, const void* v_cache, void* out,
    uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
    uint32_t bucket_kv_len, uint32_t max_seq_len,
    float scale, const void* pos_ptr, void* stream);

cudaError_t apxinf_llmdec_rope_decode_f32(
    const void* input, void* output,
    uint32_t head_dim, uint32_t n_heads,
    float rope_theta, const void* pos_ptr, void* stream);

cudaError_t apxinf_llmdec_rope_decode_bf16(
    const void* input, void* output,
    uint32_t head_dim, uint32_t n_heads,
    float rope_theta, const void* pos_ptr, void* stream);

cudaError_t apxinf_llmdec_rope_k_write_bf16(
    const void* k_in, void* k_cache,
    uint32_t head_dim, uint32_t n_kv_heads, uint32_t max_seq_len,
    float rope_theta, const void* pos_ptr, void* stream);

cudaError_t apxinf_llmdec_rope_mrope_decode_bf16(
    const void* input, void* output,
    uint32_t head_dim, uint32_t n_heads,
    float theta, const void* pos_ids,
    uint32_t sec_h, uint32_t sec_w, void* stream);

cudaError_t apxinf_llmdec_kv_cache_append_decode_f32(
    void* cache, const void* new_data,
    uint32_t n_kv_heads, uint32_t head_dim, uint32_t max_seq_len,
    const void* pos_ptr, void* stream);

cudaError_t apxinf_llmdec_kv_cache_append_f32(
    void* cache, const void* new_data,
    uint32_t n_kv_heads, uint32_t head_dim, uint32_t max_seq_len,
    uint32_t seq_len, uint32_t append_len, void* stream);

cudaError_t apxinf_llmdec_kv_cache_append_bf16(
    void* cache, const void* new_data,
    uint32_t n_kv_heads, uint32_t head_dim, uint32_t max_seq_len,
    uint32_t seq_len, uint32_t append_len, void* stream);

cudaError_t apxinf_llmdec_kv_cache_append_decode_bf16(
    void* cache, const void* new_data,
    uint32_t n_kv_heads, uint32_t head_dim, uint32_t max_seq_len,
    const void* pos_ptr, void* stream);

cudaError_t apxinf_llmdec_kv_cache_gather_bf16(
    const void* src, void* dst,
    uint32_t tokens, uint32_t n_kv_heads, uint32_t head_dim,
    uint32_t max_seq_len, uint32_t kv_offset, void* stream);

#ifdef __cplusplus
}
#endif
