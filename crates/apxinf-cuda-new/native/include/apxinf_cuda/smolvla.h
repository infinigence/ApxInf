#pragma once

#include <cuda_runtime.h>

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* SmolVLA FP16 kernel surface plus the BF16 pixel-shuffle. Copied verbatim
   from the legacy apxinf-cuda adapters with the apxinf_smolvla_ prefix,
   because the legacy crate links into the same binary and the legacy symbols
   are already present. */

cudaError_t apxinf_smolvla_rms_norm_f16(
    const void* input, const void* weight, void* output,
    int rows, int cols, float eps, cudaStream_t stream);

cudaError_t apxinf_smolvla_layer_norm_f16(
    const void* input, const void* weight, const void* bias, void* output,
    int rows, int cols, float eps, cudaStream_t stream);

cudaError_t apxinf_smolvla_swiglu_f16(
    const void* gate_up, void* output, int rows, int inner,
    cudaStream_t stream);

cudaError_t apxinf_smolvla_bias_gelu_f16(
    const void* input, const void* bias, void* output,
    int rows, int cols, cudaStream_t stream);

cudaError_t apxinf_smolvla_bias_silu_f16(
    const void* input, const void* bias, void* output, int rows, int cols,
    cudaStream_t stream);

cudaError_t apxinf_smolvla_bias_f16(
    const void* input, const void* bias, void* output, int rows, int cols,
    cudaStream_t stream);

cudaError_t apxinf_smolvla_euler_update_f16(
    const void* state, const void* velocity, void* output, int64_t count,
    float dt, cudaStream_t stream);

cudaError_t apxinf_smolvla_add_f16(
    const void* a, const void* b, void* output, uint32_t count,
    cudaStream_t stream);

cudaError_t apxinf_smolvla_scale_f16(
    const void* input, void* output, uint32_t count, float scale,
    cudaStream_t stream);

cudaError_t apxinf_smolvla_pixel_shuffle_4_bf16(
    const void* input, void* output, int views, int tokens_per_view, int width,
    cudaStream_t stream);

cudaError_t apxinf_smolvla_pixel_shuffle_4_f16(
    const void* input, void* output, int views, int tokens_per_view, int width,
    cudaStream_t stream);

cudaError_t apxinf_smolvla_rope_half_split_f16(
    const void* input, void* output, uint32_t head_dim, uint32_t n_heads,
    uint32_t seq_len, float rope_theta, uint32_t pos_offset,
    cudaStream_t stream);

/* cuBLAS-composed FP16 GQA. Returns a cublasStatus_t as int (0 == success),
   matching the legacy adapters. */

int apxinf_smolvla_cublas_gqa_f16(
    const void* q, const void* k, const void* v, void* output,
    int query_tokens, int key_tokens, int q_heads, int kv_heads,
    int head_dim, cudaStream_t stream);

int apxinf_smolvla_cublas_gqa_causal_f16(
    const void* q, const void* k, const void* v, void* output,
    int query_tokens, int key_tokens, int q_heads, int kv_heads,
    int head_dim, int key_offset, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
