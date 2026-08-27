// Copyright 2026 apxinf contributors.
// Stable C ABI adapter for the vendored FlashAttention-2 BF16 operator.

#include "../kernels/cutlass/fa2_bf16_sm80.cu"

extern "C" int apxinf_static_fa2_bf16(
    const void* q, const void* k, const void* v, void* output,
    void* softmax_lse, int batch, int query_tokens, int key_tokens,
    int query_heads, int kv_heads, int head_dim, float softmax_scale,
    cudaStream_t stream) {
  return apxinf::cuda::cutlass_ops::fa2_bf16(
      q, k, v, output, softmax_lse, batch, query_tokens, key_tokens,
      query_heads, kv_heads, head_dim, softmax_scale, stream);
}

extern "C" int apxinf_static_fa2_varlen_bf16(
    const void* q, const void* k, const void* v, void* output,
    void* softmax_lse, const void* cu_seqlens, int batch,
    int total_tokens, int max_tokens, int query_heads, int kv_heads,
    int head_dim, float softmax_scale, cudaStream_t stream) {
  return apxinf::cuda::cutlass_ops::fa2_varlen_bf16(
      q, k, v, output, softmax_lse,
      static_cast<const int*>(cu_seqlens), batch, total_tokens, max_tokens,
      query_heads, kv_heads, head_dim, softmax_scale, stream);
}

#if defined(APXINF_FA2_CAUSAL_HDIM128)
extern "C" int apxinf_static_fa2_causal_strided_kv_bf16(
    const void* q, const void* k, const void* v, void* output,
    void* softmax_lse, int query_tokens, int key_tokens,
    int query_heads, int kv_heads, int head_dim, int max_seq_len,
    float softmax_scale, cudaStream_t stream) {
  return apxinf::cuda::cutlass_ops::fa2_causal_strided_kv_bf16(
      q, k, v, output, softmax_lse, query_tokens, key_tokens,
      query_heads, kv_heads, head_dim, max_seq_len, softmax_scale, stream);
}
#endif

#if defined(APXINF_FA2_SM80)
extern "C" int apxinf_static_fa2_bf16_splitkv(
    const void* q, const void* k, const void* v, void* output,
    void* softmax_lse, void* softmax_lse_accum, void* o_accum, int batch,
    int query_tokens, int key_tokens, int query_heads, int kv_heads,
    int head_dim, float softmax_scale, int num_sms, cudaStream_t stream) {
  return apxinf::cuda::cutlass_ops::fa2_bf16_splitkv(
      q, k, v, output, softmax_lse, softmax_lse_accum, o_accum, batch,
      query_tokens, key_tokens, query_heads, kv_heads, head_dim, softmax_scale,
      num_sms, stream);
}
#endif

extern "C" int apxinf_static_fa2_f16(
    const void* q, const void* k, const void* v, void* output,
    void* softmax_lse, int batch, int query_tokens, int key_tokens,
    int query_heads, int kv_heads, int head_dim, float softmax_scale,
    cudaStream_t stream) {
  return apxinf::cuda::cutlass_ops::fa2_f16(
      q, k, v, output, softmax_lse, batch, query_tokens, key_tokens,
      query_heads, kv_heads, head_dim, softmax_scale, stream);
}
