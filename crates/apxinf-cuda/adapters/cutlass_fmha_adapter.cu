// Copyright 2026 apxinf contributors.
// Stable C ABI adapter for the CUTLASS SM100/SM110 FMHA operator.

#include "../kernels/cutlass/fmha_sm100.cu"

extern "C" int apxinf_static_prepare_cutlass_mha_f16(
    const void* q, const void* k, const void* v, void* output,
    int batches, int query_tokens, int key_tokens, int query_heads,
    int kv_heads, int head_dim, cudaStream_t stream) {
  return apxinf::cuda::cutlass_ops::prepare_mha_f16(
      q, k, v, output, batches, query_tokens, key_tokens, query_heads,
      kv_heads, head_dim, stream);
}

extern "C" int apxinf_static_cutlass_mha_f16(
    const void* q, const void* k, const void* v, void* output,
    int batches, int query_tokens, int key_tokens, int query_heads,
    int kv_heads, int head_dim, cudaStream_t stream) {
  return apxinf::cuda::cutlass_ops::mha_f16(
      q, k, v, output, batches, query_tokens, key_tokens, query_heads,
      kv_heads, head_dim, stream);
}

extern "C" int apxinf_static_prepare_cutlass_mha_packed_qkv_f16(
    const void* qkv, void* output, int batches, int tokens, int heads,
    int head_dim, cudaStream_t stream) {
  return apxinf::cuda::cutlass_ops::prepare_mha_packed_qkv_f16(
      qkv, output, batches, tokens, heads, head_dim, stream);
}

extern "C" int apxinf_static_cutlass_mha_packed_qkv_f16(
    const void* qkv, void* output, int batches, int tokens, int heads,
    int head_dim, cudaStream_t stream) {
  return apxinf::cuda::cutlass_ops::mha_packed_qkv_f16(
      qkv, output, batches, tokens, heads, head_dim, stream);
}

extern "C" int apxinf_static_prepare_cutlass_mha_bf16(
    const void* q, const void* k, const void* v, void* output,
    int batches, int query_tokens, int key_tokens, int query_heads,
    int kv_heads, int head_dim, cudaStream_t stream) {
  return apxinf::cuda::cutlass_ops::prepare_mha_bf16(
      q, k, v, output, batches, query_tokens, key_tokens, query_heads,
      kv_heads, head_dim, stream);
}

extern "C" int apxinf_static_cutlass_mha_bf16(
    const void* q, const void* k, const void* v, void* output,
    int batches, int query_tokens, int key_tokens, int query_heads,
    int kv_heads, int head_dim, cudaStream_t stream) {
  return apxinf::cuda::cutlass_ops::mha_bf16(
      q, k, v, output, batches, query_tokens, key_tokens, query_heads,
      kv_heads, head_dim, stream);
}
