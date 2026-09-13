// Copyright 2026 apxinf contributors.
// Stable C ABI adapter for the vendored FlashAttention-2 BF16 operator.

#include "../kernels/cutlass/fa2_bf16_sm80.cu"

#if defined(APXINF_FA2_SM80)
namespace FLASH_NAMESPACE {
int run_bf16_head64_splitkv(Flash_fwd_params& params, cudaStream_t stream);
int run_bf16_head256_causal(Flash_fwd_params& params, cudaStream_t stream);
int run_bf16_head256_splitkv(Flash_fwd_params& params, cudaStream_t stream);
}
#endif

extern "C" int apxinf_static_fa2_bf16(
    const void* q, const void* k, const void* v, void* output,
    void* softmax_lse, int batch, int query_tokens, int key_tokens,
    int query_heads, int kv_heads, int head_dim, float softmax_scale,
    cudaStream_t stream) {
  return apxinf::cuda::cutlass_ops::fa2_bf16(
      q, k, v, output, softmax_lse, batch, query_tokens, key_tokens,
      query_heads, kv_heads, head_dim, softmax_scale, stream);
}

extern "C" int apxinf_static_fa2_bf16_causal(
    const void* q, const void* k, const void* v, void* output,
    void* softmax_lse, int batch, int query_tokens, int key_tokens,
    int query_heads, int kv_heads, int head_dim, float softmax_scale,
    cudaStream_t stream) {
#if defined(APXINF_FA2_SM80)
  if(head_dim==256 && query_tokens==key_tokens && query_tokens>1) {
    if(!q||!k||!v||!output||!softmax_lse||batch<=0||query_heads<=0||kv_heads<=0||query_heads%kv_heads)
      return static_cast<int>(cudaErrorInvalidValue);
    FLASH_NAMESPACE::Flash_fwd_params params;
    fill_params(params,true,q,k,v,output,softmax_lse,batch,query_tokens,key_tokens,
                query_heads,kv_heads,head_dim,softmax_scale);
    params.is_causal=true;params.window_size_right=0;
    return FLASH_NAMESPACE::run_bf16_head256_causal(params,stream);
  }
#endif
  return apxinf::cuda::cutlass_ops::fa2_bf16_causal(
      q, k, v, output, softmax_lse, batch, query_tokens, key_tokens,
      query_heads, kv_heads, head_dim, softmax_scale, stream);
}

#if defined(APXINF_FA2_SPLITKV)
extern "C" int apxinf_static_fa2_bf16_splitkv(
    const void* q, const void* k, const void* v, void* output,
    void* softmax_lse, void* softmax_lse_accum, void* o_accum, int batch,
    int query_tokens, int key_tokens, int query_heads, int kv_heads,
    int head_dim, float softmax_scale, int num_sms, cudaStream_t stream) {
#if defined(APXINF_FA2_SM80)
  if(head_dim==256 && query_tokens>0 && query_tokens<=64) {
    if(!q||!k||!v||!output||!softmax_lse||!softmax_lse_accum||!o_accum||batch<=0||key_tokens<=0||
        query_heads<=0||kv_heads<=0||query_heads%kv_heads||num_sms<=0)
      return static_cast<int>(cudaErrorInvalidValue);
    FLASH_NAMESPACE::Flash_fwd_params params;
    fill_params(params,true,q,k,v,output,softmax_lse,batch,query_tokens,key_tokens,
                query_heads,kv_heads,head_dim,softmax_scale);
    if(query_tokens==1 && query_heads>kv_heads) {
      const int groups=query_heads/kv_heads;
      params.seqlen_q=groups;params.h=kv_heads;params.h_h_k_ratio=1;
      params.q_row_stride=params.o_row_stride=head_dim;
      params.q_head_stride=params.o_head_stride=groups*head_dim;
      params.seqlen_q_rounded=((groups+127)/128)*128;
    }
    setup_splitkv(params,softmax_lse_accum,o_accum,num_sms,params.seqlen_q,
                  key_tokens,head_dim,batch,params.h);
    return FLASH_NAMESPACE::run_bf16_head256_splitkv(params,stream);
  }
  if (head_dim == 64) {
    if (!q || !k || !v || !output || !softmax_lse || !softmax_lse_accum || !o_accum ||
        batch <= 0 || query_tokens <= 0 || key_tokens <= 0 || query_heads <= 0 ||
        kv_heads <= 0 || query_heads % kv_heads != 0 || num_sms <= 0)
      return static_cast<int>(cudaErrorInvalidValue);
    FLASH_NAMESPACE::Flash_fwd_params params;
    fill_params(params, true, q, k, v, output, softmax_lse, batch, query_tokens,
                key_tokens, query_heads, kv_heads, head_dim, softmax_scale);
    setup_splitkv(params, softmax_lse_accum, o_accum, num_sms, query_tokens,
                  key_tokens, head_dim, batch, query_heads);
    return FLASH_NAMESPACE::run_bf16_head64_splitkv(params, stream);
  }
#endif
  return apxinf::cuda::cutlass_ops::fa2_bf16_splitkv(
      q, k, v, output, softmax_lse, softmax_lse_accum, o_accum, batch,
      query_tokens, key_tokens, query_heads, kv_heads, head_dim, softmax_scale,
      num_sms, stream);
}

extern "C" int apxinf_static_fa2_bf16_causal_splitkv(
    const void* q, const void* k, const void* v, void* output,
    void* softmax_lse, void* softmax_lse_accum, void* o_accum, int batch,
    int query_tokens, int key_tokens, int query_heads, int kv_heads,
    int head_dim, float softmax_scale, int num_sms, cudaStream_t stream) {
  return apxinf::cuda::cutlass_ops::fa2_bf16_causal_splitkv(
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
