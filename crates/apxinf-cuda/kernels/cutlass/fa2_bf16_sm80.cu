// Raw-pointer BF16 forward wrapper for the vendored FlashAttention-2 SM80
// kernels. The upstream kernel sources and their license live under fa2/.

#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>

#include <cstdint>
#include <type_traits>

#include "flash_attn/flash.h"
#include "flash_attn/namespace_config.h"

namespace FLASH_NAMESPACE {

template <typename Element, int HeadDim, bool IsCausal>
void run_mha_fwd_(Flash_fwd_params& params, cudaStream_t stream);

}  // namespace FLASH_NAMESPACE

namespace {

constexpr float kLog2E = 1.4426950408889634074f;

void fill_params(FLASH_NAMESPACE::Flash_fwd_params& params, bool is_bf16,
                 const void* q, const void* k, const void* v, void* output,
                 void* softmax_lse, int batch, int query_tokens,
                 int key_tokens, int query_heads, int kv_heads, int head_dim,
                 float softmax_scale) {
  params = {};
  params.is_bf16 = is_bf16;
  params.q_ptr = const_cast<void*>(q);
  params.k_ptr = const_cast<void*>(k);
  params.v_ptr = const_cast<void*>(v);
  params.o_ptr = output;
  params.softmax_lse_ptr = softmax_lse;

  const int64_t q_row_stride = static_cast<int64_t>(query_heads) * head_dim;
  const int64_t kv_row_stride = static_cast<int64_t>(kv_heads) * head_dim;
  params.q_batch_stride = static_cast<int64_t>(query_tokens) * q_row_stride;
  params.k_batch_stride = static_cast<int64_t>(key_tokens) * kv_row_stride;
  params.v_batch_stride = params.k_batch_stride;
  params.o_batch_stride = params.q_batch_stride;
  params.q_row_stride = q_row_stride;
  params.k_row_stride = kv_row_stride;
  params.v_row_stride = kv_row_stride;
  params.o_row_stride = q_row_stride;
  params.q_head_stride = head_dim;
  params.k_head_stride = head_dim;
  params.v_head_stride = head_dim;
  params.o_head_stride = head_dim;

  params.b = batch;
  params.h = query_heads;
  params.h_k = kv_heads;
  params.h_h_k_ratio = query_heads / kv_heads;
  params.seqlen_q = query_tokens;
  params.seqlen_k = key_tokens;
  params.seqlen_q_rounded = ((query_tokens + 127) / 128) * 128;
  params.seqlen_k_rounded = ((key_tokens + 127) / 128) * 128;
  params.d = head_dim;
  params.d_rounded = (head_dim + 31) & ~31;

  params.scale_softmax = softmax_scale;
  params.scale_softmax_log2 = softmax_scale * kLog2E;
  params.scale_softmax_rp_dropout = softmax_scale;
  params.p_dropout = 1.0f;
  params.p_dropout_in_uint8_t = 255;
  params.rp_dropout = 1.0f;

  params.is_causal = false;
  params.window_size_left = -1;
  params.window_size_right = -1;
  params.is_seqlens_k_cumulative = true;
  params.num_splits = 1;
}

}  // namespace

namespace apxinf::cuda::cutlass_ops {

template <typename Element>
int fa2(
    const void* q, const void* k, const void* v, void* output,
    void* softmax_lse, int batch, int query_tokens, int key_tokens,
    int query_heads, int kv_heads, int head_dim, float softmax_scale,
    cudaStream_t stream) {
  if (q == nullptr || k == nullptr || v == nullptr || output == nullptr ||
      softmax_lse == nullptr || batch <= 0 || query_tokens <= 0 ||
      key_tokens <= 0 || query_heads <= 0 || kv_heads <= 0 || head_dim <= 0 ||
      head_dim > 256 || query_heads % kv_heads != 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  FLASH_NAMESPACE::Flash_fwd_params params;
  fill_params(params, std::is_same<Element, cutlass::bfloat16_t>::value,
              q, k, v, output, softmax_lse, batch, query_tokens,
              key_tokens, query_heads, kv_heads, head_dim, softmax_scale);
  if (head_dim <= 96) {
    FLASH_NAMESPACE::run_mha_fwd_<Element, 96, false>(params, stream);
  } else {
    FLASH_NAMESPACE::run_mha_fwd_<Element, 256, false>(params, stream);
  }
  return static_cast<int>(cudaSuccess);
}

int fa2_bf16(
    const void* q, const void* k, const void* v, void* output,
    void* softmax_lse, int batch, int query_tokens, int key_tokens,
    int query_heads, int kv_heads, int head_dim, float softmax_scale,
    cudaStream_t stream) {
  return fa2<cutlass::bfloat16_t>(
      q, k, v, output, softmax_lse, batch, query_tokens, key_tokens,
      query_heads, kv_heads, head_dim, softmax_scale, stream);
}

int fa2_f16(
    const void* q, const void* k, const void* v, void* output,
    void* softmax_lse, int batch, int query_tokens, int key_tokens,
    int query_heads, int kv_heads, int head_dim, float softmax_scale,
    cudaStream_t stream) {
  return fa2<cutlass::half_t>(
      q, k, v, output, softmax_lse, batch, query_tokens, key_tokens,
      query_heads, kv_heads, head_dim, softmax_scale, stream);
}

}  // namespace apxinf::cuda::cutlass_ops
