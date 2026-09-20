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

#if defined(APXINF_FA2_SPLITKV)
template <typename Element, int HeadDim, bool IsCausal>
void run_mha_fwd_splitkv_dispatch(Flash_fwd_params& params,
                                  cudaStream_t stream);
#endif

}  // namespace FLASH_NAMESPACE

namespace apxinf::cuda::cutlass_ops {

void run_mha_fwd_hdim64_bf16_apx(
    FLASH_NAMESPACE::Flash_fwd_params& params, cudaStream_t stream);
bool use_mha_fwd_hdim64_bf16_apx(
    const FLASH_NAMESPACE::Flash_fwd_params& params);

}  // namespace apxinf::cuda::cutlass_ops

namespace {

constexpr float kLog2E = 1.4426950408889634074f;

// Minimal contiguous inference adapter for the public Flash_fwd_params
// contract. The field mapping follows official FlashAttention 2.7.4.post1's
// set_params_fprop (BSD-3-Clause); ApxInf supplies raw pointers and owns the
// host-side split policy.
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
  params.is_causal = false;
  if constexpr (std::is_same<Element, cutlass::bfloat16_t>::value) {
    if (head_dim == 64 &&
        apxinf::cuda::cutlass_ops::use_mha_fwd_hdim64_bf16_apx(params)) {
      apxinf::cuda::cutlass_ops::run_mha_fwd_hdim64_bf16_apx(params, stream);
      return static_cast<int>(cudaSuccess);
    }
  }
  if (head_dim <= 96) {
    FLASH_NAMESPACE::run_mha_fwd_<Element, 96, false>(params, stream);
  } else if (head_dim <= 128) {
    FLASH_NAMESPACE::run_mha_fwd_<Element, 128, false>(params, stream);
  } else {
    FLASH_NAMESPACE::run_mha_fwd_<Element, 256, false>(params, stream);
  }
  return static_cast<int>(cudaSuccess);
}

template <typename Element>
int fa2_causal(
    const void* q, const void* k, const void* v, void* output,
    void* softmax_lse, int batch, int query_tokens, int key_tokens,
    int query_heads, int kv_heads, int head_dim, float softmax_scale,
    cudaStream_t stream) {
  if (q == nullptr || k == nullptr || v == nullptr || output == nullptr ||
      softmax_lse == nullptr || batch <= 0 || query_tokens <= 0 ||
      key_tokens < query_tokens || query_heads <= 0 || kv_heads <= 0 ||
      head_dim <= 0 || head_dim > 256 || query_heads % kv_heads != 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  FLASH_NAMESPACE::Flash_fwd_params params;
  fill_params(params, std::is_same<Element, cutlass::bfloat16_t>::value,
              q, k, v, output, softmax_lse, batch, query_tokens,
              key_tokens, query_heads, kv_heads, head_dim, softmax_scale);
  params.is_causal = true;
  params.window_size_right = 0;
  if (head_dim <= 96) {
    FLASH_NAMESPACE::run_mha_fwd_<Element, 96, true>(params, stream);
  } else if (head_dim <= 128) {
    FLASH_NAMESPACE::run_mha_fwd_<Element, 128, true>(params, stream);
  } else {
    FLASH_NAMESPACE::run_mha_fwd_<Element, 256, true>(params, stream);
  }
  return static_cast<int>(cudaSuccess);
}

template <typename Element>
int fa2_strided_qkv(
    const void* qkv, void* output, void* softmax_lse, int batch,
    int tokens, int heads, int head_dim, float softmax_scale,
    cudaStream_t stream) {
  if (qkv == nullptr || output == nullptr || softmax_lse == nullptr ||
      batch <= 0 || tokens <= 0 || heads <= 0 || head_dim <= 0 ||
      head_dim > 256) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const int hidden = heads * head_dim;
  const auto* base = static_cast<const Element*>(qkv);
  FLASH_NAMESPACE::Flash_fwd_params params;
  fill_params(params, std::is_same<Element, cutlass::bfloat16_t>::value,
              base, base + hidden, base + 2 * hidden, output, softmax_lse,
              batch, tokens, tokens, heads, heads, head_dim, softmax_scale);
  const int64_t row_stride = static_cast<int64_t>(3) * hidden;
  params.q_batch_stride = static_cast<int64_t>(tokens) * row_stride;
  params.k_batch_stride = params.q_batch_stride;
  params.v_batch_stride = params.q_batch_stride;
  params.q_row_stride = row_stride;
  params.k_row_stride = row_stride;
  params.v_row_stride = row_stride;
  if constexpr (std::is_same<Element, cutlass::bfloat16_t>::value) {
    if (head_dim == 64 &&
        apxinf::cuda::cutlass_ops::use_mha_fwd_hdim64_bf16_apx(params)) {
      apxinf::cuda::cutlass_ops::run_mha_fwd_hdim64_bf16_apx(params, stream);
      return static_cast<int>(cudaSuccess);
    }
  }
  if (head_dim <= 96) {
    FLASH_NAMESPACE::run_mha_fwd_<Element, 96, false>(params, stream);
  } else {
    FLASH_NAMESPACE::run_mha_fwd_<Element, 256, false>(params, stream);
  }
  return static_cast<int>(cudaSuccess);
}

#if defined(APXINF_FA2_SPLITKV)
template <typename Element, bool IsCausal>
int fa2_splitkv(
    const void* q, const void* k, const void* v, void* output,
    void* softmax_lse, void* softmax_lse_accum, void* o_accum, int batch,
    int query_tokens, int key_tokens, int query_heads, int kv_heads,
    int head_dim, float softmax_scale, int num_splits, cudaStream_t stream) {
  if (q == nullptr || k == nullptr || v == nullptr || output == nullptr ||
      softmax_lse == nullptr || softmax_lse_accum == nullptr ||
      o_accum == nullptr || batch <= 0 || query_tokens <= 0 ||
      key_tokens <= 0 || query_heads <= 0 || kv_heads <= 0 || head_dim <= 0 ||
      head_dim > 256 || query_heads % kv_heads != 0 || num_splits <= 0 ||
      num_splits > 128) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  FLASH_NAMESPACE::Flash_fwd_params params;
  fill_params(params, std::is_same<Element, cutlass::bfloat16_t>::value,
              q, k, v, output, softmax_lse, batch, query_tokens,
              key_tokens, query_heads, kv_heads, head_dim, softmax_scale);
  params.is_causal = IsCausal;
  if constexpr (IsCausal) {
    params.window_size_right = 0;
  }
  params.num_splits = num_splits;
  params.softmax_lseaccum_ptr = num_splits > 1 ? softmax_lse_accum : nullptr;
  params.oaccum_ptr = num_splits > 1 ? o_accum : nullptr;
  if (num_splits <= 1) {
    if (head_dim <= 96) {
      FLASH_NAMESPACE::run_mha_fwd_<Element, 96, IsCausal>(params, stream);
    } else {
      FLASH_NAMESPACE::run_mha_fwd_<Element, 256, IsCausal>(params, stream);
    }
    return static_cast<int>(cudaSuccess);
  }
  if (head_dim <= 128) {
    FLASH_NAMESPACE::run_mha_fwd_splitkv_dispatch<Element, 128, IsCausal>(
        params, stream);
  } else if (head_dim <= 256) {
    FLASH_NAMESPACE::run_mha_fwd_splitkv_dispatch<Element, 256, IsCausal>(
        params, stream);
  } else {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  return static_cast<int>(cudaSuccess);
}
#endif

int fa2_bf16(
    const void* q, const void* k, const void* v, void* output,
    void* softmax_lse, int batch, int query_tokens, int key_tokens,
    int query_heads, int kv_heads, int head_dim, float softmax_scale,
    cudaStream_t stream) {
  return fa2<cutlass::bfloat16_t>(
      q, k, v, output, softmax_lse, batch, query_tokens, key_tokens,
      query_heads, kv_heads, head_dim, softmax_scale, stream);
}

int fa2_bf16_causal(
    const void* q, const void* k, const void* v, void* output,
    void* softmax_lse, int batch, int query_tokens, int key_tokens,
    int query_heads, int kv_heads, int head_dim, float softmax_scale,
    cudaStream_t stream) {
  return fa2_causal<cutlass::bfloat16_t>(
      q, k, v, output, softmax_lse, batch, query_tokens, key_tokens,
      query_heads, kv_heads, head_dim, softmax_scale, stream);
}

int fa2_bf16_strided_qkv(
    const void* qkv, void* output, void* softmax_lse, int batch,
    int tokens, int heads, int head_dim, float softmax_scale,
    cudaStream_t stream) {
  return fa2_strided_qkv<cutlass::bfloat16_t>(
      qkv, output, softmax_lse, batch, tokens, heads, head_dim,
      softmax_scale, stream);
}

#if defined(APXINF_FA2_SPLITKV)
int fa2_bf16_splitkv(
    const void* q, const void* k, const void* v, void* output,
    void* softmax_lse, void* softmax_lse_accum, void* o_accum, int batch,
    int query_tokens, int key_tokens, int query_heads, int kv_heads,
    int head_dim, float softmax_scale, int num_splits, cudaStream_t stream) {
  return fa2_splitkv<cutlass::bfloat16_t, false>(
      q, k, v, output, softmax_lse, softmax_lse_accum, o_accum, batch,
      query_tokens, key_tokens, query_heads, kv_heads, head_dim, softmax_scale,
      num_splits, stream);
}

int fa2_bf16_causal_splitkv(
    const void* q, const void* k, const void* v, void* output,
    void* softmax_lse, void* softmax_lse_accum, void* o_accum, int batch,
    int query_tokens, int key_tokens, int query_heads, int kv_heads,
    int head_dim, float softmax_scale, int num_splits, cudaStream_t stream) {
  return fa2_splitkv<cutlass::bfloat16_t, true>(
      q, k, v, output, softmax_lse, softmax_lse_accum, o_accum, batch,
      query_tokens, key_tokens, query_heads, kv_heads, head_dim, softmax_scale,
      num_splits, stream);
}
#endif

int fa2_f16(
    const void* q, const void* k, const void* v, void* output,
    void* softmax_lse, int batch, int query_tokens, int key_tokens,
    int query_heads, int kv_heads, int head_dim, float softmax_scale,
    cudaStream_t stream) {
  return fa2<cutlass::half_t>(
      q, k, v, output, softmax_lse, batch, query_tokens, key_tokens,
      query_heads, kv_heads, head_dim, softmax_scale, stream);
}

int fa2_f16_strided_qkv(
    const void* qkv, void* output, void* softmax_lse, int batch,
    int tokens, int heads, int head_dim, float softmax_scale,
    cudaStream_t stream) {
  return fa2_strided_qkv<cutlass::half_t>(
      qkv, output, softmax_lse, batch, tokens, heads, head_dim,
      softmax_scale, stream);
}

}  // namespace apxinf::cuda::cutlass_ops
