// Copyright 2026 ApxInf contributors.
// Exact-shape PI0.5 MQA implementation with a fused static-scale E4M3 output.

#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>

#include <cmath>
#include <cstdint>

#include "flash_attn/flash.h"
#include "flash_attn/namespace_config.h"

namespace FLASH_NAMESPACE {
template <typename Element, int HeadDim, bool IsCausal>
void run_mha_fwd_(Flash_fwd_params& params, cudaStream_t stream);
}  // namespace FLASH_NAMESPACE

namespace {

constexpr float kLog2E = 1.4426950408889634074f;

void fill_params(FLASH_NAMESPACE::Flash_fwd_params& params, const void* q,
                 const void* k, const void* v, void* output,
                 void* softmax_lse, float softmax_scale,
                 float output_inverse_scale) {
  params = {};
  params.is_bf16 = false;
  params.q_ptr = const_cast<void*>(q);
  params.k_ptr = const_cast<void*>(k);
  params.v_ptr = const_cast<void*>(v);
  params.o_ptr = output;
  params.softmax_lse_ptr = softmax_lse;

  constexpr int64_t q_row_stride = 8 * 256;
  constexpr int64_t kv_row_stride = 256;
  params.q_batch_stride = 522 * q_row_stride;
  params.k_batch_stride = 522 * kv_row_stride;
  params.v_batch_stride = params.k_batch_stride;
  params.o_batch_stride = params.q_batch_stride;
  params.q_row_stride = q_row_stride;
  params.k_row_stride = kv_row_stride;
  params.v_row_stride = kv_row_stride;
  params.o_row_stride = q_row_stride;
  params.q_head_stride = 256;
  params.k_head_stride = 256;
  params.v_head_stride = 256;
  params.o_head_stride = 256;

  params.b = 1;
  params.h = 8;
  params.h_k = 1;
  params.h_h_k_ratio = 8;
  params.seqlen_q = 522;
  params.seqlen_k = 522;
  params.seqlen_q_rounded = 640;
  params.seqlen_k_rounded = 640;
  params.d = 256;
  params.d_rounded = 256;

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

  // Softcap is compile-time disabled for this translation unit. Reuse its
  // inert ABI slot to carry the static inverse output scale to the ApxInf
  // E4M3 epilogue without changing the vendored parameter structure.
  params.softcap = output_inverse_scale;
}

}  // namespace

namespace apxinf::cuda::cutlass_ops {

int fa2_f16_direct_e4m3_522(
    const void* q, const void* k, const void* v, void* output,
    void* softmax_lse, int batch, int query_tokens, int key_tokens,
    int query_heads, int kv_heads, int head_dim, float softmax_scale,
    float output_scale, cudaStream_t stream) {
  if (q == nullptr || k == nullptr || v == nullptr || output == nullptr ||
      softmax_lse == nullptr || batch != 1 || query_tokens != 522 ||
      key_tokens != 522 || query_heads != 8 || kv_heads != 1 ||
      head_dim != 256 || !std::isfinite(softmax_scale) ||
      softmax_scale <= 0.0f || !std::isfinite(output_scale) ||
      output_scale <= 0.0f) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  FLASH_NAMESPACE::Flash_fwd_params params;
  fill_params(params, q, k, v, output, softmax_lse, softmax_scale,
              1.0f / output_scale);
  FLASH_NAMESPACE::run_mha_fwd_<cutlass::half_t, 256, false>(params, stream);
  return static_cast<int>(cudaSuccess);
}

}  // namespace apxinf::cuda::cutlass_ops
