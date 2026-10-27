// Non-causal BF16 attention through Composable Kernel's flash-attention forward
// kernels (vendored subset, see ck/VENDOR.md).
//
// Same layout contract as apxinf_hip_native_attention: Q and O are
// [batch, q_tokens, q_heads, head_dim]; K and V hold, per batch, rows of
// [kv_heads, head_dim] at `kv_token_stride` elements apart, batches
// `kv_batch_stride` elements apart, of which the first `kv_tokens` rows are
// read. Query head h reads KV head h / (q_heads / kv_heads), which CK handles
// natively through nhead_q / nhead_k.
//
// `num_splits` <= 1 runs fmha_fwd; > 1 runs fmha_fwd_splitkv, which splits the
// keys across workgroups and needs F32 scratch from the caller:
//   lse_acc  batch * q_heads * num_splits * q_tokens           floats
//   o_acc    batch * q_heads * num_splits * q_tokens * head_dim floats
// Kernel choice within each API is CK's generated dispatcher's. Nothing here
// synchronizes or allocates; every launch goes on `stream`.

#include <hip/hip_runtime.h>

#include "common.h"
#include "fmha_fwd.hpp"

namespace {

using apxinf_hip::kInvalidArgument;
// CK's dispatchers return a negative time when no instance matches.
using apxinf_hip::kNoCkInstance;

template <typename Args>
void set_layout(Args& a, int batch, int q_tokens, int q_heads, int kv_heads, int head_dim,
                int kv_tokens, int64_t kv_batch_stride, int64_t kv_token_stride) {
  a.seqlen_q = q_tokens;
  a.seqlen_k = kv_tokens;
  a.batch = batch;
  a.max_seqlen_q = q_tokens;
  a.hdim_q = head_dim;
  a.hdim_v = head_dim;
  a.nhead_q = q_heads;
  a.nhead_k = kv_heads;
  // Strides are in elements: [batch, seq, head, dim] with dim innermost.
  a.stride_q = q_heads * head_dim;
  a.stride_k = static_cast<ck_tile::index_t>(kv_token_stride);
  a.stride_v = static_cast<ck_tile::index_t>(kv_token_stride);
  a.stride_o = q_heads * head_dim;
  a.nhead_stride_q = head_dim;
  a.nhead_stride_k = head_dim;
  a.nhead_stride_v = head_dim;
  a.nhead_stride_o = head_dim;
  a.batch_stride_q = q_tokens * q_heads * head_dim;
  a.batch_stride_k = static_cast<ck_tile::index_t>(kv_batch_stride);
  a.batch_stride_v = static_cast<ck_tile::index_t>(kv_batch_stride);
  a.batch_stride_o = q_tokens * q_heads * head_dim;
  a.window_size_left = -1;
  a.window_size_right = -1;
  a.mask_type = static_cast<ck_tile::index_t>(mask_enum::no_mask);
}

int run_fwd(const void* q, const void* k, const void* v, void* out, int batch, int q_tokens,
            int q_heads, int kv_heads, int head_dim, int kv_tokens, int64_t kv_batch_stride,
            int64_t kv_token_stride, float scale, hipStream_t stream) {
  fmha_fwd_traits traits{};
  traits.hdim_q = head_dim;
  traits.hdim_v = head_dim;
  traits.data_type = "bf16";
  traits.is_group_mode = false;
  traits.is_v_rowmajor = true;
  traits.has_logits_soft_cap = false;
  traits.mask_type = mask_enum::no_mask;
  traits.bias_type = bias_enum::no_bias;
  traits.has_lse = false;
  traits.has_dropout = false;
  traits.qscale_type = quant_scale_enum::no_scale;

  fmha_fwd_args a{};
  a.q_ptr = q;
  a.k_ptr = k;
  a.v_ptr = v;
  a.o_ptr = out;
  set_layout(a, batch, q_tokens, q_heads, kv_heads, head_dim, kv_tokens, kv_batch_stride,
             kv_token_stride);
  a.scale_s = scale;
  a.p_drop = 0.0f;
  a.s_randval = false;
  a.drop_seed_offset = std::make_pair(uint64_t{0}, uint64_t{0});

  const float time = fmha_fwd(traits, a, ck_tile::stream_config{stream, false, 0});
  if (time < 0) return kNoCkInstance;
  return static_cast<int>(hipGetLastError());
}

int run_splitkv(const void* q, const void* k, const void* v, void* out, int batch, int q_tokens,
                int q_heads, int kv_heads, int head_dim, int kv_tokens, int64_t kv_batch_stride,
                int64_t kv_token_stride, int num_splits, void* lse_acc, void* o_acc, float scale,
                hipStream_t stream) {
  fmha_fwd_splitkv_traits traits{};
  traits.hdim_q = head_dim;
  traits.hdim_v = head_dim;
  traits.data_type = "bf16";
  traits.is_group_mode = false;
  traits.is_v_rowmajor = true;
  traits.has_logits_soft_cap = false;
  traits.mask_type = mask_enum::no_mask;
  traits.bias_type = bias_enum::no_bias;
  traits.has_lse = false;

  fmha_fwd_splitkv_args a{};
  a.q_ptr = q;
  a.k_ptr = k;
  a.v_ptr = v;
  a.o_ptr = out;
  a.lse_acc_ptr = lse_acc;
  a.o_acc_ptr = o_acc;
  set_layout(a, batch, q_tokens, q_heads, kv_heads, head_dim, kv_tokens, kv_batch_stride,
             kv_token_stride);
  a.num_splits = num_splits;
  a.scale_s = scale;
  a.scale_p = 1.0f;
  a.scale_o = 1.0f;
  // Partial results: [batch, q_heads, num_splits, q_tokens(, head_dim)], F32.
  a.stride_o_acc = head_dim;
  a.split_stride_lse_acc = q_tokens;
  a.split_stride_o_acc = q_tokens * head_dim;
  a.nhead_stride_lse_acc = num_splits * q_tokens;
  a.nhead_stride_o_acc = num_splits * q_tokens * head_dim;
  a.batch_stride_lse_acc = q_heads * num_splits * q_tokens;
  a.batch_stride_o_acc = q_heads * num_splits * q_tokens * head_dim;

  const float time = fmha_fwd_splitkv(traits, a, ck_tile::stream_config{stream, false, 0});
  if (time < 0) return kNoCkInstance;
  return static_cast<int>(hipGetLastError());
}

}  // namespace

extern "C" int apxinf_hip_ck_attention(const void* q, const void* k, const void* v, void* out,
                                       int batch, int q_tokens, int q_heads, int kv_heads,
                                       int head_dim, int kv_tokens, int64_t kv_batch_stride,
                                       int64_t kv_token_stride, int num_splits, void* lse_acc,
                                       void* o_acc, float scale, void* stream) {
  if (batch <= 0 || q_tokens <= 0 || q_heads <= 0 || kv_heads <= 0 || q_heads % kv_heads != 0 ||
      head_dim <= 0 || kv_tokens <= 0 || kv_token_stride < static_cast<int64_t>(kv_heads) * head_dim ||
      kv_batch_stride > INT32_MAX || kv_token_stride > INT32_MAX) {
    return kInvalidArgument;
  }
  hipStream_t s = static_cast<hipStream_t>(stream);
  if (num_splits <= 1) {
    return run_fwd(q, k, v, out, batch, q_tokens, q_heads, kv_heads, head_dim, kv_tokens,
                   kv_batch_stride, kv_token_stride, scale, s);
  }
  if (lse_acc == nullptr || o_acc == nullptr) return kInvalidArgument;
  return run_splitkv(q, k, v, out, batch, q_tokens, q_heads, kv_heads, head_dim, kv_tokens,
                     kv_batch_stride, kv_token_stride, num_splits, lse_acc, o_acc, scale, s);
}
