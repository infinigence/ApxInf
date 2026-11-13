// apxinf's C ABI over Composable Kernel's FMHA forward.
//
// This file is apxinf's, not CK's: it marshals the extents
// `apxinf_hip::native::attention` already validated into CK's argument struct
// and reports, through its return value, whether CK can serve the call at all.
// Everything under `ck/` is upstream material; nothing there knows about
// apxinf. See `ck/VENDOR.md`.
//
// Why CK is here. Profiling one GR00T inference puts apxinf's own attention
// kernel at 37.7 ms of 115.9 ms -- 32% of GPU time -- and a roofline on the
// vision shape puts it at 1.7% of the measured 31.4 TFLOP/s ceiling and 1.5%
// of the 238 GB/s one. It is reaching neither, because its launch geometry is
// one block per query row and its Q.K is a scalar FMA loop: nothing touches
// the matrix cores, and every query block re-reads all of K and V. CK tiles
// the problem and issues WMMA. Measured side by side on GR00T's four
// attention shapes, CK runs 6.9x to 17.3x faster
// (`devlocal/gr00t-hip-port/perf/attention_probe.log`).
//
// What CK costs. Against an F64 host reference CK sits at 5.6e-3 relative
// where apxinf's kernel sits at 1.7e-3 -- about 1.4 BF16 ulp against 0.4. The
// cause was not established: it is uniform across head dims including the
// unpadded one (so not the head padding) and unchanged with
// CK_TILE_FMHA_FWD_FAST_EXP2 off (so not the fast softmax). What *was*
// measured is that it does not matter end to end: emulating that loss of
// precision in apxinf's kernel moves `final_action` from 0.00235 to 0.00243
// relative L2, against a 0.05 gate. The softmax normalizes by a denominator
// built from the same rounded weights, so most of the error cancels.
//
// What CK cannot serve, and why the caller must be able to continue without
// it. The vendored set covers no mask and causal masks, in both alignments,
// but apxinf's `causal_query_start` is more general than either: it allows any
// offset, where CK's two alignments are the offsets 0 and `kv - q`. An offset
// in between has no instance. `kUnsupported` says so without being an error,
// and the caller continues on apxinf's own kernel, which covers every
// geometry.

#include <hip/hip_runtime.h>

#include "ck/example/ck_tile/01_fmha/fmha_fwd.hpp"

#include "common.h"

namespace {

using namespace apxinf_hip;

// Returned when no vendored instance covers the call. The caller falls back to
// apxinf's own kernel, which is correct for every geometry; this is a routing
// outcome, not a failure.
constexpr int kUnsupported = kInvalidArgument;

// What the generated set can dispatch. The head-dim rule is CK's own, from
// `generated/fmha_fwd_api.cpp`: the 128 tile takes `hdim <= 128`, and within
// it `hdim % 128 == 0` selects the unpadded pipeline while `hdim % 8 == 0`
// selects the head-padded one. GR00T's 64 and 48 and pi05's 72 all land on
// the padded pipeline and pay for a 128-wide head tile.
// CK's mask, from apxinf's `causal_query_start`.
//
// apxinf states the mask as an offset: query row `i` of `q_tokens` sees keys
// `[0, start + i]`. CK states it as an alignment of the causal diagonal, and
// the vendored instances cover the two that matter:
//
//   start == 0            -> `mask_top_left`, row i sees [0, i]
//   start == kv - q       -> `mask_bottom_right`, the prefill-over-cache case
//
// Both set `window_size_left = -1` and `window_size_right = 0`; only the
// alignment differs (`mask.hpp`'s `decode`, the "t" and "b" branches). Any
// other offset is a mask CK has no instance for.
//
// Returns -1 when CK cannot express the mask.
int ck_mask_type(int query_start, int q_tokens, int kv_tokens) {
  if (query_start < 0) return 0;                        // no_mask
  if (query_start == 0) return 1;                       // mask_top_left
  if (query_start == kv_tokens - q_tokens) return 2;    // mask_bottom_right
  return -1;
}

bool supported(int dtype, int head_dim, int query_start, int q_tokens, int kv_tokens) {
  if (dtype != kDTypeBF16) return false;
  if (ck_mask_type(query_start, q_tokens, kv_tokens) < 0) return false;
  if (head_dim <= 0 || head_dim > 256 || head_dim % 8 != 0) return false;
  return true;
}

}  // namespace

extern "C" int apxinf_hip_ck_attention_supported(int dtype, int head_dim, int query_start,
                                                 int q_tokens, int kv_tokens) {
  return supported(dtype, head_dim, query_start, q_tokens, kv_tokens) ? 1 : 0;
}

// `out = softmax(q k^T * scale) v`, with the same layout contract as
// `apxinf_hip_native_attention`: q and out are
// `[batch, q_tokens, q_heads, head_dim]`, and k and v are
// `[batch, rows, kv_heads, head_dim]` where `rows >= kv_tokens` so a cache may
// hold more rows than the call reads. `kv_batch_stride` and `kv_token_stride`
// carry that, which is why they are passed rather than derived.
// `q_token_stride` is the address stride between q rows, in elements: at least
// `q_heads * head_dim` (contiguous), or wider when q is a column section of a
// packed `[rows, q_width + 2 * kv_width]` projection. `out` is always
// contiguous; q's batch stride is derived as `q_tokens * q_token_stride`.
extern "C" int apxinf_hip_ck_attention(int dtype, const void* q, const void* k,
                                       const void* v, void* out, int batch, int q_tokens,
                                       int q_heads, int kv_heads, int head_dim,
                                       int kv_tokens, long long kv_batch_stride,
                                       long long kv_token_stride, long long q_token_stride,
                                       float scale, int query_start, void* stream) {
  if (!supported(dtype, head_dim, query_start, q_tokens, kv_tokens)) return kUnsupported;
  if (batch <= 0 || q_tokens <= 0 || q_heads <= 0 || kv_heads <= 0 || kv_tokens <= 0) {
    return kInvalidArgument;
  }
  if (q_heads % kv_heads != 0) return kInvalidArgument;
  if (q_token_stride < static_cast<long long>(q_heads) * head_dim) return kInvalidArgument;

  fmha_fwd_traits traits{};
  traits.hdim_q = head_dim;
  traits.hdim_v = head_dim;
  traits.data_type = "bf16";
  traits.is_group_mode = false;
  traits.is_v_rowmajor = true;
  traits.has_logits_soft_cap = false;
  const int mask = ck_mask_type(query_start, q_tokens, kv_tokens);
  traits.mask_type = static_cast<mask_enum>(mask);
  traits.bias_type = bias_enum::no_bias;
  traits.has_lse = false;
  traits.has_dropout = false;
  traits.qscale_type = quant_scale_enum::no_scale;

  fmha_fwd_args args{};
  args.q_ptr = q;
  args.k_ptr = k;
  args.v_ptr = v;
  args.o_ptr = out;
  args.seqlen_q = q_tokens;
  args.seqlen_k = kv_tokens;
  args.batch = batch;
  args.max_seqlen_q = q_tokens;
  args.hdim_q = head_dim;
  args.hdim_v = head_dim;
  args.nhead_q = q_heads;
  args.nhead_k = kv_heads;
  args.scale_s = scale;
  args.logits_soft_cap = 0.0f;
  // Row (token) strides. q's comes from the caller: contiguous, or a packed
  // q/k/v section whose rows are `q_width + 2 * kv_width` elements apart.
  args.stride_q = static_cast<ck_tile::index_t>(q_token_stride);
  args.stride_o = q_heads * head_dim;
  args.stride_k = static_cast<ck_tile::index_t>(kv_token_stride);
  args.stride_v = static_cast<ck_tile::index_t>(kv_token_stride);
  // Head strides: heads are the innermost grouping before head_dim.
  args.nhead_stride_q = head_dim;
  args.nhead_stride_o = head_dim;
  args.nhead_stride_k = head_dim;
  args.nhead_stride_v = head_dim;
  args.batch_stride_q = static_cast<ck_tile::index_t>(q_tokens) *
                        static_cast<ck_tile::index_t>(q_token_stride);
  args.batch_stride_o = q_tokens * q_heads * head_dim;
  args.batch_stride_k = static_cast<ck_tile::index_t>(kv_batch_stride);
  args.batch_stride_v = static_cast<ck_tile::index_t>(kv_batch_stride);
  // A causal mask is a sliding window unbounded to the left and closed to the
  // right; the alignment is what `mask_type` carries.
  args.window_size_left = -1;
  args.window_size_right = mask == 0 ? -1 : 0;
  args.sink_size = 0;
  args.mask_type = mask;

  // `time_kernel = false`: no events, no synchronization. CK returns a
  // negative duration when it found no instance, which is the same routing
  // outcome as `supported()` returning false and must not become an error.
  const ck_tile::stream_config config{static_cast<hipStream_t>(stream), false};
  const float elapsed = fmha_fwd(traits, args, config);
  if (elapsed < 0.0f) return kUnsupported;
  return last_launch_status();
}
