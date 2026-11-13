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

#include <limits>

#include "ck/example/ck_tile/01_fmha/fmha_fwd.hpp"

#include "common.h"

namespace {

using namespace apxinf_hip;

// Returned when no vendored instance covers the call. The caller falls back to
// apxinf's own kernel, which is correct for every geometry; this is a routing
// outcome, not a failure. It is a distinct code from `kInvalidArgument` so the
// Rust side can tell "fall back" from "report an error".
constexpr int kUnsupported = kNoCkInstance;

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
  // CK's `index_t` is 32-bit; the strides arrive as 64-bit. A stride that does
  // not fit wraps to a negative offset and reads K/V from the wrong place, so
  // it is a routing miss rather than an error: apxinf's own kernel takes
  // 64-bit strides. `batch_stride_q` is checked at its computed value because
  // two factors that each fit can still overflow the product.
  const auto fits_int32 = [](long long v) {
    return v >= 0 && v <= static_cast<long long>(std::numeric_limits<int>::max());
  };
  if (!fits_int32(q_token_stride) || !fits_int32(kv_token_stride) ||
      !fits_int32(kv_batch_stride) ||
      !fits_int32(static_cast<long long>(q_tokens) * q_token_stride)) {
    return kUnsupported;
  }

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
  // CK reports a failed launch by throwing (`launch_and_check` ->
  // `HIP_CHECK_ERROR` -> `throw`), and an exception unwinding through
  // `extern "C"` into Rust is UB. Catch it here and treat it as a routing
  // miss, so the documented fallback to apxinf's own kernel actually happens.
  float elapsed;
  try {
    elapsed = fmha_fwd(traits, args, config);
  } catch (...) {
    // The error that made CK throw latches on the device; clear it so the
    // fallback kernel's `last_launch_status()` reports only its own launch.
    (void)hipGetLastError();
    return kUnsupported;
  }
  if (elapsed < 0.0f) {
    // No instance for this geometry. Whatever CK did while deciding may have
    // left a sticky error; clear it for the same reason as above.
    (void)hipGetLastError();
    return kUnsupported;
  }
  return last_launch_status();
}

// `out = softmax(q k^T * scale) v` through CK's split-KV forward: the keys are
// split across `num_splits` workgroups per query tile and combined, which pays
// off when few query rows face many keys — one 16-row query tile per head
// cannot fill the GPU. pi05's action expert (10 queries over 532 keys) is the
// caller this exists for; 4 splits measured fastest on gfx1151.
//
// The partial log-sum-exps and outputs land in caller-provided F32 scratch:
//   lse_acc  batch * q_heads * num_splits * q_tokens            floats
//   o_acc    batch * q_heads * num_splits * q_tokens * head_dim floats
// The Rust side sizes and aligns this; here it is two pointers.
//
// q and out are [batch, q_tokens, q_heads, head_dim] contiguous; k and v
// follow the same cache layout contract as `apxinf_hip_ck_attention`. The
// vendored split-KV instances are unmasked, which is why this entry has no
// mask parameter: a causal call belongs on the fused entry or apxinf's kernel.
// 128 is CK's own split ceiling (its `num_splits_heuristic`); empty trailing
// splits are handled by the combine kernel, so `kv_tokens` is not a bound.
extern "C" int apxinf_hip_ck_attention_splitkv(
    int dtype, const void* q, const void* k, const void* v, void* out, int batch, int q_tokens,
    int q_heads, int kv_heads, int head_dim, int kv_tokens, long long kv_batch_stride,
    long long kv_token_stride, int num_splits, void* lse_acc, void* o_acc, float scale,
    void* stream) {
  if (dtype != kDTypeBF16) return kInvalidArgument;
  if (batch <= 0 || q_tokens <= 0 || q_heads <= 0 || kv_heads <= 0 ||
      q_heads % kv_heads != 0 || head_dim <= 0 || kv_tokens <= 0 || num_splits < 1 ||
      num_splits > 128 || lse_acc == nullptr || o_acc == nullptr) {
    return kInvalidArgument;
  }
  if (kv_token_stride < static_cast<long long>(kv_heads) * head_dim) return kInvalidArgument;
  // CK's `index_t` is 32-bit; see the fused entry for why a stride that does
  // not fit is a routing miss rather than an error.
  const auto fits_int32 = [](long long v) {
    return v >= 0 && v <= static_cast<long long>(std::numeric_limits<int>::max());
  };
  if (!fits_int32(kv_batch_stride) || !fits_int32(kv_token_stride) ||
      !fits_int32(static_cast<long long>(q_tokens) * q_heads * head_dim)) {
    return kUnsupported;
  }

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
  a.seqlen_q = q_tokens;
  a.seqlen_k = kv_tokens;
  a.batch = batch;
  a.max_seqlen_q = q_tokens;
  a.hdim_q = head_dim;
  a.hdim_v = head_dim;
  a.nhead_q = q_heads;
  a.nhead_k = kv_heads;
  a.num_splits = num_splits;
  a.scale_s = scale;
  a.scale_p = 1.0f;
  a.scale_o = 1.0f;
  // Strides are in elements: [batch, seq, head, dim] with dim innermost. q
  // and out are contiguous here; k and v carry the caller's cache strides.
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
  // Partial results: [batch, q_heads, num_splits, q_tokens(, head_dim)], F32.
  a.stride_o_acc = head_dim;
  a.split_stride_lse_acc = q_tokens;
  a.split_stride_o_acc = q_tokens * head_dim;
  a.nhead_stride_lse_acc = num_splits * q_tokens;
  a.nhead_stride_o_acc = num_splits * q_tokens * head_dim;
  a.batch_stride_lse_acc = q_heads * num_splits * q_tokens;
  a.batch_stride_o_acc = q_heads * num_splits * q_tokens * head_dim;

  // `time_kernel = false`: no events, no synchronization. A negative duration
  // is a routing outcome, and an exception is caught for the same reason as
  // in the fused entry — it must not unwind through `extern "C"` into Rust.
  const ck_tile::stream_config config{static_cast<hipStream_t>(stream), false};
  float elapsed;
  try {
    elapsed = fmha_fwd_splitkv(traits, a, config);
  } catch (...) {
    (void)hipGetLastError();
    return kUnsupported;
  }
  if (elapsed < 0.0f) {
    (void)hipGetLastError();
    return kUnsupported;
  }
  return last_launch_status();
}
