// Host side of the vendored mp_31 D256 FMHA kernels.
//
// The kernels themselves are `kernels/mutlass_fmha/fmha_fwd/*.mu`, vendored
// from MATE and compiled by `build.rs` with MATE's own device flags. This file
// is what supplies them with what the render expected its caller to supply: the
// `cu_seqlens` pair the ragged kernel reads, the LSE buffer the epilogue writes
// even when the attention is not split, and the input validation.
//
// Why this exists at all
// ----------------------
// On mp_31 the operator used to run through the torch eager bridge, which
// answered `attention.mqa_bf16` with `mate.flash_attn_varlen_func`. That is a
// correctness scaffold, and it is also what pinned the whole PI0.5 path to
// `--execution eager`: `eager_bridge::dispatch` refuses during capture, because
// a graph would replay without the bridged work. With the kernel here, nothing
// on the BF16 path reaches the bridge and `mate` is not imported at all.
//
// Contract
// --------
// BF16, inference only, non-causal, no mask. Q and out are packed
// `[tokens, 8, 256]` contiguous; K and V are `[key_tokens, 256]` contiguous,
// which is how the model holds them and which gives the same row and head
// stride as the `[key_tokens, 1, 256]` view the bridge built. `key_tokens` is
// how many rows take part and the kernel reads no row past it, so a reserved
// prefix -- NaNs included -- never reaches the softmax.
//
// The scale is `1/sqrt(256)`. It is not a parameter: the vendored
// instantiation's tile shape and head dimension are compile-time constants, and
// 256 is the only head dimension these two kernels have.
//
// Threading and lifetime
// ----------------------
// The metadata and LSE buffers are `thread_local`, as in `mubin_fa_adapter.mu`
// and for the same reasons: one host thread drives one context **at a time**,
// and nothing here is safe to share. Nothing is ever freed, also as there --
// teardown would run after the driver may already be gone.
//
// One context at a time, not one for the thread's life: a thread that has driven
// a context on another card keeps that card's entries, so the table below
// carries the device too (`adapter_common.h` says where it comes from).
//
// The metadata is cached per `(device, tokens, key_tokens)` rather than uploaded
// per call: an inference reaches this kernel a few hundred times with a handful
// of distinct argument pairs, so uploading every time would be hundreds of small
// copies to say the same thing a few times. The table grows and is never
// evicted; each entry is 16 device bytes plus the LSE buffer -- which is
// `kHeads * tokens * 4`, 32 bytes per token, so ~16 KiB at PI0.5's 522-token
// prefix rather than the 16 bytes an earlier version of this line implied.
//
// `mubin_fa_adapter.mu` has its own table that is deliberately not shared with
// this one. The two kernels take different metadata -- that one needs
// `{cu_seqlens_q, seqused_k}` and this one `{cu_seqlens_q, cu_seqlens_k}` --
// they serve mutually exclusive targets, and that table is in another
// translation unit's anonymous namespace. The duplication is the price of the
// two kernels having different ABIs, not an oversight.

#include <musa.h>
#include <musa_bf16.h>
#include <musa_runtime.h>

#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "adapter_common.h"
#include "fmha_fwd_args.hpp"

namespace {

constexpr int kHeads = 8;
constexpr int kHeadDim = 256;

using Launcher = const char* (*)(const void* q, const void* k, const void* v, void* out,
                                 float* lse, const uint32_t* cu_seqlens_q,
                                 const uint32_t* cu_seqlens_k, int32_t tokens,
                                 int32_t key_tokens, void* stream);

// ── Variants ────────────────────────────────────────────────────────────────
//
// The two vendored instantiations and their names in one table, indexed by the
// caller's `variant` argument. Which kernel to run and what to call it in an
// error are the same fact, and a third variant that had to update two switches
// would be one edit away from being wrong in a way nothing would catch.
//
// Indexing is unchecked: `apxinf_musa_mutlass_fa_d256` rejects an out-of-range
// variant before anything here sees it, and nothing else calls these.

enum Variant {
  kPackedDecodeTile128 = 0,
  kGeneralTile192 = 1,
  kVariantCount = 2,
};

struct VariantInfo {
  const char* name;
  Launcher launch;
};

const VariantInfo kVariants[kVariantCount] = {
    {"packed-decode-tile128", apxinf_musa_mp31_fmha_fwd_tile128_packed},
    {"general-tile192", apxinf_musa_mp31_fmha_fwd_tile192_general},
};

using apxinf_adapters::capturing;
using apxinf_adapters::error_buffer;
using apxinf_adapters::set_error;

// ── Metadata and LSE, per (device, tokens, key_tokens) ──────────────────────
//
// `cu_seqlens_q` and `cu_seqlens_k` are separate device int32 arrays here,
// unlike `mubin_fa_adapter.mu`'s single `{cu_q, cu_q, seqused_k}` triple: the
// rendered host entry passed `cu_seqlens_k.value().data_ptr()` to a parameter
// of its own. Both are functions of `(tokens, key_tokens)` alone, and one
// sequence per call starts at zero.
//
// The LSE buffer is `[kHeads, tokens]` f32. It is not optional: the epilogue
// writes it from registers whenever the attention is not split, and this
// instantiation is never split. The bridge discarded MATE's LSE and copied only
// the output, so nothing reads this buffer -- it exists because the kernel
// stores to it either way, and a null there is an out-of-bounds write.
//
// The device is in the key because both are *written* by the kernel as well as
// read by it, and because a pointer only names memory on the card that
// allocated it: measured on two S5000s, a same-thread second card either
// returned a mix of zeros and the first card's values, or faulted with
// `MUSA error 700`, depending on which card came second.

struct Metadata {
  /// Which card the two pointers below were allocated on.
  int device_id = -1;
  int32_t tokens = 0;
  int32_t key_tokens = 0;
  uint32_t* cu_seqlens = nullptr;  // [4] = {0, tokens, 0, key_tokens}
  float* lse = nullptr;            // [kHeads * tokens] f32: 32 * tokens bytes, not 4
};

thread_local std::vector<Metadata> g_metadata;

bool ensure_metadata(int32_t tokens, int32_t key_tokens, int device_id, void* stream,
                     const uint32_t** out_cu_seqlens, float** out_lse) {
  for (const Metadata& entry : g_metadata) {
    if (entry.device_id == device_id && entry.tokens == tokens &&
        entry.key_tokens == key_tokens) {
      *out_cu_seqlens = entry.cu_seqlens;
      *out_lse = entry.lse;
      return true;
    }
  }

  // The one allocation on this path, and the reason the warmup traversal has to
  // see every shape before the capture begins.
  if (capturing(stream)) {
    set_error(
        "the D256 attention metadata for %d x %d was first needed during graph "
        "capture; a captured graph bakes the pointers it recorded, so every shape "
        "must be seen once on the warmup traversal before musaStreamBeginCapture",
        tokens, key_tokens);
    return false;
  }

  // Both pointers below are device memory, and `musaMalloc`/`musaMemcpy` resolve
  // against the current device rather than against the stream that will use them.
  if (!apxinf_adapters::select_device(device_id)) {
    set_error("the D256 attention metadata for %d x %d cannot be allocated: MUSA %d could not "
              "be selected",
              tokens, key_tokens, device_id);
    return false;
  }

  const size_t lse_count = size_t(kHeads) * size_t(tokens);
  Metadata entry;
  entry.device_id = device_id;
  entry.tokens = tokens;
  entry.key_tokens = key_tokens;
  if (musaMalloc(reinterpret_cast<void**>(&entry.cu_seqlens), 4 * sizeof(uint32_t)) !=
      musaSuccess) {
    set_error("musaMalloc failed for the D256 attention metadata");
    return false;
  }
  if (musaMalloc(reinterpret_cast<void**>(&entry.lse), lse_count * sizeof(float)) !=
      musaSuccess) {
    musaFree(entry.cu_seqlens);
    set_error("musaMalloc failed for the D256 attention's %zu element LSE buffer", lse_count);
    return false;
  }
  // One sequence per call: each `cu_seqlens` is a single segment starting at
  // zero, so the ragged kernel sees a batch of one of the given length.
  const uint32_t host[4] = {0, static_cast<uint32_t>(tokens), 0,
                            static_cast<uint32_t>(key_tokens)};
  if (musaMemcpy(entry.cu_seqlens, host, sizeof(host), musaMemcpyHostToDevice) != musaSuccess) {
    musaFree(entry.cu_seqlens);
    musaFree(entry.lse);
    set_error("musaMemcpy failed for the D256 attention metadata");
    return false;
  }
  g_metadata.push_back(entry);
  *out_cu_seqlens = entry.cu_seqlens;
  *out_lse = entry.lse;
  return true;
}

}  // namespace

extern "C" const char* apxinf_musa_mutlass_fa_d256_last_error(void) { return error_buffer(); }

// Packed BF16 attention over a dense BF16 KV, on the vendored mp_31 kernels.
//
// `variant` is one of the `Variant` values; `tokens` is the query length and the
// query and output tensors' first dimension; `key_tokens` is how many KV rows
// take part.
//
// Returns 0 on success. On failure it returns non-zero and
// `apxinf_musa_mutlass_fa_d256_last_error()` names the reason. Nothing here
// falls back to another variant: the two instantiations tile the same
// arithmetic differently, and a caller that asked for one and silently got the
// other would be measuring a kernel it did not choose.
extern "C" int apxinf_musa_mutlass_fa_d256(const void* q, const void* k, const void* v, void* out,
                                           int32_t tokens, int32_t key_tokens, int32_t variant,
                                           void* stream) {
  if (q == nullptr || k == nullptr || v == nullptr || out == nullptr) {
    set_error("the attention operands must not be null");
    return -1;
  }
  if (tokens <= 0 || key_tokens <= 0) {
    set_error("the attention shapes must be positive, got tokens %d key_tokens %d", tokens,
              key_tokens);
    return -1;
  }
  if (variant < 0 || variant >= kVariantCount) {
    set_error("unknown D256 attention variant %d", variant);
    return -1;
  }

  // Which card this call runs on, asked of the stream rather than of the thread:
  // the table below outlives any one context, and a thread that has served
  // another card still holds that card's entries -- see `adapter_common.h`.
  const int device_id = apxinf_adapters::stream_device(stream);

  const uint32_t* cu_seqlens = nullptr;
  float* lse = nullptr;
  if (!ensure_metadata(tokens, key_tokens, device_id, stream, &cu_seqlens, &lse)) {
    return -1;
  }

  const char* failure =
      kVariants[variant].launch(q, k, v, out, lse, cu_seqlens, cu_seqlens + 2, tokens,
                                key_tokens, stream);
  if (failure != nullptr) {
    set_error("the %s D256 attention failed: %s", kVariants[variant].name, failure);
    return -1;
  }
  return 0;
}
