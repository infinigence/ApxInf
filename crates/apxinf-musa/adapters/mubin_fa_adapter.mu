// Host side of the MP22 D256 attention assembly.
//
// The kernels themselves are `kernels/asm/*.asm`, assembled by `build.rs` into
// three `muModuleLoadData` images. This file is what loads them and launches
// one; it owns no arithmetic that the assembly does not.
//
// Why this is not a `.cuh` like the other operators: the assembly is a
// loadable module, not a translation unit to be compiled. `mcc` has no way to
// consume `.asm`, and the driver wants a module rather than a linked symbol, so
// the two meet here through the driver API instead of the runtime one. That is
// also why this file is an adapter and not a kernel -- see
// `crates/apxinf-cuda/adapters/README.md` for the same split on the CUDA side.
//
// Contract
// --------
// BF16, inference only, non-causal, no masks, no LSE. Q and out are packed
// `[tokens, 8, 256]`; K and V are `[capacity, 1, 256]` contiguous, which is how
// a `kv_heads == 1` cache is laid out whether the caller holds it as 2-D
// `[rows, head_dim]` or 3-D. `key_tokens` bounds how many leading rows take
// part, so a reserved prefix can be passed whole -- rows past it are excluded
// from both the softmax and PV, NaNs included.
//
// The scale is `1/sqrt(256)`, folded into log2 because the assembly
// exponentiates with `exp2`. It is not a parameter: the assembly hard-codes the
// head dimension in its fragment layouts, and 256 is the only one it has.
//
// Variants
// --------
// The three images export the same kernel symbol, so the variant is chosen by
// which image is loaded, not by what is looked up. `apxinf_musa_fa_d256` takes
// the choice from the caller (see `kernels/attention.rs`, which also logs the
// fallback) because it depends on the model's action horizon, not on anything
// this file can see.
//
// The split-KV variant is the only one that is not self-contained: it writes
// unnormalized partials and per-split statistics into a workspace and a second
// kernel merges them. That merge kernel is below, and its constants are tied to
// the assembly's packed row count -- 10 tokens x 8 heads = 80. Changing one
// without the other is silently wrong, which is why they sit in one file.
//
// Threading and lifetime
// ----------------------
// The loaded modules, the metadata buffers and the split-KV workspace are
// `thread_local`, as in `mublas_adapter.mu` and for the same reasons: one
// host thread drives one context **at a time**, and nothing here is safe to
// share. Nothing is ever freed, also as there -- teardown would run after the
// driver may already be gone.
//
// One context at a time, not one for the thread's life: a thread that has
// driven a context on another card keeps that card's entries, so every cache
// below carries the device too (`adapter_common.h` says where it comes from).
//
// The metadata is cached per `(device, tokens, key_tokens)` rather than
// uploaded per call: an inference reaches these kernels ~200 times with a
// handful of distinct argument pairs -- one per prefix length it runs -- so
// uploading every time would be ~200 small copies to say the same thing a few
// times.

#include <musa.h>
#include <musa_bf16.h>
#include <musa_runtime.h>

#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
#include <utility>
#include <vector>

#include "adapter_common.h"
#include "mubin_images.h"

namespace {

constexpr int kHeads = 8;
constexpr int kHeadDim = 256;
constexpr int kTokensPerRowGroup = 10;  // the packed decode layouts' fixed query length
constexpr int kSplits = 4;
constexpr int kPackedRows = kTokensPerRowGroup * kHeads;

// The assembly's parameter block: every tensor is an (address, byte length)
// pair rather than a bare pointer.
struct RobustPointer {
  uint64_t address;
  uint64_t bytes;
};

RobustPointer robust(const void* pointer, uint64_t bytes) {
  return {reinterpret_cast<uint64_t>(pointer), bytes};
}

// ── Variants ────────────────────────────────────────────────────────────────
//
// The three images, their names, the symbol each exports and the launch shape
// each wants, in one table indexed by the caller's `variant` argument. Which
// image to load, what to call it in an error and how many blocks to launch are
// properties of the same kernel, and a fourth variant that had to update two
// switches would be one edit away from being wrong in a way nothing would
// catch.
//
// The launch shape is data rather than a formula at the launch site for the
// same reason, and it is the one that would fail *silently*: a variant whose
// kernel tiles differently, launched with the previous variant's grid, loads
// fine, resolves its symbol, returns `MUSA_SUCCESS`, and computes the wrong
// thing. Nothing downstream can tell that from a correct launch.
//
// Indexing is unchecked: `apxinf_musa_fa_d256` rejects an out-of-range variant
// before anything here sees it, and nothing else calls these.

enum Variant {
  kGeneral = 0,
  kPackedDecode = 1,
  kSplitKv = 2,
  kVariantCount = 3,
};

struct VariantInfo {
  const char* name;
  const unsigned char* image;
  unsigned long size;
  /// The symbol this image exports. The three MP22 images export the same one
  /// -- the variant is chosen by which image is loaded, not by what is looked
  /// up -- but it is a field because a next generation's kernel will not be
  /// called `mate_fa_d256_mp22`, and this is where that edit belongs.
  const char* kernel_symbol;
  /// Rows of the variant's row space that one block covers along x.
  unsigned rows_per_block;
  /// Whether x counts `tokens` (false) or the packed `tokens * kHeads` rows the
  /// decode layouts lay out (true).
  bool x_is_packed_rows;
  /// Blocks along y.
  unsigned grid_y;
  /// Threads per block.
  unsigned block;
};

const VariantInfo kVariants[kVariantCount] = {
    {"general", kApxinfMusaFaD256Mp22Image, kApxinfMusaFaD256Mp22ImageSize,
     "mate_fa_d256_mp22", 32, false, kHeads, 128},
    {"packed-decode", kApxinfMusaFaD256Mp22DecodeImage,
     kApxinfMusaFaD256Mp22DecodeImageSize,
     "mate_fa_d256_mp22", 32, true, kSplits, 128},
    {"split-kv", kApxinfMusaFaD256Mp22SplitkvImage,
     kApxinfMusaFaD256Mp22SplitkvImageSize,
     "mate_fa_d256_mp22", 32, true, kSplits, 128},
};

/// Whether a variant's image has been brought up **on one device**, and how it
/// went.
///
/// Three states rather than the one `attempted` flag this replaces, because the
/// failure modes are not alike. An image that is empty or a symbol the image
/// does not export is a build-time fact: retrying would produce the same answer
/// forever, so it is latched and only re-reported. `muModuleLoadData` can fail
/// for a reason that is *not* the image — an out-of-memory on a busy device is
/// the plausible one — and a thread that latched that would never retry for the
/// life of the process. So it is not latched, and a later call tries again.
///
/// The other two adapters have the same choice to make; `mudnn_adapter.mu`
/// rebuilds its per-shape state from scratch on every call, and
/// `mutlass_fa_adapter.mu` has no such cache at all. This one is the only place
/// that had a negative cache, and it was the wrong shape.
enum class LoadState {
  kUntried,
  kReady,
  /// Reported once, reported again on each later call, never retried.
  kPermanentFailure,
};

struct Loaded {
  LoadState state = LoadState::kUntried;
  MUfunction function = nullptr;
};

/// Keyed by `(device, variant)`, not by variant alone: a module is loaded into
/// the context of whichever device was current at `muModuleLoadData`, and the
/// function handle it yields launches on that device. A handle is therefore
/// **device** state -- unlike the muBLAS handle next door, which is stream
/// state and keyed `(device, stream)` -- and a thread that has served two cards
/// keeps one entry per card.
///
/// `kPermanentFailure` follows the same key, so the per-(thread, device,
/// variant) answer is given once per device. Two of its three causes (an empty
/// image, a symbol the image does not export) are properties of the image
/// rather than of the device, so the second card re-asks and re-reports the
/// same sentence. Telling those apart would mean a second container to say what
/// this one already knows; this file's point is the least host code that is
/// still honest, and a repeated sentence is the cheaper half of that trade.
thread_local std::map<std::pair<int, int>, Loaded> g_loaded;

using apxinf_adapters::capturing;
using apxinf_adapters::error_buffer;
using apxinf_adapters::select_device;
using apxinf_adapters::set_error;
using apxinf_adapters::stream_device;

/// Load one image, once per (thread, device). Null with the error set otherwise.
MUfunction load_variant(int variant, int device) {
  Loaded& loaded = g_loaded[std::make_pair(device, variant)];
  const VariantInfo& info = kVariants[variant];
  if (loaded.state == LoadState::kReady) {
    return loaded.function;
  }
  if (loaded.state == LoadState::kPermanentFailure) {
    set_error(
        "the %s attention image is unusable and this thread already reported why; the "
        "message from the first attempt has been overwritten by later calls",
        info.name);
    return nullptr;
  }

  if (info.image == nullptr || info.size == 0) {
    // `build.rs` emits an empty array when `llvm-mc` or `lld` was missing, or
    // when assembling failed. Reported by name rather than as a null deref.
    loaded.state = LoadState::kPermanentFailure;
    set_error(
        "the %s attention image is empty: the MUSA toolkit's llvm-mc/lld were "
        "unavailable or the assembly failed to build (see the build warnings)",
        info.name);
    return nullptr;
  }

  // The key is `(device, variant)`; load the module into that device's
  // context, not into whichever one the thread last selected.
  if (!select_device(device)) {
    set_error("the %s attention image cannot select device %d", info.name, device);
    return nullptr;
  }

  MUmodule module = nullptr;
  MUresult status = muModuleLoadData(&module, info.image);
  if (status != MUSA_SUCCESS) {
    // Deliberately not latched: see `LoadState`.
    set_error("muModuleLoadData failed for the %s attention image (MUresult %d)",
              info.name, static_cast<int>(status));
    return nullptr;
  }
  MUfunction function = nullptr;
  status = muModuleGetFunction(&function, module, info.kernel_symbol);
  if (status != MUSA_SUCCESS) {
    // Permanent -- the symbol is either in the image or it is not -- so the
    // module is unloaded rather than kept for a retry that will not come. The
    // success path keeps its module loaded for the thread's lifetime on purpose;
    // this path used to keep it loaded for nothing.
    set_error("muModuleGetFunction failed for the %s attention image (MUresult %d)",
              info.name, static_cast<int>(status));
    muModuleUnload(module);
    loaded.state = LoadState::kPermanentFailure;
    return nullptr;
  }
  loaded.function = function;
  loaded.state = LoadState::kReady;
  return function;
}

// ── Metadata: cu_seqlens_q and seqused_k ────────────────────────────────────
//
// The assembly takes both as device int32 tensors. They are a function of
// `(device, tokens, key_tokens)`, and an inference uses exactly two pairs on
// one card, so a tiny per-thread table covers it and each pair is uploaded
// once.

// Grows, and only grows. `key_tokens` is the prompt prefix length, so a
// long-lived process accumulates one entry per distinct task string it is
// asked about -- a fixed table would turn the ninth into a permanent failure
// for a model whose prompts vary. There is nothing to evict: each entry is 12
// device bytes and the device pointers are never freed, as the lifetime note
// at the top of this file says.

struct Metadata {
  /// Which card `device` was allocated on. Without it, a thread that has served
  /// one card hands the other card's launch a pointer into memory it cannot
  /// read: measured on two S5000s, the second card came back with a mix of
  /// zeros and the first card's values (no error), and the reverse order
  /// faulted with `MUSA error 700`.
  int device_id = -1;
  int32_t tokens = 0;
  int32_t key_tokens = 0;
  int32_t* device = nullptr;  // [cu_seqlens_q(2), seqused_k(1)]
};

thread_local std::vector<Metadata> g_metadata;

bool ensure_metadata(int32_t tokens, int32_t key_tokens, int device_id, void* stream,
                     int32_t** out_device) {
  for (const Metadata& entry : g_metadata) {
    if (entry.device_id == device_id && entry.tokens == tokens &&
        entry.key_tokens == key_tokens) {
      *out_device = entry.device;
      return true;
    }
  }

  // A hit is free; a miss allocates, and that is what cannot happen inside a
  // capture. The warmup traversal is what puts every shape this model uses into
  // the table before `musaStreamBeginCapture`, so a miss here means a shape was
  // first needed while recording.
  if (capturing(stream)) {
    set_error(
        "the MP22 D256 attention metadata for %d x %d was first needed during graph capture; "
        "a captured graph bakes the pointers it recorded, so every shape must be seen once on "
        "the warmup traversal before musaStreamBeginCapture",
        static_cast<int>(tokens), static_cast<int>(key_tokens));
    return false;
  }

  // The metadata is device memory, and `musaMalloc`/`musaMemcpy` resolve against
  // the current device rather than against the stream that will read it.
  if (!select_device(device_id)) {
    set_error("the attention metadata for %d x %d cannot be allocated: MUSA %d could not be "
              "selected",
              static_cast<int>(tokens), static_cast<int>(key_tokens), device_id);
    return false;
  }
  int32_t* device = nullptr;
  if (musaMalloc(reinterpret_cast<void**>(&device), 3 * sizeof(int32_t)) != musaSuccess) {
    set_error("musaMalloc failed for the attention metadata");
    return false;
  }
  // One sequence per call: `cu_seqlens_q` is a single segment starting at zero.
  const int32_t host[3] = {0, tokens, key_tokens};
  if (musaMemcpy(device, host, sizeof(host), musaMemcpyHostToDevice) != musaSuccess) {
    musaFree(device);
    set_error("musaMemcpy failed for the attention metadata");
    return false;
  }
  g_metadata.push_back(Metadata{device_id, tokens, key_tokens, device});
  *out_device = device;
  return true;
}

// ── Split-KV merge ──────────────────────────────────────────────────────────
//
// Written to match the assembly's partial layout: `kSplits` blocks of
// `[kPackedRows, kHeadDim]` f32 partials, then `kSplits` blocks of
// `[kPackedRows, 2]` (max, denominator). An empty partition reports a zero
// accumulator and denominator with a finite -FLT_MAX, so it contributes
// nothing rather than a NaN.
__global__ void fa_d256_merge_kernel(const float* part, const float* stats,
                                     __mt_bfloat16* out) {
  int row = blockIdx.x, col = threadIdx.x;
  float maximum = -3.402823466e38f;
  float maxima[kSplits], denominators[kSplits], weights[kSplits];
#pragma unroll
  for (int split = 0; split < kSplits; ++split) {
    maxima[split] = stats[(split * kPackedRows + row) * 2];
    denominators[split] = stats[(split * kPackedRows + row) * 2 + 1];
    maximum = fmaxf(maximum, maxima[split]);
  }
  float total = 0;
#pragma unroll
  for (int split = 0; split < kSplits; ++split) {
    weights[split] = exp2f(maxima[split] - maximum);
    total += weights[split] * denominators[split];
  }
  float inverse = 1.f / fmaxf(total, 1.f);
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    float value = 0;
#pragma unroll
    for (int split = 0; split < kSplits; ++split) {
      value += weights[split] * part[(split * kPackedRows + row) * kHeadDim + col + j * 128];
    }
    out[row * kHeadDim + col + j * 128] = __float2bfloat16(value * inverse);
  }
}

constexpr size_t kWorkspaceBytes = (size_t(kSplits) * kPackedRows * kHeadDim +
                                    size_t(kSplits) * kPackedRows * 2) * sizeof(float);

/// One workspace per device, allocated on first use and never freed -- the
/// lifetime note at the top of this file. A second card gets a second buffer
/// rather than a pointer into the first one's; the split-KV merge kernel below
/// *writes* through this pointer, so a shared entry would corrupt the other
/// card's scratch as readily as it would read it.
thread_local std::map<int, float*> g_workspaces;

bool ensure_workspace(int device_id, void* stream) {
  if (g_workspaces.count(device_id) != 0) {
    return true;
  }
  // One block for the whole process and independent of shape, but it is still
  // allocated on first use -- and the first split-KV launch can be inside a
  // capture, which is the same failure as the metadata's.
  if (capturing(stream)) {
    set_error(
        "the %zu byte split-KV workspace was first needed during graph capture; a captured "
        "graph bakes the pointers it recorded, so a split-KV shape must be seen once on the "
        "warmup traversal before musaStreamBeginCapture",
        kWorkspaceBytes);
    return false;
  }
  if (!select_device(device_id)) {
    set_error("the %zu byte split-KV workspace cannot be allocated: MUSA %d could not be selected",
              kWorkspaceBytes, device_id);
    return false;
  }
  float* workspace = nullptr;
  if (musaMalloc(reinterpret_cast<void**>(&workspace), kWorkspaceBytes) != musaSuccess) {
    set_error("musaMalloc failed for the %zu byte split-KV workspace", kWorkspaceBytes);
    return false;
  }
  g_workspaces.emplace(device_id, workspace);
  return true;
}

/// The workspace `ensure_workspace` allocated for this device, or null.
float* split_kv_workspace(int device_id) {
  const auto found = g_workspaces.find(device_id);
  return found == g_workspaces.end() ? nullptr : found->second;
}

}  // namespace

extern "C" const char* apxinf_musa_fa_d256_last_error(void) { return error_buffer(); }

// Packed BF16 attention over a dense BF16 KV.
//
// `variant` is one of the `Variant` values; `tokens` is the query length and
// the query tensor's first dimension; `capacity` is the number of KV rows
// present; `key_tokens` is how many of them participate.
//
// Returns 0 on success. On failure it returns non-zero and
// `apxinf_musa_fa_d256_last_error()` names the reason. Nothing here falls back
// to another variant: a caller that asked for split-KV and got the general
// kernel would be slower with no way to tell.
extern "C" int apxinf_musa_fa_d256(const void* q, const void* k, const void* v, void* out,
                                  int32_t tokens, int32_t capacity, int32_t key_tokens,
                                  int32_t variant, void* stream) {
  if (q == nullptr || k == nullptr || v == nullptr || out == nullptr) {
    set_error("the attention operands must not be null");
    return -1;
  }
  if (tokens <= 0 || capacity <= 0) {
    set_error("the attention shapes must be positive, got tokens %d capacity %d", tokens,
              capacity);
    return -1;
  }
  if (key_tokens <= 0 || key_tokens > capacity) {
    set_error("key_tokens must be in (0, %d], got %d", capacity, key_tokens);
    return -1;
  }
  if (variant < 0 || variant >= kVariantCount) {
    set_error("unknown attention variant %d", variant);
    return -1;
  }
  if ((variant == kPackedDecode || variant == kSplitKv) && tokens != kTokensPerRowGroup) {
    set_error("the %s variant is built for %d query rows, got %d",
              kVariants[variant].name, kTokensPerRowGroup, tokens);
    return -1;
  }
  // There is deliberately no check on `capacity` for the split-KV variant.
  // `kernels/attention.rs`'s `SPLIT_KV_CAPACITY` (512..=640) is the range the
  // port has *measured* the split to win in, not a bound the kernel needs: the
  // split is over the KV axis at run time, and a capacity outside the band
  // computes the same answer, just without the measurement to justify the
  // partitioning. Refusing it here would turn "untested on this shape" into
  // "cannot run", which is the opposite of what an unmeasured shape deserves --
  // see `MusaArchFamily` for the same principle applied to generations.

  // Which card this call runs on, asked of the stream rather than of the thread:
  // every cache below outlives any one context, and a thread that has served
  // another card still holds that card's entries -- see `adapter_common.h`.
  const int device_id = stream_device(stream);

  MUfunction function = load_variant(variant, device_id);
  if (function == nullptr) {
    return -1;
  }

  int32_t* metadata = nullptr;
  if (!ensure_metadata(tokens, key_tokens, device_id, stream, &metadata)) {
    return -1;
  }

  const size_t q_bytes = size_t(tokens) * kHeads * kHeadDim * sizeof(__mt_bfloat16);
  const size_t kv_bytes = size_t(capacity) * kHeadDim * sizeof(__mt_bfloat16);

  // Split-KV writes its partials where a plain launch would write the output,
  // so the workspace takes the output's slot in the parameter block and the
  // merge kernel produces the real output afterwards.
  void* primary = out;
  size_t primary_bytes = q_bytes;
  float* workspace = nullptr;
  if (variant == kSplitKv) {
    if (!ensure_workspace(device_id, stream)) {
      return -1;
    }
    workspace = split_kv_workspace(device_id);
    primary = workspace;
    primary_bytes = kWorkspaceBytes;
  }

  auto primary_rp = robust(primary, primary_bytes), q_rp = robust(q, q_bytes);
  auto k_rp = robust(k, kv_bytes), v_rp = robust(v, kv_bytes);
  auto cq_rp = robust(metadata, 2 * sizeof(int32_t));
  auto uk_rp = robust(metadata + 2, sizeof(int32_t));
  int32_t max_seqlen_q = tokens, kv_capacity = capacity;
  int32_t heads = kHeads, kv_heads = 1;
  // The assembly exponentiates with exp2, so the scale arrives pre-multiplied by
  // log2(e). Computed here rather than passed in because it is a property of the
  // kernel's arithmetic, not of the model.
  float scale_log2 = static_cast<float>((1.0 / std::sqrt(double(kHeadDim))) * std::log2(M_E));
  void* params[] = {&primary_rp, &q_rp, &k_rp, &v_rp, &cq_rp, &uk_rp,
                    &max_seqlen_q, &kv_capacity, &heads, &kv_heads, &scale_log2};

  const VariantInfo& info = kVariants[variant];
  const unsigned rows =
      info.x_is_packed_rows ? unsigned(tokens) * kHeads : unsigned(tokens);
  const unsigned grid_x = (rows + info.rows_per_block - 1) / info.rows_per_block;
  MUresult status = muLaunchKernel(function, grid_x, info.grid_y, 1, info.block, 1, 1, 0,
                                   reinterpret_cast<MUstream>(stream), params, nullptr);
  if (status != MUSA_SUCCESS) {
    set_error("muLaunchKernel failed for the %s attention (MUresult %d)",
              info.name, static_cast<int>(status));
    return -1;
  }

  if (variant == kSplitKv) {
    fa_d256_merge_kernel<<<kPackedRows, 128, 0, reinterpret_cast<musaStream_t>(stream)>>>(
        workspace, workspace + size_t(kSplits) * kPackedRows * kHeadDim,
        static_cast<__mt_bfloat16*>(out));
    const musaError_t launch = musaGetLastError();
    if (launch != musaSuccess) {
      set_error("the split-KV merge kernel failed: %s", musaGetErrorString(launch));
      return -1;
    }
  }
  return 0;
}
