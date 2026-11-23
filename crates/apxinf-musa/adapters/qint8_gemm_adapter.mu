// Host side of the MP22 INT8 (w8a8) GEMM tiles.
//
// The kernels are `kernels/asm/qint8_*.asm`, rendered from the muDNN
// `ComputeAsmKern` `s8mma` convolution templates configured as a 1x1 GEMM and
// assembled by `build.rs` into `muModuleLoadData` images. This file loads them
// and launches one; it owns no arithmetic that the assembly does not.
//
// Contract
// --------
// INT8, inference only, no bias and no C: `D[m,n] = a_scale[m] * b_scale[n] *
// (A[m,k] x B[n,k]^T)`. A is `[m,k]` row-major INT8, B is `[n,k]` row-major
// INT8 -- **output-major**, which is the physical layout the model layer's
// `int8_dynamic` weights already store (`weights/int8_dynamic.rs`'s
// `quantize_output_channels` writes `quantized[output*input_dim + input]`) -- and
// the two scale vectors are FP16 with a 512-element zero pad past their length.
// D is BF16, written directly by the tile: the vendored templates carry a local
// patch that narrows the f32 accumulator to BF16 in the epilogue (the upstream
// `output_dtype: half` configuration packed FP16 instead), so the single launch
// produces the contract's BF16 result with no widening pass. The two-step
// dequant in that epilogue -- an f32 in-place multiply by the FP16 scales,
// then an f32->bf16 narrowing move -- exists because an MP22 `FOP.MUL` with a
// BF16 destination rejects an FP16 scale source; numerically it rounds the
// same product once, where the old FP16 tile plus widening pass rounded twice
// (f32->f16, then f16->bf16).
//
// The template is a convolution kernel, so the 67-argument ABI it exports is
// that kernel's, not a GEMM's. Every convolution-specific field is pinned to the
// 1x1-NCHW degenerate case (H=W=1, kernel 1x1, stride 1, no padding, one group,
// no bias, identity activation) which is exactly how the vendored reference
// launcher `kernels/qint8_mma_tiles/qint8_asm_launcher.cc` drives it, and the
// argument order is that launcher's -- `kernels/asm/README.md` records it.
//
// Falling back
// ------------
// Unlike the attention adapter, a failure here is not the end of the operator:
// `kernels/gemm/w8a8.rs` reports it as an error, and every failure below happens
// *before* the kernel runs -- an empty image, a module that will not load, a
// symbol the module does not export, a launch the driver refuses -- so nothing
// has been written to the output. The one exception is the row split near the
// end: the whole-row block can have run before the remainder's launch is
// refused, which leaves the remainder's rows unwritten. The caller sees only
// the error -- the operator hands back no output -- so those rows are never
// read. A remainder tile that merely fails to *load*, which is what a first use
// inside a capture would be, is not fatal: the single-launch path below then
// computes the whole shape with padding.
//
// Threading and lifetime
// ----------------------
// The loaded modules are `thread_local` and keyed by `(device, tile)`, as in
// `asm_gemm_adapter.mu` and for the same reasons.

#include <musa.h>
#include <musa_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <utility>

#include "adapter_common.h"
#include "mubin_images.h"

namespace {

/// One tile: what to load, what to call, and the launch shape it wants.
///
/// **The order is load-bearing.** `build.rs`'s `QINT8_ASM_VARIANTS` and
/// `src/kernels/gemm/qint8_tiles.rs`'s `QINT8_TILES` are the same list in the
/// same order, and the selector hands back an index into it.
struct TileInfo {
  const char* name;
  const unsigned char* image;
  unsigned long size;
  /// The symbol the image exports.
  const char* kernel_symbol;
  uint32_t tile_m;
  uint32_t tile_n;
  /// The contraction depth one pass of the tile covers, in INT8 elements. `k`
  /// must be a multiple of it.
  uint32_t tile_k;
  /// Threads per block.
  unsigned block_x;
};

const TileInfo kTiles[] = {
    {"256x128", kApxinfMusaQint8Gemm256x128Image, kApxinfMusaQint8Gemm256x128ImageSize,
     "musa_asm_s8mma_nhwc_conv_KRSC_256_256x128_lin_ol_outbf16", 256, 128, 64, 256},
    {"256x256", kApxinfMusaQint8Gemm256x256Image, kApxinfMusaQint8Gemm256x256ImageSize,
     "musa_asm_s8mma_nhwc_conv_KRSC_512_256x256_lin_ol_outbf16", 256, 256, 64, 512},
    {"128x128", kApxinfMusaQint8Gemm128x128Image, kApxinfMusaQint8Gemm128x128ImageSize,
     "musa_asm_s8mma_nhwc_conv_KRSC_128_128x128_lin_ol_outbf16", 128, 128, 64, 128},
    {"256x64", kApxinfMusaQint8Gemm256x64Image, kApxinfMusaQint8Gemm256x64ImageSize,
     "musa_asm_s8mma_nhwc_conv_KRSC_256_256x64_lin_ol_outbf16", 256, 64, 64, 256},
    {"32x128", kApxinfMusaQint8Gemm32x128Image, kApxinfMusaQint8Gemm32x128ImageSize,
     "musa_asm_s8mma_nhwc_conv_KRSC_128_32x128_lin_ol_outbf16", 32, 128, 64, 128},
    {"32x64", kApxinfMusaQint8Gemm32x64Image, kApxinfMusaQint8Gemm32x64ImageSize,
     "musa_asm_s8mma_nhwc_conv_KRSC_128_32x64_lin_ol_outbf16", 32, 64, 64, 128},
};
constexpr int kTileCount = static_cast<int>(sizeof(kTiles) / sizeof(kTiles[0]));

/// The index of the tile called `name` in [`kTiles`], or `-1`.
int tile_index(const char* name) {
  for (int i = 0; i < kTileCount; ++i) {
    if (std::strcmp(kTiles[i].name, name) == 0) {
      return i;
    }
  }
  return -1;
}

/// A remainder tile the probe measured ahead of [`select_tail_tile`]'s rule.
struct TailOverride {
  uint32_t n;
  uint32_t k;
  const char* name;
};

/// The remainders the probe found the shortest-narrowest rule wrong on.
///
/// The rule picks `32x64` for every π0.5 remainder. An INT8 tile's ten-row block
/// is small enough that how its column blocks land on a wave decides the launch:
/// `32x64` makes `n / 64` blocks against the 32 a full MP22 wave holds, so
/// `n = 2560` (40 blocks) needs two waves where `32x128`'s 20 fit in one, and the
/// second, eighth-full wave costs more than the wider tile's extra work. The
/// encode `(n, k)` pairs at their ten-row remainder (50 warmups / 200 launches,
/// median of three, device-event timing):
///
///   `(2560, 2048)`   `32x64` 0.0974 ms  `32x128` 0.0771 ms   <- override, 26%
///   `(2560, 1024)`   `32x64` 0.0588 ms  `32x128` 0.0460 ms   <- override, 28%
///   `(32768, 2048)`  `32x64` 0.9768 ms  `32x128` 1.0294 ms
///   `(2048, 2048)`   `32x64` 0.0624 ms  `32x128` 0.0682 ms
///   `(2048, 16384)`  `32x64` 0.5344 ms  `32x128` 0.6036 ms
///
/// so the two `n = 2560` pairs override the rule, and `k` is carried in the
/// table so that a contraction depth not timed here does not inherit their
/// decision. `32x64` stays the aggregate winner, which is why it leads
/// `qint8_tiles.rs`'s short-output preference too. A shape not listed keeps the
/// rule.
constexpr TailOverride kTailOverrides[] = {
    {2560, 2048, "32x128"},
    {2560, 1024, "32x128"},
};

/// The FP16 zero pad both scale vectors carry past their length.
///
/// The template indexes a scale by tile group, and a masked group can reach past
/// the tensor's own length; the reference launcher pads by 512 and so does this
/// one, so the indexed rows are always inside an allocation the caller sized.
constexpr uint32_t kScalePad = 512;

enum class LoadState {
  kUntried,
  kReady,
  kPermanentFailure,
};

struct Loaded {
  LoadState state = LoadState::kUntried;
  MUfunction function = nullptr;
};

thread_local std::map<std::pair<int, int>, Loaded> g_loaded;

using apxinf_adapters::capturing;
using apxinf_adapters::error_buffer;
using apxinf_adapters::select_device;
using apxinf_adapters::set_error;
using apxinf_adapters::stream_device;

MUfunction load_tile(int tile, int device, void* stream) {
  Loaded& loaded = g_loaded[std::make_pair(device, tile)];
  const TileInfo& info = kTiles[tile];
  if (loaded.state == LoadState::kReady) {
    return loaded.function;
  }
  if (loaded.state == LoadState::kPermanentFailure) {
    set_error(
        "the %s INT8 GEMM tile is unusable and this thread already reported why; the message "
        "from the first attempt has been overwritten by later calls",
        info.name);
    return nullptr;
  }

  if (info.image == nullptr || info.size == 0) {
    loaded.state = LoadState::kPermanentFailure;
    set_error(
        "the %s INT8 GEMM tile image is empty: the MUSA toolkit's llvm-mc/lld were unavailable, "
        "the assembly failed to build, or this is not an MP22 target (see the build warnings)",
        info.name);
    return nullptr;
  }

  if (capturing(stream)) {
    set_error(
        "the %s INT8 GEMM tile was first needed during graph capture; a captured graph bakes "
        "the handles it recorded, so every tile this model uses must be seen once on the warmup "
        "traversal before musaStreamBeginCapture",
        info.name);
    return nullptr;
  }

  // `muModuleLoadData` loads into the *current* context, so the key and the
  // context have to name the same card. `asm_gemm_adapter.mu` gets this by
  // keying on `muCtxGetCurrent`; here the key is the stream's device, so that
  // device is selected first.
  if (!select_device(device)) {
    set_error("the %s INT8 GEMM tile cannot select device %d", info.name, device);
    return nullptr;
  }

  MUmodule module = nullptr;
  MUresult status = muModuleLoadData(&module, info.image);
  if (status != MUSA_SUCCESS) {
    set_error("muModuleLoadData failed for the %s INT8 GEMM tile (MUresult %d)", info.name,
              static_cast<int>(status));
    return nullptr;
  }
  MUfunction function = nullptr;
  status = muModuleGetFunction(&function, module, info.kernel_symbol);
  if (status != MUSA_SUCCESS) {
    set_error("muModuleGetFunction failed for the %s INT8 GEMM tile (MUresult %d)", info.name,
              static_cast<int>(status));
    muModuleUnload(module);
    loaded.state = LoadState::kPermanentFailure;
    return nullptr;
  }
  loaded.function = function;
  loaded.state = LoadState::kReady;
  return function;
}

/// The tile to run a row block's remainder with: the shortest, then narrowest.
///
/// The `m = 522` shapes leave a ten-row remainder, and a tile as tall as the one
/// that served the whole block would spend most of that height padding. The
/// narrow half of the tiebreak keeps the remainder's weight re-read on the
/// smallest possible column block.
///
/// A remainder the probe timed against the rule is taken from
/// [`kTailOverrides`] first, since an INT8 remainder's launch is short enough
/// that the rule's own margin can invert.
int select_tail_tile(uint32_t n, uint32_t k) {
  int best = -1;
  for (const TailOverride& candidate : kTailOverrides) {
    if (candidate.n == n && candidate.k == k) {
      const int index = tile_index(candidate.name);
      if (index >= 0) {
        const TileInfo& info = kTiles[index];
        if (n % info.tile_n == 0 && k % info.tile_k == 0) {
          return index;
        }
      }
    }
  }
  for (int i = 0; i < kTileCount; ++i) {
    const TileInfo& info = kTiles[i];
    if (n % info.tile_n != 0 || k % info.tile_k != 0) {
      continue;
    }
    if (best < 0 || info.tile_m < kTiles[best].tile_m ||
        (info.tile_m == kTiles[best].tile_m && info.tile_n < kTiles[best].tile_n)) {
      best = i;
    }
  }
  return best;
}

/// Launch one tile on `rows` rows of `A[rows,k] x B[n,k]^T -> D[rows,n]`.
///
/// `a`, `d` and `a_scale` are already offset to the block's first row; `b` and
/// `b_scale` are the whole weight and its whole scale vector, because every row
/// block contracts the same `k` rows of `B` against the same output columns.
///
/// The four `_bytes` guard registers the template's guarded loads take are
/// computed here from this launch's own extents, not inherited from the caller's
/// allocation: the scale guards especially must be this block's padded length
/// (`rows + kScalePad` entries), because a row-split tail sits at an offset into
/// the caller's scale vector and a guard sized for the whole vector would let the
/// kernel read past the end of that allocation.
MUresult launch_rows(MUfunction function, const TileInfo& info, const void* a, const void* b,
                     void* d, const void* a_scale, const void* b_scale, uint32_t rows, uint32_t n,
                     uint32_t k, void* stream) {
  void* null_ptr = nullptr;
  uint32_t zero = 0;

  void* d_arg = d;
  void* a_arg = const_cast<void*>(a);
  void* b_arg = const_cast<void*>(b);
  void* a_scale_arg = const_cast<void*>(a_scale);
  void* b_scale_arg = const_cast<void*>(b_scale);
  uint32_t d_bytes_arg = static_cast<uint32_t>(static_cast<size_t>(rows) * n * 2);
  uint32_t a_bytes_arg = static_cast<uint32_t>(static_cast<size_t>(rows) * k);
  uint32_t b_bytes_arg = static_cast<uint32_t>(static_cast<size_t>(k) * n);
  uint32_t a_scale_bytes = (rows + kScalePad) * 2;
  uint32_t b_scale_bytes = (n + kScalePad) * 2;

  // The 1x1 conv-as-GEMM degenerate configuration. Every field the template's
  // convolution machinery reads is pinned to the case where the spatial extent
  // is one, so `nr_batch` is the row count, `icpg`/`ocpg` are `k`/`n`, and the
  // strides are the row lengths.
  uint32_t nr_batch = rows, icpg = k, h = 1, w = 1, fh = 1, fw = 1;
  uint32_t ocpg = n, oh = 1, ow = 1, sh = 1, sw = 1, ph = 0, pw = 0, dh = 1, dw = 1;
  // MakeFastDivmod(1) for both: the divisor is one, so the packed reciprocal the
  // template multiplies by is unused and the launcher's convention is {1, 0, 0}.
  uint32_t fast_ohw_divisor = 1, fast_ohw_multiplier = 0, fast_ohw_shift = 0;
  uint32_t fast_ow_divisor = 1, fast_ow_multiplier = 0, fast_ow_shift = 0;
  uint32_t nr_m = rows, oc = n, ic_align_tilek = k, nr_k = k, fw_mul_dw = 1, rsc = k, rs = 1;
  uint32_t pa_stride = 64, pa_rewind = k, pa_rewind_h = 0;
  uint32_t pb_ocpg = n, pb_step = 2 * n, pb_stride = 64, pb_rewind = 0;
  uint32_t ldb = k, icpg_sub_1 = k - 1, icpg_sub_tilek = k - 64;
  uint32_t batch_stride = n, group = 1, with_bias = 0;
  float rcp_scale_o = 32.0f, alpha = 1.0f, beta = 0.0f, leaky_relu_alpha = 0.0f;
  uint32_t act_mode = 0;
  uint32_t bias_buffer_size = 0, input_z_buffer_size = 0;

  // This order is the 67 explicit parameters in the generated `.s` metadata, and
  // the count was pinned by the reference launcher's `static_assert`.
  void* args[] = {
      &d_arg, &d_bytes_arg, &zero,
      &a_arg, &a_bytes_arg, &zero,
      &b_arg, &b_bytes_arg, &zero,
      &null_ptr, &bias_buffer_size, &zero,
      &null_ptr, &input_z_buffer_size, &zero,
      &nr_batch, &icpg, &h, &w, &fh, &fw, &ocpg, &oh, &ow,
      &sh, &sw, &ph, &pw, &dh, &dw,
      &fast_ohw_divisor, &fast_ohw_multiplier, &fast_ohw_shift,
      &fast_ow_divisor, &fast_ow_multiplier, &fast_ow_shift,
      &nr_m, &oc, &ic_align_tilek, &nr_k, &fw_mul_dw, &rsc, &rs,
      &pa_stride, &pa_rewind, &pa_rewind_h, &pb_ocpg, &pb_step,
      &pb_stride, &pb_rewind, &ldb, &icpg_sub_1, &icpg_sub_tilek,
      &batch_stride, &group, &with_bias, &rcp_scale_o, &act_mode,
      &alpha, &beta, &leaky_relu_alpha,
      &a_scale_arg, &a_scale_bytes, &zero,
      &b_scale_arg, &b_scale_bytes, &zero,
  };
  static_assert(sizeof(args) / sizeof(args[0]) == 67, "generated ABI parameter count changed");

  // `n` is a multiple of `tile_n` by construction (the caller gates on it), so
  // `grid_x` is exact; `m` is rounded up because the kernel writes a partial row
  // tile with the `remain_m` epilogue predicate.
  const unsigned grid_x = n / info.tile_n;
  const unsigned grid_y = (rows + info.tile_m - 1) / info.tile_m;
  return muLaunchKernel(function, grid_x, grid_y, 1, info.block_x, 1, 1, 0,
                        reinterpret_cast<MUstream>(stream), args, nullptr);
}

}  // namespace

extern "C" const char* apxinf_musa_asm_qint8_gemm_last_error(void) { return error_buffer(); }

/// One INT8 GEMM: `A[m,k] x B[n,k]^T -> D[m,n]` (BF16), with FP16 row/column
/// scales. Returns 0 on success; on failure it returns non-zero and
/// [`apxinf_musa_asm_qint8_gemm_last_error`] names the reason.
extern "C" int apxinf_musa_asm_qint8_gemm(const void* a, const void* b, void* d,
                                          const void* a_scale, const void* b_scale,
                                          uint32_t a_scale_bytes, uint32_t b_scale_bytes,
                                          int32_t m, int32_t n, int32_t k, int32_t tile,
                                          void* stream) {
  if (a == nullptr || b == nullptr || d == nullptr || a_scale == nullptr || b_scale == nullptr) {
    set_error("the INT8 GEMM operands must not be null");
    return -1;
  }
  if (tile < 0 || tile >= kTileCount) {
    set_error("unknown INT8 GEMM tile %d", tile);
    return -1;
  }
  if (m <= 0 || n <= 0 || k <= 0) {
    set_error("the INT8 GEMM shapes must be positive, got m %d n %d k %d", m, n, k);
    return -1;
  }
  const TileInfo& info = kTiles[tile];
  // Repeated rather than trusted: a tile launched on a shape it does not cover
  // computes the wrong thing and returns success, which is the one failure mode
  // nothing downstream can tell from a correct launch. These tiles mask a
  // partial `M` row but not a partial `N` column or `K` row, so `n` must be a
  // whole number of tile columns and `k` a whole number of tile rows.
  if (static_cast<uint32_t>(n) < info.tile_n || static_cast<uint32_t>(n) % info.tile_n != 0) {
    set_error("the %s INT8 GEMM tile needs n to be a positive multiple of %u, got %d", info.name,
              info.tile_n, n);
    return -1;
  }
  if (static_cast<uint32_t>(k) < info.tile_k || static_cast<uint32_t>(k) % info.tile_k != 0) {
    set_error("the %s INT8 GEMM tile needs k to be a positive multiple of %u, got %d", info.name,
              info.tile_k, k);
    return -1;
  }
  // The scale vectors are FP16 with a 512-element zero pad, and the launch below
  // reads up to `m + kScalePad` / `n + kScalePad` entries of them; a short
  // allocation would be read past rather than caught by the guarded load.
  const uint32_t a_scale_floor = (static_cast<uint32_t>(m) + kScalePad) * 2;
  const uint32_t b_scale_floor = (static_cast<uint32_t>(n) + kScalePad) * 2;
  if (a_scale_bytes < a_scale_floor || b_scale_bytes < b_scale_floor) {
    set_error("the INT8 GEMM scale vectors need %u and %u bytes, got %u and %u", a_scale_floor,
              b_scale_floor, a_scale_bytes, b_scale_bytes);
    return -1;
  }

  const int device = stream_device(stream);
  MUfunction function = load_tile(tile, device, stream);
  if (function == nullptr) {
    return -1;
  }

  const uint32_t arg_n = static_cast<uint32_t>(n);
  const uint32_t arg_k = static_cast<uint32_t>(k);
  // The rows the tile cannot fill, split off exactly as the BF16 tiles are: the
  // whole-tile rows run first, then the remainder starts where they ended -- in
  // the activation, its scale and the output, but not in the weight, which both
  // blocks contract in full. What the split recovers is the padding a tile tall
  // enough to hold the last rows computes past them: `m = 522` rounds to 768
  // rows on the 256x128 tile and to 544 with a 32-row remainder tile.
  const uint32_t whole_rows = (static_cast<uint32_t>(m) / info.tile_m) * info.tile_m;
  if (whole_rows > 0 && whole_rows < static_cast<uint32_t>(m)) {
    const int tail_tile = select_tail_tile(arg_n, arg_k);
    MUfunction tail_function = tail_tile < 0 ? nullptr : load_tile(tail_tile, device, stream);
    // A remainder tile that will not load is not a reason to fail the operator:
    // the one-launch path below still computes the right answer, just with the
    // padding.
    if (tail_function != nullptr) {
      const TileInfo& tail_info = kTiles[tail_tile];
      const uint32_t tail_rows = static_cast<uint32_t>(m) - whole_rows;

      MUresult status =
          launch_rows(function, info, a, b, d, a_scale, b_scale, whole_rows, arg_n, arg_k, stream);
      if (status != MUSA_SUCCESS) {
        set_error(
            "muLaunchKernel failed for the %s INT8 GEMM tile on its %u whole rows (MUresult %d)",
            info.name, whole_rows, static_cast<int>(status));
        return -1;
      }
      // Row offset times each operand's row length: `whole_rows * k` bytes into
      // a row-major `[m,k]` INT8 activation, `whole_rows * 2` into the FP16
      // scale vector, `whole_rows * n * 2` into the FP16 output.
      const char* tail_a = static_cast<const char*>(a) + static_cast<size_t>(whole_rows) * arg_k;
      const char* tail_a_scale =
          static_cast<const char*>(a_scale) + static_cast<size_t>(whole_rows) * 2;
      char* tail_d = static_cast<char*>(d) + static_cast<size_t>(whole_rows) * arg_n * 2;
      status = launch_rows(tail_function, tail_info, tail_a, b, tail_d, tail_a_scale, b_scale,
                           tail_rows, arg_n, arg_k, stream);
      if (status != MUSA_SUCCESS) {
        set_error(
            "muLaunchKernel failed for the %s INT8 GEMM tile on its %u tail rows (MUresult %d)",
            tail_info.name, tail_rows, static_cast<int>(status));
        return -1;
      }
      return 0;
    }
  }

  MUresult status =
      launch_rows(function, info, a, b, d, a_scale, b_scale, static_cast<uint32_t>(m), arg_n,
                  arg_k, stream);
  if (status != MUSA_SUCCESS) {
    set_error("muLaunchKernel failed for the %s INT8 GEMM tile (MUresult %d)", info.name,
              static_cast<int>(status));
    return -1;
  }
  return 0;
}
