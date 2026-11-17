// Stable C ABI and launch configuration for the custom MUSA operators.
//
// The pattern is `crates/apxinf-cuda/adapters/custom_kernels.cu`'s: the device
// code lives in `kernels/custom/` and is included here (inside an anonymous
// namespace, so it keeps internal linkage), and every `extern "C"` symbol plus
// every `<<<>>>` lives in this file. Operator implementation files do not
// export C ABI symbols; safe Rust contracts reach these through `src/ffi/`.
//
// Contract, and how the status is read
// ------------------------------------
// Every entry point launches its kernel and returns `musaGetLastError()`, which
// **returns the runtime's sticky error and clears it**. The clearing is what
// makes the attribution work, so it is part of the contract rather than an
// implementation detail:
//
// * the slot is empty when a call starts, because every call empties it on the
//   way out — the argument checks at the top return before setting one;
// * so the value returned is this call's, not an earlier one's.
//
// `musaPeekAtLastError()` would break the second property rather than fix
// anything: nothing would clear the slot, so one failure would be reported again
// by every later call for the life of the process. The one place that can leave
// the slot dirty is a caller that discards a runtime status without consuming
// it; `MusaAllocation`'s two `musaFree` sites are the only ones, and both clear.

#include <musa_runtime.h>
#include <cstdlib>
#include <cstring>
#include <musa_bf16.h>
#include <musa_fp16.h>
#ifdef APXINF_MUSA_MP31_FP8
#include <musa_fp8.h>
#endif

#include <cstdint>
#include <map>
#include <tuple>

#include "adapter_common.h"

namespace {
#ifdef APXINF_MUSA_MP31_FP8
#include "../kernels/custom/fp8_quantization.cuh"
#include "../kernels/custom/fp8_activation.cuh"
#endif
#include "../kernels/custom/graph.cuh"
#include "../kernels/custom/math.cuh"
#include "../kernels/custom/activation.cuh"
#include "../kernels/custom/attention.cuh"
#include "../kernels/custom/elementwise.cuh"
#include "../kernels/custom/embedding.cuh"
#include "../kernels/custom/normalization.cuh"
#include "../kernels/custom/rope.cuh"
#include "../kernels/custom/fused.cuh"
#include "../kernels/custom/preprocess.cuh"
#include "../kernels/custom/quantization.cuh"
#include "../kernels/custom/norm_quantization.cuh"
#ifdef APXINF_MUSA_MP31_FP8
#include "../kernels/custom/fp8_normalization.cuh"
#endif
}  // namespace

extern "C" musaError_t apxinf_musa_norm_e4m3(
    const void* input, const void* weight, const void* bias,
    const void* residual, const void* projection_bias, void* hidden, void* output,
    int rows, int cols, int kind, int round_bias, float eps, float scale,
    musaStream_t stream) {
#ifdef APXINF_MUSA_MP31_FP8
  if (!input || !weight || !output || rows <= 0 || cols <= 0 ||
      !(eps > 0.0f) || !std::isfinite(eps) || !(scale > 0.0f) ||
      !std::isfinite(1.0f / scale) || !std::isfinite(scale) || kind < 0 || kind > 2 ||
      (kind == 0 && !bias) || (residual && !hidden)) return musaErrorInvalidValue;
  auto source = static_cast<const __mt_bfloat16*>(input);
  auto weights = static_cast<const __mt_bfloat16*>(weight);
  auto shifts = kind == 2 ? weights + cols : static_cast<const __mt_bfloat16*>(bias);
  auto destination = static_cast<__mt_fp8_e4m3*>(output);
  if (residual) {
    auto skip = static_cast<const __mt_bfloat16*>(residual);
    auto projection_shift = static_cast<const __mt_bfloat16*>(projection_bias);
    auto hidden_output = static_cast<__mt_bfloat16*>(hidden);
    if (cols == 1024 && !round_bias && kind == 0) {
      residual_layer_norm_e4m3_kernel<1024, false, false><<<rows, 256, 0, stream>>>(
          source, projection_shift, skip, weights, shifts, hidden_output, destination, eps, 1.0f / scale);
    } else if (cols == 1536 && round_bias && projection_bias && kind != 1) {
      if (kind == 2)
        residual_layer_norm_e4m3_kernel<1536, true, true><<<rows, 256, 0, stream>>>(
            source, projection_shift, skip, weights, shifts, hidden_output, destination, eps, 1.0f / scale);
      else
        residual_layer_norm_e4m3_kernel<1536, false, true><<<rows, 256, 0, stream>>>(
            source, projection_shift, skip, weights, shifts, hidden_output, destination, eps, 1.0f / scale);
    } else return musaErrorInvalidValue;
  } else if (kind == 1) {
    if (cols != 2048) return musaErrorInvalidValue;
    rms_norm_e4m3_2048_kernel<<<rows, 256, 0, stream>>>(source, weights, destination, eps, 1.0f / scale);
  } else if (kind == 2) {
    layer_norm_e4m3_kernel<true><<<rows, 256, 0, stream>>>(source, weights, shifts, destination, cols, eps, 1.0f / scale);
  } else {
    layer_norm_e4m3_kernel<false><<<rows, 256, 0, stream>>>(source, weights, shifts, destination, cols, eps, 1.0f / scale);
  }
  return musaGetLastError();
#else
  return musaErrorNotSupported;
#endif
}

extern "C" musaError_t apxinf_musa_capture_anchor(musaStream_t stream) {
  capture_anchor_kernel<<<1, 128, 0, stream>>>();
  return musaGetLastError();
}

extern "C" musaError_t apxinf_musa_quantize_bf16_e4m3(
    const void* input, void* output, size_t count, float scale, musaStream_t stream) {
#ifdef APXINF_MUSA_MP31_FP8
  if (!input || !output || !count || !std::isfinite(scale) || scale <= 0)
    return musaErrorInvalidValue;
  const unsigned blocks = static_cast<unsigned>(std::min(size_t(4096), (count + 2047) / 2048));
  quantize_bf16_e4m3_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const __mt_bfloat16*>(input), static_cast<__mt_fp8_e4m3*>(output),
      count, 1.0f / scale);
  return musaGetLastError();
#else
  return musaErrorNotSupported;
#endif
}

// Grid size for a grid-stride elementwise launch.
//
// Capped rather than sized exactly: the loop covers any `count`, and M1000 has
// 8 SMs, so a grid far larger than the device can hold only adds scheduling
// overhead. The cap is deliberately the same shape as CUDA's; what differs is
// that it is not tuned, which this milestone does not require.
namespace {
unsigned int elementwise_grid(int64_t count) {
  const int64_t needed = (count + kThreads - 1) / kThreads;
  constexpr int64_t kMaxBlocks = 4096;
  if (needed > kMaxBlocks) return static_cast<unsigned int>(kMaxBlocks);
  if (needed < 1) return 1;
  return static_cast<unsigned int>(needed);
}

// Grid for the row-major kernels: `x` splits a row into column chunks, `y` is
// the row.
//
// The point of the second dimension is that a thread's column is then an offset
// from its own block's base rather than a remainder. The flat grid-stride form
// these replaced computed `index % cols` (and, in `geglu`, `index / inner` too)
// once per element -- and on M1000 a 64-bit integer division is a software
// routine, measured at 6x the whole kernel's runtime.
//
// `y` is capped at `kMaxGridY`; the kernels loop `row += gridDim.y`, so a
// taller matrix costs occupancy rather than coverage.
//
// **`x` targets about `kElementsPerThread` elements per thread, not one block
// per `kThreads` columns.** Sizing it as `ceil(cols / kThreads)` -- one block
// per 256-column chunk, which is what this did first -- launches 33408 blocks
// for `[522, 32768]` and measures 1.29 ms; the same kernel at 4 chunks per row,
// 2088 blocks, measures 0.83. The same holds for `[512, 4304]`: 0.328 ms at 17
// chunks against 0.194 at 1. Both optima land at ~16 elements per thread, which
// is where `kElementsPerThread` comes from (`rowgrid_variants.mu`). M1000 has
// 8 SMs, so a grid in the tens of thousands is dispatch, not parallelism.
//
// Two floors keep the small shapes from collapsing to one block: at least
// `kMinBlocks` blocks in total, and never more chunks than there are
// 256-column groups (which is what made the chunking cheap in the first place
// -- see the division note above).
dim3 row_grid(int rows, int cols) {
  if (rows < 1) rows = 1;
  if (cols < 1) cols = 1;
  constexpr int64_t kElementsPerThread = 16;
  constexpr int64_t kMinBlocks = 256;
  const int64_t widest = (static_cast<int64_t>(cols) + kThreads - 1) / kThreads;
  int64_t chunks = (static_cast<int64_t>(cols) + kThreads * kElementsPerThread - 1) /
                   (kThreads * kElementsPerThread);
  const int64_t for_occupancy =
      (kMinBlocks + static_cast<int64_t>(rows) - 1) / static_cast<int64_t>(rows);
  if (chunks < for_occupancy) chunks = for_occupancy;
  if (chunks < 1) chunks = 1;
  if (chunks > widest) chunks = widest;
  if (chunks < 1) chunks = 1;
  const unsigned int y = static_cast<unsigned int>(rows < kMaxGridY ? rows : kMaxGridY);
  return dim3(static_cast<unsigned int>(chunks), y, 1);
}

// Whether a device pointer is aligned enough for a vector load of this width.
// The vectorized dispatches below are only legal when every pointer they
// reinterpret-cast satisfies this, so each of them guards on it and falls back
// to the scalar kernel rather than faulting.
bool aligned_to(const void* pointer, uintptr_t alignment) {
  return reinterpret_cast<uintptr_t>(pointer) % alignment == 0;
}
}  // namespace

extern "C" musaError_t apxinf_musa_bias_activation_bf16(
    const void* input, const void* bias, void* output, int rows, int cols,
    int activation, musaStream_t stream) {
  if (input == nullptr || output == nullptr || rows <= 0 || cols <= 0) {
    return musaErrorInvalidValue;
  }
  const auto kind = static_cast<Activation>(activation);
  if (kind != Activation::None && kind != Activation::Gelu &&
      kind != Activation::Silu && kind != Activation::Relu) {
    // Reported rather than silently treated as "none": an unknown code reaching
    // here means the safe facade and this ABI disagree about the enum, and
    // computing the identity would look like a successful run.
    return musaErrorInvalidValue;
  }
  // Eight columns per thread whenever the width divides by eight and every
  // pointer is 16-byte aligned for the load; the arithmetic is the scalar
  // kernel's, so the two agree bit for bit. `[512, 4304]` -- the vision tower's
  // fc1 epilogue -- is `8 * 538` and takes this path.
  if (cols % 8 == 0 && aligned_to(input, 16) && aligned_to(output, 16) &&
      (bias == nullptr || aligned_to(bias, 16))) {
    const dim3 grid = row_grid(rows, cols / 8);
    const auto* wide_in = static_cast<const __mt_bfloat16*>(input);
    const auto* wide_bias = static_cast<const __mt_bfloat16*>(bias);
    auto* wide_out = static_cast<__mt_bfloat16*>(output);
    // `activation` is one of four validated codes, and the bias is a
    // compile-time branch, so the two together are eight instantiations of a
    // kernel whose element loop is otherwise branch-free.
    if (bias == nullptr) {
      if (kind == Activation::None) {
        bias_activation_bf16_vector8_kernel<Activation::None, false>
            <<<grid, kThreads, 0, stream>>>(wide_in, wide_bias, wide_out, rows, cols);
      } else if (kind == Activation::Gelu) {
        bias_activation_bf16_vector8_kernel<Activation::Gelu, false>
            <<<grid, kThreads, 0, stream>>>(wide_in, wide_bias, wide_out, rows, cols);
      } else if (kind == Activation::Silu) {
        bias_activation_bf16_vector8_kernel<Activation::Silu, false>
            <<<grid, kThreads, 0, stream>>>(wide_in, wide_bias, wide_out, rows, cols);
      } else {
        bias_activation_bf16_vector8_kernel<Activation::Relu, false>
            <<<grid, kThreads, 0, stream>>>(wide_in, wide_bias, wide_out, rows, cols);
      }
    } else {
      if (kind == Activation::None) {
        bias_activation_bf16_vector8_kernel<Activation::None, true>
            <<<grid, kThreads, 0, stream>>>(wide_in, wide_bias, wide_out, rows, cols);
      } else if (kind == Activation::Gelu) {
        bias_activation_bf16_vector8_kernel<Activation::Gelu, true>
            <<<grid, kThreads, 0, stream>>>(wide_in, wide_bias, wide_out, rows, cols);
      } else if (kind == Activation::Silu) {
        bias_activation_bf16_vector8_kernel<Activation::Silu, true>
            <<<grid, kThreads, 0, stream>>>(wide_in, wide_bias, wide_out, rows, cols);
      } else {
        bias_activation_bf16_vector8_kernel<Activation::Relu, true>
            <<<grid, kThreads, 0, stream>>>(wide_in, wide_bias, wide_out, rows, cols);
      }
    }
    return musaGetLastError();
  }
  bias_activation_bf16_kernel<<<row_grid(rows, cols), kThreads, 0, stream>>>(
      static_cast<const __mt_bfloat16*>(input),
      static_cast<const __mt_bfloat16*>(bias),
      static_cast<__mt_bfloat16*>(output), rows, cols, kind);
  return musaGetLastError();
}

extern "C" musaError_t apxinf_musa_silu_mul_bf16(
    const void* gate, const void* up, void* output, int64_t count,
    musaStream_t stream) {
  if (gate == nullptr || up == nullptr || output == nullptr || count <= 0) {
    return musaErrorInvalidValue;
  }
  silu_mul_bf16_kernel<<<elementwise_grid(count), kThreads, 0, stream>>>(
      static_cast<const __mt_bfloat16*>(gate),
      static_cast<const __mt_bfloat16*>(up),
      static_cast<__mt_bfloat16*>(output), count);
  return musaGetLastError();
}

extern "C" musaError_t apxinf_musa_silu_bf16(
    const void* input, void* output, int64_t count, musaStream_t stream) {
  if (input == nullptr || output == nullptr || count <= 0) {
    return musaErrorInvalidValue;
  }
  silu_bf16_kernel<<<elementwise_grid(count), kThreads, 0, stream>>>(
      static_cast<const __mt_bfloat16*>(input),
      static_cast<__mt_bfloat16*>(output), count);
  return musaGetLastError();
}

extern "C" musaError_t apxinf_musa_bias_gelu_quant_bf16_e4m3(
    const void* input, const void* bias, void* output, int rows, int cols,
    float scale, musaStream_t stream) {
#ifdef APXINF_MUSA_MP31_FP8
  if (input == nullptr || bias == nullptr || output == nullptr || rows <= 0 ||
      cols <= 0 || !std::isfinite(scale) || scale <= 0.0f) {
    return musaErrorInvalidValue;
  }
  const size_t count = size_t(rows) * size_t(cols);
  if (cols % 8 == 0 && aligned_to(input, 16) && aligned_to(output, 8) &&
      aligned_to(bias, 16)) {
    const dim3 grid = row_grid(rows, cols / 8);
    bias_gelu_quant_bf16_e4m3_vector8_kernel<true>
        <<<grid, kThreads, 0, stream>>>(
            static_cast<const __mt_bfloat16*>(input),
            static_cast<const __mt_bfloat16*>(bias),
            static_cast<__mt_fp8_e4m3*>(output), rows, cols, 1.0f / scale);
    return musaGetLastError();
  }
  const unsigned blocks = static_cast<unsigned>(
      std::min(size_t(4096), (count + 2047) / 2048));
  bias_gelu_quant_bf16_e4m3_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const __mt_bfloat16*>(input),
      static_cast<const __mt_bfloat16*>(bias),
      static_cast<__mt_fp8_e4m3*>(output), count, cols, 1.0f / scale);
  return musaGetLastError();
#else
  return musaErrorNotSupported;
#endif
}

extern "C" musaError_t apxinf_musa_silu_mul_quant_bf16_e4m3(
    const void* gate, const void* up, void* output, int64_t count,
    float scale, musaStream_t stream) {
#ifdef APXINF_MUSA_MP31_FP8
  if (gate == nullptr || up == nullptr || output == nullptr || count <= 0 ||
      !std::isfinite(scale) || scale <= 0.0f) {
    return musaErrorInvalidValue;
  }
  if (count % 8 == 0 && aligned_to(gate, 16) && aligned_to(up, 16) &&
      aligned_to(output, 8)) {
    const unsigned blocks = static_cast<unsigned>(
        std::min(size_t(4096), (size_t(count) / 8 + kThreads - 1) / kThreads));
    silu_mul_quant_bf16_e4m3_vector8_kernel<<<blocks, kThreads, 0, stream>>>(
        static_cast<const __mt_bfloat16*>(gate),
        static_cast<const __mt_bfloat16*>(up),
        static_cast<__mt_fp8_e4m3*>(output), count, 1.0f / scale);
    return musaGetLastError();
  }
  const unsigned blocks = static_cast<unsigned>(
      std::min(size_t(4096), (size_t(count) + 2047) / 2048));
  silu_mul_quant_bf16_e4m3_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const __mt_bfloat16*>(gate),
      static_cast<const __mt_bfloat16*>(up),
      static_cast<__mt_fp8_e4m3*>(output), size_t(count), 1.0f / scale);
  return musaGetLastError();
#else
  return musaErrorNotSupported;
#endif
}

// One block per row. `blockDim.x` must be a multiple of 32 (the reduction's
// contract) and at most 8 * 32, because the kernel's `__shared__ float
// scratch[8]` holds one partial per software warp.
extern "C" musaError_t apxinf_musa_ada_gate_residual_rms_norm_bf16(
    const void* projection, const void* residual, const void* gate_style,
    const void* norm_style, void* hidden, void* normalized, int rows, int cols,
    float eps, musaStream_t stream) {
  if (projection == nullptr || residual == nullptr || gate_style == nullptr ||
      norm_style == nullptr || hidden == nullptr || normalized == nullptr ||
      rows <= 0 || cols <= 0 || !(eps > 0.0f)) {
    return musaErrorInvalidValue;
  }
  // `[10, 1024]` -- the action expert's recorded shape -- is exactly one block
  // of `kThreads` with four columns per thread, so the register-cached
  // single-pass kernel applies and is bit-identical (it keeps the block
  // reduction). Anything else falls to the scalar two-pass kernel.
  if (cols == kThreads * 4 && aligned_to(projection, 8) &&
      aligned_to(residual, 8) && aligned_to(gate_style, 8) &&
      aligned_to(norm_style, 8) && aligned_to(hidden, 8) &&
      aligned_to(normalized, 8)) {
    ada_gate_residual_rms_norm_bf16_vector_kernel<4>
        <<<rows, kThreads, 0, stream>>>(
            static_cast<const __mt_bfloat16*>(projection),
            static_cast<const __mt_bfloat16*>(residual),
            static_cast<const __mt_bfloat16*>(gate_style),
            static_cast<const __mt_bfloat16*>(norm_style),
            static_cast<__mt_bfloat16*>(hidden),
            static_cast<__mt_bfloat16*>(normalized), rows, cols, eps);
    return musaGetLastError();
  }
  ada_gate_residual_rms_norm_bf16_kernel<<<rows, kThreads, 0, stream>>>(
      static_cast<const __mt_bfloat16*>(projection),
      static_cast<const __mt_bfloat16*>(residual),
      static_cast<const __mt_bfloat16*>(gate_style),
      static_cast<const __mt_bfloat16*>(norm_style),
      static_cast<__mt_bfloat16*>(hidden),
      static_cast<__mt_bfloat16*>(normalized), rows, cols, eps);
  return musaGetLastError();
}

// ── RMSNorm ──────────────────────────────────────────────────────────────

extern "C" musaError_t apxinf_musa_rms_norm_bf16(
    const void* input, const void* weight, void* output, int rows, int cols,
    float eps, musaStream_t stream) {
  if (input == nullptr || weight == nullptr || output == nullptr || rows <= 0 ||
      cols <= 0 || !(eps > 0.0f)) {
    return musaErrorInvalidValue;
  }
  const auto* in = static_cast<const __mt_bfloat16*>(input);
  const auto* w = static_cast<const __mt_bfloat16*>(weight);
  auto* out = static_cast<__mt_bfloat16*>(output);
  // The register-cached, single-pass kernel only applies where the row divides
  // the block exactly; every other width keeps the two-pass kernel. `[522, 2048]`
  // -- the recorded shape -- takes the 8-wide branch.
  if (cols == kThreads * 8 && aligned_to(input, 16) &&
      aligned_to(weight, 16) && aligned_to(output, 16)) {
    rms_norm_bf16_vector_kernel<8, 8><<<rows, kThreads, 0, stream>>>(in, w, out,
                                                                     rows, cols, eps);
  } else if (cols == kThreads * 4 && aligned_to(input, 8) &&
             aligned_to(weight, 8) && aligned_to(output, 8)) {
    rms_norm_bf16_vector_kernel<4, 4><<<rows, kThreads, 0, stream>>>(in, w, out,
                                                                     rows, cols, eps);
  } else {
    rms_norm_bf16_kernel<<<rows, kThreads, 0, stream>>>(in, w, out, rows, cols, eps);
  }
  return musaGetLastError();
}

// The scalar two-pass RMSNorm, selected explicitly by the accuracy harness.
// The ordinary entry above takes the register-cached lane kernel for
// `[522, 2048]`, so a switch that only disables the fused producer would never
// exercise this kernel; this entry makes that path reachable.
extern "C" musaError_t apxinf_musa_rms_norm_bf16_scalar(
    const void* input, const void* weight, void* output, int rows, int cols,
    float eps, musaStream_t stream) {
  if (input == nullptr || weight == nullptr || output == nullptr || rows <= 0 ||
      cols <= 0 || !(eps > 0.0f)) {
    return musaErrorInvalidValue;
  }
  rms_norm_bf16_kernel<<<rows, kThreads, 0, stream>>>(
      static_cast<const __mt_bfloat16*>(input),
      static_cast<const __mt_bfloat16*>(weight),
      static_cast<__mt_bfloat16*>(output), rows, cols, eps);
  return musaGetLastError();
}

extern "C" musaError_t apxinf_musa_ada_rms_norm_bf16(
    const void* input, const void* style, void* output, int rows, int cols,
    float eps, musaStream_t stream) {
  if (input == nullptr || style == nullptr || output == nullptr || rows <= 0 ||
      cols <= 0 || !(eps > 0.0f)) {
    return musaErrorInvalidValue;
  }
  // `[10, 1024]` -- the action expert's recorded shape -- is exactly one block
  // of `kThreads` with four columns per thread, so the register-cached
  // single-pass kernel applies, as it does for the gated sibling
  // `apxinf_musa_adaptive_gate_residual_rms_bf16`. Anything else falls to the
  // scalar two-pass kernel.
  if (cols == kThreads * 4 && aligned_to(input, 8) && aligned_to(style, 8) &&
      aligned_to(output, 8)) {
    ada_rms_norm_bf16_vector_kernel<4><<<rows, kThreads, 0, stream>>>(
        static_cast<const __mt_bfloat16*>(input),
        static_cast<const __mt_bfloat16*>(style),
        static_cast<__mt_bfloat16*>(output), rows, cols, eps);
    return musaGetLastError();
  }
  ada_rms_norm_bf16_kernel<<<rows, kThreads, 0, stream>>>(
      static_cast<const __mt_bfloat16*>(input),
      static_cast<const __mt_bfloat16*>(style),
      static_cast<__mt_bfloat16*>(output), rows, cols, eps);
  return musaGetLastError();
}

// ── Fused bias + residual ────────────────────────────────────────────────

extern "C" musaError_t apxinf_musa_bias_residual_bf16(
    const void* projection, const void* bias, const void* residual,
    void* output, int rows, int cols, musaStream_t stream) {
  if (projection == nullptr || residual == nullptr || output == nullptr ||
      rows <= 0 || cols <= 0) {
    return musaErrorInvalidValue;
  }
  // Eight columns per thread where the width divides by eight and the pointers
  // are 16-byte aligned; `[512, 1152]`, the vision tower's fc2 residual, is
  // `8 * 144`. The arithmetic is the scalar kernel's, so the two agree bit for
  // bit.
  if (cols % 8 == 0 && aligned_to(projection, 16) && aligned_to(residual, 16) &&
      aligned_to(output, 16) && (bias == nullptr || aligned_to(bias, 16))) {
    const dim3 grid = row_grid(rows, cols / 8);
    if (bias == nullptr) {
      bias_residual_bf16_vector8_kernel<false><<<grid, kThreads, 0, stream>>>(
          static_cast<const __mt_bfloat16*>(projection), nullptr,
          static_cast<const __mt_bfloat16*>(residual),
          static_cast<__mt_bfloat16*>(output), rows, cols);
    } else {
      bias_residual_bf16_vector8_kernel<true><<<grid, kThreads, 0, stream>>>(
          static_cast<const __mt_bfloat16*>(projection),
          static_cast<const __mt_bfloat16*>(bias),
          static_cast<const __mt_bfloat16*>(residual),
          static_cast<__mt_bfloat16*>(output), rows, cols);
    }
    return musaGetLastError();
  }
  bias_residual_bf16_kernel<<<row_grid(rows, cols), kThreads, 0, stream>>>(
      static_cast<const __mt_bfloat16*>(projection),
      static_cast<const __mt_bfloat16*>(bias),
      static_cast<const __mt_bfloat16*>(residual),
      static_cast<__mt_bfloat16*>(output), rows, cols);
  return musaGetLastError();
}

extern "C" musaError_t apxinf_musa_bias_then_residual_bf16(
    const void* projection, const void* bias, const void* residual,
    void* output, int rows, int cols, musaStream_t stream) {
  if (projection == nullptr || residual == nullptr || output == nullptr ||
      rows <= 0 || cols <= 0) {
    return musaErrorInvalidValue;
  }
  bias_then_residual_bf16_kernel<<<row_grid(rows, cols), kThreads, 0, stream>>>(
      static_cast<const __mt_bfloat16*>(projection),
      static_cast<const __mt_bfloat16*>(bias),
      static_cast<const __mt_bfloat16*>(residual),
      static_cast<__mt_bfloat16*>(output), rows, cols);
  return musaGetLastError();
}

extern "C" musaError_t
apxinf_musa_bias_then_residual_adaptive_layer_norm_bf16(
    const void* projection, const void* projection_bias,
    const void* residual, const void* modulation, void* hidden,
    void* normalized, int rows, int cols, float eps, musaStream_t stream) {
  if (projection == nullptr || projection_bias == nullptr ||
      residual == nullptr || modulation == nullptr || hidden == nullptr ||
      normalized == nullptr || rows != 41 || cols != 1536 || !(eps > 0.0f)) {
    return musaErrorInvalidValue;
  }
  bias_then_residual_adaptive_layer_norm_bf16_cached_1536_kernel
      <<<rows, kThreads, 0, stream>>>(
          static_cast<const __mt_bfloat16*>(projection),
          static_cast<const __mt_bfloat16*>(projection_bias),
          static_cast<const __mt_bfloat16*>(residual),
          static_cast<const __mt_bfloat16*>(modulation),
          static_cast<__mt_bfloat16*>(hidden),
          static_cast<__mt_bfloat16*>(normalized), rows, eps);
  return musaGetLastError();
}

extern "C" musaError_t apxinf_musa_bias_then_residual_layer_norm_bf16(
    const void* projection, const void* projection_bias,
    const void* residual, const void* norm_weight, const void* norm_bias,
    void* hidden, void* normalized, int rows, int cols, float eps,
    musaStream_t stream) {
  if (projection == nullptr || projection_bias == nullptr ||
      residual == nullptr || norm_weight == nullptr || norm_bias == nullptr ||
      hidden == nullptr || normalized == nullptr || rows != 41 ||
      cols != 1536 || !(eps > 0.0f)) {
    return musaErrorInvalidValue;
  }
  bias_then_residual_layer_norm_bf16_cached_1536_kernel
      <<<rows, kThreads, 0, stream>>>(
          static_cast<const __mt_bfloat16*>(projection),
          static_cast<const __mt_bfloat16*>(projection_bias),
          static_cast<const __mt_bfloat16*>(residual),
          static_cast<const __mt_bfloat16*>(norm_weight),
          static_cast<const __mt_bfloat16*>(norm_bias),
          static_cast<__mt_bfloat16*>(hidden),
          static_cast<__mt_bfloat16*>(normalized), rows, eps);
  return musaGetLastError();
}

extern "C" musaError_t apxinf_musa_bias_residual_rms_norm_bf16(
    const void* projection, const void* bias, const void* residual,
    const void* weight, void* hidden, void* normalized, int rows, int cols,
    float eps, musaStream_t stream) {
  if (projection == nullptr || residual == nullptr || weight == nullptr ||
      hidden == nullptr || normalized == nullptr || rows <= 0 || cols <= 0 ||
      !(eps > 0.0f)) {
    return musaErrorInvalidValue;
  }
  // `[522, 2048]` -- the recorded shape -- fits the register-cached kernel
  // exactly (one block of `kThreads`, eight columns per thread), and that kernel
  // is bit-identical to the two-pass one because it keeps the block reduction.
  if (cols == kThreads * 8 && aligned_to(projection, 16) &&
      aligned_to(residual, 16) && aligned_to(weight, 16) &&
      aligned_to(hidden, 16) && aligned_to(normalized, 16) &&
      (bias == nullptr || aligned_to(bias, 16))) {
    // The bias branch is a compile-time parameter: a runtime null test inside the
    // unrolled loop costs more than the load (see `fused.cuh`).
    if (bias == nullptr) {
      bias_residual_rms_norm_bf16_vector8_kernel<false>
          <<<rows, kThreads, 0, stream>>>(
              static_cast<const __mt_bfloat16*>(projection), nullptr,
              static_cast<const __mt_bfloat16*>(residual),
              static_cast<const __mt_bfloat16*>(weight),
              static_cast<__mt_bfloat16*>(hidden),
              static_cast<__mt_bfloat16*>(normalized), rows, cols, eps);
    } else {
      bias_residual_rms_norm_bf16_vector8_kernel<true>
          <<<rows, kThreads, 0, stream>>>(
              static_cast<const __mt_bfloat16*>(projection),
              static_cast<const __mt_bfloat16*>(bias),
              static_cast<const __mt_bfloat16*>(residual),
              static_cast<const __mt_bfloat16*>(weight),
              static_cast<__mt_bfloat16*>(hidden),
              static_cast<__mt_bfloat16*>(normalized), rows, cols, eps);
    }
    return musaGetLastError();
  }
  bias_residual_rms_norm_bf16_kernel<<<rows, kThreads, 0, stream>>>(
      static_cast<const __mt_bfloat16*>(projection),
      static_cast<const __mt_bfloat16*>(bias),
      static_cast<const __mt_bfloat16*>(residual),
      static_cast<const __mt_bfloat16*>(weight),
      static_cast<__mt_bfloat16*>(hidden),
      static_cast<__mt_bfloat16*>(normalized), rows, cols, eps);
  return musaGetLastError();
}

// ── Elementwise ──────────────────────────────────────────────────────────

extern "C" musaError_t apxinf_musa_concat_rows_bf16(
    const void* first, const void* second, void* output, int first_rows,
    int second_rows, int cols, musaStream_t stream) {
  if (first == nullptr || second == nullptr || output == nullptr ||
      first_rows <= 0 || second_rows <= 0 || cols <= 0) {
    return musaErrorInvalidValue;
  }
  const int64_t first_count = static_cast<int64_t>(first_rows) * cols;
  const int64_t total_count = static_cast<int64_t>(first_rows + second_rows) * cols;
  // The vectorized pass needs all three pointers 16-byte aligned and the source
  // boundary on an 8-element boundary, so that `second`'s offset stays a
  // multiple of the load width.
  if (aligned_to(first, 16) && aligned_to(second, 16) && aligned_to(output, 16) &&
      first_count % 8 == 0) {
    const int64_t lanes = (total_count + 7) / 8;
    concat_rows_bf16_vector_kernel<<<elementwise_grid(lanes), kThreads, 0, stream>>>(
        static_cast<const __mt_bfloat16*>(first),
        static_cast<const __mt_bfloat16*>(second),
        static_cast<__mt_bfloat16*>(output), first_count, total_count);
    return musaGetLastError();
  }
  concat_rows_bf16_kernel<<<elementwise_grid(total_count), kThreads, 0, stream>>>(
      static_cast<const __mt_bfloat16*>(first),
      static_cast<const __mt_bfloat16*>(second),
      static_cast<__mt_bfloat16*>(output), first_count, total_count);
  return musaGetLastError();
}

extern "C" musaError_t apxinf_musa_concat_columns_bf16(
    const void* first, const void* second, const void* third, void* output,
    int rows, int first_cols, int second_cols, int third_cols,
    musaStream_t stream) {
  if (first == nullptr || output == nullptr || rows <= 0 || first_cols <= 0 ||
      second_cols < 0 || third_cols < 0 ||
      (second_cols > 0 && second == nullptr) ||
      (third_cols > 0 && third == nullptr)) {
    return musaErrorInvalidValue;
  }
  const int total_cols = first_cols + second_cols + third_cols;
  concat_columns_bf16_kernel<<<row_grid(rows, total_cols), kThreads, 0, stream>>>(
      static_cast<const __mt_bfloat16*>(first),
      static_cast<const __mt_bfloat16*>(second),
      static_cast<const __mt_bfloat16*>(third),
      static_cast<__mt_bfloat16*>(output), rows, first_cols, second_cols,
      third_cols);
  return musaGetLastError();
}

extern "C" musaError_t apxinf_musa_gather_rows_bf16(
    const void* input, const void* indices, void* output, int rows, int cols,
    musaStream_t stream) {
  if (input == nullptr || indices == nullptr || output == nullptr || rows <= 0 ||
      cols <= 0) {
    return musaErrorInvalidValue;
  }
  gather_rows_bf16_kernel<<<row_grid(rows, cols), kThreads, 0, stream>>>(
      static_cast<const __mt_bfloat16*>(input),
      static_cast<const uint32_t*>(indices),
      static_cast<__mt_bfloat16*>(output), rows, cols);
  return musaGetLastError();
}
extern "C" musaError_t apxinf_musa_add_bf16(
    const void* a, const void* b, void* output, int64_t count,
    musaStream_t stream) {
  if (a == nullptr || b == nullptr || output == nullptr || count <= 0) {
    return musaErrorInvalidValue;
  }
  add_bf16_kernel<<<elementwise_grid(count), kThreads, 0, stream>>>(
      static_cast<const __mt_bfloat16*>(a),
      static_cast<const __mt_bfloat16*>(b), static_cast<__mt_bfloat16*>(output),
      count);
  return musaGetLastError();
}

extern "C" musaError_t apxinf_musa_mul_bf16(
    const void* a, const void* b, void* output, int64_t count,
    musaStream_t stream) {
  if (a == nullptr || b == nullptr || output == nullptr || count <= 0) {
    return musaErrorInvalidValue;
  }
  mul_bf16_kernel<<<elementwise_grid(count), kThreads, 0, stream>>>(
      static_cast<const __mt_bfloat16*>(a),
      static_cast<const __mt_bfloat16*>(b), static_cast<__mt_bfloat16*>(output),
      count);
  return musaGetLastError();
}

extern "C" musaError_t apxinf_musa_scatter_rows_bf16(
    const void* destination, const void* source, const void* row_to_source,
    void* output, int destination_rows, int source_rows, int cols, int add,
    musaStream_t stream) {
  if (destination == nullptr || source == nullptr || row_to_source == nullptr ||
      output == nullptr || destination_rows <= 0 || source_rows <= 0 ||
      source_rows > destination_rows || cols <= 0 || (add != 0 && add != 1)) {
    return musaErrorInvalidValue;
  }
  scatter_rows_bf16_kernel<<<row_grid(destination_rows, cols), kThreads, 0, stream>>>(
      static_cast<const __mt_bfloat16*>(destination),
      static_cast<const __mt_bfloat16*>(source),
      static_cast<const uint32_t*>(row_to_source),
      static_cast<__mt_bfloat16*>(output), destination_rows, source_rows, cols,
      add);
  return musaGetLastError();
}

extern "C" musaError_t apxinf_musa_euler_update_bf16(
    const void* state, const void* velocity, void* output, int64_t count,
    float dt, musaStream_t stream) {
  if (state == nullptr || velocity == nullptr || output == nullptr ||
      count <= 0) {
    return musaErrorInvalidValue;
  }
  euler_update_bf16_kernel<<<elementwise_grid(count), kThreads, 0, stream>>>(
      static_cast<const __mt_bfloat16*>(state),
      static_cast<const __mt_bfloat16*>(velocity),
      static_cast<__mt_bfloat16*>(output), count, dt);
  return musaGetLastError();
}

extern "C" musaError_t apxinf_musa_add_position_bf16(
    const void* projection, const void* bias, const void* position, void* output,
    int rows, int cols, int tokens_per_view, musaStream_t stream) {
  if (projection == nullptr || position == nullptr || output == nullptr ||
      rows <= 0 || cols <= 0 || tokens_per_view <= 0 || tokens_per_view > rows ||
      rows % tokens_per_view != 0) {
    return musaErrorInvalidValue;
  }
  add_position_bf16_kernel<<<row_grid(rows, cols), kThreads, 0, stream>>>(
      static_cast<const __mt_bfloat16*>(projection),
      static_cast<const __mt_bfloat16*>(bias),
      static_cast<const __mt_bfloat16*>(position),
      static_cast<__mt_bfloat16*>(output), rows, cols, tokens_per_view);
  return musaGetLastError();
}

// Multiply a BF16 `[rows, cols]` matrix by an f32 `[cols]` per-channel scale,
// rounded once -- the activation half of the W8A8 equalization fold.
extern "C" musaError_t apxinf_musa_mul_channels_bf16(
    const void* input, const void* channel_scale, void* output, int rows,
    int cols, musaStream_t stream) {
  if (input == nullptr || channel_scale == nullptr || output == nullptr ||
      rows <= 0 || cols <= 0) {
    return musaErrorInvalidValue;
  }
  mul_channels_bf16_kernel<<<row_grid(rows, cols), kThreads, 0, stream>>>(
      static_cast<const __mt_bfloat16*>(input),
      static_cast<const float*>(channel_scale),
      static_cast<__mt_bfloat16*>(output), rows, cols);
  return musaGetLastError();
}

// ── Embedding ────────────────────────────────────────────────────────────

// `normalizer` is `bf16(sqrtf(width))` as an f32, computed by the caller: the
// rounding is part of the operator's contract and belongs where it can be read
// next to the comment that explains it, not inside a launch wrapper.
extern "C" musaError_t apxinf_musa_embedding_bf16_scaled(
    const void* table, const void* ids, void* output, int tokens, int width,
    float normalizer, musaStream_t stream) {
  if (table == nullptr || ids == nullptr || output == nullptr || tokens <= 0 ||
      width <= 0) {
    return musaErrorInvalidValue;
  }
  embedding_bf16_scaled_kernel<<<row_grid(tokens, width), kThreads, 0, stream>>>(
      static_cast<const __mt_bfloat16*>(table),
      static_cast<const uint32_t*>(ids), static_cast<__mt_bfloat16*>(output),
      tokens, width, normalizer);
  return musaGetLastError();
}

// ── Attention-side split ─────────────────────────────────────────────────

extern "C" musaError_t apxinf_musa_qkv_split_bias_bf16(
    const void* qkv, const void* bias, void* q, void* k, void* v, int tokens,
    int projection_width, musaStream_t stream) {
  if (qkv == nullptr || q == nullptr || k == nullptr || v == nullptr ||
      tokens <= 0 || projection_width <= 0) {
    return musaErrorInvalidValue;
  }
  // `projection_width % 8 == 0` is what keeps an eight-column lane inside one
  // `q | k | v` segment, so the segment branch never straddles a boundary --
  // `[512, 3456]`, the vision tower's split, has width 1152 = `8 * 144`. The
  // arithmetic is the scalar kernel's, so the two agree bit for bit.
  if (projection_width % 8 == 0 && aligned_to(qkv, 16) && aligned_to(q, 16) &&
      aligned_to(k, 16) && aligned_to(v, 16) &&
      (bias == nullptr || aligned_to(bias, 16))) {
    if (bias == nullptr) {
      qkv_split_bias_bf16_vector8_kernel<false><<<tokens, kThreads, 0, stream>>>(
          static_cast<const __mt_bfloat16*>(qkv), nullptr,
          static_cast<__mt_bfloat16*>(q), static_cast<__mt_bfloat16*>(k),
          static_cast<__mt_bfloat16*>(v), projection_width);
    } else {
      qkv_split_bias_bf16_vector8_kernel<true><<<tokens, kThreads, 0, stream>>>(
          static_cast<const __mt_bfloat16*>(qkv),
          static_cast<const __mt_bfloat16*>(bias), static_cast<__mt_bfloat16*>(q),
          static_cast<__mt_bfloat16*>(k), static_cast<__mt_bfloat16*>(v),
          projection_width);
    }
    return musaGetLastError();
  }
  qkv_split_bias_bf16_kernel<<<tokens, kThreads, 0, stream>>>(
      static_cast<const __mt_bfloat16*>(qkv),
      static_cast<const __mt_bfloat16*>(bias), static_cast<__mt_bfloat16*>(q),
      static_cast<__mt_bfloat16*>(k), static_cast<__mt_bfloat16*>(v),
      projection_width);
  return musaGetLastError();
}

// ── Vision attention's data-movement helpers ─────────────────────────────

// Move between `[tokens, heads, dim]` and `[batch * heads, tokens_per_batch, dim]`.
//
// Three entries, because the two halves of the vision attention do not agree on
// a type. Q and K make the trip as BF16 -- they arrive that way from the QKV
// split and the scores GEMM takes BF16 operands, so for them the permutation
// moves bytes and nothing else, and it only ever runs into the head-major
// layout. V is widened to f32, because the score matrix it multiplies is f32
// (softmax output) and mublas does not mix operand types, so one side has to
// give. The output permutation narrows back on the way out.
//
// Widening Q and K was the shape this started with, and the reason is worth
// keeping: rounding the scores to bf16 costs 0.2% relative on a 72-term dot
// product, and that reaches 1.3% by the time it is a weighted sum of V. Kept
// wide, end-to-end cosine measured 0.99991; taken narrow, it measures 0.99987
// -- both far inside the 0.999 this port is held to. What the narrow path gives
// up is the claim that the port differs from the reference only in summation
// order: a tensor core's intermediate rounding is not a summation order.
// (the port's vision-attention probe and its latency-reduction notes §23-24.)
extern "C" musaError_t apxinf_musa_permute_heads_bf16(
    const void* in, void* out, int tokens, int heads, int dim,
    int tokens_per_batch, musaStream_t stream) {
  if (in == nullptr || out == nullptr || tokens <= 0 || heads <= 0 || dim <= 0 ||
      tokens_per_batch <= 0 || tokens % tokens_per_batch != 0) {
    return musaErrorInvalidValue;
  }
  permute_heads_kernel<__mt_bfloat16, __mt_bfloat16>
      <<<tokens, kPermuteThreads, 0, stream>>>(
          static_cast<const __mt_bfloat16*>(in),
          static_cast<__mt_bfloat16*>(out), tokens, heads, dim, tokens_per_batch,
          1);
  return musaGetLastError();
}

extern "C" musaError_t apxinf_musa_permute_heads_to_f32(
    const void* in, void* out, int tokens, int heads, int dim,
    int tokens_per_batch, musaStream_t stream) {
  if (in == nullptr || out == nullptr || tokens <= 0 || heads <= 0 || dim <= 0 ||
      tokens_per_batch <= 0 || tokens % tokens_per_batch != 0) {
    return musaErrorInvalidValue;
  }
  permute_heads_kernel<__mt_bfloat16, float>
      <<<tokens, kPermuteThreads, 0, stream>>>(
          static_cast<const __mt_bfloat16*>(in), static_cast<float*>(out), tokens,
          heads, dim, tokens_per_batch, 1);
  return musaGetLastError();
}

extern "C" musaError_t apxinf_musa_permute_heads_from_f32(
    const void* in, void* out, int tokens, int heads, int dim,
    int tokens_per_batch, musaStream_t stream) {
  if (in == nullptr || out == nullptr || tokens <= 0 || heads <= 0 || dim <= 0 ||
      tokens_per_batch <= 0 || tokens % tokens_per_batch != 0) {
    return musaErrorInvalidValue;
  }
  permute_heads_kernel<float, __mt_bfloat16>
      <<<tokens, kPermuteThreads, 0, stream>>>(
          static_cast<const float*>(in), static_cast<__mt_bfloat16*>(out), tokens,
          heads, dim, tokens_per_batch, 0);
  return musaGetLastError();
}

// In-place row softmax. One block per row, `kThreads` per block -- the block
// size is fixed by the kernel's shared reduction buffer.
extern "C" musaError_t apxinf_musa_permute_gqa_kv_bf16(
    const void* in, void* out, int tokens, int kv_heads, int q_heads, int dim,
    musaStream_t stream) {
  if (in == nullptr || out == nullptr || tokens <= 0 || kv_heads <= 0 ||
      q_heads <= 0 || q_heads % kv_heads != 0 || dim <= 0) {
    return musaErrorInvalidValue;
  }
  permute_gqa_kv_kernel<__mt_bfloat16><<<tokens, kPermuteThreads, 0, stream>>>(
      static_cast<const __mt_bfloat16*>(in),
      static_cast<__mt_bfloat16*>(out), tokens, kv_heads, q_heads, dim);
  return musaGetLastError();
}

extern "C" musaError_t apxinf_musa_permute_gqa_kv_to_f32(
    const void* in, void* out, int tokens, int kv_heads, int q_heads, int dim,
    musaStream_t stream) {
  if (in == nullptr || out == nullptr || tokens <= 0 || kv_heads <= 0 ||
      q_heads <= 0 || q_heads % kv_heads != 0 || dim <= 0) {
    return musaErrorInvalidValue;
  }
  permute_gqa_kv_kernel<float><<<tokens, kPermuteThreads, 0, stream>>>(
      static_cast<const __mt_bfloat16*>(in), static_cast<float*>(out), tokens,
      kv_heads, q_heads, dim);
  return musaGetLastError();
}
extern "C" musaError_t apxinf_musa_causal_row_softmax_f32(
    void* data, int rows, int tokens, musaStream_t stream) {
  if (data == nullptr || rows <= 0 || tokens <= 0 || rows % tokens != 0) {
    return musaErrorInvalidValue;
  }
  const unsigned int warps_per_block = kThreads / 32;
  const unsigned int blocks =
      static_cast<unsigned int>((static_cast<unsigned int>(rows) + warps_per_block - 1) /
                                warps_per_block);
  causal_row_softmax_f32_kernel<<<blocks, kThreads, 0, stream>>>(
      static_cast<float*>(data), rows, tokens);
  return musaGetLastError();
}

extern "C" musaError_t apxinf_musa_row_softmax_f32(void* data, int rows,
                                                   int cols,
                                                   musaStream_t stream) {
  if (data == nullptr || rows <= 0 || cols <= 0) {
    return musaErrorInvalidValue;
  }
  // One warp per row now, so the grid carries the row count divided by the
  // block's warps; the kernel guards the remainder.
  const unsigned int warps_per_block = kThreads / 32;
  const unsigned int blocks =
      static_cast<unsigned int>((static_cast<unsigned int>(rows) + warps_per_block - 1) /
                                warps_per_block);
  row_softmax_f32_kernel<<<blocks, kThreads, 0, stream>>>(static_cast<float*>(data),
                                                          rows, cols);
  return musaGetLastError();
}

// ── LayerNorm ────────────────────────────────────────────────────────────

extern "C" musaError_t apxinf_musa_layer_norm_bf16(
    const void* input, const void* weight, const void* bias, void* output,
    int rows, int cols, float eps, musaStream_t stream) {
  if (input == nullptr || weight == nullptr || bias == nullptr ||
      output == nullptr || rows <= 0 || cols <= 0 || !(eps > 0.0f)) {
    return musaErrorInvalidValue;
  }
  // `[512, 1152]` -- the vision tower's width -- is `32 * 4 * 9`, so one warp
  // covers a row in nine `uint64_t` loads and the whole row stays in registers
  // across a shuffle-only reduction. That barrier-free shape measures 0.0444 ms
  // against the three-pass kernel's 0.0832 on a clean device; every other width
  // keeps the three-pass kernel.
  if (cols == 32 * 4 * 9 && aligned_to(input, 8) && aligned_to(weight, 8) &&
      aligned_to(bias, 8) && aligned_to(output, 8)) {
    constexpr int kLayerWarpWarps = 4;
    layer_norm_bf16_warp_kernel<4, 9, kLayerWarpWarps>
        <<<(rows + kLayerWarpWarps - 1) / kLayerWarpWarps, kLayerWarpWarps * 32, 0,
           stream>>>(static_cast<const __mt_bfloat16*>(input),
                     static_cast<const __mt_bfloat16*>(weight),
                     static_cast<const __mt_bfloat16*>(bias),
                     static_cast<__mt_bfloat16*>(output), rows, cols, eps);
    return musaGetLastError();
  }
  layer_norm_bf16_kernel<<<rows, kThreads, 0, stream>>>(
      static_cast<const __mt_bfloat16*>(input),
      static_cast<const __mt_bfloat16*>(weight),
      static_cast<const __mt_bfloat16*>(bias), static_cast<__mt_bfloat16*>(output),
      rows, cols, eps);
  return musaGetLastError();
}

extern "C" musaError_t apxinf_musa_adaptive_layer_norm_bf16(
    const void* input, const void* modulation, void* output, int rows, int cols,
    float eps, musaStream_t stream) {
  if (input == nullptr || modulation == nullptr || output == nullptr ||
      rows <= 0 || cols <= 0 || !(eps > 0.0f)) {
    return musaErrorInvalidValue;
  }
  adaptive_layer_norm_bf16_kernel<<<rows, kThreads, 0, stream>>>(
      static_cast<const __mt_bfloat16*>(input),
      static_cast<const __mt_bfloat16*>(modulation),
      static_cast<__mt_bfloat16*>(output), rows, cols, eps);
  return musaGetLastError();
}

extern "C" musaError_t apxinf_musa_bias_residual_layer_norm_bf16(
    const void* projection, const void* projection_bias, const void* residual,
    const void* norm_weight, const void* norm_bias, void* hidden,
    void* normalized, int rows, int cols, float eps, musaStream_t stream) {
  if (projection == nullptr || residual == nullptr || norm_weight == nullptr ||
      norm_bias == nullptr || hidden == nullptr || normalized == nullptr ||
      rows <= 0 || cols <= 0 || !(eps > 0.0f)) {
    return musaErrorInvalidValue;
  }
  if (cols == 1024) {
    bias_residual_layer_norm_bf16_cached_1024_kernel
        <<<rows, kThreads, 0, stream>>>(
            static_cast<const __mt_bfloat16*>(projection),
            static_cast<const __mt_bfloat16*>(projection_bias),
            static_cast<const __mt_bfloat16*>(residual),
            static_cast<const __mt_bfloat16*>(norm_weight),
            static_cast<const __mt_bfloat16*>(norm_bias),
            static_cast<__mt_bfloat16*>(hidden),
            static_cast<__mt_bfloat16*>(normalized), rows, eps);
    return musaGetLastError();
  }
  // Same `[512, 1152]` shape as `layer_norm`, with the projection bias the
  // vision tower passes folded into the register-cached sum. The bias is a
  // compile-time branch -- see the kernel -- so the two cases are separate
  // instantiations, measured at 0.0737 ms (with bias) and 0.0669 ms (without)
  // against the three-pass kernel's 0.1213.
  if (cols == 32 * 4 * 9 && aligned_to(projection, 8) &&
      aligned_to(residual, 8) && aligned_to(norm_weight, 8) &&
      aligned_to(norm_bias, 8) && aligned_to(hidden, 8) &&
      aligned_to(normalized, 8) &&
      (projection_bias == nullptr || aligned_to(projection_bias, 8))) {
    constexpr int kWarpWarps = 8;
    const dim3 grid((rows + kWarpWarps - 1) / kWarpWarps);
    if (projection_bias != nullptr) {
      bias_residual_layer_norm_bf16_warp_kernel<4, 9, kWarpWarps, true>
          <<<grid, kWarpWarps * 32, 0, stream>>>(
              static_cast<const __mt_bfloat16*>(projection),
              static_cast<const __mt_bfloat16*>(projection_bias),
              static_cast<const __mt_bfloat16*>(residual),
              static_cast<const __mt_bfloat16*>(norm_weight),
              static_cast<const __mt_bfloat16*>(norm_bias),
              static_cast<__mt_bfloat16*>(hidden),
              static_cast<__mt_bfloat16*>(normalized), rows, cols, eps);
    } else {
      bias_residual_layer_norm_bf16_warp_kernel<4, 9, kWarpWarps, false>
          <<<grid, kWarpWarps * 32, 0, stream>>>(
              static_cast<const __mt_bfloat16*>(projection),
              static_cast<const __mt_bfloat16*>(projection_bias),
              static_cast<const __mt_bfloat16*>(residual),
              static_cast<const __mt_bfloat16*>(norm_weight),
              static_cast<const __mt_bfloat16*>(norm_bias),
              static_cast<__mt_bfloat16*>(hidden),
              static_cast<__mt_bfloat16*>(normalized), rows, cols, eps);
    }
    return musaGetLastError();
  }
  bias_residual_layer_norm_bf16_kernel<<<rows, kThreads, 0, stream>>>(
      static_cast<const __mt_bfloat16*>(projection),
      static_cast<const __mt_bfloat16*>(projection_bias),
      static_cast<const __mt_bfloat16*>(residual),
      static_cast<const __mt_bfloat16*>(norm_weight),
      static_cast<const __mt_bfloat16*>(norm_bias),
      static_cast<__mt_bfloat16*>(hidden),
      static_cast<__mt_bfloat16*>(normalized), rows, cols, eps);
  return musaGetLastError();
}

// ── Rotary position embedding ────────────────────────────────────────────

// The one place in this file where the grid is not a flat elementwise one, and
// so the one where an argument can be wrong in a way that silently writes
// outside the buffers rather than merely computing nothing: `blockIdx.y` walks
// the q | k | v segments and `threadIdx.x` walks the head dimension. The
// CUDA launcher validates none of this (`static_bf16_adapter.mu` passes
// `cudaGetLastError()` straight through), so a bad `head_dim` there is a silent
// out-of-bounds write. Here it is refused.
// The rope frequency table, kept per thread because the handle and the FP8
// scratch next door are too, and one entry per `(device, half_dim, theta)`:
// the device because the table is memory on the card that allocated it, and a
// thread that serves two cards keeps an entry for each rather than reallocating
// on every alternation. On PI0.5 that is one entry per process: both towers use
// theta 10000 and head_dim 256, on one card.
// `rope_variants.mu` measures what the table is worth -- 0.935 -> 0.184
// ms per call on the prefix shape, 80% of that kernel, with identical values
// because the table comes from the device's own `powf`.
namespace {
thread_local std::map<std::tuple<int, int, float>, const float*> g_rope_frequency;

// Returns nullptr on allocation failure; the caller reports it.
//
// **A superseded table is never freed, and that is the point.** A captured
// graph bakes the pointer it recorded, so freeing a table any graph might still
// name makes it read memory the driver has taken back. Both FA adapters meet
// the same hazard by refusing to allocate during capture; that answer needs a
// message the caller can report, and this adapter has no `..._last_error()` for
// one to land in. Retaining instead removes the hazard rather than guarding it,
// and it is affordable here in a way it is not for those two: their buffers are
// per shape, while this table is per `(device, theta, half_dim)` set -- the
// model config bounds theta and `half_dim`, and a process serves the cards it
// was pointed at. So the cost is a few kilobytes in a process that
// reconfigures, and nothing at all in one that does not.
const float* rope_frequency(int half_dim, float theta, musaStream_t stream) {
  const int device = apxinf_adapters::stream_device(stream);
  const auto key = std::make_tuple(device, half_dim, theta);
  const auto found = g_rope_frequency.find(key);
  if (found != g_rope_frequency.end()) {
    return found->second;
  }
  // The table is device memory, and `musaMalloc` resolves against the current
  // device rather than against the stream that will read it.
  if (!apxinf_adapters::select_device(device)) {
    return nullptr;
  }
  float* table = nullptr;
  if (musaMalloc(reinterpret_cast<void**>(&table),
                 static_cast<size_t>(half_dim) * sizeof(float)) != musaSuccess) {
    return nullptr;
  }
  const int threads = 128;
  rope_frequency_kernel<<<(half_dim + threads - 1) / threads, threads, 0, stream>>>(
      table, half_dim, theta);
  if (musaGetLastError() != musaSuccess) {
    // The launch failed, so nothing is writing this table and it can go. No
    // other entry is touched: each is still the table an earlier call -- and
    // any captured graph -- is entitled to.
    musaFree(table);
    return nullptr;
  }
  g_rope_frequency.emplace(key, table);
  return table;
}
}  // namespace

extern "C" musaError_t apxinf_musa_vision_2d_rope_bf16(
    const void* input, const uint32_t* positions, void* output, int tokens,
    int heads, int head_dim, float theta, musaStream_t stream) {
  if (input == nullptr || positions == nullptr || output == nullptr ||
      tokens <= 0 || heads <= 0 || head_dim <= 0 || head_dim % 4 != 0 ||
      theta <= 0.0f) {
    return musaErrorInvalidValue;
  }
  const int half_dim = head_dim / 2;
  const dim3 grid(
      static_cast<unsigned int>((half_dim + kThreads - 1) / kThreads),
      static_cast<unsigned int>(heads), static_cast<unsigned int>(tokens));
  vision_2d_rope_bf16_kernel<<<grid, kThreads, 0, stream>>>(
      static_cast<const __mt_bfloat16*>(input), positions,
      static_cast<__mt_bfloat16*>(output), tokens, heads, head_dim, theta);
  return musaGetLastError();
}

extern "C" musaError_t apxinf_musa_build_vision_rotation_table_f32(
    const uint32_t* positions, void* rotation_table, int tokens, int head_dim,
    float theta, musaStream_t stream) {
  if (positions == nullptr || rotation_table == nullptr || tokens <= 0 ||
      head_dim <= 0 || head_dim % 4 != 0 || theta <= 0.0f) {
    return musaErrorInvalidValue;
  }
  const int64_t total = static_cast<int64_t>(tokens) * (head_dim / 2);
  build_vision_rotation_table_f32_kernel<<<elementwise_grid(total), kThreads, 0,
                                           stream>>>(
      positions, static_cast<float*>(rotation_table), tokens, head_dim, theta);
  return musaGetLastError();
}

extern "C" musaError_t
apxinf_musa_qkv_split_bias_vision_rope_precomputed_bf16(
    const void* qkv, const void* bias, void* query, void* key, void* value,
    const void* rotation_table, int tokens, int heads, int head_dim,
    musaStream_t stream) {
  if (qkv == nullptr || bias == nullptr || query == nullptr || key == nullptr ||
      value == nullptr || rotation_table == nullptr || tokens <= 0 ||
      heads <= 0 || head_dim <= 0 || head_dim % 4 != 0) {
    return musaErrorInvalidValue;
  }
  const int64_t total =
      static_cast<int64_t>(tokens) * heads * (head_dim / 2);
  qkv_split_bias_vision_rope_precomputed_bf16_kernel<<<
      elementwise_grid(total), kThreads, 0, stream>>>(
      static_cast<const __mt_bfloat16*>(qkv),
      static_cast<const __mt_bfloat16*>(bias),
      static_cast<__mt_bfloat16*>(query),
      static_cast<__mt_bfloat16*>(key),
      static_cast<__mt_bfloat16*>(value),
      static_cast<const float*>(rotation_table), tokens, heads, head_dim);
  return musaGetLastError();
}

extern "C" musaError_t apxinf_musa_mrope_bf16(
    const void* input, const uint32_t* positions, void* output, int tokens,
    int heads, int head_dim, int section_t, int section_h, int section_w,
    float theta, musaStream_t stream) {
  if (input == nullptr || positions == nullptr || output == nullptr ||
      tokens <= 0 || heads <= 0 || head_dim <= 0 || head_dim % 2 != 0 ||
      section_t < 0 || section_h < 0 || section_w < 0 || theta <= 0.0f) {
    return musaErrorInvalidValue;
  }
  (void)section_t;  // temporal is the default axis for the remaining pairs.
  const int half_dim = head_dim / 2;
  const dim3 grid(
      static_cast<unsigned int>((half_dim + kThreads - 1) / kThreads),
      static_cast<unsigned int>(heads), static_cast<unsigned int>(tokens));
  mrope_bf16_kernel<<<grid, kThreads, 0, stream>>>(
      static_cast<const __mt_bfloat16*>(input), positions,
      static_cast<__mt_bfloat16*>(output), tokens, heads, head_dim, section_h,
      section_w, theta);
  return musaGetLastError();
}

extern "C" musaError_t apxinf_musa_build_mrope_rotation_table_f32(
    const uint32_t* positions, void* rotation_table, int tokens, int head_dim,
    int section_t, int section_h, int section_w, float theta,
    musaStream_t stream) {
  if (positions == nullptr || rotation_table == nullptr || tokens <= 0 ||
      head_dim <= 0 || head_dim % 2 != 0 || section_t < 0 || section_h < 0 ||
      section_w < 0 || theta <= 0.0f) {
    return musaErrorInvalidValue;
  }
  (void)section_t;
  const int64_t total = static_cast<int64_t>(tokens) * (head_dim / 2);
  build_mrope_rotation_table_f32_kernel<<<elementwise_grid(total), kThreads, 0,
                                          stream>>>(
      positions, static_cast<float*>(rotation_table), tokens, head_dim,
      section_h, section_w, theta);
  return musaGetLastError();
}

extern "C" musaError_t apxinf_musa_qk_rms_norm_mrope_precomputed_bf16(
    const void* query, const void* query_weight, void* query_output,
    const void* key, const void* key_weight, void* key_output,
    const void* rotation_table, int tokens, int query_heads, int key_heads,
    int head_dim, float eps, int block_threads, musaStream_t stream) {
  if (query == nullptr || query_weight == nullptr || query_output == nullptr ||
      key == nullptr || key_weight == nullptr || key_output == nullptr ||
      rotation_table == nullptr || tokens <= 0 || query_heads <= 0 ||
      key_heads <= 0 || head_dim != 128 || !(eps > 0.0f) ||
      block_threads != 128) {
    return musaErrorInvalidValue;
  }
  qk_rms_norm_mrope_precomputed_bf16_kernel<<<
      dim3(static_cast<unsigned int>(tokens),
           static_cast<unsigned int>(query_heads + key_heads)),
      block_threads, 0, stream>>>(
      static_cast<const __mt_bfloat16*>(query),
      static_cast<const __mt_bfloat16*>(query_weight),
      static_cast<__mt_bfloat16*>(query_output),
      static_cast<const __mt_bfloat16*>(key),
      static_cast<const __mt_bfloat16*>(key_weight),
      static_cast<__mt_bfloat16*>(key_output),
      static_cast<const float*>(rotation_table), tokens, query_heads,
      key_heads, head_dim, eps);
  return musaGetLastError();
}

extern "C" musaError_t apxinf_musa_qkv_rope_bf16(
    const void* qkv, const void* bias, void* q, void* k, void* v, int tokens,
    int q_heads, int kv_heads, int head_dim, float theta, int position_offset,
    int kv_output_offset, musaStream_t stream) {
  if (qkv == nullptr || q == nullptr || k == nullptr || v == nullptr ||
      tokens <= 0 || q_heads <= 0 || kv_heads <= 0 || head_dim <= 0 ||
      head_dim % 2 != 0 || head_dim > 1024 || position_offset < 0 ||
      kv_output_offset < 0) {
    return musaErrorInvalidValue;
  }
  const int half_dim = head_dim / 2;
  // `theta` goes straight into `powf`, so a non-positive one is refused here.
  if (theta <= 0.0f) {
    return musaErrorInvalidValue;
  }
  const float* frequency = rope_frequency(half_dim, theta, stream);
  if (frequency == nullptr) {
    return musaErrorMemoryAllocation;
  }
  const dim3 grid(static_cast<unsigned int>(tokens),
                  static_cast<unsigned int>(q_heads + 2 * kv_heads));
  qkv_rope_bf16_kernel<<<grid, half_dim, 0, stream>>>(
      static_cast<const __mt_bfloat16*>(qkv),
      static_cast<const __mt_bfloat16*>(bias), static_cast<__mt_bfloat16*>(q),
      static_cast<__mt_bfloat16*>(k), static_cast<__mt_bfloat16*>(v), q_heads,
      kv_heads, head_dim, frequency, position_offset, kv_output_offset);
  return musaGetLastError();
}

// ── Gated GELU ───────────────────────────────────────────────────────────

extern "C" musaError_t apxinf_musa_geglu_bf16(const void* gate_up, void* output,
                                              int rows, int inner,
                                              musaStream_t stream) {
  if (gate_up == nullptr || output == nullptr || rows <= 0 || inner <= 0) {
    return musaErrorInvalidValue;
  }
  if (inner % 4 == 0 &&
      reinterpret_cast<uintptr_t>(gate_up) % 8 == 0 &&
      reinterpret_cast<uintptr_t>(output) % 8 == 0) {
    // The packed4 kernel is grid-strided in both dimensions, so the launch
    // policy is a separate decision from the kernel. `row_grid` collapses to one
    // column chunk at this width, which leaves the eight M1000 SMs underfilled;
    // aiming at a few thousand blocks and using 512-thread blocks is ~17% faster
    // on `[522, 32768]` and bit-identical.
    constexpr int kGegluThreads = 512;
    constexpr int64_t kGegluTargetBlocks = 4096;
    const int64_t groups = inner / 4;
    const int64_t widest = (groups + kGegluThreads - 1) / kGegluThreads;
    int64_t chunks = (kGegluTargetBlocks + rows - 1) / rows;
    if (chunks > widest) chunks = widest;
    if (chunks < 1) chunks = 1;
    geglu_bf16_packed4_kernel<<<dim3(static_cast<unsigned>(chunks),
                                     static_cast<unsigned>(rows)),
                                kGegluThreads, 0, stream>>>(
        static_cast<const __mt_bfloat16*>(gate_up),
        static_cast<__mt_bfloat16*>(output), rows, inner);
  } else {
    geglu_bf16_kernel<<<row_grid(rows, inner), kThreads, 0, stream>>>(
        static_cast<const __mt_bfloat16*>(gate_up),
        static_cast<__mt_bfloat16*>(output), rows, inner);
  }
  return musaGetLastError();
}

// ── Raw RGB preprocessing ────────────────────────────────────────────────

// `count` below is the input byte count *and* the output element count --
// `views * 3 * image_size^2` either way, since a patch grid covers the image
// exactly. The kernel's loop bound is derived the same way from the same three
// numbers, so the grid here only has to cover it, which `elementwise_grid`
// does (capped, and the kernel is grid-stride).
//
// `layout` is validated rather than defaulted: the safe facade maps the Rust
// enum to 0/1 and is the only caller, so a third value means the facade and
// this ABI have drifted apart -- and treating that as NHWC would answer a
// transposed buffer with a plausible-looking wrong tensor.
extern "C" musaError_t apxinf_musa_rgb_u8_to_patches_bf16(
    const void* images, void* patches, int views, int image_size, int patch_size,
    int layout, musaStream_t stream) {
  if (images == nullptr || patches == nullptr || views <= 0 || image_size <= 0 ||
      patch_size <= 0 || image_size % patch_size != 0 ||
      (layout != 0 && layout != 1)) {
    return musaErrorInvalidValue;
  }
  const int64_t count =
      static_cast<int64_t>(views) * 3 * image_size * image_size;
  if (layout == 0) {
    rgb_u8_to_patches_bf16_kernel<true>
        <<<elementwise_grid(count), kThreads, 0, stream>>>(
            static_cast<const uint8_t*>(images),
            static_cast<__mt_bfloat16*>(patches), views, image_size,
            patch_size);
  } else {
    rgb_u8_to_patches_bf16_kernel<false>
        <<<elementwise_grid(count), kThreads, 0, stream>>>(
            static_cast<const uint8_t*>(images),
            static_cast<__mt_bfloat16*>(patches), views, image_size,
            patch_size);
  }
  return musaGetLastError();
}

// ── W8A8 activation quantization ─────────────────────────────────────────

// Row-quantize a BF16 `[rows, cols]` activation to INT8, with FP16 row scales.
//
// `output` is `rows * cols` INT8 and `scales` is `rows + kW8A8ScalePad` FP16
// entries; the pad is zeroed here, because the `s8mma` tile's guarded scale load
// can reach into it. One block per row, so `cols` may be any positive width and
// `kThreads` stays inside the quantizer's `scratch[8]` bound (a multiple of 32,
// at most 256).
//
// Two kernels answer this and they compute the same thing, so the choice is
// purely about the shape. The lane form reads the row once into registers and
// moves it in 16-byte loads and 8-byte stores; the scalar port reads it twice,
// two bytes at a time. The lane form therefore needs a row whose every lane is
// a legal 16-byte load and 8-byte store (`cols % 8 == 0`, both pointers
// aligned) and whose whole width fits in one thread's register budget
// (`kW8A8MaxLanes` lanes of `kThreads` threads); anything else keeps the scalar
// kernel, which is correct for every shape.
extern "C" musaError_t apxinf_musa_quantize_rows_bf16_int8(
    const void* input, void* output, void* scales, int rows, int cols,
    const void* inv_c, musaStream_t stream) {
  if (input == nullptr || output == nullptr || scales == nullptr || rows <= 0 ||
      cols <= 0 || !aligned_to(scales, 2)) {
    return musaErrorInvalidValue;
  }
  const auto* source = static_cast<const __mt_bfloat16*>(input);
  auto* destination = static_cast<int8_t*>(output);
  auto* row_scales = static_cast<__half*>(scales);
  // Per-channel equalization factors, or nullptr: `cols` f32 the row is
  // multiplied by before its amax is measured.
  const auto* factors = static_cast<const float*>(inv_c);
  constexpr int kLaneWidth = 8;
  const int64_t per_block = static_cast<int64_t>(kThreads) * kLaneWidth;
  const int64_t lanes = (static_cast<int64_t>(cols) + per_block - 1) / per_block;
  if (cols % kLaneWidth == 0 && lanes <= kW8A8MaxLanes &&
      aligned_to(input, 16) && aligned_to(output, 8) &&
      (factors == nullptr || aligned_to(factors, 16))) {
    // The lane count is a register array's extent, so it has to be a constant:
    // each case is the smallest of 1, 2, 4, 8 that covers `cols`, and the lanes
    // past `cols` load and store nothing.
    if (lanes <= 1) {
      quantize_rows_bf16_int8_lanes_kernel<1><<<rows, kThreads, 0, stream>>>(
          source, destination, row_scales, rows, cols, factors);
    } else if (lanes <= 2) {
      quantize_rows_bf16_int8_lanes_kernel<2><<<rows, kThreads, 0, stream>>>(
          source, destination, row_scales, rows, cols, factors);
    } else if (lanes <= 4) {
      quantize_rows_bf16_int8_lanes_kernel<4><<<rows, kThreads, 0, stream>>>(
          source, destination, row_scales, rows, cols, factors);
    } else {
      quantize_rows_bf16_int8_lanes_kernel<8><<<rows, kThreads, 0, stream>>>(
          source, destination, row_scales, rows, cols, factors);
    }
    return musaGetLastError();
  }
  quantize_rows_bf16_int8_kernel<<<rows, kThreads, 0, stream>>>(
      source, destination, row_scales, rows, cols, factors);
  return musaGetLastError();
}

// Packed GeGLU followed by the same row quantization: `input` is
// `[rows, 2 * cols]` (gate in the first `cols`, up in the second) and the
// output is `[rows, cols]` INT8 with the padded FP16 row scales. The producer
// is the down projection's, and materializing its BF16 intermediate only to
// read it back in the quantizer is what this avoids.
//
// One kernel, one lane form: unlike the quantizer above there is no scalar
// fallback, so the caller checks the shape first (even width, at most 16384
// columns, 16-byte input and 8-byte output alignment, and the same for the
// optional equalization factors). Anything else keeps the two-pass path.
extern "C" musaError_t apxinf_musa_geglu_quantize_rows_bf16_int8(
    const void* input, void* output, void* scales, int rows, int cols,
    const void* inv_c, musaStream_t stream) {
  if (input == nullptr || output == nullptr || scales == nullptr || rows <= 0 ||
      cols <= 0 || cols % 8 != 0 || cols > 16384 || !aligned_to(scales, 2) ||
      !aligned_to(input, 16) || !aligned_to(output, 8) ||
      (inv_c != nullptr && !aligned_to(inv_c, 16))) {
    return musaErrorInvalidValue;
  }
  const auto* source = static_cast<const __mt_bfloat16*>(input);
  auto* destination = static_cast<int8_t*>(output);
  auto* row_scales = static_cast<__half*>(scales);
  const auto* factors = static_cast<const float*>(inv_c);
  // Wider blocks bound the live GELU work to 8/16 values per thread; the
  // quantizer-only 256-thread geometry is slower for this producer on MP22.
  if (cols <= 4096) {
    geglu_quantize_rows_bf16_int8_lanes_kernel<1><<<rows, 512, 0, stream>>>(
        source, destination, row_scales, rows, cols, factors);
  } else {
    geglu_quantize_rows_bf16_int8_lanes_kernel<2><<<rows, 1024, 0, stream>>>(
        source, destination, row_scales, rows, cols, factors);
  }
  return musaGetLastError();
}

extern "C" musaError_t apxinf_musa_bias_gelu_quantize_rows_bf16_int8(
    const void* input, const void* bias, void* output, void* scales,
    int rows, int cols, const void* inv_c, musaStream_t stream) {
  // Measured 4352-column specialization; the Rust facade keeps other widths
  // and unaligned views on bias/GELU followed by the ordinary row quantizer.
  if (!input || !output || !scales || rows <= 0 || cols != 4352 ||
      !aligned_to(scales, 2) || !aligned_to(input, 16) ||
      !aligned_to(output, 8) || (bias && !aligned_to(bias, 16)) ||
      (inv_c && !aligned_to(inv_c, 16)))
    return musaErrorInvalidValue;
  const auto* source = static_cast<const __mt_bfloat16*>(input);
  const auto* bias_values = static_cast<const __mt_bfloat16*>(bias);
  auto* destination = static_cast<int8_t*>(output);
  auto* row_scales = static_cast<__half*>(scales);
  const auto* factors = static_cast<const float*>(inv_c);
  // Five hardware warps keep one eight-value lane per thread. This avoids
  // the register arrays of the 256/512-thread variants on MP22.
  if (bias) {
    bias_gelu_quantize_rows_bf16_int8_lanes_kernel<1, true>
        <<<rows, 640, 0, stream>>>(source, bias_values, destination, row_scales,
                                 rows, cols, factors);
  } else {
    bias_gelu_quantize_rows_bf16_int8_lanes_kernel<1, false>
        <<<rows, 640, 0, stream>>>(source, bias_values, destination, row_scales,
                                 rows, cols, factors);
  }
  return musaGetLastError();
}

extern "C" musaError_t apxinf_musa_ada_rms_quantize_bf16_int8(
    const void* input, const void* residual, const void* gate_style,
    const void* norm_style, void* hidden, void* output, void* scales,
    int rows, int cols, float eps, const void* inv_c, musaStream_t stream) {
  if (!input || !norm_style || !output || !scales || rows <= 0 || cols != 1024 ||
      !aligned_to(scales, 2) || !aligned_to(input, 8) ||
      !aligned_to(norm_style, 8) || !aligned_to(output, 4) ||
      (inv_c && !aligned_to(inv_c, 16)) ||
      (residual && (!gate_style || !hidden || !aligned_to(residual, 8) ||
                    !aligned_to(gate_style, 8) || !aligned_to(hidden, 8))))
    return musaErrorInvalidValue;
  if (residual) {
    ada_rms_quantize_bf16_int8_kernel<true><<<rows, 256, 0, stream>>>(
        static_cast<const __mt_bfloat16*>(input), static_cast<const __mt_bfloat16*>(residual),
        static_cast<const __mt_bfloat16*>(gate_style), static_cast<const __mt_bfloat16*>(norm_style),
        static_cast<__mt_bfloat16*>(hidden), static_cast<int8_t*>(output),
        static_cast<__half*>(scales), rows, cols, eps, static_cast<const float*>(inv_c));
  } else {
    ada_rms_quantize_bf16_int8_kernel<false><<<rows, 256, 0, stream>>>(
        static_cast<const __mt_bfloat16*>(input), nullptr, nullptr,
        static_cast<const __mt_bfloat16*>(norm_style), nullptr, static_cast<int8_t*>(output),
        static_cast<__half*>(scales), rows, cols, eps, static_cast<const float*>(inv_c));
  }
  return musaGetLastError();
}

extern "C" musaError_t apxinf_musa_norm_quantize_bf16_int8(
    const void* input, const void* residual, const void* projection_bias,
    const void* weight, const void* norm_bias, void* hidden, void* output,
    void* scales, int rows, int cols, float eps, const void* inv_c,
    musaStream_t stream) {
  const bool layer = norm_bias != nullptr;
  const int alignment = layer ? 8 : 16;
  if (!input || !weight || !output || !scales || rows <= 0 ||
      cols != (layer ? 1152 : 2048) || !aligned_to(input, alignment) ||
      !aligned_to(weight, alignment) || !aligned_to(output, 8) ||
      !aligned_to(scales, 2) ||
      (norm_bias && !aligned_to(norm_bias, alignment)) ||
      (inv_c && !aligned_to(inv_c, 16)) ||
      (projection_bias &&
       (!residual || !aligned_to(projection_bias, alignment))) ||
      (residual && (!hidden || !aligned_to(residual, alignment) ||
                    !aligned_to(hidden, alignment))))
    return musaErrorInvalidValue;
  if (!residual) {
    if (layer)
      layer_quantize_bf16_int8_kernel<4, false, false>
          <<<(rows + 3) / 4, 128, 0, stream>>>(
              static_cast<const __mt_bfloat16*>(input),
              static_cast<const __mt_bfloat16*>(residual),
              static_cast<const __mt_bfloat16*>(projection_bias),
              static_cast<const __mt_bfloat16*>(weight),
              static_cast<const __mt_bfloat16*>(norm_bias),
              static_cast<__mt_bfloat16*>(hidden), static_cast<int8_t*>(output),
              static_cast<__half*>(scales), rows, cols, eps,
              static_cast<const float*>(inv_c));
    else
      rms_quantize_bf16_int8_kernel<false, false><<<rows, 256, 0, stream>>>(
          static_cast<const __mt_bfloat16*>(input),
          static_cast<const __mt_bfloat16*>(residual),
          static_cast<const __mt_bfloat16*>(projection_bias),
          static_cast<const __mt_bfloat16*>(weight),
          static_cast<const __mt_bfloat16*>(norm_bias),
          static_cast<__mt_bfloat16*>(hidden), static_cast<int8_t*>(output),
          static_cast<__half*>(scales), rows, cols, eps,
          static_cast<const float*>(inv_c));
  } else if (!projection_bias) {
    if (layer)
      layer_quantize_bf16_int8_kernel<8, true, false>
          <<<(rows + 7) / 8, 256, 0, stream>>>(
              static_cast<const __mt_bfloat16*>(input),
              static_cast<const __mt_bfloat16*>(residual),
              static_cast<const __mt_bfloat16*>(projection_bias),
              static_cast<const __mt_bfloat16*>(weight),
              static_cast<const __mt_bfloat16*>(norm_bias),
              static_cast<__mt_bfloat16*>(hidden), static_cast<int8_t*>(output),
              static_cast<__half*>(scales), rows, cols, eps,
              static_cast<const float*>(inv_c));
    else
      rms_quantize_bf16_int8_kernel<true, false><<<rows, 256, 0, stream>>>(
          static_cast<const __mt_bfloat16*>(input),
          static_cast<const __mt_bfloat16*>(residual),
          static_cast<const __mt_bfloat16*>(projection_bias),
          static_cast<const __mt_bfloat16*>(weight),
          static_cast<const __mt_bfloat16*>(norm_bias),
          static_cast<__mt_bfloat16*>(hidden), static_cast<int8_t*>(output),
          static_cast<__half*>(scales), rows, cols, eps,
          static_cast<const float*>(inv_c));
  } else {
    if (layer)
      layer_quantize_bf16_int8_kernel<8, true, true>
          <<<(rows + 7) / 8, 256, 0, stream>>>(
              static_cast<const __mt_bfloat16*>(input),
              static_cast<const __mt_bfloat16*>(residual),
              static_cast<const __mt_bfloat16*>(projection_bias),
              static_cast<const __mt_bfloat16*>(weight),
              static_cast<const __mt_bfloat16*>(norm_bias),
              static_cast<__mt_bfloat16*>(hidden), static_cast<int8_t*>(output),
              static_cast<__half*>(scales), rows, cols, eps,
              static_cast<const float*>(inv_c));
    else
      rms_quantize_bf16_int8_kernel<true, true><<<rows, 256, 0, stream>>>(
          static_cast<const __mt_bfloat16*>(input),
          static_cast<const __mt_bfloat16*>(residual),
          static_cast<const __mt_bfloat16*>(projection_bias),
          static_cast<const __mt_bfloat16*>(weight),
          static_cast<const __mt_bfloat16*>(norm_bias),
          static_cast<__mt_bfloat16*>(hidden), static_cast<int8_t*>(output),
          static_cast<__half*>(scales), rows, cols, eps,
          static_cast<const float*>(inv_c));
  }
  return musaGetLastError();
}
