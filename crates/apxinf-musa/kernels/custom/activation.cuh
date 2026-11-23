#pragma once

// Derived in part from `crates/apxinf-cuda/kernels/custom/` (Apache-2.0,
// Copyright 2026 apxinf contributors); the differences are explained below.

#include "math.cuh"

// Pure device code, grouped by physical operation. Launch policy and the C ABI
// live under `adapters/`; nothing here is `extern "C"`.
//
// BF16 epilogues retain FP32 activation math and explicit BF16 output
// rounding. GeGLU uses packed4 for aligned widths and scalar tails otherwise.
//
// The activation codes are the CUDA kernel ABI's, so a diagnosable name never
// has to be renumbered across backends: 0 none, 1 GELU (tanh approximation),
// 2 SiLU.

__global__ void bias_activation_bf16_kernel(const __mt_bfloat16* input,
                                            const __mt_bfloat16* bias,
                                            __mt_bfloat16* output, int rows,
                                            int cols, int activation) {
  // Rows on `gridDim.y` and column chunks on `gridDim.x`, so the column is an
  // offset from the block's own base and never a remainder. The flat
  // grid-stride form this replaced took `index % cols` per element, and on
  // M1000 a 64-bit integer division is a software routine: measuring it (the
  // same work with and without) put 6x of this kernel's runtime in the divide
  // measured directly, with and without the division.
  const int stride = blockDim.x * gridDim.x;
  for (int row = blockIdx.y; row < rows; row += gridDim.y) {
    const int64_t base = static_cast<int64_t>(row) * cols;
    // `bias` is row-broadcast: one value per column, for every row. The bridge
    // reaches the same place through tensor broadcasting, which is why the
    // recorded tensors do not depend on the row count.
    for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < cols;
         col += stride) {
      float value = __bfloat162float(input[base + col]);
      if (bias != nullptr) value += __bfloat162float(bias[col]);
      if (activation == 1) value = gelu_tanh(value);
      if (activation == 2) value = value / (1.0f + expf(-value));
      output[base + col] = __float2bfloat16(value);
    }
  }
}

union __align__(16) ActivationPacked8 {
  uint4 bits;
  __mt_bfloat16 lanes[8];
};

// Vectorized bias + activation epilogue: eight columns per thread in one 16-byte
// load and one 16-byte store.
//
// Both branches the scalar kernel above takes per element are **compile-time**
// parameters here -- the bias presence, and which activation -- for the reason
// `fused.cuh`'s register-cached kernels record: a runtime null test or code test
// inside the element loop costs more than the load it guards. The scalar form
// also moves 2 bytes per access, which is what its ceiling is really made of:
// on `[512, 4304]` -- the vision tower's fc1 epilogue -- it measures 0.2163 ms
// (40.8 GB/s) with GELU and 0.1862 (47.4) with the activation off, so the
// `tanh` is 13% and the rest is scalar traffic. Requires `cols % 8 == 0`; the
// launcher checks that and the pointer alignment, and every other width keeps
// the scalar kernel.
//
// The arithmetic and the single BF16 rounding are unchanged, so this is
// bit-identical to the scalar form.
template <int ACTIVATION, bool HAS_BIAS>
__global__ void bias_activation_bf16_vector8_kernel(
    const __mt_bfloat16* input, const __mt_bfloat16* bias,
    __mt_bfloat16* output, int rows, int cols) {
  static_assert(ACTIVATION >= 0 && ACTIVATION <= 2, "unknown activation code");
  const int width = cols / 8;
  const int stride = blockDim.x * gridDim.x;
  for (int row = blockIdx.y; row < rows; row += gridDim.y) {
    const int64_t base = static_cast<int64_t>(row) * cols;
    for (int lane = blockIdx.x * blockDim.x + threadIdx.x; lane < width;
         lane += stride) {
      const int64_t column = static_cast<int64_t>(lane) * 8;
      ActivationPacked8 input_raw, bias_raw, output_raw;
      input_raw.bits = *reinterpret_cast<const uint4*>(input + base + column);
      if (HAS_BIAS) {
        bias_raw.bits = *reinterpret_cast<const uint4*>(bias + column);
      }
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        float value = __bfloat162float(input_raw.lanes[i]);
        if (HAS_BIAS) value += __bfloat162float(bias_raw.lanes[i]);
        if (ACTIVATION == 1) value = gelu_tanh(value);
        if (ACTIVATION == 2) value = value / (1.0f + expf(-value));
        output_raw.lanes[i] = __float2bfloat16(value);
      }
      *reinterpret_cast<uint4*>(output + base + column) = output_raw.bits;
    }
  }
}

// Gated GELU: `gelu_tanh(gate) * up`, both halves of the row.
//
// The gate is the first half of the columns and the up projection the second,
// which is the plain layout; the interleaved packings belong to the fused GEMM
// that consumes this, not here.
//
// Arithmetic in f32 with one rounding on the way out, which is what the bridge
// did (`_geglu_bf16`: `_f32(gate_up)`, `_gelu_tanh`, multiply, `_write`). A
// single multiply has no add to contract into, so unlike the euler step there
// is nothing here for the compiler to fuse away.
__global__ void geglu_bf16_kernel(const __mt_bfloat16* gate_up,
                                  __mt_bfloat16* output, int rows, int inner) {
  // As above: the row/column split comes from the grid rather than from a
  // 64-bit divide and remainder per element, which is where six sevenths of
  // this kernel's time went on M1000.
  const int stride = blockDim.x * gridDim.x;
  for (int row = blockIdx.y; row < rows; row += gridDim.y) {
    const __mt_bfloat16* row_input =
        gate_up + static_cast<int64_t>(row) * 2 * inner;
    __mt_bfloat16* row_output = output + static_cast<int64_t>(row) * inner;
    for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < inner;
         col += stride) {
      row_output[col] = __float2bfloat16(
          gelu_tanh(__bfloat162float(row_input[col])) *
          __bfloat162float(row_input[inner + col]));
    }
  }
}

union __align__(8) GegluPacked4 {
  uint64_t bits;
  __mt_bfloat16 lanes[4];
};

// Same FP32 GELU and the same BF16 rounding as the scalar path. Vectorized
// gate/up reads and output writes reduce traffic/CTAs without changing math.
__global__ void geglu_bf16_packed4_kernel(const __mt_bfloat16* input,
                                        __mt_bfloat16* output,
                                        int64_t rows, int64_t hidden) {
  for (int64_t row = blockIdx.y; row < rows; row += gridDim.y) {
    const int64_t base = row * hidden;
    for (int64_t col = 4LL * (blockIdx.x * blockDim.x + threadIdx.x);
         col < hidden; col += 4LL * blockDim.x * gridDim.x) {
      GegluPacked4 gate, up, result;
      gate.bits = *reinterpret_cast<const uint64_t*>(input + 2 * base + col);
      up.bits = *reinterpret_cast<const uint64_t*>(input + 2 * base + hidden + col);
#pragma unroll
      for (int lane = 0; lane < 4; ++lane) {
        result.lanes[lane] = __float2bfloat16(
            gelu_tanh(__bfloat162float(gate.lanes[lane])) *
            __bfloat162float(up.lanes[lane]));
      }
      *reinterpret_cast<uint64_t*>(output + base + col) = result.bits;
    }
  }
}
