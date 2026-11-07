#pragma once

// Derived in part from `crates/apxinf-cuda/kernels/custom/` (Apache-2.0,
// Copyright 2026 apxinf contributors); the differences are explained below.

// Quantization and dtype-conversion kernels for the W8A8 path.
//
// A port of the parts of `crates/apxinf-cuda/kernels/custom/quantization.cuh` the
// MUSA W8A8 GEMM consumes, in the place the two backends differ:
//
// * the MUSA `s8mma` tile reads its dequantization scales as **FP16**, not f32,
//   and pads them by 512 entries, so `quantize_rows_bf16_int8_kernel` writes the
//   row scales half-precision and this file zero-fills the pad.
//
// The tile writes the BF16 result itself (the vendored templates carry a local
// BF16 epilogue patch), so the FP16 -> BF16 widening pass the upstream
// FP16-only templates needed has no caller and is gone.
//
// The arithmetic is the CUDA kernel's -- per-row `amax`, scale
// `max(amax / 127, 1e-12)`, the quotient rounded to nearest and clamped to
// `[-128, 127]` -- with one substitution: the rounding is
// `quantize_rows_value`'s one-instruction form rather than `roundf`, which it
// agrees with on every value this file can produce but not by definition.
//
// The row quantizer has two forms, both with that arithmetic and the same
// output: `quantize_rows_bf16_int8_kernel` is the CUDA port, one two-byte
// element per lane and a second read of the row to quantize what the first pass
// measured; `quantize_rows_bf16_int8_lanes_kernel` holds its share of the row in
// registers, so the row is read once and both accesses are wide lanes.
// `adapters/custom_kernels.mu` picks between them per shape.
//
// Measured on M1000 (a probe that timed both forms over these widths and
// checked them byte for byte): the lane form is
// 1.1x-1.5x the scalar form over the widths the encode layers quantize, 1152 to
// 16384, and a wash below those, where the row is a handful of microseconds and
// the launch dominates either form. What is left of the gap is arithmetic rather
// than traffic -- the same lane pattern with its arithmetic removed reaches
// 78.9 GB/s against the device's 96.6 GB/s copy anchor -- and `roundf` was the
// largest single piece of it, 26% of the kernel's time at 16384 columns, where
// the kernel with `roundf` runs 0.6505 ms and with the one-instruction rounding
// 0.5052 ms.

#include <musa_fp16.h>

#include "math.cuh"
#include "reduction.cuh"

// The FP16 zero pad both scale vectors carry past their length. Must match
// `kScalePad` in `adapters/qint8_gemm_adapter.mu` and the reference launcher's
// `scale_pad`.
constexpr int kW8A8ScalePad = 512;

// The quantizer's per-element arithmetic, shared by both forms below so that
// they cannot drift apart: `value / scale`, rounded to nearest, clamped to the
// INT8 range.
//
// The CUDA reference rounds the quotient with `roundf`. This computes the same
// INT8 by a route that costs one instruction where MUSA's `roundf` costs about
// eight: `scaled + copysignf(0.5f, scaled)` is the quotient shifted half a unit
// toward its own sign, and the `static_cast<int8_t>` truncation of the sum is
// therefore the quotient rounded to nearest. The two are not the same function
// -- `roundf` breaks an exact tie away from zero, while the add rounds the *sum*
// to its own ulp first, so a quotient within an ulp below an exact half can
// truncate to the neighbouring odd integer. The smallest such quotient is
// `0.5 - 2**-25`, which no BF16 value and scale in this file's domain reaches:
// a one-off host probe over 41.3 million BF16 pairs finds no divergence, and a
// device run of both forms over 12.6 million elements of the encode shapes
// quantizes them identically. The clamp is applied to the sum, which keeps a
// quotient outside `[-128, 127]` inside it exactly as clamping the rounded
// value does.
__device__ inline int8_t quantize_rows_value(float value, float scale) {
  const float scaled = value / scale;
  return static_cast<int8_t>(
      fminf(127.0f, fmaxf(-128.0f, scaled + copysignf(0.5f, scaled))));
}

// Per-channel equalization: `inv_c` is either nullptr or `cols` f32 factors the
// activation is multiplied by before the amax is measured and the quotient
// taken (the SmoothQuant identity `x @ W^T == (x ./ c) @ (W .* c)^T`, with the
// weight's `.* c` folded in at pack time). The product is rounded back to BF16
// with the same `__float2bfloat16` the elementwise pass this fusion replaces
// used, so the fused kernel is bitwise-identical to pre-scaling the activation
// and quantizing it -- while the factors ride the row's register pass instead
// of a separate 17 MB round trip per call.
__device__ inline __mt_bfloat16 equalize_bf16(__mt_bfloat16 value,
                                              const float* inv_c, int column) {
  if (inv_c == nullptr) return value;
  return __float2bfloat16(__bfloat162float(value) * inv_c[column]);
}

// The lanes form's per-lane factors: `column` is a multiple of 8, so both
// halves are legal 16-byte loads. Only called with a non-null `inv_c`.
__device__ inline void load_inv_c_lane(const float* inv_c, int column,
                                       float factors[8]) {
  const float4 low = *reinterpret_cast<const float4*>(inv_c + column);
  const float4 high = *reinterpret_cast<const float4*>(inv_c + column + 4);
  factors[0] = low.x;
  factors[1] = low.y;
  factors[2] = low.z;
  factors[3] = low.w;
  factors[4] = high.x;
  factors[5] = high.y;
  factors[6] = high.z;
  factors[7] = high.w;
}

// Row-quantize a BF16 `[rows, cols]` matrix to INT8, writing FP16 row scales.
//
// One block per row, as in the CUDA reference: `blockDim.x` must be a multiple of
// 32 and at most 8 * 32 for the reduction's `scratch[8]`. `scales` holds `rows`
// FP16 scales followed by `kW8A8ScalePad` zeros, which the first block writes --
// the tile's guarded scale load can reach into that pad, and it has to read zero
// there.
__global__ void quantize_rows_bf16_int8_kernel(const __mt_bfloat16* input,
                                               int8_t* output, __half* scales,
                                               int rows, int cols,
                                               const float* inv_c) {
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int64_t base = static_cast<int64_t>(row) * cols;
  float maximum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    maximum = fmaxf(maximum,
                    fabsf(__bfloat162float(
                        equalize_bf16(input[base + col], inv_c, col))));
  }
  const float scale =
      fmaxf(block_max_parallel_unsafe(maximum, scratch) / 127.0f, 1.0e-12f);
  if (threadIdx.x == 0) scales[row] = __float2half_rn(scale);
  if (blockIdx.x == 0) {
    for (int pad = threadIdx.x; pad < kW8A8ScalePad; pad += blockDim.x) {
      scales[rows + pad] = __float2half_rn(0.0f);
    }
  }
  // `block_max_parallel_unsafe` leaves its result in `scratch[0]` after a
  // barrier, but does not protect the read of it: every thread reads the value
  // into a register before the next writer. There is no next writer here, so the
  // returned value is used directly.
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    output[base + col] = quantize_rows_value(
        __bfloat162float(equalize_bf16(input[base + col], inv_c, col)), scale);
  }
}

// Eight BF16 in a 16-byte lane, as `activation.cuh`'s packed unions do. MUSA's
// widest load is 16 bytes, so eight elements is one instruction where the
// scalar kernel above takes eight -- and eight two-byte lanes per thread is
// half a 128-byte sector per warp, which is the other half of why that form
// tops out well under the device's copy bandwidth.
union __align__(16) PackedBf16x8 {
  uint4 bits;
  __mt_bfloat16 lanes[8];
};

// Eight INT8 in an 8-byte lane: the quantizer's output width, one store per
// lane group instead of eight.
union __align__(8) PackedI8x8 {
  uint64_t bits;
  int8_t lanes[8];
};

// The most lanes one thread may hold: `kW8A8MaxLanes * 16` bytes of the row in
// registers, which is 128 bytes -- 32 registers of data -- at 8 lanes.
constexpr int kW8A8MaxLanes = 8;

// Row quantize, vectorized, with the row read once.
//
// Same arithmetic and same output as `quantize_rows_bf16_int8_kernel` above;
// the difference is where the row lives between the two halves of the work. The
// amax of a row is not known until the whole row has been seen, so a second
// pass over the row is inherent to per-row quantization -- but not necessarily
// a second *read*: here each thread keeps its `kLanes` sixteen-byte lanes in
// registers while the block reduces the amax, and quantizes out of those.
// Traffic is therefore one 16-byte read and one 8-byte write per eight
// elements, against the scalar form's two two-byte reads and one two-byte
// write, and the instruction count per eight elements drops from roughly
// twenty-four to two.
//
// Requires `cols <= kLanes * blockDim.x * 8`, `cols % 8 == 0` and pointers
// aligned so that every lane is a legal 16-byte load and 8-byte store;
// `adapters/custom_kernels.mu` checks all of that and keeps the scalar form for
// the shapes that fail it. `blockDim.x` obeys the reduction's bounds: a
// multiple of 32 and at most 8 * 32.
template <int kLanes>
__global__ void quantize_rows_bf16_int8_lanes_kernel(const __mt_bfloat16* input,
                                                     int8_t* output,
                                                     __half* scales, int rows,
                                                     int cols,
                                                     const float* inv_c) {
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int64_t base = static_cast<int64_t>(row) * cols;
  PackedBf16x8 cached[kLanes];
  float maximum = 0.0f;
#pragma unroll
  for (int lane = 0; lane < kLanes; ++lane) {
    const int column = (lane * blockDim.x + threadIdx.x) * 8;
    if (column < cols) {
      cached[lane].bits =
          *reinterpret_cast<const uint4*>(input + base + column);
      if (inv_c == nullptr) {
#pragma unroll
        for (int index = 0; index < 8; ++index) {
          maximum = fmaxf(
              maximum, fabsf(__bfloat162float(cached[lane].lanes[index])));
        }
      } else {
        // Equalize in place: the rounded product overwrites the cached lane,
        // so the quantization loop below runs unmodified and the factors are
        // read exactly once.
        float factors[8];
        load_inv_c_lane(inv_c, column, factors);
#pragma unroll
        for (int index = 0; index < 8; ++index) {
          cached[lane].lanes[index] = __float2bfloat16(
              __bfloat162float(cached[lane].lanes[index]) * factors[index]);
          maximum = fmaxf(
              maximum, fabsf(__bfloat162float(cached[lane].lanes[index])));
        }
      }
    }
  }
  const float scale =
      fmaxf(block_max_parallel_unsafe(maximum, scratch) / 127.0f, 1.0e-12f);
  if (threadIdx.x == 0) scales[row] = __float2half_rn(scale);
  if (blockIdx.x == 0) {
    for (int pad = threadIdx.x; pad < kW8A8ScalePad; pad += blockDim.x) {
      scales[rows + pad] = __float2half_rn(0.0f);
    }
  }
  // Same contract as the scalar form: the scale is read out of `scratch[0]`
  // once and every thread works from its register copy of it.
#pragma unroll
  for (int lane = 0; lane < kLanes; ++lane) {
    const int column = (lane * blockDim.x + threadIdx.x) * 8;
    if (column < cols) {
      PackedI8x8 quantized;
#pragma unroll
      for (int index = 0; index < 8; ++index) {
        quantized.lanes[index] = quantize_rows_value(
            __bfloat162float(cached[lane].lanes[index]), scale);
      }
      *reinterpret_cast<uint64_t*>(output + base + column) = quantized.bits;
    }
  }
}

// GeGLU producer: retain the unfused BF16 intermediate rounding before
// equalization and row quantization. The cached lane never leaves registers.
// Launch with 512 or 1024 threads; scratch covers all 32 software warps.
template <int kLanes>
__global__ void geglu_quantize_rows_bf16_int8_lanes_kernel(const __mt_bfloat16* input,
                                                     int8_t* output,
                                                     __half* scales, int rows,
                                                     int cols,
                                                     const float* inv_c) {
  __shared__ float scratch[32];
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int64_t base = static_cast<int64_t>(row) * cols;
  PackedBf16x8 cached[kLanes];
  float maximum = 0.0f;
#pragma unroll
  for (int lane = 0; lane < kLanes; ++lane) {
    const int column = (lane * blockDim.x + threadIdx.x) * 8;
    if (column < cols) {
      PackedBf16x8 gate, up;
      gate.bits = *reinterpret_cast<const uint4*>(input + 2 * base + column);
      up.bits = *reinterpret_cast<const uint4*>(input + 2 * base + cols + column);
#pragma unroll
      for (int index = 0; index < 8; ++index) {
        cached[lane].lanes[index] = __float2bfloat16(
            gelu_tanh(__bfloat162float(gate.lanes[index])) *
            __bfloat162float(up.lanes[index]));
      }
      if (inv_c == nullptr) {
#pragma unroll
        for (int index = 0; index < 8; ++index) {
          maximum = fmaxf(
              maximum, fabsf(__bfloat162float(cached[lane].lanes[index])));
        }
      } else {
        // Equalize in place: the rounded product overwrites the cached lane,
        // so the quantization loop below runs unmodified and the factors are
        // read exactly once.
        float factors[8];
        load_inv_c_lane(inv_c, column, factors);
#pragma unroll
        for (int index = 0; index < 8; ++index) {
          cached[lane].lanes[index] = __float2bfloat16(
              __bfloat162float(cached[lane].lanes[index]) * factors[index]);
          maximum = fmaxf(
              maximum, fabsf(__bfloat162float(cached[lane].lanes[index])));
        }
      }
    }
  }
  const float scale =
      fmaxf(block_max_parallel_unsafe(maximum, scratch) / 127.0f, 1.0e-12f);
  if (threadIdx.x == 0) scales[row] = __float2half_rn(scale);
  if (blockIdx.x == 0) {
    for (int pad = threadIdx.x; pad < kW8A8ScalePad; pad += blockDim.x) {
      scales[rows + pad] = __float2half_rn(0.0f);
    }
  }
  // Same contract as the scalar form: the scale is read out of `scratch[0]`
  // once and every thread works from its register copy of it.
#pragma unroll
  for (int lane = 0; lane < kLanes; ++lane) {
    const int column = (lane * blockDim.x + threadIdx.x) * 8;
    if (column < cols) {
      PackedI8x8 quantized;
#pragma unroll
      for (int index = 0; index < 8; ++index) {
        quantized.lanes[index] = quantize_rows_value(
            __bfloat162float(cached[lane].lanes[index]), scale);
      }
      *reinterpret_cast<uint64_t*>(output + base + column) = quantized.bits;
    }
  }
}
