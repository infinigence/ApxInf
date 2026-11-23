#pragma once

// Derived in part from `crates/apxinf-cuda/kernels/custom/` (Apache-2.0,
// Copyright 2026 apxinf contributors); the differences are explained below.

#include "normalization.cuh"
#include "reduction.cuh"

// Fused operators, a port of the parts of
// `crates/apxinf-cuda/kernels/custom/fused.cuh` this port reaches. The CUDA
// file is much larger; these are added as their operators move off the bridge,
// not in advance.

// Gated residual followed by an adaLN RMSNorm, one block per row.
//
// The order matters and is the whole reason this is fused rather than composed:
// `hidden` is written **rounded to bf16** in the first pass and read back in the
// second, so the rounding is part of the arithmetic rather than an artifact of
// storage. `normalized` is computed from the rounded value.
//
// `gate_style` and `norm_style` are both `[3 * cols]` and only partly used:
// the gate is the third segment, and the norm's scale and shift are the first
// two. That is the model layer's layout, not this kernel's choice.
__global__ void ada_gate_residual_rms_norm_bf16_kernel(
    const __mt_bfloat16* projection, const __mt_bfloat16* residual,
    const __mt_bfloat16* gate_style, const __mt_bfloat16* norm_style,
    __mt_bfloat16* hidden, __mt_bfloat16* normalized, int rows, int cols,
    float eps) {
  // 8 floats for a 256-thread block's 8 software warps.
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  float square_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    const __mt_bfloat16 rounded = __float2bfloat16(
        __bfloat162float(residual[index]) +
        __bfloat162float(projection[index]) *
            __bfloat162float(gate_style[2 * cols + col]));
    hidden[index] = rounded;
    const float value = __bfloat162float(rounded);
    square_sum += value * value;
  }
  const float inverse_rms =
      rsqrtf(block_sum_parallel_unsafe(square_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    const float value = __bfloat162float(hidden[index]) * inverse_rms;
    normalized[index] = __float2bfloat16(
        value * (1.0f + __bfloat162float(norm_style[col])) +
        __bfloat162float(norm_style[cols + col]));
  }
}

// Vectorized, register-cached form of the gated residual + adaLN RMSNorm above,
// for widths that divide the block exactly (`cols == blockDim.x * VEC`).
//
// Two things change against the scalar kernel, and both are about the shape the
// action expert actually reaches -- `[10, 1024]`, one block per row, so only ten
// blocks on an eight-SM device and no occupancy to hide latency with:
//
//   * each thread loads its `VEC` gate/projection/residual columns in one
//     `RmsPack<VEC>::Type` transaction instead of `cols / blockDim.x` scalar
//     2-byte loads, and loads the two `norm_style` segments the same way;
//   * the rounded `hidden` value the row was normalized from is held in
//     registers across the block reduction, so `hidden` is written once and
//     never read back -- the scalar kernel re-reads the whole row for the second
//     pass, a full extra body per call.
//
// The reduction and the arithmetic are unchanged, so this is bit-identical to
// the scalar form.
template <int VEC>
__global__ void ada_gate_residual_rms_norm_bf16_vector_kernel(
    const __mt_bfloat16* projection, const __mt_bfloat16* residual,
    const __mt_bfloat16* gate_style, const __mt_bfloat16* norm_style,
    __mt_bfloat16* hidden, __mt_bfloat16* normalized, int rows, int cols,
    float eps) {
  static_assert(VEC == 4 || VEC == 8, "VEC must be 4 or 8 bf16 lanes");
  using LoadT = typename RmsPack<VEC>::Type;
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  const int column = threadIdx.x * VEC;
  const int64_t index = static_cast<int64_t>(row) * cols + column;
  const LoadT projection_raw =
      *reinterpret_cast<const LoadT*>(projection + index);
  const LoadT residual_raw = *reinterpret_cast<const LoadT*>(residual + index);
  // The gate is the third `gate_style` segment, as in the scalar kernel.
  const LoadT gate_raw =
      *reinterpret_cast<const LoadT*>(gate_style + 2 * cols + column);
  const __mt_bfloat16* packed_projection =
      reinterpret_cast<const __mt_bfloat16*>(&projection_raw);
  const __mt_bfloat16* packed_residual =
      reinterpret_cast<const __mt_bfloat16*>(&residual_raw);
  const __mt_bfloat16* packed_gate =
      reinterpret_cast<const __mt_bfloat16*>(&gate_raw);
  float values[VEC];
  float square_sum = 0.0f;
  LoadT hidden_raw;
  __mt_bfloat16* packed_hidden = reinterpret_cast<__mt_bfloat16*>(&hidden_raw);
#pragma unroll
  for (int lane = 0; lane < VEC; ++lane) {
    const __mt_bfloat16 rounded = __float2bfloat16(
        __bfloat162float(packed_residual[lane]) +
        __bfloat162float(packed_projection[lane]) *
            __bfloat162float(packed_gate[lane]));
    packed_hidden[lane] = rounded;
    values[lane] = __bfloat162float(rounded);
    square_sum += values[lane] * values[lane];
  }
  *reinterpret_cast<LoadT*>(hidden + index) = hidden_raw;
  const float inverse_rms =
      rsqrtf(block_sum_parallel_unsafe(square_sum, scratch) / cols + eps);
  // The norm's scale and shift are the first two `norm_style` segments.
  const LoadT scale_raw =
      *reinterpret_cast<const LoadT*>(norm_style + column);
  const LoadT shift_raw =
      *reinterpret_cast<const LoadT*>(norm_style + cols + column);
  const __mt_bfloat16* packed_scale =
      reinterpret_cast<const __mt_bfloat16*>(&scale_raw);
  const __mt_bfloat16* packed_shift =
      reinterpret_cast<const __mt_bfloat16*>(&shift_raw);
  LoadT out_raw;
  __mt_bfloat16* packed_out = reinterpret_cast<__mt_bfloat16*>(&out_raw);
#pragma unroll
  for (int lane = 0; lane < VEC; ++lane) {
    packed_out[lane] = __float2bfloat16(
        values[lane] * inverse_rms *
            (1.0f + __bfloat162float(packed_scale[lane])) +
        __bfloat162float(packed_shift[lane]));
  }
  *reinterpret_cast<LoadT*>(normalized + index) = out_raw;
}

// Projection plus an optional bias, added into the residual, rounded once.
//
// Grid-stride rather than one block per row: this operator has no reduction, so
// the row structure only matters for broadcasting the bias.
__global__ void bias_residual_bf16_kernel(const __mt_bfloat16* projection,
                                          const __mt_bfloat16* bias,
                                          const __mt_bfloat16* residual,
                                          __mt_bfloat16* output, int rows,
                                          int cols) {
  // Rows on `gridDim.y`, column chunks on `gridDim.x`: the column is an offset
  // from the block's base, not a remainder. See `activation.cuh` for why the
  // flat form's `index % cols` was worth this.
  const int stride = blockDim.x * gridDim.x;
  for (int row = blockIdx.y; row < rows; row += gridDim.y) {
    const int64_t base = static_cast<int64_t>(row) * cols;
    for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < cols;
         col += stride) {
      float value = __bfloat162float(projection[base + col]) +
                    __bfloat162float(residual[base + col]);
      if (bias != nullptr) value += __bfloat162float(bias[col]);
      output[base + col] = __float2bfloat16(value);
    }
  }
}

// Vectorized form of the above: eight columns per thread in one 16-byte load
// per operand and one 16-byte store, which is where the scalar form's ceiling
// is. `[512, 1152]` -- the vision tower's fc2 residual -- measures 0.0518 ms
// (68.3 GB/s) scalar; the width is `8 * 144`, so the same shape takes the
// eight-lane branch here.
//
// `HAS_BIAS` is a compile-time parameter for the reason the register-cached
// kernels below record, and the arithmetic is the scalar kernel's unchanged, so
// the two are bit-identical. Requires `cols % 8 == 0`, which the launcher
// checks along with the pointer alignment.
template <bool HAS_BIAS>
__global__ void bias_residual_bf16_vector8_kernel(
    const __mt_bfloat16* projection, const __mt_bfloat16* bias,
    const __mt_bfloat16* residual, __mt_bfloat16* output, int rows, int cols) {
  const int width = cols / 8;
  const int stride = blockDim.x * gridDim.x;
  for (int row = blockIdx.y; row < rows; row += gridDim.y) {
    const int64_t base = static_cast<int64_t>(row) * cols;
    for (int lane = blockIdx.x * blockDim.x + threadIdx.x; lane < width;
         lane += stride) {
      const int64_t column = static_cast<int64_t>(lane) * 8;
      const uint4 projection_raw =
          *reinterpret_cast<const uint4*>(projection + base + column);
      const uint4 residual_raw =
          *reinterpret_cast<const uint4*>(residual + base + column);
      uint4 bias_raw;
      if (HAS_BIAS) bias_raw = *reinterpret_cast<const uint4*>(bias + column);
      const __mt_bfloat16* packed_projection =
          reinterpret_cast<const __mt_bfloat16*>(&projection_raw);
      const __mt_bfloat16* packed_residual =
          reinterpret_cast<const __mt_bfloat16*>(&residual_raw);
      const __mt_bfloat16* packed_bias =
          reinterpret_cast<const __mt_bfloat16*>(&bias_raw);
      uint4 output_raw;
      __mt_bfloat16* packed_output = reinterpret_cast<__mt_bfloat16*>(&output_raw);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        float value = __bfloat162float(packed_projection[i]) +
                      __bfloat162float(packed_residual[i]);
        if (HAS_BIAS) value += __bfloat162float(packed_bias[i]);
        packed_output[i] = __float2bfloat16(value);
      }
      *reinterpret_cast<uint4*>(output + base + column) = output_raw;
    }
  }
}

// The same, followed by an RMSNorm with a weight and no shift.
//
// `hidden` holds the bias+residual sum **rounded to bf16**, and the norm reads
// it back rather than keeping the f32 value in a register -- the rounding is
// part of the arithmetic and the bridge does the same thing explicitly.
__global__ void bias_residual_rms_norm_bf16_kernel(
    const __mt_bfloat16* projection, const __mt_bfloat16* bias,
    const __mt_bfloat16* residual, const __mt_bfloat16* weight,
    __mt_bfloat16* hidden, __mt_bfloat16* normalized, int rows, int cols,
    float eps) {
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  float square_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    float value = __bfloat162float(projection[index]) +
                  __bfloat162float(residual[index]);
    if (bias != nullptr) value += __bfloat162float(bias[col]);
    const __mt_bfloat16 rounded = __float2bfloat16(value);
    hidden[index] = rounded;
    value = __bfloat162float(rounded);
    square_sum += value * value;
  }
  const float inverse_rms =
      rsqrtf(block_sum_parallel_unsafe(square_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    normalized[index] = __float2bfloat16(
        __bfloat162float(hidden[index]) * inverse_rms *
        __bfloat162float(weight[col]));
  }
}

// Projection plus an optional bias, added into the residual, then a LayerNorm
// over the result -- both the sum and the normalized row come back.
//
// `hidden` is written **rounded to bf16** and read back for the mean and
// variance, so that rounding is part of the arithmetic. The eager bridge this
// replaces is close to that but not the same: `_bias_residual_layer_bf16`
// writes `hidden` (rounding it) and then takes the statistics from the
// *unrounded* f32 value it still holds. CUDA's kernel reads the rounded row
// back, which is what the reference model does -- it adds two bf16 tensors and
// normalizes the result -- so this port follows CUDA rather than the bridge.
// The difference is one bf16 rounding inside the statistics, measured at
// ~1.4 bf16 ulp on `fused.bias_residual_rms_bf16`'s identical structure
// (ledger #12), and gated rather than expected to be bit-identical.
//
// Pass order and the barrier are CUDA's; see `layer_norm_bf16_kernel` for why
// the second `__syncthreads()` cannot be dropped.
__global__ void bias_residual_layer_norm_bf16_kernel(
    const __mt_bfloat16* projection, const __mt_bfloat16* projection_bias,
    const __mt_bfloat16* residual, const __mt_bfloat16* norm_weight,
    const __mt_bfloat16* norm_bias, __mt_bfloat16* hidden,
    __mt_bfloat16* normalized, int rows, int cols, float eps) {
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  float sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    float value = __bfloat162float(projection[index]) +
                  __bfloat162float(residual[index]);
    if (projection_bias != nullptr) value += __bfloat162float(projection_bias[col]);
    const __mt_bfloat16 rounded = __float2bfloat16(value);
    hidden[index] = rounded;
    sum += __bfloat162float(rounded);
  }
  const float mean = block_sum_parallel_unsafe(sum, scratch) / cols;
  float variance_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float centered =
        __bfloat162float(hidden[static_cast<int64_t>(row) * cols + col]) - mean;
    variance_sum += centered * centered;
  }
  // Finish reading the previous reduction before reusing scratch.
  __syncthreads();
  const float inverse_std =
      rsqrtf(block_sum_parallel_unsafe(variance_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    const float value =
        (__bfloat162float(hidden[index]) - mean) * inverse_std *
            __bfloat162float(norm_weight[col]) +
        __bfloat162float(norm_bias[col]);
    normalized[index] = __float2bfloat16(value);
  }
}

// Register-cached, barrier-free warp-per-row form of the above.
//
// The same shape as `layer_norm_bf16_warp_kernel`, extended with the projection
// bias: the warp caches the rounded `projection + residual (+ bias)` in
// registers, publishes it to `hidden`, and takes its statistics and its affine
// transform from those same registers -- one read of each input, one write to
// each output, no shared memory and no `__syncthreads()`. Requires
// `cols == 32 * VEC * NITER`.
//
// `projection_bias` is nullable, as in the three-pass kernel, and `HAS_BIAS`
// is a **compile-time** parameter rather than a runtime null test: a branch on
// the pointer inside the unrolled load loop costs more than the load itself
// (0.147 ms against 0.067 on `[512, 1152]`), so the two cases are separate
// instantiations and the launcher picks by pointer. With the bias present this
// measures 0.0737 ms against the three-pass 0.1213, and like the LayerNorm it
// takes `E[x^2] - mean^2` rather than the sum of squared deviations -- one
// fused reduction, agreeing with the three-pass form well inside the
// per-tensor gate.
template <int VEC, int NITER, int WARPS, bool HAS_BIAS>
__global__ void bias_residual_layer_norm_bf16_warp_kernel(
    const __mt_bfloat16* projection, const __mt_bfloat16* projection_bias,
    const __mt_bfloat16* residual, const __mt_bfloat16* norm_weight,
    const __mt_bfloat16* norm_bias, __mt_bfloat16* hidden,
    __mt_bfloat16* normalized, int rows, int cols, float eps) {
  static_assert(VEC == 4 || VEC == 8, "VEC must be 4 or 8 bf16 lanes");
  using LoadT = typename RmsPack<VEC>::Type;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int row = blockIdx.x * WARPS + warp;
  if (row >= rows) return;
  const int64_t base = static_cast<int64_t>(row) * cols;
  float values[NITER * VEC];
  float sum = 0.0f;
  float square_sum = 0.0f;
#pragma unroll
  for (int chunk = 0; chunk < NITER; ++chunk) {
    const int64_t column = static_cast<int64_t>(chunk * 32 + lane) * VEC;
    const LoadT projection_raw =
        *reinterpret_cast<const LoadT*>(projection + base + column);
    const LoadT residual_raw =
        *reinterpret_cast<const LoadT*>(residual + base + column);
    const __mt_bfloat16* packed_projection =
        reinterpret_cast<const __mt_bfloat16*>(&projection_raw);
    const __mt_bfloat16* packed_residual =
        reinterpret_cast<const __mt_bfloat16*>(&residual_raw);
    LoadT bias_raw;
    const __mt_bfloat16* packed_bias = nullptr;
    if (HAS_BIAS) {
      bias_raw = *reinterpret_cast<const LoadT*>(projection_bias + column);
      packed_bias = reinterpret_cast<const __mt_bfloat16*>(&bias_raw);
    }
    LoadT hidden_raw;
    __mt_bfloat16* packed_hidden = reinterpret_cast<__mt_bfloat16*>(&hidden_raw);
#pragma unroll
    for (int lane_in = 0; lane_in < VEC; ++lane_in) {
      float value = __bfloat162float(packed_projection[lane_in]) +
                    __bfloat162float(packed_residual[lane_in]);
      if (HAS_BIAS) value += __bfloat162float(packed_bias[lane_in]);
      packed_hidden[lane_in] = __float2bfloat16(value);
      const float rounded = __bfloat162float(packed_hidden[lane_in]);
      values[chunk * VEC + lane_in] = rounded;
      sum += rounded;
      square_sum += rounded * rounded;
    }
    *reinterpret_cast<LoadT*>(hidden + base + column) = hidden_raw;
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_xor_sync(0xffffffffu, sum, offset);
    square_sum += __shfl_xor_sync(0xffffffffu, square_sum, offset);
  }
  const float mean = sum / cols;
  const float inverse_std = rsqrtf(square_sum / cols - mean * mean + eps);
#pragma unroll
  for (int chunk = 0; chunk < NITER; ++chunk) {
    const int64_t column = static_cast<int64_t>(chunk * 32 + lane) * VEC;
    const LoadT weight_raw = *reinterpret_cast<const LoadT*>(norm_weight + column);
    const LoadT bias_raw = *reinterpret_cast<const LoadT*>(norm_bias + column);
    const __mt_bfloat16* packed_weight =
        reinterpret_cast<const __mt_bfloat16*>(&weight_raw);
    const __mt_bfloat16* packed_bias =
        reinterpret_cast<const __mt_bfloat16*>(&bias_raw);
    LoadT out_raw;
    __mt_bfloat16* packed_out = reinterpret_cast<__mt_bfloat16*>(&out_raw);
#pragma unroll
    for (int lane_in = 0; lane_in < VEC; ++lane_in) {
      packed_out[lane_in] = __float2bfloat16(
          (values[chunk * VEC + lane_in] - mean) * inverse_std *
              __bfloat162float(packed_weight[lane_in]) +
          __bfloat162float(packed_bias[lane_in]));
    }
    *reinterpret_cast<LoadT*>(normalized + base + column) = out_raw;
  }
}

// Vectorized, register-cached bias + residual + RMSNorm: one block per row,
// each thread loads its eight columns once, holds them across the block
// reduction, and writes both `hidden` and `normalized` from those registers, so
// the row is read once instead of twice. Requires `cols == blockDim.x * 8`,
// which is what lets the whole row live in eight registers per thread.
//
// The one-block-per-row reduction cost is the same as
// `bias_residual_rms_norm_bf16_kernel`'s, so this is its larger gain of the two
// register-cached shapes here, and it is bit-identical because the reduction
// order is unchanged.
//
// `HAS_BIAS` is a **compile-time** parameter for the same reason it is on
// `bias_residual_layer_norm_bf16_warp_kernel` below: a runtime `bias != nullptr`
// test inside the unrolled load loop costs more than the load itself. Measured
// with the null test, `[522, 2048]` runs at 0.2344 ms (36.5 GB/s) against the
// sibling `rms_norm_bf16_vector_kernel`'s 0.0477 -- 4.9x for twice the traffic.
// Two separate instantiations remove it; the launcher picks by the pointer, as
// the warp-layer kernel's does.
template <bool HAS_BIAS>
__global__ void bias_residual_rms_norm_bf16_vector8_kernel(
    const __mt_bfloat16* projection, const __mt_bfloat16* bias,
    const __mt_bfloat16* residual, const __mt_bfloat16* weight,
    __mt_bfloat16* hidden, __mt_bfloat16* normalized, int rows, int cols,
    float eps) {
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  const int64_t column = static_cast<int64_t>(threadIdx.x) * 8;
  const int64_t index = static_cast<int64_t>(row) * cols + column;
  const uint4 projection_raw = *reinterpret_cast<const uint4*>(projection + index);
  const uint4 residual_raw = *reinterpret_cast<const uint4*>(residual + index);
  const __mt_bfloat16* packed_projection =
      reinterpret_cast<const __mt_bfloat16*>(&projection_raw);
  const __mt_bfloat16* packed_residual =
      reinterpret_cast<const __mt_bfloat16*>(&residual_raw);
  uint4 bias_raw;
  const __mt_bfloat16* packed_bias = nullptr;
  if (HAS_BIAS) {
    bias_raw = *reinterpret_cast<const uint4*>(bias + column);
    packed_bias = reinterpret_cast<const __mt_bfloat16*>(&bias_raw);
  }
  float values[8];
  float square_sum = 0.0f;
  uint4 hidden_raw;
  __mt_bfloat16* packed_hidden = reinterpret_cast<__mt_bfloat16*>(&hidden_raw);
#pragma unroll
  for (int lane_in = 0; lane_in < 8; ++lane_in) {
    float value = __bfloat162float(packed_projection[lane_in]) +
                  __bfloat162float(packed_residual[lane_in]);
    if (HAS_BIAS) value += __bfloat162float(packed_bias[lane_in]);
    packed_hidden[lane_in] = __float2bfloat16(value);
    values[lane_in] = __bfloat162float(packed_hidden[lane_in]);
    square_sum += values[lane_in] * values[lane_in];
  }
  *reinterpret_cast<uint4*>(hidden + index) = hidden_raw;
  const float inverse_rms =
      rsqrtf(block_sum_parallel_unsafe(square_sum, scratch) / cols + eps);
  const uint4 weight_raw = *reinterpret_cast<const uint4*>(weight + column);
  const __mt_bfloat16* packed_weight =
      reinterpret_cast<const __mt_bfloat16*>(&weight_raw);
  uint4 out_raw;
  __mt_bfloat16* packed_out = reinterpret_cast<__mt_bfloat16*>(&out_raw);
#pragma unroll
  for (int lane_in = 0; lane_in < 8; ++lane_in) {
    packed_out[lane_in] = __float2bfloat16(
        values[lane_in] * inverse_rms * __bfloat162float(packed_weight[lane_in]));
  }
  *reinterpret_cast<uint4*>(normalized + index) = out_raw;
}
