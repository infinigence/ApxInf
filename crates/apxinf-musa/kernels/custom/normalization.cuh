#pragma once

// Derived in part from `crates/apxinf-cuda/kernels/custom/` (Apache-2.0,
// Copyright 2026 apxinf contributors); the differences are explained below.

#include "reduction.cuh"

// Normalization kernels, a port of the parts of
// `crates/apxinf-cuda/kernels/custom/normalization.cuh` this port reaches.
//
// CUDA's file also carries the quantized FP8 outputs and a register-cached
// LayerNorm for widths that divide the block; those arrive with their operators
// rather than in advance, which is also what keeps this file free of the
// aggregate-header shape the CUDA crate's README forbids.

// RMSNorm: one block per row, block-wide sum of squares, then the weight.
//
// **Deliberately not bit-identical to the bridge.** The port started with CUDA's
// shape dispatch here: a single-warp `rms_vector_square_mean_bf16` for
// `rows >= 16 && cols > 128 && cols % 4 == 0`, a block reduction otherwise.
// That reproduced the reference's bits on `[522, 2048]`, at a price measured
// with the port's rms-variant sweep: the reduction was **89% of the
// kernel** (0.686 ms of 0.773), and the same reduction over all 256 threads
// instead of 32 ran it in 0.108 ms -- **6.4x**. On `[522, 2048]` that is
// 10.5 ms of a pass.
//
// The two branches summed in different orders, which is the only reason the
// dispatch existed; the reference is a `mean`, not a hand-rolled tree, so
// neither order is *the* reference order. Collapsing to one block reduction
// costs the bit-identity the ledger recorded for #1 and moves the statistic by
// well under a bf16 ulp -- the same trade `layer_norm_bf16_kernel` and
// `bias_residual_rms_norm_bf16_kernel` already make, and both remain in the
// gate. It is recorded in the port's latency-reduction notes rather than here.
__global__ void rms_norm_bf16_kernel(const __mt_bfloat16* input,
                                     const __mt_bfloat16* weight,
                                     __mt_bfloat16* output, int rows, int cols,
                                     float eps) {
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  const __mt_bfloat16* row_input =
      input + static_cast<int64_t>(row) * cols;
  float square_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float value = __bfloat162float(row_input[col]);
    square_sum += value * value;
  }
  const float mean = block_sum_parallel_unsafe(square_sum, scratch) / cols;
  const float inverse_rms = rsqrtf(__fadd_rn(mean, eps));
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    output[index] = __float2bfloat16(__bfloat162float(input[index]) *
                                     inverse_rms *
                                     __bfloat162float(weight[col]));
  }
}

// The load width a register-cached RMSNorm uses for a given lanes-per-thread.
// A `std::conditional_t` would pull in `<type_traits>`; this file stays free of
// host headers.
template <int VEC>
struct RmsPack;
template <>
struct RmsPack<4> {
  using Type = uint64_t;
};
template <>
struct RmsPack<8> {
  using Type = uint4;
};

// Vectorized, register-cached RMSNorm: one block per row, each thread loads its
// `VEC`-wide column group once, holds it in registers across the block
// reduction, and writes from those registers -- so the row is read once instead
// of twice. Requires `cols == blockDim.x * PER`, which is what lets the whole
// row live in `PER` registers per thread.
//
// This is the "register-cached LayerNorm for widths that divide the block" the
// header note above says CUDA carries and this port had not brought over. It
// reads the row once and writes it once, against the two reads and one write of
// `rms_norm_bf16_kernel`, and on `[522, 2048]` that is the whole of the gain.
//
// The per-thread summation order is grouped-by-thread rather than strided, so
// like the single block reduction above it is not bit-identical to the earlier
// kernel; it stays inside the same per-tensor gate for the same reason.
template <int VEC, int PER>
__global__ void rms_norm_bf16_vector_kernel(const __mt_bfloat16* input,
                                            const __mt_bfloat16* weight,
                                            __mt_bfloat16* output, int rows,
                                            int cols, float eps) {
  static_assert(PER % VEC == 0, "PER must be a multiple of VEC");
  static_assert(VEC == 4 || VEC == 8, "VEC must be 4 or 8 bf16 lanes");
  using LoadT = typename RmsPack<VEC>::Type;
  constexpr int kChunks = PER / VEC;
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  const int64_t base = static_cast<int64_t>(row) * cols;
  float values[PER];
  float square_sum = 0.0f;
#pragma unroll
  for (int chunk = 0; chunk < kChunks; ++chunk) {
    const int64_t column =
        (static_cast<int64_t>(chunk) * blockDim.x + threadIdx.x) * VEC;
    const LoadT raw = *reinterpret_cast<const LoadT*>(input + base + column);
    const __mt_bfloat16* packed = reinterpret_cast<const __mt_bfloat16*>(&raw);
#pragma unroll
    for (int lane = 0; lane < VEC; ++lane) {
      const float value = __bfloat162float(packed[lane]);
      values[chunk * VEC + lane] = value;
      square_sum += value * value;
    }
  }
  const float mean = block_sum_parallel_unsafe(square_sum, scratch) / cols;
  const float inverse_rms = rsqrtf(__fadd_rn(mean, eps));
#pragma unroll
  for (int chunk = 0; chunk < kChunks; ++chunk) {
    const int64_t column =
        (static_cast<int64_t>(chunk) * blockDim.x + threadIdx.x) * VEC;
    const LoadT weight_raw =
        *reinterpret_cast<const LoadT*>(weight + column);
    const __mt_bfloat16* packed_weight =
        reinterpret_cast<const __mt_bfloat16*>(&weight_raw);
    LoadT out_raw;
    __mt_bfloat16* packed_out = reinterpret_cast<__mt_bfloat16*>(&out_raw);
#pragma unroll
    for (int lane = 0; lane < VEC; ++lane) {
      packed_out[lane] = __float2bfloat16(values[chunk * VEC + lane] *
                                          inverse_rms *
                                          __bfloat162float(packed_weight[lane]));
    }
    *reinterpret_cast<LoadT*>(output + base + column) = out_raw;
  }
}

// adaLN RMSNorm: scale and shift from the first two segments of `style`.
//
// `style` is `[3 * cols]` and only `[0, 2 * cols)` is read -- the third segment
// belongs to the gated variant. That is the model layer's layout, not this
// kernel's choice.
__global__ void ada_rms_norm_bf16_kernel(const __mt_bfloat16* input,
                                         const __mt_bfloat16* style,
                                         __mt_bfloat16* output, int rows,
                                         int cols, float eps) {
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  float square_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float value =
        __bfloat162float(input[static_cast<int64_t>(row) * cols + col]);
    square_sum += value * value;
  }
  const float inverse_rms =
      rsqrtf(block_sum_parallel_unsafe(square_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    const float normalized = __bfloat162float(input[index]) * inverse_rms;
    output[index] = __float2bfloat16(
        normalized * (1.0f + __bfloat162float(style[col])) +
        __bfloat162float(style[cols + col]));
  }
}

// Vectorized, register-cached form of the adaLN RMSNorm above, for widths that
// divide the block exactly (`cols == blockDim.x * VEC`).
//
// This is `[10, 1024]` -- the action expert's own shape, one block per row, so
// ten blocks on an eight-SM device with no occupancy to hide latency behind.
// Each thread loads its `VEC` input columns and both `style` segments in one
// `RmsPack<VEC>::Type` transaction, holds the row in registers across the block
// reduction, and writes the output from those registers: the row is read once
// instead of twice, the same trade
// `ada_gate_residual_rms_norm_bf16_vector_kernel` makes for its sibling.
//
// The per-thread summation is grouped `VEC` columns per thread instead of the
// scalar kernel's `cols / blockDim.x` strided ones, so the accumulation order
// differs slightly; this operator is already gated rather than bit-identical to
// the reference (see the `# Numerics` note on `adaptive_rms_bf16`), and the
// reorder stays inside that gate.
template <int VEC>
__global__ void ada_rms_norm_bf16_vector_kernel(const __mt_bfloat16* input,
                                                const __mt_bfloat16* style,
                                                __mt_bfloat16* output, int rows,
                                                int cols, float eps) {
  static_assert(VEC == 4 || VEC == 8, "VEC must be 4 or 8 bf16 lanes");
  using LoadT = typename RmsPack<VEC>::Type;
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  const int column = threadIdx.x * VEC;
  const int64_t index = static_cast<int64_t>(row) * cols + column;
  const LoadT input_raw = *reinterpret_cast<const LoadT*>(input + index);
  const __mt_bfloat16* packed_input =
      reinterpret_cast<const __mt_bfloat16*>(&input_raw);
  float values[VEC];
  float square_sum = 0.0f;
#pragma unroll
  for (int lane = 0; lane < VEC; ++lane) {
    values[lane] = __bfloat162float(packed_input[lane]);
    square_sum += values[lane] * values[lane];
  }
  const float inverse_rms =
      rsqrtf(block_sum_parallel_unsafe(square_sum, scratch) / cols + eps);
  // The norm's scale and shift are the first two `style` segments, as in the
  // scalar kernel.
  const LoadT scale_raw = *reinterpret_cast<const LoadT*>(style + column);
  const LoadT shift_raw =
      *reinterpret_cast<const LoadT*>(style + cols + column);
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
  *reinterpret_cast<LoadT*>(output + index) = out_raw;
}

// LayerNorm: one block per row, mean and variance over the row, then the
// affine transform.
//
// Three passes over the row and **no cached specialization**. CUDA's file has a
// `cached<2|4|8>` variant that keeps the row in registers and short-circuits
// the reloads, selected for `threads * per_thread == cols` with
// `cols ∈ {512, 1024, 2048}`; PI0.5's `norm.layer_bf16` is `cols = 1152`, which
// selects none of them. A port that skipped the branch because 1152 "does not
// fit anyway" would be right about this model and wrong about the operator's
// contract -- so the branch is not here at all, and adding it later means
// adding a variant whose accumulation order matches CUDA's.
//
// The `__syncthreads()` before the second reduction is CUDA's, and it is
// load-bearing: `block_sum_parallel_unsafe`'s trailing barrier publishes
// `scratch[0]` but does not protect the reads of it, so a fast thread could
// overwrite the mean while a slow one is still reading it.
__global__ void layer_norm_bf16_kernel(const __mt_bfloat16* input,
                                       const __mt_bfloat16* weight,
                                       const __mt_bfloat16* bias,
                                       __mt_bfloat16* output, int rows, int cols,
                                       float eps) {
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  float sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x)
    sum += __bfloat162float(input[static_cast<int64_t>(row) * cols + col]);
  const float mean = block_sum_parallel_unsafe(sum, scratch) / cols;
  float variance_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float centered =
        __bfloat162float(input[static_cast<int64_t>(row) * cols + col]) - mean;
    variance_sum += centered * centered;
  }
  // Finish reading the previous reduction before reusing scratch.
  __syncthreads();
  const float inverse_std =
      rsqrtf(block_sum_parallel_unsafe(variance_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    const float value =
        (__bfloat162float(input[index]) - mean) * inverse_std *
            __bfloat162float(weight[col]) +
        __bfloat162float(bias[col]);
    output[index] = __float2bfloat16(value);
  }
}

// Register-cached, barrier-free warp-per-row LayerNorm.
//
// A whole row belongs to one warp: each lane walks `NITER` column groups of
// `VEC` elements, holds the row in registers across the statistics, and writes
// the affine result from those registers, so the row is read once and written
// once. The mean and variance come from a single `__shfl_xor_sync` reduction of
// (sum, sum of squares) -- **no shared memory and no `__syncthreads()`**, which
// is what the one-block-per-row shapes pay twice. Requires
// `cols == 32 * VEC * NITER`, which is what keeps every access in range without
// a predicate.
//
// The variance is `E[x^2] - mean^2` rather than the sum of squared deviations
// the three-pass kernel uses: one fused reduction instead of two. On
// `[512, 1152]` this measures 0.0444 ms against the three-pass 0.0832, and the
// two forms agree to 2.4e-4 on the per-tensor gate.
template <int VEC, int NITER, int WARPS>
__global__ void layer_norm_bf16_warp_kernel(const __mt_bfloat16* input,
                                            const __mt_bfloat16* weight,
                                            const __mt_bfloat16* bias,
                                            __mt_bfloat16* output, int rows,
                                            int cols, float eps) {
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
    const LoadT raw = *reinterpret_cast<const LoadT*>(input + base + column);
    const __mt_bfloat16* packed = reinterpret_cast<const __mt_bfloat16*>(&raw);
#pragma unroll
    for (int lane_in = 0; lane_in < VEC; ++lane_in) {
      const float value = __bfloat162float(packed[lane_in]);
      values[chunk * VEC + lane_in] = value;
      sum += value;
      square_sum += value * value;
    }
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
    const LoadT weight_raw = *reinterpret_cast<const LoadT*>(weight + column);
    const LoadT bias_raw = *reinterpret_cast<const LoadT*>(bias + column);
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
    *reinterpret_cast<LoadT*>(output + base + column) = out_raw;
  }
}
