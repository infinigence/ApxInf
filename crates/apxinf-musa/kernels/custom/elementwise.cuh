#pragma once

// Derived in part from `crates/apxinf-cuda/kernels/custom/` (Apache-2.0,
// Copyright 2026 apxinf contributors); the differences are explained below.

// Elementwise and row-manipulation kernels, a port of the parts of
// `crates/apxinf-cuda/kernels/custom/elementwise.cuh` this port reaches.

// Row concatenation: `first`'s rows then `second`'s, into one buffer.
//
// A single grid-stride pass over the *total* length, selecting the source by
// index, rather than one launch per input -- the boundary is where the two
// sources meet and needs no special case.
__global__ void concat_rows_bf16_kernel(const __mt_bfloat16* first,
                                        const __mt_bfloat16* second,
                                        __mt_bfloat16* output,
                                        int64_t first_count,
                                        int64_t total_count) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < total_count; index += stride) {
    output[index] =
        index < first_count ? first[index] : second[index - first_count];
  }
}

// Vectorized form of the same pass: eight bf16 (one `uint4`) per lane.
//
// `concat_rows_bf16_kernel` moves two bytes per lane, which is half a 128-byte
// sector per 32-lane warp on M1000 -- a scalar bf16 grid-stride copy is
// sector-limited and measures 30.1 GB/s on `[10, 512] ++ [512, 2048]`. Widening
// the lane to 16 bytes fills the sector and measures 113.1 GB/s for the same
// bytes.
//
// Only reachable when every pointer and the source boundary are 16-byte
// aligned; `first_count % 8 == 0` keeps `second`'s offset a multiple of the
// load width too, and the launcher checks all of it. A group that straddles the
// boundary or the end falls back to the scalar path element by element.
__global__ void concat_rows_bf16_vector_kernel(const __mt_bfloat16* first,
                                               const __mt_bfloat16* second,
                                               __mt_bfloat16* output,
                                               int64_t first_count,
                                               int64_t total_count) {
  const int64_t lanes = (total_count + 7) / 8;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t lane = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       lane < lanes; lane += stride) {
    const int64_t index = lane * 8;
    if (index + 8 <= total_count &&
        (index + 8 <= first_count || index >= first_count)) {
      const __mt_bfloat16* source =
          index >= first_count ? second + (index - first_count) : first + index;
      *reinterpret_cast<uint4*>(output + index) =
          *reinterpret_cast<const uint4*>(source);
    } else {
      for (int within = 0; within < 8; ++within) {
        const int64_t at = index + within;
        if (at < total_count) {
          output[at] = at < first_count ? first[at] : second[at - first_count];
        }
      }
    }
  }
}

// One explicit Euler step of the flow sampler: `state + dt * velocity`.
//
// The multiply-add is written out rather than left to contract into an FMA:
// the bridge this replaces computed `_f32(state) + dt * _f32(velocity)` in
// torch, and torch does not fuse those. Letting the compiler do it would change
// the last bit of every element.
__global__ void euler_update_bf16_kernel(const __mt_bfloat16* state,
                                         const __mt_bfloat16* velocity,
                                         __mt_bfloat16* output, int64_t count,
                                         float dt) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    const float scaled = __fmul_rn(dt, __bfloat162float(velocity[index]));
    output[index] = __float2bfloat16(
        __fadd_rn(__bfloat162float(state[index]), scaled));
  }
}

// Position embedding: add a per-position row to a row-per-token projection,
// plus an optional per-column bias, rounded once.
//
// The position table holds `tokens_per_view` rows and is tiled down the
// projection, which is what makes the row index a modulo rather than a direct
// lookup: the projection covers several views and each one is offset by the
// same set of positions. Rows and columns are derived from a flat index, as in
// the CUDA kernel, so the launch stays grid-stride over any length.
__global__ void add_position_bf16_kernel(const __mt_bfloat16* projection,
                                         const __mt_bfloat16* bias,
                                         const __mt_bfloat16* position,
                                         __mt_bfloat16* output, int rows,
                                         int cols, int tokens_per_view) {
  // The modulo that tiles the position table down the projection is over the
  // *row*, so it is one per block rather than one per element -- the flat form
  // needed `(index / cols) % tokens_per_view` on every element, and a 64-bit
  // division is a software routine on M1000. See `activation.cuh`.
  const int stride = blockDim.x * gridDim.x;
  for (int row = blockIdx.y; row < rows; row += gridDim.y) {
    const int64_t base = static_cast<int64_t>(row) * cols;
    const __mt_bfloat16* position_row =
        position + static_cast<int64_t>(row % tokens_per_view) * cols;
    for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < cols;
         col += stride) {
      float value = __bfloat162float(projection[base + col]) +
                    __bfloat162float(position_row[col]);
      if (bias != nullptr) value += __bfloat162float(bias[col]);
      output[base + col] = __float2bfloat16(value);
    }
  }
}

// Per-channel scale: a BF16 `[rows, cols]` matrix times an f32 `[cols]`
// vector, rounded once. This is the activation half of the W8A8 per-channel
// equalization (`apxinf-model`'s `weights/int8_dynamic.rs`): the weight's
// contraction row `k` was multiplied by `c[k]` at pack time and the activation
// is divided by it here, which leaves the product unchanged and moves the
// residual stream's outlier channels off the activation's per-row amax.
//
// One row per `blockIdx.y`, so the channel index is a plain block offset and
// never a 64-bit modulo (see `add_position_bf16_kernel`); every row-block with
// the same `blockIdx.x` reads the same run of `channel_scale`, which
// broadcasts out of L2.
__global__ void mul_channels_bf16_kernel(const __mt_bfloat16* input,
                                         const float* channel_scale,
                                         __mt_bfloat16* output, int rows,
                                         int cols) {
  const int stride = blockDim.x * gridDim.x;
  for (int row = blockIdx.y; row < rows; row += gridDim.y) {
    const int64_t base = static_cast<int64_t>(row) * cols;
    for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < cols;
         col += stride) {
      output[base + col] = __float2bfloat16(
          __bfloat162float(input[base + col]) * channel_scale[col]);
    }
  }
}
