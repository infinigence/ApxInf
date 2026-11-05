// Copyright 2026 ApxInf contributors.
// Gated RMSNorm kernels moved mechanically from the historical aggregate.
#pragma once

// Triton's sigmoid lowering used by FLA gated RMSNorm: exp(x) becomes
// ex2.approx(x*log2(e)) and the reciprocal uses div.full.f32.
__device__ __forceinline__ float la_triton_sigmoid(float x) {
  const float exponent = __fmul_rn(-x, 1.4426950408889634f);
  float power;
  asm("ex2.approx.f32 %0, %1;" : "=f"(power) : "f"(exponent));
  const float denominator = __fadd_rn(1.0f, power);
  float result;
  asm("div.full.f32 %0, %1, %2;" : "=f"(result) : "f"(1.0f), "f"(denominator));
  return result;
}

// Fused gated RMSNorm: FP32 normalization, weight and SiLU gate; one BF16 store.
// x rows are [rows, cols]; z rows are strided slices z[(row/z_heads) *
// z_row_stride + z_col_offset + (row%z_heads)*cols].
// One element per thread at the shipped shape (cols 128, blockDim 128), which
// leaves each thread with a single load in flight; ncu attributes 55% of the
// stalls to long_scoreboard at 94% occupancy, so there are no more warps to
// hide that with. Two things follow.
//
// The shared staging array is not needed: `la_gated[i]` is written and read by
// the same thread, never shared, so a register does it and the shared round
// trip goes away.
//
// And z and weight do not depend on the reduction, so their loads can be issued
// before it rather than after the two barriers it needs. Three loads in flight
// instead of one, with two of the latencies overlapping the reduction.
//
// Values, expressions and the reduction tree are all unchanged.
// A block walks GRS_ROWS rows, one thread per column, rather than taking a row
// each: at the shipped shape that is 108k blocks of 128 threads doing one
// 128-wide reduction apiece. The per-row reduction tree below is untouched --
// same partials, same shuffle order, same two-stage warp sum -- so the rows
// come out bit-for-bit as they did. Only the weight load and the block launch
// are amortised.
#define GRS_ROWS 8


// Fixed 128-column gated RMSNorm. One warp owns one row, and each lane owns
// columns lane, lane+32, lane+64, lane+96. Four warps handle four rows in
// parallel, twice per CTA. Reconstruct the original four warp sums and second
// XOR tree within each row's warp, including the +0 lanes.
__global__ __launch_bounds__(128, 4) void gated_rms_silu_bf16_warp4_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ z,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ out,
    int rows, int cols, int z_heads, int64_t z_row_stride,
    int64_t z_col_offset, float eps) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  float wv[4];
#pragma unroll
  for (int s = 0; s < 4; ++s)
    wv[s] = __bfloat162float(weight[lane + 32 * s]);

#pragma unroll
  for (int wave = 0; wave < 2; ++wave) {
    const int row = blockIdx.x * 8 + wave * 4 + warp;
    if (row >= rows) continue;  // Warp-uniform; no CTA barrier in this kernel.
    const int64_t base = static_cast<int64_t>(row) * 128;
    const int64_t z_base = static_cast<int64_t>(row / z_heads) * z_row_stride +
                           z_col_offset + static_cast<int64_t>(row % z_heads) * 128;
    float xv[4], zf[4], partial[4];
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const int col = lane + 32 * s;
      xv[s] = __bfloat162float(x[base + col]);
      zf[s] = __bfloat162float(z[z_base + col]);
      partial[s] = xv[s] * xv[s];
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
      for (int s = 0; s < 4; ++s)
        partial[s] += __shfl_xor_sync(0xffffffff, partial[s], offset);
    }

    // Only lane 0 of each original warp was written to warp_sums[s].
    const float w0 = __shfl_sync(0xffffffff, partial[0], 0);
    const float w1 = __shfl_sync(0xffffffff, partial[1], 0);
    const float w2 = __shfl_sync(0xffffffff, partial[2], 0);
    const float w3 = __shfl_sync(0xffffffff, partial[3], 0);
    float v = lane == 0 ? w0 : (lane == 1 ? w1 :
              (lane == 2 ? w2 : (lane == 3 ? w3 : 0.0f)));
    // Match old warp 0: lanes 4..31 enter as +0 and participate in
    // XOR16,8,4,2,1 rather than directly summing four warp totals.
    for (int offset = 16; offset > 0; offset >>= 1)
      v += __shfl_xor_sync(0xffffffff, v, offset);
    const float sum = __shfl_sync(0xffffffff, v, 0);
    const float rms = rsqrtf(sum / cols + eps);

#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const float y1 = wv[s] * (xv[s] * rms);
      out[base + lane + 32 * s] = __float2bfloat16(
          __fmul_rn(__fmul_rn(y1, zf[s]), la_triton_sigmoid(zf[s])));
    }
  }
}


__global__ void gated_rms_silu_bf16_rowfit_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ z,
    const __nv_bfloat16* __restrict__ weight, __nv_bfloat16* __restrict__ out,
    int rows, int cols, int z_heads, int64_t z_row_stride, int64_t z_col_offset,
    float eps) {
  const int i = threadIdx.x;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const float wv = __bfloat162float(weight[i]);
  __shared__ float warp_sums[32];
  const int row_base = blockIdx.x * GRS_ROWS;

  for (int r = 0; r < GRS_ROWS; ++r) {
    const int row = row_base + r;
    if (row >= rows) return;
    const int64_t base = static_cast<int64_t>(row) * cols;
    const int64_t z_base = static_cast<int64_t>(row / z_heads) * z_row_stride +
                           z_col_offset + static_cast<int64_t>(row % z_heads) * cols;
    const float xv = __bfloat162float(x[base + i]);
    const float zf = __bfloat162float(z[z_base + i]);
    float partial = xv * xv;
    for (int offset = 16; offset > 0; offset >>= 1)
      partial += __shfl_xor_sync(0xffffffff, partial, offset);
    if (lane == 0) warp_sums[warp] = partial;
    __syncthreads();
    if (warp == 0) {
      float v = (lane < (blockDim.x + 31) / 32) ? warp_sums[lane] : 0.0f;
      for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, offset);
      if (lane == 0) warp_sums[0] = v;
    }
    __syncthreads();
    const float rms = rsqrtf(warp_sums[0] / cols + eps);
    const float y1 = wv * (xv * rms);
    out[base + i] = __float2bfloat16(
        __fmul_rn(__fmul_rn(y1, zf), la_triton_sigmoid(zf)));
    // The next row reuses warp_sums, so no lane may still be reading it.
    __syncthreads();
  }
}

__global__ void gated_rms_silu_bf16_kernel(
    const __nv_bfloat16* x, const __nv_bfloat16* z, const __nv_bfloat16* weight,
    __nv_bfloat16* out, int cols, int z_heads, int64_t z_row_stride,
    int64_t z_col_offset, float eps) {
  const int row = blockIdx.x;
  extern __shared__ float la_gated[];
  const int64_t base = static_cast<int64_t>(row) * cols;
  const int64_t z_base = static_cast<int64_t>(row / z_heads) * z_row_stride + z_col_offset
      + static_cast<int64_t>(row % z_heads) * cols;
  float partial = 0.0f;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    const float v = __bfloat162float(x[base + i]);
    la_gated[i] = v;
    partial += v * v;
  }
  __shared__ float warp_sums[32];
  for (int offset = 16; offset > 0; offset >>= 1)
    partial += __shfl_xor_sync(0xffffffff, partial, offset);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) warp_sums[warp] = partial;
  __syncthreads();
  if (warp == 0) {
    float v = (lane < (blockDim.x + 31) / 32) ? warp_sums[lane] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1)
      v += __shfl_xor_sync(0xffffffff, v, offset);
    if (lane == 0) warp_sums[0] = v;
  }
  __syncthreads();
  const float rms = rsqrtf(warp_sums[0] / cols + eps);
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    const float y0 = la_gated[i] * rms;
    const float y1 = __bfloat162float(weight[i]) * y0;
    const float zf = __bfloat162float(z[z_base + i]);
    out[base + i] = __float2bfloat16(
        __fmul_rn(__fmul_rn(y1, zf), la_triton_sigmoid(zf)));
  }
}
