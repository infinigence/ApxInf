#pragma once

// Copyright 2026 apxinf contributors.
// Pure CUDA operators grouped by physical operation; launch policy lives under adapters/.

__global__ void quantize_f16_e4m3_kernel(
    const half* input, __nv_fp8_e4m3* output, int64_t count,
    float inverse_scale) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    float value = fminf(448.0f, fmaxf(-448.0f,
        __half2float(input[index]) * inverse_scale));
    output[index] = static_cast<__nv_fp8_e4m3>(value);
  }
}

// Four values per thread with one half2 pair per load and one uint32 store.
// The scalar kernel above remains the fallback for unaligned buffers and the
// final 0..3 values.
__global__ void quantize_f16_e4m3_packed4_kernel(
    const half* input, __nv_fp8_e4m3* output, int64_t vector_count,
    float inverse_scale) {
  int64_t index =
      (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
  const int64_t stride =
      static_cast<int64_t>(blockDim.x) * gridDim.x * 4;
  const half2* input2 = reinterpret_cast<const half2*>(input);
  for (; index < vector_count; index += stride) {
    const half2 first = input2[index / 2];
    const half2 second = input2[index / 2 + 1];
    __nv_fp8_e4m3 values[4];
    values[0] = static_cast<__nv_fp8_e4m3>(fminf(
        448.0f, fmaxf(-448.0f, __half2float(first.x) * inverse_scale)));
    values[1] = static_cast<__nv_fp8_e4m3>(fminf(
        448.0f, fmaxf(-448.0f, __half2float(first.y) * inverse_scale)));
    values[2] = static_cast<__nv_fp8_e4m3>(fminf(
        448.0f, fmaxf(-448.0f, __half2float(second.x) * inverse_scale)));
    values[3] = static_cast<__nv_fp8_e4m3>(fminf(
        448.0f, fmaxf(-448.0f, __half2float(second.y) * inverse_scale)));
    reinterpret_cast<uint32_t*>(output)[index / 4] =
        *reinterpret_cast<const uint32_t*>(values);
  }
}

__global__ void dequantize_e4m3_f16_kernel(
    const __nv_fp8_e4m3* input, half* output, int64_t count, float scale) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    output[index] = __float2half_rn(static_cast<float>(input[index]) * scale);
  }
}


__global__ void quantize_rows_bf16_int8_kernel(
    const __nv_bfloat16* input, int8_t* output, float* scales,
    int rows, int cols) {
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  float maximum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    maximum = fmaxf(
        maximum,
        fabsf(__bfloat162float(input[static_cast<int64_t>(row) * cols + col])));
  }
  const float scale = fmaxf(block_max(maximum, scratch) / 127.0f, 1.0e-12f);
  if (threadIdx.x == 0) scales[row] = scale;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    const float quantized = roundf(__bfloat162float(input[index]) / scale);
    output[index] = static_cast<int8_t>(fminf(127.0f, fmaxf(-128.0f, quantized)));
  }
}

__global__ void dequantize_int32_bf16_kernel(
    const int32_t* accumulators, const float* row_scales,
    const float* column_scales, __nv_bfloat16* output,
    int rows, int cols) {
  const int row = blockIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows && col < cols) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    output[index] = __float2bfloat16(
        static_cast<float>(accumulators[index]) * row_scales[row] *
        column_scales[col]);
  }
}

// ── AutoAWQ INT4 (GEMM packing, group-wise asymmetric) ───────────────────
//
// Layout produced by AutoAWQ `version="gemm"` for a Linear with K inputs and
// N outputs (all arrays row-major):
//   qweight : int32 [K, N/8]      8 nibbles per word
//   qzeros  : int32 [K/G, N/8]    same nibble packing as qweight
//   scales  : half  [K/G, N]      logical column order
// Nibble `i` (bits 4i..4i+3) of word `j` belongs to logical output column
// `8*j + awq_nibble_column(i)`, i.e. the AutoAWQ interleave {0,2,4,6,1,3,5,7}.
// Dequantized value: W[k, n] = (q[k, n] - z[k/G, n]) * s[k/G, n].
//
// Kernels take per-expert element strides so a whole MoE layer can be stored
// in one allocation and addressed by expert index (`blockIdx.y` /
// `expert_ids[slot]`) without host-side pointer arithmetic.

__device__ __forceinline__ int awq_nibble_column(int nibble) {
  return ((nibble & 3) << 1) | (nibble >> 2);
}

// Unpack one expert's [K, N] weight to BF16 row-major. One thread per packed
// word (k, j) writes 8 consecutive columns as a single 16-byte store.
__global__ void awq_dequant_bf16_kernel(
    const int32_t* __restrict__ qweight, const int32_t* __restrict__ qzeros,
    const half* __restrict__ scales, __nv_bfloat16* __restrict__ output,
    int rows, int packed_cols, int group_size, int64_t stride_q,
    int64_t stride_z, int64_t stride_s, int64_t stride_out) {
  const int expert = blockIdx.y;
  qweight += static_cast<int64_t>(expert) * stride_q;
  qzeros += static_cast<int64_t>(expert) * stride_z;
  scales += static_cast<int64_t>(expert) * stride_s;
  output += static_cast<int64_t>(expert) * stride_out;
  const int cols = packed_cols * 8;
  const int64_t total = static_cast<int64_t>(rows) * packed_cols;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total; index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int k = static_cast<int>(index / packed_cols);
    const int j = static_cast<int>(index - static_cast<int64_t>(k) * packed_cols);
    const int g = k / group_size;
    const int32_t q = qweight[index];
    const int32_t z = qzeros[static_cast<int64_t>(g) * packed_cols + j];
    const uint4 packed_scales = *reinterpret_cast<const uint4*>(
        scales + static_cast<int64_t>(g) * cols + j * 8);
    const half* s = reinterpret_cast<const half*>(&packed_scales);
    __align__(16) __nv_bfloat16 values[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int column = awq_nibble_column(i);
      const float w = static_cast<float>((q >> (4 * i)) & 0xF) -
                      static_cast<float>((z >> (4 * i)) & 0xF);
      values[column] = __float2bfloat16(w * __half2float(s[column]));
    }
    *reinterpret_cast<uint4*>(output + static_cast<int64_t>(k) * cols + j * 8) =
        *reinterpret_cast<const uint4*>(values);
  }
}

// Split-K GEMV over AWQ INT4 weights for M=1 decode.
//
//   partial[slot][split][n] = scale(slot) * sum_{k in split} x_slot[k] * W_e[k, n]
//
// grid = (ceil(N/256), splits, slots), block = 256 threads (8 warps).
// Each block owns 256 output columns (32 packed words, one per lane) and a
// contiguous row range of `rows_per_block`; the 8 warps split that range.
// `expert_ids` (nullable) maps slot -> expert; the expert offset is applied on
// device so the launch is CUDA-graph capturable with data-dependent routing.
// `x_slot_stride` is 0 when all slots share the activation and K when each
// slot has its own input row. `slot_scale` (nullable) folds router weights in.
//
// Nibble accumulation trick: within one quantization group the zero point and
// scale are constant, so acc_i = sum x[k]*q_i[k] is accumulated on the raw
// nibble and the affine correction s*(acc - z*sum(x)) is applied once per
// group. This keeps the inner loop at one 32-bit load and eight FMAs.
//
// How many loads each thread keeps in flight. The kernel is latency-bound,
// not bandwidth- or ALU-bound: with a single dependent load per thread it ran
// at ~15% of Thor's 259 GB/s, because ~40 warps per SM holding one 128-byte
// request each is only ~5 KB in flight against a bandwidth-latency product
// two orders of magnitude larger. The smallest warp row range is 64 and the
// group size is 128, so 8 divides every segment evenly.
#define W4A16_GEMV_UNROLL 8
__global__ void __launch_bounds__(256) w4a16_gemv_partial_kernel(
    const __nv_bfloat16* __restrict__ x, int64_t x_slot_stride,
    const int32_t* __restrict__ qweight, const int32_t* __restrict__ qzeros,
    const half* __restrict__ scales, const int32_t* __restrict__ expert_ids,
    int64_t stride_q, int64_t stride_z, int64_t stride_s,
    const float* __restrict__ slot_scale, float* __restrict__ partial,
    int rows, int packed_cols, int group_size, int rows_per_block) {
  extern __shared__ float smem[];
  float* xs = smem;                       // [rows_per_block]
  float* red = smem + rows_per_block;     // [8 warps][256]

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int slot = blockIdx.z;
  const int split = blockIdx.y;
  const int cols = packed_cols * 8;
  const int expert = expert_ids == nullptr ? slot : expert_ids[slot];
  qweight += static_cast<int64_t>(expert) * stride_q;
  qzeros += static_cast<int64_t>(expert) * stride_z;
  scales += static_cast<int64_t>(expert) * stride_s;
  x += static_cast<int64_t>(slot) * x_slot_stride;

  const int k_begin = split * rows_per_block;
  const int k_end = min(rows, k_begin + rows_per_block);
  for (int k = k_begin + threadIdx.x; k < k_end; k += blockDim.x) {
    xs[k - k_begin] = __bfloat162float(x[k]);
  }
  __syncthreads();

  const int j = blockIdx.x * 32 + lane;
  const bool active = j < packed_cols;
  const int rows_per_warp = rows_per_block / 8;
  const int kw_begin = k_begin + warp * rows_per_warp;
  const int kw_end = min(k_end, kw_begin + rows_per_warp);

  float total[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    total[i] = 0.0f;
  }
  const int32_t* wp = active ? qweight + j : qweight;

  // Walk one quantization group at a time. Within a group the scale and zero
  // point are constant, so the affine correction s*(acc - z*sum(x)) leaves the
  // inner loop entirely and the body reduces to W4A16_GEMV_UNROLL
  // *independent* loads followed by their FMAs.
  //
  // The unroll is the point, not the arithmetic — see W4A16_GEMV_UNROLL above.
  // The old loop tested the group boundary inside the k loop, which both
  // serialized the loads and inhibited unrolling.
  for (int k0 = kw_begin; k0 < kw_end;) {
    const int g = k0 / group_size;
    const int k1 = min(kw_end, (g + 1) * group_size);

    float zero[8];
    float scale[8];
    if (active) {
      const int32_t z = qzeros[static_cast<int64_t>(g) * packed_cols + j];
      const uint4 packed_scales = *reinterpret_cast<const uint4*>(
          scales + static_cast<int64_t>(g) * cols + j * 8);
      const half* s = reinterpret_cast<const half*>(&packed_scales);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        zero[i] = static_cast<float>((z >> (4 * i)) & 0xF);
        scale[i] = __half2float(s[awq_nibble_column(i)]);
      }
    } else {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        zero[i] = 0.0f;
        scale[i] = 0.0f;
      }
    }

    float acc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      acc[i] = 0.0f;
    }
    float xsum = 0.0f;

    int k = k0;
    for (; k + W4A16_GEMV_UNROLL <= k1; k += W4A16_GEMV_UNROLL) {
      const int64_t base = static_cast<int64_t>(k) * packed_cols;
      // Issued together: nothing below consumes a value until all of them are
      // in flight, which is the whole point of the unroll.
      int32_t q[W4A16_GEMV_UNROLL];
      float xv[W4A16_GEMV_UNROLL];
#pragma unroll
      for (int u = 0; u < W4A16_GEMV_UNROLL; ++u) {
        q[u] = active ? wp[base + static_cast<int64_t>(u) * packed_cols] : 0;
        xv[u] = xs[k + u - k_begin];
      }
#pragma unroll
      for (int u = 0; u < W4A16_GEMV_UNROLL; ++u) {
        xsum += xv[u];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          acc[i] += xv[u] * static_cast<float>((q[u] >> (4 * i)) & 0xF);
        }
      }
    }
    for (; k < k1; ++k) {
      const float xv = xs[k - k_begin];
      const int32_t q = active ? wp[static_cast<int64_t>(k) * packed_cols] : 0;
      xsum += xv;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        acc[i] += xv * static_cast<float>((q >> (4 * i)) & 0xF);
      }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      total[i] += scale[i] * (acc[i] - zero[i] * xsum);
    }
    k0 = k1;
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    red[warp * 256 + lane * 8 + awq_nibble_column(i)] = total[i];
  }
  __syncthreads();
  const int column = threadIdx.x;  // 0..255 within the tile
  float sum = 0.0f;
#pragma unroll
  for (int w = 0; w < 8; ++w) {
    sum += red[w * 256 + column];
  }
  const int n = blockIdx.x * 256 + column;
  if (n < cols) {
    if (slot_scale != nullptr) sum *= slot_scale[slot];
    partial[(static_cast<int64_t>(slot) * gridDim.y + split) * cols + n] = sum;
  }
}

// out[n] = sum_{c < count} partial[c][n], as BF16.
__global__ void partial_sum_bf16_kernel(
    const float* __restrict__ partial, __nv_bfloat16* __restrict__ output,
    int cols, int count) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= cols) return;
  float sum = 0.0f;
  for (int c = 0; c < count; ++c) {
    sum += partial[static_cast<int64_t>(c) * cols + n];
  }
  output[n] = __float2bfloat16(sum);
}

// h[slot][i] = silu(sum_splits gate) * sum_splits up, where each partial row
// holds [gate | up] of width 2*inter.
__global__ void partial_silu_mul_bf16_kernel(
    const float* __restrict__ partial, __nv_bfloat16* __restrict__ output,
    int inter, int splits, int slots) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= static_cast<int64_t>(inter) * slots) return;
  const int slot = static_cast<int>(index / inter);
  const int i = static_cast<int>(index - static_cast<int64_t>(slot) * inter);
  const float* base = partial + static_cast<int64_t>(slot) * splits * 2 * inter;
  float gate = 0.0f;
  float up = 0.0f;
  for (int s = 0; s < splits; ++s) {
    gate += base[static_cast<int64_t>(s) * 2 * inter + i];
    up += base[static_cast<int64_t>(s) * 2 * inter + inter + i];
  }
  const float silu = gate / (1.0f + expf(-gate));
  output[index] = __float2bfloat16(silu * up);
}
