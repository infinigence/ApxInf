#pragma once

// Correctness-first asymmetric W4 projection. Weights pack eight K nibbles
// per uint32; zero-points pack eight N nibbles per uint32 for each K group.
// One block computes one [row, out] scalar with fixed 256-float shared scratch.
__global__ void qwen35_w4_project_bf16_kernel(
    const __nv_bfloat16* activation, const uint32_t* weight_packed,
    const __nv_bfloat16* scales, const uint32_t* zero_points,
    __nv_bfloat16* output, uint32_t* error_flags, int rows, int out_features,
    int in_features, int group_size) {
  __shared__ float partial[256];
  const int output_index = blockIdx.x;
  const int row = output_index / out_features;
  const int out = output_index - row * out_features;
  const int groups = (in_features + group_size - 1) / group_size;
  const int packed_k_columns = (in_features + 7) / 8;
  float sum = 0.0f;
  for (int k = threadIdx.x; k < in_features; k += blockDim.x) {
    const int group = k / group_size;
    const uint32_t packed_weight =
        weight_packed[static_cast<int64_t>(out) * packed_k_columns + k / 8];
    const float quantized =
        static_cast<float>((packed_weight >> (4 * (k & 7))) & 0x0fU);
    const uint32_t packed_zero =
        zero_points[static_cast<int64_t>(out / 8) * groups + group];
    const float zero_point =
        static_cast<float>((packed_zero >> (4 * (out & 7))) & 0x0fU);
    const float scale =
        __bfloat162float(scales[static_cast<int64_t>(out) * groups + group]);
    if (!isfinite(scale)) atomicOr(error_flags, 1U);
    const float value =
        __bfloat162float(activation[static_cast<int64_t>(row) * in_features + k]);
    const float dequantized = (quantized - zero_point) * scale;
    // compressed-tensors decompresses each packed weight to the model's
    // native BF16 dtype before the BF16 linear consumes it. Preserve that
    // rounding boundary instead of multiplying by an unrounded FP32 value.
    const float weight = __bfloat162float(__float2bfloat16(dequantized));
    sum += value * weight;
  }
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride != 0; stride /= 2) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    if (!isfinite(partial[0])) atomicOr(error_flags, 2U);
    const __nv_bfloat16 converted = __float2bfloat16(partial[0]);
    if (!isfinite(__bfloat162float(converted))) atomicOr(error_flags, 2U);
    output[output_index] = converted;
  }
}

// Bandwidth-oriented packed-W4 GEMV for decode-shaped calls.
//
// The baseline kernel above assigns one thread per K element, so the eight
// threads covering one packed uint32 each issue their own load for it, and
// every thread re-reads its group's scale and zero-point. This variant gives
// each thread one whole packed uint32 (eight consecutive K values), which
// always lies inside a single group because group_size is a multiple of eight.
// Per uint32 that is one weight load, one scale load and one zero-point load
// instead of eight of each.
//
// Weight decompression is unchanged: each nibble is scaled in FP32 and rounded
// to BF16 exactly once before the multiply. Only the accumulation order and
// the reduction differ (warp shuffle plus a short shared-memory stage rather
// than a 256-entry tree), which is the same class of sub-ulp reassociation as
// the prefill GEMM path.
__global__ void qwen35_w4_project_bf16_packed_kernel(
    const __nv_bfloat16* activation, const uint32_t* weight_packed,
    const __nv_bfloat16* scales, const uint32_t* zero_points,
    __nv_bfloat16* output, uint32_t* error_flags, int rows, int out_features,
    int in_features, int group_size) {
  __shared__ float warp_sums[32];
  const int output_index = blockIdx.x;
  const int row = output_index / out_features;
  const int out = output_index - row * out_features;
  const int groups = (in_features + group_size - 1) / group_size;
  const int packed_k_columns = (in_features + 7) / 8;
  const int64_t weight_base = static_cast<int64_t>(out) * packed_k_columns;
  const int64_t activation_base = static_cast<int64_t>(row) * in_features;
  const int64_t scale_base = static_cast<int64_t>(out) * groups;
  const int64_t zero_base = static_cast<int64_t>(out / 8) * groups;
  const int zero_shift = 4 * (out & 7);

  float sum = 0.0f;
  for (int column = threadIdx.x; column < packed_k_columns;
       column += blockDim.x) {
    const int k_begin = column * 8;
    const int group = k_begin / group_size;
    const uint32_t packed_weight = weight_packed[weight_base + column];
    const float zero_point =
        static_cast<float>((zero_points[zero_base + group] >> zero_shift) & 0x0fU);
    const float scale = __bfloat162float(scales[scale_base + group]);
    if (!isfinite(scale)) atomicOr(error_flags, 1U);
    const int lanes = min(8, in_features - k_begin);
    for (int nibble = 0; nibble < lanes; ++nibble) {
      const float quantized =
          static_cast<float>((packed_weight >> (4 * nibble)) & 0x0fU);
      const float weight =
          __bfloat162float(__float2bfloat16((quantized - zero_point) * scale));
      const float value =
          __bfloat162float(activation[activation_base + k_begin + nibble]);
      sum += value * weight;
    }
  }

  // Warp-level reduction, then one shared-memory stage across warps.
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffffU, sum, offset);
  }
  const int lane = threadIdx.x & (warpSize - 1);
  const int warp = threadIdx.x / warpSize;
  const int warps = (blockDim.x + warpSize - 1) / warpSize;
  if (lane == 0) warp_sums[warp] = sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = 0.0f;
    for (int index = 0; index < warps; ++index) total += warp_sums[index];
    if (!isfinite(total)) atomicOr(error_flags, 2U);
    const __nv_bfloat16 converted = __float2bfloat16(total);
    if (!isfinite(__bfloat162float(converted))) atomicOr(error_flags, 2U);
    output[output_index] = converted;
  }
}

// Warp-per-output packed-W4 GEMV.
//
// The packed kernel above still assigns one 256-thread block per output
// element, so with in_features=5120 each thread handles only 2.5 packed
// uint32 values and a whole block streams just 2.5 KB — too little work to
// hide memory latency (measured ~20% of peak bandwidth). Here one warp owns
// one output and a block owns `WARPS_PER_BLOCK` consecutive outputs, giving
// each thread ~20 uint32 values, replacing the shared-memory tree with a pure
// warp shuffle, and letting the activation row (loaded once into shared
// memory) be reused by every warp in the block.
//
// Decompression rounding is identical to both kernels above: one FP32 scale
// followed by a single BF16 round per weight, before any multiply.
template <int WARPS_PER_BLOCK>
__global__ void qwen35_w4_project_bf16_warp_kernel(
    const __nv_bfloat16* activation, const uint32_t* weight_packed,
    const __nv_bfloat16* scales, const uint32_t* zero_points,
    __nv_bfloat16* output, uint32_t* error_flags, int rows, int out_features,
    int in_features, int group_size) {
  extern __shared__ float shared_activation[];
  const int warp = threadIdx.x / warpSize;
  const int lane = threadIdx.x % warpSize;
  const int row = blockIdx.y;
  const int out = blockIdx.x * WARPS_PER_BLOCK + warp;
  const int groups = (in_features + group_size - 1) / group_size;
  const int packed_k_columns = (in_features + 7) / 8;
  const int64_t activation_base = static_cast<int64_t>(row) * in_features;

  // Cooperative load of this row's activation; every warp reads all of it.
  for (int index = threadIdx.x; index < in_features; index += blockDim.x) {
    shared_activation[index] =
        __bfloat162float(activation[activation_base + index]);
  }
  __syncthreads();

  if (out >= out_features) return;

  const int64_t weight_base = static_cast<int64_t>(out) * packed_k_columns;
  const int64_t scale_base = static_cast<int64_t>(out) * groups;
  const int64_t zero_base = static_cast<int64_t>(out / 8) * groups;
  const int zero_shift = 4 * (out & 7);

  float sum = 0.0f;
  for (int column = lane; column < packed_k_columns; column += warpSize) {
    const int k_begin = column * 8;
    const int group = k_begin / group_size;
    const uint32_t packed_weight = weight_packed[weight_base + column];
    const float zero_point = static_cast<float>(
        (zero_points[zero_base + group] >> zero_shift) & 0x0fU);
    const float scale = __bfloat162float(scales[scale_base + group]);
    if (!isfinite(scale)) atomicOr(error_flags, 1U);
    const int lanes = min(8, in_features - k_begin);
    for (int nibble = 0; nibble < lanes; ++nibble) {
      const float quantized =
          static_cast<float>((packed_weight >> (4 * nibble)) & 0x0fU);
      const float weight =
          __bfloat162float(__float2bfloat16((quantized - zero_point) * scale));
      sum += shared_activation[k_begin + nibble] * weight;
    }
  }
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffffU, sum, offset);
  }
  if (lane == 0) {
    if (!isfinite(sum)) atomicOr(error_flags, 2U);
    const __nv_bfloat16 converted = __float2bfloat16(sum);
    if (!isfinite(__bfloat162float(converted))) atomicOr(error_flags, 2U);
    output[static_cast<int64_t>(row) * out_features + out] = converted;
  }
}

// Diagnostic GEMV variants used to attribute the W4 decode cost. These are
// selected only by `APXINF_Q35_W4_DIAG_KERNEL` for measurement; none is a
// production path, and each documents exactly which property it drops.
//
// Variant 1 (`nodequant`): keeps the identical memory access pattern of the
// packed kernel but replaces the per-nibble dequantize+round chain with a
// single multiply. It isolates "how much of the cost is the dequantization
// arithmetic" from "how much is streaming the packed bytes". NUMERICALLY
// WRONG on purpose — never selectable in production.
__global__ void qwen35_w4_diag_nodequant_kernel(
    const __nv_bfloat16* activation, const uint32_t* weight_packed,
    const __nv_bfloat16* scales, const uint32_t* zero_points,
    __nv_bfloat16* output, uint32_t* error_flags, int rows, int out_features,
    int in_features, int group_size) {
  __shared__ float warp_sums[32];
  const int output_index = blockIdx.x;
  const int out = output_index % out_features;
  const int row = output_index / out_features;
  const int packed_k_columns = (in_features + 7) / 8;
  const int64_t weight_base = static_cast<int64_t>(out) * packed_k_columns;
  const int64_t activation_base = static_cast<int64_t>(row) * in_features;
  float sum = 0.0f;
  for (int column = threadIdx.x; column < packed_k_columns;
       column += blockDim.x) {
    const uint32_t packed_weight = weight_packed[weight_base + column];
    const int k_begin = column * 8;
    const int lanes = min(8, in_features - k_begin);
    for (int nibble = 0; nibble < lanes; ++nibble) {
      const float quantized =
          static_cast<float>((packed_weight >> (4 * nibble)) & 0x0fU);
      sum += __bfloat162float(activation[activation_base + k_begin + nibble]) *
             quantized;
    }
  }
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffffU, sum, offset);
  }
  const int lane = threadIdx.x & (warpSize - 1);
  const int warp = threadIdx.x / warpSize;
  const int warps = (blockDim.x + warpSize - 1) / warpSize;
  if (lane == 0) warp_sums[warp] = sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = 0.0f;
    for (int index = 0; index < warps; ++index) total += warp_sums[index];
    output[output_index] = __float2bfloat16(total);
  }
  (void)scales;
  (void)zero_points;
  (void)error_flags;
  (void)rows;
}

// Variant 2 (`streamonly`): reads the packed weights and accumulates their raw
// bits, touching no activation and doing no dequantization. It measures the
// pure achievable read bandwidth for this exact layout and grid shape, i.e.
// the floor any correct kernel with this access pattern must respect.
// NUMERICALLY MEANINGLESS on purpose.
__global__ void qwen35_w4_diag_streamonly_kernel(
    const __nv_bfloat16* activation, const uint32_t* weight_packed,
    const __nv_bfloat16* scales, const uint32_t* zero_points,
    __nv_bfloat16* output, uint32_t* error_flags, int rows, int out_features,
    int in_features, int group_size) {
  __shared__ float warp_sums[32];
  const int output_index = blockIdx.x;
  const int out = output_index % out_features;
  const int packed_k_columns = (in_features + 7) / 8;
  const int64_t weight_base = static_cast<int64_t>(out) * packed_k_columns;
  float sum = 0.0f;
  for (int column = threadIdx.x; column < packed_k_columns;
       column += blockDim.x) {
    sum += static_cast<float>(weight_packed[weight_base + column] & 0xffU);
  }
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffffU, sum, offset);
  }
  const int lane = threadIdx.x & (warpSize - 1);
  const int warp = threadIdx.x / warpSize;
  const int warps = (blockDim.x + warpSize - 1) / warpSize;
  if (lane == 0) warp_sums[warp] = sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = 0.0f;
    for (int index = 0; index < warps; ++index) total += warp_sums[index];
    output[output_index] = __float2bfloat16(total);
  }
  (void)activation;
  (void)scales;
  (void)zero_points;
  (void)error_flags;
  (void)rows;
  (void)group_size;
}

// Variant 3 (`vec4`): production-equivalent arithmetic, but each thread loads
// four consecutive packed uint32 values as one 16-byte `uint4` transaction
// (32 K values per thread-step) instead of one uint32. Accumulation order
// changes, so it is a candidate rather than a diagnostic; correctness is
// checked against the packed kernel at reduction tolerance.
__global__ void qwen35_w4_project_bf16_vec4_kernel(
    const __nv_bfloat16* activation, const uint32_t* weight_packed,
    const __nv_bfloat16* scales, const uint32_t* zero_points,
    __nv_bfloat16* output, uint32_t* error_flags, int rows, int out_features,
    int in_features, int group_size) {
  __shared__ float warp_sums[32];
  const int output_index = blockIdx.x;
  const int out = output_index % out_features;
  const int row = output_index / out_features;
  const int groups = (in_features + group_size - 1) / group_size;
  const int packed_k_columns = (in_features + 7) / 8;
  const int64_t weight_base = static_cast<int64_t>(out) * packed_k_columns;
  const int64_t activation_base = static_cast<int64_t>(row) * in_features;
  const int64_t scale_base = static_cast<int64_t>(out) * groups;
  const int64_t zero_base = static_cast<int64_t>(out / 8) * groups;
  const int zero_shift = 4 * (out & 7);
  const int vec_columns = packed_k_columns / 4;

  float sum = 0.0f;
  // Vectorized body: 16-byte loads over the aligned prefix.
  const uint4* weight_vec4 =
      reinterpret_cast<const uint4*>(weight_packed + weight_base);
  for (int vec = threadIdx.x; vec < vec_columns; vec += blockDim.x) {
    const uint4 packed = weight_vec4[vec];
    const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
    for (int word = 0; word < 4; ++word) {
      const int k_begin = (vec * 4 + word) * 8;
      const int group = k_begin / group_size;
      const float zero_point = static_cast<float>(
          (zero_points[zero_base + group] >> zero_shift) & 0x0fU);
      const float scale = __bfloat162float(scales[scale_base + group]);
      if (!isfinite(scale)) atomicOr(error_flags, 1U);
      for (int nibble = 0; nibble < 8; ++nibble) {
        const float quantized =
            static_cast<float>((words[word] >> (4 * nibble)) & 0x0fU);
        const float weight = __bfloat162float(
            __float2bfloat16((quantized - zero_point) * scale));
        sum += __bfloat162float(activation[activation_base + k_begin + nibble]) *
               weight;
      }
    }
  }
  // Scalar tail for the columns the vectorized body could not cover.
  for (int column = vec_columns * 4 + threadIdx.x; column < packed_k_columns;
       column += blockDim.x) {
    const int k_begin = column * 8;
    const int group = k_begin / group_size;
    const uint32_t packed_weight = weight_packed[weight_base + column];
    const float zero_point = static_cast<float>(
        (zero_points[zero_base + group] >> zero_shift) & 0x0fU);
    const float scale = __bfloat162float(scales[scale_base + group]);
    if (!isfinite(scale)) atomicOr(error_flags, 1U);
    const int lanes = min(8, in_features - k_begin);
    for (int nibble = 0; nibble < lanes; ++nibble) {
      const float quantized =
          static_cast<float>((packed_weight >> (4 * nibble)) & 0x0fU);
      const float weight =
          __bfloat162float(__float2bfloat16((quantized - zero_point) * scale));
      sum += __bfloat162float(activation[activation_base + k_begin + nibble]) *
             weight;
    }
  }

  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffffU, sum, offset);
  }
  const int lane = threadIdx.x & (warpSize - 1);
  const int warp = threadIdx.x / warpSize;
  const int warps = (blockDim.x + warpSize - 1) / warpSize;
  if (lane == 0) warp_sums[warp] = sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = 0.0f;
    for (int index = 0; index < warps; ++index) total += warp_sums[index];
    if (!isfinite(total)) atomicOr(error_flags, 2U);
    const __nv_bfloat16 converted = __float2bfloat16(total);
    if (!isfinite(__bfloat162float(converted))) atomicOr(error_flags, 2U);
    output[output_index] = converted;
  }
}

// Variant 4/5 candidate: production-equivalent decompression with the two
// arithmetic inefficiencies the micro-benchmark exposed removed.
//
// Micro-benchmark on the MLP gate shape (44.6 MB packed): streaming the packed
// bytes alone costs 52 us (815 GB/s, 81% of peak), adding the activation load
// and multiply-accumulate costs 79 us, and the full kernel costs 128 us — so
// ~39% of the time is dequantization arithmetic, not memory.
//
// Two changes, both measurable in isolation via `FUSE_ZERO_POINT`:
//   * four independent accumulators break the serial `sum +=` dependency
//     chain (one FP32 FMA latency per nibble otherwise);
//   * with FUSE_ZERO_POINT, `(q - z) * s` becomes `fmaf(q, s, -z*s)`, one FMA
//     instead of a subtract plus a multiply, with `-z*s` hoisted per group.
//
// The BF16 rounding boundary is preserved exactly: each weight is still
// rounded once via `__float2bfloat16` before the multiply. Accumulation order
// changes (four partials, pairwise combined), which is the same class of
// reassociation as the accepted prefill GEMM path, so correctness is asserted
// at reduction tolerance against the packed kernel rather than bitwise.
template <bool FUSE_ZERO_POINT>
__global__ void qwen35_w4_project_bf16_fast_kernel(
    const __nv_bfloat16* activation, const uint32_t* weight_packed,
    const __nv_bfloat16* scales, const uint32_t* zero_points,
    __nv_bfloat16* output, uint32_t* error_flags, int rows, int out_features,
    int in_features, int group_size) {
  __shared__ float warp_sums[32];
  const int output_index = blockIdx.x;
  const int out = output_index % out_features;
  const int row = output_index / out_features;
  const int groups = (in_features + group_size - 1) / group_size;
  const int packed_k_columns = (in_features + 7) / 8;
  const int64_t weight_base = static_cast<int64_t>(out) * packed_k_columns;
  const int64_t activation_base = static_cast<int64_t>(row) * in_features;
  const int64_t scale_base = static_cast<int64_t>(out) * groups;
  const int64_t zero_base = static_cast<int64_t>(out / 8) * groups;
  const int zero_shift = 4 * (out & 7);

  float sum0 = 0.0f;
  float sum1 = 0.0f;
  float sum2 = 0.0f;
  float sum3 = 0.0f;
  for (int column = threadIdx.x; column < packed_k_columns;
       column += blockDim.x) {
    const int k_begin = column * 8;
    const int group = k_begin / group_size;
    const uint32_t packed_weight = weight_packed[weight_base + column];
    const float zero_point = static_cast<float>(
        (zero_points[zero_base + group] >> zero_shift) & 0x0fU);
    const float scale = __bfloat162float(scales[scale_base + group]);
    if (!isfinite(scale)) atomicOr(error_flags, 1U);
    const float neg_zero_scale = -zero_point * scale;
    const int lanes = min(8, in_features - k_begin);
    // Fully unrolled with named accumulators. An earlier attempt indexed a
    // `float partial[4]` by `nibble & 3`; that dynamic index forces the array
    // into local (device) memory and measured 17% SLOWER than the baseline.
    // Fixed indices keep all four partials in registers.
    const __nv_bfloat16* row_activation =
        activation + activation_base + k_begin;
#define APXINF_W4_NIBBLE(index, accumulator)                                  \
  if (lanes > (index)) {                                                      \
    const float quantized =                                                   \
        static_cast<float>((packed_weight >> (4 * (index))) & 0x0fU);          \
    const float dequantized =                                                 \
        FUSE_ZERO_POINT ? __fmaf_rn(quantized, scale, neg_zero_scale)          \
                        : (quantized - zero_point) * scale;                    \
    (accumulator) = __fmaf_rn(__bfloat162float(row_activation[index]),         \
                              __bfloat162float(__float2bfloat16(dequantized)), \
                              (accumulator));                                  \
  }
    APXINF_W4_NIBBLE(0, sum0)
    APXINF_W4_NIBBLE(1, sum1)
    APXINF_W4_NIBBLE(2, sum2)
    APXINF_W4_NIBBLE(3, sum3)
    APXINF_W4_NIBBLE(4, sum0)
    APXINF_W4_NIBBLE(5, sum1)
    APXINF_W4_NIBBLE(6, sum2)
    APXINF_W4_NIBBLE(7, sum3)
#undef APXINF_W4_NIBBLE
  }
  float sum = (sum0 + sum1) + (sum2 + sum3);
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffffU, sum, offset);
  }
  const int lane = threadIdx.x & (warpSize - 1);
  const int warp = threadIdx.x / warpSize;
  const int warps = (blockDim.x + warpSize - 1) / warpSize;
  if (lane == 0) warp_sums[warp] = sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = 0.0f;
    for (int index = 0; index < warps; ++index) total += warp_sums[index];
    if (!isfinite(total)) atomicOr(error_flags, 2U);
    const __nv_bfloat16 converted = __float2bfloat16(total);
    if (!isfinite(__bfloat162float(converted))) atomicOr(error_flags, 2U);
    output[output_index] = converted;
  }
}

// A per-group dequantization lookup table was designed and discarded before
// implementation. Within a group only 16 distinct weights exist, so a table
// looked attractive, but the thread-to-column mapping defeats it: with each
// thread striding by blockDim.x, the 32 lanes of a warp cover columns that
// span eight different groups, so a per-warp table is invalid, and remapping
// threads to own whole groups yields a build-to-use ratio of 16:32 — an upper
// bound of half the dequantization arithmetic, i.e. ~19.5% of kernel time,
// for a substantial restructuring. Recorded here so the reasoning is not
// repeated.

// Marlin-style bit-trick dequantization for packed W4 GEMV.
//
// The micro-benchmark matrix localized 39% of this kernel's time to
// dequantization arithmetic (128.5 us total, 78.7 us with dequant removed,
// 52.2 us streaming weights alone). The production path spends, per nibble,
// a shift, a mask, an int->float convert, a subtract, a multiply, a
// float->bf16 round and a bf16->float convert. This variant replaces the
// convert/subtract/multiply chain with SIMD bf16x2 operations built directly
// from the packed bits, processing two nibbles at a time.
//
// The trick: BF16 has 8 significant bits, so the bit pattern `0x4300 | q`
// is exactly the value `128 + q` for any q in 0..15 (verified exhaustively).
// Two nibbles can therefore be materialized as one bf16x2 with a mask, a
// shift and an OR, with no numeric conversion instructions at all.
//
// Bitwise equality with the production kernel is guaranteed, not hoped for:
//   * `(128+q) - (128+z)` is exact in BF16 for q,z in 0..15 (both operands
//     and the result are small integers inside BF16's 8-bit significand);
//   * `(q-z)` needs at most 5 significant bits and `s` has 8, so the exact
//     product needs at most 13 — FP32's 24-bit significand holds it exactly.
//     The production path therefore performs exactly one rounding (FP32
//     product -> BF16), which is precisely what the BF16 multiply computes.
// Accumulation stays FP32 and in the same order, so the whole kernel is
// bit-identical to the packed kernel.
//
// MEASURED OUTCOME: REJECTED, 54% slower (203 us vs 131 us on the MLP gate
// shape, two independent GPUs per arm). SASS disassembly shows this variant
// issues FEWER instructions than the production kernel, and `__hmul2` does
// compile to native `HFMA2` on SM89 — so neither instruction count nor
// missing hardware support explains it. The cause is latency, not throughput:
// the `HSUB2 -> HMUL2 -> unpack -> FFMA` chain is longer per element than the
// scalar FP32 chain, and an M=1 GEMV has too little per-thread work to hide
// it. The technique pays off in batched GEMM, where occupancy hides latency
// and tensor-core MMA consumes the packed pairs directly; it does not
// transfer to GEMV. Kept as a diagnostic variant only.
__global__ void qwen35_w4_project_bf16_marlin_kernel(
    const __nv_bfloat16* activation, const uint32_t* weight_packed,
    const __nv_bfloat16* scales, const uint32_t* zero_points,
    __nv_bfloat16* output, uint32_t* error_flags, int rows, int out_features,
    int in_features, int group_size) {
  __shared__ float warp_sums[32];
  const int output_index = blockIdx.x;
  const int out = output_index % out_features;
  const int row = output_index / out_features;
  const int groups = (in_features + group_size - 1) / group_size;
  const int packed_k_columns = (in_features + 7) / 8;
  const int64_t weight_base = static_cast<int64_t>(out) * packed_k_columns;
  const int64_t activation_base = static_cast<int64_t>(row) * in_features;
  const int64_t scale_base = static_cast<int64_t>(out) * groups;
  const int64_t zero_base = static_cast<int64_t>(out / 8) * groups;
  const int zero_shift = 4 * (out & 7);

  float sum = 0.0f;
  for (int column = threadIdx.x; column < packed_k_columns;
       column += blockDim.x) {
    const int k_begin = column * 8;
    const int group = k_begin / group_size;
    const uint32_t packed_weight = weight_packed[weight_base + column];
    const uint32_t zero_code =
        (zero_points[zero_base + group] >> zero_shift) & 0x0fU;
    const __nv_bfloat16 scale = scales[scale_base + group];
    if (!isfinite(__bfloat162float(scale))) atomicOr(error_flags, 1U);
    // Broadcast `128 + z` and the scale across both bf16x2 lanes. All bf16
    // values are built with `__ushort_as_bfloat16` / `__halves2bfloat162`,
    // never by taking the address of a local: a `reinterpret_cast` on a stack
    // variable forces a local-memory round trip and measured 54% SLOWER than
    // the production kernel.
    const __nv_bfloat162 zero_pair = __bfloat162bfloat162(
        __ushort_as_bfloat16(static_cast<unsigned short>(0x4300u | zero_code)));
    const __nv_bfloat162 scale_pair = __bfloat162bfloat162(scale);
    const int lanes = min(8, in_features - k_begin);

    for (int pair = 0; pair * 2 < lanes; ++pair) {
      const int shift = 8 * pair;
      // Two nibbles -> two bf16 values `128 + q`, by bit construction only.
      const unsigned short low =
          static_cast<unsigned short>(0x4300u | ((packed_weight >> shift) & 0x0fU));
      const unsigned short high = static_cast<unsigned short>(
          0x4300u | ((packed_weight >> (shift + 4)) & 0x0fU));
      const __nv_bfloat162 code_pair = __halves2bfloat162(
          __ushort_as_bfloat16(low), __ushort_as_bfloat16(high));
      // (128+q) - (128+z) = q - z exactly, then one BF16 multiply by s.
      const __nv_bfloat162 weight_pair =
          __hmul2(__hsub2(code_pair, zero_pair), scale_pair);
      const int k = k_begin + pair * 2;
      sum = __fmaf_rn(__bfloat162float(activation[activation_base + k]),
                      __bfloat162float(__low2bfloat16(weight_pair)), sum);
      if (pair * 2 + 1 < lanes) {
        sum = __fmaf_rn(__bfloat162float(activation[activation_base + k + 1]),
                        __bfloat162float(__high2bfloat16(weight_pair)), sum);
      }
    }
  }

  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffffU, sum, offset);
  }
  const int lane = threadIdx.x & (warpSize - 1);
  const int warp = threadIdx.x / warpSize;
  const int warps = (blockDim.x + warpSize - 1) / warpSize;
  if (lane == 0) warp_sums[warp] = sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = 0.0f;
    for (int index = 0; index < warps; ++index) total += warp_sums[index];
    if (!isfinite(total)) atomicOr(error_flags, 2U);
    const __nv_bfloat16 converted = __float2bfloat16(total);
    if (!isfinite(__bfloat162float(converted))) atomicOr(error_flags, 2U);
    output[output_index] = converted;
  }
}

// Dequantize the packed asymmetric W4 matrix into a dense BF16 checkpoint-
// layout [out_features, in_features] scratch buffer, so a large-M prefill can
// hand the result to a tensor-core BF16 GEMM instead of re-reading the packed
// weight once per activation row.
//
// The rounding boundary is identical to the GEMV kernel above: each weight is
// decompressed to FP32 and then rounded to BF16 exactly once, before any
// multiply. The dot-product accumulation order does change (cuBLAS tiling
// instead of the fixed 256-thread tree), which is the same class of sub-ulp
// reassociation already present between the eager and chunked GDN paths.
__global__ void qwen35_w4_dequantize_bf16_kernel(
    const uint32_t* weight_packed, const __nv_bfloat16* scales,
    const uint32_t* zero_points, __nv_bfloat16* dequantized,
    uint32_t* error_flags, int out_features, int in_features, int group_size) {
  const int64_t total =
      static_cast<int64_t>(out_features) * static_cast<int64_t>(in_features);
  const int groups = (in_features + group_size - 1) / group_size;
  const int packed_k_columns = (in_features + 7) / 8;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total; index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int out = static_cast<int>(index / in_features);
    const int k = static_cast<int>(index - static_cast<int64_t>(out) * in_features);
    const int group = k / group_size;
    const uint32_t packed_weight =
        weight_packed[static_cast<int64_t>(out) * packed_k_columns + k / 8];
    const float quantized =
        static_cast<float>((packed_weight >> (4 * (k & 7))) & 0x0fU);
    const uint32_t packed_zero =
        zero_points[static_cast<int64_t>(out / 8) * groups + group];
    const float zero_point =
        static_cast<float>((packed_zero >> (4 * (out & 7))) & 0x0fU);
    const float scale =
        __bfloat162float(scales[static_cast<int64_t>(out) * groups + group]);
    if (!isfinite(scale)) atomicOr(error_flags, 1U);
    const __nv_bfloat16 weight =
        __float2bfloat16((quantized - zero_point) * scale);
    if (!isfinite(__bfloat162float(weight))) atomicOr(error_flags, 2U);
    dequantized[index] = weight;
  }
}
