#pragma once

// Vectorized FP32 mean reduction for contiguous BF16 rows. This preserves the
// four independent accumulators and warp reduction of a separate square/mean
// operation for batches of at least 16 rows and widths divisible by four.
__device__ __forceinline__ float rms_vector_square_mean_bf16(
    const __nv_bfloat16* input, int cols) {
  __shared__ float mean;
  if (threadIdx.x < 32) {
    float sums[4] = {0, 0, 0, 0};
    for (int col = threadIdx.x * 4; col < cols; col += 128) {
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float value = __bfloat162float(input[col + j]);
        sums[j] = __fadd_rn(sums[j], __fmul_rn(value, value));
      }
    }
    float sum = ((sums[0] + sums[1]) + sums[2]) + sums[3];
    for (int offset = 16; offset > 0; offset >>= 1)
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (threadIdx.x == 0)
      mean = __fmul_rn(sum, __fdiv_rn(1.f, static_cast<float>(cols)));
  }
  __syncthreads();
  return mean;
}
