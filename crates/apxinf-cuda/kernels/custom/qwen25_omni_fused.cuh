#pragma once

union __align__(16) Qwen25OmniBf16Pack8 {
  uint4 raw;
  __nv_bfloat16 value[8];
};

// Fixed H2048/M1 residual-add + RMSNorm for the Qwen2.5-Omni short graph.
// Global I/O uses one aligned 128-bit transaction per eight BF16 values, but
// the square-sum is deliberately reconstructed in the incumbent
// threadIdx + 256*j order so BF16 residual and normalized output stay exact.
__global__ void qwen25_omni_residual_rmsnorm_pack8_bf16_kernel(
    __nv_bfloat16* residual, const __nv_bfloat16* delta,
    const __nv_bfloat16* weight, __nv_bfloat16* output, float eps) {
  constexpr int kColumns = 2048;
  constexpr int kThreads = 256;
  const int thread = threadIdx.x;
  extern __shared__ float updated_values[];
  __shared__ float reduced_sum;
  __shared__ float warp_sums[8];

  Qwen25OmniBf16Pack8 residual_pack;
  Qwen25OmniBf16Pack8 delta_pack;
  Qwen25OmniBf16Pack8 updated_pack;
  residual_pack.raw = reinterpret_cast<const uint4*>(residual)[thread];
  delta_pack.raw = reinterpret_cast<const uint4*>(delta)[thread];
  const int contiguous_base = thread * 8;
#pragma unroll
  for (int item = 0; item < 8; ++item) {
    updated_pack.value[item] = __float2bfloat16(
        __bfloat162float(residual_pack.value[item]) +
        __bfloat162float(delta_pack.value[item]));
    updated_values[contiguous_base + item] =
        __bfloat162float(updated_pack.value[item]);
  }
  reinterpret_cast<uint4*>(residual)[thread] = updated_pack.raw;
  __syncthreads();

  float partial = 0.0f;
#pragma unroll 1
  for (int item = 0; item < 8; ++item) {
    const float value = updated_values[thread + item * kThreads];
    partial += value * value;
  }
  for (int shift = 16; shift > 0; shift >>= 1)
    partial += __shfl_xor_sync(0xffffffff, partial, shift);
  const int warp = thread / 32;
  const int lane = thread % 32;
  if (lane == 0) warp_sums[warp] = partial;
  __syncthreads();
  if (warp == 0) {
    float value = lane < 8 ? warp_sums[lane] : 0.0f;
    for (int shift = 16; shift > 0; shift >>= 1)
      value += __shfl_xor_sync(0xffffffff, value, shift);
    if (lane == 0) reduced_sum = value;
  }
  __syncthreads();
  const float inverse_rms =
      rsqrtf(reduced_sum / static_cast<float>(kColumns) + eps);

  Qwen25OmniBf16Pack8 weight_pack;
  Qwen25OmniBf16Pack8 output_pack;
  weight_pack.raw = reinterpret_cast<const uint4*>(weight)[thread];
#pragma unroll
  for (int item = 0; item < 8; ++item) {
    output_pack.value[item] = __float2bfloat16(
        updated_values[contiguous_base + item] * inverse_rms *
        __bfloat162float(weight_pack.value[item]));
  }
  reinterpret_cast<uint4*>(output)[thread] = output_pack.raw;
}
