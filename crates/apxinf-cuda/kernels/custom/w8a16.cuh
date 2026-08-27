#pragma once

// Weight-only INT8/BF16 decode GEMV. One CTA owns eight output rows and one
// warp owns each row. The BF16 activation and eight FP32 row scales are staged
// once per CTA; INT8 weights remain the sole large streaming input.

__global__ void w8a16_gemv_bf16_kernel(
    const __nv_bfloat16* activation, const int8_t* weight,
    const float* scales, __nv_bfloat16* output, int input_dim) {
  constexpr int kRowsPerBlock = 8;
  extern __shared__ __align__(16) unsigned char storage[];
  __nv_bfloat16* shared_activation =
      reinterpret_cast<__nv_bfloat16*>(storage);
  float* shared_scales = reinterpret_cast<float*>(
      shared_activation + input_dim);
  const int activation_words = input_dim / 8;
  for (int word = threadIdx.x; word < activation_words;
       word += blockDim.x) {
    reinterpret_cast<uint4*>(shared_activation)[word] =
        reinterpret_cast<const uint4*>(activation)[word];
  }
  if (threadIdx.x < kRowsPerBlock) {
    shared_scales[threadIdx.x] =
        scales[static_cast<int>(blockIdx.x) * kRowsPerBlock + threadIdx.x];
  }
  __syncthreads();

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = static_cast<int>(blockIdx.x) * kRowsPerBlock + warp;
  const int packed_cols = input_dim / 4;
  float accumulator = 0.0f;
  for (int packed_col = lane; packed_col < packed_cols; packed_col += 32) {
    const uint32_t weight_word = reinterpret_cast<const uint32_t*>(
        weight + static_cast<int64_t>(row) * input_dim)[packed_col];
    const uint2 activation_word =
        reinterpret_cast<const uint2*>(shared_activation)[packed_col];
    const __nv_bfloat16* activation_values =
        reinterpret_cast<const __nv_bfloat16*>(&activation_word);
#pragma unroll
    for (int index = 0; index < 4; ++index) {
      const int quantized = static_cast<int>(static_cast<int8_t>(
          (weight_word >> (index * 8)) & 0xffU));
      accumulator = fmaf(
          __bfloat162float(activation_values[index]),
          static_cast<float>(quantized), accumulator);
    }
  }
#pragma unroll
  for (int delta = 16; delta > 0; delta >>= 1) {
    accumulator += __shfl_down_sync(0xffffffffU, accumulator, delta);
  }
  if (lane == 0) {
    output[row] = __float2bfloat16(accumulator * shared_scales[warp]);
  }
}
