#pragma once

// compressed-tensors pack-quantized W4A16 decode GEMV.
//
// Physical layout for logical weight [N,K]:
//   packed I32 [N,K/8]         eight offset-binary INT4 values per word
//   scale  BF16 [N,K/group]
//   zero   I32 [N/8,K/group]   eight output-row zero points per word
//
// One CTA owns eight adjacent output rows and one warp owns each row. This
// makes the zero-point word common to all warps in a CTA and keeps every weight
// row a single streaming owner. The activation is small (10 KiB at K=5120) and
// reused through cache; no intermediate dequantized weight is materialized.

__device__ __forceinline__ int w4_offset_binary(uint32_t word, int lane) {
  return static_cast<int>((word >> (lane * 4)) & 0x0fU) - 8;
}

__global__ void w4a16_gemv_bf16_kernel(
    const __nv_bfloat16* activation, const int32_t* packed,
    const __nv_bfloat16* scales, const int32_t* packed_zero_points,
    __nv_bfloat16* output, int input_dim) {
  constexpr int kRowsPerBlock = 8;
  constexpr int kGroupSize = 32;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = static_cast<int>(blockIdx.x) * kRowsPerBlock + warp;

  const int packed_cols = input_dim / 8;
  const int groups = input_dim / kGroupSize;
  float accumulator = 0.0f;

  for (int packed_col = lane; packed_col < packed_cols; packed_col += 32) {
    const int group = packed_col >> 2;
    const int group_leader = lane & ~3;
    uint32_t scale_bits = 0;
    uint32_t zero_word = 0;
    if ((lane & 3) == 0) {
      scale_bits = static_cast<uint32_t>(
          reinterpret_cast<const uint16_t*>(scales)[
              static_cast<int64_t>(row) * groups + group]);
      zero_word = static_cast<uint32_t>(packed_zero_points[
          static_cast<int64_t>(row / 8) * groups + group]);
    }
    scale_bits = __shfl_sync(0xffffffffU, scale_bits, group_leader);
    zero_word = __shfl_sync(0xffffffffU, zero_word, group_leader);

    const __nv_bfloat16 scale =
        reinterpret_cast<const __nv_bfloat16*>(&scale_bits)[0];
    const float scale_f32 = __bfloat162float(scale);
    const int zero = w4_offset_binary(zero_word, row & 7);
    const uint32_t weight_word = static_cast<uint32_t>(
        packed[static_cast<int64_t>(row) * packed_cols + packed_col]);
    const uint4 activation_words = reinterpret_cast<const uint4*>(activation)[packed_col];
    const __nv_bfloat16* activation_values =
        reinterpret_cast<const __nv_bfloat16*>(&activation_words);

#pragma unroll
    for (int index = 0; index < 8; ++index) {
      const int quantized = w4_offset_binary(weight_word, index);
      const float weight = static_cast<float>(quantized - zero) * scale_f32;
      accumulator = fmaf(__bfloat162float(activation_values[index]), weight,
                         accumulator);
    }
  }

#pragma unroll
  for (int delta = 16; delta > 0; delta >>= 1) {
    accumulator += __shfl_down_sync(0xffffffffU, accumulator, delta);
  }
  if (lane == 0) output[row] = __float2bfloat16(accumulator);
}

__global__ void w4a16_gemv_bf16_staged_kernel(
    const __nv_bfloat16* activation, const int32_t* packed,
    const __nv_bfloat16* scales, const int32_t* packed_zero_points,
    __nv_bfloat16* output, int input_dim) {
  constexpr int kRowsPerBlock = 8;
  constexpr int kGroupSize = 32;
  extern __shared__ __align__(16) unsigned char storage[];

  const int packed_cols = input_dim / 8;
  const int groups = input_dim / kGroupSize;
  __nv_bfloat16* shared_activation =
      reinterpret_cast<__nv_bfloat16*>(storage);
  __nv_bfloat16* shared_scales = shared_activation + input_dim;
  int32_t* shared_zero_points = reinterpret_cast<int32_t*>(
      shared_scales + kRowsPerBlock * groups);

  for (int packed_col = threadIdx.x; packed_col < packed_cols;
       packed_col += blockDim.x) {
    reinterpret_cast<uint4*>(shared_activation)[packed_col] =
        reinterpret_cast<const uint4*>(activation)[packed_col];
  }
  const int scale_words = kRowsPerBlock * groups / 8;
  const int64_t scale_word_offset =
      static_cast<int64_t>(blockIdx.x) * scale_words;
  for (int word = threadIdx.x; word < scale_words; word += blockDim.x) {
    reinterpret_cast<uint4*>(shared_scales)[word] =
        reinterpret_cast<const uint4*>(scales)[scale_word_offset + word];
  }
  const int64_t zero_offset = static_cast<int64_t>(blockIdx.x) * groups;
  for (int group = threadIdx.x; group < groups; group += blockDim.x) {
    shared_zero_points[group] = packed_zero_points[zero_offset + group];
  }
  __syncthreads();

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = static_cast<int>(blockIdx.x) * kRowsPerBlock + warp;
  float accumulator = 0.0f;
  for (int packed_col = lane; packed_col < packed_cols; packed_col += 32) {
    const int group = packed_col >> 2;
    const float scale = __bfloat162float(shared_scales[warp * groups + group]);
    const int zero = w4_offset_binary(
        static_cast<uint32_t>(shared_zero_points[group]), row & 7);
    const uint32_t weight_word = static_cast<uint32_t>(
        packed[static_cast<int64_t>(row) * packed_cols + packed_col]);
    const uint4 activation_words =
        reinterpret_cast<const uint4*>(shared_activation)[packed_col];
    const __nv_bfloat16* activation_values =
        reinterpret_cast<const __nv_bfloat16*>(&activation_words);

#pragma unroll
    for (int index = 0; index < 8; ++index) {
      const int quantized = w4_offset_binary(weight_word, index);
      const float weight = static_cast<float>(quantized - zero) * scale;
      accumulator = fmaf(__bfloat162float(activation_values[index]), weight,
                         accumulator);
    }
  }
#pragma unroll
  for (int delta = 16; delta > 0; delta >>= 1) {
    accumulator += __shfl_down_sync(0xffffffffU, accumulator, delta);
  }
  if (lane == 0) output[row] = __float2bfloat16(accumulator);
}

// Small-M prefill projection. One CTA still owns eight output rows, but each
// warp reuses its packed weight/scales/zero value across up to eight tokens.
// Activations are read from global/L2 because M*K does not fit shared memory;
// the high-volume checkpoint weight is streamed only once per M-token tile.
__global__ void w4a16_gemm_m8_bf16_kernel(
    const __nv_bfloat16* activation, const int32_t* packed,
    const __nv_bfloat16* scales, const int32_t* packed_zero_points,
    __nv_bfloat16* output, int tokens, int input_dim, int output_dim) {
  constexpr int kRowsPerBlock = 8;
  constexpr int kGroupSize = 32;
  constexpr int kMaxTokens = 8;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = static_cast<int>(blockIdx.x) * kRowsPerBlock + warp;
  const int packed_cols = input_dim / 8;
  const int groups = input_dim / kGroupSize;
  float accumulator[kMaxTokens];
#pragma unroll
  for (int token = 0; token < kMaxTokens; ++token) accumulator[token] = 0.0f;

  for (int packed_col = lane; packed_col < packed_cols; packed_col += 32) {
    const int group = packed_col >> 2;
    const int group_leader = lane & ~3;
    uint32_t scale_bits = 0;
    uint32_t zero_word = 0;
    if ((lane & 3) == 0) {
      scale_bits = static_cast<uint32_t>(
          reinterpret_cast<const uint16_t*>(scales)[
              static_cast<int64_t>(row) * groups + group]);
      zero_word = static_cast<uint32_t>(packed_zero_points[
          static_cast<int64_t>(row / 8) * groups + group]);
    }
    scale_bits = __shfl_sync(0xffffffffU, scale_bits, group_leader);
    zero_word = __shfl_sync(0xffffffffU, zero_word, group_leader);
    const float scale = __bfloat162float(
        reinterpret_cast<const __nv_bfloat16*>(&scale_bits)[0]);
    const int zero = w4_offset_binary(zero_word, row & 7);
    const uint32_t weight_word = static_cast<uint32_t>(
        packed[static_cast<int64_t>(row) * packed_cols + packed_col]);
#pragma unroll
    for (int token = 0; token < kMaxTokens; ++token) {
      if (token < tokens) {
        const uint4 activation_words = reinterpret_cast<const uint4*>(
            activation + static_cast<int64_t>(token) * input_dim)[packed_col];
        const __nv_bfloat16* activation_values =
            reinterpret_cast<const __nv_bfloat16*>(&activation_words);
#pragma unroll
        for (int index = 0; index < 8; ++index) {
          const int quantized = w4_offset_binary(weight_word, index);
          const float weight = static_cast<float>(quantized - zero) * scale;
          accumulator[token] = fmaf(
              __bfloat162float(activation_values[index]), weight,
              accumulator[token]);
        }
      }
    }
  }
#pragma unroll
  for (int token = 0; token < kMaxTokens; ++token) {
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1)
      accumulator[token] +=
          __shfl_down_sync(0xffffffffU, accumulator[token], delta);
    if (lane == 0 && token < tokens)
      output[static_cast<int64_t>(token) * output_dim + row] =
          __float2bfloat16(accumulator[token]);
  }
}
