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
// Adapted from an upstream quantize_fp8_static FP16 path at commit
// a260835599e9406067196968cfe7a67bf8d13b4d (MIT). The scalar kernel above
// remains the fallback for unaligned buffers and the final 0..3 values.
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
