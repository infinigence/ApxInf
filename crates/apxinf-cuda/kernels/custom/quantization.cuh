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



__global__ void dequantize_w4a16_asym_bf16_kernel(
    const int32_t* weight_packed,
    const __nv_bfloat16* weight_scale,
    const int32_t* weight_zero_point,
    __nv_bfloat16* dense,
    int in_cols,
    int out_cols,
    int groups) {
  const int64_t total = static_cast<int64_t>(out_cols) * in_cols;
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  const int packed_cols = (in_cols + 7) / 8;
  const int group_size = (in_cols + groups - 1) / groups;
  for (; index < total; index += stride) {
    const int out_col = static_cast<int>(index / in_cols);
    const int in_col = static_cast<int>(index - static_cast<int64_t>(out_col) * in_cols);
    const int group = min(in_col / group_size, groups - 1);
    const uint32_t word = static_cast<uint32_t>(
        weight_packed[static_cast<int64_t>(out_col) * packed_cols + in_col / 8]);
    const int q = static_cast<int>((word >> ((in_col & 7) * 4)) & 0xFU);
    const int zp_row = out_col / 8;
    const int zp_shift = (out_col & 7) * 4;
    const uint32_t zp_word = static_cast<uint32_t>(
        weight_zero_point[static_cast<int64_t>(zp_row) * groups + group]);
    const int zp = static_cast<int>((zp_word >> zp_shift) & 0xFU);
    const float scale = __bfloat162float(
        weight_scale[static_cast<int64_t>(out_col) * groups + group]);
    dense[index] = __float2bfloat16(static_cast<float>(q - zp) * scale);
  }
}

__global__ void matmul_bf16_w4a16_asym_kernel(
    const __nv_bfloat16* activation,
    const int32_t* weight_packed,
    const __nv_bfloat16* weight_scale,
    const int32_t* weight_zero_point,
    __nv_bfloat16* output,
    int rows,
    int in_cols,
    int out_cols,
    int groups) {
  const int out_col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y;
  if (row >= rows || out_col >= out_cols) return;

  const int packed_cols = (in_cols + 7) / 8;
  const int group_size = (in_cols + groups - 1) / groups;
  const int zp_row = out_col / 8;
  const int zp_shift = (out_col & 7) * 4;
  float acc = 0.0f;
  for (int in_col = 0; in_col < in_cols; ++in_col) {
    const int group = min(in_col / group_size, groups - 1);
    const uint32_t word = static_cast<uint32_t>(
        weight_packed[static_cast<int64_t>(out_col) * packed_cols + in_col / 8]);
    const int q = static_cast<int>((word >> ((in_col & 7) * 4)) & 0xFU);
    const uint32_t zp_word = static_cast<uint32_t>(
        weight_zero_point[static_cast<int64_t>(zp_row) * groups + group]);
    const int zp = static_cast<int>((zp_word >> zp_shift) & 0xFU);
    const float scale = __bfloat162float(
        weight_scale[static_cast<int64_t>(out_col) * groups + group]);
    const float x = __bfloat162float(
        activation[static_cast<int64_t>(row) * in_cols + in_col]);
    acc += x * static_cast<float>(q - zp) * scale;
  }
  output[static_cast<int64_t>(row) * out_cols + out_col] = __float2bfloat16(acc);
}
