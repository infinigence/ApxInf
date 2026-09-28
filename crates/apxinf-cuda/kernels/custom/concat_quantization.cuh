#pragma once

// Copyright 2026 apxinf contributors.
// Exact fusion of row concatenation with calibrated BF16-to-E4M3 quantization.

__global__ void concat_rows_quantize_bf16_e4m3_kernel(
    const __nv_bfloat16* first, const __nv_bfloat16* second,
    __nv_fp8_e4m3* output, int64_t first_count, int64_t total_count,
    float inverse_scale) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < total_count; index += stride) {
    const __nv_bfloat16 input =
        index < first_count ? first[index] : second[index - first_count];
    const float value = fminf(
        448.0f,
        fmaxf(-448.0f, __bfloat162float(input) * inverse_scale));
    output[index] = static_cast<__nv_fp8_e4m3>(value);
  }
}
