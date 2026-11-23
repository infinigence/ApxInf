#include "../../kernels/quantization/fixed_bf16.cuh"

extern "C" cudaError_t apxinf_quantization_quantize_bf16_e4m3_packed8(
    const void* input, void* output, int64_t count, float scale,
    cudaStream_t stream) {
  if (input == nullptr || output == nullptr || count <= 0 || !(scale > 0.0f))
    return cudaErrorInvalidValue;
  constexpr int threads = 256;
  const bool aligned =
      (reinterpret_cast<uintptr_t>(input) % alignof(Bf16Pack8) == 0) &&
      (reinterpret_cast<uintptr_t>(output) % alignof(Fp8Pack8) == 0);
  if (!aligned || count < 8) {
    int blocks = static_cast<int>((count + threads - 1) / threads);
    blocks = blocks > 4096 ? 4096 : blocks;
    quantize_bf16_e4m3_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input),
        static_cast<__nv_fp8_e4m3*>(output), count, 1.0f / scale);
    return cudaGetLastError();
  }
  const int64_t vector_count = count / 8;
  int blocks = static_cast<int>((vector_count + threads - 1) / threads);
  blocks = blocks > 4096 ? 4096 : blocks;
  quantize_bf16_e4m3_packed8_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const Bf16Pack8*>(input),
      static_cast<Fp8Pack8*>(output), vector_count, 1.0f / scale);
  const int64_t tail = count - vector_count * 8;
  if (tail != 0) {
    quantize_bf16_e4m3_kernel<<<1, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input) + vector_count * 8,
        static_cast<__nv_fp8_e4m3*>(output) + vector_count * 8, tail,
        1.0f / scale);
  }
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_quantization_concat_rows_quantize_bf16_e4m3(
    const void* first, const void* second, void* output, int first_rows,
    int second_rows, int cols, float scale, cudaStream_t stream) {
  if (first == nullptr || second == nullptr || output == nullptr ||
      first_rows <= 0 || second_rows <= 0 || cols <= 0 || !(scale > 0.0f))
    return cudaErrorInvalidValue;
  const int64_t first_count = static_cast<int64_t>(first_rows) * cols;
  const int64_t total_count =
      static_cast<int64_t>(first_rows + second_rows) * cols;
  int blocks = static_cast<int>((total_count + 255) / 256);
  blocks = blocks > 4096 ? 4096 : blocks;
  concat_rows_quantize_bf16_e4m3_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(first),
      static_cast<const __nv_bfloat16*>(second),
      static_cast<__nv_fp8_e4m3*>(output), first_count, total_count,
      1.0f / scale);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_quantization_dequantize_int32_bf16(
    const void* accumulators, const void* row_scales,
    const void* column_scales, void* output, int rows, int cols,
    cudaStream_t stream) {
  if (accumulators == nullptr || row_scales == nullptr ||
      column_scales == nullptr || output == nullptr || rows <= 0 || cols <= 0)
    return cudaErrorInvalidValue;
  const dim3 grid((cols + kBlockSize - 1) / kBlockSize, rows);
  dequantize_int32_bf16_kernel<<<grid, kBlockSize, 0, stream>>>(
      static_cast<const int32_t*>(accumulators),
      static_cast<const float*>(row_scales),
      static_cast<const float*>(column_scales),
      static_cast<__nv_bfloat16*>(output), rows, cols);
  return cudaGetLastError();
}

// ── W8A8 (INT8) quantization launchers ───────────────────────────────────
// Copied from the legacy w8a8_adapter.cu. These produce the dynamically
// row-quantized INT8 activations the W8A8 GEMM consumes.

extern "C" cudaError_t apxinf_quantization_quantize_rows_bf16_int8(
    const void* input, void* output, void* scales, int rows, int cols,
    cudaStream_t stream) {
  if (input == nullptr || output == nullptr || scales == nullptr ||
      rows <= 0 || cols <= 0)
    return cudaErrorInvalidValue;
  quantize_rows_bf16_int8_kernel<<<rows, kBlockSize, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input), static_cast<int8_t*>(output),
      static_cast<float*>(scales), rows, cols);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_quantization_quantize_rows_bf16_int8_packed4(
    const void* input, void* output, void* scales, int rows, int cols,
    cudaStream_t stream) {
  if (input == nullptr || output == nullptr || scales == nullptr ||
      rows <= 0 || cols <= 0 || cols % 4 != 0)
    return cudaErrorInvalidValue;
  quantize_rows_bf16_int8_vec4_kernel<<<
      rows, 512, static_cast<size_t>(cols) * sizeof(__nv_bfloat16), stream>>>(
      static_cast<const __nv_bfloat16*>(input), static_cast<int8_t*>(output),
      static_cast<float*>(scales), rows, cols);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_quantization_bias_gelu_quantize_rows_bf16_int8(
    const void* input, const void* bias, void* output, void* scales, int rows,
    int cols, cudaStream_t stream) {
  if (input == nullptr || bias == nullptr || output == nullptr ||
      scales == nullptr || rows <= 0 || cols <= 0)
    return cudaErrorInvalidValue;
  if (cols % 4 == 0) {
    bias_gelu_quantize_rows_bf16_int8_vec4_kernel<<<
        rows, 512, static_cast<size_t>(cols) * sizeof(__nv_bfloat16), stream>>>(
        static_cast<const __nv_bfloat16*>(input),
        static_cast<const __nv_bfloat16*>(bias), static_cast<int8_t*>(output),
        static_cast<float*>(scales), rows, cols);
  } else {
    bias_gelu_quantize_rows_bf16_int8_kernel<<<
        rows, kBlockSize,
        static_cast<size_t>(cols) * sizeof(__nv_bfloat16), stream>>>(
        static_cast<const __nv_bfloat16*>(input),
        static_cast<const __nv_bfloat16*>(bias), static_cast<int8_t*>(output),
        static_cast<float*>(scales), rows, cols);
  }
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_quantization_silu_mul_quantize_rows_bf16_int8(
    const void* gate, const void* up, void* output, void* scales, int rows,
    int cols, cudaStream_t stream) {
  if (gate == nullptr || up == nullptr || output == nullptr ||
      scales == nullptr || rows <= 0 || cols <= 0)
    return cudaErrorInvalidValue;
  silu_mul_quantize_rows_bf16_int8_kernel<<<
      rows, kBlockSize,
      static_cast<size_t>(cols) * sizeof(__nv_bfloat16), stream>>>(
      static_cast<const __nv_bfloat16*>(gate),
      static_cast<const __nv_bfloat16*>(up), static_cast<int8_t*>(output),
      static_cast<float*>(scales), rows, cols);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_quantization_silu_mul_quantize_rows_bf16_int8_packed4(
    const void* gate, const void* up, void* output, void* scales, int rows,
    int cols, cudaStream_t stream) {
  if (gate == nullptr || up == nullptr || output == nullptr ||
      scales == nullptr || rows <= 0 || cols <= 0 || cols % 4 != 0 ||
      (reinterpret_cast<uintptr_t>(gate) & 7) != 0 ||
      (reinterpret_cast<uintptr_t>(up) & 7) != 0 ||
      (reinterpret_cast<uintptr_t>(output) & 3) != 0)
    return cudaErrorInvalidValue;
  silu_mul_quantize_rows_bf16_int8_vec4_kernel<<<
      rows, 512, static_cast<size_t>(cols) * sizeof(__nv_bfloat16), stream>>>(
      static_cast<const __nv_bfloat16*>(gate),
      static_cast<const __nv_bfloat16*>(up), static_cast<int8_t*>(output),
      static_cast<float*>(scales), rows, cols);
  return cudaGetLastError();
}
