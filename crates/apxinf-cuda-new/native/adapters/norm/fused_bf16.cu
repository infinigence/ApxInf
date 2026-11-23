#include "../../kernels/norm/fused_bf16.cuh"

extern "C" cudaError_t apxinf_norm_bias_then_residual_bf16(
    const void* projection, const void* bias, const void* residual,
    void* output, int rows, int cols, cudaStream_t stream) {
  const int64_t count = static_cast<int64_t>(rows) * cols;
  const int blocks = static_cast<int>((count + kBlockSize - 1) / kBlockSize);
  bias_then_residual_bf16_kernel<<<blocks, kBlockSize, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(projection),
      static_cast<const __nv_bfloat16*>(bias),
      static_cast<const __nv_bfloat16*>(residual),
      static_cast<__nv_bfloat16*>(output), count, cols);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_norm_bias_residual_bf16_packed4(
    const void* projection, const void* bias, const void* residual,
    void* output, int rows, int cols, cudaStream_t stream) {
  if (projection == nullptr || bias == nullptr || residual == nullptr ||
      output == nullptr || rows <= 0 || cols <= 0 || (cols & 3) != 0)
    return cudaErrorInvalidValue;
  const uintptr_t pointers = reinterpret_cast<uintptr_t>(projection) |
      reinterpret_cast<uintptr_t>(residual) |
      reinterpret_cast<uintptr_t>(output) |
      reinterpret_cast<uintptr_t>(bias);
  if ((pointers & (alignof(Bf16x4) - 1)) != 0) return cudaErrorInvalidValue;
  const int64_t packed_count = static_cast<int64_t>(rows) * (cols / 4);
  const int blocks =
      static_cast<int>((packed_count + kBlockSize - 1) / kBlockSize);
  bias_residual_bf16_packed4_kernel<<<blocks, kBlockSize, 0, stream>>>(
      static_cast<const Bf16x4*>(projection), static_cast<const Bf16x4*>(bias),
      static_cast<const Bf16x4*>(residual), static_cast<Bf16x4*>(output),
      packed_count, cols / 4);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_norm_bias_then_residual_bf16_packed4(
    const void* projection, const void* bias, const void* residual,
    void* output, int rows, int cols, cudaStream_t stream) {
  if (projection == nullptr || residual == nullptr || output == nullptr ||
      rows <= 0 || cols <= 0 || (cols & 3) != 0)
    return cudaErrorInvalidValue;
  const uintptr_t pointers = reinterpret_cast<uintptr_t>(projection) |
      reinterpret_cast<uintptr_t>(residual) |
      reinterpret_cast<uintptr_t>(output) |
      reinterpret_cast<uintptr_t>(bias);
  if ((pointers & (alignof(Bf16x4) - 1)) != 0) return cudaErrorInvalidValue;
  const int64_t packed_count = static_cast<int64_t>(rows) * (cols / 4);
  const int blocks =
      static_cast<int>((packed_count + kBlockSize - 1) / kBlockSize);
  bias_then_residual_bf16_packed4_kernel<<<blocks, kBlockSize, 0, stream>>>(
      static_cast<const Bf16x4*>(projection), static_cast<const Bf16x4*>(bias),
      static_cast<const Bf16x4*>(residual), static_cast<Bf16x4*>(output),
      packed_count, cols / 4);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_norm_bias_residual_layer_norm_bf16_cached_1024(
    const void* projection, const void* projection_bias, const void* residual,
    const void* norm_weight, const void* norm_bias, void* hidden,
    void* normalized, int rows, int cols, float eps, cudaStream_t stream) {
  if (projection == nullptr || residual == nullptr || norm_weight == nullptr ||
      norm_bias == nullptr || hidden == nullptr || normalized == nullptr ||
      rows <= 0 || cols != 1024)
    return cudaErrorInvalidValue;
  bias_residual_layer_norm_bf16_cached_1024_kernel<<<rows, kBlockSize, 0,
                                                     stream>>>(
      static_cast<const __nv_bfloat16*>(projection),
      static_cast<const __nv_bfloat16*>(projection_bias),
      static_cast<const __nv_bfloat16*>(residual),
      static_cast<const __nv_bfloat16*>(norm_weight),
      static_cast<const __nv_bfloat16*>(norm_bias),
      static_cast<__nv_bfloat16*>(hidden),
      static_cast<__nv_bfloat16*>(normalized), rows, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_norm_bias_then_residual_layer_norm_bf16_cached_1536(
    const void* projection, const void* projection_bias, const void* residual,
    const void* norm_weight, const void* norm_bias, void* hidden,
    void* normalized, int rows, int cols, float eps, cudaStream_t stream) {
  if (projection == nullptr || projection_bias == nullptr ||
      residual == nullptr || norm_weight == nullptr || norm_bias == nullptr ||
      hidden == nullptr || normalized == nullptr || rows <= 0 || cols != 1536)
    return cudaErrorInvalidValue;
  bias_then_residual_layer_norm_bf16_cached_1536_kernel<<<rows, kBlockSize,
                                                          0, stream>>>(
      static_cast<const __nv_bfloat16*>(projection),
      static_cast<const __nv_bfloat16*>(projection_bias),
      static_cast<const __nv_bfloat16*>(residual),
      static_cast<const __nv_bfloat16*>(norm_weight),
      static_cast<const __nv_bfloat16*>(norm_bias),
      static_cast<__nv_bfloat16*>(hidden),
      static_cast<__nv_bfloat16*>(normalized), rows, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t
apxinf_norm_bias_then_residual_adaptive_layer_norm_bf16_cached_1536(
    const void* projection, const void* projection_bias, const void* residual,
    const void* modulation, void* hidden, void* normalized, int rows, int cols,
    float eps, cudaStream_t stream) {
  if (projection == nullptr || projection_bias == nullptr ||
      residual == nullptr || modulation == nullptr || hidden == nullptr ||
      normalized == nullptr || rows <= 0 || cols != 1536)
    return cudaErrorInvalidValue;
  bias_then_residual_adaptive_layer_norm_bf16_cached_1536_kernel
      <<<rows, kBlockSize, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(projection),
          static_cast<const __nv_bfloat16*>(projection_bias),
          static_cast<const __nv_bfloat16*>(residual),
          static_cast<const __nv_bfloat16*>(modulation),
          static_cast<__nv_bfloat16*>(hidden),
          static_cast<__nv_bfloat16*>(normalized), rows, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_norm_bias_residual_layer_norm_quant_bf16_e4m3(
    const void* projection, const void* projection_bias, const void* residual,
    const void* norm_weight, const void* norm_bias, void* hidden,
    void* normalized, int rows, int cols, float eps, float scale,
    cudaStream_t stream) {
  if (projection == nullptr || residual == nullptr || norm_weight == nullptr ||
      norm_bias == nullptr || hidden == nullptr || normalized == nullptr ||
      rows <= 0 || cols <= 0 || !std::isfinite(scale) || scale <= 0.0f)
    return cudaErrorInvalidValue;
  bias_residual_layer_norm_quant_bf16_e4m3_kernel<<<rows, kBlockSize, 0,
                                                     stream>>>(
      static_cast<const __nv_bfloat16*>(projection),
      static_cast<const __nv_bfloat16*>(projection_bias),
      static_cast<const __nv_bfloat16*>(residual),
      static_cast<const __nv_bfloat16*>(norm_weight),
      static_cast<const __nv_bfloat16*>(norm_bias),
      static_cast<__nv_bfloat16*>(hidden),
      static_cast<__nv_fp8_e4m3*>(normalized), rows, cols, eps, 1.0f / scale);
  return cudaGetLastError();
}
