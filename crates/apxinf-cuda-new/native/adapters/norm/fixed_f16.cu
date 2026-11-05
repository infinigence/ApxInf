#include "../../kernels/norm/fixed_f16.cuh"

extern "C" cudaError_t apxinf_norm_rms_norm_f16(
    const void* input, const void* weight, void* output,
    int rows, int cols, float eps, cudaStream_t stream) {
  if (input == nullptr || weight == nullptr || output == nullptr ||
      rows <= 0 || cols <= 0 || !(eps > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  const size_t shared_bytes = static_cast<size_t>(cols) * sizeof(float);
  rms_norm_f16_kernel<<<rows, kThreads, shared_bytes, stream>>>(
      static_cast<const half*>(input), static_cast<const half*>(weight),
      static_cast<half*>(output), cols, rows, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_norm_layer_norm_f16(
    const void* input, const void* weight, const void* bias, void* output,
    int rows, int cols, float eps, cudaStream_t stream) {
  if (input == nullptr || weight == nullptr || bias == nullptr ||
      output == nullptr || rows <= 0 || cols <= 0 || !(eps > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  layer_norm_f16_kernel<<<rows, kThreads, 0, stream>>>(
      static_cast<const half*>(input), static_cast<const half*>(weight),
      static_cast<const half*>(bias), static_cast<half*>(output),
      rows, cols, eps);
  return cudaGetLastError();
}
