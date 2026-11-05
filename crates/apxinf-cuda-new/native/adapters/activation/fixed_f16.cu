#include "../../kernels/activation/fixed_f16.cuh"

extern "C" cudaError_t apxinf_activation_swiglu_f16(
    const void* gate_up, void* output, int rows, int inner,
    cudaStream_t stream) {
  const int64_t count = static_cast<int64_t>(rows) * inner;
  if (gate_up == nullptr || output == nullptr || rows <= 0 || inner <= 0)
    return cudaErrorInvalidValue;
  swiglu_f16_kernel<<<blocks_for(count), kThreads, 0, stream>>>(
      static_cast<const half*>(gate_up), static_cast<half*>(output),
      rows, inner);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_activation_bias_gelu_f16(
    const void* input, const void* bias, void* output,
    int rows, int cols, cudaStream_t stream) {
  if (input == nullptr || bias == nullptr || output == nullptr ||
      rows <= 0 || cols <= 0) {
    return cudaErrorInvalidValue;
  }
  const int64_t count = static_cast<int64_t>(rows) * cols;
  bias_gelu_f16_kernel<<<blocks_for(count), kThreads, 0, stream>>>(
      static_cast<const half*>(input), static_cast<const half*>(bias),
      static_cast<half*>(output), rows, cols);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_activation_bias_silu_f16(
    const void* input, const void* bias, void* output, int rows, int cols,
    cudaStream_t stream) {
  if (rows <= 0 || cols <= 0) return cudaErrorInvalidValue;
  int64_t count = static_cast<int64_t>(rows) * cols;
  bias_silu_f16_kernel<<<clamped_blocks(count), 256, 0, stream>>>(
      static_cast<const half*>(input), static_cast<const half*>(bias),
      static_cast<half*>(output), count, cols);
  return cudaGetLastError();
}
