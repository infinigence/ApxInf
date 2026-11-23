#include "../../kernels/attention/fixed_bf16.cuh"

extern "C" cudaError_t apxinf_attention_row_softmax_f32_bf16(
    const void* input, void* output, uint32_t cols, uint32_t rows,
    cudaStream_t stream) {
  if (input == nullptr || output == nullptr || cols == 0 || rows == 0) {
    return cudaErrorInvalidValue;
  }
  row_softmax_f32_bf16_kernel<<<rows, 256, 0, stream>>>(
      static_cast<const float*>(input), static_cast<__nv_bfloat16*>(output),
      cols, rows);
  return cudaGetLastError();
}
extern "C" cudaError_t apxinf_attention_row_softmax_causal_f32(
    const void* input, void* output, uint32_t cols, uint32_t rows,
    uint32_t kv_offset, uint32_t n_heads, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || cols == 0 || rows == 0 ||
      n_heads == 0 || n_heads > rows) {
    return cudaErrorInvalidValue;
  }
  row_softmax_causal_f32_kernel<<<rows, 256, 0, stream>>>(
      static_cast<const float*>(input), static_cast<float*>(output), cols, rows,
      kv_offset, n_heads);
  return cudaGetLastError();
}
