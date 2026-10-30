#include "../../kernels/elementwise/fixed_bf16.cuh"
#include "../../kernels/elementwise/broadcast_bf16_f32.cuh"

extern "C" cudaError_t apxinf_elementwise_broadcast_bf16_f32_rows(
    const void* bias, void* output, int rows, int cols, cudaStream_t stream) {
  if (bias == nullptr || output == nullptr || rows <= 0 || cols <= 0)
    return cudaErrorInvalidValue;
  const int64_t count = static_cast<int64_t>(rows) * cols;
  const int64_t requested_blocks = (count + 255) / 256;
  const int blocks = static_cast<int>(requested_blocks > 4096 ? 4096 : requested_blocks);
  broadcast_bf16_f32_rows_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(bias), static_cast<float*>(output), count, cols);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_elementwise_scatter_rows_bf16(
    const void* source, const void* rows, void* output,
    int row_count, int cols, int add, cudaStream_t stream) {
  if (source == nullptr || rows == nullptr || output == nullptr ||
      row_count <= 0 || cols <= 0 || (add != 0 && add != 1))
    return cudaErrorInvalidValue;
  const int64_t count = static_cast<int64_t>(row_count) * cols;
  const int blocks = static_cast<int>((count + kBlockSize - 1) / kBlockSize);
  scatter_rows_bf16_kernel<<<blocks, kBlockSize, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(source),
      static_cast<const uint32_t*>(rows),
      static_cast<__nv_bfloat16*>(output), count, cols, add != 0);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_elementwise_bias_qkv_in_place_bf16(
    void* query, void* key, void* value, const void* query_bias,
    const void* key_bias, const void* value_bias, int rows, int cols,
    cudaStream_t stream) {
  if (!query || !key || !value || !query_bias || !key_bias || !value_bias ||
      rows <= 0 || cols <= 0 || cols % 4 != 0)
    return cudaErrorInvalidValue;
  constexpr int threads = 256;
  const int64_t groups = static_cast<int64_t>(rows) * cols / 4;
  int blocks = static_cast<int>((groups + threads - 1) / threads);
  blocks = blocks > 1024 ? 1024 : blocks;
  bias_qkv_in_place_bf16_packed4_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<__nv_bfloat16*>(query),
      static_cast<__nv_bfloat16*>(key),
      static_cast<__nv_bfloat16*>(value),
      static_cast<const __nv_bfloat16*>(query_bias),
      static_cast<const __nv_bfloat16*>(key_bias),
      static_cast<const __nv_bfloat16*>(value_bias), groups, cols / 4);
  return cudaGetLastError();
}
