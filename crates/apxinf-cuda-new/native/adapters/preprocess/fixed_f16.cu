#include "../../kernels/preprocess/fixed_f16.cuh"

extern "C" cudaError_t apxinf_preprocess_pixel_shuffle_4_bf16(
    const void* input, void* output, int views, int tokens_per_view, int width,
    cudaStream_t stream) {
  if (views <= 0 || tokens_per_view <= 0 || tokens_per_view % 16 != 0 || width <= 0)
    return cudaErrorInvalidValue;
  const int64_t count = static_cast<int64_t>(views) * (tokens_per_view / 16) *
                        static_cast<int64_t>(width) * 16;
  pixel_shuffle_4_bf16_kernel<<<blocks_for(count), kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<__nv_bfloat16*>(output), views, tokens_per_view, width);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_preprocess_pixel_shuffle_4_f16(
    const void* input, void* output, int views, int tokens_per_view, int width,
    cudaStream_t stream) {
  if (views <= 0 || tokens_per_view <= 0 || tokens_per_view % 16 != 0 || width <= 0)
    return cudaErrorInvalidValue;
  const int64_t count = static_cast<int64_t>(views) * (tokens_per_view / 16) *
                        static_cast<int64_t>(width) * 16;
  pixel_shuffle_4_f16_kernel<<<blocks_for(count), kThreads, 0, stream>>>(
      static_cast<const half*>(input), static_cast<half*>(output),
      views, tokens_per_view, width);
  return cudaGetLastError();
}
