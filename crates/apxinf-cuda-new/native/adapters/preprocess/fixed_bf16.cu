#include "../../kernels/preprocess/fixed_bf16.cuh"

extern "C" cudaError_t apxinf_preprocess_rgb_u8_to_temporal2_merge2_rect_bf16(
    const void* rgb, void* patches, const void* lut, int grid_h, int grid_w,
    cudaStream_t stream) {
  if (!rgb || !patches || !lut || grid_h <= 0 || grid_w <= 0 ||
      (grid_h & 1) || (grid_w & 1) ||
      static_cast<int64_t>(grid_h) * grid_w > std::numeric_limits<int>::max())
    return cudaErrorInvalidValue;
  rgb_u8_to_temporal2_merge2_rect_bf16_kernel<<<grid_h * grid_w, 256, 0, stream>>>(
      static_cast<const uint8_t*>(rgb), static_cast<uint16_t*>(patches),
      static_cast<const uint16_t*>(lut), grid_h, grid_w);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_preprocess_pillow_bicubic_u8_axis(
    const void* input, void* output, const void* input_offsets,
    const void* output_offsets, const void* bounds, const void* weights,
    int ksize, int in_w, int in_h, int out_w, int out_h,
    int batch, bool horizontal, cudaStream_t stream) {
  if (!input || !output || !input_offsets || !output_offsets || !bounds || !weights ||
      ksize <= 0 || in_w <= 0 || in_h <= 0 || out_w <= 0 || out_h <= 0 ||
      (horizontal ? out_h != in_h : out_w != in_w) ||
      batch <= 0 || batch > 65535 ||
      static_cast<int64_t>(out_w) * out_h > std::numeric_limits<int>::max())
    return cudaErrorInvalidValue;
  const int64_t output_pixels = static_cast<int64_t>(out_w) * out_h;
  const int64_t blocks = (output_pixels + 255) / 256;
  if (blocks > std::numeric_limits<int>::max()) return cudaErrorInvalidValue;
  pillow_bicubic_u8_axis_kernel<<<dim3(static_cast<unsigned int>(blocks), batch), 256, 0, stream>>>(
      static_cast<const uint8_t*>(input), static_cast<uint8_t*>(output),
      static_cast<const int64_t*>(input_offsets),
      static_cast<const int64_t*>(output_offsets),
      static_cast<const int32_t*>(bounds), static_cast<const int32_t*>(weights),
      ksize, in_w, in_h, out_w, out_h, horizontal);
  return cudaGetLastError();
}
