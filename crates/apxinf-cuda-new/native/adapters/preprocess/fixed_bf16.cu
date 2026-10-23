// Copyright 2026 ApxInf contributors.
// Fixed implementation for this operator family; ported without numerical changes.
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace {
#include "../../kernels/custom/gdn_policy.h"
#include "../../kernels/custom/math.cuh"
#include "../../kernels/custom/reduction.cuh"
#include "../../kernels/custom/pillow_bicubic_u8.cuh"
#include "../../kernels/custom/linear_attention.cuh"
#include "../../kernels/custom/gdn_raw_inverse_f1.cuh"
#include "../../kernels/custom/gdn_chunk_state_wmma.cuh"
#include "../../kernels/custom/gdn_chunk_gemm_tri.cuh"
__global__ void rgb_u8_to_temporal2_merge2_rect_bf16_kernel(
    const uint8_t* __restrict__ rgb, uint16_t* __restrict__ patches,
    const uint16_t* __restrict__ lut, int grid_h, int grid_w) {
  constexpr int kPatch = 16;
  constexpr int kArea = 256;
  constexpr int kRowWidth = 3 * 2 * kArea;
  const int row = blockIdx.x;
  int rem = row;
  const int merge_x = rem & 1;
  rem >>= 1;
  const int merge_y = rem & 1;
  rem >>= 1;
  const int groups_w = grid_w / 2;
  const int group_x = rem % groups_w;
  const int group_y = rem / groups_w;
  const int patch_y = group_y * 2 + merge_y;
  const int patch_x = group_x * 2 + merge_x;
  const int dy = threadIdx.x / kPatch;
  const int dx = threadIdx.x % kPatch;
  const int64_t pixel = ((static_cast<int64_t>(patch_y) * kPatch + dy) *
                         (static_cast<int64_t>(grid_w) * kPatch) +
                         static_cast<int64_t>(patch_x) * kPatch + dx) * 3;
  const int64_t base = static_cast<int64_t>(row) * kRowWidth + threadIdx.x;
#pragma unroll
  for (int channel = 0; channel < 3; ++channel) {
    const uint16_t normalized = lut[rgb[pixel + channel]];
    patches[base + channel * 2 * kArea] = normalized;
    patches[base + channel * 2 * kArea + kArea] = normalized;
  }
}

}  // namespace

extern "C" cudaError_t apxinf_hybrid_rgb_u8_to_temporal2_merge2_rect_bf16(
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

extern "C" cudaError_t apxinf_hybrid_pillow_bicubic_u8_axis(
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
