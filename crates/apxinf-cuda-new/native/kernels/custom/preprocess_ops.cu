// Copyright 2026 ApxInf contributors.
//
// Ported bit-identically from the legacy
// rgb_u8_to_normalized_temporal_merged_patches_bf16_kernel
// (crates/apxinf-cuda/kernels/custom/preprocess.cuh): float64 rescale with a
// single round to float32, float32 mean/std normalization, one final round
// to BF16.

#include "preprocess_ops.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <limits>

namespace apxinf::cuda_new::preprocess_ops {
namespace {

template <bool kNhwc>
__global__ void rgb_u8_to_normalized_temporal_merged_patches_bf16_kernel(
    const uint8_t* images, __nv_bfloat16* patches, int views, int image_size,
    int patch_size, int temporal_patch_size, int merge_size,
    double rescale_factor, float mean0, float mean1, float mean2, float std0,
    float std1, float std2) {
  const int grid_size = image_size / patch_size;
  const int groups_per_side = grid_size / merge_size;
  const int rows_per_view = grid_size * grid_size;
  const int patch_area = patch_size * patch_size;
  const int patch_width = 3 * temporal_patch_size * patch_area;
  const int64_t count =
      static_cast<int64_t>(views) * rows_per_view * patch_width;
  int64_t output_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t output_stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

  for (; output_index < count; output_index += output_stride) {
    int patch_element = static_cast<int>(output_index % patch_width);
    int row = static_cast<int>(output_index / patch_width);
    const int view = row / rows_per_view;
    row -= view * rows_per_view;

    const int merge_x = row % merge_size;
    row /= merge_size;
    const int merge_y = row % merge_size;
    row /= merge_size;
    const int group_x = row % groups_per_side;
    const int group_y = row / groups_per_side;

    const int dx = patch_element % patch_size;
    patch_element /= patch_size;
    const int dy = patch_element % patch_size;
    patch_element /= patch_size;
    // A still image is repeated along temporal-merged's temporal patch dimension.
    patch_element /= temporal_patch_size;
    const int channel = patch_element;

    const int y = (group_y * merge_size + merge_y) * patch_size + dy;
    const int x = (group_x * merge_size + merge_x) * patch_size + dx;
    const int64_t input_index = kNhwc
        ? ((static_cast<int64_t>(view) * image_size + y) * image_size + x) * 3 + channel
        : ((static_cast<int64_t>(view) * 3 + channel) * image_size + y) * image_size + x;

    const float mean = channel == 0 ? mean0 : (channel == 1 ? mean1 : mean2);
    const float std = channel == 0 ? std0 : (channel == 1 ? std1 : std2);
    // Transformers rescales in float64, then rounds to float32 before normalization.
    const float scaled = __double2float_rn(
        __dmul_rn(static_cast<double>(images[input_index]), rescale_factor));
    // Match Transformers' float32 normalization boundary: subtract the
    // float32 channel mean, divide by the float32 channel std, then round once
    // to BF16 for the model input.
    const float normalized = __fdiv_rn(__fsub_rn(scaled, mean), std);
    patches[output_index] = __float2bfloat16(normalized);
  }
}

}  // namespace

int rgb_u8_to_normalized_temporal_merged_patches_bf16(
    const void* images, void* patches, int views, int image_size,
    int patch_size, int temporal_patch_size, int merge_size, bool nhwc,
    double rescale_factor, const float mean[3], const float std[3],
    cudaStream_t stream) {
  if (images == nullptr || patches == nullptr || views <= 0 ||
      image_size <= 0 || patch_size <= 0 || temporal_patch_size <= 0 ||
      merge_size <= 0 || !(rescale_factor > 0.0)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  constexpr int64_t kMaxKernelInt = std::numeric_limits<int>::max();
  constexpr int64_t kMaxKernelIndex = std::numeric_limits<int64_t>::max();
  const int64_t patch_merge = static_cast<int64_t>(patch_size) * merge_size;
  if (patch_merge > image_size || image_size % patch_merge != 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const int64_t grid_size64 = image_size / patch_size;
  const int64_t rows_per_view64 = grid_size64 * grid_size64;
  const int64_t patch_area64 = static_cast<int64_t>(patch_size) * patch_size;
  if (rows_per_view64 > kMaxKernelInt || patch_area64 > kMaxKernelInt) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const int64_t three_patch_area64 = 3 * patch_area64;
  if (temporal_patch_size > kMaxKernelInt / three_patch_area64 ||
      views > kMaxKernelInt / rows_per_view64) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const int64_t patch_width64 =
      static_cast<int64_t>(temporal_patch_size) * three_patch_area64;
  const int64_t patch_rows64 = static_cast<int64_t>(views) * rows_per_view64;
  if (patch_rows64 > kMaxKernelIndex / patch_width64) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const int64_t count = patch_rows64 * patch_width64;
  constexpr int threads = 256;
  // Avoid overflowing at the accepted int64_t element-count boundary.
  const int64_t requested_blocks =
      count / threads + (count % threads != 0 ? 1 : 0);
  const int blocks =
      static_cast<int>(requested_blocks > 1024 ? 1024 : requested_blocks);
  if (nhwc) {
    rgb_u8_to_normalized_temporal_merged_patches_bf16_kernel<true>
        <<<blocks, threads, 0, stream>>>(
            static_cast<const uint8_t*>(images),
            static_cast<__nv_bfloat16*>(patches), views, image_size,
            patch_size, temporal_patch_size, merge_size, rescale_factor,
            mean[0], mean[1], mean[2], std[0], std[1], std[2]);
  } else {
    rgb_u8_to_normalized_temporal_merged_patches_bf16_kernel<false>
        <<<blocks, threads, 0, stream>>>(
            static_cast<const uint8_t*>(images),
            static_cast<__nv_bfloat16*>(patches), views, image_size,
            patch_size, temporal_patch_size, merge_size, rescale_factor,
            mean[0], mean[1], mean[2], std[0], std[1], std[2]);
  }
  return static_cast<int>(cudaGetLastError());
}

}  // namespace apxinf::cuda_new::preprocess_ops
