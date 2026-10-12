// Copyright 2026 ApxInf contributors.
#pragma once

#include <cuda_runtime_api.h>

// Image preprocessing ported bit-identically from the legacy walloss kernel:
// u8 RGB to normalized temporal-merged BF16 patches (the Qwen2-VL-style
// patchification with temporal repetition and spatial merge reordering).
namespace apxinf::cuda_new::preprocess_ops {

// patches[row, elem] = bf16((f32(u8 * rescale) - mean_c) / std_c), with the
// still image repeated along the temporal patch dimension and rows ordered
// by (view, merge group, merge offset). `nhwc` selects the input layout.
// Rescaling happens in float64 then rounds to float32, matching Transformers.
int rgb_u8_to_normalized_temporal_merged_patches_bf16(
    const void* images, void* patches, int views, int image_size,
    int patch_size, int temporal_patch_size, int merge_size, bool nhwc,
    double rescale_factor, const float mean[3], const float std[3],
    cudaStream_t stream);

}  // namespace apxinf::cuda_new::preprocess_ops
