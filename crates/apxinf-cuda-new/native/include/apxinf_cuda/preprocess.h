#pragma once

#include "types.h"
#include "status.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Temporal-merged BF16 image preprocessing (walloss family). */

apxinf_status_t apxinf_preprocess_temporal_merged_patches_bf16(
    const void* images, void* patches, int32_t views, int32_t image_size,
    int32_t patch_size, int32_t temporal_patch_size, int32_t merge_size,
    int32_t nhwc, double rescale_factor, float mean0, float mean1, float mean2,
    float std0, float std1, float std2, apxinf_cuda_stream_t stream);

#ifdef __cplusplus
}
#endif
