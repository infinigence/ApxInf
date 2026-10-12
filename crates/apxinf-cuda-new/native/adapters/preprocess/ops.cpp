// Temporal-merged image preprocessing: direct C-ABI forwarder per
// doc/adding-new-kernels.md section 6 (single implementation, nothing to
// tune).

#include "../../include/apxinf_cuda/preprocess.h"

#include "../../framework/runtime_internal.h"
#include "../../kernels/custom/preprocess_ops.h"

#include <cmath>
#include <cstdint>
#include <string>

namespace {

using apxinf::framework::Failure;
using apxinf::framework::abi_boundary;

}  // namespace

extern "C" apxinf_status_t apxinf_preprocess_temporal_merged_patches_bf16(
    const void* images, void* patches, int32_t views, int32_t image_size,
    int32_t patch_size, int32_t temporal_patch_size, int32_t merge_size,
    int32_t nhwc, double rescale_factor, float mean0, float mean1, float mean2,
    float std0, float std1, float std2, apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (images == nullptr || patches == nullptr || views <= 0 ||
        image_size <= 0 || patch_size <= 0 || temporal_patch_size <= 0 ||
        merge_size <= 0 || (nhwc != 0 && nhwc != 1) ||
        !std::isfinite(rescale_factor) || !(rescale_factor > 0.0) ||
        !std::isfinite(mean0) || !std::isfinite(mean1) ||
        !std::isfinite(mean2) || !std::isfinite(std0) ||
        !std::isfinite(std1) || !std::isfinite(std2) || !(std0 > 0.0F) ||
        !(std1 > 0.0F) || !(std2 > 0.0F)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid temporal-merged preprocessing arguments");
    }
    const float mean[3] = {mean0, mean1, mean2};
    const float std[3] = {std0, std1, std2};
    const int status =
        apxinf::cuda_new::preprocess_ops::
            rgb_u8_to_normalized_temporal_merged_patches_bf16(
                images, patches, views, image_size, patch_size,
                temporal_patch_size, merge_size, nhwc == 1, rescale_factor,
                mean, std, static_cast<cudaStream_t>(stream));
    if (status != 0) {
      throw Failure(APXINF_STATUS_PROVIDER_ERROR,
                    "temporal-merged preprocessing failed with status " +
                        std::to_string(status));
    }
  });
}
