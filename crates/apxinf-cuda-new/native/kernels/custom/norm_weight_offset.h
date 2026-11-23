// Copyright 2026 ApxInf contributors.
#pragma once

#include <cuda_runtime_api.h>

namespace apxinf::cuda_new::norm {

int rms_norm_weight_offset(const void* input, const void* weight, void* output,
                           int rows, int width, float epsilon,
                           cudaStream_t stream);

}  // namespace apxinf::cuda_new::norm
