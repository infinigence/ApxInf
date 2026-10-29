// Copyright 2026 ApxInf contributors.
#pragma once

#include <cuda_runtime_api.h>

namespace apxinf::cuda_new::norm {

int head_rms_norm(void* data, const void* weight, int rows, int head_dim,
                  float epsilon, cudaStream_t stream);

}  // namespace apxinf::cuda_new::norm
