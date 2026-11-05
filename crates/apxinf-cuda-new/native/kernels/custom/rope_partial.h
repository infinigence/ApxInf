// Copyright 2026 ApxInf contributors.
#pragma once

#include <cuda_runtime_api.h>

namespace apxinf::cuda_new::rope {

int partial_rope(void* data, const void* positions, int tokens, int heads,
                 int head_dim, int rotary_dim, float theta,
                 cudaStream_t stream);

}  // namespace apxinf::cuda_new::rope
