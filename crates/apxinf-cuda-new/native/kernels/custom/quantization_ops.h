// Copyright 2026 ApxInf contributors.
#pragma once

#include <cuda_runtime_api.h>

namespace apxinf::cuda_new::quantization {

int fp8_per_tensor(const void* input, void* output, long long count,
                   float input_scale, cudaStream_t stream);

}  // namespace apxinf::cuda_new::quantization
