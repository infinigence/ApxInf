// Copyright 2026 ApxInf contributors.
#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>

namespace apxinf::cuda_new::sampling {

int argmax_bf16(const void* logits, int32_t* index, int count,
                cudaStream_t stream);

}  // namespace apxinf::cuda_new::sampling
