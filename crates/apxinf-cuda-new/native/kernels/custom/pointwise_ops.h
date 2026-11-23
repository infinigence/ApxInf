// Copyright 2026 ApxInf contributors.
#pragma once

#include <cuda_runtime_api.h>

namespace apxinf::cuda_new::pointwise {

int swiglu(const void* fused_gate_up, void* output, int rows, int width,
           cudaStream_t stream);
int add_into(const void* addend, void* accumulator, long long count,
             cudaStream_t stream);

}  // namespace apxinf::cuda_new::pointwise
