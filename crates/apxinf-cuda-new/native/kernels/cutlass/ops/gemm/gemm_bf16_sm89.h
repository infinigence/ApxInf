// Copyright 2026 ApxInf contributors.
#pragma once

#include <cuda_runtime_api.h>

namespace apxinf::cuda::cutlass_ops::bf16_sm89_detail {

int interleaved_geglu(
    const void* activation, const void* interleaved_weight, void* output,
    int m, int n, int k, int full_n, int tactic, cudaStream_t stream);

}  // namespace apxinf::cuda::cutlass_ops::bf16_sm89_detail
