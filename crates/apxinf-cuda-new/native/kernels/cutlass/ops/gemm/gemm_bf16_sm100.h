// Copyright 2026 ApxInf contributors.
#pragma once

#include <cuda_runtime_api.h>

namespace apxinf::cuda::cutlass_ops {

int bf16_gemm_geglu(
    const void* activation, const void* packed_weight, const void* gate,
    void* output, int m, int n, int k, int full_n, int tactic,
    cudaStream_t stream);

namespace bf16_dual_geglu_detail {

int production_dual_geglu_bf16(
    const void* activation, const void* interleaved_weight, void* output,
    int m, int n, int k, int full_n, cudaStream_t stream);

}  // namespace bf16_dual_geglu_detail
}  // namespace apxinf::cuda::cutlass_ops
