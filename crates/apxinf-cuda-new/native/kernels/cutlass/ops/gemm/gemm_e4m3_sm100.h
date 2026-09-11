// Copyright 2026 ApxInf contributors.
#pragma once

#include <cuda_runtime_api.h>

namespace apxinf::cuda::cutlass_ops {

int fp8_gemm_f16(
    const void* activation, const void* weight, void* output,
    int m, int n, int k, float alpha, int tactic, cudaStream_t stream);

int fp8_rowwise_gemm_bf16(
    const void* activation, const void* weight_nk,
    const float* activation_scales, const float* weight_scales,
    const void* bias, void* output, int m, int n, int k, int tactic,
    cudaStream_t stream);

int fp8_gemm_geglu_e4m3(
    const void* activation, const void* packed_weight, const void* gate,
    void* output, int m, int n, int k, int full_n, float alpha,
    float output_scale, int tactic, cudaStream_t stream);

namespace fp8_dual_geglu_detail {

int production_dual_geglu(
    const void* activation, const void* interleaved_weight, void* output,
    int m, int n, int k, int full_n, float alpha, float output_scale,
    cudaStream_t stream);

}  // namespace fp8_dual_geglu_detail
}  // namespace apxinf::cuda::cutlass_ops
