// Copyright 2026 ApxInf contributors.
#pragma once

#include <cuda_runtime_api.h>

namespace apxinf::cuda_new::cutlass_ops {
struct ExactGeluExecution;
size_t exact_gelu_workspace();
size_t exact_gelu_lut_workspace();
ExactGeluExecution* prepare_exact_gelu_lut(const void*, const void*, void*, int, size_t, cudaStream_t);
ExactGeluExecution* prepare_exact_gelu(const void*, const void*, void*, int, size_t, cudaStream_t);
int enqueue_exact_gelu(ExactGeluExecution*, cudaStream_t);
void destroy_exact_gelu(ExactGeluExecution*);


int bf16_gemm_geglu(
    const void* activation, const void* packed_weight, const void* gate,
    void* output, int m, int n, int k, int full_n, int tactic,
    cudaStream_t stream);

namespace bf16_dual_geglu_detail {

int production_dual_geglu_bf16(
    const void* activation, const void* interleaved_weight, void* output,
    int m, int n, int k, int full_n, cudaStream_t stream);

// Same paired-GEMM mainloop with a SwiGLU epilogue. The implementation owns
// its validated shape constraints; model names and call-site layout do not
// enter this L0 interface.
int production_dual_swiglu_bf16(
    const void* activation, const void* interleaved_weight, void* output,
    int m, int n, int k, int full_n, cudaStream_t stream);

}  // namespace bf16_dual_geglu_detail
}  // namespace apxinf::cuda_new::cutlass_ops
