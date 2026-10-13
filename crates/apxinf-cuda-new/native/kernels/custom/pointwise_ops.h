// Copyright 2026 ApxInf contributors.
#pragma once

#include <cuda_runtime_api.h>

namespace apxinf::cuda_new::pointwise_ops {

// The pointwise family's non-templated members.
//
// The Spec-dispatched semantics of this family live in
// kernels/custom/pointwise.cuh and launch from adapters/pointwise.cu. These two
// are BF16-only with a single implementation each, so they keep flat parameters
// per doc/cuda-new-design.md section 5 and launch from here.

// y[r,c] = silu(fused[r,c]) * fused[r,width+c]
//
// `fused_gate_up` is [rows, 2*width] with gate first, which is what one fused
// gate/up GEMM produces. This is SwiGLU, not the GELU-based `gemm_geglu`.
int swiglu_bf16(const void* fused_gate_up, void* output, int rows, int width,
                cudaStream_t stream);

// accumulator += addend, elementwise. Residual connections.
int add_bf16(const void* addend, void* accumulator, long long count,
             cudaStream_t stream);

}  // namespace apxinf::cuda_new::pointwise_ops
