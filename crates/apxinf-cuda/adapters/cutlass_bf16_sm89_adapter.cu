// Copyright 2026 ApxInf contributors.
// Stable C ABI adapter for the CUTLASS SM89 BF16 GeGLU operator.

#include "../kernels/cutlass/bf16_operators_sm89.h"

extern "C" int apxinf_static_cutlass_bf16_interleaved_geglu_sm89(
    const void* activation, const void* interleaved_weight, void* output,
    int m, int n, int k, int full_n, int tactic, cudaStream_t stream) {
  return apxinf::cuda::cutlass_ops::bf16_sm89_detail::interleaved_geglu(
      activation, interleaved_weight, output, m, n, k, full_n, tactic, stream);
}
