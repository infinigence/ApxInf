// Copyright 2026 ApxInf contributors.
#include "../../include/apxinf_cuda/rope_types.h"
#include "../../kernels/rope/interleaved.cuh"

extern "C" cudaError_t apxinf_rope_apply_interleaved_table(
    const void* input, const void* cos_values, const void* sin_values,
    void* output, int64_t tokens, int64_t time, int64_t heads, int64_t head_dim,
    uint32_t dtype, cudaStream_t stream) {
  switch (dtype) {
    case APXINF_DTYPE_F32:
      return launch_rope_apply_interleaved<float>(
          input, cos_values, sin_values, output, tokens, time, heads, head_dim,
          stream);
    case APXINF_DTYPE_F16:
      return launch_rope_apply_interleaved<__half>(
          input, cos_values, sin_values, output, tokens, time, heads, head_dim,
          stream);
    case APXINF_DTYPE_BF16:
      return launch_rope_apply_interleaved<__nv_bfloat16>(
          input, cos_values, sin_values, output, tokens, time, heads, head_dim,
          stream);
    default:
      return cudaErrorInvalidValue;
  }
}
