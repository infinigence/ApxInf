// Copyright 2026 ApxInf contributors.
#pragma once

#include <cuda_runtime_api.h>

namespace apxinf::cuda_new::norm_offset {

// y[r,c] = x[r,c] / sqrt(mean(x[r,:]^2) + epsilon) * (1 + weight[c])
//
// The `(1 + weight)` affine of the norm family, i.e. the Qwen3.8 RMSNorm.
// Reduction accumulates in f32: summing thousands of squared BF16 values in
// BF16 drops the small terms outright.
//
// This is not a variant of the `norm.cuh` RMSNorm with a flag. Its block size
// follows `width` (up to 1024 threads against that family's fixed 256) and it
// reduces through its own warp tree into dynamic shared memory, so the two
// sum in different orders and are not bit-comparable beyond the `1 +`. See
// norm_offset.cu for why the code lives in its own translation unit.
//
// Returns the launch status directly, and consumes it: this function calls
// cudaGetLastError() itself, so a caller's own check would see success.
cudaError_t rms_norm_bf16(const void* input, const void* weight, void* output,
                          int rows, int width, float epsilon,
                          cudaStream_t stream);

}  // namespace apxinf::cuda_new::norm_offset
