// Copyright 2026 ApxInf contributors.
#pragma once

#include <cuda_runtime_api.h>

namespace apxinf::cuda_new::norm_head {

// Per-head RMSNorm in place over `[rows, head_dim]`, for q_norm and k_norm.
//
// A member of the norm family, but not of its Spec, for two reasons: it writes
// its input in place where every Spec pipeline writes a separate output, and it
// reduces over the last axis of a `[tokens, heads, head_dim]` activation that
// the Spec's `[rows, cols]` shape cannot name without folding heads into rows.
//
// Its affine is `(1 + weight)`, the same form as AFFINE_WEIGHT_OFFSET, but it
// is a third reduction tree again -- a warp shuffle into 32 floats of static
// shared memory, summed serially by thread 0. Not bit-comparable with either
// norm.cuh or norm_offset.cu.
int head_rms_norm(void* data, const void* weight, int rows, int head_dim,
                  float epsilon, cudaStream_t stream);

}  // namespace apxinf::cuda_new::norm_head
