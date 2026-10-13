// Copyright 2026 ApxInf contributors.
#pragma once

#include <cuda_runtime_api.h>

namespace apxinf::cuda_new::rope_partial {

// Partial rotary embedding over `[tokens, heads, head_dim]` BF16, in place.
//
// Only the first `rotary_dim` elements of each head rotate
// (partial_rotary_factor 0.25 over head_dim 256 leaves 192 untouched).
// `positions` is `[tokens]` int32.
//
// This is the text-only form: with a single position id per token, the
// config's mrope_section [11, 11, 10] split rotates every section by the same
// angle, so it collapses to plain partial RoPE. It stops being correct once
// image or video tokens carry distinct temporal/height/width positions.
//
// A member of the rope family, but not of its Spec: the Spec-dispatched
// semantics in adapters/rope.cu fuse a QKV split with the rotation and write
// three outputs, while this one rotates a single tensor in place. See
// rope_partial.cu for the pairing convention, which is part of the contract.
int partial_rope(void* data, const void* positions, int tokens, int heads,
                 int head_dim, int rotary_dim, float theta,
                 cudaStream_t stream);

}  // namespace apxinf::cuda_new::rope_partial
