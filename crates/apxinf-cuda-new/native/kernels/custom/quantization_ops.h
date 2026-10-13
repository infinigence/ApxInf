// Copyright 2026 ApxInf contributors.
#pragma once

#include <cuda_runtime_api.h>

namespace apxinf::cuda_new::quantization_ops {

// The quantization family's non-templated per-tensor FP8 member.
//
// The Spec-dispatched semantics of this family (including
// QuantizationSemantic::FixedScaleE4m3, which computes the same arithmetic over
// a [rows, cols] Spec) live in kernels/primitives/quantization.cuh and launch
// from adapters/quantization.cu. This one is kept separate on purpose: its
// default kernel is 8-wide (uint4 loads, uint64 stores) where the family's BF16
// path is scalar, and it maps NaN to the FP8 NaN encoding where the family's
// clamp maps NaN to -448. Merging the two is a measured change, not a move.

// Quantize BF16 to E4M3 against a single per-tensor scale.
//
// This is the activation side of the FP8 projections in a ModelOpt checkpoint,
// whose weight_scale and input_scale are scalars rather than the [M]/[N]
// vectors the scaled-FP8 GEMM contract expects. Quantizing against
// input_scale lets the projection run as unit-scale FP8 with
// `alpha = weight_scale * input_scale`, so no new quantization contract is
// needed for attention or GDN.
int quantize_fp8_per_tensor(const void* input, void* output, long long count,
                            float input_scale, cudaStream_t stream);

}  // namespace apxinf::cuda_new::quantization_ops
