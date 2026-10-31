// Copyright 2026 ApxInf contributors. SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>
namespace apxinf::cuda_new::indexed_attention {
struct Bindings {
 const void* query; const void* key; const void* value; void* output;
 uint32_t* flags; int* counts; int* indices; int* cumulative;
 float* native_lse; float* fa2_lse;
 int queries; int keys; int heads; float scale;
};
// The first implementation fixes the numerical boundary margin to 16 FP32 ULP.
// It is part of the implementation identity, not a model-visible tuning knob.
// Configuration 0 retains the original two-Q-stage one-CTA kernel; 1 uses
// one Q stage in a two-CTA cluster with a base-aware 256-column TMEM allocation.
int prepare_native(const Bindings&, void** plan, int configuration = 0);
cudaError_t enqueue_native(void* plan, cudaStream_t);
void destroy_native(void* plan) noexcept;
int prepare_mapped(const Bindings&, void** plan);
cudaError_t enqueue_mapped(void* plan, cudaStream_t);
void destroy_mapped(void* plan) noexcept;
cudaError_t compact(const Bindings&, cudaStream_t);
} // namespace apxinf::cuda_new::indexed_attention
