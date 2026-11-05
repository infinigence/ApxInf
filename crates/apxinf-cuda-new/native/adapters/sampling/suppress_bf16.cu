// Copyright 2026 ApxInf contributors.
// Fixed family launcher; kernel body and launch parameters are unchanged.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace {
#include "../../kernels/sampling/suppress_bf16.cuh"
}  // namespace

extern "C" cudaError_t apxinf_sampling_suppress_logits_bf16(
    void* logits, const uint32_t* ids, int count, cudaStream_t stream) {
  if (logits == nullptr || ids == nullptr || count <= 0) {
    return cudaErrorInvalidValue;
  }
  const int blocks = (count + 63) / 64;
  suppress_logits_bf16_kernel<<<blocks, 64, 0, stream>>>(
      static_cast<__nv_bfloat16*>(logits), ids, count);
  return cudaGetLastError();
}
