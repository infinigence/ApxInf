// Copyright 2026 ApxInf contributors.
// Fixed launcher for the device-position KV token write; the kernel lives in
// `kernels/cache/write_token.cuh`. F16 and BF16 both store one 16-bit word
// per element, so the copy is dtype-agnostic; the Rust wrapper restricts
// callers to those two.
#include "../../kernels/cache/write_token.cuh"

extern "C" cudaError_t apxinf_cache_write_token(
    const void* input, void* cache, const void* meta, int64_t capacity,
    int64_t elements, cudaStream_t stream) {
  if (input == nullptr || cache == nullptr || meta == nullptr ||
      capacity <= 0 || elements <= 0 || capacity > INT32_MAX) {
    return cudaErrorInvalidValue;
  }
  const int64_t blocks = (elements + kTokenWriteThreads - 1) / kTokenWriteThreads;
  if (blocks > INT32_MAX) return cudaErrorInvalidValue;
  cache_write_token_kernel<<<static_cast<int>(blocks), kTokenWriteThreads, 0, stream>>>(
      static_cast<const uint16_t*>(input), static_cast<uint16_t*>(cache),
      static_cast<const uint32_t*>(meta), capacity, elements);
  return cudaGetLastError();
}
