// Copyright 2026 apxinf contributors.
// Fixed launcher for exponential-race sampling; the kernel lives in
// `kernels/sampling/exponential_race.cuh` and runs as a single block.

#include "../../kernels/sampling/exponential_race.cuh"

// dtype: 1 f16, 2 bf16.
extern "C" cudaError_t apxinf_sampling_exponential_race(
    const void* logits, const void* seen, void* output, int64_t vocab,
    uint64_t seed, uint64_t step, uint32_t top_k, float temperature,
    float repetition, uint32_t dtype, cudaStream_t stream) {
  if (logits == nullptr || seen == nullptr || output == nullptr ||
      vocab <= 0 || vocab > kExponentialRaceMaxVocab ||
      temperature <= 0.0f || top_k > kExponentialRaceMaxVocab) {
    return cudaErrorInvalidValue;
  }
  switch (dtype) {
    case 1:
      apxinf_sampling_exponential_race_kernel<__half>
          <<<1, kExponentialRaceThreads, 0, stream>>>(
              static_cast<const __half*>(logits),
              static_cast<uint8_t*>(const_cast<void*>(seen)),
              static_cast<uint32_t*>(output), static_cast<uint32_t>(vocab),
              seed, step, top_k, temperature, repetition);
      break;
    case 2:
      apxinf_sampling_exponential_race_kernel<__nv_bfloat16>
          <<<1, kExponentialRaceThreads, 0, stream>>>(
              static_cast<const __nv_bfloat16*>(logits),
              static_cast<uint8_t*>(const_cast<void*>(seen)),
              static_cast<uint32_t*>(output), static_cast<uint32_t>(vocab),
              seed, step, top_k, temperature, repetition);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}
