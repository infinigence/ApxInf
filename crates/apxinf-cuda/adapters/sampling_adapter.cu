// Copyright 2026 apxinf contributors.
// Stable C ABI and launch policy for categorical sampling and device RNG.

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "../kernels/custom/sampling.cuh"

namespace {

constexpr int kThreads = 256;

template <typename T>
cudaError_t launch_prepare(
    const void* logits, const uint32_t* counts, float* adjusted,
    uint32_t* token_ids, uint32_t vocab_size, float repetition,
    float frequency, float presence, float inverse_temperature,
    cudaStream_t stream) {
  int blocks = static_cast<int>((vocab_size + kThreads - 1) / kThreads);
  blocks = std::min(blocks, 1024);
  apxinf_prepare_logits_kernel<<<blocks, kThreads, 0, stream>>>(
      static_cast<const T*>(logits), counts, adjusted, token_ids, vocab_size,
      repetition, frequency, presence, inverse_temperature);
  return cudaGetLastError();
}

}  // namespace

extern "C" cudaError_t apxinf_token_sampling_workspace_sizes(
    uint32_t vocab_size, size_t* sort_bytes, size_t* scan_bytes) {
  if (vocab_size == 0 || sort_bytes == nullptr || scan_bytes == nullptr)
    return cudaErrorInvalidValue;
  *sort_bytes = 0;
  *scan_bytes = 0;
  cudaError_t status = cub::DeviceRadixSort::SortPairsDescending(
      nullptr, *sort_bytes, static_cast<const float*>(nullptr),
      static_cast<float*>(nullptr), static_cast<const uint32_t*>(nullptr),
      static_cast<uint32_t*>(nullptr), vocab_size, 0, 32);
  if (status != cudaSuccess) return status;
  return cub::DeviceScan::InclusiveSum(
      nullptr, *scan_bytes, static_cast<const float*>(nullptr),
      static_cast<float*>(nullptr), vocab_size);
}

extern "C" cudaError_t apxinf_sample_token(
    const void* logits, int dtype, uint32_t vocab_size, uint32_t* counts,
    float repetition, float frequency, float presence, int selection,
    float temperature, uint32_t top_k, float top_p, uint64_t seed,
    uint64_t sequence, uint64_t draw, uint32_t return_logprob,
    float* adjusted, uint32_t* token_ids, float* sorted_logits,
    uint32_t* sorted_tokens, float* weights, float* cdf,
    float* partial_values, uint32_t* partial_tokens, uint32_t partial_count,
    void* sort_workspace, size_t sort_workspace_bytes, void* scan_workspace,
    size_t scan_workspace_bytes, ApxInfSamplingOutput* output,
    cudaStream_t stream) {
  if (logits == nullptr || counts == nullptr || adjusted == nullptr ||
      token_ids == nullptr || output == nullptr || vocab_size == 0 ||
      partial_count == 0 || !(repetition > 0.0f))
    return cudaErrorInvalidValue;
  const float inverse_temperature =
      selection == 0 ? 1.0f : 1.0f / temperature;
  cudaError_t status;
  switch (dtype) {
    case 0:
      status = launch_prepare<float>(
          logits, counts, adjusted, token_ids, vocab_size, repetition,
          frequency, presence, inverse_temperature, stream);
      break;
    case 1:
      status = launch_prepare<half>(
          logits, counts, adjusted, token_ids, vocab_size, repetition,
          frequency, presence, inverse_temperature, stream);
      break;
    case 2:
      status = launch_prepare<__nv_bfloat16>(
          logits, counts, adjusted, token_ids, vocab_size, repetition,
          frequency, presence, inverse_temperature, stream);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  if (status != cudaSuccess) return status;

  if (selection == 0 && return_logprob == 0) {
    apxinf_argmax_stage1_kernel<<<partial_count, kThreads, 0, stream>>>(
        adjusted, vocab_size, partial_values, partial_tokens);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    apxinf_argmax_stage2_kernel<<<1, kThreads, 0, stream>>>(
        partial_values, partial_tokens, partial_count, counts, output);
    return cudaGetLastError();
  }

  if (sorted_logits == nullptr || sorted_tokens == nullptr ||
      weights == nullptr || cdf == nullptr || sort_workspace == nullptr ||
      scan_workspace == nullptr)
    return cudaErrorInvalidValue;
  status = cub::DeviceRadixSort::SortPairsDescending(
      sort_workspace, sort_workspace_bytes, adjusted, sorted_logits, token_ids,
      sorted_tokens, vocab_size, 0, 32, stream);
  if (status != cudaSuccess) return status;
  const uint32_t candidate_limit =
      top_k == 0 ? vocab_size : std::min(top_k, vocab_size);
  int blocks = static_cast<int>((vocab_size + kThreads - 1) / kThreads);
  blocks = std::min(blocks, 1024);
  apxinf_softmax_weights_kernel<<<blocks, kThreads, 0, stream>>>(
      sorted_logits, weights, vocab_size, candidate_limit);
  status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  status = cub::DeviceScan::InclusiveSum(
      scan_workspace, scan_workspace_bytes, weights, cdf, vocab_size, stream);
  if (status != cudaSuccess) return status;
  apxinf_select_cdf_kernel<<<1, 1, 0, stream>>>(
      sorted_tokens, weights, cdf, candidate_limit,
      selection == 0 ? 1.0f : top_p, selection != 0, return_logprob, seed,
      sequence, draw, counts, output);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_fill_standard_normal(
    void* output, int dtype, uint64_t count, uint64_t seed,
    uint64_t sequence, uint64_t draw, cudaStream_t stream) {
  if (output == nullptr || count == 0) return cudaErrorInvalidValue;
  int blocks = static_cast<int>(((count + 1) / 2 + kThreads - 1) / kThreads);
  blocks = std::min(blocks, 1024);
  switch (dtype) {
    case 0:
      apxinf_standard_normal_kernel<<<blocks, kThreads, 0, stream>>>(
          static_cast<float*>(output), count, seed, sequence, draw);
      break;
    case 1:
      apxinf_standard_normal_kernel<<<blocks, kThreads, 0, stream>>>(
          static_cast<half*>(output), count, seed, sequence, draw);
      break;
    case 2:
      apxinf_standard_normal_kernel<<<blocks, kThreads, 0, stream>>>(
          static_cast<__nv_bfloat16*>(output), count, seed, sequence, draw);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}
