// SPDX-License-Identifier: Apache-2.0
// L1 bindings for the SM110 segmented-attention candidates. The translated
// CUTLASS kernels own the arithmetic; this file owns execution-scoped provider
// state, fixed geometry, and split-batch glue.

#include "../internal.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>

extern "C" size_t apxinf_attention_d256_causal_state_size();
extern "C" int apxinf_attention_d256_causal_forward(
    const void*, const void*, const void*, void*, int, int, int, float,
    cudaStream_t, void*, int);

extern "C" size_t apxinf_attention_d256_split_batch_state_size();
extern "C" size_t apxinf_attention_d256_split_batch_lse_elements(int, int,
                                                                    int);
extern "C" int apxinf_attention_d256_split_batch_forward(
    const void*, const void*, const void*, void*, float*, int, int, int, int*,
    int, int, int, int, int, float, cudaStream_t, void*, int);

extern "C" size_t apxinf_attention_strided_groups_state_size();
extern "C" void apxinf_attention_strided_groups_state_destroy(void*);
extern "C" int apxinf_attention_strided_groups_forward(
    const void*, const void*, const void*, void*, int, int, int, int, int,
    int, int, float, cudaStream_t, void*, int);

namespace {
constexpr int kBatch = 2;
constexpr int kQueryTokens = 50;
constexpr int kQueryHeads = 16;
constexpr int kKvHeads = 4;
constexpr int kDim = 256;
constexpr int kKvViewRows = 1720;
constexpr int kKvBatchRows = 1718;
constexpr int kQueryWords = kQueryTokens * kQueryHeads * kDim;
constexpr int kMergeRows = kQueryTokens * kQueryHeads;

__global__ void duplicate_q_vec8(const uint4* source, uint4* destination) {
  constexpr int vectors = kQueryWords / 8;
  int index = int(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < vectors) {
    uint4 value = source[index];
    destination[index] = value;
    destination[index + vectors] = value;
  }
}

__global__ void repack_lse(const float* native_lse, float* dense_lse) {
  int index = int(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= kBatch * kMergeRows) return;
  int batch = index / kMergeRows;
  int row = index % kMergeRows;
  int query_head = row / kQueryTokens;
  int token = row % kQueryTokens;
  constexpr int rounded_query = 256;
  int kv_head = query_head / 4;
  int lane = query_head % 4;
  dense_lse[index] = native_lse[
      batch * kKvHeads * rounded_query + kv_head * rounded_query +
      token * 4 + lane];
}

union Bf16x8 {
  uint4 vector;
  uint16_t words[8];
};

__global__ void merge_partial_vec8(const uint4* partial, const float* lse,
                                   uint4* output) {
  constexpr int vectors_per_row = kDim / 8;
  constexpr int vectors_per_batch = kQueryWords / 8;
  int warp = (int(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
  if (warp >= kMergeRows) return;
  int lane = threadIdx.x & 31;
  int token = warp / kQueryHeads;
  int head = warp % kQueryHeads;
  int lse_index = head * kQueryTokens + token;
  float weight0 = 0.0f, weight1 = 0.0f, sum = 0.0f;
  if (lane == 0) {
    float first = lse[lse_index];
    float second = lse[kMergeRows + lse_index];
    float maximum = fmaxf(first, second);
    weight0 = expf(first - maximum);
    weight1 = expf(second - maximum);
    sum = weight0 + weight1;
  }
  weight0 = __shfl_sync(0xffffffff, weight0, 0);
  weight1 = __shfl_sync(0xffffffff, weight1, 0);
  sum = __shfl_sync(0xffffffff, sum, 0);
  int vector = warp * vectors_per_row + lane;
  Bf16x8 first{}, second{}, result{};
  first.vector = partial[vector];
  second.vector = partial[vectors_per_batch + vector];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    float a = __bfloat162float(__ushort_as_bfloat16(first.words[i]));
    float b = __bfloat162float(__ushort_as_bfloat16(second.words[i]));
    result.words[i] = __bfloat16_as_ushort(
        __float2bfloat16_rn((weight0 * a + weight1 * b) / sum));
  }
  output[vector] = result.vector;
}

struct CandidateCausalState {
  void* kernel = nullptr;
};

struct CandidateSplitState {
  void* kernel = nullptr;
  int32_t* cu_seqlens = nullptr;
  float* native_lse = nullptr;
  void* q2 = nullptr;
  void* partial = nullptr;
  float* dense_lse = nullptr;
};

struct CandidateVisionState {
  void* length_624 = nullptr;
  void* length_2200 = nullptr;
};

void destroy_vision_group_state(void* state) noexcept {
  if (state == nullptr) return;
  apxinf_attention_strided_groups_state_destroy(state);
  std::free(state);
}

void destroy_candidate_split(CandidateSplitState* state) {
  if (state == nullptr) return;
  cudaFree(state->dense_lse);
  cudaFree(state->partial);
  cudaFree(state->q2);
  cudaFree(state->native_lse);
  cudaFree(state->cu_seqlens);
  std::free(state->kernel);
  delete state;
}

int launch_vision_group_with_state(int group, const void* q, const void* k,
                                   const void* v, void* output,
                                   cudaStream_t stream, int prepare_only,
                                   void* state_624, void* state_2200,
                                   int value_stride) {
  constexpr int starts[4] = {0, 624, 1248, 1872};
  constexpr int lengths[4] = {624, 624, 624, 2200};
  constexpr int qko_stride = 16 * 64;
  constexpr int batch_tokens = 4072;
  int start = starts[group];
  int length = lengths[group];
  const char* qp = static_cast<const char*>(q) + int64_t(start) * qko_stride * 2;
  const char* kp = static_cast<const char*>(k) + int64_t(start) * qko_stride * 2;
  const char* vp =
      static_cast<const char*>(v) + int64_t(start) * value_stride * 2;
  char* op = static_cast<char*>(output) + int64_t(start) * qko_stride * 2;
  return apxinf_attention_strided_groups_forward(
      qp, kp, vp, op, 3, length, 16, 64, qko_stride, value_stride,
      batch_tokens, 0.125f, stream,
      length == 624 ? state_624 : state_2200, prepare_only);
}

}  // namespace

namespace apxinf::attention {

size_t qwen_d256_causal_resource_requirements(const Spec&, int) {
  return apxinf_attention_d256_causal_state_size();
}

void prepare_qwen_d256_causal(Execution& execution) {
  auto state = std::make_unique<CandidateCausalState>();
  state->kernel =
      std::calloc(1, apxinf_attention_d256_causal_state_size());
  if (state->kernel == nullptr) {
    throw Failure(APXINF_STATUS_PROVIDER_ERROR,
                  "Qwen D256 causal state allocation failed");
  }
  const int status = apxinf_attention_d256_causal_forward(
      execution.bindings.query, execution.bindings.key,
      execution.bindings.value, execution.bindings.output,
      static_cast<int>(execution.spec.query_tokens),
      static_cast<int>(execution.spec.query_heads),
      static_cast<int>(execution.spec.kv_heads), execution.bindings.scale,
      static_cast<cudaStream_t>(execution.bindings.stream), state->kernel, 1);
  if (status != 0) {
    std::free(state->kernel);
    throw Failure(APXINF_STATUS_PROVIDER_ERROR,
                  "Qwen D256 causal preparation failed with status " +
                      std::to_string(status));
  }
  execution.resource_bytes = apxinf_attention_d256_causal_state_size();
  execution.provider_state = state.release();
}

cudaError_t launch_qwen_d256_causal(Execution& execution) {
  auto* state = static_cast<CandidateCausalState*>(execution.provider_state);
  const int status = apxinf_attention_d256_causal_forward(
      execution.bindings.query, execution.bindings.key,
      execution.bindings.value, execution.bindings.output,
      static_cast<int>(execution.spec.query_tokens),
      static_cast<int>(execution.spec.query_heads),
      static_cast<int>(execution.spec.kv_heads), execution.bindings.scale,
      static_cast<cudaStream_t>(execution.bindings.stream), state->kernel, 0);
  return status == 0 ? cudaGetLastError() : cudaErrorUnknown;
}

void destroy_qwen_d256_causal(Execution& execution) noexcept {
  auto* state = static_cast<CandidateCausalState*>(execution.provider_state);
  if (state == nullptr) return;
  std::free(state->kernel);
  delete state;
  execution.provider_state = nullptr;
}

size_t qwen_d256_split_resource_requirements(const Spec&, int) {
  const size_t query_bytes = size_t(kQueryWords) * sizeof(uint16_t);
  const size_t native_lse = apxinf_attention_d256_split_batch_lse_elements(
                                kQueryTokens, kKvHeads, kBatch) *
                            sizeof(float);
  return apxinf_attention_d256_split_batch_state_size() +
         4 * query_bytes + 2 * kMergeRows * sizeof(float) + native_lse +
         3 * sizeof(int32_t);
}

void prepare_qwen_d256_split(Execution& execution) {
  auto* state = new CandidateSplitState();
  try {
    state->kernel =
        std::calloc(1, apxinf_attention_d256_split_batch_state_size());
    if (state->kernel == nullptr) {
      throw Failure(APXINF_STATUS_PROVIDER_ERROR,
                    "Qwen D256 split state allocation failed");
    }
    const size_t query_bytes = size_t(kQueryWords) * sizeof(uint16_t);
    const size_t native_lse_bytes =
        apxinf_attention_d256_split_batch_lse_elements(
            kQueryTokens, kKvHeads, kBatch) *
        sizeof(float);
    check_cuda(cudaMalloc(&state->q2, 2 * query_bytes));
    check_cuda(cudaMalloc(&state->partial, 2 * query_bytes));
    check_cuda(
        cudaMalloc(&state->dense_lse, 2 * kMergeRows * sizeof(float)));
    check_cuda(cudaMalloc(&state->native_lse, native_lse_bytes));
    int32_t host[3] = {0, kKvBatchRows,
                       static_cast<int32_t>(execution.spec.key_tokens)};
    check_cuda(cudaMalloc(&state->cu_seqlens, sizeof(host)));
    check_cuda(cudaMemcpy(state->cu_seqlens, host, sizeof(host),
                          cudaMemcpyHostToDevice));
    duplicate_q_vec8<<<100, 256, 0,
                       static_cast<cudaStream_t>(execution.bindings.stream)>>>(
        static_cast<const uint4*>(execution.bindings.query),
        static_cast<uint4*>(state->q2));
    check_cuda(cudaGetLastError());
    const int status = apxinf_attention_d256_split_batch_forward(
        state->q2, execution.bindings.key, execution.bindings.value,
        state->partial, state->native_lse, kQueryTokens, kKvViewRows,
        static_cast<int>(execution.spec.key_tokens), state->cu_seqlens,
        kQueryHeads, kKvHeads, kBatch, kQueryWords,
        kKvBatchRows * kKvHeads * kDim, execution.bindings.scale,
        static_cast<cudaStream_t>(execution.bindings.stream), state->kernel,
        1);
    if (status != 0) {
      throw Failure(APXINF_STATUS_PROVIDER_ERROR,
                    "Qwen D256 split preparation failed with status " +
                        std::to_string(status));
    }
  } catch (...) {
    destroy_candidate_split(state);
    throw;
  }
  execution.resource_bytes =
      qwen_d256_split_resource_requirements(execution.spec, 0);
  execution.provider_state = state;
}

cudaError_t launch_qwen_d256_split(Execution& execution) {
  auto* state = static_cast<CandidateSplitState*>(execution.provider_state);
  const auto stream = static_cast<cudaStream_t>(execution.bindings.stream);
  duplicate_q_vec8<<<100, 256, 0, stream>>>(
      static_cast<const uint4*>(execution.bindings.query),
      static_cast<uint4*>(state->q2));
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) return error;
  const int status = apxinf_attention_d256_split_batch_forward(
      state->q2, execution.bindings.key, execution.bindings.value,
      state->partial, state->native_lse, kQueryTokens, kKvViewRows,
      static_cast<int>(execution.spec.key_tokens), state->cu_seqlens,
      kQueryHeads, kKvHeads, kBatch, kQueryWords,
      kKvBatchRows * kKvHeads * kDim, execution.bindings.scale, stream,
      state->kernel, 0);
  if (status != 0) return cudaErrorUnknown;
  repack_lse<<<(kBatch * kMergeRows + 255) / 256, 256, 0, stream>>>(
      state->native_lse, state->dense_lse);
  if ((error = cudaGetLastError()) != cudaSuccess) return error;
  merge_partial_vec8<<<100, 256, 0, stream>>>(
      static_cast<const uint4*>(state->partial), state->dense_lse,
      static_cast<uint4*>(execution.bindings.output));
  return cudaGetLastError();
}

void destroy_qwen_d256_split(Execution& execution) noexcept {
  destroy_candidate_split(
      static_cast<CandidateSplitState*>(execution.provider_state));
  execution.provider_state = nullptr;
}

size_t qwen_vision_segmented_resource_requirements(const Spec&, int) {
  return 2 * apxinf_attention_strided_groups_state_size();
}

void prepare_qwen_vision_segmented(Execution& execution) {
  auto state = std::make_unique<CandidateVisionState>();
  const size_t bytes = apxinf_attention_strided_groups_state_size();
  state->length_624 = std::calloc(1, bytes);
  state->length_2200 = std::calloc(1, bytes);
  if (state->length_624 == nullptr || state->length_2200 == nullptr) {
    destroy_vision_group_state(state->length_624);
    destroy_vision_group_state(state->length_2200);
    throw Failure(APXINF_STATUS_PROVIDER_ERROR,
                  "Qwen vision attention state allocation failed");
  }
  for (int group : {0, 3}) {
    const int status = launch_vision_group_with_state(
        group, execution.bindings.query, execution.bindings.key,
        execution.bindings.value, execution.bindings.output,
        static_cast<cudaStream_t>(execution.bindings.stream), 1,
        state->length_624, state->length_2200, 16 * 64);
    if (status != 0) {
      destroy_vision_group_state(state->length_624);
      destroy_vision_group_state(state->length_2200);
      throw Failure(APXINF_STATUS_PROVIDER_ERROR,
                    "Qwen vision attention preparation failed with status " +
                        std::to_string(status));
    }
  }
  execution.resource_bytes = 2 * bytes;
  execution.provider_state = state.release();
}

cudaError_t launch_qwen_vision_segmented(Execution& execution) {
  auto* state = static_cast<CandidateVisionState*>(execution.provider_state);
  for (int group = 0; group < 4; ++group) {
    const int status = launch_vision_group_with_state(
        group, execution.bindings.query, execution.bindings.key,
        execution.bindings.value, execution.bindings.output,
        static_cast<cudaStream_t>(execution.bindings.stream), 0,
        state->length_624, state->length_2200, 16 * 64);
    if (status != 0) return cudaErrorUnknown;
  }
  return cudaGetLastError();
}

void destroy_qwen_vision_segmented(Execution& execution) noexcept {
  auto* state = static_cast<CandidateVisionState*>(execution.provider_state);
  if (state == nullptr) return;
  destroy_vision_group_state(state->length_624);
  destroy_vision_group_state(state->length_2200);
  delete state;
  execution.provider_state = nullptr;
}

}  // namespace apxinf::attention
