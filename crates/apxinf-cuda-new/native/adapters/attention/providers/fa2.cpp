#include "../internal.h"

#if defined(APXINF_ATTENTION_FA2)

#include <algorithm>
#include <limits>

namespace apxinf::cuda_new::cutlass_ops {

int fa2_bf16(const void* q, const void* k, const void* v, void* output,
             void* softmax_lse, int batch, int query_tokens, int key_tokens,
             int query_heads, int kv_heads, int head_dim, float softmax_scale,
             cudaStream_t stream);
int fa2_bf16_causal(const void* q, const void* k, const void* v, void* output,
                    void* softmax_lse, int batch, int query_tokens,
                    int key_tokens, int query_heads, int kv_heads,
                    int head_dim, float softmax_scale, cudaStream_t stream);
int fa2_bf16_splitkv(
    const void* q, const void* k, const void* v, void* output,
    void* softmax_lse, void* softmax_lse_accum, void* o_accum, int batch,
    int query_tokens, int key_tokens, int query_heads, int kv_heads,
    int head_dim, float softmax_scale, int num_splits, cudaStream_t stream);
int fa2_bf16_causal_splitkv(
    const void* q, const void* k, const void* v, void* output,
    void* softmax_lse, void* softmax_lse_accum, void* o_accum, int batch,
    int query_tokens, int key_tokens, int query_heads, int kv_heads,
    int head_dim, float softmax_scale, int num_splits, cudaStream_t stream);
int fa2_f16(const void* q, const void* k, const void* v, void* output,
            void* softmax_lse, int batch, int query_tokens, int key_tokens,
            int query_heads, int kv_heads, int head_dim, float softmax_scale,
            cudaStream_t stream);
int fa2_f16_packed_qkv(const void* q, const void* k, const void* v,
                       void* output, void* softmax_lse, int batch,
                       int tokens, int heads, int head_dim,
                       float softmax_scale, cudaStream_t stream);
#if defined(APXINF_ATTENTION_FA2_E4M3)
int fa2_f16_direct_e4m3_522(
    const void* q, const void* k, const void* v, void* output,
    void* softmax_lse, int batch, int query_tokens, int key_tokens,
    int query_heads, int kv_heads, int head_dim, float softmax_scale,
    float output_scale, cudaStream_t stream);
#endif
}  // namespace apxinf::cuda_new::cutlass_ops

namespace apxinf::attention {
namespace {

struct Fa2State {
  float* softmax_lse = nullptr;
};

struct Fa2SplitKvState {
  float* softmax_lse = nullptr;
  float* softmax_lse_accum = nullptr;
  float* output_accum = nullptr;

  ~Fa2SplitKvState() {
    if (output_accum != nullptr) cudaFree(output_accum);
    if (softmax_lse_accum != nullptr) cudaFree(softmax_lse_accum);
    if (softmax_lse != nullptr) cudaFree(softmax_lse);
  }
};

size_t checked_multiply(size_t lhs, size_t rhs, const char* label) {
  if (rhs != 0 && lhs > std::numeric_limits<size_t>::max() / rhs) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, label);
  }
  return lhs * rhs;
}

size_t splitkv_lse_elements(const Spec& spec) {
  return checked_multiply(
      checked_multiply(static_cast<size_t>(spec.batch),
                       static_cast<size_t>(spec.query_heads),
                       "FA2 split-KV LSE workspace size overflow"),
      static_cast<size_t>(spec.query_tokens),
      "FA2 split-KV LSE workspace size overflow");
}

}  // namespace

size_t fa2_resource_requirements(const Spec& spec, int) {
  const uint64_t elements = static_cast<uint64_t>(spec.batch) *
                            spec.query_heads * spec.query_tokens;
  if (elements > SIZE_MAX / sizeof(float)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "FA2 softmax LSE workspace size overflow");
  }
  return static_cast<size_t>(elements) * sizeof(float);
}

void prepare_fa2(Execution& execution) {
  auto state = std::make_unique<Fa2State>();
  const size_t bytes =
      fa2_resource_requirements(execution.spec, execution.configuration);
  if (bytes > execution.resource_limit) {
    throw Failure(APXINF_STATUS_UNSUPPORTED,
                  "FA2 softmax LSE workspace exceeds policy");
  }
  check_cuda(cudaMalloc(&state->softmax_lse, bytes));
  execution.resource_bytes = bytes;
  execution.provider_state = state.release();
}

size_t fa2_splitkv_resource_requirements(const Spec& spec,
                                         int configuration) {
  if (configuration <= 1 || configuration > 128) {
    throw Failure(APXINF_STATUS_UNSUPPORTED,
                  "invalid FA2 split-KV configuration");
  }
  const size_t splits = static_cast<size_t>(configuration);
  const size_t lse_elements = splitkv_lse_elements(spec);
  const size_t lse_bytes = checked_multiply(
      lse_elements, sizeof(float),
      "FA2 split-KV LSE workspace byte size overflow");
  const size_t lse_accum_bytes = checked_multiply(
      checked_multiply(splits, lse_elements,
                       "FA2 split-KV LSE accumulation size overflow"),
      sizeof(float),
      "FA2 split-KV LSE accumulation byte size overflow");
  const size_t output_accum_bytes = checked_multiply(
      checked_multiply(
          checked_multiply(splits, lse_elements,
                           "FA2 split-KV output accumulation size overflow"),
          static_cast<size_t>(spec.head_dim),
          "FA2 split-KV output accumulation size overflow"),
      sizeof(float),
      "FA2 split-KV output accumulation byte size overflow");
  if (lse_bytes > std::numeric_limits<size_t>::max() - lse_accum_bytes ||
      lse_bytes + lse_accum_bytes >
          std::numeric_limits<size_t>::max() - output_accum_bytes) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "FA2 split-KV total workspace size overflow");
  }
  return lse_bytes + lse_accum_bytes + output_accum_bytes;
}

void prepare_fa2_splitkv(Execution& execution) {
  const auto& spec = execution.spec;
  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceProperties(&properties, execution.device));
  const size_t key_tile = spec.head_dim == 128 ? 128 : 64;
  const size_t key_tiles =
      (static_cast<size_t>(spec.key_tokens) + key_tile - 1) / key_tile;
  const size_t device_limit = std::min<size_t>(
      128, std::min(key_tiles,
                    static_cast<size_t>(properties.multiProcessorCount) * 2));
  if (static_cast<size_t>(execution.configuration) > device_limit) {
    throw Failure(APXINF_STATUS_UNSUPPORTED,
                  "FA2 split-KV configuration exceeds device split limit");
  }
  const size_t required =
      fa2_splitkv_resource_requirements(spec, execution.configuration);
  if (required > execution.resource_limit) {
    throw Failure(APXINF_STATUS_UNSUPPORTED,
                  "FA2 split-KV workspace exceeds policy");
  }
  auto state = std::make_unique<Fa2SplitKvState>();
  const size_t lse_elements = splitkv_lse_elements(spec);
  const size_t lse_bytes = lse_elements * sizeof(float);
  const size_t lse_accum_bytes =
      static_cast<size_t>(execution.configuration) * lse_bytes;
  const size_t output_accum_bytes =
      static_cast<size_t>(execution.configuration) * lse_bytes * spec.head_dim;
  check_cuda(cudaMalloc(&state->softmax_lse, lse_bytes));
  check_cuda(cudaMalloc(&state->softmax_lse_accum, lse_accum_bytes));
  check_cuda(cudaMalloc(&state->output_accum, output_accum_bytes));
  execution.resource_bytes = required;
  execution.provider_state = state.release();
}

cudaError_t launch_fa2_splitkv(Execution& execution) {
  auto* state = static_cast<Fa2SplitKvState*>(execution.provider_state);
  const auto& spec = execution.spec;
  const auto& bindings = execution.bindings;
  const auto stream = static_cast<cudaStream_t>(bindings.stream);
  const auto launch = spec.mask == APXINF_ATTENTION_MASK_CAUSAL
                          ? apxinf::cuda_new::cutlass_ops::fa2_bf16_causal_splitkv
                          : apxinf::cuda_new::cutlass_ops::fa2_bf16_splitkv;
  return static_cast<cudaError_t>(launch(
      bindings.query, bindings.key, bindings.value, bindings.output,
      state->softmax_lse, state->softmax_lse_accum, state->output_accum,
      static_cast<int>(spec.batch), static_cast<int>(spec.query_tokens),
      static_cast<int>(spec.key_tokens), static_cast<int>(spec.query_heads),
      static_cast<int>(spec.kv_heads), static_cast<int>(spec.head_dim),
      bindings.scale, execution.configuration, stream));
}

void destroy_fa2_splitkv(Execution& execution) noexcept {
  delete static_cast<Fa2SplitKvState*>(execution.provider_state);
  execution.provider_state = nullptr;
}

cudaError_t launch_fa2(Execution& execution) {
  auto* state = static_cast<Fa2State*>(execution.provider_state);
  const auto& spec = execution.spec;
  const auto& bindings = execution.bindings;
  const auto stream = static_cast<cudaStream_t>(bindings.stream);
  if (spec.semantic == APXINF_ATTENTION_SEMANTIC_PACKED_QKV) {
    return static_cast<cudaError_t>(
        apxinf::cuda_new::cutlass_ops::fa2_f16_packed_qkv(
            bindings.query, bindings.key, bindings.value, bindings.output,
            state->softmax_lse, static_cast<int>(spec.batch),
            static_cast<int>(spec.query_tokens),
            static_cast<int>(spec.query_heads), static_cast<int>(spec.head_dim),
            bindings.scale, stream));
  }
#if defined(APXINF_ATTENTION_FA2_E4M3)
  if (spec.dtype == APXINF_DTYPE_F16 &&
      spec.output_dtype == APXINF_DTYPE_E4M3) {
    return static_cast<cudaError_t>(
        apxinf::cuda_new::cutlass_ops::fa2_f16_direct_e4m3_522(
            bindings.query, bindings.key, bindings.value, bindings.output,
            state->softmax_lse, static_cast<int>(spec.batch),
            static_cast<int>(spec.query_tokens),
            static_cast<int>(spec.key_tokens),
            static_cast<int>(spec.query_heads),
            static_cast<int>(spec.kv_heads), static_cast<int>(spec.head_dim),
            bindings.scale, bindings.output_scale, stream));
  }
#endif
  int status = static_cast<int>(cudaErrorInvalidValue);
  if (spec.dtype == APXINF_DTYPE_BF16) {
    const auto launch = spec.mask == APXINF_ATTENTION_MASK_CAUSAL
                            ? apxinf::cuda_new::cutlass_ops::fa2_bf16_causal
                            : apxinf::cuda_new::cutlass_ops::fa2_bf16;
    status = launch(
        bindings.query, bindings.key, bindings.value, bindings.output,
        state->softmax_lse, static_cast<int>(spec.batch),
        static_cast<int>(spec.query_tokens), static_cast<int>(spec.key_tokens),
        static_cast<int>(spec.query_heads), static_cast<int>(spec.kv_heads),
        static_cast<int>(spec.head_dim), bindings.scale, stream);
  } else if (spec.dtype == APXINF_DTYPE_F16 &&
             spec.mask == APXINF_ATTENTION_MASK_NONE) {
    status = apxinf::cuda_new::cutlass_ops::fa2_f16(
        bindings.query, bindings.key, bindings.value, bindings.output,
        state->softmax_lse, static_cast<int>(spec.batch),
        static_cast<int>(spec.query_tokens), static_cast<int>(spec.key_tokens),
        static_cast<int>(spec.query_heads), static_cast<int>(spec.kv_heads),
        static_cast<int>(spec.head_dim), bindings.scale, stream);
  }
  return static_cast<cudaError_t>(status);
}

void destroy_fa2(Execution& execution) noexcept {
  auto* state = static_cast<Fa2State*>(execution.provider_state);
  if (state == nullptr) return;
  if (state->softmax_lse != nullptr) cudaFree(state->softmax_lse);
  delete state;
  execution.provider_state = nullptr;
}

}  // namespace apxinf::attention

#endif
