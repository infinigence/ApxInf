#include "../internal.h"
#include <cstdlib>
#include <cstring>

#if defined(APXINF_ATTENTION_FA2)

namespace apxinf::cuda_new::cutlass_ops {

int fa2_bf16_decode_splitkv(const void* q, const void* k, const void* v,
                           void* output, void* workspace, int key_tokens,
                           float softmax_scale, cudaStream_t stream);

int fa2_bf16(const void* q, const void* k, const void* v, void* output,
             void* softmax_lse, int batch, int query_tokens, int key_tokens,
             int query_heads, int kv_heads, int head_dim, float softmax_scale,
             cudaStream_t stream);
int fa2_bf16_causal(const void* q, const void* k, const void* v, void* output,
                    void* softmax_lse, int batch, int query_tokens,
                    int key_tokens, int query_heads, int kv_heads,
                    int head_dim, float softmax_scale, cudaStream_t stream);
int fa2_f16(const void* q, const void* k, const void* v, void* output,
            void* softmax_lse, int batch, int query_tokens, int key_tokens,
            int query_heads, int kv_heads, int head_dim, float softmax_scale,
            cudaStream_t stream);
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
  bool splitkv = false;
};

bool use_decode_splitkv(const Spec& spec) {
  const char* enabled = std::getenv("APXINF_FA2_DECODE_SPLITKV");
  return enabled != nullptr && std::strcmp(enabled, "1") == 0 &&
         spec.dtype == APXINF_DTYPE_BF16 &&
         spec.output_dtype == APXINF_DTYPE_BF16 && spec.batch == 1 &&
         spec.query_tokens == 1 && spec.query_heads == 24 &&
         spec.kv_heads == 4 && spec.head_dim == 256 &&
         spec.key_tokens >= 128 &&
         (spec.mask == APXINF_ATTENTION_MASK_NONE ||
          (spec.mask == APXINF_ATTENTION_MASK_CAUSAL &&
           spec.query_start == spec.key_tokens - 1));
}

}  // namespace

size_t fa2_resource_requirements(const Spec& spec) {
  if (use_decode_splitkv(spec)) return (24 + 5 * 24 * (1 + 256)) * sizeof(float);
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
  state->splitkv = use_decode_splitkv(execution.spec);
  const size_t bytes = fa2_resource_requirements(execution.spec);
  if (bytes > execution.resource_limit) {
    throw Failure(APXINF_STATUS_UNSUPPORTED,
                  "FA2 softmax LSE workspace exceeds policy");
  }
  check_cuda(cudaMalloc(&state->softmax_lse, bytes));
  execution.resource_bytes = bytes;
  execution.provider_state = state.release();
}

cudaError_t launch_fa2(Execution& execution) {
  auto* state = static_cast<Fa2State*>(execution.provider_state);
  const auto& spec = execution.spec;
  const auto& bindings = execution.bindings;
  const auto stream = static_cast<cudaStream_t>(bindings.stream);
  if (state->splitkv) {
    return static_cast<cudaError_t>(
        apxinf::cuda_new::cutlass_ops::fa2_bf16_decode_splitkv(
            bindings.query, bindings.key, bindings.value, bindings.output,
            state->softmax_lse, static_cast<int>(spec.key_tokens),
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
