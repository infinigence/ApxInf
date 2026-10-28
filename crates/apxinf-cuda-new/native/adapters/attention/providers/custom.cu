#include "../internal.h"
#include "../../../kernels/custom/attention.cuh"
#include "../../../kernels/custom/attention_short_f32.cuh"

namespace apxinf::attention {
namespace {

struct CustomState {
  float* probabilities = nullptr;
};

}  // namespace

size_t custom_resource_requirements(const Spec& spec, int) {
  if (spec.semantic == APXINF_ATTENTION_SEMANTIC_SEGMENTED) return 0;
  const auto rows = static_cast<uint64_t>(spec.batch) * spec.query_tokens *
                    spec.query_heads;
  const auto elements = rows * spec.key_tokens;
  if (elements > SIZE_MAX / sizeof(float)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "Attention probability workspace size overflow");
  }
  return static_cast<size_t>(elements) * sizeof(float);
}

void prepare_custom(Execution& execution) {
  auto state = std::make_unique<CustomState>();
  const size_t bytes =
      custom_resource_requirements(execution.spec, execution.configuration);
  if (bytes > execution.resource_limit) {
    throw Failure(APXINF_STATUS_UNSUPPORTED,
                  "Attention probability workspace exceeds policy");
  }
  if (execution.spec.semantic == APXINF_ATTENTION_SEMANTIC_SEGMENTED) {
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, execution.device));
    if (execution.spec.query_heads > properties.maxGridSize[1] ||
        execution.spec.segments > properties.maxGridSize[2]) {
      throw Failure(APXINF_STATUS_UNSUPPORTED,
                    "segmented Attention grid exceeds device limits");
    }
    const uint64_t shared_bytes =
        static_cast<uint64_t>(execution.spec.max_segment_tokens) *
        sizeof(float);
    int max_shared_bytes = 0;
    check_cuda(cudaDeviceGetAttribute(
        &max_shared_bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin,
        execution.device));
    if (shared_bytes > static_cast<uint64_t>(max_shared_bytes) ||
        shared_bytes > static_cast<uint64_t>(INT32_MAX)) {
      throw Failure(APXINF_STATUS_UNSUPPORTED,
                    "segmented Attention segment exceeds shared-memory limit");
    }
    if (shared_bytes > static_cast<uint64_t>(properties.sharedMemPerBlock)) {
      if (execution.spec.dtype == APXINF_DTYPE_F16) {
        check_cuda(cudaFuncSetAttribute(
            kernels::segmented_attention_kernel<__half>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes)));
      } else {
        check_cuda(cudaFuncSetAttribute(
            kernels::segmented_attention_kernel<__nv_bfloat16>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes)));
      }
    }
  }
  if (bytes != 0) check_cuda(cudaMalloc(&state->probabilities, bytes));
  execution.resource_bytes = bytes;
  execution.provider_state = state.release();
}

cudaError_t launch_custom(Execution& execution) {
  auto* state = static_cast<CustomState*>(execution.provider_state);
  const auto& spec = execution.spec;
  const auto& bindings = execution.bindings;
  const auto stream = static_cast<cudaStream_t>(bindings.stream);
  if (spec.semantic == APXINF_ATTENTION_SEMANTIC_SEGMENTED) {
    if (spec.dtype == APXINF_DTYPE_F16) {
      return kernels::launch_segmented_attention<__half>(
          bindings.query, bindings.key, bindings.value, bindings.offsets,
          bindings.output, static_cast<int>(spec.segments),
          static_cast<int>(spec.max_segment_tokens),
          static_cast<int>(spec.query_tokens),
          static_cast<int>(spec.query_heads), static_cast<int>(spec.head_dim),
          bindings.scale, stream);
    }
    return kernels::launch_segmented_attention<__nv_bfloat16>(
        bindings.query, bindings.key, bindings.value, bindings.offsets,
        bindings.output, static_cast<int>(spec.segments),
        static_cast<int>(spec.max_segment_tokens),
        static_cast<int>(spec.query_tokens),
        static_cast<int>(spec.query_heads), static_cast<int>(spec.head_dim),
        bindings.scale, stream);
  }
  const int key_stride = spec.semantic == APXINF_ATTENTION_SEMANTIC_KV_CACHE
                             ? static_cast<int>(spec.key_capacity)
                             : static_cast<int>(spec.key_tokens);
  if (spec.dtype == APXINF_DTYPE_F32) {
    return kernels::launch_attention<float>(
        bindings.query, bindings.key, bindings.value, bindings.output,
        state->probabilities, static_cast<int>(spec.batch),
        static_cast<int>(spec.query_tokens), static_cast<int>(spec.key_tokens),
        key_stride,
        static_cast<int>(spec.query_heads), static_cast<int>(spec.kv_heads),
        static_cast<int>(spec.head_dim),
        spec.mask == APXINF_ATTENTION_MASK_CAUSAL,
        static_cast<int>(spec.query_start), bindings.decode_meta,
        bindings.scale, stream, static_cast<const float*>(bindings.bias),
        spec.bias_broadcast_batch != 0);
  }
  if (spec.dtype == APXINF_DTYPE_F16) {
    return kernels::launch_attention<__half>(
        bindings.query, bindings.key, bindings.value, bindings.output,
        state->probabilities, static_cast<int>(spec.batch),
        static_cast<int>(spec.query_tokens), static_cast<int>(spec.key_tokens),
        key_stride,
        static_cast<int>(spec.query_heads), static_cast<int>(spec.kv_heads),
        static_cast<int>(spec.head_dim),
        spec.mask == APXINF_ATTENTION_MASK_CAUSAL,
        static_cast<int>(spec.query_start), bindings.decode_meta,
        bindings.scale, stream, static_cast<const float*>(bindings.bias),
        spec.bias_broadcast_batch != 0);
  }
  return kernels::launch_attention<__nv_bfloat16>(
      bindings.query, bindings.key, bindings.value, bindings.output,
      state->probabilities, static_cast<int>(spec.batch),
      static_cast<int>(spec.query_tokens), static_cast<int>(spec.key_tokens),
      key_stride,
      static_cast<int>(spec.query_heads), static_cast<int>(spec.kv_heads),
      static_cast<int>(spec.head_dim),
      spec.mask == APXINF_ATTENTION_MASK_CAUSAL,
      static_cast<int>(spec.query_start), bindings.decode_meta,
      bindings.scale, stream, static_cast<const float*>(bindings.bias),
        spec.bias_broadcast_batch != 0);
}

template <bool Cooperative>
cudaError_t launch_parallel_impl(Execution& execution) {
  auto* state = static_cast<CustomState*>(execution.provider_state);
  const auto& spec = execution.spec;
  const auto& b = execution.bindings;
#define APXINF_LAUNCH_PARALLEL(T) kernels::launch_parallel_attention<T, Cooperative>( \
      b.query, b.key, b.value, b.output, state->probabilities, \
      static_cast<int>(spec.batch), static_cast<int>(spec.query_tokens), \
      static_cast<int>(spec.key_tokens), static_cast<int>(spec.query_heads), \
      static_cast<int>(spec.kv_heads), static_cast<int>(spec.head_dim), b.scale, \
      spec.mask == APXINF_ATTENTION_MASK_CAUSAL, static_cast<int>(spec.query_start), \
      static_cast<const float*>(b.bias), spec.bias_broadcast_batch != 0, \
      static_cast<cudaStream_t>(b.stream))
  if (spec.dtype == APXINF_DTYPE_F32) return APXINF_LAUNCH_PARALLEL(float);
  if (spec.dtype == APXINF_DTYPE_F16) return APXINF_LAUNCH_PARALLEL(__half);
  return APXINF_LAUNCH_PARALLEL(__nv_bfloat16);
#undef APXINF_LAUNCH_PARALLEL
}

cudaError_t launch_parallel_custom(Execution& execution) {
  return launch_parallel_impl<false>(execution);
}

cudaError_t launch_cooperative_custom(Execution& execution) {
  return launch_parallel_impl<true>(execution);
}

cudaError_t launch_short_f32_custom(Execution& execution) {
  const auto& s = execution.spec;
  const auto& b = execution.bindings;
  return kernels::launch_attention_short_f32(
      b.query, b.key, b.value, b.output, s.query_tokens, s.key_tokens,
      s.query_heads, b.scale, static_cast<cudaStream_t>(b.stream));
}

void destroy_custom(Execution& execution) noexcept {
  auto* state = static_cast<CustomState*>(execution.provider_state);
  if (state == nullptr) return;
  if (state->probabilities != nullptr) cudaFree(state->probabilities);
  delete state;
  execution.provider_state = nullptr;
}

}  // namespace apxinf::attention
