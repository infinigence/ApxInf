#include "../include/apxinf_cuda/relative_attention.h"
#include "../framework/runtime_internal.h"
#include "../kernels/custom/relative_attention.cuh"

#include <climits>
#include <cmath>
#include <cstdint>

namespace {
using apxinf::framework::Failure;
namespace kernels = apxinf::relative_attention::kernels;

constexpr int kRelativeThreads = 128;

void validate_spec(const apxinf_relative_attention_spec_t& spec) {
  const bool supported_dtype = spec.dtype == APXINF_DTYPE_F32 ||
                               spec.dtype == APXINF_DTYPE_F16 ||
                               spec.dtype == APXINF_DTYPE_BF16;
  if (spec.version != APXINF_RELATIVE_ATTENTION_SPEC_VERSION ||
      !supported_dtype || spec.batch <= 0 || spec.length <= 0 ||
      spec.heads <= 0 || spec.head_dim <= 0 || spec.window < 0 ||
      spec.heads > INT32_MAX || spec.head_dim > INT32_MAX ||
      spec.length > INT32_MAX || spec.reserved != 0 ||
      !std::isfinite(spec.scale)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "invalid RelativeAttention Spec");
  }
  if (spec.heads_share_relative > 1) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "invalid RelativeAttention head sharing");
  }
}

void validate_bindings(const apxinf_relative_attention_spec_t& spec,
                       const apxinf_relative_attention_bindings_t& bindings) {
  if (bindings.query == nullptr || bindings.key == nullptr ||
      bindings.value == nullptr || bindings.relative_key == nullptr ||
      bindings.relative_value == nullptr || bindings.output == nullptr) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "missing RelativeAttention binding");
  }
}

template <class T>
cudaError_t launch_typed(const apxinf_relative_attention_spec_t& spec,
                         const apxinf_relative_attention_bindings_t& bindings) {
  const auto* query = static_cast<const T*>(bindings.query);
  const auto* key = static_cast<const T*>(bindings.key);
  const auto* value = static_cast<const T*>(bindings.value);
  const auto* relative_key = static_cast<const T*>(bindings.relative_key);
  const auto* relative_value = static_cast<const T*>(bindings.relative_value);
  auto* output = static_cast<T*>(bindings.output);
  auto stream = static_cast<cudaStream_t>(bindings.stream);
  const int64_t blocks = spec.batch * spec.heads * spec.length;
  if (blocks > INT32_MAX) {
    return cudaErrorInvalidValue;
  }
  const size_t shared_bytes =
      static_cast<size_t>(spec.length) * sizeof(float);
  kernels::relative_attention_kernel<T><<<
      static_cast<int>(blocks), kRelativeThreads, shared_bytes, stream>>>(
      query, key, value, relative_key, relative_value, output, spec.batch,
      spec.length, spec.heads, spec.head_dim, spec.window, spec.scale);
  return cudaGetLastError();
}

cudaError_t launch(const apxinf_relative_attention_spec_t& spec,
                   const apxinf_relative_attention_bindings_t& bindings) {
  switch (spec.dtype) {
    case APXINF_DTYPE_F32:
      return launch_typed<float>(spec, bindings);
    case APXINF_DTYPE_F16:
      return launch_typed<__half>(spec, bindings);
    case APXINF_DTYPE_BF16:
      return launch_typed<__nv_bfloat16>(spec, bindings);
    default:
      return cudaErrorInvalidValue;
  }
}

}  // namespace

extern "C" apxinf_status_t apxinf_relative_attention_launch(
    apxinf_runtime_t runtime, const apxinf_relative_attention_spec_t* spec,
    const apxinf_relative_attention_bindings_t* bindings) {
  return apxinf::framework::abi_boundary([&] {
    if (runtime == nullptr || spec == nullptr || bindings == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "null RelativeAttention argument");
    }
    validate_spec(*spec);
    validate_bindings(*spec, *bindings);
    apxinf::framework::check_cuda(cudaSetDevice(runtime->device));
    apxinf::framework::check_cuda(launch(*spec, *bindings));
  });
}
