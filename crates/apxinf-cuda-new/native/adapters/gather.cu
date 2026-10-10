#include "../include/apxinf_cuda/gather.h"
#include "../framework/runtime_internal.h"
#include "../kernels/custom/gather.cuh"

#include <climits>
#include <cstdint>

namespace {

using apxinf::framework::Failure;
namespace kernels = apxinf::gather::kernels;

constexpr int kThreads = 256;
constexpr int kMaxBlocks = 4096;

bool valid_alignment(uint32_t alignment) {
  return alignment <= 256 && alignment != 0 &&
         (alignment & (alignment - 1)) == 0;
}

bool may_have_bias(uint32_t semantic) {
  return semantic == APXINF_GATHER_SEMANTIC_BIAS_POSITION;
}

void validate_spec(const apxinf_gather_spec_t& spec) {
  if (spec.version != APXINF_GATHER_SPEC_VERSION ||
      spec.semantic > APXINF_GATHER_SEMANTIC_RGB_TO_PATCHES ||
      (spec.dtype != APXINF_DTYPE_F16 && spec.dtype != APXINF_DTYPE_BF16) ||
      spec.has_bias > 1 || spec.nhwc > 1 || spec.rows <= 0 || spec.cols <= 0 ||
      spec.rows > INT32_MAX || spec.cols > INT32_MAX ||
      spec.vocab_size > INT32_MAX || spec.tokens_per_view > INT32_MAX ||
      spec.views > INT32_MAX || spec.image_size > INT32_MAX ||
      spec.patch_size > INT32_MAX || !valid_alignment(spec.input_alignment) ||
      !valid_alignment(spec.bias_alignment) ||
      !valid_alignment(spec.output_alignment)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid Gather Spec");
  }
  if (spec.has_bias != 0 && !may_have_bias(spec.semantic)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "Gather semantic does not take a bias");
  }
  if (spec.semantic == APXINF_GATHER_SEMANTIC_EMBEDDING_LOOKUP &&
      spec.vocab_size == 0) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "embedding lookup requires a vocabulary size");
  }
  if (spec.semantic == APXINF_GATHER_SEMANTIC_BIAS_POSITION &&
      (spec.tokens_per_view == 0 ||
       spec.rows % static_cast<int64_t>(spec.tokens_per_view) != 0)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "bias-position rows must be a whole number of views");
  }
  if (spec.semantic == APXINF_GATHER_SEMANTIC_RGB_TO_PATCHES) {
    if (spec.views == 0 || spec.image_size == 0 || spec.patch_size == 0 ||
        spec.image_size % spec.patch_size != 0) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid RGB-to-patches geometry");
    }
    const int64_t per_side = spec.image_size / spec.patch_size;
    const int64_t expected_rows =
        static_cast<int64_t>(spec.views) * per_side * per_side;
    const int64_t expected_cols =
        3 * static_cast<int64_t>(spec.patch_size) * spec.patch_size;
    if (spec.rows != expected_rows || spec.cols != expected_cols) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "RGB-to-patches shape disagrees with the geometry");
    }
  }
}

void validate_bindings(const apxinf_gather_spec_t& spec,
                       const apxinf_gather_bindings_t& bindings) {
  if (bindings.input == nullptr || bindings.output == nullptr) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "missing Gather binding");
  }
  if (spec.semantic == APXINF_GATHER_SEMANTIC_EMBEDDING_LOOKUP &&
      bindings.ids == nullptr) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "embedding lookup requires token ids");
  }
  if (spec.semantic == APXINF_GATHER_SEMANTIC_BIAS_POSITION &&
      bindings.position == nullptr) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "bias-position requires a position embedding");
  }
  if ((spec.has_bias != 0) != (bindings.bias != nullptr)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "Gather bias binding disagrees with Spec.has_bias");
  }
}

int blocks_for(int64_t count) {
  const int64_t blocks = (count + kThreads - 1) / kThreads;
  if (blocks < 1) return 1;
  return static_cast<int>(blocks < kMaxBlocks ? blocks : kMaxBlocks);
}

template <class T>
cudaError_t launch(const apxinf_gather_spec_t& spec,
                   const apxinf_gather_bindings_t& bindings) {
  const auto* input = static_cast<const T*>(bindings.input);
  const auto* bias = static_cast<const T*>(bindings.bias);
  const auto* position = static_cast<const T*>(bindings.position);
  auto* output = static_cast<T*>(bindings.output);
  const int rows = static_cast<int>(spec.rows);
  const int cols = static_cast<int>(spec.cols);
  const int64_t count = spec.rows * spec.cols;
  auto stream = static_cast<cudaStream_t>(bindings.stream);
  const int grid = blocks_for(count);

  switch (spec.semantic) {
    case APXINF_GATHER_SEMANTIC_EMBEDDING_LOOKUP:
      kernels::embedding_lookup<T><<<grid, kThreads, 0, stream>>>(
          input, bindings.ids, output, rows, cols,
          static_cast<int>(spec.vocab_size));
      break;
    case APXINF_GATHER_SEMANTIC_BIAS_POSITION:
      kernels::bias_position<T><<<grid, kThreads, 0, stream>>>(
          input, bias, position, output, count, cols,
          static_cast<int>(spec.tokens_per_view));
      break;
    case APXINF_GATHER_SEMANTIC_RGB_TO_PATCHES: {
      const auto* images = static_cast<const uint8_t*>(bindings.input);
      if (spec.nhwc != 0) {
        kernels::rgb_u8_to_patches<T, true><<<grid, kThreads, 0, stream>>>(
            images, output, static_cast<int>(spec.views),
            static_cast<int>(spec.image_size),
            static_cast<int>(spec.patch_size));
      } else {
        kernels::rgb_u8_to_patches<T, false><<<grid, kThreads, 0, stream>>>(
            images, output, static_cast<int>(spec.views),
            static_cast<int>(spec.image_size),
            static_cast<int>(spec.patch_size));
      }
      break;
    }
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

}  // namespace

extern "C" apxinf_status_t apxinf_gather_launch(
    apxinf_runtime_t runtime, const apxinf_gather_spec_t* spec,
    const apxinf_gather_bindings_t* bindings) {
  return apxinf::framework::abi_boundary([&] {
    if (runtime == nullptr || spec == nullptr || bindings == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "null Gather argument");
    }
    validate_spec(*spec);
    validate_bindings(*spec, *bindings);
    apxinf::framework::check_cuda(cudaSetDevice(runtime->device));
    apxinf::framework::check_cuda(
        spec->dtype == APXINF_DTYPE_BF16
            ? launch<__nv_bfloat16>(*spec, *bindings)
            : launch<__half>(*spec, *bindings));
  });
}
