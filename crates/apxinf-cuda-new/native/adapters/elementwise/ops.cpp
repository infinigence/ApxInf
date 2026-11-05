// Out-of-place elementwise and activation kernels backing the portable
// `apxinf_core::Backend` trait.
//
// One fixed implementation each and no persisted selection, so per
// `doc/adding-new-kernels.md` section 6 they carry no candidate registry,
// tuning key, or autotuner; they are direct C-ABI forwarders.

#include "../../include/apxinf_cuda/elementwise.h"

#include "../../framework/runtime_internal.h"
#include "../../kernels/custom/elementwise_ops.h"

#include <cstdint>
#include <string>

namespace {

using apxinf::framework::Failure;
using apxinf::framework::abi_boundary;

void check(int status, const char* what) {
  if (status != 0) {
    throw Failure(APXINF_STATUS_PROVIDER_ERROR,
                  std::string(what) + " failed with status " +
                      std::to_string(status));
  }
}

}  // namespace

extern "C" apxinf_status_t apxinf_elementwise_activation_bf16(
    const void* input, void* output, int64_t count, int32_t activation,
    apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (input == nullptr || output == nullptr || count <= 0 || activation < 0 ||
        activation > 2) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid activation arguments");
    }
    check(apxinf::cuda_new::elementwise_ops::activation_bf16(
              input, output, count, static_cast<int>(activation),
              static_cast<cudaStream_t>(stream)),
          "activation");
  });
}

extern "C" apxinf_status_t apxinf_elementwise_mul_bf16(
    const void* a, const void* b, void* output, int64_t count,
    apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (a == nullptr || b == nullptr || output == nullptr || count <= 0) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid mul arguments");
    }
    check(apxinf::cuda_new::elementwise_ops::mul_bf16(
              a, b, output, count, static_cast<cudaStream_t>(stream)),
          "mul");
  });
}

extern "C" apxinf_status_t apxinf_elementwise_add_bf16(
    const void* a, const void* b, void* output, int64_t count,
    apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (a == nullptr || b == nullptr || output == nullptr || count <= 0) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid add arguments");
    }
    check(apxinf::cuda_new::elementwise_ops::add_bf16(
              a, b, output, count, static_cast<cudaStream_t>(stream)),
          "add");
  });
}

extern "C" apxinf_status_t apxinf_elementwise_scale_bf16(
    const void* input, void* output, int64_t count, float factor,
    apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (input == nullptr || output == nullptr || count <= 0) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid scale arguments");
    }
    check(apxinf::cuda_new::elementwise_ops::scale_bf16(
              input, output, count, factor, static_cast<cudaStream_t>(stream)),
          "scale");
  });
}

extern "C" apxinf_status_t apxinf_elementwise_add_bias_bf16(
    const void* input, const void* bias, void* output, int64_t rows,
    int64_t cols, apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (input == nullptr || bias == nullptr || output == nullptr || rows <= 0 ||
        cols <= 0) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid add-bias arguments");
    }
    check(apxinf::cuda_new::elementwise_ops::add_bias_bf16(
              input, bias, output, rows, cols,
              static_cast<cudaStream_t>(stream)),
          "add-bias");
  });
}

extern "C" apxinf_status_t apxinf_elementwise_gather_rows_bf16(
    const void* input, const void* indices, void* output, int64_t rows,
    int64_t cols, apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (input == nullptr || indices == nullptr || output == nullptr ||
        rows <= 0 || cols <= 0) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid gather-rows arguments");
    }
    check(apxinf::cuda_new::elementwise_ops::gather_rows_bf16(
              input, indices, output, rows, cols,
              static_cast<cudaStream_t>(stream)),
          "gather-rows");
  });
}

extern "C" apxinf_status_t apxinf_elementwise_replace_rows_bf16(
    const void* base, const void* replacement, const void* row_map,
    void* output, int64_t rows, int64_t cols, apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (base == nullptr || replacement == nullptr || row_map == nullptr ||
        output == nullptr || rows <= 0 || cols <= 0) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid replace-rows arguments");
    }
    check(apxinf::cuda_new::elementwise_ops::replace_rows_bf16(
              base, replacement, row_map, output, rows, cols,
              static_cast<cudaStream_t>(stream)),
          "replace-rows");
  });
}

extern "C" apxinf_status_t apxinf_elementwise_bias_position_f32_bf16(
    const void* projection, const void* bias, const void* position,
    void* output, int64_t count, int32_t cols, int32_t tokens_per_view,
    apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (projection == nullptr || position == nullptr || output == nullptr ||
        count <= 0 || cols <= 0 || tokens_per_view <= 0) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid bias-position arguments");
    }
    check(apxinf::cuda_new::elementwise_ops::bias_position_f32_bf16(
              projection, bias, position, output, count, cols, tokens_per_view,
              static_cast<cudaStream_t>(stream)),
          "bias-position");
  });
}

extern "C" apxinf_status_t apxinf_elementwise_argmax_remap_bf16(
    const void* logits, uint32_t n, const void* remap, void* out,
    apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (logits == nullptr || remap == nullptr || out == nullptr || n == 0) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid argmax-remap arguments");
    }
    check(apxinf::cuda_new::elementwise_ops::argmax_remap_bf16(
              logits, n, remap, out, static_cast<cudaStream_t>(stream)),
          "argmax-remap");
  });
}
