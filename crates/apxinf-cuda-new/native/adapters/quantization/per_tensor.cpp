#include "../../include/apxinf_cuda/quantization.h"

#include "../../framework/runtime_internal.h"
#include "../../kernels/custom/quantization_ops.h"

#include <string>

namespace {
using apxinf::framework::Failure;
using apxinf::framework::abi_boundary;
void check(int status, const char* what) {
  if (status != 0)
    throw Failure(APXINF_STATUS_PROVIDER_ERROR,
                  std::string(what) + " failed with status " +
                      std::to_string(status));
}
}  // namespace

extern "C" apxinf_status_t apxinf_quantization_fp8_per_tensor(
    const void* input, void* output, int64_t count, float input_scale,
    apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (input == nullptr || output == nullptr || count <= 0 ||
        !(input_scale > 0.0F))
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid FP8 quantization arguments");
    check(apxinf::cuda_new::quantization::fp8_per_tensor(
              input, output, count, input_scale,
              static_cast<cudaStream_t>(stream)),
          "FP8 quantization");
  });
}
