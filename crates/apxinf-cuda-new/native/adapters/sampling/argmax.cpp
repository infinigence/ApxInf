#include "../../include/apxinf_cuda/sampling.h"

#include "../../framework/runtime_internal.h"
#include "../../kernels/custom/sampling_argmax.h"

#include <cstdint>
#include <string>

namespace {
using apxinf::framework::Failure;
using apxinf::framework::abi_boundary;
bool extent(int64_t value) { return value > 0 && value <= INT32_MAX; }
void check(int status, const char* what) {
  if (status != 0)
    throw Failure(APXINF_STATUS_PROVIDER_ERROR,
                  std::string(what) + " failed with status " +
                      std::to_string(status));
}
}  // namespace

extern "C" apxinf_status_t apxinf_sampling_argmax(
    const void* logits, void* index, int64_t count,
    apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (logits == nullptr || index == nullptr || !extent(count))
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid argmax arguments");
    check(apxinf::cuda_new::sampling::argmax_bf16(
              logits, static_cast<int32_t*>(index), static_cast<int>(count),
              static_cast<cudaStream_t>(stream)),
          "argmax");
  });
}
