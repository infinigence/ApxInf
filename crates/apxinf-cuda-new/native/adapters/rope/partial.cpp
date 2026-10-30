#include "../../include/apxinf_cuda/rope.h"

#include "../../framework/runtime_internal.h"
#include "../../kernels/custom/rope_partial.h"

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

extern "C" apxinf_status_t apxinf_rope_partial(
    void* data, const void* positions, int64_t tokens, int64_t heads,
    int64_t head_dim, int64_t rotary_dim, float theta,
    apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (data == nullptr || positions == nullptr || !extent(tokens) ||
        !extent(heads) || !extent(head_dim) || !extent(rotary_dim))
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid RoPE arguments");
    check(apxinf::cuda_new::rope::partial_rope(
              data, positions, static_cast<int>(tokens),
              static_cast<int>(heads), static_cast<int>(head_dim),
              static_cast<int>(rotary_dim), theta,
              static_cast<cudaStream_t>(stream)),
          "partial RoPE");
  });
}
