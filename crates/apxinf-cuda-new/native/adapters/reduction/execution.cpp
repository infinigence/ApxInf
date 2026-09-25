// Reduction / model-head kernels: embedding gather, argmax.
//
// These ops have a single fixed implementation and no persisted selection,
// so per doc/adding-new-kernels.md §6 they carry no candidate registry,
// tuning key, or autotuner; they are direct C-ABI forwarders.

#include "../../include/apxinf_cuda/model.h"

#include "../../framework/runtime_internal.h"
#include "../../kernels/custom/model_ops.h"

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

bool extent(int64_t value) { return value > 0 && value <= INT32_MAX; }

}  // namespace

extern "C" apxinf_status_t apxinf_model_embedding_gather(
    const void* table, const void* ids, void* output, int64_t tokens,
    int64_t hidden, int64_t vocab, apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (table == nullptr || ids == nullptr || output == nullptr ||
        !extent(tokens) || !extent(hidden) || !extent(vocab)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid embedding gather arguments");
    }
    check(apxinf::cuda::model_ops::embedding_gather(
              table, static_cast<const int32_t*>(ids), output,
              static_cast<int>(tokens), static_cast<int>(hidden),
              static_cast<int>(vocab), static_cast<cudaStream_t>(stream)),
          "embedding gather");
  });
}

extern "C" apxinf_status_t apxinf_model_argmax_bf16(
    const void* logits, void* index, int64_t count,
    apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (logits == nullptr || index == nullptr || !extent(count)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid argmax arguments");
    }
    check(apxinf::cuda::model_ops::argmax_bf16(
              logits, static_cast<int32_t*>(index), static_cast<int>(count),
              static_cast<cudaStream_t>(stream)),
          "argmax");
  });
}
