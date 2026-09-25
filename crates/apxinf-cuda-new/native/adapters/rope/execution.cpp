// RoPE and attention-helper kernels: partial RoPE, per-head RMSNorm, query/gate split, output gate.
//
// These ops have a single fixed implementation and no persisted selection,
// so per doc/adding-new-kernels.md §6 they carry no candidate registry,
// tuning key, or autotuner; they are direct C-ABI forwarders.

#include "../../include/apxinf_cuda/attn.h"

#include "../../framework/runtime_internal.h"
#include "../../kernels/custom/attn_ops.h"

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

extern "C" apxinf_status_t apxinf_attn_partial_rope(
    void* data, const void* positions, int64_t tokens, int64_t heads,
    int64_t head_dim, int64_t rotary_dim, float theta,
    apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (data == nullptr || positions == nullptr || !extent(tokens) ||
        !extent(heads) || !extent(head_dim) || !extent(rotary_dim)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid RoPE arguments");
    }
    check(apxinf::cuda::attn_ops::partial_rope(
              data, positions, static_cast<int>(tokens),
              static_cast<int>(heads), static_cast<int>(head_dim),
              static_cast<int>(rotary_dim), theta,
              static_cast<cudaStream_t>(stream)),
          "partial RoPE");
  });
}

extern "C" apxinf_status_t apxinf_attn_head_rms_norm(
    void* data, const void* weight, int64_t rows, int64_t head_dim,
    float epsilon, apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (data == nullptr || weight == nullptr || !extent(rows) ||
        !extent(head_dim)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid head RMSNorm arguments");
    }
    check(apxinf::cuda::attn_ops::head_rms_norm(
              data, weight, static_cast<int>(rows), static_cast<int>(head_dim),
              epsilon, static_cast<cudaStream_t>(stream)),
          "head RMSNorm");
  });
}

extern "C" apxinf_status_t apxinf_attn_split_query_and_gate(
    const void* fused, void* query, void* gate, int64_t tokens, int64_t heads,
    int64_t head_dim, apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (fused == nullptr || query == nullptr || gate == nullptr ||
        !extent(tokens) || !extent(heads) || !extent(head_dim)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid query/gate split arguments");
    }
    check(apxinf::cuda::attn_ops::split_query_and_gate(
              fused, query, gate, static_cast<int>(tokens),
              static_cast<int>(heads), static_cast<int>(head_dim),
              static_cast<cudaStream_t>(stream)),
          "query/gate split");
  });
}

extern "C" apxinf_status_t apxinf_attn_apply_output_gate(
    void* data, const void* gate, int64_t count,
    apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (data == nullptr || gate == nullptr || count <= 0) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid output gate arguments");
    }
    check(apxinf::cuda::attn_ops::apply_output_gate(
              data, gate, count, static_cast<cudaStream_t>(stream)),
          "attention output gate");
  });
}
