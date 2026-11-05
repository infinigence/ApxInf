// KV-cache append for the portable `apxinf_core::KvCache` trait.
//
// Single fixed implementation, no tuning: a direct C-ABI forwarder per
// `doc/adding-new-kernels.md` section 6.

#include "../../include/apxinf_cuda/cache.h"

#include "../../framework/runtime_internal.h"
#include "../../kernels/custom/cache_ops.h"

#include <cstdint>
#include <string>

namespace {

using apxinf::framework::Failure;
using apxinf::framework::abi_boundary;

}  // namespace

extern "C" apxinf_status_t apxinf_cache_append_bf16(
    const void* new_data, void* cache, int32_t n_kv_heads, int32_t head_dim,
    int32_t max_seq_len, int32_t seq_len, int32_t append_len,
    apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (new_data == nullptr || cache == nullptr || n_kv_heads <= 0 ||
        head_dim <= 0 || max_seq_len <= 0 || seq_len < 0 || append_len <= 0) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid cache append arguments");
    }
    const int status = apxinf::cuda_new::cache_ops::append_bf16(
        cache, new_data, n_kv_heads, head_dim, max_seq_len, seq_len, append_len,
        static_cast<cudaStream_t>(stream));
    if (status != 0) {
      throw Failure(APXINF_STATUS_PROVIDER_ERROR,
                    "KV cache append failed with status " +
                        std::to_string(status));
    }
  });
}
