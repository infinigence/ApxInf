// The fused per-head norm+rope decode stage and the device-indexed KV row
// append.

#include "../include/apxinf_cuda/qk_norm_rope.h"

#include "../framework/runtime_internal.h"

#include "../kernels/custom/qk_norm_rope.cuh"

#include <cuda_runtime.h>

#include <climits>
#include <cstdint>

namespace {

using apxinf::framework::Failure;

// Same value as the product RMSNorm block size: the fused kernel must match
// it to keep the reduction tree bitwise-identical to norm-then-rope.
constexpr int kQkNormRopeThreads = 256;

constexpr int kCacheAppendRowThreads = 256;

void check_geometry(int64_t tokens, int64_t heads, int64_t head_dim,
                    int64_t rotary_dim) {
  if (tokens <= 0 || heads <= 0 || head_dim <= 0) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid attention geometry");
  }
  if (tokens > INT_MAX || heads > INT_MAX || head_dim > INT_MAX) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "attention geometry exceeds the 32-bit kernel index");
  }
  // The fused launch is dim3(tokens, heads): heads lands on grid.y, whose
  // hardware limit is 65535 blocks.
  if (heads > 65535) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "heads exceed the 65535 grid.y block limit");
  }
  if (rotary_dim <= 0 || rotary_dim > head_dim || rotary_dim % 2 != 0) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "rotary width must be even and at most head_dim");
  }
}

}  // namespace

extern "C" apxinf_status_t apxinf_qk_norm_rope_fused(
    const void* input, const void* weight, const void* positions, void* output,
    int64_t tokens, int64_t heads, int64_t head_dim, int64_t rotary_dim,
    float theta, float epsilon, apxinf_cuda_stream_t stream) {
  return apxinf::framework::abi_boundary([&] {
    if (input == nullptr || weight == nullptr || positions == nullptr ||
        output == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "null qk norm rope argument");
    }
    check_geometry(tokens, heads, head_dim, rotary_dim);
    const auto device_stream = static_cast<cudaStream_t>(stream);
    apxinf::qk_norm_rope::kernels::qk_norm_rope_kernel<<<
        dim3(static_cast<unsigned>(tokens), static_cast<unsigned>(heads)),
        kQkNormRopeThreads, 0, device_stream>>>(
        static_cast<const __nv_bfloat16*>(input),
        static_cast<const __nv_bfloat16*>(weight),
        static_cast<const int32_t*>(positions),
        static_cast<__nv_bfloat16*>(output), static_cast<int>(tokens),
        static_cast<int>(heads), static_cast<int>(head_dim),
        static_cast<int>(rotary_dim), theta, epsilon);
    apxinf::framework::check_cuda(cudaGetLastError());
  });
}

extern "C" apxinf_status_t apxinf_qk_norm_rope_cache_append_row(
    const void* src, void* cache, const void* position, int64_t row_elems,
    int64_t capacity, apxinf_cuda_stream_t stream) {
  return apxinf::framework::abi_boundary([&] {
    if (src == nullptr || cache == nullptr || position == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "null cache append argument");
    }
    if (row_elems <= 0 || capacity <= 0 || row_elems > INT_MAX) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid cache append geometry");
    }
    const auto device_stream = static_cast<cudaStream_t>(stream);
    apxinf::qk_norm_rope::kernels::cache_append_row_kernel<<<
        1, kCacheAppendRowThreads, 0, device_stream>>>(
        static_cast<const __nv_bfloat16*>(src),
        static_cast<__nv_bfloat16*>(cache),
        static_cast<const int32_t*>(position), static_cast<int>(row_elems),
        static_cast<long long>(capacity));
    apxinf::framework::check_cuda(cudaGetLastError());
  });
}
