// VLA attention primitives: direct C-ABI forwarders per
// doc/adding-new-kernels.md section 6 (single implementation, nothing to
// tune).

#include "../../include/apxinf_cuda/vla_attn.h"

#include "../../framework/runtime_internal.h"
#include "../../kernels/custom/vla_attn_ops.h"

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

extern "C" apxinf_status_t apxinf_vla_gqa_qkv_mrope_cache_bf16(
    const void* qkv, const void* bias, const void* position_ids, void* q,
    void* k_cache, void* v_cache, int32_t tokens, int32_t q_heads,
    int32_t kv_heads, int32_t head_dim, float theta, int32_t section_h,
    int32_t section_w, int32_t cache_offset, apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (qkv == nullptr || position_ids == nullptr || q == nullptr ||
        k_cache == nullptr || v_cache == nullptr || tokens <= 0 ||
        q_heads <= 0 || kv_heads <= 0 || q_heads % kv_heads != 0 ||
        head_dim <= 0 || cache_offset < 0) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid GQA QKV mRoPE cache arguments");
    }
    check(apxinf::cuda_new::vla_attn_ops::gqa_qkv_mrope_cache_bf16(
              qkv, bias, static_cast<const uint32_t*>(position_ids), q,
              k_cache, v_cache, tokens, q_heads, kv_heads, head_dim, theta,
              section_h, section_w, cache_offset,
              static_cast<cudaStream_t>(stream)),
          "GQA QKV mRoPE cache");
  });
}

extern "C" apxinf_status_t apxinf_vla_vision_qkv_rope_bf16(
    const void* qkv, const void* bias, const void* position_ids, void* q,
    void* k, void* v, int32_t tokens, int32_t heads, int32_t head_dim,
    float theta, apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (qkv == nullptr || position_ids == nullptr || q == nullptr ||
        k == nullptr || v == nullptr || tokens <= 0 || heads <= 0 ||
        head_dim <= 0) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid vision QKV RoPE arguments");
    }
    check(apxinf::cuda_new::vla_attn_ops::vision_qkv_rope_bf16(
              qkv, bias, static_cast<const uint32_t*>(position_ids), q, k, v,
              tokens, heads, head_dim, theta,
              static_cast<cudaStream_t>(stream)),
          "vision QKV RoPE");
  });
}

extern "C" apxinf_status_t apxinf_vla_segmented_mha_bf16(
    const void* q, const void* k, const void* v, const void* offsets,
    void* output, int32_t segments, int32_t max_tokens, int32_t heads,
    int32_t head_dim, apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (q == nullptr || k == nullptr || v == nullptr || offsets == nullptr ||
        output == nullptr || segments <= 0 || max_tokens <= 0 || heads <= 0 ||
        head_dim <= 0) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid segmented MHA arguments");
    }
    check(apxinf::cuda_new::vla_attn_ops::segmented_mha_bf16(
              q, k, v, offsets, output, segments, max_tokens, heads, head_dim,
              static_cast<cudaStream_t>(stream)),
          "segmented MHA");
  });
}
