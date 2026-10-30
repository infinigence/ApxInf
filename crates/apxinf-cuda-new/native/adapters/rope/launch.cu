#include "../../include/apxinf_cuda/rope.h"
#include "../../framework/runtime_internal.h"
#include "../../kernels/custom/rope.cuh"
#include "../../kernels/rope/multimodal.cuh"

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cuda_bf16.h>
#include <type_traits>

extern "C" cudaError_t apxinf_rope_qkv_split_bias_vision_rope_bf16(
    const void* qkv, const void* bias, void* q_out, void* k_out, void* v_out,
    uint32_t head_dim, uint32_t n_heads, uint32_t seq_len, float theta,
    const void* pos_ids, cudaStream_t stream);
extern "C" cudaError_t
apxinf_rope_qkv_split_bias_vision_rope_precomputed_bf16(
    const void* qkv, const void* bias, void* q_out, void* k_out, void* v_out,
    uint32_t head_dim, uint32_t n_heads, uint32_t seq_len,
    const void* rotation_table, cudaStream_t stream);
extern "C" cudaError_t
apxinf_rope_qkv_split_bias_vision_rope_precomputed_vec2_bf16(
    const void* qkv, const void* bias, void* q_out, void* k_out, void* v_out,
    uint32_t head_dim, uint32_t n_heads, uint32_t seq_len,
    const void* rotation_table, cudaStream_t stream);
extern "C" cudaError_t apxinf_rope_half_split_f16(
    const void* input, void* output, uint32_t head_dim, uint32_t n_heads,
    uint32_t seq_len, float rope_theta, uint32_t pos_offset,
    cudaStream_t stream);

namespace {

#include "../../kernels/primitives/cache.cuh"
#include "../../kernels/primitives/rope.cuh"

using apxinf::framework::Failure;
namespace kernels = apxinf::rope::kernels;

constexpr int kDecodeThreads = 256;

cudaError_t launch_rope_decode_bf16(
    const void* input, void* output, uint32_t head_dim, uint32_t n_heads,
    float rope_theta, const void* position, cudaStream_t stream) {
  dim3 grid((head_dim / 2 + kDecodeThreads - 1) / kDecodeThreads, n_heads, 1);
  dim3 block(kDecodeThreads, 1, 1);
  rope_decode_bf16_kernel<<<grid, block, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<__nv_bfloat16*>(output), head_dim, n_heads, rope_theta,
      static_cast<const uint32_t*>(position));
  return cudaGetLastError();
}
cudaError_t launch_kv_cache_append_decode_bf16(
    void* cache, const void* new_data, uint32_t n_kv_heads,
    uint32_t head_dim, uint32_t max_seq_len, const void* position,
    cudaStream_t stream) {
  dim3 grid((head_dim + kDecodeThreads - 1) / kDecodeThreads, n_kv_heads, 1);
  dim3 block(kDecodeThreads, 1, 1);
  kv_cache_append_decode_bf16_kernel<<<grid, block, 0, stream>>>(
      static_cast<__nv_bfloat16*>(cache),
      static_cast<const __nv_bfloat16*>(new_data), n_kv_heads, head_dim,
      max_seq_len, static_cast<const uint32_t*>(position));
  return cudaGetLastError();
}

cudaError_t launch_rope_k_write_bf16(
    const void* input, void* cache, uint32_t head_dim, uint32_t n_kv_heads,
    uint32_t max_seq_len, float rope_theta, const void* position,
    cudaStream_t stream) {
  dim3 grid((head_dim / 2 + kDecodeThreads - 1) / kDecodeThreads, n_kv_heads,
            1);
  dim3 block(kDecodeThreads, 1, 1);
  rope_k_write_bf16_kernel<<<grid, block, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<__nv_bfloat16*>(cache), head_dim, n_kv_heads, max_seq_len,
      rope_theta, static_cast<const uint32_t*>(position));
  return cudaGetLastError();
}

bool valid_alignment(uint32_t alignment) {
  return alignment <= 256 && alignment != 0 &&
         (alignment & (alignment - 1)) == 0;
}

bool applies_rope(uint32_t semantic) {
  return semantic == APXINF_ROPE_SEMANTIC_SPLIT_QKV_ROPE ||
         semantic == APXINF_ROPE_SEMANTIC_DECODE_QKV_CACHE ||
         semantic == APXINF_ROPE_SEMANTIC_SPLIT_QKV_AXIAL ||
         semantic == APXINF_ROPE_SEMANTIC_PARTIAL_MROPE ||
         semantic == APXINF_ROPE_SEMANTIC_SPLIT_QKV_VISION ||
         semantic == APXINF_ROPE_SEMANTIC_SPLIT_QKV_MROPE_CACHE ||
         semantic == APXINF_ROPE_SEMANTIC_APPLY;
}

bool valid_contract_fields(const apxinf_rope_spec_t& spec) {
  switch (spec.semantic) {
    case APXINF_ROPE_SEMANTIC_SPLIT_QKV_ROPE:
      return spec.layout == APXINF_ROPE_LAYOUT_PACKED_QKV &&
             spec.cache_write == APXINF_ROPE_CACHE_WRITE_TOKEN_MAJOR &&
             spec.rotation_source == APXINF_ROPE_ROTATION_POSITION_OFFSET &&
             spec.rotary_convention == APXINF_ROPE_CONVENTION_HALF_SPLIT;
    case APXINF_ROPE_SEMANTIC_SPLIT_QKV_BIAS:
      return spec.layout == APXINF_ROPE_LAYOUT_PACKED_QKV &&
             spec.cache_write == APXINF_ROPE_CACHE_WRITE_TOKEN_MAJOR &&
             spec.rotation_source == APXINF_ROPE_ROTATION_NONE &&
             spec.rotary_convention == APXINF_ROPE_CONVENTION_NONE;
    case APXINF_ROPE_SEMANTIC_DECODE_QKV_CACHE:
      return spec.layout == APXINF_ROPE_LAYOUT_SEPARATE_QKV &&
             spec.cache_write == APXINF_ROPE_CACHE_WRITE_HEAD_MAJOR &&
             spec.rotation_source == APXINF_ROPE_ROTATION_DEVICE_SCALAR &&
             spec.rotary_convention == APXINF_ROPE_CONVENTION_HALF_SPLIT;
    case APXINF_ROPE_SEMANTIC_SPLIT_QKV_AXIAL:
      return spec.layout == APXINF_ROPE_LAYOUT_PACKED_QKV &&
             spec.cache_write == APXINF_ROPE_CACHE_WRITE_TOKEN_MAJOR &&
             spec.rotation_source == APXINF_ROPE_ROTATION_AXIAL_IDS &&
             spec.rotary_convention ==
                 APXINF_ROPE_CONVENTION_HALF_SPLIT_AXIAL;
    case APXINF_ROPE_SEMANTIC_PARTIAL_MROPE:
      return spec.layout == APXINF_ROPE_LAYOUT_IN_PLACE &&
             spec.cache_write == APXINF_ROPE_CACHE_WRITE_NONE &&
             spec.rotation_source == APXINF_ROPE_ROTATION_MULTIMODAL_IDS &&
             spec.rotary_convention ==
                 APXINF_ROPE_CONVENTION_HALF_SPLIT_INTERLEAVED_AXES;
    case APXINF_ROPE_SEMANTIC_SPLIT_QKV_VISION:
      return spec.layout == APXINF_ROPE_LAYOUT_PACKED_QKV &&
             spec.cache_write == APXINF_ROPE_CACHE_WRITE_TOKEN_MAJOR &&
             (spec.rotation_source == APXINF_ROPE_ROTATION_AXIAL_IDS ||
              spec.rotation_source ==
                  APXINF_ROPE_ROTATION_PRECOMPUTED_TABLE) &&
             (spec.rotary_convention ==
                  APXINF_ROPE_CONVENTION_HALF_SPLIT_AXIAL ||
              spec.rotary_convention ==
                  APXINF_ROPE_CONVENTION_HALF_SPLIT_AXIAL_ROUNDED_INPUT);
    case APXINF_ROPE_SEMANTIC_SPLIT_QKV_MROPE_CACHE:
      return spec.layout == APXINF_ROPE_LAYOUT_PACKED_QKV &&
             spec.cache_write == APXINF_ROPE_CACHE_WRITE_TOKEN_MAJOR &&
             spec.rotation_source == APXINF_ROPE_ROTATION_MULTIMODAL_IDS &&
             spec.rotary_convention ==
                 APXINF_ROPE_CONVENTION_HALF_SPLIT_INTERLEAVED_AXES;
    case APXINF_ROPE_SEMANTIC_APPLY:
      return spec.layout == APXINF_ROPE_LAYOUT_SINGLE_TENSOR &&
             spec.cache_write == APXINF_ROPE_CACHE_WRITE_NONE &&
             ((spec.rotation_source == APXINF_ROPE_ROTATION_POSITION_OFFSET &&
               spec.rotary_convention == APXINF_ROPE_CONVENTION_HALF_SPLIT) ||
              (spec.rotation_source == APXINF_ROPE_ROTATION_MULTIMODAL_IDS &&
               spec.rotary_convention ==
                   APXINF_ROPE_CONVENTION_HALF_SPLIT_INTERLEAVED_AXES) ||
              (spec.rotation_source == APXINF_ROPE_ROTATION_AXIAL_IDS &&
               spec.rotary_convention ==
                   APXINF_ROPE_CONVENTION_HALF_SPLIT_AXIAL));
    default:
      return false;
  }
}

void validate_spec(const apxinf_rope_spec_t& spec) {
  const bool dtype_ok =
      spec.dtype == APXINF_DTYPE_F16 || spec.dtype == APXINF_DTYPE_BF16 ||
      (spec.dtype == APXINF_DTYPE_F32 &&
       spec.semantic == APXINF_ROPE_SEMANTIC_APPLY &&
       spec.rotation_source == APXINF_ROPE_ROTATION_POSITION_OFFSET);
  if (spec.version != APXINF_ROPE_SPEC_VERSION ||
      spec.semantic > APXINF_ROPE_SEMANTIC_APPLY ||
      !dtype_ok ||
      spec.has_bias > 1 || spec.tokens <= 0 || spec.tokens > INT32_MAX ||
      spec.q_heads == 0 || spec.kv_heads == 0 || spec.head_dim == 0 ||
      spec.q_heads > INT32_MAX || spec.kv_heads > INT32_MAX ||
      spec.head_dim > 2048 || spec.head_dim % 2 != 0 ||
      spec.q_heads % spec.kv_heads != 0 ||
      !valid_alignment(spec.qkv_alignment) ||
      !valid_alignment(spec.bias_alignment) ||
      !valid_alignment(spec.q_alignment) ||
      !valid_alignment(spec.kv_alignment) ||
      !valid_alignment(spec.rotation_alignment) ||
      !valid_contract_fields(spec)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid RoPE Spec");
  }
  if (spec.semantic != APXINF_ROPE_SEMANTIC_DECODE_QKV_CACHE &&
      spec.semantic != APXINF_ROPE_SEMANTIC_PARTIAL_MROPE &&
      spec.semantic != APXINF_ROPE_SEMANTIC_APPLY &&
      static_cast<uint64_t>(spec.q_heads) + 2ull * spec.kv_heads > 65535) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid RoPE grid");
  }
  if ((spec.semantic == APXINF_ROPE_SEMANTIC_SPLIT_QKV_BIAS ||
       spec.semantic == APXINF_ROPE_SEMANTIC_SPLIT_QKV_AXIAL) &&
      spec.q_heads != spec.kv_heads) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "unrotated QKV split requires q_heads == kv_heads");
  }
  if (spec.semantic == APXINF_ROPE_SEMANTIC_DECODE_QKV_CACHE &&
      (spec.dtype != APXINF_DTYPE_BF16 || spec.tokens != 1 ||
       spec.has_bias != 0 || spec.cache_capacity <= 0 ||
       spec.cache_capacity > UINT32_MAX)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "invalid dynamic decode RoPE Spec");
  }
  if (spec.semantic != APXINF_ROPE_SEMANTIC_SPLIT_QKV_AXIAL &&
      spec.semantic != APXINF_ROPE_SEMANTIC_PARTIAL_MROPE &&
      spec.semantic != APXINF_ROPE_SEMANTIC_SPLIT_QKV_VISION &&
      spec.semantic != APXINF_ROPE_SEMANTIC_SPLIT_QKV_MROPE_CACHE &&
      spec.semantic != APXINF_ROPE_SEMANTIC_APPLY &&
      (spec.rotary_dim != 0 || spec.section_t != 0 ||
       spec.section_h != 0 || spec.section_w != 0)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "RoPE semantic does not take multimodal fields");
  }
  if (spec.semantic == APXINF_ROPE_SEMANTIC_SPLIT_QKV_AXIAL &&
      (spec.dtype != APXINF_DTYPE_BF16 || spec.head_dim > 256 ||
       spec.head_dim % 4 != 0 || spec.rotary_dim != spec.head_dim ||
       spec.section_t != 0 || spec.section_h != spec.head_dim / 4 ||
       spec.section_w != spec.head_dim / 4 || spec.cache_capacity != 0)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "invalid axial RoPE Spec");
  }
  if (spec.semantic == APXINF_ROPE_SEMANTIC_PARTIAL_MROPE &&
      (spec.dtype != APXINF_DTYPE_BF16 || spec.has_bias != 0 ||
       spec.q_heads > 65535 ||
       spec.rotary_dim != 64 || spec.rotary_dim > spec.head_dim ||
       spec.section_t != 11 || spec.section_h != 11 ||
       spec.section_w != 10 ||
       static_cast<uint64_t>(spec.section_t) + spec.section_h +
               spec.section_w !=
           spec.rotary_dim / 2 ||
       spec.cache_capacity != 0)) {
    throw Failure(APXINF_STATUS_UNSUPPORTED,
                  "unsupported multimodal RoPE section layout");
  }
  if (spec.semantic == APXINF_ROPE_SEMANTIC_SPLIT_QKV_VISION &&
      (spec.dtype != APXINF_DTYPE_BF16 || spec.has_bias != 1 ||
       spec.q_heads != spec.kv_heads || spec.head_dim > 256 ||
       spec.head_dim % 4 != 0 || spec.rotary_dim != spec.head_dim ||
       spec.section_t != 0 || spec.section_h != spec.head_dim / 4 ||
       spec.section_w != spec.head_dim / 4 || spec.cache_capacity != 0)) {
    throw Failure(APXINF_STATUS_UNSUPPORTED,
                  "unsupported packed vision RoPE Spec");
  }
  if (spec.semantic == APXINF_ROPE_SEMANTIC_SPLIT_QKV_MROPE_CACHE &&
      (spec.dtype != APXINF_DTYPE_BF16 || spec.head_dim > 256 ||
       spec.rotary_dim != spec.head_dim ||
       static_cast<uint64_t>(spec.section_t) + spec.section_h +
               spec.section_w !=
           spec.head_dim / 2 ||
       spec.cache_capacity <= 0 || spec.cache_capacity > INT32_MAX ||
       spec.tokens > spec.cache_capacity)) {
    throw Failure(APXINF_STATUS_UNSUPPORTED,
                  "unsupported packed GQA mRoPE cache Spec");
  }
  if (spec.semantic == APXINF_ROPE_SEMANTIC_APPLY) {
    const bool one_dimensional =
        spec.rotation_source == APXINF_ROPE_ROTATION_POSITION_OFFSET &&
        spec.rotary_dim == spec.head_dim && spec.section_t == 0 &&
        spec.section_h == 0 && spec.section_w == 0;
    const bool multimodal =
        spec.dtype == APXINF_DTYPE_BF16 &&
        spec.rotation_source == APXINF_ROPE_ROTATION_MULTIMODAL_IDS &&
        spec.rotary_dim == spec.head_dim &&
        static_cast<uint64_t>(spec.section_h) + spec.section_w <=
            spec.head_dim / 2;
    const bool axial =
        spec.dtype == APXINF_DTYPE_BF16 &&
        spec.rotation_source == APXINF_ROPE_ROTATION_AXIAL_IDS &&
        spec.rotary_dim == spec.head_dim && spec.section_t == 0 &&
        static_cast<uint64_t>(spec.section_h) + spec.section_w ==
            spec.head_dim / 2;
    if (spec.has_bias != 0 || spec.q_heads != spec.kv_heads ||
        spec.cache_capacity != 0 ||
        (!one_dimensional && !multimodal && !axial)) {
      throw Failure(APXINF_STATUS_UNSUPPORTED,
                    "unsupported standalone RoPE Spec");
    }
  }
}

void validate_bindings(const apxinf_rope_spec_t& spec,
                       const apxinf_rope_bindings_t& bindings) {
  const bool partial_mrope =
      spec.semantic == APXINF_ROPE_SEMANTIC_PARTIAL_MROPE;
  const bool single_tensor = spec.semantic == APXINF_ROPE_SEMANTIC_APPLY;
  if ((!partial_mrope && !single_tensor &&
       (bindings.qkv == nullptr || bindings.q == nullptr ||
        bindings.k == nullptr || bindings.v == nullptr)) ||
      (partial_mrope && bindings.q == nullptr) ||
      (single_tensor && (bindings.qkv == nullptr || bindings.q == nullptr))) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "missing RoPE binding");
  }
  if (partial_mrope &&
      (bindings.qkv != nullptr || bindings.bias != nullptr ||
       bindings.k != nullptr || bindings.v != nullptr ||
       bindings.key_input != nullptr || bindings.value_input != nullptr ||
       bindings.rotation_table != nullptr)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "partial multimodal RoPE received unrelated bindings");
  }
  if (single_tensor &&
      (bindings.bias != nullptr || bindings.k != nullptr ||
       bindings.v != nullptr || bindings.key_input != nullptr ||
       bindings.value_input != nullptr || bindings.rotation_table != nullptr ||
       bindings.kv_output_offset != 0)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "standalone RoPE received unrelated bindings");
  }
  if ((spec.has_bias != 0) != (bindings.bias != nullptr)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "RoPE bias binding disagrees with Spec.has_bias");
  }
  if (bindings.kv_output_offset < 0 || bindings.position_offset < 0) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "negative RoPE offset");
  }
  const bool precomputed =
      spec.rotation_source == APXINF_ROPE_ROTATION_PRECOMPUTED_TABLE;
  const bool uses_position_binding =
      spec.rotation_source == APXINF_ROPE_ROTATION_DEVICE_SCALAR ||
      spec.rotation_source == APXINF_ROPE_ROTATION_AXIAL_IDS ||
      spec.rotation_source == APXINF_ROPE_ROTATION_MULTIMODAL_IDS;
  if ((uses_position_binding != (bindings.position != nullptr)) ||
      (precomputed != (bindings.rotation_table != nullptr))) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "RoPE rotation bindings disagree with Spec");
  }
  if (applies_rope(spec.semantic) && !precomputed &&
      (!std::isfinite(bindings.theta) || bindings.theta <= 0.0f)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid RoPE theta");
  }
  if (spec.semantic == APXINF_ROPE_SEMANTIC_SPLIT_QKV_BIAS &&
      (bindings.position_offset != 0 || bindings.kv_output_offset != 0)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "unrotated QKV split does not take offsets");
  }
  if ((spec.semantic == APXINF_ROPE_SEMANTIC_SPLIT_QKV_AXIAL ||
       partial_mrope ||
       (spec.semantic == APXINF_ROPE_SEMANTIC_SPLIT_QKV_VISION &&
        !precomputed)) &&
      (bindings.position == nullptr || bindings.position_offset != 0 ||
       bindings.kv_output_offset != 0)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "missing multimodal RoPE position binding");
  }
  if (spec.semantic == APXINF_ROPE_SEMANTIC_SPLIT_QKV_MROPE_CACHE &&
      (bindings.position == nullptr || bindings.position_offset != 0 ||
       bindings.kv_output_offset < 0 ||
       static_cast<int64_t>(bindings.kv_output_offset) + spec.tokens >
           spec.cache_capacity)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "invalid GQA mRoPE cache binding");
  }
  if (spec.semantic == APXINF_ROPE_SEMANTIC_DECODE_QKV_CACHE &&
      (bindings.key_input == nullptr || bindings.value_input == nullptr ||
       bindings.position == nullptr || bindings.position_offset != 0 ||
       bindings.kv_output_offset != 0)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "missing dynamic decode RoPE binding");
  }
  if (spec.semantic == APXINF_ROPE_SEMANTIC_SPLIT_QKV_VISION &&
      (bindings.position_offset != 0 || bindings.kv_output_offset != 0)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "vision RoPE does not take row offsets");
  }
  if (single_tensor) {
    const bool uses_position_offset =
        spec.rotation_source == APXINF_ROPE_ROTATION_POSITION_OFFSET;
    if ((uses_position_offset && bindings.position != nullptr) ||
        (!uses_position_offset &&
         (bindings.position == nullptr || bindings.position_offset != 0))) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "standalone RoPE offset/position binding mismatch");
    }
  }
  const void* rotation = precomputed
                             ? bindings.rotation_table
                             : static_cast<const void*>(bindings.position);
  if (rotation != nullptr &&
      reinterpret_cast<uintptr_t>(rotation) % spec.rotation_alignment != 0) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "RoPE rotation pointer contradicts recorded alignment");
  }
}

int rope_threads(int head_dim) {
  const int half = head_dim / 2;
  int threads = 32;
  while (threads < half && threads < 1024) threads *= 2;
  return threads;
}

template <class T>
cudaError_t launch_split(const apxinf_rope_spec_t& spec,
                         const apxinf_rope_bindings_t& bindings) {
  const auto* qkv = static_cast<const T*>(bindings.qkv);
  const auto* bias = static_cast<const T*>(bindings.bias);
  auto* q = static_cast<T*>(bindings.q);
  auto* kv_k = static_cast<T*>(bindings.k);
  auto* kv_v = static_cast<T*>(bindings.v);
  const int tokens = static_cast<int>(spec.tokens);
  const int q_heads = static_cast<int>(spec.q_heads);
  const int kv_heads = static_cast<int>(spec.kv_heads);
  const int head_dim = static_cast<int>(spec.head_dim);
  auto stream = static_cast<cudaStream_t>(bindings.stream);

  switch (spec.semantic) {
    case APXINF_ROPE_SEMANTIC_SPLIT_QKV_ROPE: {
      const dim3 grid(static_cast<unsigned>(tokens),
                      static_cast<unsigned>(q_heads + 2 * kv_heads));
      kernels::split_qkv_rope<T><<<grid, rope_threads(head_dim), 0, stream>>>(
          qkv, bias, q, kv_k, kv_v, tokens, q_heads, kv_heads, head_dim,
          bindings.theta, bindings.position_offset, bindings.kv_output_offset);
      break;
    }
    case APXINF_ROPE_SEMANTIC_SPLIT_QKV_BIAS:
      kernels::split_qkv_bias<T><<<tokens, 256, 0, stream>>>(
          qkv, bias, q, kv_k, kv_v, tokens, q_heads * head_dim);
      break;
    case APXINF_ROPE_SEMANTIC_SPLIT_QKV_AXIAL:
      if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        kernels::split_qkv_axial_rope<<<
            tokens, 256, static_cast<size_t>(head_dim) * sizeof(float), stream>>>(
            qkv, bias, bindings.position, q, kv_k, kv_v, tokens, q_heads,
            head_dim, bindings.theta);
      } else {
        return cudaErrorInvalidValue;
      }
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

cudaError_t launch_partial_mrope(const apxinf_rope_spec_t& spec,
                                 const apxinf_rope_bindings_t& bindings) {
  const int half = static_cast<int>(spec.rotary_dim / 2);
  const int threads = half >= 128 ? 128 : ((half + 31) / 32) * 32;
  kernels::partial_multimodal_rope<<<
      dim3(static_cast<unsigned>(spec.tokens), spec.q_heads), threads, 0,
      static_cast<cudaStream_t>(bindings.stream)>>>(
      static_cast<__nv_bfloat16*>(bindings.q),
      bindings.position, static_cast<int>(spec.tokens),
      static_cast<int>(spec.q_heads), static_cast<int>(spec.head_dim),
      static_cast<int>(spec.rotary_dim), static_cast<int>(spec.section_h),
      static_cast<int>(spec.section_w), bindings.theta);
  return cudaGetLastError();
}

cudaError_t launch_vision(apxinf_runtime_t runtime,
                          const apxinf_rope_spec_t& spec,
                          const apxinf_rope_bindings_t& bindings) {
  const auto stream = static_cast<cudaStream_t>(bindings.stream);
  if (spec.rotary_convention ==
      APXINF_ROPE_CONVENTION_HALF_SPLIT_AXIAL_ROUNDED_INPUT) {
    return static_cast<cudaError_t>(
        apxinf::rope::multimodal::vision_qkv_rope_bf16(
            bindings.qkv, bindings.bias, bindings.position, bindings.q,
            bindings.k, bindings.v, static_cast<int>(spec.tokens),
            static_cast<int>(spec.q_heads), static_cast<int>(spec.head_dim),
            bindings.theta, stream));
  }
  if (spec.rotation_source == APXINF_ROPE_ROTATION_AXIAL_IDS) {
    return apxinf_rope_qkv_split_bias_vision_rope_bf16(
        bindings.qkv, bindings.bias, bindings.q, bindings.k, bindings.v,
        spec.head_dim, spec.q_heads, static_cast<uint32_t>(spec.tokens),
        bindings.theta, bindings.position, stream);
  }
  const int sm = runtime->compute_major * 10 + runtime->compute_minor;
  const bool use_vec2 =
      sm == 87 && spec.tokens == 512 && spec.q_heads == 16 &&
      spec.head_dim == 64 &&
      std::getenv("APXINF_GR00T_BF16_LEGACY_VISION_POINTWISE_PACK") == nullptr &&
      std::getenv("APXINF_GR00T_W8A8_LEGACY_VISION_POINTWISE_PACK") == nullptr;
  if (use_vec2) {
    return apxinf_rope_qkv_split_bias_vision_rope_precomputed_vec2_bf16(
        bindings.qkv, bindings.bias, bindings.q, bindings.k, bindings.v,
        spec.head_dim, spec.q_heads, static_cast<uint32_t>(spec.tokens),
        bindings.rotation_table, stream);
  }
  return apxinf_rope_qkv_split_bias_vision_rope_precomputed_bf16(
      bindings.qkv, bindings.bias, bindings.q, bindings.k, bindings.v,
      spec.head_dim, spec.q_heads, static_cast<uint32_t>(spec.tokens),
      bindings.rotation_table, stream);
}

cudaError_t launch_mrope_cache(const apxinf_rope_spec_t& spec,
                               const apxinf_rope_bindings_t& bindings) {
  return static_cast<cudaError_t>(
      apxinf::rope::multimodal::gqa_qkv_mrope_cache_bf16(
          bindings.qkv, bindings.bias, bindings.position, bindings.q,
          bindings.k, bindings.v, static_cast<int>(spec.tokens),
          static_cast<int>(spec.q_heads), static_cast<int>(spec.kv_heads),
          static_cast<int>(spec.head_dim), bindings.theta,
          static_cast<int>(spec.section_h),
          static_cast<int>(spec.section_w), bindings.kv_output_offset,
          static_cast<cudaStream_t>(bindings.stream)));
}

cudaError_t launch_decode(const apxinf_rope_spec_t& spec,
                          const apxinf_rope_bindings_t& bindings) {
  const uint32_t capacity = static_cast<uint32_t>(spec.cache_capacity);
  auto stream = static_cast<cudaStream_t>(bindings.stream);
  cudaError_t status = launch_rope_decode_bf16(
      bindings.qkv, bindings.q, spec.head_dim, spec.q_heads, bindings.theta,
      bindings.position, stream);
  if (status != cudaSuccess) return status;
  status = launch_rope_k_write_bf16(
      bindings.key_input, bindings.k, spec.head_dim, spec.kv_heads, capacity,
      bindings.theta, bindings.position, stream);
  if (status != cudaSuccess) return status;
  return launch_kv_cache_append_decode_bf16(
      bindings.v, bindings.value_input, spec.kv_heads, spec.head_dim, capacity,
      bindings.position, stream);
}

cudaError_t launch_apply(const apxinf_rope_spec_t& spec,
                         const apxinf_rope_bindings_t& bindings) {
  const dim3 block(kDecodeThreads, 1, 1);
  const dim3 grid((spec.head_dim / 2 + kDecodeThreads - 1) / kDecodeThreads,
                  spec.q_heads, static_cast<uint32_t>(spec.tokens));
  auto stream = static_cast<cudaStream_t>(bindings.stream);
  if (spec.rotation_source == APXINF_ROPE_ROTATION_POSITION_OFFSET) {
    if (spec.dtype == APXINF_DTYPE_F16) {
      return apxinf_rope_half_split_f16(
          bindings.qkv, bindings.q, spec.head_dim, spec.q_heads,
          static_cast<uint32_t>(spec.tokens), bindings.theta,
          static_cast<uint32_t>(bindings.position_offset), stream);
    }
    if (spec.dtype == APXINF_DTYPE_F32) {
      rope_batched_f32_kernel<<<grid, block, 0, stream>>>(
          static_cast<const float*>(bindings.qkv),
          static_cast<float*>(bindings.q), spec.head_dim, spec.q_heads,
          static_cast<uint32_t>(spec.tokens), bindings.theta,
          static_cast<uint32_t>(bindings.position_offset));
      return cudaGetLastError();
    }
    rope_batched_bf16_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(bindings.qkv),
        static_cast<__nv_bfloat16*>(bindings.q), spec.head_dim, spec.q_heads,
        static_cast<uint32_t>(spec.tokens), bindings.theta,
        static_cast<uint32_t>(bindings.position_offset));
    return cudaGetLastError();
  }
  if (spec.rotation_source == APXINF_ROPE_ROTATION_MULTIMODAL_IDS) {
    rope_mrope_bf16_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(bindings.qkv),
        static_cast<__nv_bfloat16*>(bindings.q), spec.head_dim, spec.q_heads,
        static_cast<uint32_t>(spec.tokens), bindings.theta, bindings.position,
        spec.section_h, spec.section_w);
    return cudaGetLastError();
  }
  if (spec.rotation_source == APXINF_ROPE_ROTATION_AXIAL_IDS) {
    rope_vision_2d_bf16_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(bindings.qkv),
        static_cast<__nv_bfloat16*>(bindings.q), spec.head_dim, spec.q_heads,
        static_cast<uint32_t>(spec.tokens), bindings.theta,
        bindings.position);
    return cudaGetLastError();
  }
  return cudaErrorInvalidValue;
}

}  // namespace

extern "C" apxinf_status_t apxinf_rope_launch(
    apxinf_runtime_t runtime, const apxinf_rope_spec_t* spec,
    const apxinf_rope_bindings_t* bindings) {
  return apxinf::framework::abi_boundary([&] {
    if (runtime == nullptr || spec == nullptr || bindings == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "null RoPE argument");
    }
    validate_spec(*spec);
    validate_bindings(*spec, *bindings);
    apxinf::framework::check_cuda(cudaSetDevice(runtime->device));
    const cudaError_t status =
        spec->semantic == APXINF_ROPE_SEMANTIC_DECODE_QKV_CACHE
            ? launch_decode(*spec, *bindings)
            : spec->semantic == APXINF_ROPE_SEMANTIC_APPLY
                  ? launch_apply(*spec, *bindings)
            : spec->semantic == APXINF_ROPE_SEMANTIC_PARTIAL_MROPE
                  ? launch_partial_mrope(*spec, *bindings)
            : spec->semantic == APXINF_ROPE_SEMANTIC_SPLIT_QKV_VISION
                  ? launch_vision(runtime, *spec, *bindings)
            : spec->semantic == APXINF_ROPE_SEMANTIC_SPLIT_QKV_MROPE_CACHE
                  ? launch_mrope_cache(*spec, *bindings)
            : (spec->dtype == APXINF_DTYPE_BF16
                   ? launch_split<__nv_bfloat16>(*spec, *bindings)
                   : launch_split<__half>(*spec, *bindings));
    apxinf::framework::check_cuda(status);
  });
}
