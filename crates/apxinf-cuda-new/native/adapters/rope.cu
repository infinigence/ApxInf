#include "../include/apxinf_cuda/rope.h"
#include "../framework/runtime_internal.h"
#include "../kernels/custom/rope.cuh"

#include <climits>
#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>

namespace {

#include "../kernels/primitives/cache.cuh"
#include "../kernels/primitives/rope.cuh"

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
         semantic == APXINF_ROPE_SEMANTIC_DECODE_QKV_CACHE;
}

void validate_spec(const apxinf_rope_spec_t& spec) {
  if (spec.version != APXINF_ROPE_SPEC_VERSION ||
      spec.semantic > APXINF_ROPE_SEMANTIC_DECODE_QKV_CACHE ||
      (spec.dtype != APXINF_DTYPE_F16 && spec.dtype != APXINF_DTYPE_BF16) ||
      spec.has_bias > 1 || spec.tokens <= 0 || spec.tokens > INT32_MAX ||
      spec.q_heads == 0 || spec.kv_heads == 0 || spec.head_dim == 0 ||
      spec.q_heads > INT32_MAX || spec.kv_heads > INT32_MAX ||
      spec.head_dim > 2048 || spec.head_dim % 2 != 0 ||
      spec.q_heads % spec.kv_heads != 0 ||
      !valid_alignment(spec.qkv_alignment) ||
      !valid_alignment(spec.bias_alignment) ||
      !valid_alignment(spec.q_alignment) ||
      !valid_alignment(spec.kv_alignment) ||
      !valid_alignment(spec.position_alignment)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid RoPE Spec");
  }
  if (spec.semantic != APXINF_ROPE_SEMANTIC_DECODE_QKV_CACHE &&
      static_cast<uint64_t>(spec.q_heads) + 2ull * spec.kv_heads > 65535) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid RoPE grid");
  }
  if (spec.semantic == APXINF_ROPE_SEMANTIC_SPLIT_QKV_BIAS &&
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
}

void validate_bindings(const apxinf_rope_spec_t& spec,
                       const apxinf_rope_bindings_t& bindings) {
  if (bindings.qkv == nullptr || bindings.q == nullptr ||
      bindings.k == nullptr || bindings.v == nullptr) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "missing RoPE binding");
  }
  if ((spec.has_bias != 0) != (bindings.bias != nullptr)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "RoPE bias binding disagrees with Spec.has_bias");
  }
  if (bindings.kv_output_offset < 0 || bindings.position_offset < 0) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "negative RoPE offset");
  }
  if (applies_rope(spec.semantic) &&
      (!std::isfinite(bindings.theta) || bindings.theta <= 0.0f)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid RoPE theta");
  }
  if (spec.semantic == APXINF_ROPE_SEMANTIC_SPLIT_QKV_BIAS &&
      (bindings.position_offset != 0 || bindings.kv_output_offset != 0)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "unrotated QKV split does not take offsets");
  }
  if (spec.semantic == APXINF_ROPE_SEMANTIC_DECODE_QKV_CACHE &&
      (bindings.key_input == nullptr || bindings.value_input == nullptr ||
       bindings.position == nullptr || bindings.position_offset != 0 ||
       bindings.kv_output_offset != 0)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "missing dynamic decode RoPE binding");
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
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
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
            : (spec->dtype == APXINF_DTYPE_BF16
                   ? launch_split<__nv_bfloat16>(*spec, *bindings)
                   : launch_split<__half>(*spec, *bindings));
    apxinf::framework::check_cuda(status);
  });
}
