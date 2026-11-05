// Copyright 2026 ApxInf contributors.

#include "quantization_ops.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace apxinf::cuda_new::quantization {
namespace {
// E4M3 with bias 7, saturating at +-448. Values are pre-divided by the
// per-tensor scale, so the representable range maps onto the calibrated one.
// The hardware SATFINITE convert keeps subnormals (the old frexpf ladder
// flushed them to signed zero -- vLLM parity defect D3, report 64).
__device__ __forceinline__ uint8_t to_e4m3(float value) {
  return __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
}

// Eight elements per thread, 16-byte input vectors and 8-byte output
// stores. Same to_e4m3 per element in element order; bit-identical.
__global__ void quantize_fp8_kernel(const __nv_bfloat16* __restrict__ input,
                                    uint8_t* __restrict__ output,
                                    long long count, float inverse_scale) {
  const long long vectors = count / 8;
  const long long index = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (index < vectors) {
    const uint4 in = reinterpret_cast<const uint4*>(input)[index];
    const __nv_bfloat16* iv = reinterpret_cast<const __nv_bfloat16*>(&in);
    uint64_t packed = 0;
#pragma unroll
    for (int element = 0; element < 8; ++element) {
      packed |= static_cast<uint64_t>(
                    to_e4m3(__bfloat162float(iv[element]) * inverse_scale))
                << (8 * element);
    }
    reinterpret_cast<uint64_t*>(output)[index] = packed;
    return;
  }
  const long long tail = vectors * 8 + (index - vectors);
  if (tail < count) {
    output[tail] = to_e4m3(__bfloat162float(input[tail]) * inverse_scale);
  }
}

// Same conversion as to_e4m3; kept as a separate name so the vector kernel's
// dispatch identity survives (it used to be a bit-twiddling variant with the
// same D3 subnormal flush).
__device__ __forceinline__ uint8_t to_e4m3_bits(float value) {
  return to_e4m3(value);
}

__global__ void quantize_fp8_vector_kernel(const uint4* __restrict__ input,
                                           uint2* __restrict__ output,
                                           long long vectors, float inverse_scale) {
  const long long index = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (index >= vectors) return;
  const uint4 packed_input = input[index];
  const auto* elements = reinterpret_cast<const __nv_bfloat16*>(&packed_input);
  uint2 packed_output;
  auto* codes = reinterpret_cast<uint8_t*>(&packed_output);
#pragma unroll
  for (int offset = 0; offset < 8; ++offset) {
    codes[offset] = to_e4m3_bits(__bfloat162float(elements[offset]) * inverse_scale);
  }
  output[index] = packed_output;
}

__device__ __forceinline__ uint16_t to_e4m3_native_pair(float low, float high) {
  // Bare SATFINITE convert: the old small-magnitude zeroing branch replicated
  // the D3 subnormal flush (report 64); the hardware keeps subnormals.
  return __nv_cvt_float2_to_fp8x2(make_float2(low, high), __NV_SATFINITE, __NV_E4M3);
}

__global__ void quantize_fp8_native_pair_kernel(const uint4* __restrict__ input,
                                               uint2* __restrict__ output,
                                               long long vectors, float inverse_scale) {
  const long long index = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (index >= vectors) return;
  const uint4 packed_input = input[index];
  const auto* elements = reinterpret_cast<const __nv_bfloat16*>(&packed_input);
  uint2 packed_output;
  auto* pairs = reinterpret_cast<uint16_t*>(&packed_output);
#pragma unroll
  for (int offset = 0; offset < 4; ++offset)
    pairs[offset] = to_e4m3_native_pair(
        __bfloat162float(elements[2 * offset]) * inverse_scale,
        __bfloat162float(elements[2 * offset + 1]) * inverse_scale);
  output[index] = packed_output;
}
bool vector_elementwise_enabled(const void* source, const void* destination,
                                long long count) {
  const char* flag = std::getenv("APXINF_QWEN38_VECTOR_ELEMENTWISE");
  return flag != nullptr && std::strcmp(flag, "1") == 0 && count % 8 == 0 &&
         reinterpret_cast<uintptr_t>(source) % 16 == 0 &&
         reinterpret_cast<uintptr_t>(destination) % 16 == 0;
}


}  // namespace

int fp8_per_tensor(const void* input, void* output, long long count,
                            float input_scale, cudaStream_t stream) {
  if (count <= 0 || !(input_scale > 0.0f)) return -1;
  const int threads = 256;
  if (vector_elementwise_enabled(input, output, count)) {
    const long long vectors = count / 8;
    const char* native_pair = std::getenv("APXINF_FP8_NATIVE_PAIR");
    if (native_pair != nullptr && std::strcmp(native_pair, "1") == 0) {
      quantize_fp8_native_pair_kernel<<<static_cast<int>((vectors + threads - 1) / threads), threads, 0, stream>>>(
          static_cast<const uint4*>(input), static_cast<uint2*>(output), vectors, 1.0f / input_scale);
      return cudaGetLastError() == cudaSuccess ? 0 : -2;
    }
    quantize_fp8_vector_kernel<<<static_cast<int>((vectors + threads - 1) / threads), threads, 0, stream>>>(
        static_cast<const uint4*>(input), static_cast<uint2*>(output), vectors, 1.0f / input_scale);
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
  }
  // One thread per 8-element vector plus one per tail element.
  const long long work = count / 8 + count % 8;
  const long long blocks = (work + threads - 1) / threads;
  quantize_fp8_kernel<<<static_cast<int>(blocks), threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input), static_cast<uint8_t*>(output),
      count, 1.0f / input_scale);
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

}  // namespace apxinf::cuda_new::quantization
