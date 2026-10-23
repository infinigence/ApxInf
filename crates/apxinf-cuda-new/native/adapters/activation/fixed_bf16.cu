// Copyright 2026 ApxInf contributors.
// Fixed BF16 implementation for this operator family; ported without numerical changes.
#include "../../include/apxinf_cuda/transformer_bf16.h"
#include "../../kernels/custom/reduction.cuh"
#ifdef APXINF_GEMM_CUTLASS_SM87_W8A8
#include "../../kernels/cutlass/ops/gemm/gemm_i8_bf16_sm80.h"
#endif

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {

constexpr int kBlockSize = 256;

__device__ __forceinline__ float gelu_tanh(float value) {
  constexpr float kAlpha = 0.7978845608028654f;
  return 0.5f * value *
         (1.0f + tanhf(kAlpha * (value + 0.044715f * value * value * value)));
}

struct alignas(8) Bf16x4 {
  __nv_bfloat162 low;
  __nv_bfloat162 high;
};

struct alignas(8) W8Bf16x4 {
  __nv_bfloat162 low;
  __nv_bfloat162 high;
};

struct alignas(16) Bf16Pairx8 {
  Bf16x4 first;
  Bf16x4 second;
};

// Separate-input variant used when gate/up projections are not packed.  The
// intermediate SiLU value is explicitly rounded to BF16 so this remains
// bit-compatible with the former `silu` then `mul` kernel sequence.
__global__ void silu_mul_separate_bf16_kernel(
    const __nv_bfloat16* gate, const __nv_bfloat16* up,
    __nv_bfloat16* output, uint32_t count) {
  const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count) return;
  const float g = __bfloat162float(gate[gid]);
  const __nv_bfloat16 activated = __float2bfloat16(g / (1.0f + expf(-g)));
  output[gid] = __float2bfloat16(
      __bfloat162float(activated) * __bfloat162float(up[gid]));
}

__global__ void silu_mul_separate_bf16_packed4_kernel(
    const Bf16x4* gate, const Bf16x4* up, Bf16x4* output,
    uint32_t quad_count) {
  uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t stride = blockDim.x * gridDim.x;
  for (; index < quad_count; index += stride) {
    const Bf16x4 g = gate[index];
    const Bf16x4 u = up[index];
    float values[4] = {
        __bfloat162float(g.low.x), __bfloat162float(g.low.y),
        __bfloat162float(g.high.x), __bfloat162float(g.high.y)};
    const float ups[4] = {
        __bfloat162float(u.low.x), __bfloat162float(u.low.y),
        __bfloat162float(u.high.x), __bfloat162float(u.high.y)};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const __nv_bfloat16 activated =
          __float2bfloat16(values[i] / (1.0f + expf(-values[i])));
      values[i] = __bfloat162float(activated) * ups[i];
    }
    output[index] = Bf16x4{
        __floats2bfloat162_rn(values[0], values[1]),
        __floats2bfloat162_rn(values[2], values[3])};
  }
}

// Preserve the existing three-stage numerical contract while avoiding the
// two BF16 intermediates: SiLU rounds to BF16, multiplication rounds to BF16,
// then the calibrated value converts to saturating E4M3.
__global__ void silu_mul_quant_bf16_e4m3_kernel(
    const __nv_bfloat16* gate, const __nv_bfloat16* up,
    __nv_fp8_e4m3* output, int64_t count, float inverse_scale) {
  int64_t pair = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t pair_count = count / 2;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; pair < pair_count; pair += stride) {
    const int64_t index = pair * 2;
    const __nv_bfloat162 gate2 =
        reinterpret_cast<const __nv_bfloat162*>(gate)[pair];
    const __nv_bfloat162 up2 =
        reinterpret_cast<const __nv_bfloat162*>(up)[pair];
    const float gx = __bfloat162float(gate2.x);
    const float gy = __bfloat162float(gate2.y);
    const __nv_bfloat16 sx =
        __float2bfloat16(gx / (1.0f + expf(-gx)));
    const __nv_bfloat16 sy =
        __float2bfloat16(gy / (1.0f + expf(-gy)));
    const __nv_bfloat16 px = __float2bfloat16(
        __bfloat162float(sx) * __bfloat162float(up2.x));
    const __nv_bfloat16 py = __float2bfloat16(
        __bfloat162float(sy) * __bfloat162float(up2.y));
    reinterpret_cast<__nv_fp8x2_e4m3*>(output)[pair] =
        __nv_fp8x2_e4m3(make_float2(
            __bfloat162float(px) * inverse_scale,
            __bfloat162float(py) * inverse_scale));
  }
  if (count % 2 != 0 && pair == pair_count) {
    const float g = __bfloat162float(gate[count - 1]);
    const __nv_bfloat16 s = __float2bfloat16(g / (1.0f + expf(-g)));
    const __nv_bfloat16 product = __float2bfloat16(
        __bfloat162float(s) * __bfloat162float(up[count - 1]));
    output[count - 1] = static_cast<__nv_fp8_e4m3>(
        __bfloat162float(product) * inverse_scale);
  }
}

__global__ void bias_gelu_quant_bf16_e4m3_packed4_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* bias,
    __nv_fp8_e4m3* output, int64_t quad_count, int cols,
    float inverse_scale) {
  int64_t quad_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  const int quads_per_row = cols / 4;
  const Bf16x4* input4 = reinterpret_cast<const Bf16x4*>(input);
  const Bf16x4* bias4 = reinterpret_cast<const Bf16x4*>(bias);
  for (; quad_index < quad_count; quad_index += stride) {
    const Bf16x4 x = input4[quad_index];
    const Bf16x4 b = bias4[quad_index % quads_per_row];
    const float y0 = __bfloat162float(__float2bfloat16(gelu_tanh(
        __bfloat162float(x.low.x) + __bfloat162float(b.low.x))));
    const float y1 = __bfloat162float(__float2bfloat16(gelu_tanh(
        __bfloat162float(x.low.y) + __bfloat162float(b.low.y))));
    const float y2 = __bfloat162float(__float2bfloat16(gelu_tanh(
        __bfloat162float(x.high.x) + __bfloat162float(b.high.x))));
    const float y3 = __bfloat162float(__float2bfloat16(gelu_tanh(
        __bfloat162float(x.high.y) + __bfloat162float(b.high.y))));
    const __nv_fp8x2_e4m3 first(make_float2(
        y0 * inverse_scale, y1 * inverse_scale));
    const __nv_fp8x2_e4m3 second(make_float2(
        y2 * inverse_scale, y3 * inverse_scale));
    reinterpret_cast<uint32_t*>(output)[quad_index] =
        static_cast<uint32_t>(first.__x) |
        (static_cast<uint32_t>(second.__x) << 16);
  }
}

__device__ __forceinline__ Bf16x4 bias_gelu_bf16_quad(
    const Bf16x4 input, const Bf16x4 bias) {
  const float v0 = gelu_tanh(
      __bfloat162float(input.low.x) + __bfloat162float(bias.low.x));
  const float v1 = gelu_tanh(
      __bfloat162float(input.low.y) + __bfloat162float(bias.low.y));
  const float v2 = gelu_tanh(
      __bfloat162float(input.high.x) + __bfloat162float(bias.high.x));
  const float v3 = gelu_tanh(
      __bfloat162float(input.high.y) + __bfloat162float(bias.high.y));
  return Bf16x4{
      __floats2bfloat162_rn(v0, v1), __floats2bfloat162_rn(v2, v3)};
}

__global__ void bias_gelu_bf16_packed8_kernel(
    const Bf16Pairx8* input, const Bf16Pairx8* bias, Bf16Pairx8* output,
    int64_t octet_count, int octets_per_row) {
  int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < octet_count; index += stride) {
    const Bf16Pairx8 value = input[index];
    const Bf16Pairx8 row_bias = bias[index % octets_per_row];
    output[index] = Bf16Pairx8{
        bias_gelu_bf16_quad(value.first, row_bias.first),
        bias_gelu_bf16_quad(value.second, row_bias.second)};
  }
}

}  // namespace

extern "C" cudaError_t apxinf_transformer_bf16_silu_mul_separate_bf16(
    const void* gate, const void* up, void* output, uint32_t count,
    cudaStream_t stream) {
  const bool packed =
      (count & 3U) == 0 &&
      (reinterpret_cast<uintptr_t>(gate) & 7U) == 0 &&
      (reinterpret_cast<uintptr_t>(up) & 7U) == 0 &&
      (reinterpret_cast<uintptr_t>(output) & 7U) == 0;
  if (packed) {
    const uint32_t quad_count = count / 4;
    int blocks = static_cast<int>(
        (quad_count + kBlockSize - 1) / kBlockSize);
    blocks = blocks > 512 ? 512 : blocks;
    silu_mul_separate_bf16_packed4_kernel<<<blocks, kBlockSize, 0,
        stream>>>(
        (const Bf16x4*)gate, (const Bf16x4*)up, (Bf16x4*)output, quad_count);
    return cudaGetLastError();
  }
  dim3 grid((count + kBlockSize - 1) / kBlockSize, 1, 1);
  dim3 block(kBlockSize, 1, 1);
  silu_mul_separate_bf16_kernel<<<grid, block, 0, stream>>>(
      (const __nv_bfloat16*)gate, (const __nv_bfloat16*)up,
      (__nv_bfloat16*)output, count);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_transformer_bf16_silu_mul_quant_bf16_e4m3(
    const void* gate, const void* up, void* output, int64_t count,
    float scale, cudaStream_t stream) {
  if (gate == nullptr || up == nullptr || output == nullptr || count <= 0 ||
      !(scale > 0.0f)) return cudaErrorInvalidValue;
  constexpr int threads = 256;
  const int64_t pairs = (count + 1) / 2;
  int blocks = static_cast<int>((pairs + threads - 1) / threads);
  blocks = blocks > 1024 ? 1024 : blocks;
  silu_mul_quant_bf16_e4m3_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(gate),
      static_cast<const __nv_bfloat16*>(up),
      static_cast<__nv_fp8_e4m3*>(output), count, 1.0f / scale);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_transformer_bf16_bias_gelu_quant_bf16_e4m3(
    const void* input, const void* bias, void* output, int rows, int cols,
    float scale, cudaStream_t stream) {
  if (!input || !bias || !output || rows <= 0 || cols <= 0 || cols % 4 != 0 ||
      !(scale > 0.0f) || reinterpret_cast<uintptr_t>(input) % alignof(Bf16x4) != 0 ||
      reinterpret_cast<uintptr_t>(bias) % alignof(Bf16x4) != 0 ||
      reinterpret_cast<uintptr_t>(output) % alignof(uint32_t) != 0)
    return cudaErrorInvalidValue;
  constexpr int threads = 256;
  const int64_t quad_count = static_cast<int64_t>(rows) * cols / 4;
  int blocks = static_cast<int>((quad_count + threads - 1) / threads);
  blocks = blocks > 1024 ? 1024 : blocks;
  bias_gelu_quant_bf16_e4m3_packed4_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(bias),
      static_cast<__nv_fp8_e4m3*>(output), quad_count, cols, 1.0f / scale);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_transformer_bf16_bias_gelu_bf16_packed8(
    const void* input, const void* bias, void* output, int rows, int cols,
    cudaStream_t stream) {
  if (input == nullptr || bias == nullptr || output == nullptr || rows <= 0 ||
      cols <= 0 || cols % 8 != 0 ||
      reinterpret_cast<uintptr_t>(input) % alignof(Bf16Pairx8) != 0 ||
      reinterpret_cast<uintptr_t>(bias) % alignof(Bf16Pairx8) != 0 ||
      reinterpret_cast<uintptr_t>(output) % alignof(Bf16Pairx8) != 0)
    return cudaErrorInvalidValue;
  const int64_t octet_count = static_cast<int64_t>(rows) * cols / 8;
  const int threads = rows >= 512 ? 256 : 128;
  const int blocks = static_cast<int>((octet_count + threads - 1) / threads);
  bias_gelu_bf16_packed8_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const Bf16Pairx8*>(input), static_cast<const Bf16Pairx8*>(bias),
      static_cast<Bf16Pairx8*>(output), octet_count, cols / 8);
  return cudaGetLastError();
}
