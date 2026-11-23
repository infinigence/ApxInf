#include "../../kernels/activation/fixed_bf16.cuh"

extern "C" cudaError_t apxinf_activation_silu_mul_separate_bf16(
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
extern "C" cudaError_t apxinf_activation_silu_mul_quant_bf16_e4m3(
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

extern "C" cudaError_t apxinf_activation_bias_gelu_quant_bf16_e4m3(
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
