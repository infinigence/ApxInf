#include "../../kernels/elementwise/fixed_f16.cuh"

extern "C" cudaError_t apxinf_elementwise_bias_f16(
    const void* input, const void* bias, void* output, int rows, int cols,
    cudaStream_t stream) {
  if (rows <= 0 || cols <= 0) return cudaErrorInvalidValue;
  int64_t count = static_cast<int64_t>(rows) * cols;
  bias_f16_kernel<<<clamped_blocks(count), 256, 0, stream>>>(
      static_cast<const half*>(input), static_cast<const half*>(bias),
      static_cast<half*>(output), count, cols);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_elementwise_euler_update_f16(
    const void* state, const void* velocity, void* output, int64_t count,
    float dt, cudaStream_t stream) {
  if (count <= 0) return cudaErrorInvalidValue;
  euler_update_f16_kernel<<<clamped_blocks(count), 256, 0, stream>>>(
      static_cast<const half*>(state), static_cast<const half*>(velocity),
      static_cast<half*>(output), count, dt);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_elementwise_add_f16(
    const void* a, const void* b, void* output, uint32_t count,
    cudaStream_t stream) {
  const uintptr_t aa = reinterpret_cast<uintptr_t>(a);
  const uintptr_t ba = reinterpret_cast<uintptr_t>(b);
  const uintptr_t oa = reinterpret_cast<uintptr_t>(output);
  if ((count % 8u) == 0u && (aa % 16u) == 0u && (ba % 16u) == 0u &&
      (oa % 16u) == 0u) {
    add_f16_vec8_kernel<<<(count / 8 + kThreads - 1) / kThreads,
                          kThreads, 0, stream>>>(
        (const float4*)a, (const float4*)b, (float4*)output, count / 8);
  } else {
    dim3 grid((count + kThreads - 1) / kThreads, 1, 1);
    dim3 block(kThreads, 1, 1);
    add_f16_kernel<<<grid, block, 0, stream>>>(
        (const half*)a, (const half*)b, (half*)output, count);
  }
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_elementwise_scale_f16(
    const void* input, void* output, uint32_t count, float scale,
    cudaStream_t stream) {
  const uintptr_t input_address = reinterpret_cast<uintptr_t>(input);
  const uintptr_t output_address = reinterpret_cast<uintptr_t>(output);
  if ((count % 8u) == 0u && (input_address % 16u) == 0u &&
      (output_address % 16u) == 0u) {
    scale_f16_vec8_kernel<<<(count / 8 + kThreads - 1) / kThreads,
                            kThreads, 0, stream>>>(
        (const float4*)input, (float4*)output, count / 8, scale);
  } else {
    dim3 grid((count + kThreads - 1) / kThreads, 1, 1);
    dim3 block(kThreads, 1, 1);
    scale_f16_kernel<<<grid, block, 0, stream>>>(
        (const half*)input, (half*)output, count, scale);
  }
  return cudaGetLastError();
}
