#include "candidates.h"

#include "../../kernels/activation/fixed_bf16.cuh"

namespace apxinf::pointwise::candidates {

cudaError_t launch_bias_gelu_bf16_packed8(
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
      static_cast<const Bf16Pairx8*>(input),
      static_cast<const Bf16Pairx8*>(bias),
      static_cast<Bf16Pairx8*>(output), octet_count, cols / 8);
  return cudaGetLastError();
}

}  // namespace apxinf::pointwise::candidates
