#pragma once

#include <cuda_runtime.h>

namespace apxinf::pointwise::candidates {

cudaError_t launch_bias_gelu_bf16_packed8(
    const void* input, const void* bias, void* output, int rows, int cols,
    cudaStream_t stream);

}  // namespace apxinf::pointwise::candidates
