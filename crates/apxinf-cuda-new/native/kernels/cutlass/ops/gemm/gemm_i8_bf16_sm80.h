#pragma once

#include <cuda_runtime.h>

namespace apxinf::cuda_new::cutlass_ops {

cudaError_t w8a8_gemm_bf16(const void* activation,
                           const void* weight_output_major,
                           const void* row_scales,
                           const void* column_scales,
                           void* output,
                           int m,
                           int n,
                           int k,
                           cudaStream_t stream);

}  // namespace apxinf::cuda_new::cutlass_ops
