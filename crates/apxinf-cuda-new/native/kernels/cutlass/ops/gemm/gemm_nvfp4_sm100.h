// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace apxinf::cuda::cutlass_ops {

// NVFP4 E2M1 x E2M1, block-16 UE4M3 scales, FP32 accumulation and F16 output.
// The three configurations correspond to the high-performing FlashRT small-M
// variants V1, V6 and V8.
int nvfp4_num_configurations();
size_t nvfp4_workspace_size(int configuration, int m, int n, int k);
size_t nvfp4_scale_workspace_size(int rows, int k);
size_t nvfp4_packed_b_size(int n, int k);
cudaError_t nvfp4_pack_scales(const void* source, void* destination, int rows,
                              int k, cudaStream_t stream);
cudaError_t nvfp4_pack_b(const void* source, void* destination, int n, int k,
                         cudaStream_t stream);
int nvfp4_create(int configuration, const void* a, const void* a_scales,
                 const void* b, const void* b_scales, void* output, int m,
                 int n, int k, float alpha, void* workspace,
                 cudaStream_t stream, void** runner);
int nvfp4_run(void* runner, cudaStream_t stream);
void nvfp4_destroy(void* runner) noexcept;

}  // namespace apxinf::cuda::cutlass_ops
