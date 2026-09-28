#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
// Returned only before any kernel launch when this process/device topology is
// unsupported. CUDA and AOT execution failures retain their own status codes.
#define APXINF_FA4_SPLIT_BATCH_UNSUPPORTED_TOPOLOGY 0x20000
int32_t apxinf_static_fa4_split_batch_init(cudaStream_t stream);
int32_t apxinf_static_fa4_split_batch_forward(
    const void* q, const void* k, const void* v,
    void* q2, void* partial, float* lse, void* out, int32_t key_tokens, cudaStream_t stream);
#ifdef __cplusplus
}
#endif
