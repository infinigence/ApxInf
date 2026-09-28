#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initialize the pinned causal SM110 D256 module outside CUDA Graph capture.
int32_t apxinf_static_fa4_d256_init(void);

// Causal Q [1,3387,16,256], KV [1,3387,4,256].
// All buffers are contiguous BF16. Return 0, a CUDA error integer, or
// 0x10000 | AOT status. No allocation, synchronization or module loading.
int32_t apxinf_static_fa4_d256_forward(
    const void* q, const void* k, const void* v,
    void* out, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
