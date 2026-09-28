#pragma once

#include <cuda_runtime.h>
#include <cstdint>

extern "C" int32_t apxinf_quack_m256n256_init();
extern "C" int32_t apxinf_quack_m256n256_forward(
    const void* x_mk, const void* b_nk, void* y_mi,
    int32_t m, int32_t max_active_clusters, cudaStream_t stream);
