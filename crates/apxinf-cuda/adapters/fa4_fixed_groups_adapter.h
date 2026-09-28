#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// The safe Rust entry validates the four complete BF16 tensor extents before
// invoking this exact-geometry C ABI. Modules live for the process lifetime.
extern "C" int32_t apxinf_static_fa4_bf16_vfixed_init(cudaStream_t stream);
extern "C" int32_t apxinf_static_fa4_bf16_vfixed_ready(void);
extern "C" int32_t apxinf_static_fa4_bf16_vfixed_forward(
    const void* q, const void* k, const void* v_strided,
    void* output, cudaStream_t stream);
