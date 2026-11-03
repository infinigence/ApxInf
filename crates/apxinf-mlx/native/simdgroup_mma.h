
#include <metal_simdgroup_matrix>

// Fragment coordinates and FP32 arithmetic follow MLX 0.32.3 Steel mma.h.
// Copyright 2024 Apple Inc. Used under the MLX MIT license.
inline float2 g07_mma(float2 a, float2 b, float2 c) {
    metal::simdgroup_float8x8 am, bm, cm, dm;
    am.thread_elements()[0] = a.x;
    am.thread_elements()[1] = a.y;
    bm.thread_elements()[0] = b.x;
    bm.thread_elements()[1] = b.y;
    cm.thread_elements()[0] = c.x;
    cm.thread_elements()[1] = c.y;
    metal::simdgroup_multiply_accumulate(dm, am, bm, cm);
    return float2(dm.thread_elements()[0], dm.thread_elements()[1]);
}
