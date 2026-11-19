#pragma once

#include <stdint.h>

#include "status.h"
#include "types.h"

/* Per-head query/key normalization and the fused norm+rope decode stage.
 *
 * Both launches take BF16 buffers and read their position from a device
 * I32 tensor, so a captured decode graph stays valid as the position
 * advances.
 */
#define APXINF_QK_NORM_ROPE_VERSION 2u

#ifdef __cplusplus
extern "C" {
#endif

/* Fused per-head RMSNorm (weighted, bf16-rounded) + partial rope over
 * [tokens, heads, head_dim]. `positions` is I32 [tokens]. `input` and
 * `output` must not alias. */
apxinf_status_t apxinf_qk_norm_rope_fused(
    const void* input, const void* weight, const void* positions, void* output,
    int64_t tokens, int64_t heads, int64_t head_dim, int64_t rotary_dim,
    float theta, float epsilon, apxinf_cuda_stream_t stream);

/* Copy one `row_elems`-wide BF16 row of `src` into `cache`
 * [capacity, row_elems] at the row index read from `position` (I32 [1]). */
apxinf_status_t apxinf_qk_norm_rope_cache_append_row(
    const void* src, void* cache, const void* position, int64_t row_elems,
    int64_t capacity, apxinf_cuda_stream_t stream);

#ifdef __cplusplus
}
#endif
