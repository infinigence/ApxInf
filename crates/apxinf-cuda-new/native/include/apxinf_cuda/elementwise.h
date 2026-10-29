#pragma once

#include "types.h"
#include "status.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Out-of-place elementwise and activation operators backing the portable
   `apxinf_core::Backend` trait. One fixed implementation each, no tuning:
   plain C-ABI entry points, not registry-backed operators. */

/* output[i] = activation(input[i]); activation is 0 none, 1 gelu_tanh,
   2 silu. `count` is the total element count. */
apxinf_status_t apxinf_elementwise_activation_bf16(const void* input,
                                                   void* output, int64_t count,
                                                   int32_t activation,
                                                   apxinf_cuda_stream_t stream);

/* output[i] = a[i] * b[i]. */
apxinf_status_t apxinf_elementwise_mul_bf16(const void* a, const void* b,
                                            void* output, int64_t count,
                                            apxinf_cuda_stream_t stream);

/* output[i] = a[i] + b[i]. */
apxinf_status_t apxinf_elementwise_add_bf16(const void* a, const void* b,
                                            void* output, int64_t count,
                                            apxinf_cuda_stream_t stream);

/* output[i] = input[i] * factor. */
apxinf_status_t apxinf_elementwise_scale_bf16(const void* input, void* output,
                                              int64_t count, float factor,
                                              apxinf_cuda_stream_t stream);

/* output[r, c] = input[r, c] + bias[c], broadcasting bias over rows. */
apxinf_status_t apxinf_elementwise_add_bias_bf16(const void* input,
                                                 const void* bias, void* output,
                                                 int64_t rows, int64_t cols,
                                                 apxinf_cuda_stream_t stream);

/* output[r, :] = input[indices[r], :], u32 row indices. */
apxinf_status_t apxinf_elementwise_gather_rows_bf16(
    const void* input, const void* indices, void* output, int64_t rows,
    int64_t cols, apxinf_cuda_stream_t stream);

/* output[r, :] = row_map[r] == 0xffffffff ? base[r, :]
                                           : replacement[row_map[r], :]. */
apxinf_status_t apxinf_elementwise_replace_rows_bf16(
    const void* base, const void* replacement, const void* row_map,
    void* output, int64_t rows, int64_t cols, apxinf_cuda_stream_t stream);

/* BF16(projection + position + bias?) with F32 inputs; the vision
   patch-embedding epilogue. */
apxinf_status_t apxinf_elementwise_bias_position_f32_bf16(
    const void* projection, const void* bias, const void* position,
    void* output, int64_t count, int32_t cols, int32_t tokens_per_view,
    apxinf_cuda_stream_t stream);

/* *out = remap[argmax(logits)], device argmax over BF16 logits with a u32
   remap table. */
apxinf_status_t apxinf_elementwise_argmax_remap_bf16(
    const void* logits, uint32_t n, const void* remap, void* out,
    apxinf_cuda_stream_t stream);

#ifdef __cplusplus
}
#endif
