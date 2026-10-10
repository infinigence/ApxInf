#pragma once

#include "rope_types.h"

#ifdef __cplusplus
extern "C" {
#endif

apxinf_status_t apxinf_rope_launch(
    apxinf_runtime_t runtime, const apxinf_rope_spec_t* spec,
    const apxinf_rope_bindings_t* bindings);

/* Standalone half-split RoPE over a contiguous [seq, n_heads, head_dim] BF16
   tensor. Backs the portable `apxinf_core::Backend::rope` trait method, which
   is not a packed-QKV split. */
apxinf_status_t apxinf_rope_apply_batched_bf16(const void* input, void* output,
                                               int32_t n_heads,
                                               int32_t head_dim, int32_t seq_len,
                                               float theta, int32_t pos_offset,
                                               apxinf_cuda_stream_t stream);

/* Multimodal 3D RoPE for Qwen3-VL. `pos_ids` is a device u32 buffer of length
   seq_len*3 holding (t, h, w) per token; sec_h/sec_w are the H and W section
   widths of the head_dim/2 frequency pairs. */
apxinf_status_t apxinf_rope_apply_mrope_bf16(
    const void* input, void* output, int32_t n_heads, int32_t head_dim,
    int32_t seq_len, float theta, const void* pos_ids, int32_t sec_h,
    int32_t sec_w, apxinf_cuda_stream_t stream);

/* Vision-tower 2D RoPE for Qwen3-VL. `pos_ids` is a device u32 buffer of
   length seq_len*2 holding (h, w) per token; the first half of the head_dim/2
   frequency pairs uses h, the second half uses w. */
apxinf_status_t apxinf_rope_apply_vision_2d_bf16(
    const void* input, void* output, int32_t n_heads, int32_t head_dim,
    int32_t seq_len, float theta, const void* pos_ids,
    apxinf_cuda_stream_t stream);

#ifdef __cplusplus
}
#endif
