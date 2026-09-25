#pragma once

#include "types.h"
#include "status.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Gated DeltaNet primitives. See kernels/custom/gdn_ops.h for the recurrence.

   The update rule is implemented from the architecture's documented form and
   has not yet been checked against a reference engine running this
   checkpoint. */

apxinf_status_t apxinf_gdn_recurrent_step(
    void* state, const void* q, const void* k, const void* v,
    const void* decay, const void* beta, void* output, int64_t v_heads,
    int64_t k_heads, int64_t v_dim, int64_t k_dim,
    apxinf_cuda_stream_t stream);

apxinf_status_t apxinf_gdn_gated_norm(const void* input, const void* gate,
                                      const void* weight, void* output,
                                      int64_t heads, int64_t head_dim,
                                      float epsilon,
                                      apxinf_cuda_stream_t stream);

apxinf_status_t apxinf_gdn_causal_conv_step(void* window, const void* input,
                                            const void* weight, void* output,
                                            int64_t channels,
                                            int64_t kernel_width,
                                            apxinf_cuda_stream_t stream);

// Chunked gated delta rule on the vendored FlashInfer Cake kernel.
//
// q/k/v/out are FP16 and q/k must already be L2-normalized; `gate_log` is the
// natural-log decay. See native/kernels/flashinfer_gdn/README.md.
apxinf_status_t apxinf_gdn_widen_f16_to_bf16(const void* input, void* output,
                                             int64_t count,
                                             apxinf_cuda_stream_t stream);

apxinf_status_t apxinf_gdn_prepare_flashinfer(
    const void* fused, void* q_out, void* k_out, void* v_out, const void* g,
    void* alpha, int64_t tokens, int64_t row_width, int64_t k_heads,
    int64_t v_heads, int64_t dim, float epsilon, apxinf_cuda_stream_t stream);

apxinf_status_t apxinf_flashinfer_gdn_prefill(
    const void* q, const void* k, const void* v, void* out,
    const void* gate_log, const void* beta, const void* cu_seqlens,
    void* state, void* tensor_map_workspace, int64_t tokens, int64_t q_heads,
    int64_t v_heads, int64_t num_seqs, float scale,
    apxinf_cuda_stream_t stream);

// Scratch bytes apxinf_flashinfer_gdn_prefill needs for TMA rewrites.
int64_t apxinf_flashinfer_gdn_workspace_bytes(int64_t v_heads,
                                              int64_t num_seqs);

apxinf_status_t apxinf_gdn_causal_conv_forward(
    const void* input, const void* weight, void* output, void* window,
    int64_t tokens, int64_t channels, int64_t kernel_width,
    apxinf_cuda_stream_t stream);

apxinf_status_t apxinf_gdn_decay_and_beta_seq(
    const void* a, const void* b, const void* a_log, const void* dt_bias,
    void* decay, void* beta, int64_t tokens, int64_t heads,
    apxinf_cuda_stream_t stream);

apxinf_status_t apxinf_gdn_gated_norm_seq(
    const void* input, const void* gate, const void* weight, void* output,
    int64_t tokens, int64_t heads, int64_t head_dim, float epsilon,
    apxinf_cuda_stream_t stream);

apxinf_status_t apxinf_gdn_l2_normalize_heads(void* data, int64_t heads,
                                              int64_t head_dim, float epsilon,
                                              apxinf_cuda_stream_t stream);

apxinf_status_t apxinf_gdn_decay_and_beta(const void* a, const void* b,
                                          const void* a_log,
                                          const void* dt_bias, void* decay,
                                          void* beta, int64_t heads,
                                          apxinf_cuda_stream_t stream);


apxinf_status_t apxinf_gdn_chunk_scan(
    const void* q, const void* k, const void* v, const void* g,
    const void* beta, void* out, void* state, int64_t seq_padded,
    int64_t v_heads, int64_t k_heads, int64_t chunk_size, int64_t k_dim,
    int64_t num_chunks, int64_t q_row_stride, int64_t k_row_stride,
    int64_t v_row_stride, apxinf_cuda_stream_t stream);

#ifdef __cplusplus
}
#endif
