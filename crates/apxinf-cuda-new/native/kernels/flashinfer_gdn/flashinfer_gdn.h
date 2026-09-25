// Copyright 2026 ApxInf contributors.
#pragma once

#include <cuda_runtime_api.h>
#include <cstddef>

namespace apxinf::cuda::flashinfer_gdn {

// Chunked gated delta rule over a whole prompt, on the vendored FlashInfer
// Cake kernel.
//
//   q, k   [tokens, q_heads, 128]  FP16, already L2-normalized by the caller
//   v      [tokens, v_heads, 128]  FP16
//   out    [tokens, v_heads, 128]  FP16
//   gate   [tokens, v_heads]       f32, NATURAL LOG decay (see the
//                                  APXINF_GDN_LOG_GATE patch in README.md)
//   beta   [tokens, v_heads]       f32
//   state  [v_heads, 128, 128]     f32, [H, V, K] order, updated in place
//   cu_seqlens [num_seqs + 1]      i32, device
//
// `scale` multiplies the q-side products; pass 1/sqrt(128) rather than
// pre-scaling q. Unlike our own scan this imposes no padding requirement on
// `tokens`: the kernel clamps its TMA descriptors per sequence.
int prefill(const void* q, const void* k, const void* v, void* out,
            const void* gate_log, const void* beta, const void* cu_seqlens,
            void* state, void* tensor_map_workspace, int tokens, int q_heads,
            int v_heads, int num_seqs, float scale, cudaStream_t stream);

// Bytes of scratch `prefill` needs for its per-CTA TMA descriptor rewrites.
size_t tensor_map_workspace_bytes(int v_heads, int num_seqs);

}  // namespace apxinf::cuda::flashinfer_gdn
