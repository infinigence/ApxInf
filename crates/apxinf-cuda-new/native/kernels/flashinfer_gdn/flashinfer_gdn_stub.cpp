// Copyright 2026 ApxInf contributors.
//
// Link stub for builds without an sm100-family target. The FlashInfer Cake
// GDN kernel (flashinfer_gdn_launch.cu) emits tcgen05 and TMA, so it only
// joins sm100 builds; this translation unit takes its place everywhere else
// and reports unsupported. The operator layer never dispatches the GDN
// prefill on devices without the kernel, so the status is a backstop.

#include <cuda_runtime.h>

extern "C" int apxinf_flashinfer_gdn_launch(
    const void* map_q, const void* map_k, const void* map_v,
    const void* map_out, float* gate_log, float* beta, int* cu_seqlens,
    float* state, int* state_indices, float* checkpoint_state,
    int* cu_checkpoints, unsigned char* tensor_map_workspace,
    long long state_stride, float scale, int num_seqs, int q_heads,
    int v_heads, int total_tiles, int grid_x, cudaStream_t stream) {
  (void)map_q;
  (void)map_k;
  (void)map_v;
  (void)map_out;
  (void)gate_log;
  (void)beta;
  (void)cu_seqlens;
  (void)state;
  (void)state_indices;
  (void)checkpoint_state;
  (void)cu_checkpoints;
  (void)tensor_map_workspace;
  (void)state_stride;
  (void)scale;
  (void)num_seqs;
  (void)q_heads;
  (void)v_heads;
  (void)total_tiles;
  (void)grid_x;
  (void)stream;
  return -5;  // unsupported on this build's targets
}
