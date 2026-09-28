// Copyright 2026 ApxInf contributors.
//
// Kernel-side translation unit for the vendored FlashInfer Cake GDN prefill.
//
// This file must NOT include <cuda.h>: gdn_prefill_generated.cuh declares its
// own CUtensorMap, and the driver header declares another, so the two cannot
// coexist in one translation unit. The descriptors are therefore built in
// flashinfer_gdn_tma.cpp and arrive here as opaque bytes.
//
// See README.md for provenance and the APXINF_GDN_LOG_GATE patch.

// Toggled off while diagnosing a single-chunk state blow-up; see the gate
// hypothesis in README.md. Stock behaviour expects linear-space alpha.
// #define APXINF_GDN_LOG_GATE 1
#include "cake_gdn_prefill_dvsplit_initial_f16io_gatepipe4_ac92807e7ba4.cu"

#include <cuda_runtime.h>

namespace {
constexpr int kBlockThreads = 384;
constexpr int kDynamicSharedBytes = 226048;
}  // namespace

extern "C" int apxinf_flashinfer_gdn_launch(
    const void* map_q, const void* map_k, const void* map_v,
    const void* map_out, float* gate_log, float* beta, int* cu_seqlens,
    float* state, int* state_indices, float* checkpoint_state,
    int* cu_checkpoints, unsigned char* tensor_map_workspace,
    long long state_stride, float scale, int num_seqs, int q_heads,
    int v_heads, int total_tiles, int grid_x, cudaStream_t stream) {
  if (cudaFuncSetAttribute(
          kernel_flashinfer_blackwell_gdn_prefill_dvsplit_initial_f16io_gatepipe4,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          kDynamicSharedBytes) != cudaSuccess) {
    return -5;
  }

  // The same pointer is both initial and output state: the kernel reads the
  // incoming state and writes the advanced one, which is the in-place update
  // decode continues from.
  kernel_flashinfer_blackwell_gdn_prefill_dvsplit_initial_f16io_gatepipe4<<<
      grid_x, kBlockThreads, kDynamicSharedBytes, stream>>>(
      *static_cast<const CUtensorMap*>(map_q),
      *static_cast<const CUtensorMap*>(map_k),
      *static_cast<const CUtensorMap*>(map_v),
      *static_cast<const CUtensorMap*>(map_out), gate_log, beta, cu_seqlens,
      state_indices, state, state, checkpoint_state, cu_checkpoints,
      tensor_map_workspace, state_stride, state_stride, /*checkpoint=*/0,
      scale, num_seqs, q_heads, v_heads, total_tiles);

  return cudaGetLastError() == cudaSuccess ? 0 : -6;
}
