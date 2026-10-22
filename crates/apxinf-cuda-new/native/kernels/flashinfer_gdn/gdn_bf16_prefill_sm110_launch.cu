// SPDX-License-Identifier: Apache-2.0
// Kernel-side translation unit for the vendored FlashInfer Cake GDN prefill
// (BF16 I/O, FP32 state, full-DV).
//
// This file must NOT include <cuda.h>: the vendored
// `gdn_prefill_generated.cuh` declares its own `CUtensorMap` and the driver
// header declares another, so the two cannot coexist in one translation unit.
// The descriptors are therefore built in `gdn_bf16_cuda_adapter.cu` and arrive
// here as opaque bytes.
//
// `APXINF_GDN_LOG_GATE` is deliberately never defined; the caller passes
// linear-space `alpha = exp(g)`. See kernels/flashinfer_gdn/README.md.

#define kernel_flashinfer_blackwell_gdn_prefill \
  kernel_flashinfer_blackwell_gdn_prefill_bf16_fulldv
#include "cake_gdn_prefill_bf16_fulldv_sm110.cu"
#undef kernel_flashinfer_blackwell_gdn_prefill

#include <cuda_runtime.h>
#include <cstdint>

namespace {
// Scraped from the vendored source rather than hardcoded independently; both
// are `#undef`'d on its last lines, so they are mirrored here and asserted
// against the build's `-D` values in the adapter's static checks.
constexpr int kBlockThreads = 384;
constexpr int kDynamicSharedBytes = 226048;
}  // namespace

extern "C" int32_t apxinf_gdn_bf16_fulldv_set_attr() {
  return cudaFuncSetAttribute(
             kernel_flashinfer_blackwell_gdn_prefill_bf16_fulldv,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              kDynamicSharedBytes) == cudaSuccess
             ? 0
             : -5;
}

extern "C" int32_t apxinf_gdn_bf16_fulldv_launch(
    const void* map_q, const void* map_k, const void* map_v,
    const void* map_out, float* gate_linear, float* beta, int* cu_seqlens,
    float* state, int* state_indices, float* checkpoint_state,
    int* cu_checkpoints, unsigned char* tensormap_workspace,
    long long state_stride, float scale, int num_seqs, int q_heads,
    int v_heads, int total_tiles, int grid_x, cudaStream_t stream) {
  // `USE_INITIAL_STATE == 0`, so the incoming state pointer is unused and the
  // same buffer is only the output. Passing it twice matches the parameter
  // list of the FP16 sibling already vendored in apxinf-cuda-new.
  kernel_flashinfer_blackwell_gdn_prefill_bf16_fulldv<<<
      grid_x, kBlockThreads, kDynamicSharedBytes, stream>>>(
      *static_cast<const CUtensorMap*>(map_q),
      *static_cast<const CUtensorMap*>(map_k),
      *static_cast<const CUtensorMap*>(map_v),
      *static_cast<const CUtensorMap*>(map_out), gate_linear, beta, cu_seqlens,
      state_indices, state, state, checkpoint_state, cu_checkpoints,
      tensormap_workspace, state_stride, state_stride, /*checkpoint=*/0, scale,
      num_seqs, q_heads, v_heads, total_tiles);
  return cudaGetLastError() == cudaSuccess ? 0 : -6;
}
