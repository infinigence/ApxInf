// SPDX-License-Identifier: Apache-2.0
// Shared contract for the BF16 first-prefill gated delta rule.
//
// Everything up to the point where `q/k/v/alpha/beta/cu_seqlens` are ready,
// plus the state transpose that follows the recurrence, lives in ONE
// translation unit (`gdn_bf16_prefill_prologue.cu`) and is reached from both
// arms through the two C entries below. The AOT object and the vendored
// in-tree kernel sit between those two halves, so the arms differ in *only*
// the GDN prefill launch. Anything duplicated into an arm stops being a
// controlled comparison.
//
// The prologue cannot be `inline` in this header: its kernels are `__global__`
// and would then be emitted with external linkage in every including
// translation unit, which is a multiple-definition link error.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace apxinf_gdn_bf16 {

constexpr int H = 16;           // qk heads
constexpr int HV = 32;          // v heads
constexpr int D = 128;          // head dim
constexpr int KEY = H * D;
constexpr int VALUE = HV * D;
constexpr int CONV = 2 * KEY + VALUE;
constexpr int FLASH_SEQ = 3387;
constexpr int FLASH_PAD = 3392;
constexpr size_t FLASH_GATE_BYTES = size_t(FLASH_PAD) * HV * sizeof(float);
// 10240 is what the AOT kernel's export metadata requires for its per-CTA TMA
// descriptor rewrites (20 CTAs x 512 B). The vendored in-tree kernel needs the
// same plus a 256-byte tail for the three optional pointers it dereferences
// even with state indices and checkpoints disabled. Both arms get the larger
// scratch so the comparison stays controlled; the AOT simply ignores the tail.
constexpr size_t FLASH_WORKSPACE_BYTES = 10240 + 256;
constexpr size_t FLASH_SCRATCH_BYTES =
    2 * FLASH_GATE_BYTES + FLASH_WORKSPACE_BYTES;
constexpr size_t RAW_QK_BYTES =
    size_t(FLASH_SEQ) * 2 * KEY * sizeof(__nv_bfloat16);
static_assert(FLASH_GATE_BYTES % 256 == 0 &&
              FLASH_SCRATCH_BYTES <= RAW_QK_BYTES);
constexpr float SCALE = 0.08838834764831845f;

inline int32_t status(cudaError_t error) { return static_cast<int32_t>(error); }

// Shared argument validation. Identical for both arms by construction.
inline bool arguments_valid(
    const void* zba, const void* conv_weight, const void* new_conv_state,
    const float* A_log, const void* dt_bias_bf16, const void* raw_qk_scratch,
    const void* q, const void* k, const void* v, const void* a, const void* b,
    const float* h_output_vmajor, const float* final_state_kmajor,
    const void* output_pad, const int32_t* cu_seqlens, int32_t seq,
    int32_t seq_pad) {
  return seq == FLASH_SEQ && seq_pad == FLASH_PAD && zba && conv_weight &&
         new_conv_state && A_log && dt_bias_bf16 && raw_qk_scratch && q && k &&
         v && a && b && h_output_vmajor && final_state_kmajor && output_pad &&
         cu_seqlens &&
         (reinterpret_cast<uintptr_t>(raw_qk_scratch) & 255) == 0 &&
         h_output_vmajor != final_state_kmajor;
}

}  // namespace apxinf_gdn_bf16

// Conv + offsets + a/b pack + linear-space gate. On success `*alpha` and
// `*beta` point into the caller's scratch and `*descriptor_workspace` is the
// tail the GDN kernel uses for its per-CTA TMA rewrites.
extern "C" int32_t apxinf_gdn_bf16_prefill_prologue(
    const void* zba, const void* conv_weight, void* new_conv_state,
    const float* A_log, const void* dt_bias_bf16, void* raw_qk_scratch,
    void* q, void* k, void* v, void* a, void* b, int32_t* cu_seqlens,
    int32_t seq, int32_t seq_pad, float** alpha, float** beta,
    void** descriptor_workspace, cudaStream_t stream);

// The GDN kernel writes the recurrent state value-major; the decode path reads
// it key-major.
extern "C" int32_t apxinf_gdn_bf16_prefill_state_transpose(
    const float* h_output_vmajor, float* final_state_kmajor,
    cudaStream_t stream);
