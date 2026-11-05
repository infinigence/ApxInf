// SPDX-License-Identifier: Apache-2.0
// Shared contract for the BF16 first-prefill gated delta rule.
//
// Everything up to the point where `q/k/v/alpha/beta/cu_seqlens` are ready,
// plus the state transpose that follows the recurrence, lives in one L0
// translation unit (`kernels/custom/gdn_bf16_prefill_prologue.cu`) and is
// reached from both arms through the C entries declared by its L0 header. The
// AOT object and the vendored
// in-tree kernel sit between those two halves, so the arms differ in *only*
// the GDN prefill launch. Anything duplicated into an arm stops being a
// controlled comparison.
//
#pragma once

#include "../../../kernels/custom/gdn_bf16_prefill.h"

namespace apxinf_gdn_bf16 {

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
