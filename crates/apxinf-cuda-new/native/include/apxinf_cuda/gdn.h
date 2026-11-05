#pragma once

#include "gdn_types.h"

#ifdef __cplusplus
extern "C" {
#endif

apxinf_status_t apxinf_gdn_launch(
    apxinf_runtime_t runtime, const apxinf_gdn_spec_t* spec,
    const apxinf_gdn_bindings_t* bindings);

apxinf_status_t apxinf_gdn_prepare_fixed(
    int32_t device, int32_t head_k_dim, int32_t head_v_dim,
    int32_t chunk_size);

int32_t apxinf_gdn_qk_bf16_supported(
    int32_t device, int32_t recurrent, int32_t head_k_dim,
    int32_t head_v_dim, int32_t chunk_size);

apxinf_status_t apxinf_gdn_chunk_scan_prepare(int32_t device);

// Scratch bytes apxinf_gdn_prefill needs for TMA rewrites.
int64_t apxinf_gdn_prefill_workspace_bytes(int64_t v_heads,
                                           int64_t num_seqs);

#ifdef __cplusplus
}
#endif
