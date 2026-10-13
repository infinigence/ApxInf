#pragma once

#include "norm_types.h"

#ifdef __cplusplus
extern "C" {
#endif

apxinf_status_t apxinf_norm_launch(
    apxinf_runtime_t runtime, const apxinf_norm_spec_t* spec,
    const apxinf_norm_bindings_t* bindings);

/* Per-head RMSNorm in place over [rows, head_dim], for q_norm and k_norm.

   Not a Spec semantic: it writes its input in place where every Spec pipeline
   writes a separate output, and its affine is `(1 + weight)` over a third
   reduction tree again -- see kernels/custom/norm_head.cu, which is kept
   separate for the same reason norm_offset.cu is. */
apxinf_status_t apxinf_norm_head_rms(void* data, const void* weight,
                                          int64_t rows, int64_t head_dim,
                                          float epsilon,
                                          apxinf_cuda_stream_t stream);

#ifdef __cplusplus
}
#endif
