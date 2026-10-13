#pragma once

#include "pointwise_types.h"

#ifdef __cplusplus
extern "C" {
#endif

apxinf_status_t apxinf_pointwise_launch(
    apxinf_runtime_t runtime, const apxinf_pointwise_spec_t* spec,
    const apxinf_pointwise_bindings_t* bindings);

/* The family's non-Spec members: BF16-only, one implementation each, so they
   take flat parameters rather than a Spec (doc/cuda-new-design.md section 5).
   Their kernels are in kernels/custom/pointwise_ops.cu. */

/* SwiGLU over a fused [rows, 2*width] gate/up projection, gate first. This is
   the SiLU-gated form; the GEGLU semantic of apxinf_pointwise_launch is the
   GELU one and is not interchangeable. */
apxinf_status_t apxinf_pointwise_swiglu(const void* fused_gate_up, void* output,
                                   int64_t rows, int64_t width,
                                   apxinf_cuda_stream_t stream);

/* accumulator += addend, elementwise. Residual connections. */
apxinf_status_t apxinf_pointwise_add_into(const void* addend, void* accumulator,
                                int64_t count, apxinf_cuda_stream_t stream);

#ifdef __cplusplus
}
#endif
