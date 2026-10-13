#pragma once

#include "quantization_types.h"

#ifdef __cplusplus
extern "C" {
#endif

apxinf_status_t apxinf_quantization_launch(
    apxinf_runtime_t runtime, const apxinf_quantization_spec_t* spec,
    const apxinf_quantization_bindings_t* bindings);

/* BF16 -> E4M3 against a single per-tensor scale, for the FP8 projections a
   ModelOpt checkpoint stores with scalar scales.

   The FIXED_SCALE_E4M3 semantic of apxinf_quantization_launch computes the same
   arithmetic, but its BF16 path is scalar where this one is 8-wide, and the two
   differ on NaN input. Kernel in kernels/custom/quantization_ops.cu; merging
   them is a measured change, not a rename. */
apxinf_status_t apxinf_quantization_fp8_per_tensor(const void* input, void* output,
                                               int64_t count,
                                               float input_scale,
                                               apxinf_cuda_stream_t stream);

#ifdef __cplusplus
}
#endif
