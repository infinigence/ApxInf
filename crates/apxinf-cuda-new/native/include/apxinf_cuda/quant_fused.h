#pragma once

#include "types.h"
#include "status.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Fused rowwise-quantizing epilogues (walloss family). Each writes an E4M3
   activation plus one F32 scale per row; padding columns are zero. */

apxinf_status_t apxinf_quant_rms_norm_rows_bf16_e4m3(
    const void* input, const void* weight, void* output, void* scales,
    int32_t rows, int32_t input_cols, int32_t output_cols, float eps,
    apxinf_cuda_stream_t stream);

apxinf_status_t apxinf_quant_bias_residual_rms_norm_rows_bf16_e4m3(
    const void* projection, const void* bias, const void* residual,
    const void* weight, void* hidden, void* normalized, void* scales,
    int32_t rows, int32_t cols, int32_t output_cols, float eps,
    apxinf_cuda_stream_t stream);

apxinf_status_t apxinf_quant_swiglu_rows_bf16_e4m3(
    const void* gate_up, const void* bias, void* output, void* scales,
    int32_t rows, int32_t input_cols, int32_t inner, int32_t output_cols,
    apxinf_cuda_stream_t stream);

#ifdef __cplusplus
}
#endif
