// Copyright 2026 ApxInf contributors.
#pragma once

#include <cuda_runtime_api.h>

// Fused rowwise-quantizing epilogues ported bit-identically from the legacy
// walloss kernels: RMS-norm + rowwise E4M3 quantization, bias/residual + RMS
// + quantization, and SwiGLU + quantization. Each writes the quantized
// activation plus one F32 scale per row; padding columns are zero.
namespace apxinf::cuda_new::quant_ops {

// normalized[r, :] = clamp(rms_norm(input[r, :]) * weight / scale_r);
// scales[r] = max|rms_norm * weight| / 448, floored at 1e-12.
int rms_norm_quantize_rows_bf16_e4m3(const void* input, const void* weight,
                                     void* output, void* scales, int rows,
                                     int input_cols, int output_cols, float eps,
                                     cudaStream_t stream);

// hidden = bf16(projection + bias? + residual); normalized = rowwise-quantized
// rms_norm(hidden) * weight. The hidden write rounds to BF16 before the
// square-sum, matching the legacy residual contract.
int bias_residual_rms_norm_quantize_rows_bf16_e4m3(
    const void* projection, const void* bias, const void* residual,
    const void* weight, void* hidden, void* normalized, void* scales, int rows,
    int cols, int output_cols, float eps, cudaStream_t stream);

// output[r, :] = rowwise-quantized silu(gate + bias?) * (up + bias?), where
// gate_up packs gate then up along the column axis.
int swiglu_quantize_rows_bf16_e4m3(const void* gate_up, const void* bias,
                                   void* output, void* scales, int rows,
                                   int input_cols, int inner, int output_cols,
                                   cudaStream_t stream);

}  // namespace apxinf::cuda_new::quant_ops
