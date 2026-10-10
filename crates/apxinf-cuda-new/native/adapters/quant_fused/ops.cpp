// Fused rowwise-quantizing epilogues: direct C-ABI forwarders per
// doc/adding-new-kernels.md section 6 (single implementation, nothing to tune).

#include "../../include/apxinf_cuda/quant_fused.h"

#include "../../framework/runtime_internal.h"
#include "../../kernels/custom/quant_ops.h"

#include <cstdint>
#include <string>

namespace {

using apxinf::framework::Failure;
using apxinf::framework::abi_boundary;

void check(int status, const char* what) {
  if (status != 0) {
    throw Failure(APXINF_STATUS_PROVIDER_ERROR,
                  std::string(what) + " failed with status " +
                      std::to_string(status));
  }
}

}  // namespace

extern "C" apxinf_status_t apxinf_quant_rms_norm_rows_bf16_e4m3(
    const void* input, const void* weight, void* output, void* scales,
    int32_t rows, int32_t input_cols, int32_t output_cols, float eps,
    apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (input == nullptr || weight == nullptr || output == nullptr ||
        scales == nullptr || rows <= 0 || input_cols <= 0 ||
        output_cols < input_cols) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid fused RMS quantization arguments");
    }
    check(apxinf::cuda_new::quant_ops::rms_norm_quantize_rows_bf16_e4m3(
              input, weight, output, scales, rows, input_cols, output_cols,
              eps, static_cast<cudaStream_t>(stream)),
          "fused RMS quantization");
  });
}

extern "C" apxinf_status_t apxinf_quant_bias_residual_rms_norm_rows_bf16_e4m3(
    const void* projection, const void* bias, const void* residual,
    const void* weight, void* hidden, void* normalized, void* scales,
    int32_t rows, int32_t cols, int32_t output_cols, float eps,
    apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (projection == nullptr || residual == nullptr || weight == nullptr ||
        hidden == nullptr || normalized == nullptr || scales == nullptr ||
        rows <= 0 || cols <= 0 || output_cols < cols) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid fused residual RMS quantization arguments");
    }
    check(
        apxinf::cuda_new::quant_ops::bias_residual_rms_norm_quantize_rows_bf16_e4m3(
            projection, bias, residual, weight, hidden, normalized, scales,
            rows, cols, output_cols, eps, static_cast<cudaStream_t>(stream)),
        "fused residual RMS quantization");
  });
}

extern "C" apxinf_status_t apxinf_quant_swiglu_rows_bf16_e4m3(
    const void* gate_up, const void* bias, void* output, void* scales,
    int32_t rows, int32_t input_cols, int32_t inner, int32_t output_cols,
    apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (gate_up == nullptr || output == nullptr || scales == nullptr ||
        rows <= 0 || input_cols <= 0 || inner <= 0 || output_cols < inner) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid fused SwiGLU quantization arguments");
    }
    check(apxinf::cuda_new::quant_ops::swiglu_quantize_rows_bf16_e4m3(
              gate_up, bias, output, scales, rows, input_cols, inner,
              output_cols, static_cast<cudaStream_t>(stream)),
          "fused SwiGLU quantization");
  });
}
