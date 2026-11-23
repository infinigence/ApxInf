#include "../../include/apxinf_cuda/gemm.h"

#include "../../framework/runtime_internal.h"
#include "../../kernels/custom/gemv.h"

#include <cstdint>
#include <string>

namespace {
using apxinf::framework::Failure;
using apxinf::framework::abi_boundary;
bool extent(int64_t value) { return value > 0 && value <= INT32_MAX; }
void check(int status, const char* what) {
  if (status != 0)
    throw Failure(APXINF_STATUS_PROVIDER_ERROR,
                  std::string(what) + " failed with status " +
                      std::to_string(status));
}
}  // namespace

extern "C" apxinf_status_t apxinf_gemm_fp8_gemv(
    const void* weight, const void* activation, void* output, int64_t n,
    int64_t k, float alpha, apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (weight == nullptr || activation == nullptr || output == nullptr ||
        !extent(n) || !extent(k))
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid FP8 GEMV arguments");
    check(apxinf::cuda_new::gemm::fp8_gemv(
              weight, activation, output, static_cast<int>(n),
              static_cast<int>(k), alpha, static_cast<cudaStream_t>(stream)),
          "FP8 GEMV");
  });
}

extern "C" apxinf_status_t apxinf_gemm_nvfp4_gemv(
    const void* weight, const void* weight_scales, const void* activation,
    const void* activation_scales, void* output, int64_t n, int64_t k,
    float alpha, apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (weight == nullptr || weight_scales == nullptr ||
        activation == nullptr || activation_scales == nullptr ||
        output == nullptr || !extent(n) || !extent(k))
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid NVFP4 GEMV arguments");
    check(apxinf::cuda_new::gemm::nvfp4_gemv(
              weight, weight_scales, activation, activation_scales, output,
              static_cast<int>(n), static_cast<int>(k), alpha,
              static_cast<cudaStream_t>(stream)),
          "NVFP4 GEMV");
  });
}
