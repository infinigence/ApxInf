#include "../include/apxinf_cuda/runtime.h"
#include "../framework/runtime_internal.h"
#include "gemm/internal.h"

namespace {
thread_local std::string last_error;
}

namespace apxinf::framework {

void set_last_error(const std::string& message) { last_error = message; }
void clear_last_error() { last_error.clear(); }

}  // namespace apxinf::framework

extern "C" const char* apxinf_last_error() { return last_error.c_str(); }

extern "C" apxinf_status_t apxinf_runtime_create(int32_t device,
                                                   apxinf_runtime_t* output) {
  if (output != nullptr) {
    *output = nullptr;
  }
  return apxinf::framework::abi_boundary([&] {
    if (output == nullptr || device < 0) {
      throw apxinf::framework::Failure(APXINF_STATUS_INVALID_ARGUMENT,
                                      "invalid runtime argument");
    }
    apxinf::framework::check_cuda(cudaSetDevice(device));
    cudaDeviceProp properties{};
    apxinf::framework::check_cuda(cudaGetDeviceProperties(&properties, device));
    const int sm = properties.major * 10 + properties.minor;
    if (apxinf::gemm::compiled_target(sm) == nullptr) {
      throw apxinf::framework::Failure(
          APXINF_STATUS_UNSUPPORTED,
          "current device SM " + std::to_string(sm) +
              " is not included in this GEMM build; rebuild with "
              "APXINF_CUDA_ARCH including this exact architecture");
    }
    auto runtime = std::make_unique<apxinf_runtime>();
    runtime->device = device;
    *output = runtime.release();
  });
}

extern "C" void apxinf_runtime_destroy(apxinf_runtime_t runtime) {
  delete runtime;
}
