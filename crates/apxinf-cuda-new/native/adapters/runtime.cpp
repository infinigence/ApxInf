#include "../include/apxinf_cuda/runtime.h"
#include "../framework/runtime_internal.h"
#include "gemm/internal.h"

#include <cstring>

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

extern "C" apxinf_status_t apxinf_runtime_device_info(
    apxinf_runtime_t runtime, apxinf_device_info_t* info) {
  return apxinf::framework::abi_boundary([&] {
    if (runtime == nullptr || info == nullptr ||
        info->version != APXINF_DEVICE_INFO_VERSION) {
      throw apxinf::framework::Failure(APXINF_STATUS_INVALID_ARGUMENT,
                                      "invalid device-info argument");
    }
    cudaDeviceProp properties{};
    apxinf::framework::check_cuda(
        cudaGetDeviceProperties(&properties, runtime->device));
    info->compute_major = static_cast<uint32_t>(properties.major);
    info->compute_minor = static_cast<uint32_t>(properties.minor);
    info->multiprocessor_count =
        static_cast<uint32_t>(properties.multiProcessorCount);
    std::strncpy(info->device_name, properties.name,
                 sizeof(info->device_name) - 1);
    info->device_name[sizeof(info->device_name) - 1] = '\0';
  });
}
