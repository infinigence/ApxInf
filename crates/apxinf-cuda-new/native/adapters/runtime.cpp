#include "../include/apxinf_cuda/runtime.h"
#include "../framework/runtime_internal.h"
#include "gemm/internal.h"

#include <algorithm>
#include <cstring>

namespace {
thread_local std::string last_error;

std::string default_cache_directory(const cudaDeviceProp& properties) {
  std::string name = properties.name;
  std::transform(name.begin(), name.end(), name.begin(), [](unsigned char character) {
    return character >= 'A' && character <= 'Z' ? character + ('a' - 'A') : character;
  });
  std::string family;
  if (name.find("thor") != std::string::npos) {
    family = "thor";
  } else if (name.find("orin") != std::string::npos) {
    family = "orin";
  } else if (name.find("4090") != std::string::npos) {
    family = "rtx4090";
  } else {
    if (name.rfind("nvidia ", 0) == 0) name.erase(0, 7);
    for (unsigned char character : name) {
      if ((character >= 'a' && character <= 'z') ||
          (character >= '0' && character <= '9')) {
        family += character;
      } else if (!family.empty() && family.back() != '-') {
        family += '-';
      }
    }
    if (!family.empty() && family.back() == '-') family.pop_back();
  }
  int cuda_version = 0;
  int cublas_major = 0;
  int cublas_minor = 0;
  apxinf::framework::check_cuda(cudaRuntimeGetVersion(&cuda_version));
  apxinf::framework::check_cublas(cublasGetProperty(MAJOR_VERSION, &cublas_major));
  apxinf::framework::check_cublas(cublasGetProperty(MINOR_VERSION, &cublas_minor));
  return "configs/tuning/nvidia/" + family + "-sm" +
      std::to_string(properties.major * 10 + properties.minor) + "/cuda" +
      std::to_string(cuda_version / 1000) + "." +
      std::to_string((cuda_version % 1000) / 10) + "-cublas" +
      std::to_string(cublas_major) + "." + std::to_string(cublas_minor);
}
}

namespace apxinf::framework {

void set_last_error(const std::string& message) { last_error = message; }
void clear_last_error() { last_error.clear(); }

}  // namespace apxinf::framework

extern "C" const char* apxinf_last_error() { return last_error.c_str(); }

extern "C" apxinf_status_t apxinf_runtime_create(int32_t device,
                                                   apxinf_runtime_t* output) {
  return apxinf_runtime_create_with_autotune(device, 1, output);
}

extern "C" apxinf_status_t apxinf_runtime_create_with_autotune(
    int32_t device, uint32_t allow_online_tune, apxinf_runtime_t* output) {
  if (output != nullptr) {
    *output = nullptr;
  }
  return apxinf::framework::abi_boundary([&] {
    if (output == nullptr || device < 0 || allow_online_tune > 1) {
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
    runtime->default_cache_dir = default_cache_directory(properties);
    runtime->allow_online_tune = allow_online_tune != 0;
    *output = runtime.release();
  });
}

extern "C" const char* apxinf_runtime_default_cache_dir(apxinf_runtime_t runtime) {
  return runtime != nullptr ? runtime->default_cache_dir.c_str() : nullptr;
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
