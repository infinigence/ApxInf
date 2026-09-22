// Shared runtime contracts for native operator adapters.
//
// Exported C ABI entry points should run inside abi_boundary(); adapters use
// Failure and the status helpers to translate native errors consistently.
// Recipe stores only a winner's provider/implementation/configuration identity,
// never pointers or provider state. On a cache hit, the adapter must resolve the
// exact implementation, revalidate it for the current Spec and Policy, and
// prepare a fresh execution. Per-operator in-memory recipe maps are protected
// by their corresponding mutexes. read_recipe()/write_recipe() provide optional
// serialized disk persistence; an empty directory disables it.
#pragma once

#include "../include/apxinf_cuda/status.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <map>
#include <mutex>
#include <stdexcept>
#include <string>

namespace apxinf::framework {

struct Failure : std::runtime_error {
  apxinf_status_t status;

  Failure(apxinf_status_t status, const std::string& message)
      : std::runtime_error(message), status(status) {}
};

void set_last_error(const std::string& message);
void clear_last_error();

template <class Function>
apxinf_status_t abi_boundary(Function&& function) {
  try {
    function();
    clear_last_error();
    return APXINF_STATUS_OK;
  } catch (const Failure& failure) {
    set_last_error(failure.what());
    return failure.status;
  } catch (const std::exception& exception) {
    set_last_error(exception.what());
    return APXINF_STATUS_INTERNAL_ERROR;
  } catch (...) {
    set_last_error("unknown native exception");
    return APXINF_STATUS_INTERNAL_ERROR;
  }
}

inline void check_cuda(cudaError_t status) {
  if (status != cudaSuccess) {
    throw Failure(APXINF_STATUS_CUDA_ERROR, cudaGetErrorString(status));
  }
}

inline void check_cublas(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw Failure(APXINF_STATUS_PROVIDER_ERROR,
                  "cuBLAS status " + std::to_string(status));
  }
}

struct Recipe {
  uint32_t provider_id;
  uint32_t implementation_id;
  uint32_t implementation_version;
  int32_t configuration;
};

std::string read_recipe(const std::string& directory, const std::string& key);
void write_recipe(const std::string& directory, const std::string& key,
                  const std::string& recipe);

}  // namespace apxinf::framework

struct apxinf_runtime {
  int device = 0;
  std::mutex gemm_mutex;
  std::map<std::string, apxinf::framework::Recipe> gemm_recipes;
  std::mutex attention_mutex;
  std::map<std::string, apxinf::framework::Recipe> attention_recipes;
};
