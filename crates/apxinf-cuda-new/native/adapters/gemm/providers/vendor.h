#pragma once

#include "../internal.h"

namespace apxinf::gemm::vendor {

// Shared only by the vendor providers. These buffers are intentionally not
// part of the common Execution: another provider can define a completely
// different private state without modifying the common framework.
struct CommonResources {
  void* projection = nullptr;
  void* unpack_a = nullptr;
  void* unpack_b = nullptr;
  size_t resource_bytes = 0;
  uint32_t projection_dtype = APXINF_DTYPE_F16;

  CommonResources() = default;
  CommonResources(const CommonResources&) = delete;
  CommonResources& operator=(const CommonResources&) = delete;
  ~CommonResources();

  void release() noexcept;
};

void allocate_common_resources(const Spec& spec,
                               CommonResources& resources,
                               bool native_fp8);
uint32_t common_projection_dtype(const Spec& spec);
size_t common_resource_requirements(const Spec& spec, bool native_fp8);
cudaError_t launch_postprocess(const Spec& spec,
                               CommonResources& resources,
                               const apxinf_gemm_bindings_t& bindings,
                               void* projection);

}  // namespace apxinf::gemm::vendor
