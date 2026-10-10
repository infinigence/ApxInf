// Copyright 2026 ApxInf contributors.
//
// Host adapter for the optional CuTe DSL AOT object selected from the shared
// APXINF_CUDA_AOT_MANIFEST bundle. Generated objects stay outside the source
// tree while their recipe, checksum and ABI are verified at build time.

#include "gemm_nvfp4_sm100.h"
#include "dense_swiglu.h"

#include <cuda_runtime_api.h>

#include <cstdlib>
#include <mutex>

namespace apxinf::cuda_new::cutlass_ops {
namespace {

std::once_flag module_once;
cudaLibrary_t module = nullptr;
cudaError_t module_status = cudaSuccess;

void load_module() {
  void** library = reinterpret_cast<void**>(&module);
  void* init_args[] = {&library, &module_status};
  _mlir_dense_swiglu_cuda_init(init_args);
  if (module_status != cudaSuccess) return;
  int device = 0;
  module_status = cudaGetDevice(&device);
  if (module_status != cudaSuccess) return;
  void* load_args[] = {&library, &device, &module_status};
  _mlir_dense_swiglu_cuda_load_to_device(load_args);
}

int32_t configured_max_active_clusters() {
  const char* value = std::getenv("APXINF_QWEN38_FUSED_FC1_CLUSTERS");
  if (value == nullptr) return 10;
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  return end != value && *end == '\0' && parsed >= 1 && parsed <= 10
             ? static_cast<int32_t>(parsed)
             : 0;
}

}  // namespace

int nvfp4_dense_swiglu_aot(
    const void* a, const void* b, const void* a_sf, const void* b_sf,
    void* c, void* c_sf, const void* alpha, const void* input_global_scale,
    const void* down_inverse_global_scale, const void* tile_groups,
    const void* tile_limits, const void* token_map, const void* tile_count,
    int rows, int n, int k, cudaStream_t stream) {
  if (rows != 2048 || n != 17408 || k != 5120) return -10;
  std::call_once(module_once, load_module);
  if (module_status != cudaSuccess || module == nullptr) {
    return -20 - static_cast<int>(module_status);
  }

  int64_t original_m = rows;
  int64_t padded_m = rows;
  int64_t interleaved_n = 2LL * n;
  int64_t hidden_k = k;
  int64_t groups = 1;
  int32_t max_active_clusters = configured_max_active_clusters();
  if (max_active_clusters == 0) return -11;
  int32_t result = 0;
  const void* a_arg = a;
  const void* b_arg = b;
  const void* a_sf_arg = a_sf;
  const void* b_sf_arg = b_sf;
  const void* c_arg = c;
  const void* c_sf_arg = c_sf;
  const void* alpha_arg = alpha;
  const void* input_scale_arg = input_global_scale;
  const void* down_scale_arg = down_inverse_global_scale;
  const void* tile_groups_arg = tile_groups;
  const void* tile_limits_arg = tile_limits;
  const void* token_map_arg = token_map;
  const void* tile_count_arg = tile_count;
  void* stream_arg = stream;
  void* args[] = {
      &a_arg,          &b_arg,          &a_sf_arg,       &b_sf_arg,
      &c_arg,          &c_sf_arg,       &alpha_arg,      &input_scale_arg,
      &down_scale_arg, &tile_groups_arg, &tile_limits_arg, &token_map_arg,
      &tile_count_arg, &original_m,     &padded_m,       &interleaved_n,
      &hidden_k,       &groups,         &max_active_clusters, &stream_arg,
      &result};
  _mlir_dense_swiglu__mlir_ciface_cutlass_single_b_wrapper_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_2048_128_34816_5120_1_128_16__CUstream0x0_5(
      args, 21);
  return result;
}

}  // namespace apxinf::cuda_new::cutlass_ops
