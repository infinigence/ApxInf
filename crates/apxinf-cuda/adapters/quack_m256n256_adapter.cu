#include <cuda_runtime.h>
#include <cstdint>
#include "apxinf_quack_swiglu_bf16_m256n256_sm110.h"

#include <atomic>
#include <mutex>

namespace {
constexpr int kK = 2560;
constexpr int kI = 9216;
constexpr int kN = 2 * kI;
constexpr int kAotErrorBase = 0x10000;
apxinf_quack_swiglu_bf16_m256n256_sm110_Kernel_Module_t module{};
std::mutex init_mutex;
std::atomic<bool> ready{false};
std::atomic<int> initialized_device{-1};
}  // namespace

extern "C" int32_t apxinf_quack_m256n256_init() {
  int device = -1;
  cudaError_t error = cudaGetDevice(&device);
  if (error != cudaSuccess) return int32_t(error);
  if (ready.load(std::memory_order_acquire))
    return device == initialized_device.load(std::memory_order_relaxed)
               ? 0 : int32_t(cudaErrorInvalidDevice);
  std::lock_guard<std::mutex> lock(init_mutex);
  if (ready.load(std::memory_order_relaxed))
    return device == initialized_device.load(std::memory_order_relaxed)
               ? 0 : int32_t(cudaErrorInvalidDevice);
  int major = 0, minor = 0, count = 0;
  error = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  if (error != cudaSuccess) return int32_t(error);
  error = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
  if (error != cudaSuccess) return int32_t(error);
  if (major != 11 || minor != 0) return int32_t(cudaErrorInvalidDeviceFunction);
  // Generated loader carries one module handle, so reject multi-GPU contexts.
  error = cudaGetDeviceCount(&count);
  if (error != cudaSuccess) return int32_t(error);
  if (count != 1) return int32_t(cudaErrorInvalidDevice);
  // The generated convenience loader only prints errors. Preserve its ABI
  // while propagating failures through our C boundary. The module is cached
  // for process lifetime, so every captured graph keeps a valid code handle.
  cudaLibrary_t* library = &module.module;
  struct InitArgs { cudaLibrary_t** library; cudaError_t* error; } init{&library, &error};
  _mlir_apxinf_quack_swiglu_bf16_m256n256_sm110_cuda_init(
      reinterpret_cast<void**>(&init));
  if (error != cudaSuccess) return int32_t(error);
  struct LoadArgs { cudaLibrary_t** library; int32_t* device; cudaError_t* error; }
      load{&library, &device, &error};
  _mlir_apxinf_quack_swiglu_bf16_m256n256_sm110_cuda_load_to_device(
      reinterpret_cast<void**>(&load));
  if (error != cudaSuccess) {
    if (module.module) cudaLibraryUnload(module.module);
    module.module = nullptr;
    return int32_t(error);
  }
  if (!module.module) return int32_t(cudaErrorInvalidResourceHandle);
  initialized_device.store(device, std::memory_order_relaxed);
  ready.store(true, std::memory_order_release);
  return 0;
}

extern "C" int32_t apxinf_quack_m256n256_forward(
    const void* x_mk, const void* b_nk, void* y_mi,
    int32_t m, int32_t max_active_clusters, cudaStream_t stream) {
  if (!ready.load(std::memory_order_acquire)) return int32_t(cudaErrorNotReady);
  int device = -1;
  cudaError_t error = cudaGetDevice(&device);
  if (error != cudaSuccess) return int32_t(error);
  if (device != initialized_device.load(std::memory_order_relaxed))
    return int32_t(cudaErrorInvalidDevice);
  if (!x_mk || !b_nk || !y_mi || m != 3387 || max_active_clusters != 20)
    return int32_t(cudaErrorInvalidValue);
  apxinf_quack_swiglu_bf16_m256n256_sm110_Tensor_a_t a{};
  a.data = const_cast<void*>(x_mk);
  a.dynamic_shapes[0] = m;
  a.dynamic_shapes[1] = kK;
  a.dynamic_strides[0] = kK;
  apxinf_quack_swiglu_bf16_m256n256_sm110_Tensor_b_t b{};
  b.data = const_cast<void*>(b_nk);
  b.dynamic_shapes[0] = kN;
  b.dynamic_shapes[1] = kK;
  b.dynamic_strides[0] = kK;
  apxinf_quack_swiglu_bf16_m256n256_sm110_Tensor_y_t y{};
  y.data = y_mi;
  y.dynamic_shapes[0] = m;
  y.dynamic_shapes[1] = kI;
  y.dynamic_strides[0] = kI;
  int32_t status = cute_dsl_apxinf_quack_swiglu_bf16_m256n256_sm110_wrapper(
      &module, &a, &b, &y, max_active_clusters, stream);
  if (status) return kAotErrorBase | (status & 0xffff);
  return int32_t(cudaGetLastError());
}
