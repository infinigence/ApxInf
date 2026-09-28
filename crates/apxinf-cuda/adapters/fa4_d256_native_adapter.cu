#include "fixed_profile.h"
#include <cuda_runtime.h>
#include <cstdint>
#include "apxinf_fa4_d256_l_sm110.h"

#include <atomic>
#include <mutex>

namespace {
constexpr int kAotErrorBase = 0x10000;
apxinf_fa4_d256_l_sm110_Kernel_Module_t module_l{};
std::mutex init_mutex;
std::atomic<bool> ready{false};
std::atomic<int> initialized_device{-1};

template <typename T>
void set_dense(T& tensor, const void* data, int seq, int heads) {
  tensor.data = const_cast<void*>(data);
  tensor.dynamic_shapes[0] = 1;
  tensor.dynamic_shapes[1] = seq;
  tensor.dynamic_shapes[2] = heads;
  tensor.dynamic_shapes[3] = 256;
  tensor.dynamic_strides[0] = int64_t(seq) * heads * 256;
  tensor.dynamic_strides[1] = int64_t(heads) * 256;
  tensor.dynamic_strides[2] = 256;
}
}  // namespace

extern "C" int32_t apxinf_static_fa4_d256_init(void) {
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
  int major = 0, minor = 0, device_count = 0;
  error = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  if (error != cudaSuccess) return int32_t(error);
  error = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
  if (error != cudaSuccess) return int32_t(error);
  if (major != 11 || minor != 0) return int32_t(cudaErrorInvalidDeviceFunction);
  // The generated loader iterates devices but retains one module handle.
  error = cudaGetDeviceCount(&device_count);
  if (error != cudaSuccess) return int32_t(error);
  if (device_count != 1) return int32_t(cudaErrorInvalidDevice);
  apxinf_fa4_d256_l_sm110_Kernel_Module_Load(&module_l);
  error = cudaGetLastError();
  if (error != cudaSuccess) return int32_t(error);
  if (module_l.module == nullptr) return int32_t(cudaErrorInvalidResourceHandle);
  initialized_device.store(device, std::memory_order_relaxed);
  ready.store(true, std::memory_order_release);
  return 0;
}

extern "C" int32_t apxinf_static_fa4_d256_forward(
    const void* q, const void* k, const void* v,
    void* out, cudaStream_t stream) {
  if (!ready.load(std::memory_order_acquire)) return int32_t(cudaErrorNotReady);
  int device = -1;
  cudaError_t error = cudaGetDevice(&device);
  if (error != cudaSuccess) return int32_t(error);
  if (device != initialized_device.load(std::memory_order_relaxed))
    return int32_t(cudaErrorInvalidDevice);
  if (!q || !k || !v || !out) return int32_t(cudaErrorInvalidValue);
  {
    apxinf_fa4_d256_l_sm110_Tensor_q_t tq{}; set_dense(tq, q, APXINF_FIXED_SCENE_TOKENS, 16);
    apxinf_fa4_d256_l_sm110_Tensor_k_t tk{}; set_dense(tk, k, APXINF_FIXED_SCENE_TOKENS, 4);
    apxinf_fa4_d256_l_sm110_Tensor_v_t tv{}; set_dense(tv, v, APXINF_FIXED_SCENE_TOKENS, 4);
    apxinf_fa4_d256_l_sm110_Tensor_out_t to{}; set_dense(to, out, APXINF_FIXED_SCENE_TOKENS, 16);
    int32_t status = cute_dsl_apxinf_fa4_d256_l_sm110_wrapper(
        &module_l, &tq, &tk, &tv, &to, 0.0625f, stream);
    if (status != 0) return kAotErrorBase | (status & 0xffff);
  }
  return int32_t(cudaGetLastError());
}
