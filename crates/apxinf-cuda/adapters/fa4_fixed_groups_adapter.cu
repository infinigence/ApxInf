#include <cuda_runtime.h>
#include <cstdint>
#include "apxinf_fa4_vfixed_b3_l624_sm110.h"
#include "apxinf_fa4_vfixed_b3_l2200_sm110.h"

#include <atomic>
#include <mutex>

namespace {
constexpr int kHeads = 16;
constexpr int kDim = 64;
constexpr int kGroupStarts[4] = {0, 624, 1248, 1872};
constexpr int kGroupLengths[4] = {624, 624, 624, 2200};
constexpr int kAotErrorBase = 0x10000;
constexpr int kUnsupported = 0x20000;

apxinf_fa4_vfixed_b3_l624_sm110_Kernel_Module_t module624{};
apxinf_fa4_vfixed_b3_l2200_sm110_Kernel_Module_t module2200{};
std::mutex init_mutex;
std::atomic<bool> ready{false};
int initialized_device = -1;  // Published before ready.store(release).

template <class T>
void set_tensor(T& t, void* ptr, int length, int token_stride) {
  t.data = ptr;
  t.dynamic_shapes[0] = 3;
  t.dynamic_shapes[1] = length;
  t.dynamic_shapes[2] = kHeads;
  t.dynamic_shapes[3] = kDim;
  t.dynamic_strides[0] = int64_t(4072) * token_stride;
  t.dynamic_strides[1] = token_stride;
  t.dynamic_strides[2] = kDim;
}

bool aligned(const void* ptr) {
  return ptr != nullptr && (reinterpret_cast<uintptr_t>(ptr) & 15) == 0;
}

int32_t launch_group(int group, const void* q, const void* k,
                     const void* v, void* out, cudaStream_t stream) {
  const int start = kGroupStarts[group], length = kGroupLengths[group];
  const int64_t q_offset = int64_t(start) * kHeads * kDim;
  const int64_t v_offset = int64_t(start) * 3 * kHeads * kDim;
  void* qp = static_cast<void*>(static_cast<char*>(const_cast<void*>(q)) + q_offset * 2);
  void* kp = static_cast<void*>(static_cast<char*>(const_cast<void*>(k)) + q_offset * 2);
  void* vp = static_cast<void*>(static_cast<char*>(const_cast<void*>(v)) + v_offset * 2);
  void* op = static_cast<void*>(static_cast<char*>(out) + q_offset * 2);
  int32_t status = 0;
  if (length == 624) {
    apxinf_fa4_vfixed_b3_l624_sm110_Tensor_q_t tq{}; set_tensor(tq, qp, length, 1024);
    apxinf_fa4_vfixed_b3_l624_sm110_Tensor_k_t tk{}; set_tensor(tk, kp, length, 1024);
    apxinf_fa4_vfixed_b3_l624_sm110_Tensor_v_t tv{}; set_tensor(tv, vp, length, 3072);
    apxinf_fa4_vfixed_b3_l624_sm110_Tensor_out_t to{}; set_tensor(to, op, length, 1024);
    status = cute_dsl_apxinf_fa4_vfixed_b3_l624_sm110_wrapper(
        &module624, &tq, &tk, &tv, &to, 0.125f, stream);
  } else {
    apxinf_fa4_vfixed_b3_l2200_sm110_Tensor_q_t tq{}; set_tensor(tq, qp, length, 1024);
    apxinf_fa4_vfixed_b3_l2200_sm110_Tensor_k_t tk{}; set_tensor(tk, kp, length, 1024);
    apxinf_fa4_vfixed_b3_l2200_sm110_Tensor_v_t tv{}; set_tensor(tv, vp, length, 3072);
    apxinf_fa4_vfixed_b3_l2200_sm110_Tensor_out_t to{}; set_tensor(to, op, length, 1024);
    status = cute_dsl_apxinf_fa4_vfixed_b3_l2200_sm110_wrapper(
        &module2200, &tq, &tk, &tv, &to, 0.125f, stream);
  }
  if (status != 0) return kAotErrorBase | (status & 0xffff);
  return int32_t(cudaGetLastError());
}
}  // namespace

extern "C" int32_t apxinf_static_fa4_bf16_vfixed_init(cudaStream_t stream) {
  cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
  cudaError_t error = cudaStreamIsCapturing(stream, &capture);
  if (error != cudaSuccess) return int32_t(error);
  if (capture != cudaStreamCaptureStatusNone) return int32_t(cudaErrorStreamCaptureUnsupported);
  int device = -1;
  error = cudaGetDevice(&device);
  if (error != cudaSuccess) return int32_t(error);
  if (ready.load(std::memory_order_acquire))
    return device == initialized_device ? 0 : kUnsupported;
  std::lock_guard<std::mutex> lock(init_mutex);
  if (ready.load(std::memory_order_relaxed))
    return device == initialized_device ? 0 : kUnsupported;
  int count = 0, major = 0, minor = 0, sms = 0;
  error = cudaGetDeviceCount(&count);
  if (error != cudaSuccess) return int32_t(error);
  if (count != 1) return kUnsupported;
  error = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  if (error != cudaSuccess) return int32_t(error);
  error = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
  if (error != cudaSuccess) return int32_t(error);
  error = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  if (error != cudaSuccess) return int32_t(error);
  if (major != 11 || minor != 0 || sms != 20) return kUnsupported;
  apxinf_fa4_vfixed_b3_l624_sm110_Kernel_Module_Load(&module624);
  error = cudaGetLastError();
  if (error != cudaSuccess || module624.module == nullptr)
    return int32_t(error == cudaSuccess ? cudaErrorInvalidResourceHandle : error);
  apxinf_fa4_vfixed_b3_l2200_sm110_Kernel_Module_Load(&module2200);
  error = cudaGetLastError();
  if (error != cudaSuccess || module2200.module == nullptr) {
    cudaLibraryUnload(module624.module);
    module624.module = nullptr;
    return int32_t(error == cudaSuccess ? cudaErrorInvalidResourceHandle : error);
  }
  initialized_device = device;
  ready.store(true, std::memory_order_release);
  return 0;
}

extern "C" int32_t apxinf_static_fa4_bf16_vfixed_ready(void) {
  if (!ready.load(std::memory_order_acquire)) return kUnsupported;
  int device = -1;
  cudaError_t error = cudaGetDevice(&device);
  if (error != cudaSuccess) return int32_t(error);
  return device == initialized_device ? 0 : kUnsupported;
}

extern "C" int32_t apxinf_static_fa4_bf16_vfixed_forward(
    const void* q, const void* k, const void* v, void* out,
    cudaStream_t stream) {
  if (!ready.load(std::memory_order_acquire)) return int32_t(cudaErrorNotReady);
  if (!aligned(q) || !aligned(k) || !aligned(v) || !aligned(out))
    return int32_t(cudaErrorInvalidValue);
  int device = -1;
  cudaError_t error = cudaGetDevice(&device);
  if (error != cudaSuccess) return int32_t(error);
  if (device != initialized_device) return kUnsupported;
  // All four descriptors and launch arguments are stack-local. The two loaded
  // modules are read-only and remain live for captured Graph replay.
  for (int group = 0; group < 4; ++group) {
    int32_t status = launch_group(group, q, k, v, out, stream);
    if (status != 0) return status;
  }
  return 0;
}
