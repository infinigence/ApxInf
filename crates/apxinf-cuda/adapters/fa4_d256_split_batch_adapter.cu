#include <cuda_runtime.h>
#include <cstdint>
#include "apxinf_fa4_d256_a_splitbatch_sm110.h"
#include <atomic>
#include <mutex>

extern "C" int32_t apxinf_split_batch_duplicate_q(const void*, void*, cudaStream_t);
extern "C" int32_t apxinf_split_batch_merge(const void*, const float*, void*, cudaStream_t);

namespace {
constexpr int kAotErrorBase = 0x10000;
constexpr int APXINF_FA4_SPLIT_BATCH_UNSUPPORTED_TOPOLOGY = 0x20000;
apxinf_fa4_d256_a_splitbatch_sm110_Kernel_Module_t module{};
std::mutex init_mutex;
std::atomic<bool> ready{false};
std::atomic<int> initialized_device{-1};
// Process/device lifetime: captured graphs retain this address across replays.
int32_t* seq_used_device = nullptr;

template <typename T>
void set_dense(T& tensor, const void* data, int batch, int seq, int heads) {
  tensor.data = const_cast<void*>(data);
  tensor.dynamic_shapes[0] = batch;
  tensor.dynamic_shapes[1] = seq;
  tensor.dynamic_shapes[2] = heads;
  tensor.dynamic_shapes[3] = 256;
  tensor.dynamic_strides[0] = int64_t(seq) * heads * 256;
  tensor.dynamic_strides[1] = int64_t(heads) * 256;
  tensor.dynamic_strides[2] = 256;
}
}

extern "C" int32_t apxinf_static_fa4_split_batch_init(cudaStream_t stream) {
  cudaStreamCaptureStatus capture_status{};
  cudaError_t error = cudaStreamIsCapturing(stream, &capture_status);
  if (error != cudaSuccess) return int32_t(error);
  if (capture_status != cudaStreamCaptureStatusNone) return int32_t(cudaErrorStreamCaptureUnsupported);
  int device = -1;
  error = cudaGetDevice(&device);
  if (error != cudaSuccess) return int32_t(error);
  if (ready.load(std::memory_order_acquire))
    return device == initialized_device.load(std::memory_order_relaxed)
               ? 0 : APXINF_FA4_SPLIT_BATCH_UNSUPPORTED_TOPOLOGY;
  std::lock_guard<std::mutex> lock(init_mutex);
  if (ready.load(std::memory_order_relaxed))
    return device == initialized_device.load(std::memory_order_relaxed)
               ? 0 : APXINF_FA4_SPLIT_BATCH_UNSUPPORTED_TOPOLOGY;
  int major = 0, minor = 0, device_count = 0;
  error = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  if (error != cudaSuccess) return int32_t(error);
  error = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
  if (error != cudaSuccess) return int32_t(error);
  if (major != 11 || minor != 0) return int32_t(cudaErrorInvalidDeviceFunction);
  error = cudaGetDeviceCount(&device_count);
  if (error != cudaSuccess) return int32_t(error);
  if (device_count != 1) return APXINF_FA4_SPLIT_BATCH_UNSUPPORTED_TOPOLOGY;
  apxinf_fa4_d256_a_splitbatch_sm110_Kernel_Module_Load(&module);
  error = cudaGetLastError();
  if (error != cudaSuccess) return int32_t(error);
  if (module.module == nullptr) return int32_t(cudaErrorInvalidResourceHandle);
  const int32_t seq_used_host[10] = {1718, 1715, 1718, 1716, 1718, 1717, 1718, 1718, 1718, 1719};
  error = cudaMalloc(&seq_used_device, sizeof(seq_used_host));
  if (error != cudaSuccess) return int32_t(error);
  error = cudaMemcpy(seq_used_device, seq_used_host, sizeof(seq_used_host), cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    cudaFree(seq_used_device);
    seq_used_device = nullptr;
    return int32_t(error);
  }
  initialized_device.store(device, std::memory_order_relaxed);
  ready.store(true, std::memory_order_release);
  return 0;
}

extern "C" int32_t apxinf_static_fa4_split_batch_ready(void) {
  if (!ready.load(std::memory_order_acquire))
    return APXINF_FA4_SPLIT_BATCH_UNSUPPORTED_TOPOLOGY;
  int device = -1;
  const cudaError_t error = cudaGetDevice(&device);
  if (error != cudaSuccess) return int32_t(error);
  return device == initialized_device.load(std::memory_order_relaxed)
             ? 0 : APXINF_FA4_SPLIT_BATCH_UNSUPPORTED_TOPOLOGY;
}

extern "C" int32_t apxinf_static_fa4_split_batch_forward(
    const void* q, const void* k, const void* v,
    void* q2, void* partial, float* lse, void* out, int32_t key_tokens, cudaStream_t stream) {
  if (key_tokens < 3433 || key_tokens > 3437) return int32_t(cudaErrorInvalidValue);
  if (!ready.load(std::memory_order_acquire)) return int32_t(cudaErrorNotReady);
  int device = -1;
  cudaError_t error = cudaGetDevice(&device);
  if (error != cudaSuccess) return int32_t(error);
  if (device != initialized_device.load(std::memory_order_relaxed))
    return APXINF_FA4_SPLIT_BATCH_UNSUPPORTED_TOPOLOGY;
  if (!q || !k || !v || !seq_used_device || !q2 || !partial || !lse || !out)
    return int32_t(cudaErrorInvalidValue);

  int32_t status = apxinf_split_batch_duplicate_q(q, q2, stream);
  if (status != 0) return status;
  apxinf_fa4_d256_a_splitbatch_sm110_Tensor_q_t tq{};
  set_dense(tq, q2, 2, 50, 16);
  apxinf_fa4_d256_a_splitbatch_sm110_Tensor_k_t tk{};
  set_dense(tk, k, 2, 1720, 4);
  // Read-only batch views overlap physically; seqused masks the first batch
  // at 1718. This preserves the accepted reduction partition for real keys.
  tk.dynamic_strides[0] = int64_t(1718) * 4 * 256;
  apxinf_fa4_d256_a_splitbatch_sm110_Tensor_v_t tv{};
  set_dense(tv, v, 2, 1720, 4);
  tv.dynamic_strides[0] = int64_t(1718) * 4 * 256;
  apxinf_fa4_d256_a_splitbatch_sm110_Tensor_out_tensor_t to{};
  set_dense(to, partial, 2, 50, 16);
  apxinf_fa4_d256_a_splitbatch_sm110_Tensor_lse_t tl{};
  tl.data = lse;
  tl.dynamic_shapes[0] = 2;
  tl.dynamic_shapes[1] = 16;
  tl.dynamic_shapes[2] = 50;
  tl.dynamic_strides[0] = 800;
  tl.dynamic_strides[1] = 50;
  apxinf_fa4_d256_a_splitbatch_sm110_Tensor_seqused_k_t ts{};
  ts.data = seq_used_device + 2 * (key_tokens - 3433);
  ts.dynamic_shapes[0] = 2;
  status = cute_dsl_apxinf_fa4_d256_a_splitbatch_sm110_wrapper(
      &module, &tq, &tk, &tv, &to, &tl, &ts, 0.0625f, stream);
  if (status != 0) return kAotErrorBase | (status & 0xffff);
  status = apxinf_split_batch_merge(partial, lse, out, stream);
  if (status != 0) return status;
  return int32_t(cudaGetLastError());
}
