#pragma once

#include "../framework/runtime_internal.h"
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <utility>

namespace apxinf::providers {

// Provider-private resources for serialized work on one concrete CUDA stream.
// Executions (and captured graphs retaining them) own this allocation. The
// lookup table is weak: it neither extends the stream lifetime nor leaks pools.
// Projection, unpack and dequantization buffers remain execution-specific.
struct CublasStreamResources {
  static constexpr size_t workspace_bytes = 4 * 1024 * 1024;
  int device = 0;
  cublasHandle_t handle = nullptr;
  void* workspace = nullptr;
  std::mutex enqueue_mutex;

  ~CublasStreamResources() {
    int previous = -1;
    cudaGetDevice(&previous);
    cudaSetDevice(device);
    if (handle != nullptr) cublasDestroy(handle);
    if (workspace != nullptr) cudaFree(workspace);
    if (previous >= 0 && previous != device) cudaSetDevice(previous);
  }
};

inline std::shared_ptr<CublasStreamResources> acquire_cublas_stream_resources(
    int device, cudaStream_t stream) {
  using namespace apxinf::framework;
  using Key = std::pair<int, uintptr_t>;
  static std::mutex cache_mutex;
  static std::map<Key, std::weak_ptr<CublasStreamResources>> cache;
  const Key key{device, reinterpret_cast<uintptr_t>(stream)};
  std::lock_guard<std::mutex> lock(cache_mutex);
  for (auto it = cache.begin(); it != cache.end();) {
    if (it->second.expired()) it = cache.erase(it);
    else ++it;
  }
  // A per-thread stream sentinel is not a concrete stream identity and must
  // never allow different host threads to share user-managed workspace.
  const bool cacheable = stream != cudaStreamPerThread;
  if (cacheable) {
    auto found = cache.find(key);
    if (found != cache.end()) {
      if (auto existing = found->second.lock()) return existing;
    }
  }
  auto resources = std::make_shared<CublasStreamResources>();
  resources->device = device;
  check_cuda(cudaSetDevice(device));
  check_cublas(cublasCreate(&resources->handle));
  check_cuda(cudaMalloc(&resources->workspace, resources->workspace_bytes));
  check_cublas(cublasSetMathMode(resources->handle, CUBLAS_PEDANTIC_MATH));
  check_cublas(cublasSetPointerMode(resources->handle, CUBLAS_POINTER_MODE_HOST));
  check_cublas(cublasSetStream(resources->handle, stream));
  check_cublas(cublasSetWorkspace(resources->handle, resources->workspace,
                                 resources->workspace_bytes));
  if (cacheable) cache[key] = resources;
  return resources;
}

}  // namespace apxinf::providers
