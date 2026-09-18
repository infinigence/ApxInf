#pragma once

// Small compatibility surface for the two c10 macros used by upstream FA2.
// Host-side configuration calls are skipped while a CUDA graph is being
// captured; the same attributes are established by the warm-up launch.

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace apxinf::fa2_compat {

inline void require_success(cudaError_t status, const char* expression,
                            const char* file, int line) {
  if (status == cudaSuccess) return;
  std::fprintf(stderr, "ApxInf FA2 CUDA failure at %s:%d (%s): %s\n", file,
               line, expression, cudaGetErrorString(status));
  std::abort();
}

template <class Operation>
inline void run_host_configuration(Operation operation, cudaStream_t stream,
                                   const char* expression, const char* file,
                                   int line) {
  cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
  require_success(cudaStreamIsCapturing(stream, &capture),
                  "cudaStreamIsCapturing", file, line);
  if (capture == cudaStreamCaptureStatusNone) {
    require_success(operation(), expression, file, line);
  }
}

}  // namespace apxinf::fa2_compat

#define C10_CUDA_CHECK(expression)                                           \
  ::apxinf::fa2_compat::run_host_configuration(                             \
      [&]() { return (expression); }, stream, #expression, __FILE__, __LINE__)

#define C10_CUDA_KERNEL_LAUNCH_CHECK()                                      \
  ::apxinf::fa2_compat::require_success(                                    \
      cudaPeekAtLastError(), "CUDA kernel launch", __FILE__, __LINE__)
