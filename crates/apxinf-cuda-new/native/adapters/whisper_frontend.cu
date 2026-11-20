// Log-mel frontend launch adapters; the kernels live in
// kernels/custom/whisper_frontend.cuh.

#include "../include/apxinf_cuda/whisper_frontend.h"

#include "../framework/runtime_internal.h"
#include "../kernels/custom/whisper_frontend.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include <cmath>

namespace {

namespace kernels = apxinf::whisper_frontend::kernels;

int blocks_for(int64_t count) {
  const int64_t blocks = (count + kernels::kThreads - 1) / kernels::kThreads;
  if (blocks > 4096) return 4096;
  return static_cast<int>(blocks == 0 ? 1 : blocks);
}

}  // namespace

extern "C" apxinf_status_t apxinf_whisper_frames_launch(
    apxinf_runtime_t runtime, int64_t n_samples, int64_t n_fft, int64_t hop,
    int64_t pad, int64_t frames,
    const apxinf_whisper_frames_bindings_t* bindings) {
  return apxinf::framework::abi_boundary([&] {
    if (runtime == nullptr || bindings == nullptr) {
      throw apxinf::framework::Failure(APXINF_STATUS_INVALID_ARGUMENT,
                                       "null log-mel frames argument");
    }
    if (n_samples <= 0 || n_fft <= 0 || hop <= 0 || frames <= 0 || pad < 0) {
      throw apxinf::framework::Failure(APXINF_STATUS_INVALID_ARGUMENT,
                                       "invalid log-mel frames geometry");
    }
    if (pad >= n_samples || pad >= n_fft) {
      throw apxinf::framework::Failure(APXINF_STATUS_INVALID_ARGUMENT,
                                       "log-mel frames pad exceeds reflect range");
    }
    apxinf::framework::check_cuda(cudaSetDevice(runtime->device));
    const auto stream = static_cast<cudaStream_t>(bindings->stream);
    kernels::frames_kernel<<<static_cast<int>(frames), kernels::kThreads, 0,
                             stream>>>(
        static_cast<const float*>(bindings->waveform),
        static_cast<const float*>(bindings->hann),
        static_cast<float*>(bindings->frames), n_samples, n_fft, hop, pad,
        frames);
    apxinf::framework::check_cuda(cudaGetLastError());
  });
}

extern "C" apxinf_status_t apxinf_whisper_power_launch(
    apxinf_runtime_t runtime, int64_t count,
    const apxinf_whisper_power_bindings_t* bindings) {
  return apxinf::framework::abi_boundary([&] {
    if (runtime == nullptr || bindings == nullptr) {
      throw apxinf::framework::Failure(APXINF_STATUS_INVALID_ARGUMENT,
                                       "null log-mel power argument");
    }
    if (count < 0) {
      throw apxinf::framework::Failure(APXINF_STATUS_INVALID_ARGUMENT,
                                       "negative log-mel power count");
    }
    if (count == 0) return;
    apxinf::framework::check_cuda(cudaSetDevice(runtime->device));
    const auto stream = static_cast<cudaStream_t>(bindings->stream);
    const int blocks = blocks_for(count);
    kernels::power_kernel<<<blocks, kernels::kThreads, 0, stream>>>(
        static_cast<const float*>(bindings->re),
        static_cast<const float*>(bindings->im),
        static_cast<float*>(bindings->power), count);
    apxinf::framework::check_cuda(cudaGetLastError());
  });
}

extern "C" apxinf_status_t apxinf_whisper_log_mel_launch(
    apxinf_runtime_t runtime, int64_t rows, int64_t cols,
    const apxinf_whisper_log_mel_bindings_t* bindings) {
  return apxinf::framework::abi_boundary([&] {
    if (runtime == nullptr || bindings == nullptr) {
      throw apxinf::framework::Failure(APXINF_STATUS_INVALID_ARGUMENT,
                                       "null log-mel argument");
    }
    if (rows <= 0 || cols <= 0) {
      throw apxinf::framework::Failure(APXINF_STATUS_INVALID_ARGUMENT,
                                       "empty log-mel geometry");
    }
    const int64_t count = rows * cols;
    apxinf::framework::check_cuda(cudaSetDevice(runtime->device));
    const auto stream = static_cast<cudaStream_t>(bindings->stream);
    const int blocks = blocks_for(count);
    kernels::block_max_kernel<<<blocks, kernels::kThreads, 0, stream>>>(
        static_cast<const float*>(bindings->spec),
        static_cast<float*>(bindings->max_scratch), count);
    apxinf::framework::check_cuda(cudaGetLastError());
    float* global_max_out = static_cast<float*>(bindings->max_scratch) + blocks;
    kernels::log_mel_kernel<<<blocks, kernels::kThreads, 0, stream>>>(
        static_cast<const float*>(bindings->spec), global_max_out,
        static_cast<const float*>(bindings->max_scratch), blocks,
        static_cast<__half*>(bindings->out_f16), count);
    apxinf::framework::check_cuda(cudaGetLastError());
  });
}
