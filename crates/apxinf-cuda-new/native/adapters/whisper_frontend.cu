// Log-mel frontend nonlinear stages.
//
// The DFT projections and the mel filterbank run through the shared GEMM
// operator; these launches cover the reflect-padded framing pass, power
// (re^2 + im^2) and the log-mel transform (clamp 1e-10, log10, floor global_max
// - 8, (x+4)/4, cast f16) with the global max reduced on device.

#include "../include/apxinf_cuda/whisper_frontend.h"

#include "../framework/runtime_internal.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include <cmath>

namespace {

constexpr int kThreads = 256;

__global__ void power_kernel(const float* re, const float* im, float* out,
                             int64_t count) {
  int64_t index = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (index >= count) return;
  const float r = re[index];
  const float i = im[index];
  out[index] = r * r + i * i;
}

__global__ void block_max_kernel(const float* spec, float* block_max,
                                 int64_t count) {
  // The reference floors log10(spec) at max(log10(spec)) - 8, so the
  // reduction runs in log space: reduce log10(clamp(spec, 1e-10)).
  __shared__ float shared[kThreads];
  const int64_t tid = threadIdx.x;
  int64_t index = blockIdx.x * static_cast<int64_t>(blockDim.x) + tid;
  float value = -3.402823466e38f;
  while (index < count) {
    value = fmaxf(value, log10f(fmaxf(spec[index], 1e-10f)));
    index += gridDim.x * static_cast<int64_t>(blockDim.x);
  }
  shared[tid] = value;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
    }
    __syncthreads();
  }
  if (tid == 0) {
    block_max[blockIdx.x] = shared[0];
  }
}

__global__ void log_mel_kernel(const float* spec, float* global_max_out,
                               const float* block_max, int blocks,
                               __half* out_f16, int64_t count) {
  // Every block folds the small per-block maxima array redundantly, so the
  // transform needs no inter-block ordering beyond block_max_kernel's
  // completion (guaranteed by stream ordering).
  float global_max = -3.402823466e38f;
  for (int b = 0; b < blocks; ++b) {
    global_max = fmaxf(global_max, block_max[b]);
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *global_max_out = global_max;
  }
  const float floor_value = global_max - 8.0f;
  const int64_t tid = threadIdx.x;
  int64_t index = blockIdx.x * static_cast<int64_t>(blockDim.x) + tid;
  while (index < count) {
    float value = spec[index];
    value = fmaxf(value, 1e-10f);
    value = log10f(value);
    value = fmaxf(value, floor_value);
    value = (value + 4.0f) / 4.0f;
    out_f16[index] = __float2half(value);
    index += gridDim.x * static_cast<int64_t>(blockDim.x);
  }
}

__device__ int64_t reflect_index(int64_t index, int64_t length) {
  // torch F.pad(..., mode="reflect") index mapping for a pad of < length.
  if (index < 0) index = -index;
  if (index >= length) {
    const int64_t over = index - (length - 1);
    index = length - 1 - over;
    if (index < 0) index = -index;  // deep reflection cannot occur with pad < length
  }
  return index;
}

__global__ void frames_kernel(const float* waveform, const float* hann,
                              float* frames, int64_t n_samples, int64_t n_fft,
                              int64_t hop, int64_t pad, int64_t frames_count) {
  const int64_t f = blockIdx.x;
  if (f >= frames_count) return;
  const int64_t base = f * hop - pad;
  float* row = frames + f * n_fft;
  for (int64_t j = threadIdx.x; j < n_fft; j += blockDim.x) {
    int64_t index = base + j;
    index = reflect_index(index, n_samples);
    row[j] = waveform[index] * hann[j];
  }
}

int blocks_for(int64_t count) {
  const int64_t blocks = (count + kThreads - 1) / kThreads;
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
    frames_kernel<<<static_cast<int>(frames), kThreads, 0, stream>>>(
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
    power_kernel<<<blocks, kThreads, 0, stream>>>(
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
    block_max_kernel<<<blocks, kThreads, 0, stream>>>(
        static_cast<const float*>(bindings->spec),
        static_cast<float*>(bindings->max_scratch), count);
    apxinf::framework::check_cuda(cudaGetLastError());
    float* global_max_out = static_cast<float*>(bindings->max_scratch) + blocks;
    log_mel_kernel<<<blocks, kThreads, 0, stream>>>(
        static_cast<const float*>(bindings->spec), global_max_out,
        static_cast<const float*>(bindings->max_scratch), blocks,
        static_cast<__half*>(bindings->out_f16), count);
    apxinf::framework::check_cuda(cudaGetLastError());
  });
}
