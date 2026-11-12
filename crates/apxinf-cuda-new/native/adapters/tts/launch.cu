#include "../../include/apxinf_cuda/tts.h"
#include "../../framework/runtime_internal.h"
#include "../../kernels/custom/tts.cuh"

#include <climits>
#include <cmath>
#include <cstdint>
#include <string>

namespace {

using apxinf::framework::Failure;
namespace kernels = apxinf::tts::kernels;

constexpr int kDirectThreads = 256;
constexpr int kMaxBlocks = 4096;

bool supported_dtype(uint32_t dtype) {
  return dtype == APXINF_DTYPE_F32 || dtype == APXINF_DTYPE_F16 ||
         dtype == APXINF_DTYPE_BF16;
}

int element_bytes(uint32_t dtype) {
  return dtype == APXINF_DTYPE_F32 ? 4 : 2;
}

int blocks_for(int64_t count) {
  const int64_t blocks = (count + kDirectThreads - 1) / kDirectThreads;
  return static_cast<int>(blocks > kMaxBlocks ? kMaxBlocks : blocks);
}

void validate_dtype(uint32_t dtype, const char* what) {
  if (!supported_dtype(dtype)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  std::string("unsupported tts dtype for ") + what);
  }
}

void validate_positive(int64_t value, const char* what) {
  if (value <= 0) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  std::string("invalid ") + what);
  }
}

void validate_pointers(const void* input, const void* output, const char* what) {
  if (input == nullptr || output == nullptr) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  std::string("null tts binding for ") + what);
  }
}

template <class T>
cudaError_t launch_bias_activation(const void* input, const void* bias,
                                   void* output, int64_t count, int64_t cols,
                                   uint32_t activation, uint32_t round_bias,
                                   cudaStream_t stream) {
  if (activation > APXINF_TTS_ACTIVATION_LEAKY_RELU_ONE_PERCENT ||
      round_bias > 1) {
    return cudaErrorInvalidValue;
  }
  kernels::bias_activation<T><<<blocks_for(count), kDirectThreads, 0, stream>>>(
      static_cast<const T*>(input), static_cast<const T*>(bias),
      static_cast<T*>(output), count, static_cast<int>(cols),
      static_cast<int>(activation), static_cast<int>(round_bias));
  return cudaGetLastError();
}

template <class T>
cudaError_t launch_residual_add_bias(const void* input, const void* bias,
                                     const void* residual, void* output,
                                     int64_t count, int64_t cols,
                                     cudaStream_t stream) {
  kernels::residual_add_bias<T>
      <<<blocks_for(count), kDirectThreads, 0, stream>>>(
          static_cast<const T*>(input), static_cast<const T*>(bias),
          static_cast<const T*>(residual), static_cast<T*>(output), count,
          static_cast<int>(cols));
  return cudaGetLastError();
}

template <class T>
cudaError_t launch_affine(const void* input, const void* style, void* output,
                          int64_t count, int64_t cols, cudaStream_t stream) {
  kernels::affine<T><<<blocks_for(count), kDirectThreads, 0, stream>>>(
      static_cast<const T*>(input), static_cast<const T*>(style),
      static_cast<T*>(output), count, static_cast<int>(cols));
  return cudaGetLastError();
}

template <class T>
cudaError_t launch_glu_separate(const void* input, const void* gate,
                                void* output, int64_t count,
                                cudaStream_t stream) {
  kernels::glu_separate<T><<<blocks_for(count), kDirectThreads, 0, stream>>>(
      static_cast<const T*>(input), static_cast<const T*>(gate),
      static_cast<T*>(output), count);
  return cudaGetLastError();
}

template <class T>
cudaError_t launch_tanh_sigmoid_gate(const void* input, const void* secondary,
                                     void* output, int64_t count, int64_t cols,
                                     cudaStream_t stream) {
  kernels::tanh_sigmoid_gate<T>
      <<<blocks_for(count), kDirectThreads, 0, stream>>>(
          static_cast<const T*>(input), static_cast<const T*>(secondary),
          static_cast<T*>(output), count, static_cast<int>(cols));
  return cudaGetLastError();
}

template <class T>
cudaError_t launch_grn(const void* input, const void* style, void* output,
                       int64_t rows, int64_t cols, cudaStream_t stream) {
  // One FP32 normalizer per column lives in dynamic shared memory. The kernel
  // never opts into the larger per-block limit, so the launch is bounded by
  // the 48 KiB default a kernel may request.
  constexpr int64_t kMaxColumns = 48 * 1024 / sizeof(float);
  if (cols > kMaxColumns) return cudaErrorInvalidValue;
  kernels::grn<T><<<1, kDirectThreads, static_cast<size_t>(cols) * sizeof(float),
                    stream>>>(static_cast<const T*>(input),
                              static_cast<const T*>(style),
                              static_cast<T*>(output), rows,
                              static_cast<int>(cols));
  return cudaGetLastError();
}

template <class T>
cudaError_t launch_column_norm(const void* input, const void* style,
                               void* output, int64_t rows, int64_t cols,
                               float eps, cudaStream_t stream) {
  const unsigned threads = rows < 512 ? 32 : 512;
  kernels::column_norm_input_statistics<T>
      <<<static_cast<unsigned>(cols), threads, 0, stream>>>(
          static_cast<const T*>(input), static_cast<const T*>(style),
          static_cast<T*>(output), rows, static_cast<int>(cols), eps);
  return cudaGetLastError();
}

template <class T>
cudaError_t launch_rope(const void* input, const void* cos_values,
                        const void* sin_values, void* output, int64_t tokens,
                        int64_t time, int64_t heads, int64_t head_dim,
                        cudaStream_t stream) {
  const int64_t pairs = tokens * heads * (head_dim / 2);
  const int64_t blocks = (pairs + kDirectThreads - 1) / kDirectThreads;
  if (blocks > INT_MAX) return cudaErrorInvalidValue;
  kernels::rope_apply_kernel<T>
      <<<static_cast<int>(blocks), kDirectThreads, 0, stream>>>(
          static_cast<const T*>(input), static_cast<const T*>(cos_values),
          static_cast<const T*>(sin_values), static_cast<T*>(output), pairs,
          time, heads, head_dim);
  return cudaGetLastError();
}

template <class T>
cudaError_t launch_mean_rows(const void* input, void* output, int64_t rows,
                             int64_t cols, cudaStream_t stream) {
  kernels::mean_rows_kernel<T><<<1, kDirectThreads, 0, stream>>>(
      static_cast<const T*>(input), static_cast<T*>(output),
      static_cast<int>(rows), static_cast<int>(cols));
  return cudaGetLastError();
}

template <class T>
cudaError_t launch_argmax_rows(const void* logits, void* index, int64_t rows,
                               int64_t cols, cudaStream_t stream) {
  kernels::argmax_rows_kernel<T>
      <<<static_cast<unsigned>(rows), kDirectThreads, 0, stream>>>(
          static_cast<const T*>(logits), static_cast<int32_t*>(index),
          static_cast<int>(rows), static_cast<int>(cols));
  return cudaGetLastError();
}

template <class T>
cudaError_t launch_sinusoidal(const void* times, void* output, int64_t rows,
                              int64_t width, float scale, float log_step,
                              cudaStream_t stream) {
  const int64_t blocks = (rows * width + kDirectThreads - 1) / kDirectThreads;
  if (blocks > INT_MAX) return cudaErrorInvalidValue;
  kernels::sinusoidal_embedding_kernel<T>
      <<<static_cast<int>(blocks), kDirectThreads, 0, stream>>>(
          static_cast<const T*>(times), static_cast<T*>(output), rows, width,
          scale, log_step);
  return cudaGetLastError();
}

#define APXINF_TTS_DISPATCH(dtype, fn, ...)                      \
  switch (dtype) {                                               \
    case APXINF_DTYPE_F32:                                       \
      apxinf::framework::check_cuda(fn<float>(__VA_ARGS__));     \
      break;                                                     \
    case APXINF_DTYPE_F16:                                       \
      apxinf::framework::check_cuda(fn<__half>(__VA_ARGS__));    \
      break;                                                     \
    case APXINF_DTYPE_BF16:                                      \
      apxinf::framework::check_cuda(                             \
          fn<__nv_bfloat16>(__VA_ARGS__));                       \
      break;                                                     \
    default:                                                     \
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid dtype"); \
  }

}  // namespace

extern "C" apxinf_status_t apxinf_tts_bias_activation(
    apxinf_runtime_t runtime, const void* input, const void* bias, void* output,
    int64_t count, int64_t cols, uint32_t dtype, uint32_t activation,
    uint32_t round_bias, apxinf_cuda_stream_t raw_stream) {
  return apxinf::framework::abi_boundary([&] {
    if (!runtime) throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "null runtime");
    validate_pointers(input, output, "bias activation");
    validate_positive(count, "bias activation count");
    validate_positive(cols, "bias activation cols");
    validate_dtype(dtype, "bias activation");
    apxinf::framework::check_cuda(cudaSetDevice(runtime->device));
    const cudaStream_t stream = static_cast<cudaStream_t>(raw_stream);
    APXINF_TTS_DISPATCH(dtype, launch_bias_activation, input, bias, output,
                        count, cols, activation, round_bias, stream);
  });
}

extern "C" apxinf_status_t apxinf_tts_residual_add_bias(
    apxinf_runtime_t runtime, const void* input, const void* bias,
    const void* residual, void* output, int64_t count, int64_t cols,
    uint32_t dtype, apxinf_cuda_stream_t raw_stream) {
  return apxinf::framework::abi_boundary([&] {
    if (!runtime) throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "null runtime");
    validate_pointers(input, output, "residual add bias");
    if (bias == nullptr || residual == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "null tts binding for residual add bias");
    }
    validate_positive(count, "residual add bias count");
    validate_positive(cols, "residual add bias cols");
    validate_dtype(dtype, "residual add bias");
    apxinf::framework::check_cuda(cudaSetDevice(runtime->device));
    const cudaStream_t stream = static_cast<cudaStream_t>(raw_stream);
    APXINF_TTS_DISPATCH(dtype, launch_residual_add_bias, input, bias, residual,
                        output, count, cols, stream);
  });
}

extern "C" apxinf_status_t apxinf_tts_affine(
    apxinf_runtime_t runtime, const void* input, const void* style, void* output,
    int64_t count, int64_t cols, uint32_t dtype, apxinf_cuda_stream_t raw_stream) {
  return apxinf::framework::abi_boundary([&] {
    if (!runtime) throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "null runtime");
    validate_pointers(input, output, "affine");
    if (style == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "null affine style");
    }
    validate_positive(count, "affine count");
    validate_positive(cols, "affine cols");
    validate_dtype(dtype, "affine");
    apxinf::framework::check_cuda(cudaSetDevice(runtime->device));
    const cudaStream_t stream = static_cast<cudaStream_t>(raw_stream);
    APXINF_TTS_DISPATCH(dtype, launch_affine, input, style, output, count,
                        cols, stream);
  });
}

extern "C" apxinf_status_t apxinf_tts_glu_separate(
    apxinf_runtime_t runtime, const void* input, const void* gate, void* output,
    int64_t count, uint32_t dtype, apxinf_cuda_stream_t raw_stream) {
  return apxinf::framework::abi_boundary([&] {
    if (!runtime) throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "null runtime");
    if (input == nullptr || gate == nullptr || output == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "null glu separate binding");
    }
    validate_positive(count, "glu separate count");
    validate_dtype(dtype, "glu separate");
    apxinf::framework::check_cuda(cudaSetDevice(runtime->device));
    const cudaStream_t stream = static_cast<cudaStream_t>(raw_stream);
    APXINF_TTS_DISPATCH(dtype, launch_glu_separate, input, gate, output,
                        count, stream);
  });
}

extern "C" apxinf_status_t apxinf_tts_tanh_sigmoid_gate(
    apxinf_runtime_t runtime, const void* input, const void* secondary,
    void* output, int64_t count, int64_t cols, uint32_t dtype,
    uint32_t has_secondary, apxinf_cuda_stream_t raw_stream) {
  return apxinf::framework::abi_boundary([&] {
    if (!runtime) throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "null runtime");
    validate_pointers(input, output, "tanh-sigmoid gate");
    if ((has_secondary != 0) != (secondary != nullptr)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "tanh-sigmoid secondary binding mismatch");
    }
    validate_positive(count, "tanh-sigmoid count");
    validate_positive(cols, "tanh-sigmoid cols");
    if (count % cols != 0) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "tanh-sigmoid count is not a row multiple");
    }
    validate_dtype(dtype, "tanh-sigmoid gate");
    apxinf::framework::check_cuda(cudaSetDevice(runtime->device));
    const cudaStream_t stream = static_cast<cudaStream_t>(raw_stream);
    APXINF_TTS_DISPATCH(dtype, launch_tanh_sigmoid_gate, input, secondary,
                        output, count, cols, stream);
  });
}

extern "C" apxinf_status_t apxinf_tts_grn(
    apxinf_runtime_t runtime, const void* input, const void* style, void* output,
    int64_t rows, int64_t cols, uint32_t dtype, apxinf_cuda_stream_t raw_stream) {
  return apxinf::framework::abi_boundary([&] {
    if (!runtime) throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "null runtime");
    validate_pointers(input, output, "grn");
    if (style == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "null grn style");
    }
    validate_positive(rows, "grn rows");
    validate_positive(cols, "grn cols");
    validate_dtype(dtype, "grn");
    apxinf::framework::check_cuda(cudaSetDevice(runtime->device));
    const cudaStream_t stream = static_cast<cudaStream_t>(raw_stream);
    APXINF_TTS_DISPATCH(dtype, launch_grn, input, style, output, rows,
                        cols, stream);
  });
}

extern "C" apxinf_status_t apxinf_tts_column_norm(
    apxinf_runtime_t runtime, const void* input, const void* style, void* output,
    int64_t rows, int64_t cols, uint32_t dtype, float eps,
    apxinf_cuda_stream_t raw_stream) {
  return apxinf::framework::abi_boundary([&] {
    if (!runtime) throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "null runtime");
    validate_pointers(input, output, "column norm");
    if (style == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "null column norm style");
    }
    validate_positive(rows, "column norm rows");
    validate_positive(cols, "column norm cols");
    if (cols > INT_MAX || !std::isfinite(eps)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid column norm geometry");
    }
    validate_dtype(dtype, "column norm");
    apxinf::framework::check_cuda(cudaSetDevice(runtime->device));
    const cudaStream_t stream = static_cast<cudaStream_t>(raw_stream);
    APXINF_TTS_DISPATCH(dtype, launch_column_norm, input, style, output,
                        rows, cols, eps, stream);
  });
}

extern "C" apxinf_status_t apxinf_tts_rope_apply(
    apxinf_runtime_t runtime, const void* input, const void* cos_values,
    const void* sin_values, void* output, int64_t tokens, int64_t time,
    int64_t heads, int64_t head_dim, uint32_t dtype,
    apxinf_cuda_stream_t raw_stream) {
  return apxinf::framework::abi_boundary([&] {
    if (!runtime) throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "null runtime");
    if (input == nullptr || cos_values == nullptr || sin_values == nullptr ||
        output == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "null rope binding");
    }
    validate_positive(tokens, "rope tokens");
    validate_positive(time, "rope time");
    validate_positive(heads, "rope heads");
    validate_positive(head_dim, "rope head dim");
    if (head_dim % 2 != 0 || tokens % time != 0) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid rope geometry");
    }
    validate_dtype(dtype, "rope");
    apxinf::framework::check_cuda(cudaSetDevice(runtime->device));
    const cudaStream_t stream = static_cast<cudaStream_t>(raw_stream);
    APXINF_TTS_DISPATCH(dtype, launch_rope, input, cos_values, sin_values,
                        output, tokens, time, heads, head_dim, stream);
  });
}

extern "C" apxinf_status_t apxinf_tts_sinusoidal(
    apxinf_runtime_t runtime, const void* times, void* output, int64_t rows,
    int64_t width, float scale, float log_step, uint32_t dtype,
    apxinf_cuda_stream_t raw_stream) {
  return apxinf::framework::abi_boundary([&] {
    if (!runtime) throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "null runtime");
    validate_pointers(times, output, "sinusoidal");
    validate_positive(rows, "sinusoidal rows");
    validate_positive(width, "sinusoidal width");
    if (width < 4 || width % 2 != 0 || !std::isfinite(scale) ||
        !std::isfinite(log_step)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid sinusoidal geometry");
    }
    validate_dtype(dtype, "sinusoidal");
    apxinf::framework::check_cuda(cudaSetDevice(runtime->device));
    const cudaStream_t stream = static_cast<cudaStream_t>(raw_stream);
    APXINF_TTS_DISPATCH(dtype, launch_sinusoidal, times, output, rows,
                        width, scale, log_step, stream);
  });
}

extern "C" apxinf_status_t apxinf_tts_mean_rows(
    apxinf_runtime_t runtime, const void* input, void* output, int64_t rows,
    int64_t cols, uint32_t dtype, apxinf_cuda_stream_t raw_stream) {
  return apxinf::framework::abi_boundary([&] {
    if (!runtime) throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "null runtime");
    validate_pointers(input, output, "mean rows");
    validate_positive(rows, "mean rows rows");
    validate_positive(cols, "mean rows cols");
    if (rows > INT_MAX || cols > INT_MAX) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "mean rows overflow");
    }
    validate_dtype(dtype, "mean rows");
    apxinf::framework::check_cuda(cudaSetDevice(runtime->device));
    const cudaStream_t stream = static_cast<cudaStream_t>(raw_stream);
    APXINF_TTS_DISPATCH(dtype, launch_mean_rows, input, output, rows, cols,
                        stream);
  });
}

extern "C" apxinf_status_t apxinf_tts_argmax_rows(
    apxinf_runtime_t runtime, const void* logits, void* index, int64_t rows,
    int64_t cols, uint32_t dtype, apxinf_cuda_stream_t raw_stream) {
  return apxinf::framework::abi_boundary([&] {
    if (!runtime) throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "null runtime");
    validate_pointers(logits, index, "argmax rows");
    validate_positive(rows, "argmax rows rows");
    validate_positive(cols, "argmax rows cols");
    if (rows > INT_MAX || cols > INT_MAX) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "argmax rows overflow");
    }
    validate_dtype(dtype, "argmax rows");
    apxinf::framework::check_cuda(cudaSetDevice(runtime->device));
    const cudaStream_t stream = static_cast<cudaStream_t>(raw_stream);
    APXINF_TTS_DISPATCH(dtype, launch_argmax_rows, logits, index, rows,
                        cols, stream);
  });
}

extern "C" apxinf_status_t apxinf_tts_write_kv_cache_token(
    apxinf_runtime_t runtime, const void* input, void* cache,
    const uint32_t* meta, int64_t capacity, int64_t row_elements,
    uint32_t dtype, apxinf_cuda_stream_t raw_stream) {
  return apxinf::framework::abi_boundary([&] {
    if (!runtime) throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "null runtime");
    if (input == nullptr || cache == nullptr || meta == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "null cache token binding");
    }
    validate_positive(capacity, "cache capacity");
    validate_positive(row_elements, "cache row elements");
    if (dtype != APXINF_DTYPE_F16 && dtype != APXINF_DTYPE_BF16) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "cache token write supports F16 and BF16 only");
    }
    const int64_t blocks = (row_elements + kDirectThreads - 1) / kDirectThreads;
    if (blocks > INT_MAX) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "cache token write overflow");
    }
    apxinf::framework::check_cuda(cudaSetDevice(runtime->device));
    const cudaStream_t stream = static_cast<cudaStream_t>(raw_stream);
    kernels::cache_write_token_kernel<<<static_cast<int>(blocks),
                                        kDirectThreads, 0, stream>>>(
        static_cast<const uint16_t*>(input), static_cast<uint16_t*>(cache),
        meta, capacity, row_elements);
    apxinf::framework::check_cuda(cudaGetLastError());
  });
}

extern "C" apxinf_status_t apxinf_tts_fill_zero(
    apxinf_runtime_t runtime, void* tensor, int64_t elements, uint32_t dtype,
    apxinf_cuda_stream_t raw_stream) {
  return apxinf::framework::abi_boundary([&] {
    if (!runtime) throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "null runtime");
    if (tensor == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "null fill-zero tensor");
    }
    validate_positive(elements, "fill-zero elements");
    validate_dtype(dtype, "fill zero");
    apxinf::framework::check_cuda(cudaSetDevice(runtime->device));
    const cudaStream_t stream = static_cast<cudaStream_t>(raw_stream);
    apxinf::framework::check_cuda(cudaMemsetAsync(
        tensor, 0, static_cast<size_t>(elements) * element_bytes(dtype), stream));
  });
}
