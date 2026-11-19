#pragma once

#include <stdint.h>

#include "status.h"
#include "types.h"

/* Log-mel frontend element-wise stages.
 *
 * The frontend reuses the GEMM operator for the DFT projections and the mel
 * filterbank; these launches cover the stages between them:
 *
 *   frames:  reflect-pad the waveform by `pad`, gather `frames` rows of
 *            `n_fft` at `hop` stride, multiply by the Hann window
 *
 *   power:   power[i] = re[i] * re[i] + im[i] * im[i]
 *            (float32 in, float32 out, count elements)
 *
 *   log_mel: out = (max(log10(clamp(spec, 1e-10)), global_max - 8) + 4) / 4
 *            cast to float16. The global max is reduced over the whole spec
 *            buffer on device (two passes: reduce, then transform).
 */
#define APXINF_WHISPER_FRONTEND_VERSION 1u

typedef struct {
  const void* re;
  const void* im;
  void* power;
  apxinf_cuda_stream_t stream;
} apxinf_whisper_power_bindings_t;

typedef struct {
  const void* spec;
  void* out_f16;
  void* max_scratch; /* float32, blocks+1: per-block maxima then global max */
  apxinf_cuda_stream_t stream;
} apxinf_whisper_log_mel_bindings_t;

typedef struct {
  const void* waveform; /* float32 [n_samples] */
  const void* hann;     /* float32 [n_fft] */
  void* frames;         /* float32 [frames, n_fft] */
  apxinf_cuda_stream_t stream;
} apxinf_whisper_frames_bindings_t;

#ifdef __cplusplus
extern "C" {
#endif

apxinf_status_t apxinf_whisper_frames_launch(
    apxinf_runtime_t runtime, int64_t n_samples, int64_t n_fft, int64_t hop,
    int64_t pad, int64_t frames,
    const apxinf_whisper_frames_bindings_t* bindings);

apxinf_status_t apxinf_whisper_power_launch(
    apxinf_runtime_t runtime, int64_t count,
    const apxinf_whisper_power_bindings_t* bindings);

apxinf_status_t apxinf_whisper_log_mel_launch(
    apxinf_runtime_t runtime, int64_t rows, int64_t cols,
    const apxinf_whisper_log_mel_bindings_t* bindings);

#ifdef __cplusplus
}
#endif
