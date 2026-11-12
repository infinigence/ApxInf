#pragma once

#include <stdint.h>

#include "gemm_types.h"
#include "status.h"
#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Speech-model elementwise and layout operators.
 *
 * These semantics are not part of the language-model Pointwise ABI: they
 * carry speech-specific activations (Mish, LeakyReLU variants, Tanh) and
 * whole-activation statistics (GRN, column normalization). Keeping them in a
 * separate adapter avoids changing the stable Pointwise encoding.
 */

typedef enum {
  APXINF_TTS_ACTIVATION_NONE = 0,
  APXINF_TTS_ACTIVATION_GELU = 1,
  APXINF_TTS_ACTIVATION_SILU = 2,
  APXINF_TTS_ACTIVATION_GELU_EXACT = 3,
  APXINF_TTS_ACTIVATION_MISH = 4,
  APXINF_TTS_ACTIVATION_LEAKY_RELU = 5,
  APXINF_TTS_ACTIVATION_TANH = 6,
  APXINF_TTS_ACTIVATION_SIGMOID = 7,
  APXINF_TTS_ACTIVATION_RELU = 8,
  APXINF_TTS_ACTIVATION_LEAKY_RELU_ONE_PERCENT = 9,
} apxinf_tts_activation_t;

/* output[i] = activation(input[i] + bias[i % cols]); `bias` may be null.
 * With `round_bias`, the sum is rounded to the tensor dtype before the
 * activation. Use it when the bias a producer deferred has to reproduce the
 * value that producer would have written itself. */
apxinf_status_t apxinf_tts_bias_activation(
    apxinf_runtime_t runtime, const void* input, const void* bias, void* output,
    int64_t count, int64_t cols, uint32_t dtype, uint32_t activation,
    uint32_t round_bias, apxinf_cuda_stream_t stream);

/* output[i] = round_dtype(round_dtype(input[i] + bias[i % cols]) + residual[i]).
 * `bias` and `residual` are required; this is the deferred-bias form of the
 * ordinary residual add, in one launch. */
apxinf_status_t apxinf_tts_residual_add_bias(
    apxinf_runtime_t runtime, const void* input, const void* bias,
    const void* residual, void* output, int64_t count, int64_t cols,
    uint32_t dtype, apxinf_cuda_stream_t stream);

/* style is [2 * cols]: output = input * style[col] + style[cols + col]. */
apxinf_status_t apxinf_tts_affine(
    apxinf_runtime_t runtime, const void* input, const void* style, void* output,
    int64_t count, int64_t cols, uint32_t dtype, apxinf_cuda_stream_t stream);

/* output = input * sigmoid(gate), same width. */
apxinf_status_t apxinf_tts_glu_separate(
    apxinf_runtime_t runtime, const void* input, const void* gate, void* output,
    int64_t count, uint32_t dtype, apxinf_cuda_stream_t stream);

/* output[row, col] = tanh(a[row, col]) * sigmoid(a[row, cols + col]),
 * where a = input (+ secondary when has_secondary is set). */
apxinf_status_t apxinf_tts_tanh_sigmoid_gate(
    apxinf_runtime_t runtime, const void* input, const void* secondary,
    void* output, int64_t count, int64_t cols, uint32_t dtype,
    uint32_t has_secondary, apxinf_cuda_stream_t stream);

/* Global response normalization; style is [2 * cols] (gamma then beta). */
apxinf_status_t apxinf_tts_grn(
    apxinf_runtime_t runtime, const void* input, const void* style, void* output,
    int64_t rows, int64_t cols, uint32_t dtype, apxinf_cuda_stream_t stream);

/* Per-column LayerNorm over rows; style is [2 * cols] (weight then bias). */
apxinf_status_t apxinf_tts_column_norm(
    apxinf_runtime_t runtime, const void* input, const void* style, void* output,
    int64_t rows, int64_t cols, uint32_t dtype, float eps,
    apxinf_cuda_stream_t stream);

/* Interleaved rotary embedding over [batch, time, heads, head_dim]. */
apxinf_status_t apxinf_tts_rope_apply(
    apxinf_runtime_t runtime, const void* input, const void* cos_values,
    const void* sin_values, void* output, int64_t tokens, int64_t time,
    int64_t heads, int64_t head_dim, uint32_t dtype,
    apxinf_cuda_stream_t stream);

/* Sin/cos features from scalar times, matching the reference table formula. */
apxinf_status_t apxinf_tts_sinusoidal(
    apxinf_runtime_t runtime, const void* times, void* output, int64_t rows,
    int64_t width, float scale, float log_step, uint32_t dtype,
    apxinf_cuda_stream_t stream);

/* Column-wise mean over rows, written as one row. */
apxinf_status_t apxinf_tts_mean_rows(
    apxinf_runtime_t runtime, const void* input, void* output, int64_t rows,
    int64_t cols, uint32_t dtype, apxinf_cuda_stream_t stream);

/* Per-row argmax with first-index tie breaking. */
apxinf_status_t apxinf_tts_argmax_rows(
    apxinf_runtime_t runtime, const void* logits, void* index, int64_t rows,
    int64_t cols, uint32_t dtype, apxinf_cuda_stream_t stream);

/* Append one token row into a flat cache at meta[1]. */
apxinf_status_t apxinf_tts_write_kv_cache_token(
    apxinf_runtime_t runtime, const void* input, void* cache,
    const uint32_t* meta, int64_t capacity, int64_t row_elements,
    uint32_t dtype, apxinf_cuda_stream_t stream);

apxinf_status_t apxinf_tts_fill_zero(
    apxinf_runtime_t runtime, void* tensor, int64_t elements, uint32_t dtype,
    apxinf_cuda_stream_t stream);

#ifdef __cplusplus
}
#endif
