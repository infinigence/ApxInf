// Normalization operators for `apxinf_hip::native`: BF16 storage, F32 math.
//
// Each kernel is a port of the cuda-new kernel of the same name
// (`apxinf-cuda-new/native/kernels/custom/norm.cuh`), including where it
// rounds. The fused residual kernels round the residual sum to BF16, store it
// as `hidden`, and normalize the stored value rather than the F32 sum — the
// observable behaviour of two separate BF16 kernels, and what pi05 was
// validated against on CUDA.
//
// One block of kReduceBlock threads per row. Only the reduction order differs
// from CUDA (a shared-memory tree instead of warp shuffles), so results agree
// to within the rounding of an F32 sum, not bit for bit.

#include "common.h"

namespace {

using namespace apxinf_hip;
using bf16 = uint16_t;

__device__ __forceinline__ float ld(const bf16* p, int64_t i) { return bf16_to_f32(p[i]); }

__global__ void layer_norm_kernel(const bf16* x, const bf16* weight, const bf16* bias, bf16* y,
                                  int cols, float eps) {
  __shared__ float scratch[kReduceBlock];
  const int64_t base = static_cast<int64_t>(blockIdx.x) * cols;
  float sum = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) sum += ld(x, base + c);
  const float mean = block_sum(sum, scratch) / cols;
  float centered_sum = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    const float centered = ld(x, base + c) - mean;
    centered_sum += centered * centered;
  }
  const float inverse_std = rsqrtf(block_sum(centered_sum, scratch) / cols + eps);
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    y[base + c] =
        f32_to_bf16((ld(x, base + c) - mean) * inverse_std * ld(weight, c) + ld(bias, c));
  }
}

// GR00T's DiT conditioning: LayerNorm statistics with the per-call
// `[scale; shift]` modulation `ada_rms_norm` uses, i.e.
// `(x - mean) * rsqrt(var + eps) * (1 + scale) + shift`. The `1 +` offset and
// the single rounding at the store are cuda-new's
// (`native/kernels/norm/fixed_bf16.cuh: adaptive_layer_norm_bf16_kernel`).
__global__ void ada_layer_norm_kernel(const bf16* x, const bf16* style, bf16* y, int cols,
                                      float eps) {
  __shared__ float scratch[kReduceBlock];
  const int64_t base = static_cast<int64_t>(blockIdx.x) * cols;
  float sum = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) sum += ld(x, base + c);
  const float mean = block_sum(sum, scratch) / cols;
  float centered_sum = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    const float centered = ld(x, base + c) - mean;
    centered_sum += centered * centered;
  }
  const float inverse_std = rsqrtf(block_sum(centered_sum, scratch) / cols + eps);
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    const float normalized = (ld(x, base + c) - mean) * inverse_std;
    y[base + c] = f32_to_bf16(normalized * (1.0f + ld(style, c)) + ld(style, cols + c));
  }
}

// `style` is [2 * cols]: scale in [0, cols), shift in [cols, 2 * cols).
__global__ void ada_rms_norm_kernel(const bf16* x, const bf16* style, bf16* y, int cols,
                                    float eps) {
  __shared__ float scratch[kReduceBlock];
  const int64_t base = static_cast<int64_t>(blockIdx.x) * cols;
  float square_sum = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    const float v = ld(x, base + c);
    square_sum += v * v;
  }
  const float inverse_rms = rsqrtf(block_sum(square_sum, scratch) / cols + eps);
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    const float normalized = ld(x, base + c) * inverse_rms;
    y[base + c] = f32_to_bf16(normalized * (1.0f + ld(style, c)) + ld(style, cols + c));
  }
}

// hidden = projection + residual (+ bias), rounded once. Elementwise.
__global__ void bias_residual_kernel(const bf16* projection, const bf16* bias, const bf16* residual,
                                     bf16* hidden, int64_t count, int cols) {
  for (int64_t i = global_index(); i < count; i += grid_stride()) {
    float v = ld(projection, i) + ld(residual, i);
    if (bias != nullptr) v += ld(bias, i % cols);
    hidden[i] = f32_to_bf16(v);
  }
}

// hidden = round(projection + residual (+ bias)); normalized = rms(hidden) * weight.
__global__ void bias_residual_rms_norm_kernel(const bf16* projection, const bf16* bias,
                                              const bf16* residual, const bf16* weight,
                                              bf16* hidden, bf16* normalized, int cols,
                                              float eps) {
  __shared__ float scratch[kReduceBlock];
  const int64_t base = static_cast<int64_t>(blockIdx.x) * cols;
  float square_sum = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    float v = ld(projection, base + c) + ld(residual, base + c);
    if (bias != nullptr) v += ld(bias, c);
    const bf16 rounded = f32_to_bf16(v);
    hidden[base + c] = rounded;
    const float stored = bf16_to_f32(rounded);
    square_sum += stored * stored;
  }
  const float inverse_rms = rsqrtf(block_sum(square_sum, scratch) / cols + eps);
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    normalized[base + c] = f32_to_bf16(ld(hidden, base + c) * inverse_rms * ld(weight, c));
  }
}

// hidden = round(projection + residual (+ bias)); normalized = layer_norm(hidden).
__global__ void bias_residual_layer_norm_kernel(const bf16* projection, const bf16* bias,
                                                const bf16* residual, const bf16* norm_weight,
                                                const bf16* norm_bias, bf16* hidden,
                                                bf16* normalized, int cols, float eps) {
  __shared__ float scratch[kReduceBlock];
  const int64_t base = static_cast<int64_t>(blockIdx.x) * cols;
  float sum = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    float v = ld(projection, base + c) + ld(residual, base + c);
    if (bias != nullptr) v += ld(bias, c);
    const bf16 rounded = f32_to_bf16(v);
    hidden[base + c] = rounded;
    sum += bf16_to_f32(rounded);
  }
  const float mean = block_sum(sum, scratch) / cols;
  float centered_sum = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    const float centered = ld(hidden, base + c) - mean;
    centered_sum += centered * centered;
  }
  const float inverse_std = rsqrtf(block_sum(centered_sum, scratch) / cols + eps);
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    normalized[base + c] = f32_to_bf16((ld(hidden, base + c) - mean) * inverse_std *
                                           ld(norm_weight, c) +
                                       ld(norm_bias, c));
  }
}

// hidden = round(residual + projection * gate), the gate being the third
// segment of this layer's [3 * cols] `gate_style`; normalized is the adaptive
// RMS norm of hidden under the next layer's [2 * cols] `norm_style`.
// hidden = round(projection + residual (+ bias)); normalized is the adaptive
// LayerNorm of hidden under `style`.
//
// The pair this replaces spans a DiT block boundary: a block ends by writing
// `hidden`, and the next begins by normalizing it under its own modulation.
// Fusing them removes a launch and a round trip of `hidden` through memory --
// at 41 rows of 1536 the launch is the larger half, since 126 KB is half a
// microsecond of traffic against about 1.8 us of dispatch.
//
// `hidden` is still written, and written *before* it is read back for the
// statistics, because the separate path rounds it to BF16 there and the
// normalization consumes the rounded value. Computing the mean from the
// unrounded sum would be more accurate and would not match.
__global__ void bias_residual_ada_layer_norm_kernel(const bf16* projection, const bf16* bias,
                                                    const bf16* residual, const bf16* style,
                                                    bf16* hidden, bf16* normalized, int cols,
                                                    float eps) {
  __shared__ float scratch[kReduceBlock];
  const int64_t base = static_cast<int64_t>(blockIdx.x) * cols;
  float sum = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    float v = ld(projection, base + c) + ld(residual, base + c);
    if (bias != nullptr) v += ld(bias, c);
    const bf16 rounded = f32_to_bf16(v);
    hidden[base + c] = rounded;
    sum += bf16_to_f32(rounded);
  }
  const float mean = block_sum(sum, scratch) / cols;
  float centered_sum = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    const float centered = ld(hidden, base + c) - mean;
    centered_sum += centered * centered;
  }
  const float inverse_std = rsqrtf(block_sum(centered_sum, scratch) / cols + eps);
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    const float v = (ld(hidden, base + c) - mean) * inverse_std;
    normalized[base + c] =
        f32_to_bf16(v * (1.0f + ld(style, c)) + ld(style, cols + c));
  }
}

__global__ void ada_gate_residual_rms_norm_kernel(const bf16* projection, const bf16* residual,
                                                  const bf16* gate_style, const bf16* norm_style,
                                                  bf16* hidden, bf16* normalized, int cols,
                                                  float eps) {
  __shared__ float scratch[kReduceBlock];
  const int64_t base = static_cast<int64_t>(blockIdx.x) * cols;
  float square_sum = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    const bf16 rounded =
        f32_to_bf16(ld(residual, base + c) + ld(projection, base + c) * ld(gate_style, 2 * cols + c));
    hidden[base + c] = rounded;
    const float stored = bf16_to_f32(rounded);
    square_sum += stored * stored;
  }
  const float inverse_rms = rsqrtf(block_sum(square_sum, scratch) / cols + eps);
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    const float v = ld(hidden, base + c) * inverse_rms;
    normalized[base + c] =
        f32_to_bf16(v * (1.0f + ld(norm_style, c)) + ld(norm_style, cols + c));
  }
}

const bf16* in(const void* p) { return static_cast<const bf16*>(p); }
bf16* out(void* p) { return static_cast<bf16*>(p); }

bool valid_rows(int64_t rows, int cols) { return rows > 0 && cols > 0 && rows <= INT32_MAX; }

}  // namespace

extern "C" int apxinf_hip_native_layer_norm(const void* x, const void* weight, const void* bias,
                                            void* y, int64_t rows, int cols, float eps,
                                            void* stream) {
  if (!valid_rows(rows, cols)) return kInvalidArgument;
  layer_norm_kernel<<<static_cast<unsigned>(rows), kReduceBlock, 0, as_stream(stream)>>>(
      in(x), in(weight), in(bias), out(y), cols, eps);
  return last_launch_status();
}

extern "C" int apxinf_hip_native_ada_layer_norm(const void* x, const void* style, void* y,
                                               int64_t rows, int cols, float eps, void* stream) {
  if (rows <= 0 || cols <= 0) return kInvalidArgument;
  ada_layer_norm_kernel<<<static_cast<unsigned>(rows), kReduceBlock, 0, as_stream(stream)>>>(
      in(x), in(style), out(y), cols, eps);
  return last_launch_status();
}

extern "C" int apxinf_hip_native_ada_rms_norm(const void* x, const void* style, void* y,
                                              int64_t rows, int cols, float eps, void* stream) {
  if (!valid_rows(rows, cols)) return kInvalidArgument;
  ada_rms_norm_kernel<<<static_cast<unsigned>(rows), kReduceBlock, 0, as_stream(stream)>>>(
      in(x), in(style), out(y), cols, eps);
  return last_launch_status();
}

extern "C" int apxinf_hip_native_bias_residual(const void* projection, const void* bias,
                                               const void* residual, void* hidden, int64_t rows,
                                               int cols, void* stream) {
  if (!valid_rows(rows, cols)) return kInvalidArgument;
  const int64_t count = rows * cols;
  bias_residual_kernel<<<grid_for(count, kElementwiseBlock), kElementwiseBlock, 0,
                         as_stream(stream)>>>(in(projection), in(bias), in(residual), out(hidden),
                                              count, cols);
  return last_launch_status();
}

extern "C" int apxinf_hip_native_bias_residual_rms_norm(const void* projection, const void* bias,
                                                        const void* residual, const void* weight,
                                                        void* hidden, void* normalized,
                                                        int64_t rows, int cols, float eps,
                                                        void* stream) {
  if (!valid_rows(rows, cols)) return kInvalidArgument;
  bias_residual_rms_norm_kernel<<<static_cast<unsigned>(rows), kReduceBlock, 0,
                                  as_stream(stream)>>>(in(projection), in(bias), in(residual),
                                                       in(weight), out(hidden), out(normalized),
                                                       cols, eps);
  return last_launch_status();
}

extern "C" int apxinf_hip_native_bias_residual_layer_norm(
    const void* projection, const void* bias, const void* residual, const void* norm_weight,
    const void* norm_bias, void* hidden, void* normalized, int64_t rows, int cols, float eps,
    void* stream) {
  if (!valid_rows(rows, cols)) return kInvalidArgument;
  bias_residual_layer_norm_kernel<<<static_cast<unsigned>(rows), kReduceBlock, 0,
                                    as_stream(stream)>>>(
      in(projection), in(bias), in(residual), in(norm_weight), in(norm_bias), out(hidden),
      out(normalized), cols, eps);
  return last_launch_status();
}

extern "C" int apxinf_hip_native_bias_residual_ada_layer_norm(
    const void* projection, const void* bias, const void* residual, const void* style,
    void* hidden, void* normalized, int64_t rows, int cols, float eps, void* stream) {
  if (rows <= 0 || cols <= 0) return kInvalidArgument;
  bias_residual_ada_layer_norm_kernel<<<static_cast<unsigned>(rows), kReduceBlock, 0,
                                        as_stream(stream)>>>(
      in(projection), in(bias), in(residual), in(style), out(hidden), out(normalized), cols,
      eps);
  return last_launch_status();
}

extern "C" int apxinf_hip_native_ada_gate_residual_rms_norm(
    const void* projection, const void* residual, const void* gate_style, const void* norm_style,
    void* hidden, void* normalized, int64_t rows, int cols, float eps, void* stream) {
  if (!valid_rows(rows, cols)) return kInvalidArgument;
  ada_gate_residual_rms_norm_kernel<<<static_cast<unsigned>(rows), kReduceBlock, 0,
                                      as_stream(stream)>>>(
      in(projection), in(residual), in(gate_style), in(norm_style), out(hidden), out(normalized),
      cols, eps);
  return last_launch_status();
}
