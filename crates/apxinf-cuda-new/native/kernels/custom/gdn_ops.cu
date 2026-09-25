// Copyright 2026 ApxInf contributors.
//
// Gated DeltaNet recurrent step, the linear-attention half of Qwen3.5.
//
// IMPORTANT: the update rule below is implemented from the architecture's
// documented form. It has NOT been checked against a reference engine running
// this checkpoint, so it validates as "this kernel computes the stated
// recurrence" and not yet as "this recurrence is the model's". Layer-wise
// comparison against a reference is still required before any accuracy claim.
//
// Per value head h, with k-head kh = h / (num_v_heads / num_k_heads):
//
//   S <- S * exp(g)                  decay, g <= 0
//   m  = S @ k                       [v_dim], what the state already predicts
//   d  = (v - m) * beta              the delta rule's correction
//   S <- S + outer(d, k)             rank-1 update
//   o  = S @ q
//
// State is [v_heads, v_dim, k_dim] in f32, which the checkpoint's
// `mamba_ssm_dtype: float32` asks for.

#include "gdn_ops.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <cstdlib>

namespace apxinf::cuda::gdn_ops {
namespace {

// One warp per state row, four rows per block, float4 loads.
//
// The recurrence has a dependency that looks like it forces two passes over
// each row: the delta needs `predicted = S @ k`, which needs the whole row,
// and only then can the row be updated. Reading the row twice costs 2x the
// state traffic, and at 151 MiB of state per token across 48 layers that is
// the dominant cost of this kernel.
//
// A warp can hold one row in registers instead: 32 lanes x float4 = 128
// elements. Decay and the S@k reduction happen on the registers, the warp
// reduces to `predicted`, the delta is applied to the same registers, and the
// row is written back once. One read, one write.
__global__ void recurrent_step_kernel(
    float* __restrict__ state, const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ decay, const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ output, int v_heads, int k_heads, int v_dim,
    int k_dim) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int global_row = blockIdx.x * (blockDim.x >> 5) + warp;
  const int total_rows = v_heads * v_dim;
  if (global_row >= total_rows) return;

  const int head = global_row / v_dim;
  const int v_index = global_row % v_dim;
  const int k_head = head / (v_heads / k_heads);

  // Each lane owns four consecutive elements of the row.
  const int chunk = lane * 4;
  if (chunk >= k_dim) return;

  float4* row = reinterpret_cast<float4*>(
      state + (long long)global_row * k_dim);
  float4 s = row[lane];

  const float head_decay = expf(decay[head]);
  const __nv_bfloat16* k_row = k + (long long)k_head * k_dim + chunk;
  const __nv_bfloat16* q_row = q + (long long)k_head * k_dim + chunk;

  const float k0 = __bfloat162float(k_row[0]);
  const float k1 = __bfloat162float(k_row[1]);
  const float k2 = __bfloat162float(k_row[2]);
  const float k3 = __bfloat162float(k_row[3]);

  s.x *= head_decay;
  s.y *= head_decay;
  s.z *= head_decay;
  s.w *= head_decay;

  float predicted = s.x * k0 + s.y * k1 + s.z * k2 + s.w * k3;
  for (int offset = 16; offset > 0; offset >>= 1) {
    predicted += __shfl_down_sync(0xFFFFFFFFu, predicted, offset);
  }
  predicted = __shfl_sync(0xFFFFFFFFu, predicted, 0);

  const float delta =
      (__bfloat162float(v[head * v_dim + v_index]) - predicted) * beta[head];

  s.x += delta * k0;
  s.y += delta * k1;
  s.z += delta * k2;
  s.w += delta * k3;
  row[lane] = s;

  float out = s.x * __bfloat162float(q_row[0]) +
              s.y * __bfloat162float(q_row[1]) +
              s.z * __bfloat162float(q_row[2]) +
              s.w * __bfloat162float(q_row[3]);
  for (int offset = 16; offset > 0; offset >>= 1) {
    out += __shfl_down_sync(0xFFFFFFFFu, out, offset);
  }
  if (lane == 0) {
    // The reference scales the query by 1/sqrt(k_dim) before the recurrence
    // (torch_recurrent_gated_delta_rule: `query = query / query.shape[-1]**0.5`,
    // applied after the optional q/k L2 norm and unconditionally). q enters
    // only here, in the readout, so scaling the reduction is the same thing
    // and costs one multiply per row instead of one per element.
    //
    // This is not cosmetic even though a uniform factor would normally wash
    // out in the gated RMSNorm that follows. At this model's magnitudes
    // mean(core^2) is ~7e-10, far below `epsilon` = 1e-6, so that norm is
    // epsilon-dominated and very nearly linear -- it passes an input scale
    // error through instead of cancelling it.
    output[head * v_dim + v_index] =
        __float2bfloat16(out * rsqrtf(static_cast<float>(k_dim)));
  }
}

// Gated RMSNorm over each value head, then the z gate.
//
// Qwen3.5 uses `output_gate_type: swish`, so the gate is silu(z) rather than
// the sigmoid some GDN variants use -- reusing a sigmoid-gated path would be
// silently wrong.
__global__ void gated_norm_kernel(const __nv_bfloat16* __restrict__ input,
                                  const __nv_bfloat16* __restrict__ gate,
                                  const __nv_bfloat16* __restrict__ weight,
                                  __nv_bfloat16* __restrict__ output,
                                  int heads, int head_dim, float epsilon) {
  const int head = blockIdx.x;
  if (head >= heads) return;
  const long long base = (long long)head * head_dim;

  float sum = 0.0f;
  for (int index = threadIdx.x; index < head_dim; index += blockDim.x) {
    const float value = __bfloat162float(input[base + index]);
    sum += value * value;
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xFFFFFFFFu, sum, offset);
  }
  __shared__ float total;
  if (threadIdx.x == 0) total = sum;
  __syncthreads();

  const float scale = rsqrtf(total / static_cast<float>(head_dim) + epsilon);
  for (int index = threadIdx.x; index < head_dim; index += blockDim.x) {
    const float normalized = __bfloat162float(input[base + index]) * scale *
                             __bfloat162float(weight[index]);
    const float z = __bfloat162float(gate[base + index]);
    output[base + index] = __float2bfloat16(normalized * z / (1.0f + __expf(-z)));
  }
}

// Causal depthwise conv1d over the recurrent window, advancing the ring of
// past inputs. Decode sees one token at a time, so the window is state.
__global__ void causal_conv_step_kernel(
    float* __restrict__ window, const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight, __nv_bfloat16* __restrict__ output,
    int channels, int kernel_width) {
  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  if (channel >= channels) return;

  float* slot = window + (long long)channel * kernel_width;
  // Shift the window by one and append the new sample. kernel_width is 4 here,
  // so an explicit shift beats any ring-buffer indexing.
  for (int index = 0; index < kernel_width - 1; ++index) {
    slot[index] = slot[index + 1];
  }
  slot[kernel_width - 1] = __bfloat162float(input[channel]);

  float accumulator = 0.0f;
  for (int index = 0; index < kernel_width; ++index) {
    accumulator +=
        slot[index] * __bfloat162float(weight[channel * kernel_width + index]);
  }
  // The conv is followed by SiLU in this architecture.
  output[channel] =
      __float2bfloat16(accumulator / (1.0f + __expf(-accumulator)));
}

// L2-normalize each head of q and k, which the delta rule requires for
// stability.
__global__ void l2_normalize_heads_kernel(__nv_bfloat16* __restrict__ data,
                                          int heads, int head_dim,
                                          float epsilon) {
  const int head = blockIdx.x;
  if (head >= heads) return;
  const long long base = (long long)head * head_dim;

  float sum = 0.0f;
  for (int index = threadIdx.x; index < head_dim; index += blockDim.x) {
    const float value = __bfloat162float(data[base + index]);
    sum += value * value;
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xFFFFFFFFu, sum, offset);
  }
  __shared__ float total;
  if (threadIdx.x == 0) total = sum;
  __syncthreads();

  const float scale = rsqrtf(total + epsilon);
  for (int index = threadIdx.x; index < head_dim; index += blockDim.x) {
    data[base + index] =
        __float2bfloat16(__bfloat162float(data[base + index]) * scale);
  }
}

// g = -exp(A_log) * softplus(a + dt_bias), the log decay; beta = sigmoid(b).
__global__ void decay_and_beta_kernel(const __nv_bfloat16* __restrict__ a,
                                      const __nv_bfloat16* __restrict__ b,
                                      const __nv_bfloat16* __restrict__ a_log,
                                      const __nv_bfloat16* __restrict__ dt_bias,
                                      float* __restrict__ decay,
                                      float* __restrict__ beta, int heads) {
  const int head = blockIdx.x * blockDim.x + threadIdx.x;
  if (head >= heads) return;
  const float shifted =
      __bfloat162float(a[head]) + __bfloat162float(dt_bias[head]);
  // softplus, guarded so large inputs do not overflow the exponential.
  const float softplus =
      shifted > 20.0f ? shifted : log1pf(__expf(shifted));
  decay[head] = -__expf(__bfloat162float(a_log[head])) * softplus;
  const float raw = __bfloat162float(b[head]);
  beta[head] = 1.0f / (1.0f + __expf(-raw));
}


// Batched causal depthwise conv1d over a whole prompt, then SiLU.
//
// Structurally this follows Dao-AILab/causal-conv1d's forward kernel, but its
// parallelization does NOT transfer, because the layouts are transposed.
// That reference stores x as [batch, channel, seqlen] -- one channel's
// timesteps are contiguous -- so it vectorizes along time and passes the
// kernel_width-1 boundary values between threads through shared memory.
//
// Here x is [tokens, channels]: time-major. One channel's timesteps are
// strided by `channels`, so vectorizing along time would scatter. Assigning
// one thread per channel instead makes adjacent threads read adjacent
// channels -- fully coalesced -- and, because each thread owns its channel for
// the whole sequence, the cross-thread history exchange disappears entirely.
// The window lives in registers and never reaches shared memory.
//
// What is taken from the reference is the arithmetic: the causal window
// indexing, the zero left boundary, and folding SiLU into the same pass.
__global__ void causal_conv_forward_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output, float* __restrict__ window,
    int tokens, int channels, int kernel_width) {
  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  if (channel >= channels) return;

  // kernel_width is 4 for this checkpoint; a fixed-size register window beats
  // indexing through memory. history[kernel_width-1] is the newest sample.
  constexpr int kMaxWidth = 8;
  float history[kMaxWidth];
#pragma unroll
  for (int index = 0; index < kMaxWidth; ++index) history[index] = 0.0f;

  float weights[kMaxWidth];
  for (int index = 0; index < kernel_width; ++index) {
    weights[index] =
        __bfloat162float(weight[(long long)channel * kernel_width + index]);
  }

  // Prefill starts from a zero window: token 0 sees only itself, which is what
  // a left-padded causal conv means. The single-token path carries its window
  // across calls instead, and this kernel reseeds it at the end.
  for (int token = 0; token < tokens; ++token) {
    const float sample =
        __bfloat162float(input[(long long)token * channels + channel]);
    for (int index = 0; index < kernel_width - 1; ++index) {
      history[index] = history[index + 1];
    }
    history[kernel_width - 1] = sample;

    float accumulator = 0.0f;
    for (int index = 0; index < kernel_width; ++index) {
      accumulator += history[index] * weights[index];
    }
    output[(long long)token * channels + channel] =
        __float2bfloat16(accumulator / (1.0f + __expf(-accumulator)));
  }

  // Hand the tail to the single-token path. Without this, the first decode
  // step after a prompt convolves against an empty window and is wrong --
  // silently, because the shapes still line up.
  if (window != nullptr) {
    float* slot = window + (long long)channel * kernel_width;
    for (int index = 0; index < kernel_width; ++index) {
      slot[index] = history[index];
    }
  }
}

// Sequence-axis decay and beta: the single-token kernel with a token axis.
// Same arithmetic, one element per (token, head).
__global__ void decay_and_beta_seq_kernel(
    const __nv_bfloat16* __restrict__ a, const __nv_bfloat16* __restrict__ b,
    const __nv_bfloat16* __restrict__ a_log,
    const __nv_bfloat16* __restrict__ dt_bias, float* __restrict__ decay,
    float* __restrict__ beta, int tokens, int heads) {
  const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const long long total = (long long)tokens * heads;
  if (index >= total) return;
  // a_log and dt_bias are per-head parameters, shared across tokens.
  const int head = (int)(index % heads);

  const float shifted =
      __bfloat162float(a[index]) + __bfloat162float(dt_bias[head]);
  const float softplus = shifted > 20.0f ? shifted : log1pf(__expf(shifted));
  decay[index] = -__expf(__bfloat162float(a_log[head])) * softplus;
  const float raw = __bfloat162float(b[index]);
  beta[index] = 1.0f / (1.0f + __expf(-raw));
}

// Sequence-axis gated norm: one block per (token, head).
__global__ void gated_norm_seq_kernel(const __nv_bfloat16* __restrict__ input,
                                      const __nv_bfloat16* __restrict__ gate,
                                      const __nv_bfloat16* __restrict__ weight,
                                      __nv_bfloat16* __restrict__ output,
                                      int tokens, int heads, int head_dim,
                                      float epsilon) {
  const long long row = blockIdx.x;
  if (row >= (long long)tokens * heads) return;
  const long long base = row * head_dim;

  float sum = 0.0f;
  for (int index = threadIdx.x; index < head_dim; index += blockDim.x) {
    const float value = __bfloat162float(input[base + index]);
    sum += value * value;
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xFFFFFFFFu, sum, offset);
  }
  __shared__ float total;
  if (threadIdx.x == 0) total = sum;
  __syncthreads();

  const float scale = rsqrtf(total / static_cast<float>(head_dim) + epsilon);
  for (int index = threadIdx.x; index < head_dim; index += blockDim.x) {
    const float normalized = __bfloat162float(input[base + index]) * scale *
                             __bfloat162float(weight[index]);
    const float z = __bfloat162float(gate[base + index]);
    output[base + index] =
        __float2bfloat16(normalized * z / (1.0f + __expf(-z)));
  }
}


// Widen FP16 back to BF16. The FlashInfer scan emits FP16; everything
// downstream of it in this model is BF16.
__global__ void widen_f16_to_bf16_kernel(const __half* __restrict__ input,
                                         __nv_bfloat16* __restrict__ output,
                                         long long count) {
  const long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  output[index] = __float2bfloat16(__half2float(input[index]));
}

// Prepare a chunk of the GDN projection for the FlashInfer prefill kernel.
//
// That kernel wants what ours does not: q, k and v split into separate
// contiguous FP16 tensors, q and k L2-normalized by the caller (it refuses to
// do it itself), q left unscaled (the scale is a kernel argument), and the
// decay as a linear-space alpha rather than our natural log.
//
// All of that is one pass over the conv output. As separate steps it would
// reread the same bytes four times, and at 2048 tokens the projection is 40 MB
// -- enough for the conversion to cost as much as the kernel it feeds.
//
//   fused   [tokens, row_width] bf16, q | k | v laid out along each row
//   q_out   [tokens, k_heads, dim] fp16, L2-normalized
//   k_out   [tokens, k_heads, dim] fp16, L2-normalized
//   v_out   [tokens, v_heads, dim] fp16
//   g_in    [tokens, v_heads] f32, natural-log decay
//   alpha   [tokens, v_heads] f32, exp(g)
//
// One block per token; the block walks the heads, using all its threads for
// each L2 reduction.
// 1024 threads is the CUDA maximum, so 32 warps bounds the partials.
constexpr int kMaxWarpsPerBlock = 32;

__global__ void gdn_prepare_flashinfer_kernel(
    const __nv_bfloat16* __restrict__ fused, __half* __restrict__ q_out,
    __half* __restrict__ k_out, __half* __restrict__ v_out,
    const float* __restrict__ g_in, float* __restrict__ alpha, int tokens,
    int row_width, int k_heads, int v_heads, int dim, float epsilon) {
  const int token = blockIdx.x;
  if (token >= tokens) return;
  const int tid = threadIdx.x;
  const long long row = (long long)token * row_width;

  // A block reduction, not a warp one. The block is `dim` = 128 threads, so
  // four warps cover a head and a bare __shfl_down_sync would reduce only the
  // first 32 elements -- an L2 norm over a quarter of the vector, which stays
  // finite and plausible-looking while being entirely wrong. (The older
  // l2_normalize_heads_kernel launches 32 threads, so a warp reduction is
  // enough there; this one cannot copy it.)
  __shared__ float partials[kMaxWarpsPerBlock];
  __shared__ float reduced;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int warps = (blockDim.x + 31) / 32;

  // q and k: normalize each head, then narrow.
  for (int pass = 0; pass < 2; ++pass) {
    const int offset = pass * k_heads * dim;
    __half* destination = pass == 0 ? q_out : k_out;
    for (int head = 0; head < k_heads; ++head) {
      const __nv_bfloat16* source = fused + row + offset + (long long)head * dim;
      float sum = 0.0f;
      for (int index = tid; index < dim; index += blockDim.x) {
        const float value = __bfloat162float(source[index]);
        sum += value * value;
      }
      for (int shift = 16; shift > 0; shift >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFFu, sum, shift);
      }
      if (lane == 0) partials[warp] = sum;
      __syncthreads();
      if (tid == 0) {
        float total = 0.0f;
        for (int index = 0; index < warps; ++index) total += partials[index];
        reduced = total;
      }
      __syncthreads();
      const float inverse = rsqrtf(reduced + epsilon);
      __half* out = destination + ((long long)token * k_heads + head) * dim;
      for (int index = tid; index < dim; index += blockDim.x) {
        out[index] = __float2half(__bfloat162float(source[index]) * inverse);
      }
      __syncthreads();
    }
  }

  // v: narrowed as-is. No normalization, no scaling.
  const int v_offset = 2 * k_heads * dim;
  const long long v_span = (long long)v_heads * dim;
  for (long long index = tid; index < v_span; index += blockDim.x) {
    v_out[(long long)token * v_span + index] =
        __float2half(__bfloat162float(fused[row + v_offset + index]));
  }

  // The kernel takes linear-space decay; ours is a natural log.
  for (int head = tid; head < v_heads; head += blockDim.x) {
    alpha[(long long)token * v_heads + head] =
        __expf(g_in[(long long)token * v_heads + head]);
  }
}

}  // namespace

int gdn_widen_f16_to_bf16(const void* input, void* output, long long count,
                          cudaStream_t stream) {
  if (count <= 0) return -1;
  const int threads = 256;
  const long long blocks = (count + threads - 1) / threads;
  widen_f16_to_bf16_kernel<<<(int)blocks, threads, 0, stream>>>(
      static_cast<const __half*>(input),
      static_cast<__nv_bfloat16*>(output), count);
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

int gdn_prepare_flashinfer(const void* fused, void* q_out, void* k_out,
                           void* v_out, const void* g, void* alpha, int tokens,
                           int row_width, int k_heads, int v_heads, int dim,
                           float epsilon, cudaStream_t stream) {
  if (tokens <= 0 || row_width <= 0 || k_heads <= 0 || v_heads <= 0 ||
      dim <= 0) {
    return -1;
  }
  if (2 * k_heads * dim + v_heads * dim > row_width) return -2;
  const int threads = dim >= 128 ? 128 : dim;
  gdn_prepare_flashinfer_kernel<<<tokens, threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(fused), static_cast<__half*>(q_out),
      static_cast<__half*>(k_out), static_cast<__half*>(v_out),
      static_cast<const float*>(g), static_cast<float*>(alpha), tokens,
      row_width, k_heads, v_heads, dim, epsilon);
  return cudaGetLastError() == cudaSuccess ? 0 : -3;
}


int gdn_causal_conv_forward(const void* input, const void* weight,
                            void* output, void* window, int tokens,
                            int channels, int kernel_width,
                            cudaStream_t stream) {
  if (tokens <= 0 || channels <= 0 || kernel_width <= 0) return -1;
  // The register window is fixed at 8; a wider kernel would silently truncate.
  if (kernel_width > 8) return -4;
  const int threads = 256;
  const int blocks = (channels + threads - 1) / threads;
  causal_conv_forward_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<__nv_bfloat16*>(output), static_cast<float*>(window), tokens,
      channels, kernel_width);
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

int gdn_decay_and_beta_seq(const void* a, const void* b, const void* a_log,
                           const void* dt_bias, void* decay, void* beta,
                           int tokens, int heads, cudaStream_t stream) {
  if (tokens <= 0 || heads <= 0) return -1;
  const int threads = 128;
  const long long total = (long long)tokens * heads;
  const long long blocks = (total + threads - 1) / threads;
  decay_and_beta_seq_kernel<<<(int)blocks, threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(a),
      static_cast<const __nv_bfloat16*>(b),
      static_cast<const __nv_bfloat16*>(a_log),
      static_cast<const __nv_bfloat16*>(dt_bias), static_cast<float*>(decay),
      static_cast<float*>(beta), tokens, heads);
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

int gdn_gated_norm_seq(const void* input, const void* gate, const void* weight,
                       void* output, int tokens, int heads, int head_dim,
                       float epsilon, cudaStream_t stream) {
  if (tokens <= 0 || heads <= 0 || head_dim <= 0) return -1;
  const int threads = head_dim >= 32 ? 32 : head_dim;
  const long long blocks = (long long)tokens * heads;
  gated_norm_seq_kernel<<<(int)blocks, threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(gate),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<__nv_bfloat16*>(output), tokens, heads, head_dim, epsilon);
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}


int gdn_recurrent_step(void* state, const void* q, const void* k,
                       const void* v, const void* decay, const void* beta,
                       void* output, int v_heads, int k_heads, int v_dim,
                       int k_dim, cudaStream_t stream) {
  if (v_heads <= 0 || k_heads <= 0 || v_dim <= 0 || k_dim <= 0) return -1;
  if (v_heads % k_heads != 0) return -2;
  // One warp holds a row as 32 lanes x float4, so k_dim must be exactly 128.
  // Rejecting is better than silently taking a slower general path.
  if (k_dim != 128) return -4;
  constexpr int kRowsPerBlock = 4;
  const int threads = kRowsPerBlock * 32;
  const int blocks = (v_heads * v_dim + kRowsPerBlock - 1) / kRowsPerBlock;
  recurrent_step_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<float*>(state), static_cast<const __nv_bfloat16*>(q),
      static_cast<const __nv_bfloat16*>(k),
      static_cast<const __nv_bfloat16*>(v), static_cast<const float*>(decay),
      static_cast<const float*>(beta), static_cast<__nv_bfloat16*>(output),
      v_heads, k_heads, v_dim, k_dim);
  return cudaGetLastError() == cudaSuccess ? 0 : -3;
}

int gdn_gated_norm(const void* input, const void* gate, const void* weight,
                   void* output, int heads, int head_dim, float epsilon,
                   cudaStream_t stream) {
  if (heads <= 0 || head_dim <= 0) return -1;
  const int threads = head_dim >= 32 ? 32 : head_dim;
  gated_norm_kernel<<<heads, threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(gate),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<__nv_bfloat16*>(output), heads, head_dim, epsilon);
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

int gdn_causal_conv_step(void* window, const void* input, const void* weight,
                         void* output, int channels, int kernel_width,
                         cudaStream_t stream) {
  if (channels <= 0 || kernel_width <= 0) return -1;
  const int threads = 256;
  const int blocks = (channels + threads - 1) / threads;
  causal_conv_step_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<float*>(window), static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<__nv_bfloat16*>(output), channels, kernel_width);
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

int gdn_l2_normalize_heads(void* data, int heads, int head_dim, float epsilon,
                           cudaStream_t stream) {
  if (heads <= 0 || head_dim <= 0) return -1;
  const int threads = head_dim >= 32 ? 32 : head_dim;
  l2_normalize_heads_kernel<<<heads, threads, 0, stream>>>(
      static_cast<__nv_bfloat16*>(data), heads, head_dim, epsilon);
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

int gdn_decay_and_beta(const void* a, const void* b, const void* a_log,
                       const void* dt_bias, void* decay, void* beta, int heads,
                       cudaStream_t stream) {
  if (heads <= 0) return -1;
  const int threads = 128;
  const int blocks = (heads + threads - 1) / threads;
  decay_and_beta_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(a),
      static_cast<const __nv_bfloat16*>(b),
      static_cast<const __nv_bfloat16*>(a_log),
      static_cast<const __nv_bfloat16*>(dt_bias), static_cast<float*>(decay),
      static_cast<float*>(beta), heads);
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}


// ---------------------------------------------------------------------------
// Gated DeltaNet chunked scan (parallel prefill).
//
// A faithful port of torch_chunk_gated_delta_rule (the forward-substitution
// export path) from modeling_qwen3_5.py. One block per value head; the chunk
// loop is sequential inside the block, but the intra-chunk work is a handful of
// 64x64 / 64x128 matmuls over the whole chunk at once, so a prefill of N tokens
// runs in N/chunk_size sequential steps instead of N.
//
// Conventions matched to the reference and to gdn_recurrent_step:
//   - q and k are L2-normalized per head in fp32 (use_qk_l2norm_in_kernel), then
//     q is scaled by k_dim**-0.5. Both are done here, in fp32.
//   - g is the log decay (<=0); beta the delta-rule gate. Both per (token, head).
//   - The recurrent state is carried in the PORT layout [v_heads, v_dim, k_dim]
//     (== reference [v_heads, k_dim, v_dim] with the last two axes transposed),
//     so prefill leaves exactly the state the single-token path expects.
//   - value head h reads key head h / (v_heads / k_heads).
//
// chunk_size is fixed at 64 and k_dim == v_dim == 128 for this checkpoint.
// ---------------------------------------------------------------------------

namespace {

constexpr int kChunk = 64;   // chunk_size the caller uses
constexpr int kDim = 128;    // k_dim == v_dim for this model

// ---------------------------------------------------------------------------
// Shared-memory scratch for one chunk of one head.
//
// Only operands that one thread reads at a column other than its own have to
// live here. Buffer lifetimes over one chunk iteration (phases as labelled in
// the body below):
//
//   phase                       q    k    pw      T      M    v/nv/kcd chain
//   A load                      W    W    -       -      -    W (private)
//   B l2norm                    RW   RW   -       -      -    -
//   C cum scan                  -    -    -       -      -    -
//   D pairwise decay            -    -    W(dec)  -      -    -
//   E ut_system + intra_attn    R    R    R(dec)  W(at)  W    -
//   F forward substitution      -    -    W(inv)  -      R    -       <- M dies
//   G w = beta*(v - e^cum*kS)   -    R    -       -      -    RW
//   H v_new = T_inv @ w         -    -    R(inv)  -      -    RW      <- pw dies
//   I out = q@S + intra@v_new   R    -    -       R(at)  -    R       <- q,T die
//   J state fold                R    R    -       -      -    R       <- k dies
//
// The chain that was `v` -> `nv`/`kcd` -> `v_new` (96 KiB of the old 209 KiB)
// is only ever touched at the owning thread's v-dim column, so it becomes a
// per-thread array. What makes that work is re-associating the v_new formula:
//
//   v_new = T @ (beta*v) - (T @ (beta*e^cum*k)) @ S      (needs kcd across k)
//         = T @ ( beta * (v - e^cum * (k @ S)) )         (all at column d)
//
// which is the same operation count, just contracted over k before the
// triangular solve instead of after it, and costs nothing in precision.
// The remaining five buffers overlap almost everywhere, so aliasing them buys
// little: only `M` dies early and nothing later needs 16 KiB. Getting below
// the 5-buffer floor therefore needs either a narrower dtype or a shorter
// chunk, which is what the template parameters below select.
//
//   CHUNK=64 float/float  112.75 KiB    CHUNK=32 float/float   44.375 KiB
//   CHUNK=64 bf16 /float   80.75 KiB    CHUNK=32 bf16 /float   28.375 KiB
//   CHUNK=64 bf16 /bf16    56.75 KiB    CHUNK=32 bf16 /bf16    22.375 KiB
//
// against 228 KiB per SM. Blocks per SM is min(smem, register) limited; the
// register side is what MINBLK (a __launch_bounds__ minimum) pins down.
// ---------------------------------------------------------------------------

__device__ __forceinline__ float ldv(float x) { return x; }
__device__ __forceinline__ float ldv(__nv_bfloat16 x) {
  return __bfloat162float(x);
}
__device__ __forceinline__ void stv(float& d, float v) { d = v; }
__device__ __forceinline__ void stv(__nv_bfloat16& d, float v) {
  d = __float2bfloat16(v);
}

// QKT is the dtype of the q/k operand planes, MT of the three CHUNK x CHUNK
// matrices (pairwise decay / UT inverse, intra-chunk attention, ut_system).
template <int CHUNK, typename QKT, typename MT>
struct ChunkSharedT {
  QKT q[CHUNK][kDim];    // l2-normed q, scaled by k_dim**-0.5
  QKT k[CHUNK][kDim];    // l2-normed k
  MT pw[CHUNK][CHUNK];   // pairwise_decay, then the UT-transform inverse
  MT at[CHUNK][CHUNK];   // intra_chunk_attn
  MT ut[CHUNK][CHUNK];   // ut_system
  float cum[CHUNK];      // cumulative decay within the chunk
  float g[CHUNK];        // per-token log decay for this head
  float beta[CHUNK];     // per-token beta for this head
};

// gdn chunked scan. Grid = v_heads blocks, block = kDim (128) threads.
//
//   q_in, k_in : [seq_padded, k_heads, kDim]  bf16   (post-conv, pre-l2norm)
//   v_in       : [seq_padded, v_heads, kDim]  bf16
//   g_in, b_in : [seq_padded, v_heads]        f32    (log decay, beta)
//   out        : [seq_padded, v_heads, kDim]  bf16   (core_attn_out)
//   state      : [v_heads, kDim(v_dim), kDim(k_dim)] f32  port layout, in/out
//
// The block loads a whole chunk, builds the UT transform by forward
// substitution, then folds the chunk into the recurrent state. State rows are
// held in global memory (port layout state[v][k]) and read/written per chunk.
template <int CHUNK, typename QKT, typename MT, int MINBLK>
__global__ __launch_bounds__(kDim, MINBLK) void chunk_scan_kernel_t(
    const __nv_bfloat16* __restrict__ q_in,
    const __nv_bfloat16* __restrict__ k_in,
    const __nv_bfloat16* __restrict__ v_in, const float* __restrict__ g_in,
    const float* __restrict__ b_in, __nv_bfloat16* __restrict__ out,
    int q_row_stride, int k_row_stride, int v_row_stride,
    float* __restrict__ state, int seq_padded, int v_heads, int k_heads,
    int num_chunks) {
  extern __shared__ char smem_raw[];
  auto& s = *reinterpret_cast<ChunkSharedT<CHUNK, QKT, MT>*>(smem_raw);

  const int head = blockIdx.x;
  if (head >= v_heads) return;
  const int tid = threadIdx.x;
  const int k_head = head / (v_heads / k_heads);

  float* head_state = state + (long long)head * kDim * kDim;

  // Thread `tid` owns v-dim column d == tid throughout.
  const int d = tid;
  float vcol[CHUNK];     // v, then the delta w, then v_new -- all at column d
  float sref_col[kDim];  // sref_col[k] = S_ref[k][d] = state_port[d][k]

  for (int c = 0; c < num_chunks; ++c) {
    const int base = c * CHUNK;  // first token of this chunk

    // --- A: load q, k, v for the chunk --------------------------------------
    // Each thread owns column `tid` (a dimension index) across all rows.
    for (int t = 0; t < CHUNK; ++t) {
      const int tok = base + t;
      const __nv_bfloat16* qp =
          q_in + (long long)tok * q_row_stride + (long long)k_head * kDim;
      const __nv_bfloat16* kp =
          k_in + (long long)tok * k_row_stride + (long long)k_head * kDim;
      const __nv_bfloat16* vp =
          v_in + (long long)tok * v_row_stride + (long long)head * kDim;
      stv(s.q[t][tid], __bfloat162float(qp[tid]));
      stv(s.k[t][tid], __bfloat162float(kp[tid]));
      vcol[t] = __bfloat162float(vp[tid]);
    }
    // g and beta for this head over the chunk (CHUNK <= kDim so tid covers it).
    if (tid < CHUNK) {
      s.g[tid] = g_in[(long long)(base + tid) * v_heads + head];
      s.beta[tid] = b_in[(long long)(base + tid) * v_heads + head];
    }
    __syncthreads();

    // --- B: L2-normalize q and k over the dimension, scale q by kDim**-0.5 --
    // Thread `tid` normalizes row `tid` (CHUNK <= blockDim.x).
    const float q_scaling = rsqrtf((float)kDim);
    if (tid < CHUNK) {
      float qn = 0.0f, kn = 0.0f;
      for (int dd = 0; dd < kDim; ++dd) {
        const float qv = ldv(s.q[tid][dd]);
        const float kv = ldv(s.k[tid][dd]);
        qn += qv * qv;
        kn += kv * kv;
      }
      const float qinv = rsqrtf(qn + 1e-6f);
      const float kinv = rsqrtf(kn + 1e-6f);
      for (int dd = 0; dd < kDim; ++dd) {
        stv(s.q[tid][dd], ldv(s.q[tid][dd]) * qinv * q_scaling);
        stv(s.k[tid][dd], ldv(s.k[tid][dd]) * kinv);
      }
    }
    __syncthreads();

    // --- C: cumulative decay ------------------------------------------------
    // cum[t] = sum_{j<=t} g[j] (inclusive). Single thread does the scan.
    if (tid == 0) {
      float acc = 0.0f;
      for (int t = 0; t < CHUNK; ++t) {
        acc += s.g[t];
        s.cum[t] = acc;
      }
    }
    __syncthreads();

    // --- D: pairwise decay --------------------------------------------------
    // pairwise_decay[i][j] = exp(cum[i] - cum[j]) for i >= j, else 0.
    // Thread `tid` (a column j) fills column j for all rows i, when tid<CHUNK.
    if (tid < CHUNK) {
      const int j = tid;
      for (int i = 0; i < CHUNK; ++i) {
        stv(s.pw[i][j], i >= j ? __expf(s.cum[i] - s.cum[j]) : 0.0f);
      }
    }
    __syncthreads();

    // --- E: ut_system and intra_chunk_attn ----------------------------------
    // v_beta and k_beta are applied on the fly. k_beta[t] = beta[t]*k[t].
    // ut_system[i][j] = (k_beta[i] . k[j]) * pw[i][j]
    // intra[i][j]     = (q[i]     . k[j]) * pw[i][j]
    // This is the heaviest block in the scan. Mapping one thread per column
    // would leave kDim - CHUNK threads idle, so the row range is split too:
    // thread t takes column t % CHUNK and the row slice selected by t / CHUNK.
    // Row slices are disjoint, so no reduction is needed.
    {
      constexpr int kRowGroups = kDim / CHUNK;
      constexpr int kRowsPer = CHUNK / kRowGroups;
      const int j = tid % CHUNK;
      const int i_begin = (tid / CHUNK) * kRowsPer;
      const int i_end = i_begin + kRowsPer;
      for (int i = i_begin; i < i_end; ++i) {
        float utv = 0.0f, atv = 0.0f;
        const float bi = s.beta[i];
        for (int dd = 0; dd < kDim; ++dd) {
          const float kjd = ldv(s.k[j][dd]);
          utv += (bi * ldv(s.k[i][dd])) * kjd;
          atv += ldv(s.q[i][dd]) * kjd;
        }
        const float pwij = ldv(s.pw[i][j]);
        stv(s.ut[i][j], utv * pwij);  // ut_system
        stv(s.at[i][j], atv * pwij);  // intra_chunk_attn, live until phase I
      }
    }
    __syncthreads();

    // --- F: forward substitution, parallel over columns ---------------------
    // This inverts the unit lower triangular (I - L) where
    // L = -tril(ut_system, -1). Done by one thread it is a triple loop --
    // CHUNK^3/6 serial iterations while the other threads wait on the barrier,
    // which measured as the dominant cost of the whole scan.
    //
    // Column j of the inverse depends only on column j, so the columns are
    // independent and one thread can own each. The row loop stays sequential
    // because row i needs rows below it. The key to avoiding the read/write
    // race that forces the serial version is not to materialize L at all:
    // L[i][p] is just -ut[i][p], so the recurrence reads ut (untouched) and
    // writes s.pw (the result), and no thread reads what another is writing.
    const int col = tid;
    if (col < CHUNK) {
      for (int i = 0; i < CHUNK; ++i) stv(s.pw[i][col], 0.0f);
    }
    __syncthreads();

    for (int i = 1; i < CHUNK; ++i) {
      if (col < i) {
        // T[i][col] = L[i][col] + sum_{p in (col, i)} L[i][p] * T[p][col]
        float acc = -ldv(s.ut[i][col]);
        for (int p = col + 1; p < i; ++p) {
          acc += (-ldv(s.ut[i][p])) * ldv(s.pw[p][col]);
        }
        stv(s.pw[i][col], acc);
      }
      __syncthreads();
    }

    if (col < CHUNK) stv(s.pw[col][col], ldv(s.pw[col][col]) + 1.0f);
    __syncthreads();
    // s.pw now holds T (the inverse). s.at holds intra_chunk_attn. s.ut free.

    // --- Scan fold into the recurrent state ---------------------------------
    // Reference (per chunk i):
    //   q_scaled = q * exp(cum)                        [applied per token]
    //   k_scaled = k * exp(cum[-1] - cum)
    //   chunk_decay = exp(cum[-1])
    //   v_new  = new_values - k_cumdecay @ S_ref       (S_ref[k][v])
    //   inter  = q_scaled @ S_ref
    //   out    = inter + intra_chunk_attn @ v_new
    //   S_ref  = S_ref * chunk_decay + k_scaled^T @ v_new
    //
    // We hold S in the PORT layout state[v][k] = S_ref[k][v]. Thread `tid` owns
    // v-column d = tid of the state (so it holds S_ref[*][d], i.e. state[d][*]).
    // Every thread reads the full k-vector of q / k from shared.
    const float cum_last = s.cum[CHUNK - 1];
    const float chunk_decay = __expf(cum_last);

    // Load this thread's column of S_ref: sref_col[k] = S_ref[k][d]
    //   = state_port[d][k] = head_state[d*kDim + k]
    for (int kk = 0; kk < kDim; ++kk) {
      sref_col[kk] = head_state[(long long)d * kDim + kk];
    }

    // --- G: w[p] = beta[p] * (v[p][d] - exp(cum[p]) * (k[p] . S_ref[*][d])) -
    // This is the contraction that used to be deferred to `k_cumdecay @ S_ref`
    // after the triangular solve; pulling it in front of the solve keeps it at
    // column d and removes the CHUNK x kDim k_cumdecay and new_values buffers.
    for (int p = 0; p < CHUNK; ++p) {
      float ks = 0.0f;
      for (int kk = 0; kk < kDim; ++kk) {
        ks += ldv(s.k[p][kk]) * sref_col[kk];
      }
      vcol[p] = s.beta[p] * (vcol[p] - __expf(s.cum[p]) * ks);
    }

    // --- H: v_new = T_inv @ w, in place -------------------------------------
    // T_inv is lower-triangular, so row i reads rows 0..i only. Walking i
    // downwards means every row it reads is still the pre-multiply value, so
    // the product needs no second buffer.
    for (int i = CHUNK - 1; i >= 0; --i) {
      float acc = 0.0f;
      for (int p = 0; p <= i; ++p) {
        acc += ldv(s.pw[i][p]) * vcol[p];
      }
      vcol[i] = acc;
    }
    // vcol now holds v_new[*][d].

    // --- I: out[i][d] = inter[i][d] + sum_{j<=i} intra[i][j] * v_new[j][d] --
    //   inter[i][d] = sum_k (q[i][k] * exp(cum[i])) * S_ref[k][d]
    for (int i = 0; i < CHUNK; ++i) {
      const float ecum_i = __expf(s.cum[i]);
      float inter = 0.0f;
      for (int kk = 0; kk < kDim; ++kk) {
        inter += (ldv(s.q[i][kk]) * ecum_i) * sref_col[kk];
      }
      float intra = 0.0f;
      for (int j = 0; j <= i; ++j) {  // intra is lower-triangular (masked)
        intra += ldv(s.at[i][j]) * vcol[j];
      }
      const int tok = base + i;
      __nv_bfloat16* op = out + ((long long)tok * v_heads + head) * kDim;
      op[d] = __float2bfloat16(inter + intra);
    }

    // --- J: S_ref[k][d] = S_ref[k][d]*chunk_decay + sum_i k_scaled[i][k]*v_new
    //   k_scaled[i][k] = k[i][k] * exp(cum_last - cum[i])
    for (int kk = 0; kk < kDim; ++kk) {
      float upd = 0.0f;
      for (int i = 0; i < CHUNK; ++i) {
        const float kscaled = ldv(s.k[i][kk]) * __expf(cum_last - s.cum[i]);
        upd += kscaled * vcol[i];
      }
      // store back in port layout state[d][k]
      head_state[(long long)d * kDim + kk] = sref_col[kk] * chunk_decay + upd;
    }
    // Phases G..J only touch this thread's own column, but the next iteration
    // overwrites s.q / s.k, so the block still has to meet here.
    __syncthreads();
  }
}

template <int CHUNK, typename QKT, typename MT, int MINBLK>
int launch_chunk_scan(const __nv_bfloat16* q, const __nv_bfloat16* k,
                      const __nv_bfloat16* v, const float* g, const float* b,
                      __nv_bfloat16* out, float* state, int seq_padded,
                      int v_heads, int k_heads, int q_row_stride,
                      int k_row_stride, int v_row_stride, cudaStream_t stream) {
  if (seq_padded % CHUNK != 0) return -3;
  const int num_chunks = seq_padded / CHUNK;
  const size_t shared = sizeof(ChunkSharedT<CHUNK, QKT, MT>);
  auto kernel = chunk_scan_kernel_t<CHUNK, QKT, MT, MINBLK>;
  static bool configured = false;
  if (!configured) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)shared);
    configured = true;
  }
  kernel<<<v_heads, kDim, shared, stream>>>(q, k, v, g, b, out, q_row_stride,
                                            k_row_stride, v_row_stride, state,
                                            seq_padded, v_heads, k_heads,
                                            num_chunks);
  return cudaGetLastError() == cudaSuccess ? 0 : -4;
}

// --- Reference shape, kept so the 209 KiB original can be timed against the
// --- new shapes inside one binary instead of across two builds. -----------
struct ChunkShared {
  float q[kChunk][kDim];    // l2-normed q, scaled by k_dim**-0.5
  float k[kChunk][kDim];    // l2-normed k
  float v[kChunk][kDim];    // raw v (value)
  float nv[kChunk][kDim];   // new_values / v_new
  float kcd[kChunk][kDim];  // k_cumdecay
  float pw[kChunk][kChunk]; // pairwise_decay (lower-tri incl diag), else 0
  float T[kChunk][kChunk];  // forward-substitution inverse (unit lower-tri)
  float M[kChunk][kChunk];  // ut_system, then intra_chunk_attn
  float cum[kChunk];        // cumulative decay within the chunk
  float g[kChunk];          // per-token log decay for this head
  float beta[kChunk];       // per-token beta for this head
};

// gdn chunked scan. Grid = v_heads blocks, block = kDim (128) threads.
//
//   q_in, k_in : [seq_padded, k_heads, kDim]  bf16   (post-conv, pre-l2norm)
//   v_in       : [seq_padded, v_heads, kDim]  bf16
//   g_in, b_in : [seq_padded, v_heads]        f32    (log decay, beta)
//   out        : [seq_padded, v_heads, kDim]  bf16   (core_attn_out)
//   state      : [v_heads, kDim(v_dim), kDim(k_dim)] f32  port layout, in/out
//
// The block loads a whole chunk, builds the UT transform by forward
// substitution, then folds the chunk into the recurrent state. State rows are
// held in global memory (port layout state[v][k]) and read/written per chunk.
__global__ void chunk_scan_kernel_v0(
    const __nv_bfloat16* __restrict__ q_in,
    const __nv_bfloat16* __restrict__ k_in,
    const __nv_bfloat16* __restrict__ v_in, const float* __restrict__ g_in,
    const float* __restrict__ b_in, __nv_bfloat16* __restrict__ out,
    int q_row_stride, int k_row_stride, int v_row_stride,
    float* __restrict__ state, int seq_padded, int v_heads, int k_heads,
    int num_chunks) {
  extern __shared__ char smem_raw[];
  ChunkShared& s = *reinterpret_cast<ChunkShared*>(smem_raw);

  const int head = blockIdx.x;
  if (head >= v_heads) return;
  const int k_head = head / (v_heads / k_heads);
  const int tid = threadIdx.x;  // 0..kDim-1

  // Recurrent state for this head, port layout state[v][k] laid out as
  // state[head * v_dim * k_dim + v * k_dim + k]. Reference state is S_ref[k][v]
  // = state_port[v][k]; we compute against S_ref and store back transposed.
  float* head_state = state + (long long)head * kDim * kDim;

  for (int c = 0; c < num_chunks; ++c) {
    const int base = c * kChunk;  // first token of this chunk

    // --- Load q, k, v for the chunk; L2-normalize q,k in fp32; scale q. ------
    // Each thread owns column `tid` (a dimension index) across all 64 rows.
    // L2 norm is over the dimension, so it needs a per-row reduction; do it the
    // straightforward way with a per-row loop and a warp/block reduction using
    // shared partials. To keep it simple and correct, every thread loads its
    // column, then rows are normalized cooperatively.
    for (int t = 0; t < kChunk; ++t) {
      const int tok = base + t;
      const __nv_bfloat16* qp =
          q_in + (long long)tok * q_row_stride + (long long)k_head * kDim;
      const __nv_bfloat16* kp =
          k_in + (long long)tok * k_row_stride + (long long)k_head * kDim;
      const __nv_bfloat16* vp =
          v_in + (long long)tok * v_row_stride + (long long)head * kDim;
      s.q[t][tid] = __bfloat162float(qp[tid]);
      s.k[t][tid] = __bfloat162float(kp[tid]);
      s.v[t][tid] = __bfloat162float(vp[tid]);
    }
    // g and beta for this head over the chunk (kChunk == kDim so tid covers it).
    if (tid < kChunk) {
      s.g[tid] = g_in[(long long)(base + tid) * v_heads + head];
      s.beta[tid] = b_in[(long long)(base + tid) * v_heads + head];
    }
    __syncthreads();

    // L2-normalize q and k over the dimension (in fp32), scale q by kDim**-0.5.
    // Thread `tid` normalizes row `tid` (kChunk <= blockDim.x).
    const float q_scaling = rsqrtf((float)kDim);
    if (tid < kChunk) {
      float qn = 0.0f, kn = 0.0f;
      for (int d = 0; d < kDim; ++d) {
        qn += s.q[tid][d] * s.q[tid][d];
        kn += s.k[tid][d] * s.k[tid][d];
      }
      const float qinv = rsqrtf(qn + 1e-6f);
      const float kinv = rsqrtf(kn + 1e-6f);
      for (int d = 0; d < kDim; ++d) {
        s.q[tid][d] = s.q[tid][d] * qinv * q_scaling;
        s.k[tid][d] = s.k[tid][d] * kinv;
      }
    }
    __syncthreads();

    // --- Cumulative decay and pairwise decay --------------------------------
    // cum[t] = sum_{j<=t} g[j] (inclusive). Single thread does the scan (64).
    if (tid == 0) {
      float acc = 0.0f;
      for (int t = 0; t < kChunk; ++t) {
        acc += s.g[t];
        s.cum[t] = acc;
      }
    }
    __syncthreads();

    // pairwise_decay[i][j] = exp(cum[i] - cum[j]) for i >= j, else 0.
    // Thread `tid` (a column j) fills column j for all rows i, when tid<kChunk.
    if (tid < kChunk) {
      const int j = tid;
      for (int i = 0; i < kChunk; ++i) {
        s.pw[i][j] = (i >= j) ? __expf(s.cum[i] - s.cum[j]) : 0.0f;
      }
    }
    __syncthreads();

    // --- ut_system and intra_chunk_attn -------------------------------------
    // v_beta and k_beta are applied on the fly. k_beta[t] = beta[t]*k[t].
    // ut_system[i][j] = (k_beta[i] . k[j]) * pw[i][j]
    // intra[i][j]     = (q[i]     . k[j]) * pw[i][j]
    // Both are 64x64, and this is the heaviest block in the scan: 64 rows x
    // 128 depth per column. Mapping one thread per column would leave half the
    // block idle, since there are 64 columns and kDim = 128 threads. Splitting
    // the row range as well keeps every thread busy -- thread t takes column
    // t % 64 and the half of the rows selected by t / 64. Rows are disjoint
    // between the two halves, so no reduction is needed.
    {
      const int j = tid % kChunk;
      const int row_half = tid / kChunk;          // 0 or 1
      const int i_begin = row_half * (kChunk / 2);
      const int i_end = i_begin + (kChunk / 2);
      for (int i = i_begin; i < i_end; ++i) {
        float ut = 0.0f, at = 0.0f;
        const float bi = s.beta[i];
        for (int d = 0; d < kDim; ++d) {
          const float kjd = s.k[j][d];
          ut += (bi * s.k[i][d]) * kjd;
          at += s.q[i][d] * kjd;
        }
        s.M[i][j] = ut * s.pw[i][j];  // ut_system
        // stash intra_chunk_attn in T temporarily (reused before T is needed)
        s.T[i][j] = at * s.pw[i][j];  // intra_chunk_attn
      }
    }
    __syncthreads();

    // Forward substitution to build the unit-lower-triangular inverse.
    //   ut = -ut_system.tril(-1)                (strictly lower)
    //   for i in 1..n: row_i[:i] += sum_p row_i[p]*sub[p][:i]
    //   ut += I
    // This is inherently sequential over rows i, and each row's update reads the
    // already-updated rows above it, so a single thread walks it (64x64x64,
    // small). Then copy into s.T's own storage? s.T currently holds intra_attn,
    // which we still need. Put the inverse in a fresh region: reuse s.pw (no
    // longer needed after ut/intra) as the inverse buffer.
    // Forward substitution, parallel over columns.
    //
    // This inverts the unit lower triangular (I - L) where
    // L = -tril(ut_system, -1). Done by one thread it is a triple loop --
    // about 64^3/6 = 44k serial iterations while the other 127 threads wait on
    // the barrier, which measured as the dominant cost of the whole scan.
    //
    // Column j of the inverse depends only on column j, so the columns are
    // independent and one thread can own each. The row loop stays sequential
    // because row i needs rows below it. The key to avoiding the read/write
    // race that forces the serial version is not to materialize L at all:
    // L[i][p] is just -M[i][p], so the recurrence reads M (untouched) and
    // writes s.pw (the result), and no thread reads what another is writing.
    const int col = tid;
    if (col < kChunk) {
      for (int i = 0; i < kChunk; ++i) s.pw[i][col] = 0.0f;
    }
    __syncthreads();

    for (int i = 1; i < kChunk; ++i) {
      if (col < i) {
        // T[i][col] = L[i][col] + sum_{p in (col, i)} L[i][p] * T[p][col]
        float acc = -s.M[i][col];
        for (int p = col + 1; p < i; ++p) {
          acc += (-s.M[i][p]) * s.pw[p][col];
        }
        s.pw[i][col] = acc;
      }
      __syncthreads();
    }

    if (col < kChunk) s.pw[col][col] += 1.0f;
    __syncthreads();
    // s.pw now holds T (the inverse). s.T holds intra_chunk_attn. s.M free.

    // new_values = T @ v_beta ;  k_cumdecay = T @ decayed_k_beta
    //   v_beta[t]        = beta[t] * v[t]
    //   decayed_k_beta[t]= beta[t] * k[t] * exp(cum[t])
    // Thread `tid` owns dimension column d = tid; compute all 64 rows i.
    {
      const int d = tid;  // 0..kDim-1
      for (int i = 0; i < kChunk; ++i) {
        float nvv = 0.0f, kc = 0.0f;
        for (int p = 0; p <= i; ++p) {  // T is lower-triangular
          const float t = s.pw[i][p];
          const float vbeta = s.beta[p] * s.v[p][d];
          const float dkb = s.beta[p] * s.k[p][d] * __expf(s.cum[p]);
          nvv += t * vbeta;
          kc += t * dkb;
        }
        s.nv[i][d] = nvv;
        s.kcd[i][d] = kc;
      }
    }
    __syncthreads();

    // --- Scan fold into the recurrent state ---------------------------------
    // Reference (per chunk i):
    //   q_scaled = q * exp(cum)                        [applied per token]
    //   k_scaled = k * exp(cum[-1] - cum)
    //   chunk_decay = exp(cum[-1])
    //   v_new  = new_values - k_cumdecay @ S_ref       (S_ref[k][v])
    //   inter  = q_scaled @ S_ref
    //   out    = inter + intra_chunk_attn @ v_new
    //   S_ref  = S_ref * chunk_decay + k_scaled^T @ v_new
    //
    // We hold S in the PORT layout state[v][k] = S_ref[k][v]. Thread `tid` owns
    // v-column d = tid of the state (so it holds S_ref[*][d], i.e. state[d][*]).
    // Every thread reads the full k-vector of q_scaled / k_cumdecay from shared.
    const float cum_last = s.cum[kChunk - 1];
    const float chunk_decay = __expf(cum_last);
    const int d = tid;  // this thread owns v-dim index d

    // Load this thread's column of S_ref: sref_col[k] = S_ref[k][d]
    //   = state_port[d][k] = head_state[d*kDim + k]
    float sref_col[kDim];
    for (int kk = 0; kk < kDim; ++kk) {
      sref_col[kk] = head_state[(long long)d * kDim + kk];
    }

    // v_new[i][d] = new_values[i][d] - sum_k k_cumdecay[i][k] * S_ref[k][d]
    for (int i = 0; i < kChunk; ++i) {
      float acc = 0.0f;
      for (int kk = 0; kk < kDim; ++kk) {
        acc += s.kcd[i][kk] * sref_col[kk];
      }
      s.nv[i][d] = s.nv[i][d] - acc;  // s.nv now holds v_new
    }
    __syncthreads();

    // inter[i][d] = sum_k q_scaled[i][k] * S_ref[k][d]
    //   q_scaled[i][k] = q[i][k] * exp(cum[i])
    // out[i][d] = inter[i][d] + sum_j intra[i][j] * v_new[j][d]
    for (int i = 0; i < kChunk; ++i) {
      const float ecum_i = __expf(s.cum[i]);
      float inter = 0.0f;
      for (int kk = 0; kk < kDim; ++kk) {
        inter += (s.q[i][kk] * ecum_i) * sref_col[kk];
      }
      float intra = 0.0f;
      for (int j = 0; j <= i; ++j) {  // intra is lower-triangular (masked)
        intra += s.T[i][j] * s.nv[j][d];
      }
      const int tok = base + i;
      __nv_bfloat16* op = out + ((long long)tok * v_heads + head) * kDim;
      op[d] = __float2bfloat16(inter + intra);
    }
    __syncthreads();

    // S_ref[k][d] = S_ref[k][d] * chunk_decay + sum_i k_scaled[i][k]*v_new[i][d]
    //   k_scaled[i][k] = k[i][k] * exp(cum_last - cum[i])
    for (int kk = 0; kk < kDim; ++kk) {
      float upd = 0.0f;
      for (int i = 0; i < kChunk; ++i) {
        const float kscaled = s.k[i][kk] * __expf(cum_last - s.cum[i]);
        upd += kscaled * s.nv[i][d];
      }
      const float newval = sref_col[kk] * chunk_decay + upd;
      // store back in port layout state[d][k]
      head_state[(long long)d * kDim + kk] = newval;
    }
    __syncthreads();
  }
}


int launch_chunk_scan_v0(const __nv_bfloat16* q, const __nv_bfloat16* k,
                         const __nv_bfloat16* v, const float* g, const float* b,
                         __nv_bfloat16* out, float* state, int seq_padded,
                         int v_heads, int k_heads, int q_row_stride,
                         int k_row_stride, int v_row_stride,
                         cudaStream_t stream) {
  const int num_chunks = seq_padded / kChunk;
  const size_t shared = sizeof(ChunkShared);
  static bool configured = false;
  if (!configured) {
    cudaFuncSetAttribute(chunk_scan_kernel_v0,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)shared);
    configured = true;
  }
  chunk_scan_kernel_v0<<<v_heads, kDim, shared, stream>>>(
      q, k, v, g, b, out, q_row_stride, k_row_stride, v_row_stride, state,
      seq_padded, v_heads, k_heads, num_chunks);
  return cudaGetLastError() == cudaSuccess ? 0 : -4;
}

using bf16 = __nv_bfloat16;

// Which shared-memory shape to run. Kept switchable while the trade between
// occupancy and the precision of the demoted planes is being measured.
int smem_variant() {
  const char* e = getenv("APXINF_GDN_SMEM_VARIANT");
  return e ? atoi(e) : 8;
}

}  // namespace

int gdn_chunk_scan(const void* q, const void* k, const void* v, const void* g,
                   const void* beta, void* out, void* state, int seq_padded,
                   int v_heads, int k_heads, int chunk_size, int k_dim,
                   int num_chunks, int q_row_stride, int k_row_stride,
                   int v_row_stride, cudaStream_t stream) {
  if (v_heads <= 0 || k_heads <= 0 || v_heads % k_heads != 0) return -1;
  if (chunk_size != kChunk || k_dim != kDim) return -2;
  if (num_chunks <= 0 || seq_padded != num_chunks * chunk_size) return -3;

  const auto* qp = static_cast<const __nv_bfloat16*>(q);
  const auto* kp = static_cast<const __nv_bfloat16*>(k);
  const auto* vp = static_cast<const __nv_bfloat16*>(v);
  const auto* gp = static_cast<const float*>(g);
  const auto* bp = static_cast<const float*>(beta);
  auto* op = static_cast<__nv_bfloat16*>(out);
  auto* sp = static_cast<float*>(state);

#define APXINF_GDN_LAUNCH(CHUNK_, QKT_, MT_, MINBLK_)                        \
  launch_chunk_scan<CHUNK_, QKT_, MT_, MINBLK_>(                             \
      qp, kp, vp, gp, bp, op, sp, seq_padded, v_heads, k_heads, q_row_stride, \
      k_row_stride, v_row_stride, stream)

  static const int variant = smem_variant();
  switch (variant) {
    case 0:
      return launch_chunk_scan_v0(qp, kp, vp, gp, bp, op, sp, seq_padded,
                                  v_heads, k_heads, q_row_stride, k_row_stride,
                                  v_row_stride, stream);             // 208.75 KiB
    case 1:  return APXINF_GDN_LAUNCH(64, float, float, 2);  // 112.75 KiB
    case 2:  return APXINF_GDN_LAUNCH(32, float, float, 4);  //  44.38 KiB
    case 3:  return APXINF_GDN_LAUNCH(64, bf16,  float, 2);  //  80.75 KiB
    case 4:  return APXINF_GDN_LAUNCH(64, bf16,  bf16,  3);  //  56.75 KiB
    case 5:  return APXINF_GDN_LAUNCH(32, bf16,  float, 6);  //  28.38 KiB
    case 6:  return APXINF_GDN_LAUNCH(32, float, float, 2);  //  44.38 KiB
    case 7:  return APXINF_GDN_LAUNCH(16, float, float, 6);  //  19.19 KiB
    case 8:  return APXINF_GDN_LAUNCH(32, float, float, 3);  //  44.38 KiB
    case 9:  return APXINF_GDN_LAUNCH(32, float, bf16,  3);  //  38.38 KiB
    case 10: return APXINF_GDN_LAUNCH(16, float, float, 2);  //  19.19 KiB
    case 11: return APXINF_GDN_LAUNCH(16, float, float, 3);  //  19.19 KiB
    case 12: return APXINF_GDN_LAUNCH(16, float, float, 4);  //  19.19 KiB
    case 13: return APXINF_GDN_LAUNCH(32, float, float, 5);  //  44.38 KiB
    default: return APXINF_GDN_LAUNCH(32, float, float, 3);
  }
#undef APXINF_GDN_LAUNCH
}

}  // namespace apxinf::cuda::gdn_ops
