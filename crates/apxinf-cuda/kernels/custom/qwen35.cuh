#pragma once

// Copyright 2026 apxinf contributors.
// Qwen3.5 hybrid linear-attention kernels (bf16 activations, f32 state).
//
// The Qwen3.5 text stack interleaves full-attention decoder layers with
// gated-delta-net linear-attention layers. This header provides the linear
// layer's causal conv + delta-rule recurrence, the gated RMSNorm, the partial
// RoPE q-split/k-append used by the full-attention layers, the gate multiply,
// and a causal GQA flash prefill.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

// ── Helpers ────────────────────────────────────────────────────────────────

__device__ __forceinline__ float sigmoidf_f32(float x) {
  return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float siluf_f32(float x) {
  return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float softplusf_f32(float x) {
  return x > 20.0f ? x : log1pf(expf(x));
}

// ── MLP: SiLU(gate) * up, elementwise over [seq, intermediate] ─────────────

__global__ void qwen35_silu_mul_kernel(
    const __nv_bfloat16* gate, const __nv_bfloat16* up, __nv_bfloat16* out,
    int64_t count) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    out[index] = __float2bfloat16(
        siluf_f32(__bfloat162float(gate[index])) *
        __bfloat162float(up[index]));
  }
}

//
// ── Linear attention: causal depthwise conv + SiLU with carry state ───────
// weight:  [channels, kernel] bf16
// state:   [(kernel-1), channels] f32, [oldest .. newest]
// output:  [seq, channels] bf16
//
// Grid: (channels + threads - 1) / threads blocks of `threads` threads.
// Each block owns a contiguous channel range, keeps its conv state slice in
// shared memory, and sweeps the sequence. y[s] = silu(Σ_k w[c,k] · x[s-k]).

__global__ void qwen35_conv_silu_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    __nv_bfloat16* output, float* state, int seq, int channels, int kernel) {
  const int tid = threadIdx.x;
  const int c = blockIdx.x * blockDim.x + tid;
  if (c >= channels) return;

  // Shared: weight [kernel] + conv state [kernel-1] per thread-chunk.
  extern __shared__ float s_w[];
  float* s_state = s_w + kernel * blockDim.x;
  for (int k = 0; k < kernel; k++)
    s_w[k * blockDim.x + tid] = __bfloat162float(weight[c * kernel + k]);
  for (int k = 0; k < kernel - 1; k++)
    s_state[k * blockDim.x + tid] = state[k * channels + c];

  // raw_hist[k] holds the RAW (pre-SiLU) input for x[s+k-(kernel-1)] whenever
  // that position is >= 0, otherwise the original carry state. The output may
  // alias the input buffer (in-place), so taps at idx < s must never re-read
  // the overwritten rows; the only live read is input[s] itself. The final
  // raw_hist is exactly the new carry state.
  float raw_hist[8];
  for (int k = 0; k < kernel - 1; k++)
    raw_hist[k] = s_state[k * blockDim.x + tid];
  for (int s = 0; s < seq; s++) {
    // Read the live input BEFORE the in-place output write: the output may
    // alias the input buffer, so any later re-read would see the SiLU'd row.
    const float x_live = __bfloat162float(input[s * channels + c]);
    float acc = 0.0f;
    for (int k = 0; k < kernel; k++) {
      const float x = (k == kernel - 1) ? x_live : raw_hist[k];
      acc += x * s_w[k * blockDim.x + tid];
    }
    output[s * channels + c] = __float2bfloat16(siluf_f32(acc));
    for (int k = 0; k < kernel - 2; k++)
      raw_hist[k] = raw_hist[k + 1];
    raw_hist[kernel - 2] = x_live;
  }
  for (int k = 0; k < kernel - 1; k++)
    state[k * channels + c] = raw_hist[k];
}

// ── Linear attention: gated delta-rule recurrence ─────────────────────────
//
// qkv: [seq, k_heads*kdim*2 + v_heads*vdim] bf16 (post-conv)
// a, b: [seq, v_heads] bf16; a_log, dt_bias: [v_heads] bf16
// recurrent: [v_heads, kdim, vdim] f32 (in/out)
// out: [seq, v_heads*vdim] bf16
//
// Grid: (vdim / QWEN35_V_TILE, v_heads) blocks of QWEN35_V_TILE threads.
// The state tile ([kdim, QWEN35_V_TILE] f32) lives in shared memory for the
// whole sequence sweep, so HBM state traffic is one read + one write.

#define QWEN35_V_TILE 128
#define QWEN35_KMAX 128

// ── DeltaNet q/k norm prepass ──────────────────────────────────────────────
// Normalizes q/k per (token, k_head) in a massively parallel pass so the
// serial recurrence below contains no per-token reductions or shuffles.
// qkv: [seq, 2*k_heads*kdim + v_heads*vdim] bf16 (post-conv)
// qk_out: [seq, k_heads, 2, kdim] bf16 normalized rows
// Grid: (seq, k_heads), block: kdim threads (kdim <= QWEN35_KMAX).
__global__ void qwen35_delta_norm_prepass_kernel(
    const __nv_bfloat16* qkv, __nv_bfloat16* qk_out, int seq, int k_heads,
    int v_heads, int kdim, int vdim) {
  const int s = blockIdx.x;
  const int k_head = blockIdx.y;
  const int lane = threadIdx.x;
  const int kdim_total = k_heads * kdim;
  const int row_stride = 2 * kdim_total + v_heads * vdim;
  __shared__ float s_sq[2 * QWEN35_KMAX];  // [2][kdim] squared sums
  float qv = 0.0f, kv = 0.0f;
  if (lane < kdim) {
    qv = __bfloat162float(qkv[s * row_stride + k_head * kdim + lane]);
    kv = __bfloat162float(
        qkv[s * row_stride + kdim_total + k_head * kdim + lane]);
  }
  s_sq[lane] = qv * qv;
  s_sq[QWEN35_KMAX + lane] = kv * kv;
  __syncthreads();
  for (int off = kdim / 2; off > 0; off >>= 1) {
    if (lane < off) {
      s_sq[lane] += s_sq[lane + off];
      s_sq[QWEN35_KMAX + lane] += s_sq[QWEN35_KMAX + lane + off];
    }
    __syncthreads();
  }
  const float q_inv = rsqrtf(s_sq[0] + 1e-6f);
  const float k_inv = rsqrtf(s_sq[QWEN35_KMAX] + 1e-6f);
  const int qk_stride = k_heads * 2 * kdim;
  if (lane < kdim) {
    qk_out[s * qk_stride + k_head * 2 * kdim + lane] =
        __float2bfloat16(qv * q_inv);
    qk_out[s * qk_stride + k_head * 2 * kdim + kdim + lane] =
        __float2bfloat16(kv * k_inv);
  }
}

// ── Linear attention: gated delta-rule recurrence ──────────────────────────
//
// qkv: [seq, 2*k_heads*kdim + v_heads*vdim] bf16 (post-conv; q/k entries
//      unused here — the normalized qk_norm rows are consumed instead)
// qk_norm: [seq, k_heads, 2, kdim] bf16 (from the prepass)
// a/b: [seq, v_heads] bf16; a_log/dt_bias: [v_heads] bf16
// recurrent: [v_heads, kdim, vdim] f32 (in/out)
// out: [seq, v_heads*vdim] bf16
//
// Grid: (vdim / QWEN35_V_TILE, v_heads) blocks of QWEN35_V_TILE threads.
// The state tile ([kdim, QWEN35_V_TILE] f32) lives in shared memory for the
// whole sequence sweep, so HBM state traffic is one read + one write.
__global__ void qwen35_delta_step_kernel(
    const __nv_bfloat16* qkv, const __nv_bfloat16* qk_norm,
    const __nv_bfloat16* a, const __nv_bfloat16* b,
    const __nv_bfloat16* a_log, const __nv_bfloat16* dt_bias,
    float* recurrent, __nv_bfloat16* out, int seq, int k_heads, int v_heads,
    int kdim, int vdim) {
  const int v_head = blockIdx.y;
  const int v_tile = blockIdx.x;
  const int lane = threadIdx.x;  // 0..QWEN35_V_TILE-1
  const int vd = v_tile * QWEN35_V_TILE + lane;
  const int repeat = v_heads / k_heads;
  const int k_head = v_head / repeat;
  const int kdim_total = k_heads * kdim;
  const int row_stride = 2 * kdim_total + v_heads * vdim;
  const int qk_stride = k_heads * 2 * kdim;

  extern __shared__ float s_delta_sh[];
  float* s_state = s_delta_sh;                      // [kdim][QWEN35_V_TILE]
  float* s_qk = s_delta_sh + kdim * QWEN35_V_TILE;  // [2][kdim]
  float* s_q = s_qk;
  float* s_k = s_qk + kdim;

  // Load the state tile into shared memory once.
  const int state_base = v_head * kdim * vdim + vd;
#pragma unroll 4
  for (int kd = 0; kd < kdim; kd++)
    s_state[kd * QWEN35_V_TILE + lane] =
        __ldg(&recurrent[state_base + kd * vdim]);

  const float a_log_h = __bfloat162float(a_log[v_head]);
  const float dt_bias_h = __bfloat162float(dt_bias[v_head]);
  const float q_scale = 1.0f / sqrtf((float)kdim);

  for (int s = 0; s < seq; s++) {
    const float a_h = __bfloat162float(__ldg(&a[s * v_heads + v_head]));
    const float b_h = __bfloat162float(__ldg(&b[s * v_heads + v_head]));
    const float decay = expf(-expf(a_log_h) * softplusf_f32(a_h + dt_bias_h));
    const float beta = sigmoidf_f32(b_h);
    const float v = __bfloat162float(
        __ldg(&qkv[s * row_stride + 2 * kdim_total + v_head * vdim + vd]));

    // Broadcast the normalized q/k row for this head into shared.
    s_q[lane] = __bfloat162float(
        __ldg(&qk_norm[s * qk_stride + k_head * 2 * kdim + lane]));
    s_k[lane] = __bfloat162float(
        __ldg(&qk_norm[s * qk_stride + k_head * 2 * kdim + kdim + lane]));
    __syncthreads();  // s_q/s_k visible to every lane

    // Decay the state tile.
#pragma unroll 4
    for (int kd = 0; kd < kdim; kd++)
      s_state[kd * QWEN35_V_TILE + lane] *= decay;

    // kv_mem[vd] = Σ_kd state[kd][vd] · k[kd]
    float mem = 0.0f;
#pragma unroll 4
    for (int kd = 0; kd < kdim; kd++)
      mem += s_state[kd * QWEN35_V_TILE + lane] * s_k[kd];
    const float delta = (v - mem) * beta;

    // state[kd][vd] += k[kd] · delta
#pragma unroll 4
    for (int kd = 0; kd < kdim; kd++)
      s_state[kd * QWEN35_V_TILE + lane] += s_k[kd] * delta;

    // out[vd] = Σ_kd state[kd][vd] · q[kd] · scale
    float acc = 0.0f;
#pragma unroll 4
    for (int kd = 0; kd < kdim; kd++)
      acc += s_state[kd * QWEN35_V_TILE + lane] * s_q[kd] * q_scale;
    out[s * v_heads * vdim + v_head * vdim + vd] = __float2bfloat16(acc);
    __syncthreads();  // s_q/s_k rewritten next iteration
  }

  // Write the final state tile back.
#pragma unroll 4
  for (int kd = 0; kd < kdim; kd++)
    recurrent[state_base + kd * vdim] = s_state[kd * QWEN35_V_TILE + lane];
}

// Prefill recurrence specialization for the exact K=V=128 geometry. A value
// coordinate has no dependency on any other value coordinate, while every
// coordinate must visit tokens in causal order. Splitting the 128-wide value
// row into four warp-sized tiles therefore increases the grid from v_heads to
// 4*v_heads blocks without changing state ordering. The q/k normalization is
// kept in the established parallel prepass, so repeated value heads do not
// recompute its reductions.
//
// Grid: (4, v_heads), block: 32 threads. State is [128, 32] f32 in ordinary
// shared memory (17 KiB including q/k), avoiding any opt-in shared-memory host
// operation on the prefill/capture path. The input and output bf16 boundaries
// are identical to qwen35_delta_step_kernel.
#define QWEN35_PREFILL_V_TILE 32
__global__ void qwen35_prefill_delta_step_kernel(
    const __nv_bfloat16* qkv, const __nv_bfloat16* qk_norm,
    const __nv_bfloat16* a, const __nv_bfloat16* b,
    const __nv_bfloat16* a_log, const __nv_bfloat16* dt_bias,
    float* recurrent, __nv_bfloat16* out, int seq, int k_heads,
    int v_heads) {
  constexpr int kdim = QWEN35_KMAX;
  constexpr int vdim = QWEN35_V_TILE;
  constexpr unsigned warp_mask = 0xffffffffu;
  const int v_head = blockIdx.y;
  const int v_tile = blockIdx.x;
  const int lane = threadIdx.x;
  const int vd = v_tile * QWEN35_PREFILL_V_TILE + lane;
  const int repeat = v_heads / k_heads;
  const int k_head = v_head / repeat;
  const int kdim_total = k_heads * kdim;
  const int row_stride = 2 * kdim_total + v_heads * vdim;
  const int qk_stride = k_heads * 2 * kdim;

  __shared__ float s_state[kdim * QWEN35_PREFILL_V_TILE];
  __shared__ float s_q[kdim];
  __shared__ float s_k[kdim];

  const int state_base = v_head * kdim * vdim + vd;
#pragma unroll 4
  for (int kd = 0; kd < kdim; kd++)
    s_state[kd * QWEN35_PREFILL_V_TILE + lane] =
        __ldg(&recurrent[state_base + kd * vdim]);

  const float a_log_h = __bfloat162float(a_log[v_head]);
  const float dt_bias_h = __bfloat162float(dt_bias[v_head]);
  const float q_scale = 1.0f / sqrtf((float)kdim);

  for (int s = 0; s < seq; s++) {
    const int qk_base = s * qk_stride + k_head * 2 * kdim;
#pragma unroll
    for (int kd = lane; kd < kdim; kd += QWEN35_PREFILL_V_TILE) {
      s_q[kd] = __bfloat162float(__ldg(&qk_norm[qk_base + kd]));
      s_k[kd] = __bfloat162float(__ldg(&qk_norm[qk_base + kdim + kd]));
    }
    __syncwarp(warp_mask);

    // These scalars are head-wide. Computing them once per tile removes 31
    // redundant exp/log/sigmoid evaluations while broadcasting the same f32
    // bits each lane computed in the established split recurrence.
    float decay = 0.0f;
    float beta = 0.0f;
    if (lane == 0) {
      const float a_h = __bfloat162float(__ldg(&a[s * v_heads + v_head]));
      const float b_h = __bfloat162float(__ldg(&b[s * v_heads + v_head]));
      decay = expf(
          -expf(a_log_h) * softplusf_f32(a_h + dt_bias_h));
      beta = sigmoidf_f32(b_h);
    }
    decay = __shfl_sync(warp_mask, decay, 0);
    beta = __shfl_sync(warp_mask, beta, 0);
    const float v = __bfloat162float(__ldg(
        &qkv[s * row_stride + 2 * kdim_total + v_head * vdim + vd]));

#pragma unroll 4
    for (int kd = 0; kd < kdim; kd++)
      s_state[kd * QWEN35_PREFILL_V_TILE + lane] *= decay;
    float mem = 0.0f;
#pragma unroll 4
    for (int kd = 0; kd < kdim; kd++)
      mem += s_state[kd * QWEN35_PREFILL_V_TILE + lane] * s_k[kd];
    const float delta = (v - mem) * beta;
#pragma unroll 4
    for (int kd = 0; kd < kdim; kd++)
      s_state[kd * QWEN35_PREFILL_V_TILE + lane] += s_k[kd] * delta;
    float acc = 0.0f;
#pragma unroll 4
    for (int kd = 0; kd < kdim; kd++)
      acc += s_state[kd * QWEN35_PREFILL_V_TILE + lane] * s_q[kd] * q_scale;
    out[(s * v_heads + v_head) * vdim + vd] = __float2bfloat16(acc);
    __syncwarp(warp_mask);
  }

#pragma unroll 4
  for (int kd = 0; kd < kdim; kd++)
    recurrent[state_base + kd * vdim] =
        s_state[kd * QWEN35_PREFILL_V_TILE + lane];
}

// Iteration-33 shared-work prefill recurrence. One 128-thread CTA owns all
// V=128 columns for one value head: warp i retains the exact recurrence for
// columns [32*i,32*i+31]. Q/K rows and scalar decay/beta are computed/loaded
// once per CTA and shared across the four warps. Token order and each column's
// K-ordered FP32 recurrence are identical to the four-CTA production path.
__global__ void qwen35_prefill_delta_step_4w_kernel(
    const __nv_bfloat16* qkv, const __nv_bfloat16* qk_norm,
    const __nv_bfloat16* a, const __nv_bfloat16* b,
    const __nv_bfloat16* a_log, const __nv_bfloat16* dt_bias,
    float* recurrent, __nv_bfloat16* out, int seq, int k_heads,
    int v_heads) {
  constexpr int kdim = QWEN35_KMAX, vdim = QWEN35_V_TILE;
  const int v_head = blockIdx.x;
  const int vd = threadIdx.x;
  const int lane = vd & 31;
  const int repeat = v_heads / k_heads;
  const int k_head = v_head / repeat;
  const int kdim_total = k_heads * kdim;
  const int row_stride = 2 * kdim_total + v_heads * vdim;
  const int qk_stride = k_heads * 2 * kdim;
  extern __shared__ float shared[];
  float* s_state = shared;
  float* s_q = s_state + kdim * vdim;
  float* s_k = s_q + kdim;
  float* s_decay = s_k + kdim;
  float* s_beta = s_decay + 1;
  const int state_base = v_head * kdim * vdim + vd;
#pragma unroll 4
  for (int kd = 0; kd < kdim; ++kd)
    s_state[kd * vdim + vd] = __ldg(&recurrent[state_base + kd * vdim]);
  const float a_log_h = __bfloat162float(a_log[v_head]);
  const float dt_bias_h = __bfloat162float(dt_bias[v_head]);
  const float q_scale = 1.0f / sqrtf((float)kdim);
  __syncthreads();
  for (int s = 0; s < seq; ++s) {
    const int qk_base = s * qk_stride + k_head * 2 * kdim;
    s_q[vd] = __bfloat162float(__ldg(&qk_norm[qk_base + vd]));
    s_k[vd] = __bfloat162float(__ldg(&qk_norm[qk_base + kdim + vd]));
    if (vd == 0) {
      const float a_h = __bfloat162float(__ldg(&a[s * v_heads + v_head]));
      const float b_h = __bfloat162float(__ldg(&b[s * v_heads + v_head]));
      *s_decay = expf(-expf(a_log_h) * softplusf_f32(a_h + dt_bias_h));
      *s_beta = sigmoidf_f32(b_h);
    }
    __syncthreads();
    const float decay = *s_decay;
    const float beta = *s_beta;
    const float v = __bfloat162float(__ldg(
        &qkv[s * row_stride + 2 * kdim_total + v_head * vdim + vd]));
#pragma unroll 4
    for (int kd = 0; kd < kdim; ++kd)
      s_state[kd * vdim + vd] *= decay;
    float mem = 0.0f;
#pragma unroll 4
    for (int kd = 0; kd < kdim; ++kd)
      mem += s_state[kd * vdim + vd] * s_k[kd];
    const float delta = (v - mem) * beta;
#pragma unroll 4
    for (int kd = 0; kd < kdim; ++kd)
      s_state[kd * vdim + vd] += s_k[kd] * delta;
    float acc = 0.0f;
#pragma unroll 4
    for (int kd = 0; kd < kdim; ++kd)
      acc += s_state[kd * vdim + vd] * s_q[kd] * q_scale;
    out[(s * v_heads + v_head) * vdim + vd] = __float2bfloat16(acc);
    __syncthreads();
  }
#pragma unroll 4
  for (int kd = 0; kd < kdim; ++kd)
    recurrent[state_base + kd * vdim] = s_state[kd * vdim + vd];
}

// Two-warp prefill recurrence alternative. Two CTAs per value head own
// disjoint 64-column state tiles; each tile preserves the historical
// 32-column recurrence order while halving the shared-state footprint.
__global__ void qwen35_prefill_delta_step_2w_kernel(
    const __nv_bfloat16* qkv, const __nv_bfloat16* qk_norm,
    const __nv_bfloat16* a, const __nv_bfloat16* b,
    const __nv_bfloat16* a_log, const __nv_bfloat16* dt_bias,
    float* recurrent, __nv_bfloat16* out, int seq, int k_heads,
    int v_heads) {
  constexpr int kdim = QWEN35_KMAX, vdim = QWEN35_V_TILE;
  constexpr int tile = 64;
  const int v_head = blockIdx.y;
  const int tile_id = blockIdx.x;
  const int vd_local = threadIdx.x;
  const int vd = tile_id * tile + vd_local;
  const int lane = vd_local & 31;
  const int repeat = v_heads / k_heads;
  const int k_head = v_head / repeat;
  const int kdim_total = k_heads * kdim;
  const int row_stride = 2 * kdim_total + v_heads * vdim;
  const int qk_stride = k_heads * 2 * kdim;
  extern __shared__ float shared[];
  float* s_state = shared;
  float* s_q = s_state + kdim * tile;
  float* s_k = s_q + kdim;
  float* s_decay = s_k + kdim;
  float* s_beta = s_decay + 1;
  const int state_base = v_head * kdim * vdim + vd;
#pragma unroll 4
  for (int kd = 0; kd < kdim; ++kd)
    s_state[kd * tile + vd_local] = __ldg(&recurrent[state_base + kd * vdim]);
  const float a_log_h = __bfloat162float(a_log[v_head]);
  const float dt_bias_h = __bfloat162float(dt_bias[v_head]);
  const float q_scale = 1.0f / sqrtf((float)kdim);
  __syncthreads();
  for (int s = 0; s < seq; ++s) {
    const int qk_base = s * qk_stride + k_head * 2 * kdim;
    for (int kd = vd_local; kd < kdim; kd += blockDim.x) {
      s_q[kd] = __bfloat162float(__ldg(&qk_norm[qk_base + kd]));
      s_k[kd] = __bfloat162float(__ldg(&qk_norm[qk_base + kdim + kd]));
    }
    if (vd_local == 0) {
      const float a_h = __bfloat162float(__ldg(&a[s * v_heads + v_head]));
      const float b_h = __bfloat162float(__ldg(&b[s * v_heads + v_head]));
      *s_decay = expf(-expf(a_log_h) * softplusf_f32(a_h + dt_bias_h));
      *s_beta = sigmoidf_f32(b_h);
    }
    __syncthreads();
    const float decay = *s_decay;
    const float beta = *s_beta;
    const float v = __bfloat162float(__ldg(
        &qkv[s * row_stride + 2 * kdim_total + v_head * vdim + vd]));
#pragma unroll 4
    for (int kd = 0; kd < kdim; ++kd)
      s_state[kd * tile + vd_local] *= decay;
    float mem = 0.0f;
#pragma unroll 4
    for (int kd = 0; kd < kdim; ++kd)
      mem += s_state[kd * tile + vd_local] * s_k[kd];
    const float delta = (v - mem) * beta;
#pragma unroll 4
    for (int kd = 0; kd < kdim; ++kd)
      s_state[kd * tile + vd_local] += s_k[kd] * delta;
    float acc = 0.0f;
#pragma unroll 4
    for (int kd = 0; kd < kdim; ++kd)
      acc += s_state[kd * tile + vd_local] * s_q[kd] * q_scale;
    out[(s * v_heads + v_head) * vdim + vd] = __float2bfloat16(acc);
    __syncthreads();
  }
#pragma unroll 4
  for (int kd = 0; kd < kdim; ++kd)
    recurrent[state_base + kd * vdim] = s_state[kd * tile + vd_local];
}


// Exact two-launch GDN tail for Qwen3.5's K=V=128 geometry. The causal
// convolution remains a separate launch because all channels must be
// materialized before the grouped recurrent blocks consume q/k/v. This
// kernel combines norm-prepass, recurrence, and gated RMSNorm while writing
// both eager workspaces and explicitly rounding at the same bf16 boundaries.
// Grid: (1, v_heads), block: 128 threads.
__global__ void qwen35_norm_delta_gated_kernel(
    const __nv_bfloat16* qkv, __nv_bfloat16* qk_out,
    const __nv_bfloat16* a, const __nv_bfloat16* b,
    const __nv_bfloat16* a_log, const __nv_bfloat16* dt_bias,
    const __nv_bfloat16* z, const __nv_bfloat16* norm_weight,
    float* recurrent, __nv_bfloat16* delta_out, __nv_bfloat16* out,
    int seq, int k_heads, int v_heads, float eps) {
  constexpr int kdim = QWEN35_KMAX;
  constexpr int vdim = QWEN35_V_TILE;
  const int v_head = blockIdx.y;
  const int lane = threadIdx.x;
  const int repeat = v_heads / k_heads;
  const int k_head = v_head / repeat;
  const int kdim_total = k_heads * kdim;
  const int row_stride = 2 * kdim_total + v_heads * vdim;
  const int qk_stride = k_heads * 2 * kdim;

  extern __shared__ float fused_shared[];
  float* s_state = fused_shared;                    // [kdim][vdim]
  float* s_q = s_state + kdim * vdim;               // [kdim]
  float* s_k = s_q + kdim;                          // [kdim]
  __nv_bfloat16* s_delta = reinterpret_cast<__nv_bfloat16*>(s_k + kdim);
  float* s_warp_sums = reinterpret_cast<float*>(s_delta + vdim);

  const int state_base = v_head * kdim * vdim + lane;
#pragma unroll 4
  for (int kd = 0; kd < kdim; kd++)
    s_state[kd * vdim + lane] =
        __ldg(&recurrent[state_base + kd * vdim]);

  const float a_log_h = __bfloat162float(a_log[v_head]);
  const float dt_bias_h = __bfloat162float(dt_bias[v_head]);
  const float q_scale = 1.0f / sqrtf((float)kdim);

  for (int s = 0; s < seq; s++) {
    const int row = s * row_stride;
    const float qv = __bfloat162float(
        __ldg(&qkv[row + k_head * kdim + lane]));
    const float kv = __bfloat162float(
        __ldg(&qkv[row + kdim_total + k_head * kdim + lane]));
    s_q[lane] = qv * qv;
    s_k[lane] = kv * kv;
    __syncthreads();
    // Match qwen35_delta_norm_prepass_kernel's tree and operation order.
    for (int off = kdim / 2; off > 0; off >>= 1) {
      if (lane < off) {
        s_q[lane] += s_q[lane + off];
        s_k[lane] += s_k[lane + off];
      }
      __syncthreads();
    }
    const float q_inv = rsqrtf(s_q[0] + 1e-6f);
    const float k_inv = rsqrtf(s_k[0] + 1e-6f);
    const __nv_bfloat16 q_norm = __float2bfloat16(qv * q_inv);
    const __nv_bfloat16 k_norm = __float2bfloat16(kv * k_inv);
    // One value-head block owns each q/k workspace row. Every repeated block
    // computes the identical bits locally, avoiding a cross-block dependency.
    if (v_head % repeat == 0) {
      const int qk_base = s * qk_stride + k_head * 2 * kdim;
      qk_out[qk_base + lane] = q_norm;
      qk_out[qk_base + kdim + lane] = k_norm;
    }
    // Explicitly cross the eager prepass bf16 boundary before recurrence.
    s_q[lane] = __bfloat162float(q_norm);
    s_k[lane] = __bfloat162float(k_norm);
    __syncthreads();

    const float a_h = __bfloat162float(__ldg(&a[s * v_heads + v_head]));
    const float b_h = __bfloat162float(__ldg(&b[s * v_heads + v_head]));
    const float decay = expf(
        -expf(a_log_h) * softplusf_f32(a_h + dt_bias_h));
    const float beta = sigmoidf_f32(b_h);
    const float v = __bfloat162float(__ldg(
        &qkv[row + 2 * kdim_total + v_head * vdim + lane]));

#pragma unroll 4
    for (int kd = 0; kd < kdim; kd++)
      s_state[kd * vdim + lane] *= decay;
    float mem = 0.0f;
#pragma unroll 4
    for (int kd = 0; kd < kdim; kd++)
      mem += s_state[kd * vdim + lane] * s_k[kd];
    const float delta = (v - mem) * beta;
#pragma unroll 4
    for (int kd = 0; kd < kdim; kd++)
      s_state[kd * vdim + lane] += s_k[kd] * delta;
    float acc = 0.0f;
#pragma unroll 4
    for (int kd = 0; kd < kdim; kd++)
      acc += s_state[kd * vdim + lane] * s_q[kd] * q_scale;

    const int out_index = (s * v_heads + v_head) * vdim + lane;
    const __nv_bfloat16 delta_bf16 = __float2bfloat16(acc);
    delta_out[out_index] = delta_bf16;
    s_delta[lane] = delta_bf16;
    __syncthreads();

    // Match qwen35_gated_norm_kernel's warp reduction and warp-sum order,
    // rereading the explicitly rounded delta value.
    const float xv = __bfloat162float(s_delta[lane]);
    float partial = xv * xv;
    for (int off = 16; off > 0; off >>= 1)
      partial += __shfl_xor_sync(0xffffffff, partial, off);
    const int warp = lane / 32;
    if ((lane & 31) == 0) s_warp_sums[warp] = partial;
    __syncthreads();
    float total = 0.0f;
#pragma unroll
    for (int i = 0; i < vdim / 32; i++) total += s_warp_sums[i];
    const float inv = rsqrtf(total / (float)vdim + eps);
    const float zv = __bfloat162float(__ldg(&z[out_index]));
    const float w = __bfloat162float(__ldg(&norm_weight[lane]));
    out[out_index] =
        __float2bfloat16(xv * inv * w * siluf_f32(zv));
    __syncthreads();
  }

#pragma unroll 4
  for (int kd = 0; kd < kdim; kd++)
    recurrent[state_base + kd * vdim] = s_state[kd * vdim + lane];
}

// Packed Qwen decode/chunk path, mirroring FLA's packed recurrent decode:
// causal conv, q/k L2 normalization, gating scalars, recurrent update, and
// gated RMSNorm share one launch. The recurrent state remains
// [v_heads, kdim, vdim] f32 and each token is completed before the next token
// begins. Conv, normalized q/k, and recurrent output are explicitly rounded
// through bf16 at the same boundaries as the separate kernels.
//
// This specialization deliberately covers Qwen3.5 K=V=128, conv width 4.
// Grid: (1, v_heads), block: 128 threads.
__global__ void qwen35_packed_delta_gated_kernel(
    const __nv_bfloat16* raw_qkv, const __nv_bfloat16* conv_weight,
    float* conv_state, const __nv_bfloat16* a, const __nv_bfloat16* b,
    const __nv_bfloat16* a_log, const __nv_bfloat16* dt_bias,
    const __nv_bfloat16* z, const __nv_bfloat16* norm_weight,
    float* recurrent, __nv_bfloat16* out, int seq, int k_heads,
    int v_heads, float eps) {
  constexpr int kdim = QWEN35_KMAX;
  constexpr int vdim = QWEN35_V_TILE;
  constexpr int conv_kernel = 4;
  const int v_head = blockIdx.y;
  const int lane = threadIdx.x;
  const int repeat = v_heads / k_heads;
  const int k_head = v_head / repeat;
  const int kdim_total = k_heads * kdim;
  const int row_stride = 2 * kdim_total + v_heads * vdim;
  const int q_channel = k_head * kdim + lane;
  const int k_channel = kdim_total + q_channel;
  const int v_channel = 2 * kdim_total + v_head * vdim + lane;

  extern __shared__ float packed_shared[];
  float* s_state = packed_shared;                    // [kdim][vdim]
  float* s_q = s_state + kdim * vdim;                // [kdim]
  float* s_k = s_q + kdim;                           // [kdim]
  __nv_bfloat16* s_delta = reinterpret_cast<__nv_bfloat16*>(s_k + kdim);
  float* s_warp_sums = reinterpret_cast<float*>(s_delta + vdim);

  const int state_base = v_head * kdim * vdim + lane;
  // Four adjacent value lanes form an aligned float4 for every K row.
  if ((lane & 3) == 0) {
#pragma unroll 4
    for (int kd = 0; kd < kdim; kd++) {
      const float4 values = *reinterpret_cast<const float4*>(
          recurrent + state_base + kd * vdim);
      *reinterpret_cast<float4*>(s_state + kd * vdim + lane) = values;
    }
  }
  __syncthreads();

  float q_hist[conv_kernel - 1], k_hist[conv_kernel - 1];
  float v_hist[conv_kernel - 1];
  float q_w[conv_kernel], k_w[conv_kernel], v_w[conv_kernel];
#pragma unroll
  for (int tap = 0; tap < conv_kernel - 1; tap++) {
    q_hist[tap] = conv_state[tap * row_stride + q_channel];
    k_hist[tap] = conv_state[tap * row_stride + k_channel];
    v_hist[tap] = conv_state[tap * row_stride + v_channel];
  }
#pragma unroll
  for (int tap = 0; tap < conv_kernel; tap++) {
    q_w[tap] = __bfloat162float(conv_weight[q_channel * conv_kernel + tap]);
    k_w[tap] = __bfloat162float(conv_weight[k_channel * conv_kernel + tap]);
    v_w[tap] = __bfloat162float(conv_weight[v_channel * conv_kernel + tap]);
  }

  const float a_log_h = __bfloat162float(a_log[v_head]);
  const float dt_bias_h = __bfloat162float(dt_bias[v_head]);
  const float q_scale = 1.0f / sqrtf((float)kdim);

  for (int s = 0; s < seq; s++) {
    const int row = s * row_stride;
    const float q_live = __bfloat162float(__ldg(&raw_qkv[row + q_channel]));
    const float k_live = __bfloat162float(__ldg(&raw_qkv[row + k_channel]));
    const float v_live = __bfloat162float(__ldg(&raw_qkv[row + v_channel]));
    // Match qwen35_conv_silu_kernel's accumulation order exactly: start from
    // zero, consume history from oldest to newest, then add the live tap.
    float q_conv = 0.0f;
    float k_conv = 0.0f;
    float v_conv = 0.0f;
#pragma unroll
    for (int tap = 0; tap < conv_kernel - 1; tap++) {
      q_conv += q_hist[tap] * q_w[tap];
      k_conv += k_hist[tap] * k_w[tap];
      v_conv += v_hist[tap] * v_w[tap];
    }
    q_conv += q_live * q_w[conv_kernel - 1];
    k_conv += k_live * k_w[conv_kernel - 1];
    v_conv += v_live * v_w[conv_kernel - 1];
#pragma unroll
    for (int tap = 0; tap < conv_kernel - 2; tap++) {
      q_hist[tap] = q_hist[tap + 1];
      k_hist[tap] = k_hist[tap + 1];
      v_hist[tap] = v_hist[tap + 1];
    }
    q_hist[conv_kernel - 2] = q_live;
    k_hist[conv_kernel - 2] = k_live;
    v_hist[conv_kernel - 2] = v_live;
    // Preserve the old conv output bf16 materialization boundary.
    const float qv = __bfloat162float(__float2bfloat16(siluf_f32(q_conv)));
    const float kv = __bfloat162float(__float2bfloat16(siluf_f32(k_conv)));
    const float v = __bfloat162float(__float2bfloat16(siluf_f32(v_conv)));

    s_q[lane] = qv * qv;
    s_k[lane] = kv * kv;
    __syncthreads();
    for (int off = kdim / 2; off > 0; off >>= 1) {
      if (lane < off) {
        s_q[lane] += s_q[lane + off];
        s_k[lane] += s_k[lane + off];
      }
      __syncthreads();
    }
    const float q_inv = rsqrtf(s_q[0] + 1e-6f);
    const float k_inv = rsqrtf(s_k[0] + 1e-6f);
    // Preserve the old norm-prepass bf16 materialization boundary.
    s_q[lane] = __bfloat162float(__float2bfloat16(qv * q_inv));
    s_k[lane] = __bfloat162float(__float2bfloat16(kv * k_inv));
    __syncthreads();

    const float a_h = __bfloat162float(__ldg(&a[s * v_heads + v_head]));
    const float b_h = __bfloat162float(__ldg(&b[s * v_heads + v_head]));
    const float decay = expf(
        -expf(a_log_h) * softplusf_f32(a_h + dt_bias_h));
    const float beta = sigmoidf_f32(b_h);

#pragma unroll 4
    for (int kd = 0; kd < kdim; kd++)
      s_state[kd * vdim + lane] *= decay;
    float mem = 0.0f;
#pragma unroll 4
    for (int kd = 0; kd < kdim; kd++)
      mem += s_state[kd * vdim + lane] * s_k[kd];
    const float delta = (v - mem) * beta;
#pragma unroll 4
    for (int kd = 0; kd < kdim; kd++)
      s_state[kd * vdim + lane] += s_k[kd] * delta;
    float acc = 0.0f;
#pragma unroll 4
    for (int kd = 0; kd < kdim; kd++)
      acc += s_state[kd * vdim + lane] * s_q[kd] * q_scale;
    // Preserve the old delta-output bf16 materialization boundary.
    s_delta[lane] = __float2bfloat16(acc);
    __syncthreads();

    const float xv = __bfloat162float(s_delta[lane]);
    float partial = xv * xv;
    for (int off = 16; off > 0; off >>= 1)
      partial += __shfl_xor_sync(0xffffffff, partial, off);
    const int warp = lane / 32;
    if ((lane & 31) == 0) s_warp_sums[warp] = partial;
    __syncthreads();
    float total = 0.0f;
#pragma unroll
    for (int i = 0; i < vdim / 32; i++) total += s_warp_sums[i];
    const float inv = rsqrtf(total / (float)vdim + eps);
    const int out_index = (s * v_heads + v_head) * vdim + lane;
    const float zv = __bfloat162float(__ldg(&z[out_index]));
    const float w = __bfloat162float(__ldg(&norm_weight[lane]));
    out[out_index] =
        __float2bfloat16(xv * inv * w * siluf_f32(zv));
    __syncthreads();
  }

  if ((lane & 3) == 0) {
#pragma unroll 4
    for (int kd = 0; kd < kdim; kd++) {
      const float4 values = *reinterpret_cast<const float4*>(
          s_state + kd * vdim + lane);
      *reinterpret_cast<float4*>(recurrent + state_base + kd * vdim) = values;
    }
  }
#pragma unroll
  for (int tap = 0; tap < conv_kernel - 1; tap++) {
    // Q/K channels are duplicated by grouped value heads; one block owns the
    // identical final write, while each value head owns its disjoint V slice.
    if (v_head % repeat == 0) {
      conv_state[tap * row_stride + q_channel] = q_hist[tap];
      conv_state[tap * row_stride + k_channel] = k_hist[tap];
    }
    conv_state[tap * row_stride + v_channel] = v_hist[tap];
  }
}

// ── Linear attention: gated RMSNorm (norm before gate, SiLU gate) ──────────
//
// input: [seq, v_heads*vdim] bf16; z: same shape; weight: [vdim] bf16
// out[s, head, vd] = rms_norm(input_head) · weight · silu(z)
// One block per (s, head); dynamic shared = vdim floats.

__global__ void qwen35_gated_norm_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* z,
    const __nv_bfloat16* weight, __nv_bfloat16* out, int seq, int v_heads,
    int vdim, float eps) {
  const int s = blockIdx.x;
  const int head = blockIdx.y;
  const int tid = threadIdx.x;
  const int base = (s * v_heads + head) * vdim;
  extern __shared__ float x_buf[];


  float partial = 0.0f;
  for (int i = tid; i < vdim; i += blockDim.x) {
    const float xv = __bfloat162float(input[base + i]);
    x_buf[i] = xv;
    partial += xv * xv;
  }
  for (int off = 16; off > 0; off >>= 1)
    partial += __shfl_xor_sync(0xffffffff, partial, off);
  __shared__ float warp_sums[8];
  const int warp = tid / 32, lane = tid % 32;
  if (lane == 0) warp_sums[warp] = partial;
  __syncthreads();
  float total = 0.0f;
  for (int i = 0; i < (blockDim.x + 31) / 32; i++) total += warp_sums[i];
  const float inv = rsqrtf(total / (float)vdim + eps);
  for (int i = tid; i < vdim; i += blockDim.x) {
    const float zv = __bfloat162float(z[base + i]);
    const float w = __bfloat162float(weight[i]);
    out[base + i] = __float2bfloat16(x_buf[i] * inv * w * siluf_f32(zv));
  }
}

// ── Full attention: q/gate split + q RMSNorm + partial RoPE ────────────────
//
// q_gate: [seq, heads, 2*head_dim] bf16 (q then gate per head chunk)
// q_norm_w: [head_dim] bf16, already (1 + weight)
// q_out: [seq, heads, head_dim] bf16; gate_out: [seq, heads*head_dim] bf16
// Rotates only the first `rotary_dim` elements of each head.
// One block per (s, head); blockDim.x = head_dim.

__global__ void qwen35_q_split_norm_rope_kernel(
    const __nv_bfloat16* q_gate, const __nv_bfloat16* q_norm_w,
    __nv_bfloat16* q_out, __nv_bfloat16* gate_out, int seq, int heads,
    int head_dim, int rotary_dim, float theta, uint32_t start_pos) {
  const int s = blockIdx.x;
  const int head = blockIdx.y;
  const int tid = threadIdx.x;
  if (tid >= head_dim) return;

  const int src = (s * heads + head) * 2 * head_dim;
  const float qv = __bfloat162float(q_gate[src + tid]);
  const float gv = __bfloat162float(q_gate[src + head_dim + tid]);
  gate_out[(s * heads + head) * head_dim + tid] = __float2bfloat16(gv);

  // RMS norm over the head.
  float partial = qv * qv;
  for (int off = 16; off > 0; off >>= 1)
    partial += __shfl_xor_sync(0xffffffff, partial, off);
  __shared__ float warp_sums[8];
  const int warp = tid / 32, lane = tid % 32;
  if (lane == 0) warp_sums[warp] = partial;
  __syncthreads();
  float total = 0.0f;
  for (int i = 0; i < (head_dim + 31) / 32; i++) total += warp_sums[i];
  const float inv = rsqrtf(total / (float)head_dim + 1e-6f);
  const float x = qv * inv * __bfloat162float(q_norm_w[tid]);

  const int dst = (s * heads + head) * head_dim;
  if (tid < rotary_dim) {
    const int half = rotary_dim / 2;
    const int i = tid < half ? tid : tid - half;
    const float freq = 1.0f / powf(theta, 2.0f * (float)i / (float)rotary_dim);
    const float angle = (float)(start_pos + (uint32_t)s) * freq;
    const float cs = cosf(angle), sn = sinf(angle);
    if (tid < half) {
      const float x2v = __bfloat162float(q_gate[src + tid + half]);
      const float x2 = x2v * inv * __bfloat162float(q_norm_w[tid + half]);
      q_out[dst + tid] = __float2bfloat16(x * cs - x2 * sn);
    } else {
      const float x1v = __bfloat162float(q_gate[src + tid - half]);
      const float x1 = x1v * inv * __bfloat162float(q_norm_w[tid - half]);
      q_out[dst + tid] = __float2bfloat16(x1 * sn + x * cs);
    }
  } else {
    q_out[dst + tid] = __float2bfloat16(x);
  }
}

// ── Full attention: k RMSNorm + partial RoPE + cache append ────────────────
//
// k_in: [seq, n_kv_heads, head_dim] bf16; k_norm_w: [head_dim] (1+w) bf16
// k_cache: [n_kv_heads, max_seq_len, head_dim] bf16
// Writes the rope'd k at positions start_pos + s.
// One block per (s, head); blockDim.x = head_dim.

__global__ void qwen35_k_norm_rope_append_kernel(
    const __nv_bfloat16* k_in, const __nv_bfloat16* k_norm_w,
    __nv_bfloat16* k_cache, int seq, int n_kv_heads, int head_dim,
    int rotary_dim, float theta, uint32_t start_pos, int max_seq_len) {
  const int s = blockIdx.x;
  const int head = blockIdx.y;
  const int tid = threadIdx.x;
  if (tid >= head_dim) return;

  const int src = (s * n_kv_heads + head) * head_dim;
  const float kv = __bfloat162float(k_in[src + tid]);

  float partial = kv * kv;
  for (int off = 16; off > 0; off >>= 1)
    partial += __shfl_xor_sync(0xffffffff, partial, off);
  __shared__ float warp_sums[8];
  const int warp = tid / 32, lane = tid % 32;
  if (lane == 0) warp_sums[warp] = partial;
  __syncthreads();
  float total = 0.0f;
  for (int i = 0; i < (head_dim + 31) / 32; i++) total += warp_sums[i];
  const float inv = rsqrtf(total / (float)head_dim + 1e-6f);
  const float x = kv * inv * __bfloat162float(k_norm_w[tid]);

  const int pos = (int)(start_pos + (uint32_t)s);
  const int dst = head * max_seq_len * head_dim + pos * head_dim;
  if (tid < rotary_dim) {
    const int half = rotary_dim / 2;
    const int i = tid < half ? tid : tid - half;
    const float freq = 1.0f / powf(theta, 2.0f * (float)i / (float)rotary_dim);
    const float angle = (float)pos * freq;
    const float cs = cosf(angle), sn = sinf(angle);
    if (tid < half) {
      const float x2v = __bfloat162float(k_in[src + tid + half]);
      const float x2 = x2v * inv * __bfloat162float(k_norm_w[tid + half]);
      k_cache[dst + tid] = __float2bfloat16(x * cs - x2 * sn);
    } else {
      const float x1v = __bfloat162float(k_in[src + tid - half]);
      const float x1 = x1v * inv * __bfloat162float(k_norm_w[tid - half]);
      k_cache[dst + tid] = __float2bfloat16(x1 * sn + x * cs);
    }
  } else {
    k_cache[dst + tid] = __float2bfloat16(x);
  }
}
// Decode-only fusion of q split/norm/RoPE and k norm/RoPE/cache append.
// The two branches retain the original per-head reduction and bf16 rounding
// order; the single launch only removes launch latency and does not alter
// cache append positions or the q/k arithmetic.
__global__ void qwen35_qk_norm_rope_append_kernel(
    const __nv_bfloat16* q_gate, const __nv_bfloat16* q_norm_w,
    const __nv_bfloat16* k_in, const __nv_bfloat16* k_norm_w,
    __nv_bfloat16* q_out, __nv_bfloat16* gate_out,
    __nv_bfloat16* k_cache, int seq, int heads, int n_kv_heads,
    int head_dim, int rotary_dim, float theta, const uint32_t* position,
    int max_seq_len) {
  const uint32_t start_pos = *position;
  const int s = blockIdx.x;
  const int group = blockIdx.y;
  const int tid = threadIdx.x;
  if (tid >= head_dim) return;
  __shared__ float warp_sums[32];

  if (group < heads) {
    const int head = group;
    const int src = (s * heads + head) * 2 * head_dim;
    const float qv = __bfloat162float(q_gate[src + tid]);
    const float gv = __bfloat162float(q_gate[src + head_dim + tid]);
    gate_out[(s * heads + head) * head_dim + tid] = __float2bfloat16(gv);
    float partial = qv * qv;
    for (int off = 16; off > 0; off >>= 1)
      partial += __shfl_xor_sync(0xffffffff, partial, off);
    const int warp = tid / 32, lane = tid % 32;
    if (lane == 0) warp_sums[warp] = partial;
    __syncthreads();
    float total = 0.0f;
    for (int i = 0; i < (head_dim + 31) / 32; i++) total += warp_sums[i];
    const float inv = rsqrtf(total / (float)head_dim + 1e-6f);
    const float x = qv * inv * __bfloat162float(q_norm_w[tid]);
    const int dst = (s * heads + head) * head_dim;
    if (tid < rotary_dim) {
      const int half = rotary_dim / 2;
      const int i = tid < half ? tid : tid - half;
      const float freq = 1.0f / powf(theta, 2.0f * (float)i / (float)rotary_dim);
      const float angle = (float)(start_pos + (uint32_t)s) * freq;
      const float cs = cosf(angle), sn = sinf(angle);
      if (tid < half) {
        const float x2v = __bfloat162float(q_gate[src + tid + half]);
        const float x2 = x2v * inv * __bfloat162float(q_norm_w[tid + half]);
        q_out[dst + tid] = __float2bfloat16(x * cs - x2 * sn);
      } else {
        const float x1v = __bfloat162float(q_gate[src + tid - half]);
        const float x1 = x1v * inv * __bfloat162float(q_norm_w[tid - half]);
        q_out[dst + tid] = __float2bfloat16(x1 * sn + x * cs);
      }
    } else {
      q_out[dst + tid] = __float2bfloat16(x);
    }
  } else {
    const int head = group - heads;
    const int src = (s * n_kv_heads + head) * head_dim;
    const float kv = __bfloat162float(k_in[src + tid]);
    float partial = kv * kv;
    for (int off = 16; off > 0; off >>= 1)
      partial += __shfl_xor_sync(0xffffffff, partial, off);
    const int warp = tid / 32, lane = tid % 32;
    if (lane == 0) warp_sums[warp] = partial;
    __syncthreads();
    float total = 0.0f;
    for (int i = 0; i < (head_dim + 31) / 32; i++) total += warp_sums[i];
    const float inv = rsqrtf(total / (float)head_dim + 1e-6f);
    const float x = kv * inv * __bfloat162float(k_norm_w[tid]);
    const int pos = (int)(start_pos + (uint32_t)s);
    const int dst = head * max_seq_len * head_dim + pos * head_dim;
    if (tid < rotary_dim) {
      const int half = rotary_dim / 2;
      const int i = tid < half ? tid : tid - half;
      const float freq = 1.0f / powf(theta, 2.0f * (float)i / (float)rotary_dim);
      const float angle = (float)pos * freq;
      const float cs = cosf(angle), sn = sinf(angle);
      if (tid < half) {
        const float x2v = __bfloat162float(k_in[src + tid + half]);
        const float x2 = x2v * inv * __bfloat162float(k_norm_w[tid + half]);
        k_cache[dst + tid] = __float2bfloat16(x * cs - x2 * sn);
      } else {
        const float x1v = __bfloat162float(k_in[src + tid - half]);
        const float x1 = x1v * inv * __bfloat162float(k_norm_w[tid - half]);
        k_cache[dst + tid] = __float2bfloat16(x1 * sn + x * cs);
      }
    } else {
      k_cache[dst + tid] = __float2bfloat16(x);
    }
  }
}

// ── Row-tiled W4A16 dequant (prefill path) ────────────────────────────────
//
// One block per requested output row; no per-element integer division. Each
// thread unpacks whole int32 words (8 nibbles) and stores 8 bf16 values
// vectorized. `dense` is compact `[row_count, in_cols]`, allowing the caller
// to dequantize a cache-sized output-column tile immediately before GEMM.

__global__ void qwen35_dequant_w4a16_bf16_rows_kernel(
    const int32_t* weight_packed, const __nv_bfloat16* weight_scale,
    const int32_t* weight_zero_point, __nv_bfloat16* dense, int in_cols,
    int out_cols, int groups, int row_start, int row_count) {
  const int local_row = blockIdx.x;
  if (local_row >= row_count) return;
  const int row = row_start + local_row;
  if (row >= out_cols) return;
  const int packed_cols = in_cols / 8;
  const int group_size = in_cols / groups;
  const int zp_row = row / 8;
  const int zp_shift = (row & 7) * 4;
  const int64_t row_base = static_cast<int64_t>(row) * packed_cols;
  const int64_t out_base = static_cast<int64_t>(local_row) * in_cols;
  for (int w = threadIdx.x; w < packed_cols; w += blockDim.x) {
    const uint32_t word =
        static_cast<uint32_t>(weight_packed[row_base + w]);
    const int group = w * 8 / group_size;
    const uint32_t zp_word = static_cast<uint32_t>(
        weight_zero_point[static_cast<int64_t>(zp_row) * groups + group]);
    const int zp = static_cast<int>((zp_word >> zp_shift) & 0xFU);
    const float scale = __bfloat162float(
        weight_scale[static_cast<int64_t>(row) * groups + group]);
    __nv_bfloat16 values[8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const int q = static_cast<int>((word >> (j * 4)) & 0xFU);
      values[j] = __float2bfloat16(static_cast<float>(q - zp) * scale);
    }
    *reinterpret_cast<float4*>(dense + out_base + w * 8) =
        *reinterpret_cast<const float4*>(values);
  }
}

// Co-launches two independent raw-layout dequantizations into disjoint dense
// buffers. Each block executes the same row mapping and BF16 conversion as the
// single-projection prefill path.
__global__ void qwen35_dequant_w4a16_bf16_pair_rows_kernel(
    const int32_t* weight_packed0, const __nv_bfloat16* weight_scale0,
    const int32_t* weight_zero_point0, __nv_bfloat16* dense0, int out_cols0,
    const int32_t* weight_packed1, const __nv_bfloat16* weight_scale1,
    const int32_t* weight_zero_point1, __nv_bfloat16* dense1, int out_cols1,
    int in_cols, int groups) {
  int local_row = blockIdx.x;
  const int32_t* weight_packed = weight_packed0;
  const __nv_bfloat16* weight_scale = weight_scale0;
  const int32_t* weight_zero_point = weight_zero_point0;
  __nv_bfloat16* dense = dense0;
  if (local_row >= out_cols0) {
    local_row -= out_cols0;
    if (local_row >= out_cols1) return;
    weight_packed = weight_packed1;
    weight_scale = weight_scale1;
    weight_zero_point = weight_zero_point1;
    dense = dense1;
  }
  const int packed_cols = in_cols / 8;
  const int group_size = in_cols / groups;
  const int zp_row = local_row / 8;
  const int zp_shift = (local_row & 7) * 4;
  const int64_t row_base = static_cast<int64_t>(local_row) * packed_cols;
  const int64_t out_base = static_cast<int64_t>(local_row) * in_cols;
  for (int w = threadIdx.x; w < packed_cols; w += blockDim.x) {
    const uint32_t word = static_cast<uint32_t>(weight_packed[row_base + w]);
    const int group = w * 8 / group_size;
    const uint32_t zp_word = static_cast<uint32_t>(
        weight_zero_point[static_cast<int64_t>(zp_row) * groups + group]);
    const int zp = static_cast<int>((zp_word >> zp_shift) & 0xFU);
    const float scale = __bfloat162float(
        weight_scale[static_cast<int64_t>(local_row) * groups + group]);
    __nv_bfloat16 values[8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const int q = static_cast<int>((word >> (j * 4)) & 0xFU);
      values[j] = __float2bfloat16(static_cast<float>(q - zp) * scale);
    }
    *reinterpret_cast<float4*>(dense + out_base + w * 8) =
        *reinterpret_cast<const float4*>(values);
  }
}

// ── Tiled fused W4A16 dequant-GEMM (decode path) ───────────────────────────
//
// Block computes 128 output elements. The packed weight tile is loaded
// cooperatively (coalesced along the row-major word axis) and unpacked into
// shared memory, so each packed word is read from HBM exactly once.

#define QWEN35_GEMM_OUT_TILE 128
#define QWEN35_GEMM_IN_TILE 256

__global__ void qwen35_gemm_w4a16_bf16_kernel(
    const __nv_bfloat16* activation, const int32_t* weight_packed,
    const __nv_bfloat16* weight_scale, const int32_t* weight_zero_point,
    __nv_bfloat16* output, int in_cols, int out_cols, int groups) {
  const int out_base = blockIdx.x * QWEN35_GEMM_OUT_TILE;
  const int tid = threadIdx.x;
  const int group_size = (in_cols + groups - 1) / groups;
  const int packed_cols = (in_cols + 7) / 8;

  extern __shared__ uint8_t sh[];
  uint8_t* s_w = sh;                    // [128][TILE] nibbles
  float* s_x = reinterpret_cast<float*>(
      sh + QWEN35_GEMM_OUT_TILE * QWEN35_GEMM_IN_TILE);  // [TILE]

  float acc = 0.0f;
  const int out = out_base + tid;
  const int zp_row = out / 8;
  const int zp_shift = (out & 7) * 4;

  for (int tile = 0; tile < in_cols; tile += QWEN35_GEMM_IN_TILE) {
    const int tile_cols = min(QWEN35_GEMM_IN_TILE, in_cols - tile);
    const int words = (tile_cols + 7) / 8;
    // Cooperative coalesced load of the [128 x words] packed tile.
    const int total_words = QWEN35_GEMM_OUT_TILE * words;
    for (int idx = tid; idx < total_words; idx += blockDim.x) {
      const int r = idx / words;
      const int w = idx - r * words;
      const uint32_t word = static_cast<uint32_t>(weight_packed[
          static_cast<int64_t>(out_base + r) * packed_cols + tile / 8 + w]);
      uint8_t* dst = s_w + r * QWEN35_GEMM_IN_TILE + w * 8;
      const uint32_t lo = word & 0xFFFFu;
      const uint32_t hi = word >> 16;
      *reinterpret_cast<uint32_t*>(dst) = lo & 0x0F0F0F0Fu |
          ((lo >> 4) & 0x0F0F0F0Fu) << 4;
      // simpler: 4-bit nibbles need compaction; store one nibble per byte.
      dst[0] = lo & 0xFu; dst[1] = (lo >> 4) & 0xFu;
      dst[2] = (lo >> 8) & 0xFu; dst[3] = (lo >> 12) & 0xFu;
      dst[4] = hi & 0xFu; dst[5] = (hi >> 4) & 0xFu;
      dst[6] = (hi >> 8) & 0xFu; dst[7] = (hi >> 12) & 0xFu;
    }
    // Load the activation tile into shared.
    for (int i = tid; i < tile_cols; i += blockDim.x)
      s_x[i] = __bfloat162float(activation[tile + i]);
    __syncthreads();

    // Per-thread: its output row's tile, dequantized with scale/zp per group.
    const uint8_t* my_row = s_w + tid * QWEN35_GEMM_IN_TILE;
    for (int c = 0; c < tile_cols; c += 8) {
      const int group = (tile + c) / group_size;
      const uint32_t zp_word = static_cast<uint32_t>(
          weight_zero_point[static_cast<int64_t>(zp_row) * groups + group]);
      const int zp = static_cast<int>((zp_word >> zp_shift) & 0xFU);
      const float scale = __bfloat162float(
          weight_scale[static_cast<int64_t>(out) * groups + group]);
#pragma unroll
      for (int j = 0; j < 8; j++) {
        const int q = static_cast<int>(my_row[c + j]);
        acc += s_x[c + j] * static_cast<float>(q - zp) * scale;
      }
    }
    __syncthreads();
  }
  output[out_base + tid] = __float2bfloat16(acc);
}


// ── Decode GEMM on tensor cores ────────────────────────────────────────────
// [1, in_cols] x W4A16 (group-32, asymmetric) -> [1, out_cols] via
// m16n8k16 bf16 MMAs. The single activation row sits in A's row 0 (the
// rest are zero), the weight tile is dequantized to bf16 in shared per
// k-tile. Each 8-warp block computes QWEN35_TC_OUT_TILE = 64 outputs.
// For the model's 128-aligned K shapes, activation fragments can be consumed
// one MMA at a time instead of keeping sixteen packed values live per lane.
// The MMA/dequant order is unchanged; the generic schedule remains the exact
// fallback for other accepted shapes.

#define QWEN35_TC_OUT_TILE 64
#define QWEN35_TC_ALT_OUT_TILE 32
#define QWEN35_TC_PAIR_2W_OUT_TILE 16
#define QWEN35_TC_PAIR_6W_OUT_TILE 48

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
__device__ __forceinline__ uint32_t qwen35_pack_bf16(
    __nv_bfloat16 lo, __nv_bfloat16 hi) {
  return (static_cast<uint32_t>(__nv_bfloat16_raw(lo).x)) |
         (static_cast<uint32_t>(__nv_bfloat16_raw(hi).x) << 16);
}

__device__ __forceinline__ void qwen35_mma_bf16(
    float& c0, float& c1, float& c2, float& c3, uint32_t a0, uint32_t a1,
    uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template <bool CacheHint, typename T>
__device__ __forceinline__ T qwen35_w4_readonly_load(const T* address) {
  if constexpr (CacheHint) {
    return __ldg(address);
  }
  return *address;
}

template <bool Pair, bool CacheHint = false>
__global__ void qwen35_gemm_w4a16_bf16_tc_kernel(
    const __nv_bfloat16* activation, const int32_t* weight_packed0,
    const __nv_bfloat16* weight_scale0, const int32_t* weight_zero_point0,
    __nv_bfloat16* output0, int out_cols0, const int32_t* weight_packed1,
    const __nv_bfloat16* weight_scale1, const int32_t* weight_zero_point1,
    __nv_bfloat16* output1, int in_cols, int groups) {
  int output_block = blockIdx.x;
  const int first_blocks = out_cols0 / QWEN35_TC_OUT_TILE;
  const int32_t* weight_packed = weight_packed0;
  const __nv_bfloat16* weight_scale = weight_scale0;
  const int32_t* weight_zero_point = weight_zero_point0;
  __nv_bfloat16* output = output0;
  if (Pair && output_block >= first_blocks) {
    output_block -= first_blocks;
    weight_packed = weight_packed1;
    weight_scale = weight_scale1;
    weight_zero_point = weight_zero_point1;
    output = output1;
  }
  const int out_base = output_block * QWEN35_TC_OUT_TILE;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int group_size = (in_cols + groups - 1) / groups;
  const int packed_cols = (in_cols + 7) / 8;
  __shared__ __nv_bfloat16 s_act[128];
  const int a_col = 2 * (lane % 4);
  // This lane's B-fragment positions: (k, n) = (2l', j), (2l'+1, j),
  // (2l'+8, j), (2l'+9, j) with l' = lane%4, j = lane/4 (the n index).
  const int l4 = lane % 4;
  const int jn = lane / 4;            // the output row within the tile
  const int out_row = out_base + warp * 8 + jn;
  const int k0 = 2 * l4;              // the first k pair
  const int k1 = k0 + 8;              // the second k pair
  float c[16] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  for (int tile128 = 0; tile128 < in_cols; tile128 += 128) {
    for (int idx = threadIdx.x; idx < 128; idx += 256)
      s_act[idx] = activation[tile128 + idx];
    __syncthreads();
    uint32_t a0[8], a2[8];
#pragma unroll
    for (int sub = 0; sub < 8; sub++) {
      a0[sub] = 0;
      a2[sub] = 0;
      if (lane / 4 == 0) {
        a0[sub] = qwen35_pack_bf16(s_act[sub * 16 + a_col],
                                   s_act[sub * 16 + a_col + 1]);
        a2[sub] = qwen35_pack_bf16(s_act[sub * 16 + a_col + 8],
                                   s_act[sub * 16 + a_col + 9]);
      }
    }
#pragma unroll
    for (int sub = 0; sub < 8; sub++) {
      const int group = (tile128 + sub * 16) / group_size;
      const int base_col = tile128 + sub * 16;
      const uint32_t zp_word = static_cast<uint32_t>(qwen35_w4_readonly_load<CacheHint>(
          &weight_zero_point[static_cast<int64_t>(out_row / 8) * groups + group]));
      const int zp =
          static_cast<int>((zp_word >> ((out_row & 7) * 4)) & 0xFu);
      const float scale = __bfloat162float(qwen35_w4_readonly_load<CacheHint>(
          &weight_scale[static_cast<int64_t>(out_row) * groups + group]));
      const uint32_t w0 = static_cast<uint32_t>(qwen35_w4_readonly_load<CacheHint>(
          &weight_packed[static_cast<int64_t>(out_row) * packed_cols +
                         (base_col + k0) / 8]));
      const uint32_t w1 = static_cast<uint32_t>(qwen35_w4_readonly_load<CacheHint>(
          &weight_packed[static_cast<int64_t>(out_row) * packed_cols +
                         (base_col + k1) / 8]));
      const int q00 = static_cast<int>((w0 >> ((k0 & 7) * 4)) & 0xFu);
      const int q01 =
          static_cast<int>((w0 >> (((k0 + 1) & 7) * 4)) & 0xFu);
      const int q10 = static_cast<int>((w1 >> ((k1 & 7) * 4)) & 0xFu);
      const int q11 =
          static_cast<int>((w1 >> (((k1 + 1) & 7) * 4)) & 0xFu);
      const uint32_t b0 = qwen35_pack_bf16(
          __float2bfloat16(static_cast<float>(q00 - zp) * scale),
          __float2bfloat16(static_cast<float>(q01 - zp) * scale));
      const uint32_t b1 = qwen35_pack_bf16(
          __float2bfloat16(static_cast<float>(q10 - zp) * scale),
          __float2bfloat16(static_cast<float>(q11 - zp) * scale));
      qwen35_mma_bf16(c[(sub % 4) * 4], c[(sub % 4) * 4 + 1],
                      c[(sub % 4) * 4 + 2], c[(sub % 4) * 4 + 3],
                      a0[sub], 0, a2[sub], 0, b0, b1);
    }
    __syncthreads();
  }
  if (lane < 4) {
    const int col = 2 * lane;
    const float r0 = c[0] + c[4] + c[8] + c[12];
    const float r1 = c[1] + c[5] + c[9] + c[13];
    output[out_base + warp * 8 + col] = __float2bfloat16(r0);
    output[out_base + warp * 8 + col + 1] = __float2bfloat16(r1);
  }
}

// APXINF_W4_SCALE_EPILOGUE candidate. Scale and packed zero-point application
// are fused into the BF16 fragment transform. Each transformed weight remains
// exactly bf16((q-zp)*scale) before the existing ordered MMA sequence; no
// integer accumulation or reordered reduction is introduced.
__device__ __forceinline__ uint32_t qwen35_w4_scale_epilogue_pair(
    uint32_t packed_word, int shift, int zero_point, float scale) {
  const int q0 = static_cast<int>((packed_word >> (shift * 4)) & 0xFu);
  const int q1 = static_cast<int>((packed_word >> ((shift + 1) * 4)) & 0xFu);
  return qwen35_pack_bf16(
      __float2bfloat16(static_cast<float>(q0 - zero_point) * scale),
      __float2bfloat16(static_cast<float>(q1 - zero_point) * scale));
}

__global__ void qwen35_gemm_w4a16_bf16_tc_scale_epilogue_kernel(
    const __nv_bfloat16* activation, const int32_t* weight_packed,
    const __nv_bfloat16* weight_scale, const int32_t* weight_zero_point,
    __nv_bfloat16* output, int in_cols, int groups) {
  const int out_base = blockIdx.x * QWEN35_TC_OUT_TILE;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int group_size = in_cols / groups;
  const int packed_cols = (in_cols + 7) / 8;
  const int a_col = 2 * (lane % 4);
  const int l4 = lane % 4;
  const int jn = lane / 4;
  const int out_row = out_base + warp * 8 + jn;
  const int k0 = 2 * l4;
  const int k1 = k0 + 8;
  __shared__ __nv_bfloat16 s_act[128];
  float c[16] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  for (int tile128 = 0; tile128 < in_cols; tile128 += 128) {
    for (int idx = threadIdx.x; idx < 128; idx += 256)
      s_act[idx] = activation[tile128 + idx];
    __syncthreads();
    uint32_t a0[8], a2[8];
#pragma unroll
    for (int sub = 0; sub < 8; sub++) {
      a0[sub] = 0;
      a2[sub] = 0;
      if (lane / 4 == 0) {
        a0[sub] = qwen35_pack_bf16(s_act[sub * 16 + a_col],
                                   s_act[sub * 16 + a_col + 1]);
        a2[sub] = qwen35_pack_bf16(s_act[sub * 16 + a_col + 8],
                                   s_act[sub * 16 + a_col + 9]);
      }
    }
#pragma unroll
    for (int sub = 0; sub < 8; sub++) {
      const int group = (tile128 + sub * 16) / group_size;
      const int base_col = tile128 + sub * 16;
      const uint32_t zp_word = static_cast<uint32_t>(weight_zero_point[
          static_cast<int64_t>(out_row / 8) * groups + group]);
      const int zp = static_cast<int>((zp_word >> ((out_row & 7) * 4)) & 0xFu);
      const float scale = __bfloat162float(
          weight_scale[static_cast<int64_t>(out_row) * groups + group]);
      const uint32_t w0 = static_cast<uint32_t>(weight_packed[
          static_cast<int64_t>(out_row) * packed_cols + (base_col + k0) / 8]);
      const uint32_t w1 = static_cast<uint32_t>(weight_packed[
          static_cast<int64_t>(out_row) * packed_cols + (base_col + k1) / 8]);
      const uint32_t b0 = qwen35_w4_scale_epilogue_pair(w0, k0 & 7, zp, scale);
      const uint32_t b1 = qwen35_w4_scale_epilogue_pair(w1, k1 & 7, zp, scale);
      qwen35_mma_bf16(c[(sub % 4) * 4], c[(sub % 4) * 4 + 1],
                      c[(sub % 4) * 4 + 2], c[(sub % 4) * 4 + 3],
                      a0[sub], 0, a2[sub], 0, b0, b1);
    }
    __syncthreads();
  }
  if (lane < 4) {
    const int col = 2 * lane;
    const float r0 = c[0] + c[4] + c[8] + c[12];
    const float r1 = c[1] + c[5] + c[9] + c[13];
    output[out_base + warp * 8 + col] = __float2bfloat16(r0);
    output[out_base + warp * 8 + col + 1] = __float2bfloat16(r1);
  }
}

// APXINF_STORE_ALT candidate. Arithmetic, MMA order, and row routing match the
// baseline kernel; only the final adjacent BF16 pair is committed as one
// naturally aligned 32-bit store. The baseline kernel above is unchanged.
template <bool Pair>
__global__ void qwen35_gemm_w4a16_bf16_tc_store_alt_kernel(
    const __nv_bfloat16* activation, const int32_t* weight_packed0,
    const __nv_bfloat16* weight_scale0, const int32_t* weight_zero_point0,
    __nv_bfloat16* output0, int out_cols0, const int32_t* weight_packed1,
    const __nv_bfloat16* weight_scale1, const int32_t* weight_zero_point1,
    __nv_bfloat16* output1, int in_cols, int groups) {
  int output_block = blockIdx.x;
  const int first_blocks = out_cols0 / QWEN35_TC_OUT_TILE;
  const int32_t* weight_packed = weight_packed0;
  const __nv_bfloat16* weight_scale = weight_scale0;
  const int32_t* weight_zero_point = weight_zero_point0;
  __nv_bfloat16* output = output0;
  if (Pair && output_block >= first_blocks) {
    output_block -= first_blocks;
    weight_packed = weight_packed1;
    weight_scale = weight_scale1;
    weight_zero_point = weight_zero_point1;
    output = output1;
  }
  const int out_base = output_block * QWEN35_TC_OUT_TILE;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int group_size = (in_cols + groups - 1) / groups;
  const int packed_cols = (in_cols + 7) / 8;
  __shared__ __nv_bfloat16 s_act[128];
  const int a_col = 2 * (lane % 4);
  const int l4 = lane % 4;
  const int jn = lane / 4;
  const int out_row = out_base + warp * 8 + jn;
  const int k0 = 2 * l4;
  const int k1 = k0 + 8;
  float c[16] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  for (int tile128 = 0; tile128 < in_cols; tile128 += 128) {
    for (int idx = threadIdx.x; idx < 128; idx += 256)
      s_act[idx] = activation[tile128 + idx];
    __syncthreads();
    uint32_t a0[8], a2[8];
#pragma unroll
    for (int sub = 0; sub < 8; sub++) {
      a0[sub] = 0;
      a2[sub] = 0;
      if (lane / 4 == 0) {
        a0[sub] = qwen35_pack_bf16(s_act[sub * 16 + a_col],
                                   s_act[sub * 16 + a_col + 1]);
        a2[sub] = qwen35_pack_bf16(s_act[sub * 16 + a_col + 8],
                                   s_act[sub * 16 + a_col + 9]);
      }
    }
#pragma unroll
    for (int sub = 0; sub < 8; sub++) {
      const int group = (tile128 + sub * 16) / group_size;
      const int base_col = tile128 + sub * 16;
      const uint32_t zp_word = static_cast<uint32_t>(weight_zero_point[
          static_cast<int64_t>(out_row / 8) * groups + group]);
      const int zp = static_cast<int>((zp_word >> ((out_row & 7) * 4)) & 0xFu);
      const float scale = __bfloat162float(
          weight_scale[static_cast<int64_t>(out_row) * groups + group]);
      const uint32_t w0 = static_cast<uint32_t>(weight_packed[
          static_cast<int64_t>(out_row) * packed_cols + (base_col + k0) / 8]);
      const uint32_t w1 = static_cast<uint32_t>(weight_packed[
          static_cast<int64_t>(out_row) * packed_cols + (base_col + k1) / 8]);
      const int q00 = static_cast<int>((w0 >> ((k0 & 7) * 4)) & 0xFu);
      const int q01 = static_cast<int>((w0 >> (((k0 + 1) & 7) * 4)) & 0xFu);
      const int q10 = static_cast<int>((w1 >> ((k1 & 7) * 4)) & 0xFu);
      const int q11 = static_cast<int>((w1 >> (((k1 + 1) & 7) * 4)) & 0xFu);
      const uint32_t b0 = qwen35_pack_bf16(
          __float2bfloat16(static_cast<float>(q00 - zp) * scale),
          __float2bfloat16(static_cast<float>(q01 - zp) * scale));
      const uint32_t b1 = qwen35_pack_bf16(
          __float2bfloat16(static_cast<float>(q10 - zp) * scale),
          __float2bfloat16(static_cast<float>(q11 - zp) * scale));
      qwen35_mma_bf16(c[(sub % 4) * 4], c[(sub % 4) * 4 + 1],
                      c[(sub % 4) * 4 + 2], c[(sub % 4) * 4 + 3],
                      a0[sub], 0, a2[sub], 0, b0, b1);
    }
    __syncthreads();
  }
  if (lane < 4) {
    const int col = 2 * lane;
    const float r0 = c[0] + c[4] + c[8] + c[12];
    const float r1 = c[1] + c[5] + c[9] + c[13];
    const uint32_t packed = qwen35_pack_bf16(__float2bfloat16(r0),
                                              __float2bfloat16(r1));
    *reinterpret_cast<uint32_t*>(output + out_base + warp * 8 + col) = packed;
  }
}

// APXINF_W4_PAIR_COARSEN candidate. One 256-thread CTA computes two adjacent
// 64-row output tiles for the selected projection. Activation tiles are loaded
// once into shared memory and consumed by independent exact BF16/MMA accumulators
// for both output tiles; pair routing remains a combined projection grid.
__global__ void qwen35_gemm_w4a16_bf16_tc_pair_coarsen_kernel(
    const __nv_bfloat16* activation, const int32_t* weight_packed0,
    const __nv_bfloat16* weight_scale0, const int32_t* weight_zero_point0,
    __nv_bfloat16* output0, int out_cols0, const int32_t* weight_packed1,
    const __nv_bfloat16* weight_scale1, const int32_t* weight_zero_point1,
    __nv_bfloat16* output1, int out_cols1, int in_cols, int groups) {
  int output_block = blockIdx.x;
  const int first_blocks = out_cols0 / (2 * QWEN35_TC_OUT_TILE);
  const int32_t* weight_packed = weight_packed0;
  const __nv_bfloat16* weight_scale = weight_scale0;
  const int32_t* weight_zero_point = weight_zero_point0;
  __nv_bfloat16* output = output0;
  int out_cols = out_cols0;
  if (output_block >= first_blocks) {
    output_block -= first_blocks;
    weight_packed = weight_packed1;
    weight_scale = weight_scale1;
    weight_zero_point = weight_zero_point1;
    output = output1;
    out_cols = out_cols1;
  }
  const int out_base = output_block * (2 * QWEN35_TC_OUT_TILE);
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int l4 = lane % 4;
  const int jn = lane / 4;
  const int out_row0 = out_base + warp * 8 + jn;
  const int out_row1 = out_row0 + QWEN35_TC_OUT_TILE;
  const int packed_cols = in_cols / 8;
  const int group_size = in_cols / groups;
  const int a_col = 2 * l4;
  const int k0 = 2 * l4;
  const int k1 = k0 + 8;
  __shared__ __nv_bfloat16 s_act[128];
  float c0[16] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  float c1[16] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  for (int tile128 = 0; tile128 < in_cols; tile128 += 128) {
    for (int idx = threadIdx.x; idx < 128; idx += blockDim.x)
      s_act[idx] = activation[tile128 + idx];
    __syncthreads();
    uint32_t a0[8], a2[8];
#pragma unroll
    for (int sub = 0; sub < 8; sub++) {
      a0[sub] = 0;
      a2[sub] = 0;
      if (lane / 4 == 0) {
        a0[sub] = qwen35_pack_bf16(s_act[sub * 16 + a_col],
                                   s_act[sub * 16 + a_col + 1]);
        a2[sub] = qwen35_pack_bf16(s_act[sub * 16 + a_col + 8],
                                   s_act[sub * 16 + a_col + 9]);
      }
    }
#pragma unroll
    for (int sub = 0; sub < 8; sub++) {
      const int group = (tile128 + sub * 16) / group_size;
      const int base_col = tile128 + sub * 16;
      const uint32_t zp_word0 = static_cast<uint32_t>(weight_zero_point[
          static_cast<int64_t>(out_row0 / 8) * groups + group]);
      const int zp0 = static_cast<int>((zp_word0 >> ((out_row0 & 7) * 4)) & 0xFu);
      const float scale0 = __bfloat162float(
          weight_scale[static_cast<int64_t>(out_row0) * groups + group]);
      const uint32_t w00 = static_cast<uint32_t>(weight_packed[
          static_cast<int64_t>(out_row0) * packed_cols + (base_col + k0) / 8]);
      const uint32_t w01 = static_cast<uint32_t>(weight_packed[
          static_cast<int64_t>(out_row0) * packed_cols + (base_col + k1) / 8]);
      const int q000 = static_cast<int>((w00 >> ((k0 & 7) * 4)) & 0xFu);
      const int q001 = static_cast<int>((w00 >> (((k0 + 1) & 7) * 4)) & 0xFu);
      const int q010 = static_cast<int>((w01 >> ((k1 & 7) * 4)) & 0xFu);
      const int q011 = static_cast<int>((w01 >> (((k1 + 1) & 7) * 4)) & 0xFu);
      const uint32_t b00 = qwen35_pack_bf16(
          __float2bfloat16(static_cast<float>(q000 - zp0) * scale0),
          __float2bfloat16(static_cast<float>(q001 - zp0) * scale0));
      const uint32_t b01 = qwen35_pack_bf16(
          __float2bfloat16(static_cast<float>(q010 - zp0) * scale0),
          __float2bfloat16(static_cast<float>(q011 - zp0) * scale0));
      qwen35_mma_bf16(c0[(sub % 4) * 4], c0[(sub % 4) * 4 + 1],
                      c0[(sub % 4) * 4 + 2], c0[(sub % 4) * 4 + 3],
                      a0[sub], 0, a2[sub], 0, b00, b01);

      const uint32_t zp_word1 = static_cast<uint32_t>(weight_zero_point[
          static_cast<int64_t>(out_row1 / 8) * groups + group]);
      const int zp1 = static_cast<int>((zp_word1 >> ((out_row1 & 7) * 4)) & 0xFu);
      const float scale1 = __bfloat162float(
          weight_scale[static_cast<int64_t>(out_row1) * groups + group]);
      const uint32_t w10 = static_cast<uint32_t>(weight_packed[
          static_cast<int64_t>(out_row1) * packed_cols + (base_col + k0) / 8]);
      const uint32_t w11 = static_cast<uint32_t>(weight_packed[
          static_cast<int64_t>(out_row1) * packed_cols + (base_col + k1) / 8]);
      const int q100 = static_cast<int>((w10 >> ((k0 & 7) * 4)) & 0xFu);
      const int q101 = static_cast<int>((w10 >> (((k0 + 1) & 7) * 4)) & 0xFu);
      const int q110 = static_cast<int>((w11 >> ((k1 & 7) * 4)) & 0xFu);
      const int q111 = static_cast<int>((w11 >> (((k1 + 1) & 7) * 4)) & 0xFu);
      const uint32_t b10 = qwen35_pack_bf16(
          __float2bfloat16(static_cast<float>(q100 - zp1) * scale1),
          __float2bfloat16(static_cast<float>(q101 - zp1) * scale1));
      const uint32_t b11 = qwen35_pack_bf16(
          __float2bfloat16(static_cast<float>(q110 - zp1) * scale1),
          __float2bfloat16(static_cast<float>(q111 - zp1) * scale1));
      qwen35_mma_bf16(c1[(sub % 4) * 4], c1[(sub % 4) * 4 + 1],
                      c1[(sub % 4) * 4 + 2], c1[(sub % 4) * 4 + 3],
                      a0[sub], 0, a2[sub], 0, b10, b11);
    }
    __syncthreads();
  }
  if (lane < 4) {
    const int col = 2 * lane;
    const float r00 = c0[0] + c0[4] + c0[8] + c0[12];
    const float r01 = c0[1] + c0[5] + c0[9] + c0[13];
    output[out_base + warp * 8 + col] = __float2bfloat16(r00);
    output[out_base + warp * 8 + col + 1] = __float2bfloat16(r01);
    if (out_row1 < out_cols) {
      const float r10 = c1[0] + c1[4] + c1[8] + c1[12];
      const float r11 = c1[1] + c1[5] + c1[9] + c1[13];
      output[out_base + QWEN35_TC_OUT_TILE + warp * 8 + col] = __float2bfloat16(r10);
      output[out_base + QWEN35_TC_OUT_TILE + warp * 8 + col + 1] = __float2bfloat16(r11);
    }
  }
}

// APXINF_W4_PAIR_SHARED candidate. One 512-thread CTA is partitioned into
// independent 8-warp projection groups. Both groups consume the same K=128
// activation tile, while each retains projection-local metadata, weights,
// accumulator sequence, reduction, and output routing from the baseline.
__global__ void qwen35_gemm_w4a16_bf16_tc_pair_shared_kernel(
    const __nv_bfloat16* activation, const int32_t* weight_packed0,
    const __nv_bfloat16* weight_scale0, const int32_t* weight_zero_point0,
    __nv_bfloat16* output0, int out_cols0, const int32_t* weight_packed1,
    const __nv_bfloat16* weight_scale1, const int32_t* weight_zero_point1,
    __nv_bfloat16* output1, int out_cols1, int in_cols, int groups) {
  const int projection = threadIdx.x / 256;
  const int projection_thread = threadIdx.x % 256;
  const int output_block = blockIdx.x;
  const int out_cols = projection == 0 ? out_cols0 : out_cols1;
  const bool active = output_block < out_cols / QWEN35_TC_OUT_TILE;
  const int32_t* weight_packed =
      projection == 0 ? weight_packed0 : weight_packed1;
  const __nv_bfloat16* weight_scale =
      projection == 0 ? weight_scale0 : weight_scale1;
  const int32_t* weight_zero_point =
      projection == 0 ? weight_zero_point0 : weight_zero_point1;
  __nv_bfloat16* output = projection == 0 ? output0 : output1;
  const int out_base = output_block * QWEN35_TC_OUT_TILE;
  const int warp = projection_thread / 32;
  const int lane = projection_thread % 32;
  const int group_size = in_cols / groups;
  const int packed_cols = in_cols / 8;
  const int a_col = 2 * (lane % 4);
  const int l4 = lane % 4;
  const int jn = lane / 4;
  const int out_row = out_base + warp * 8 + jn;
  const int k0 = 2 * l4;
  const int k1 = k0 + 8;
  __shared__ __nv_bfloat16 s_act[128];
  float c[16] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  for (int tile128 = 0; tile128 < in_cols; tile128 += 128) {
    if (threadIdx.x < 128)
      s_act[threadIdx.x] = activation[tile128 + threadIdx.x];
    __syncthreads();
    uint32_t a0[8], a2[8];
#pragma unroll
    for (int sub = 0; sub < 8; sub++) {
      a0[sub] = 0;
      a2[sub] = 0;
      if (lane / 4 == 0) {
        a0[sub] = qwen35_pack_bf16(s_act[sub * 16 + a_col],
                                   s_act[sub * 16 + a_col + 1]);
        a2[sub] = qwen35_pack_bf16(s_act[sub * 16 + a_col + 8],
                                   s_act[sub * 16 + a_col + 9]);
      }
    }
#pragma unroll
    for (int sub = 0; sub < 8; sub++) {
      if (active) {
        const int group = (tile128 + sub * 16) / group_size;
        const int base_col = tile128 + sub * 16;
        const uint32_t zp_word = static_cast<uint32_t>(weight_zero_point[
            static_cast<int64_t>(out_row / 8) * groups + group]);
        const int zp =
            static_cast<int>((zp_word >> ((out_row & 7) * 4)) & 0xFu);
        const float scale = __bfloat162float(
            weight_scale[static_cast<int64_t>(out_row) * groups + group]);
        const uint32_t w0 = static_cast<uint32_t>(weight_packed[
            static_cast<int64_t>(out_row) * packed_cols + (base_col + k0) / 8]);
        const uint32_t w1 = static_cast<uint32_t>(weight_packed[
            static_cast<int64_t>(out_row) * packed_cols + (base_col + k1) / 8]);
        const int q00 = static_cast<int>((w0 >> ((k0 & 7) * 4)) & 0xFu);
        const int q01 =
            static_cast<int>((w0 >> (((k0 + 1) & 7) * 4)) & 0xFu);
        const int q10 = static_cast<int>((w1 >> ((k1 & 7) * 4)) & 0xFu);
        const int q11 =
            static_cast<int>((w1 >> (((k1 + 1) & 7) * 4)) & 0xFu);
        const uint32_t b0 = qwen35_pack_bf16(
            __float2bfloat16(static_cast<float>(q00 - zp) * scale),
            __float2bfloat16(static_cast<float>(q01 - zp) * scale));
        const uint32_t b1 = qwen35_pack_bf16(
            __float2bfloat16(static_cast<float>(q10 - zp) * scale),
            __float2bfloat16(static_cast<float>(q11 - zp) * scale));
        qwen35_mma_bf16(c[(sub % 4) * 4], c[(sub % 4) * 4 + 1],
                        c[(sub % 4) * 4 + 2], c[(sub % 4) * 4 + 3],
                        a0[sub], 0, a2[sub], 0, b0, b1);
      }
    }
    __syncthreads();
  }
  if (active && lane < 4) {
    const int col = 2 * lane;
    const float r0 = c[0] + c[4] + c[8] + c[12];
    const float r1 = c[1] + c[5] + c[9] + c[13];
    output[out_base + warp * 8 + col] = __float2bfloat16(r0);
    output[out_base + warp * 8 + col + 1] = __float2bfloat16(r1);
  }
}


// APXINF_W4_PAIR_REUSE candidate. One CTA owns the same 64-row tile in both
// projections, stages each K=128 activation tile once, and consumes that
// shared tile with separate weights, scales, zero points, and accumulators.
// Each projection retains the baseline K-ordered MMA sequence and reduction.
__global__ void qwen35_gemm_w4a16_bf16_tc_pair_reuse_kernel(
    const __nv_bfloat16* activation, const int32_t* weight_packed0,
    const __nv_bfloat16* weight_scale0, const int32_t* weight_zero_point0,
    __nv_bfloat16* output0, int out_cols0, const int32_t* weight_packed1,
    const __nv_bfloat16* weight_scale1, const int32_t* weight_zero_point1,
    __nv_bfloat16* output1, int out_cols1, int in_cols, int groups) {
  const int output_block = blockIdx.x;
  const bool active0 = output_block < out_cols0 / QWEN35_TC_OUT_TILE;
  const bool active1 = output_block < out_cols1 / QWEN35_TC_OUT_TILE;
  const int out_base = output_block * QWEN35_TC_OUT_TILE;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int group_size = (in_cols + groups - 1) / groups;
  const int packed_cols = (in_cols + 7) / 8;
  __shared__ __nv_bfloat16 s_act[128];
  const int a_col = 2 * (lane % 4);
  const int l4 = lane % 4;
  const int jn = lane / 4;
  const int out_row = out_base + warp * 8 + jn;
  const int k0 = 2 * l4;
  const int k1 = k0 + 8;
  float c0[16] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  float c1[16] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  for (int tile128 = 0; tile128 < in_cols; tile128 += 128) {
    for (int idx = threadIdx.x; idx < 128; idx += blockDim.x)
      s_act[idx] = activation[tile128 + idx];
    __syncthreads();
    uint32_t a0[8], a2[8];
#pragma unroll
    for (int sub = 0; sub < 8; sub++) {
      a0[sub] = 0;
      a2[sub] = 0;
      if (lane / 4 == 0) {
        a0[sub] = qwen35_pack_bf16(s_act[sub * 16 + a_col],
                                   s_act[sub * 16 + a_col + 1]);
        a2[sub] = qwen35_pack_bf16(s_act[sub * 16 + a_col + 8],
                                   s_act[sub * 16 + a_col + 9]);
      }
    }
#pragma unroll
    for (int sub = 0; sub < 8; sub++) {
      const int group = (tile128 + sub * 16) / group_size;
      const int base_col = tile128 + sub * 16;
      if (active0) {
        const uint32_t zp_word = static_cast<uint32_t>(weight_zero_point0[
            static_cast<int64_t>(out_row / 8) * groups + group]);
        const int zp =
            static_cast<int>((zp_word >> ((out_row & 7) * 4)) & 0xFu);
        const float scale = __bfloat162float(
            weight_scale0[static_cast<int64_t>(out_row) * groups + group]);
        const uint32_t w0 = static_cast<uint32_t>(weight_packed0[
            static_cast<int64_t>(out_row) * packed_cols + (base_col + k0) / 8]);
        const uint32_t w1 = static_cast<uint32_t>(weight_packed0[
            static_cast<int64_t>(out_row) * packed_cols + (base_col + k1) / 8]);
        const int q00 = static_cast<int>((w0 >> ((k0 & 7) * 4)) & 0xFu);
        const int q01 =
            static_cast<int>((w0 >> (((k0 + 1) & 7) * 4)) & 0xFu);
        const int q10 = static_cast<int>((w1 >> ((k1 & 7) * 4)) & 0xFu);
        const int q11 =
            static_cast<int>((w1 >> (((k1 + 1) & 7) * 4)) & 0xFu);
        const uint32_t b0 = qwen35_pack_bf16(
            __float2bfloat16(static_cast<float>(q00 - zp) * scale),
            __float2bfloat16(static_cast<float>(q01 - zp) * scale));
        const uint32_t b1 = qwen35_pack_bf16(
            __float2bfloat16(static_cast<float>(q10 - zp) * scale),
            __float2bfloat16(static_cast<float>(q11 - zp) * scale));
        qwen35_mma_bf16(c0[(sub % 4) * 4], c0[(sub % 4) * 4 + 1],
                        c0[(sub % 4) * 4 + 2], c0[(sub % 4) * 4 + 3],
                        a0[sub], 0, a2[sub], 0, b0, b1);
      }
      if (active1) {
        const uint32_t zp_word = static_cast<uint32_t>(weight_zero_point1[
            static_cast<int64_t>(out_row / 8) * groups + group]);
        const int zp =
            static_cast<int>((zp_word >> ((out_row & 7) * 4)) & 0xFu);
        const float scale = __bfloat162float(
            weight_scale1[static_cast<int64_t>(out_row) * groups + group]);
        const uint32_t w0 = static_cast<uint32_t>(weight_packed1[
            static_cast<int64_t>(out_row) * packed_cols + (base_col + k0) / 8]);
        const uint32_t w1 = static_cast<uint32_t>(weight_packed1[
            static_cast<int64_t>(out_row) * packed_cols + (base_col + k1) / 8]);
        const int q00 = static_cast<int>((w0 >> ((k0 & 7) * 4)) & 0xFu);
        const int q01 =
            static_cast<int>((w0 >> (((k0 + 1) & 7) * 4)) & 0xFu);
        const int q10 = static_cast<int>((w1 >> ((k1 & 7) * 4)) & 0xFu);
        const int q11 =
            static_cast<int>((w1 >> (((k1 + 1) & 7) * 4)) & 0xFu);
        const uint32_t b0 = qwen35_pack_bf16(
            __float2bfloat16(static_cast<float>(q00 - zp) * scale),
            __float2bfloat16(static_cast<float>(q01 - zp) * scale));
        const uint32_t b1 = qwen35_pack_bf16(
            __float2bfloat16(static_cast<float>(q10 - zp) * scale),
            __float2bfloat16(static_cast<float>(q11 - zp) * scale));
        qwen35_mma_bf16(c1[(sub % 4) * 4], c1[(sub % 4) * 4 + 1],
                        c1[(sub % 4) * 4 + 2], c1[(sub % 4) * 4 + 3],
                        a0[sub], 0, a2[sub], 0, b0, b1);
      }
    }
    __syncthreads();
  }
  if (lane < 4) {
    const int col = 2 * lane;
    if (active0) {
      const float r0 = c0[0] + c0[4] + c0[8] + c0[12];
      const float r1 = c0[1] + c0[5] + c0[9] + c0[13];
      output0[out_base + warp * 8 + col] = __float2bfloat16(r0);
      output0[out_base + warp * 8 + col + 1] = __float2bfloat16(r1);
    }
    if (active1) {
      const float r0 = c1[0] + c1[4] + c1[8] + c1[12];
      const float r1 = c1[1] + c1[5] + c1[9] + c1[13];
      output1[out_base + warp * 8 + col] = __float2bfloat16(r0);
      output1[out_base + warp * 8 + col + 1] = __float2bfloat16(r1);
    }
  }
}
// APXINF_PAIR_OCCUPANCY candidate. The arithmetic, K-ordered MMA sequence,
// accumulator reduction, and pair routing are shared with the established
// register-tuned kernel; only this distinct launch symbol requests three
// resident 256-thread CTAs when the hardware permits it.
template <int MinBlocks>
__global__ __launch_bounds__(256, MinBlocks)
void qwen35_gemm_w4a16_bf16_tc_pair_occupancy_kernel(
    const __nv_bfloat16* activation, const int32_t* weight_packed0,
    const __nv_bfloat16* weight_scale0, const int32_t* weight_zero_point0,
    __nv_bfloat16* output0, int out_cols0, const int32_t* weight_packed1,
    const __nv_bfloat16* weight_scale1, const int32_t* weight_zero_point1,
    __nv_bfloat16* output1, int in_cols, int groups) {
  int output_block = blockIdx.x;
  const int first_blocks = out_cols0 / QWEN35_TC_OUT_TILE;
  const int32_t* weight_packed = weight_packed0;
  const __nv_bfloat16* weight_scale = weight_scale0;
  const int32_t* weight_zero_point = weight_zero_point0;
  __nv_bfloat16* output = output0;
  if (output_block >= first_blocks) {
    output_block -= first_blocks;
    weight_packed = weight_packed1;
    weight_scale = weight_scale1;
    weight_zero_point = weight_zero_point1;
    output = output1;
  }
  const int out_base = output_block * QWEN35_TC_OUT_TILE;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int group_size = (in_cols + groups - 1) / groups;
  const int packed_cols = (in_cols + 7) / 8;
  __shared__ __nv_bfloat16 s_act[128];
  const int a_col = 2 * (lane % 4);
  const int l4 = lane % 4;
  const int jn = lane / 4;
  const int out_row = out_base + warp * 8 + jn;
  const int k0 = 2 * l4;
  const int k1 = k0 + 8;
  float c[16] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  for (int tile128 = 0; tile128 < in_cols; tile128 += 128) {
    for (int idx = threadIdx.x; idx < 128; idx += 256)
      s_act[idx] = activation[tile128 + idx];
    __syncthreads();
    uint32_t a0[8], a2[8];
#pragma unroll
    for (int sub = 0; sub < 8; sub++) {
      a0[sub] = 0;
      a2[sub] = 0;
      if (lane / 4 == 0) {
        a0[sub] = qwen35_pack_bf16(s_act[sub * 16 + a_col],
                                   s_act[sub * 16 + a_col + 1]);
        a2[sub] = qwen35_pack_bf16(s_act[sub * 16 + a_col + 8],
                                   s_act[sub * 16 + a_col + 9]);
      }
    }
#pragma unroll
    for (int sub = 0; sub < 8; sub++) {
      const int group = (tile128 + sub * 16) / group_size;
      const int base_col = tile128 + sub * 16;
      const uint32_t zp_word = static_cast<uint32_t>(weight_zero_point[
          static_cast<int64_t>(out_row / 8) * groups + group]);
      const int zp = static_cast<int>((zp_word >> ((out_row & 7) * 4)) & 0xFu);
      const float scale = __bfloat162float(
          weight_scale[static_cast<int64_t>(out_row) * groups + group]);
      const uint32_t w0 = static_cast<uint32_t>(weight_packed[
          static_cast<int64_t>(out_row) * packed_cols + (base_col + k0) / 8]);
      const uint32_t w1 = static_cast<uint32_t>(weight_packed[
          static_cast<int64_t>(out_row) * packed_cols + (base_col + k1) / 8]);
      const int q00 = static_cast<int>((w0 >> ((k0 & 7) * 4)) & 0xFu);
      const int q01 = static_cast<int>((w0 >> (((k0 + 1) & 7) * 4)) & 0xFu);
      const int q10 = static_cast<int>((w1 >> ((k1 & 7) * 4)) & 0xFu);
      const int q11 = static_cast<int>((w1 >> (((k1 + 1) & 7) * 4)) & 0xFu);
      const uint32_t b0 = qwen35_pack_bf16(
          __float2bfloat16(static_cast<float>(q00 - zp) * scale),
          __float2bfloat16(static_cast<float>(q01 - zp) * scale));
      const uint32_t b1 = qwen35_pack_bf16(
          __float2bfloat16(static_cast<float>(q10 - zp) * scale),
          __float2bfloat16(static_cast<float>(q11 - zp) * scale));
      qwen35_mma_bf16(c[(sub % 4) * 4], c[(sub % 4) * 4 + 1],
                      c[(sub % 4) * 4 + 2], c[(sub % 4) * 4 + 3],
                      a0[sub], 0, a2[sub], 0, b0, b1);
    }
    __syncthreads();
  }
  if (lane < 4) {
    const int col = 2 * lane;
    const float r0 = c[0] + c[4] + c[8] + c[12];
    const float r1 = c[1] + c[5] + c[9] + c[13];
    output[out_base + warp * 8 + col] = __float2bfloat16(r0);
    output[out_base + warp * 8 + col + 1] = __float2bfloat16(r1);
  }
}

__global__ __launch_bounds__(256, 2)
void qwen35_gemm_w4a16_bf16_tc_pair_reg_kernel(
    const __nv_bfloat16* activation, const int32_t* weight_packed0,
    const __nv_bfloat16* weight_scale0, const int32_t* weight_zero_point0,
    __nv_bfloat16* output0, int out_cols0, const int32_t* weight_packed1,
    const __nv_bfloat16* weight_scale1, const int32_t* weight_zero_point1,
    __nv_bfloat16* output1, int in_cols, int groups) {
  int output_block = blockIdx.x;
  const int first_blocks = out_cols0 / QWEN35_TC_OUT_TILE;
  const int32_t* weight_packed = weight_packed0;
  const __nv_bfloat16* weight_scale = weight_scale0;
  const int32_t* weight_zero_point = weight_zero_point0;
  __nv_bfloat16* output = output0;
  if (output_block >= first_blocks) {
    output_block -= first_blocks;
    weight_packed = weight_packed1;
    weight_scale = weight_scale1;
    weight_zero_point = weight_zero_point1;
    output = output1;
  }
  const int out_base = output_block * QWEN35_TC_OUT_TILE;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int group_size = (in_cols + groups - 1) / groups;
  const int packed_cols = (in_cols + 7) / 8;
  __shared__ __nv_bfloat16 s_act[128];
  const int a_col = 2 * (lane % 4);
  const int l4 = lane % 4;
  const int jn = lane / 4;
  const int out_row = out_base + warp * 8 + jn;
  const int k0 = 2 * l4;
  const int k1 = k0 + 8;
  float c[16] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  for (int tile128 = 0; tile128 < in_cols; tile128 += 128) {
    for (int idx = threadIdx.x; idx < 128; idx += 256)
      s_act[idx] = activation[tile128 + idx];
    __syncthreads();
    uint32_t a0[8], a2[8];
#pragma unroll
    for (int sub = 0; sub < 8; sub++) {
      a0[sub] = 0;
      a2[sub] = 0;
      if (lane / 4 == 0) {
        a0[sub] = qwen35_pack_bf16(s_act[sub * 16 + a_col],
                                   s_act[sub * 16 + a_col + 1]);
        a2[sub] = qwen35_pack_bf16(s_act[sub * 16 + a_col + 8],
                                   s_act[sub * 16 + a_col + 9]);
      }
    }
#pragma unroll
    for (int sub = 0; sub < 8; sub++) {
      const int group = (tile128 + sub * 16) / group_size;
      const int base_col = tile128 + sub * 16;
      const uint32_t zp_word = static_cast<uint32_t>(weight_zero_point[
          static_cast<int64_t>(out_row / 8) * groups + group]);
      const int zp = static_cast<int>((zp_word >> ((out_row & 7) * 4)) & 0xFu);
      const float scale = __bfloat162float(
          weight_scale[static_cast<int64_t>(out_row) * groups + group]);
      const uint32_t w0 = static_cast<uint32_t>(weight_packed[
          static_cast<int64_t>(out_row) * packed_cols + (base_col + k0) / 8]);
      const uint32_t w1 = static_cast<uint32_t>(weight_packed[
          static_cast<int64_t>(out_row) * packed_cols + (base_col + k1) / 8]);
      const int q00 = static_cast<int>((w0 >> ((k0 & 7) * 4)) & 0xFu);
      const int q01 = static_cast<int>((w0 >> (((k0 + 1) & 7) * 4)) & 0xFu);
      const int q10 = static_cast<int>((w1 >> ((k1 & 7) * 4)) & 0xFu);
      const int q11 = static_cast<int>((w1 >> (((k1 + 1) & 7) * 4)) & 0xFu);
      const uint32_t b0 = qwen35_pack_bf16(
          __float2bfloat16(static_cast<float>(q00 - zp) * scale),
          __float2bfloat16(static_cast<float>(q01 - zp) * scale));
      const uint32_t b1 = qwen35_pack_bf16(
          __float2bfloat16(static_cast<float>(q10 - zp) * scale),
          __float2bfloat16(static_cast<float>(q11 - zp) * scale));
      qwen35_mma_bf16(c[(sub % 4) * 4], c[(sub % 4) * 4 + 1],
                      c[(sub % 4) * 4 + 2], c[(sub % 4) * 4 + 3],
                      a0[sub], 0, a2[sub], 0, b0, b1);
    }
    __syncthreads();
  }
  if (lane < 4) {
    const int col = 2 * lane;
    const float r0 = c[0] + c[4] + c[8] + c[12];
    const float r1 = c[1] + c[5] + c[9] + c[13];
    output[out_base + warp * 8 + col] = __float2bfloat16(r0);
    output[out_base + warp * 8 + col + 1] = __float2bfloat16(r1);
  }
}

// APXINF_W4_VECTOR_MMA candidate. Vectorized uint2 packed-word and BF16
// fragment loads preserve the baseline dequant operands, MMA order,
// accumulator reduction, and output ownership. It is selected only by the
// explicitly gated ABI/model path; the baseline kernel remains unchanged.
template <bool Pair>
__global__ void qwen35_gemm_w4a16_bf16_tc_vector_mma_kernel(
    const __nv_bfloat16* activation, const int32_t* weight_packed0,
    const __nv_bfloat16* weight_scale0, const int32_t* weight_zero_point0,
    __nv_bfloat16* output0, int out_cols0, const int32_t* weight_packed1,
    const __nv_bfloat16* weight_scale1, const int32_t* weight_zero_point1,
    __nv_bfloat16* output1, int in_cols, int groups) {
  int output_block = blockIdx.x;
  const int first_blocks = out_cols0 / QWEN35_TC_OUT_TILE;
  const int32_t* weight_packed = weight_packed0;
  const __nv_bfloat16* weight_scale = weight_scale0;
  const int32_t* weight_zero_point = weight_zero_point0;
  __nv_bfloat16* output = output0;
  if (Pair && output_block >= first_blocks) {
    output_block -= first_blocks;
    weight_packed = weight_packed1;
    weight_scale = weight_scale1;
    weight_zero_point = weight_zero_point1;
    output = output1;
  }
  const int out_base = output_block * QWEN35_TC_OUT_TILE;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int group_size = (in_cols + groups - 1) / groups;
  const int packed_cols = (in_cols + 7) / 8;
  const int l4 = lane % 4;
  const int jn = lane / 4;
  const int out_row = out_base + warp * 8 + jn;
  const int k0 = 2 * l4;
  const int k1 = k0 + 8;
  float c[16] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  for (int tile128 = 0; tile128 < in_cols; tile128 += 128) {
    uint32_t a0[8], a2[8];
#pragma unroll
    for (int sub = 0; sub < 8; sub++) {
      a0[sub] = 0;
      a2[sub] = 0;
      if (lane / 4 == 0) {
        const int a_base = tile128 + sub * 16 + 2 * l4;
        a0[sub] = *reinterpret_cast<const uint32_t*>(activation + a_base);
        a2[sub] =
            *reinterpret_cast<const uint32_t*>(activation + a_base + 8);
      }
    }
#pragma unroll
    for (int sub = 0; sub < 8; sub++) {
      const int group = (tile128 + sub * 16) / group_size;
      const int base_col = tile128 + sub * 16;
      const uint32_t zp_word = static_cast<uint32_t>(weight_zero_point[
          static_cast<int64_t>(out_row / 8) * groups + group]);
      const int zp = static_cast<int>((zp_word >> ((out_row & 7) * 4)) & 0xFu);
      const float scale = __bfloat162float(
          weight_scale[static_cast<int64_t>(out_row) * groups + group]);
      const uint2 wpair = *reinterpret_cast<const uint2*>(
          weight_packed + static_cast<int64_t>(out_row) * packed_cols +
          base_col / 8);
      const uint32_t w0 = wpair.x;
      const uint32_t w1 = wpair.y;
      const int q00 = static_cast<int>((w0 >> ((k0 & 7) * 4)) & 0xFu);
      const int q01 = static_cast<int>((w0 >> (((k0 + 1) & 7) * 4)) & 0xFu);
      const int q10 = static_cast<int>((w1 >> ((k1 & 7) * 4)) & 0xFu);
      const int q11 = static_cast<int>((w1 >> (((k1 + 1) & 7) * 4)) & 0xFu);
      const uint32_t b0 = qwen35_pack_bf16(
          __float2bfloat16(static_cast<float>(q00 - zp) * scale),
          __float2bfloat16(static_cast<float>(q01 - zp) * scale));
      const uint32_t b1 = qwen35_pack_bf16(
          __float2bfloat16(static_cast<float>(q10 - zp) * scale),
          __float2bfloat16(static_cast<float>(q11 - zp) * scale));
      qwen35_mma_bf16(c[(sub % 4) * 4], c[(sub % 4) * 4 + 1],
                      c[(sub % 4) * 4 + 2], c[(sub % 4) * 4 + 3],
                      a0[sub], 0, a2[sub], 0, b0, b1);
    }
  }
  if (lane < 4) {
    const int col = 2 * lane;
    const float r0 = c[0] + c[4] + c[8] + c[12];
    const float r1 = c[1] + c[5] + c[9] + c[13];
    output[out_base + warp * 8 + col] = __float2bfloat16(r0);
    output[out_base + warp * 8 + col + 1] = __float2bfloat16(r1);
  }
}


// Raw compressed-tensors decode with metadata staged per exact group/output
// tile. The shared indices mirror scale[out_row, group] and
// zero_point[out_row / 8, group]; packed-weight loads, BF16 conversion, MMA
// order, row ownership, and accumulator reduction match the baseline kernel.
__global__ void qwen35_gemm_w4a16_bf16_tc_meta_shared_kernel(
    const __nv_bfloat16* activation, const int32_t* weight_packed,
    const __nv_bfloat16* weight_scale, const int32_t* weight_zero_point,
    __nv_bfloat16* output, int in_cols, int groups) {
  const int out_base = blockIdx.x * QWEN35_TC_OUT_TILE;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int l4 = lane % 4;
  const int jn = lane / 4;
  const int out_row = out_base + warp * 8 + jn;
  const int packed_cols = (in_cols + 7) / 8;
  const int a_col = 2 * l4;
  const int k0 = 2 * l4;
  const int k1 = k0 + 8;
  __shared__ __nv_bfloat16 s_act[128];
  __shared__ __nv_bfloat16 s_scale[4 * QWEN35_TC_OUT_TILE];
  __shared__ uint32_t s_zp[4 * (QWEN35_TC_OUT_TILE / 8)];
  float c[16] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  for (int tile128 = 0; tile128 < in_cols; tile128 += 128) {
    for (int idx = threadIdx.x; idx < 128; idx += blockDim.x)
      s_act[idx] = activation[tile128 + idx];

    // Group size is exactly 32 for this gated kernel. Four consecutive
    // groups cover the K=128 tile, and every scale row retains its original
    // raw-layout index.
    const int tile_group = tile128 / 32;
    for (int idx = threadIdx.x; idx < 4 * QWEN35_TC_OUT_TILE;
         idx += blockDim.x) {
      const int group_local = idx / QWEN35_TC_OUT_TILE;
      const int row_local = idx % QWEN35_TC_OUT_TILE;
      s_scale[idx] = weight_scale[
          static_cast<int64_t>(out_base + row_local) * groups +
          tile_group + group_local];
    }
    for (int idx = threadIdx.x;
         idx < 4 * (QWEN35_TC_OUT_TILE / 8); idx += blockDim.x) {
      const int group_local = idx / (QWEN35_TC_OUT_TILE / 8);
      const int row_pack_local = idx % (QWEN35_TC_OUT_TILE / 8);
      s_zp[idx] = static_cast<uint32_t>(weight_zero_point[
          static_cast<int64_t>(out_base / 8 + row_pack_local) * groups +
          tile_group + group_local]);
    }
    __syncthreads();

    uint32_t a0[8], a2[8];
#pragma unroll
    for (int sub = 0; sub < 8; sub++) {
      a0[sub] = 0;
      a2[sub] = 0;
      if (lane / 4 == 0) {
        a0[sub] = qwen35_pack_bf16(s_act[sub * 16 + a_col],
                                   s_act[sub * 16 + a_col + 1]);
        a2[sub] = qwen35_pack_bf16(s_act[sub * 16 + a_col + 8],
                                   s_act[sub * 16 + a_col + 9]);
      }
    }
#pragma unroll
    for (int sub = 0; sub < 8; sub++) {
      const int group_local = sub / 2;
      const int base_col = tile128 + sub * 16;
      const uint32_t zp_word =
          s_zp[group_local * (QWEN35_TC_OUT_TILE / 8) + warp];
      const int zp = static_cast<int>((zp_word >> (jn * 4)) & 0xFu);
      const float scale = __bfloat162float(
          s_scale[group_local * QWEN35_TC_OUT_TILE + warp * 8 + jn]);
      const uint32_t w0 = static_cast<uint32_t>(weight_packed[
          static_cast<int64_t>(out_row) * packed_cols + (base_col + k0) / 8]);
      const uint32_t w1 = static_cast<uint32_t>(weight_packed[
          static_cast<int64_t>(out_row) * packed_cols + (base_col + k1) / 8]);
      const int q00 = static_cast<int>((w0 >> ((k0 & 7) * 4)) & 0xFu);
      const int q01 =
          static_cast<int>((w0 >> (((k0 + 1) & 7) * 4)) & 0xFu);
      const int q10 = static_cast<int>((w1 >> ((k1 & 7) * 4)) & 0xFu);
      const int q11 =
          static_cast<int>((w1 >> (((k1 + 1) & 7) * 4)) & 0xFu);
      const uint32_t b0 = qwen35_pack_bf16(
          __float2bfloat16(static_cast<float>(q00 - zp) * scale),
          __float2bfloat16(static_cast<float>(q01 - zp) * scale));
      const uint32_t b1 = qwen35_pack_bf16(
          __float2bfloat16(static_cast<float>(q10 - zp) * scale),
          __float2bfloat16(static_cast<float>(q11 - zp) * scale));
      qwen35_mma_bf16(c[(sub % 4) * 4], c[(sub % 4) * 4 + 1],
                      c[(sub % 4) * 4 + 2], c[(sub % 4) * 4 + 3],
                      a0[sub], 0, a2[sub], 0, b0, b1);
    }
    __syncthreads();
  }
  if (lane < 4) {
    const int col = 2 * lane;
    const float r0 = c[0] + c[4] + c[8] + c[12];
    const float r1 = c[1] + c[5] + c[9] + c[13];
    output[out_base + warp * 8 + col] = __float2bfloat16(r0);
    output[out_base + warp * 8 + col + 1] = __float2bfloat16(r1);
  }
}

// Alternate raw-layout geometry: four warps compute 32 output rows per CTA.
// Each warp retains the baseline row ownership, dequantization, K-ordered MMA
// sequence, accumulator layout, and final reduction; only CTA output coverage
// and activation/metadata staging participation change.
__global__ void qwen35_gemm_w4a16_bf16_tc_tile_alt_kernel(
    const __nv_bfloat16* activation, const int32_t* weight_packed,
    const __nv_bfloat16* weight_scale, const int32_t* weight_zero_point,
    __nv_bfloat16* output, int in_cols, int groups) {
  const int out_base = blockIdx.x * QWEN35_TC_ALT_OUT_TILE;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int l4 = lane % 4;
  const int jn = lane / 4;
  const int out_row = out_base + warp * 8 + jn;
  const int packed_cols = (in_cols + 7) / 8;
  const int a_col = 2 * l4;
  const int k0 = 2 * l4;
  const int k1 = k0 + 8;
  __shared__ __nv_bfloat16 s_act[128];
  __shared__ __nv_bfloat16 s_scale[4 * QWEN35_TC_ALT_OUT_TILE];
  __shared__ uint32_t s_zp[4 * (QWEN35_TC_ALT_OUT_TILE / 8)];
  float c[16] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  for (int tile128 = 0; tile128 < in_cols; tile128 += 128) {
    for (int idx = threadIdx.x; idx < 128; idx += blockDim.x)
      s_act[idx] = activation[tile128 + idx];

    const int tile_group = tile128 / 32;
    for (int idx = threadIdx.x; idx < 4 * QWEN35_TC_ALT_OUT_TILE;
         idx += blockDim.x) {
      const int group_local = idx / QWEN35_TC_ALT_OUT_TILE;
      const int row_local = idx % QWEN35_TC_ALT_OUT_TILE;
      s_scale[idx] = weight_scale[
          static_cast<int64_t>(out_base + row_local) * groups +
          tile_group + group_local];
    }
    for (int idx = threadIdx.x;
         idx < 4 * (QWEN35_TC_ALT_OUT_TILE / 8); idx += blockDim.x) {
      const int group_local = idx / (QWEN35_TC_ALT_OUT_TILE / 8);
      const int row_pack_local = idx % (QWEN35_TC_ALT_OUT_TILE / 8);
      s_zp[idx] = static_cast<uint32_t>(weight_zero_point[
          static_cast<int64_t>(out_base / 8 + row_pack_local) * groups +
          tile_group + group_local]);
    }
    __syncthreads();

    uint32_t a0[8], a2[8];
#pragma unroll
    for (int sub = 0; sub < 8; sub++) {
      a0[sub] = 0;
      a2[sub] = 0;
      if (lane / 4 == 0) {
        a0[sub] = qwen35_pack_bf16(s_act[sub * 16 + a_col],
                                   s_act[sub * 16 + a_col + 1]);
        a2[sub] = qwen35_pack_bf16(s_act[sub * 16 + a_col + 8],
                                   s_act[sub * 16 + a_col + 9]);
      }
    }
#pragma unroll
    for (int sub = 0; sub < 8; sub++) {
      const int group_local = sub / 2;
      const int base_col = tile128 + sub * 16;
      const uint32_t zp_word =
          s_zp[group_local * (QWEN35_TC_ALT_OUT_TILE / 8) + warp];
      const int zp = static_cast<int>((zp_word >> (jn * 4)) & 0xFu);
      const float scale = __bfloat162float(
          s_scale[group_local * QWEN35_TC_ALT_OUT_TILE + warp * 8 + jn]);
      const uint32_t w0 = static_cast<uint32_t>(weight_packed[
          static_cast<int64_t>(out_row) * packed_cols + (base_col + k0) / 8]);
      const uint32_t w1 = static_cast<uint32_t>(weight_packed[
          static_cast<int64_t>(out_row) * packed_cols + (base_col + k1) / 8]);
      const int q00 = static_cast<int>((w0 >> ((k0 & 7) * 4)) & 0xFu);
      const int q01 =
          static_cast<int>((w0 >> (((k0 + 1) & 7) * 4)) & 0xFu);
      const int q10 = static_cast<int>((w1 >> ((k1 & 7) * 4)) & 0xFu);
      const int q11 =
          static_cast<int>((w1 >> (((k1 + 1) & 7) * 4)) & 0xFu);
      const uint32_t b0 = qwen35_pack_bf16(
          __float2bfloat16(static_cast<float>(q00 - zp) * scale),
          __float2bfloat16(static_cast<float>(q01 - zp) * scale));
      const uint32_t b1 = qwen35_pack_bf16(
          __float2bfloat16(static_cast<float>(q10 - zp) * scale),
          __float2bfloat16(static_cast<float>(q11 - zp) * scale));
      qwen35_mma_bf16(c[(sub % 4) * 4], c[(sub % 4) * 4 + 1],
                      c[(sub % 4) * 4 + 2], c[(sub % 4) * 4 + 3],
                      a0[sub], 0, a2[sub], 0, b0, b1);
    }
    __syncthreads();
  }
  if (lane < 4) {
    const int col = 2 * lane;
    const float r0 = c[0] + c[4] + c[8] + c[12];
    const float r1 = c[1] + c[5] + c[9] + c[13];
    output[out_base + warp * 8 + col] = __float2bfloat16(r0);
    output[out_base + warp * 8 + col + 1] = __float2bfloat16(r1);
  }
}

// Exact raw-layout gate/up fusion for batch-one Qwen MLP. One CTA owns the
// same 32 output rows in both projections, preserves each projection's K/MMA
// and final reduction order, explicitly rounds both projection results to
// BF16, then applies the established BF16 SiLU*up boundary without writing
// gate/up intermediates to global memory.
__global__ void qwen35_gemm_w4a16_bf16_gate_up_silu_kernel(
    const __nv_bfloat16* activation,
    const int32_t* gate_packed, const __nv_bfloat16* gate_scale,
    const int32_t* gate_zero_point,
    const int32_t* up_packed, const __nv_bfloat16* up_scale,
    const int32_t* up_zero_point, __nv_bfloat16* output,
    int in_cols, int out_cols, int groups) {
  const int out_base = blockIdx.x * QWEN35_TC_ALT_OUT_TILE;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int l4 = lane % 4;
  const int jn = lane / 4;
  const int out_row = out_base + warp * 8 + jn;
  const int packed_cols = in_cols / 8;
  const int a_col = 2 * l4;
  const int k0 = 2 * l4;
  const int k1 = k0 + 8;
  __shared__ __nv_bfloat16 s_act[128];
  __shared__ __nv_bfloat16 s_scale[2][4 * QWEN35_TC_ALT_OUT_TILE];
  __shared__ uint32_t s_zp[2][4 * (QWEN35_TC_ALT_OUT_TILE / 8)];
  float c_gate[16] = {};
  float c_up[16] = {};

  for (int tile128 = 0; tile128 < in_cols; tile128 += 128) {
    s_act[threadIdx.x] = activation[tile128 + threadIdx.x];
    const int tile_group = tile128 / 32;
    for (int idx = threadIdx.x; idx < 4 * QWEN35_TC_ALT_OUT_TILE;
         idx += blockDim.x) {
      const int group_local = idx / QWEN35_TC_ALT_OUT_TILE;
      const int row_local = idx % QWEN35_TC_ALT_OUT_TILE;
      const int64_t source =
          static_cast<int64_t>(out_base + row_local) * groups +
          tile_group + group_local;
      s_scale[0][idx] = gate_scale[source];
      s_scale[1][idx] = up_scale[source];
    }
    for (int idx = threadIdx.x;
         idx < 4 * (QWEN35_TC_ALT_OUT_TILE / 8); idx += blockDim.x) {
      const int group_local = idx / (QWEN35_TC_ALT_OUT_TILE / 8);
      const int row_pack_local = idx % (QWEN35_TC_ALT_OUT_TILE / 8);
      const int64_t source =
          static_cast<int64_t>(out_base / 8 + row_pack_local) * groups +
          tile_group + group_local;
      s_zp[0][idx] = static_cast<uint32_t>(gate_zero_point[source]);
      s_zp[1][idx] = static_cast<uint32_t>(up_zero_point[source]);
    }
    __syncthreads();

    uint32_t a0[8], a2[8];
#pragma unroll
    for (int sub = 0; sub < 8; ++sub) {
      a0[sub] = a2[sub] = 0;
      if (lane / 4 == 0) {
        a0[sub] = qwen35_pack_bf16(s_act[sub * 16 + a_col],
                                   s_act[sub * 16 + a_col + 1]);
        a2[sub] = qwen35_pack_bf16(s_act[sub * 16 + a_col + 8],
                                   s_act[sub * 16 + a_col + 9]);
      }
    }
#pragma unroll
    for (int sub = 0; sub < 8; ++sub) {
      const int group_local = sub / 2;
      const int base_col = tile128 + sub * 16;
      const int zp_index =
          group_local * (QWEN35_TC_ALT_OUT_TILE / 8) + warp;
      const int zp_gate = (s_zp[0][zp_index] >> (jn * 4)) & 0xf;
      const int zp_up = (s_zp[1][zp_index] >> (jn * 4)) & 0xf;
      const int scale_index =
          group_local * QWEN35_TC_ALT_OUT_TILE + warp * 8 + jn;
      const float scale_gate = __bfloat162float(s_scale[0][scale_index]);
      const float scale_up = __bfloat162float(s_scale[1][scale_index]);
      const int64_t word_base =
          static_cast<int64_t>(out_row) * packed_cols + base_col / 8;
      const uint32_t gate_w0 = static_cast<uint32_t>(gate_packed[word_base]);
      const uint32_t gate_w1 = static_cast<uint32_t>(gate_packed[word_base + 1]);
      const uint32_t up_w0 = static_cast<uint32_t>(up_packed[word_base]);
      const uint32_t up_w1 = static_cast<uint32_t>(up_packed[word_base + 1]);
      const auto fragment = [&](uint32_t w0, uint32_t w1, int zp,
                                float scale, uint32_t* b0, uint32_t* b1) {
        *b0 = qwen35_pack_bf16(
            __float2bfloat16((static_cast<int>((w0 >> ((k0 & 7) * 4)) & 0xf) - zp) * scale),
            __float2bfloat16((static_cast<int>((w0 >> (((k0 + 1) & 7) * 4)) & 0xf) - zp) * scale));
        *b1 = qwen35_pack_bf16(
            __float2bfloat16((static_cast<int>((w1 >> ((k1 & 7) * 4)) & 0xf) - zp) * scale),
            __float2bfloat16((static_cast<int>((w1 >> (((k1 + 1) & 7) * 4)) & 0xf) - zp) * scale));
      };
      uint32_t gate_b0, gate_b1, up_b0, up_b1;
      fragment(gate_w0, gate_w1, zp_gate, scale_gate, &gate_b0, &gate_b1);
      fragment(up_w0, up_w1, zp_up, scale_up, &up_b0, &up_b1);
      qwen35_mma_bf16(c_gate[(sub % 4) * 4], c_gate[(sub % 4) * 4 + 1],
                      c_gate[(sub % 4) * 4 + 2], c_gate[(sub % 4) * 4 + 3],
                      a0[sub], 0, a2[sub], 0, gate_b0, gate_b1);
      qwen35_mma_bf16(c_up[(sub % 4) * 4], c_up[(sub % 4) * 4 + 1],
                      c_up[(sub % 4) * 4 + 2], c_up[(sub % 4) * 4 + 3],
                      a0[sub], 0, a2[sub], 0, up_b0, up_b1);
    }
    __syncthreads();
  }
  if (lane < 4) {
    const int col = 2 * lane;
    const int output0 = out_base + warp * 8 + col;
    const int output1 = output0 + 1;
    const __nv_bfloat16 gate0 =
        __float2bfloat16(c_gate[0] + c_gate[4] + c_gate[8] + c_gate[12]);
    const __nv_bfloat16 gate1 =
        __float2bfloat16(c_gate[1] + c_gate[5] + c_gate[9] + c_gate[13]);
    const __nv_bfloat16 up0 =
        __float2bfloat16(c_up[0] + c_up[4] + c_up[8] + c_up[12]);
    const __nv_bfloat16 up1 =
        __float2bfloat16(c_up[1] + c_up[5] + c_up[9] + c_up[13]);
    if (output0 < out_cols)
      output[output0] = __float2bfloat16(
          siluf_f32(__bfloat162float(gate0)) * __bfloat162float(up0));
    if (output1 < out_cols)
      output[output1] = __float2bfloat16(
          siluf_f32(__bfloat162float(gate1)) * __bfloat162float(up1));
  }
}

// Paired form of the accepted alternate tile. The combined grid routes each
// CTA to exactly one projection, then preserves the single-tile computation.
__global__ void qwen35_gemm_w4a16_bf16_tc_pair_alt_kernel(
    const __nv_bfloat16* activation, const int32_t* weight_packed0,
    const __nv_bfloat16* weight_scale0, const int32_t* weight_zero_point0,
    __nv_bfloat16* output0, int out_cols0, const int32_t* weight_packed1,
    const __nv_bfloat16* weight_scale1, const int32_t* weight_zero_point1,
    __nv_bfloat16* output1, int in_cols, int groups) {
  int output_block = blockIdx.x;
  const int first_blocks = out_cols0 / QWEN35_TC_ALT_OUT_TILE;
  const int32_t* weight_packed = weight_packed0;
  const __nv_bfloat16* weight_scale = weight_scale0;
  const int32_t* weight_zero_point = weight_zero_point0;
  __nv_bfloat16* output = output0;
  if (output_block >= first_blocks) {
    output_block -= first_blocks;
    weight_packed = weight_packed1;
    weight_scale = weight_scale1;
    weight_zero_point = weight_zero_point1;
    output = output1;
  }
  const int out_base = output_block * QWEN35_TC_ALT_OUT_TILE;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int l4 = lane % 4;
  const int jn = lane / 4;
  const int out_row = out_base + warp * 8 + jn;
  const int packed_cols = (in_cols + 7) / 8;
  const int a_col = 2 * l4;
  const int k0 = 2 * l4;
  const int k1 = k0 + 8;
  __shared__ __nv_bfloat16 s_act[128];
  __shared__ __nv_bfloat16 s_scale[4 * QWEN35_TC_ALT_OUT_TILE];
  __shared__ uint32_t s_zp[4 * (QWEN35_TC_ALT_OUT_TILE / 8)];
  float c[16] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  for (int tile128 = 0; tile128 < in_cols; tile128 += 128) {
    for (int idx = threadIdx.x; idx < 128; idx += blockDim.x)
      s_act[idx] = activation[tile128 + idx];

    const int tile_group = tile128 / 32;
    for (int idx = threadIdx.x; idx < 4 * QWEN35_TC_ALT_OUT_TILE;
         idx += blockDim.x) {
      const int group_local = idx / QWEN35_TC_ALT_OUT_TILE;
      const int row_local = idx % QWEN35_TC_ALT_OUT_TILE;
      s_scale[idx] = weight_scale[
          static_cast<int64_t>(out_base + row_local) * groups +
          tile_group + group_local];
    }
    for (int idx = threadIdx.x;
         idx < 4 * (QWEN35_TC_ALT_OUT_TILE / 8); idx += blockDim.x) {
      const int group_local = idx / (QWEN35_TC_ALT_OUT_TILE / 8);
      const int row_pack_local = idx % (QWEN35_TC_ALT_OUT_TILE / 8);
      s_zp[idx] = static_cast<uint32_t>(weight_zero_point[
          static_cast<int64_t>(out_base / 8 + row_pack_local) * groups +
          tile_group + group_local]);
    }
    __syncthreads();

    uint32_t a0[8], a2[8];
#pragma unroll
    for (int sub = 0; sub < 8; sub++) {
      a0[sub] = 0;
      a2[sub] = 0;
      if (lane / 4 == 0) {
        a0[sub] = qwen35_pack_bf16(s_act[sub * 16 + a_col],
                                   s_act[sub * 16 + a_col + 1]);
        a2[sub] = qwen35_pack_bf16(s_act[sub * 16 + a_col + 8],
                                   s_act[sub * 16 + a_col + 9]);
      }
    }
#pragma unroll
    for (int sub = 0; sub < 8; sub++) {
      const int group_local = sub / 2;
      const int base_col = tile128 + sub * 16;
      const uint32_t zp_word =
          s_zp[group_local * (QWEN35_TC_ALT_OUT_TILE / 8) + warp];
      const int zp = static_cast<int>((zp_word >> (jn * 4)) & 0xFu);
      const float scale = __bfloat162float(
          s_scale[group_local * QWEN35_TC_ALT_OUT_TILE + warp * 8 + jn]);
      const uint32_t w0 = static_cast<uint32_t>(weight_packed[
          static_cast<int64_t>(out_row) * packed_cols + (base_col + k0) / 8]);
      const uint32_t w1 = static_cast<uint32_t>(weight_packed[
          static_cast<int64_t>(out_row) * packed_cols + (base_col + k1) / 8]);
      const int q00 = static_cast<int>((w0 >> ((k0 & 7) * 4)) & 0xFu);
      const int q01 =
          static_cast<int>((w0 >> (((k0 + 1) & 7) * 4)) & 0xFu);
      const int q10 = static_cast<int>((w1 >> ((k1 & 7) * 4)) & 0xFu);
      const int q11 =
          static_cast<int>((w1 >> (((k1 + 1) & 7) * 4)) & 0xFu);
      const uint32_t b0 = qwen35_pack_bf16(
          __float2bfloat16(static_cast<float>(q00 - zp) * scale),
          __float2bfloat16(static_cast<float>(q01 - zp) * scale));
      const uint32_t b1 = qwen35_pack_bf16(
          __float2bfloat16(static_cast<float>(q10 - zp) * scale),
          __float2bfloat16(static_cast<float>(q11 - zp) * scale));
      qwen35_mma_bf16(c[(sub % 4) * 4], c[(sub % 4) * 4 + 1],
                      c[(sub % 4) * 4 + 2], c[(sub % 4) * 4 + 3],
                      a0[sub], 0, a2[sub], 0, b0, b1);
    }
    __syncthreads();
  }
  if (lane < 4) {
    const int col = 2 * lane;
    const float r0 = c[0] + c[4] + c[8] + c[12];
    const float r1 = c[1] + c[5] + c[9] + c[13];
    output[out_base + warp * 8 + col] = __float2bfloat16(r0);
    output[out_base + warp * 8 + col + 1] = __float2bfloat16(r1);
  }
}

// Paired raw-layout decode candidate with the same metadata staging and
// arithmetic schedule as the single-projection shared-metadata kernel. Each
// CTA selects exactly one projection/output tile, then stages that tile's
// scale and zero-point metadata once before consuming it in K order.
__global__ void qwen35_gemm_w4a16_bf16_tc_pair_meta_kernel(
    const __nv_bfloat16* activation, const int32_t* weight_packed0,
    const __nv_bfloat16* weight_scale0, const int32_t* weight_zero_point0,
    __nv_bfloat16* output0, int out_cols0, const int32_t* weight_packed1,
    const __nv_bfloat16* weight_scale1, const int32_t* weight_zero_point1,
    __nv_bfloat16* output1, int in_cols, int groups) {
  int output_block = blockIdx.x;
  const int first_blocks = out_cols0 / QWEN35_TC_OUT_TILE;
  const int32_t* weight_packed = weight_packed0;
  const __nv_bfloat16* weight_scale = weight_scale0;
  const int32_t* weight_zero_point = weight_zero_point0;
  __nv_bfloat16* output = output0;
  if (output_block >= first_blocks) {
    output_block -= first_blocks;
    weight_packed = weight_packed1;
    weight_scale = weight_scale1;
    weight_zero_point = weight_zero_point1;
    output = output1;
  }
  const int out_base = output_block * QWEN35_TC_OUT_TILE;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int l4 = lane % 4;
  const int jn = lane / 4;
  const int out_row = out_base + warp * 8 + jn;
  const int packed_cols = in_cols / 8;
  const int a_col = 2 * l4;
  const int k0 = 2 * l4;
  const int k1 = k0 + 8;
  __shared__ __nv_bfloat16 s_act[128];
  __shared__ __nv_bfloat16 s_scale[4 * QWEN35_TC_OUT_TILE];
  __shared__ uint32_t s_zp[4 * (QWEN35_TC_OUT_TILE / 8)];
  float c[16] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  for (int tile128 = 0; tile128 < in_cols; tile128 += 128) {
    for (int idx = threadIdx.x; idx < 128; idx += blockDim.x)
      s_act[idx] = activation[tile128 + idx];

    const int tile_group = tile128 / 32;
    for (int idx = threadIdx.x; idx < 4 * QWEN35_TC_OUT_TILE;
         idx += blockDim.x) {
      const int group_local = idx / QWEN35_TC_OUT_TILE;
      const int row_local = idx % QWEN35_TC_OUT_TILE;
      s_scale[idx] = weight_scale[
          static_cast<int64_t>(out_base + row_local) * groups +
          tile_group + group_local];
    }
    for (int idx = threadIdx.x;
         idx < 4 * (QWEN35_TC_OUT_TILE / 8); idx += blockDim.x) {
      const int group_local = idx / (QWEN35_TC_OUT_TILE / 8);
      const int row_pack_local = idx % (QWEN35_TC_OUT_TILE / 8);
      s_zp[idx] = static_cast<uint32_t>(weight_zero_point[
          static_cast<int64_t>(out_base / 8 + row_pack_local) * groups +
          tile_group + group_local]);
    }
    __syncthreads();

    uint32_t a0[8], a2[8];
#pragma unroll
    for (int sub = 0; sub < 8; sub++) {
      a0[sub] = 0;
      a2[sub] = 0;
      if (lane / 4 == 0) {
        a0[sub] = qwen35_pack_bf16(s_act[sub * 16 + a_col],
                                   s_act[sub * 16 + a_col + 1]);
        a2[sub] = qwen35_pack_bf16(s_act[sub * 16 + a_col + 8],
                                   s_act[sub * 16 + a_col + 9]);
      }
    }
#pragma unroll
    for (int sub = 0; sub < 8; sub++) {
      const int group_local = sub / 2;
      const int base_col = tile128 + sub * 16;
      const uint32_t zp_word =
          s_zp[group_local * (QWEN35_TC_OUT_TILE / 8) + warp];
      const int zp = static_cast<int>((zp_word >> (jn * 4)) & 0xFu);
      const float scale = __bfloat162float(
          s_scale[group_local * QWEN35_TC_OUT_TILE + warp * 8 + jn]);
      const uint32_t w0 = static_cast<uint32_t>(weight_packed[
          static_cast<int64_t>(out_row) * packed_cols + (base_col + k0) / 8]);
      const uint32_t w1 = static_cast<uint32_t>(weight_packed[
          static_cast<int64_t>(out_row) * packed_cols + (base_col + k1) / 8]);
      const int q00 = static_cast<int>((w0 >> ((k0 & 7) * 4)) & 0xFu);
      const int q01 =
          static_cast<int>((w0 >> (((k0 + 1) & 7) * 4)) & 0xFu);
      const int q10 = static_cast<int>((w1 >> ((k1 & 7) * 4)) & 0xFu);
      const int q11 =
          static_cast<int>((w1 >> (((k1 + 1) & 7) * 4)) & 0xFu);
      const uint32_t b0 = qwen35_pack_bf16(
          __float2bfloat16(static_cast<float>(q00 - zp) * scale),
          __float2bfloat16(static_cast<float>(q01 - zp) * scale));
      const uint32_t b1 = qwen35_pack_bf16(
          __float2bfloat16(static_cast<float>(q10 - zp) * scale),
          __float2bfloat16(static_cast<float>(q11 - zp) * scale));
      qwen35_mma_bf16(c[(sub % 4) * 4], c[(sub % 4) * 4 + 1],
                      c[(sub % 4) * 4 + 2], c[(sub % 4) * 4 + 3],
                      a0[sub], 0, a2[sub], 0, b0, b1);
    }
    __syncthreads();
  }
  if (lane < 4) {
    const int col = 2 * lane;
    const float r0 = c[0] + c[4] + c[8] + c[12];
    const float r1 = c[1] + c[5] + c[9] + c[13];
    output[out_base + warp * 8 + col] = __float2bfloat16(r0);
    output[out_base + warp * 8 + col + 1] = __float2bfloat16(r1);
  }
}

// APXINF_W4_WEIGHT_STAGE candidate. Each CTA selects one projection and
// coalesces the raw 64-row by K=128 packed-weight tile plus its exact BF16
// scales and packed zero points into shared memory. Shared indexing preserves
// the original raw row/word layout; dequantization, MMA, and reduction order
// remain identical to the baseline paired kernel.
__global__ void qwen35_gemm_w4a16_bf16_tc_pair_weight_stage_kernel(
    const __nv_bfloat16* activation, const int32_t* weight_packed0,
    const __nv_bfloat16* weight_scale0, const int32_t* weight_zero_point0,
    __nv_bfloat16* output0, int out_cols0, const int32_t* weight_packed1,
    const __nv_bfloat16* weight_scale1, const int32_t* weight_zero_point1,
    __nv_bfloat16* output1, int in_cols, int groups) {
  int output_block = blockIdx.x;
  const int first_blocks = out_cols0 / QWEN35_TC_OUT_TILE;
  const int32_t* weight_packed = weight_packed0;
  const __nv_bfloat16* weight_scale = weight_scale0;
  const int32_t* weight_zero_point = weight_zero_point0;
  __nv_bfloat16* output = output0;
  if (output_block >= first_blocks) {
    output_block -= first_blocks;
    weight_packed = weight_packed1;
    weight_scale = weight_scale1;
    weight_zero_point = weight_zero_point1;
    output = output1;
  }
  const int out_base = output_block * QWEN35_TC_OUT_TILE;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int l4 = lane % 4;
  const int jn = lane / 4;
  const int row_local = warp * 8 + jn;
  const int packed_cols = in_cols / 8;
  const int a_col = 2 * l4;
  const int k0 = 2 * l4;
  const int k1 = k0 + 8;
  __shared__ __nv_bfloat16 s_act[128];
  __shared__ uint32_t s_weight[QWEN35_TC_OUT_TILE * 16];
  __shared__ __nv_bfloat16 s_scale[4 * QWEN35_TC_OUT_TILE];
  __shared__ uint32_t s_zp[4 * (QWEN35_TC_OUT_TILE / 8)];
  float c[16] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  for (int tile128 = 0; tile128 < in_cols; tile128 += 128) {
    for (int idx = threadIdx.x; idx < 128; idx += blockDim.x)
      s_act[idx] = activation[tile128 + idx];

    const int packed_tile_base = tile128 / 8;
    for (int idx = threadIdx.x; idx < QWEN35_TC_OUT_TILE * 16;
         idx += blockDim.x) {
      const int staged_row = idx / 16;
      const int staged_word = idx % 16;
      s_weight[idx] = static_cast<uint32_t>(weight_packed[
          static_cast<int64_t>(out_base + staged_row) * packed_cols +
          packed_tile_base + staged_word]);
    }

    const int tile_group = tile128 / 32;
    for (int idx = threadIdx.x; idx < 4 * QWEN35_TC_OUT_TILE;
         idx += blockDim.x) {
      const int group_local = idx / QWEN35_TC_OUT_TILE;
      const int staged_row = idx % QWEN35_TC_OUT_TILE;
      s_scale[idx] = weight_scale[
          static_cast<int64_t>(out_base + staged_row) * groups +
          tile_group + group_local];
    }
    for (int idx = threadIdx.x;
         idx < 4 * (QWEN35_TC_OUT_TILE / 8); idx += blockDim.x) {
      const int group_local = idx / (QWEN35_TC_OUT_TILE / 8);
      const int row_pack_local = idx % (QWEN35_TC_OUT_TILE / 8);
      s_zp[idx] = static_cast<uint32_t>(weight_zero_point[
          static_cast<int64_t>(out_base / 8 + row_pack_local) * groups +
          tile_group + group_local]);
    }
    __syncthreads();

    uint32_t a0[8], a2[8];
#pragma unroll
    for (int sub = 0; sub < 8; sub++) {
      a0[sub] = 0;
      a2[sub] = 0;
      if (lane / 4 == 0) {
        a0[sub] = qwen35_pack_bf16(s_act[sub * 16 + a_col],
                                   s_act[sub * 16 + a_col + 1]);
        a2[sub] = qwen35_pack_bf16(s_act[sub * 16 + a_col + 8],
                                   s_act[sub * 16 + a_col + 9]);
      }
    }
#pragma unroll
    for (int sub = 0; sub < 8; sub++) {
      const int group_local = sub / 2;
      const uint32_t zp_word_leader =
          l4 == 0 ? s_zp[group_local * (QWEN35_TC_OUT_TILE / 8) + warp]
                  : 0u;
      const uint32_t zp_word =
          __shfl_sync(0xffffffffu, zp_word_leader, lane - l4);
      const int zp = static_cast<int>((zp_word >> (jn * 4)) & 0xFu);
      const float scale_leader =
          l4 == 0 ? __bfloat162float(
                         s_scale[group_local * QWEN35_TC_OUT_TILE + row_local])
                  : 0.0f;
      const float scale =
          __shfl_sync(0xffffffffu, scale_leader, lane - l4);
      const uint32_t w0_leader =
          l4 == 0 ? s_weight[row_local * 16 + sub * 2] : 0u;
      const uint32_t w1_leader =
          l4 == 0 ? s_weight[row_local * 16 + sub * 2 + 1] : 0u;
      const uint32_t w0 =
          __shfl_sync(0xffffffffu, w0_leader, lane - l4);
      const uint32_t w1 =
          __shfl_sync(0xffffffffu, w1_leader, lane - l4);
      const int q00 = static_cast<int>((w0 >> ((k0 & 7) * 4)) & 0xFu);
      const int q01 =
          static_cast<int>((w0 >> (((k0 + 1) & 7) * 4)) & 0xFu);
      const int q10 = static_cast<int>((w1 >> ((k1 & 7) * 4)) & 0xFu);
      const int q11 =
          static_cast<int>((w1 >> (((k1 + 1) & 7) * 4)) & 0xFu);
      const uint32_t b0 = qwen35_pack_bf16(
          __float2bfloat16(static_cast<float>(q00 - zp) * scale),
          __float2bfloat16(static_cast<float>(q01 - zp) * scale));
      const uint32_t b1 = qwen35_pack_bf16(
          __float2bfloat16(static_cast<float>(q10 - zp) * scale),
          __float2bfloat16(static_cast<float>(q11 - zp) * scale));
      qwen35_mma_bf16(c[(sub % 4) * 4], c[(sub % 4) * 4 + 1],
                      c[(sub % 4) * 4 + 2], c[(sub % 4) * 4 + 3],
                      a0[sub], 0, a2[sub], 0, b0, b1);
    }
    __syncthreads();
  }
  if (lane < 4) {
    const int col = 2 * lane;
    const float r0 = c[0] + c[4] + c[8] + c[12];
    const float r1 = c[1] + c[5] + c[9] + c[13];
    output[out_base + warp * 8 + col] = __float2bfloat16(r0);
    output[out_base + warp * 8 + col + 1] = __float2bfloat16(r1);
  }
}

// Exact raw in_proj_qkv decode for N=10240, K=5120, group-32. Four warps
// compute 64 rows while cooperatively staging the complete raw 64x128 weight
// slice. Each warp owns two eight-row MMA tiles; BF16 dequantization, K order,
// accumulator layout, and final reduction match the established kernel.
__global__ __launch_bounds__(128, 2)
void qwen35_gemm_w4a16_bf16_qkv_10240x5120_kernel(
    const __nv_bfloat16* activation, const int32_t* weight_packed,
    const __nv_bfloat16* weight_scale, const int32_t* weight_zero_point,
    __nv_bfloat16* output) {
  constexpr int kIn = 5120, kGroups = 160, kPacked = 640;
  const int out_base = blockIdx.x * 64;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int l4 = lane % 4, jn = lane / 4;
  const int row0 = warp * 16 + jn, row1 = row0 + 8;
  const int k0 = 2 * l4, k1 = k0 + 8, a_col = k0;
  __shared__ __nv_bfloat16 s_act[128];
  __shared__ uint32_t s_weight[64 * 16];
  __shared__ __nv_bfloat16 s_scale[4 * 64];
  __shared__ uint32_t s_zp[4 * 8];
  float c0[16] = {}, c1[16] = {};
  for (int tile = 0; tile < kIn; tile += 128) {
    s_act[threadIdx.x] = activation[tile + threadIdx.x];
    for (int idx = threadIdx.x; idx < 64 * 16; idx += 128)
      s_weight[idx] = static_cast<uint32_t>(weight_packed[
          static_cast<int64_t>(out_base + idx / 16) * kPacked + tile / 8 + idx % 16]);
    for (int idx = threadIdx.x; idx < 4 * 64; idx += 128)
      s_scale[idx] = weight_scale[
          static_cast<int64_t>(out_base + idx % 64) * kGroups + tile / 32 + idx / 64];
    for (int idx = threadIdx.x; idx < 4 * 8; idx += 128)
      s_zp[idx] = static_cast<uint32_t>(weight_zero_point[
          static_cast<int64_t>(out_base / 8 + idx % 8) * kGroups + tile / 32 + idx / 8]);
    __syncthreads();
    uint32_t a0[8], a2[8];
#pragma unroll
    for (int sub = 0; sub < 8; ++sub) {
      a0[sub] = a2[sub] = 0;
      if (jn == 0) {
        a0[sub] = qwen35_pack_bf16(s_act[sub * 16 + a_col], s_act[sub * 16 + a_col + 1]);
        a2[sub] = qwen35_pack_bf16(s_act[sub * 16 + a_col + 8], s_act[sub * 16 + a_col + 9]);
      }
    }
#pragma unroll
    for (int sub = 0; sub < 8; ++sub) {
      const int g = sub / 2;
      const int zp0 = (s_zp[g * 8 + row0 / 8] >> (jn * 4)) & 0xf;
      const int zp1 = (s_zp[g * 8 + row1 / 8] >> (jn * 4)) & 0xf;
      const float sc0 = __bfloat162float(s_scale[g * 64 + row0]);
      const float sc1 = __bfloat162float(s_scale[g * 64 + row1]);
      const uint32_t w00 = s_weight[row0 * 16 + sub * 2];
      const uint32_t w01 = s_weight[row0 * 16 + sub * 2 + 1];
      const uint32_t w10 = s_weight[row1 * 16 + sub * 2];
      const uint32_t w11 = s_weight[row1 * 16 + sub * 2 + 1];
      const uint32_t b00 = qwen35_pack_bf16(
          __float2bfloat16((static_cast<int>((w00 >> (k0 * 4)) & 0xf) - zp0) * sc0),
          __float2bfloat16((static_cast<int>((w00 >> ((k0 + 1) * 4)) & 0xf) - zp0) * sc0));
      const uint32_t b01 = qwen35_pack_bf16(
          __float2bfloat16((static_cast<int>((w01 >> ((k1 & 7) * 4)) & 0xf) - zp0) * sc0),
          __float2bfloat16((static_cast<int>((w01 >> (((k1 + 1) & 7) * 4)) & 0xf) - zp0) * sc0));
      const uint32_t b10 = qwen35_pack_bf16(
          __float2bfloat16((static_cast<int>((w10 >> (k0 * 4)) & 0xf) - zp1) * sc1),
          __float2bfloat16((static_cast<int>((w10 >> ((k0 + 1) * 4)) & 0xf) - zp1) * sc1));
      const uint32_t b11 = qwen35_pack_bf16(
          __float2bfloat16((static_cast<int>((w11 >> ((k1 & 7) * 4)) & 0xf) - zp1) * sc1),
          __float2bfloat16((static_cast<int>((w11 >> (((k1 + 1) & 7) * 4)) & 0xf) - zp1) * sc1));
      qwen35_mma_bf16(c0[(sub % 4) * 4], c0[(sub % 4) * 4 + 1], c0[(sub % 4) * 4 + 2], c0[(sub % 4) * 4 + 3], a0[sub], 0, a2[sub], 0, b00, b01);
      qwen35_mma_bf16(c1[(sub % 4) * 4], c1[(sub % 4) * 4 + 1], c1[(sub % 4) * 4 + 2], c1[(sub % 4) * 4 + 3], a0[sub], 0, a2[sub], 0, b10, b11);
    }
    __syncthreads();
  }
  if (lane < 4) {
    const int col = 2 * lane;
    output[out_base + warp * 16 + col] = __float2bfloat16(c0[0] + c0[4] + c0[8] + c0[12]);
    output[out_base + warp * 16 + col + 1] = __float2bfloat16(c0[1] + c0[5] + c0[9] + c0[13]);
    output[out_base + warp * 16 + 8 + col] = __float2bfloat16(c1[0] + c1[4] + c1[8] + c1[12]);
    output[out_base + warp * 16 + 8 + col + 1] = __float2bfloat16(c1[1] + c1[5] + c1[9] + c1[13]);
  }
}

// Iteration-33 exact QKV scheduling family. Per-output arithmetic is identical
// to qwen35_gemm_w4a16_bf16_qkv_10240x5120_kernel; only CTA row ownership and
// activation/weight-metadata staging change. Rows=64 -> 4 warps/128 threads;
// Rows=128 -> 8 warps/256 threads. ActStages and MetaStages are independently
// one or two so the complete scheduling matrix can be exact-gated.
template <int Rows, int ActStages, int MetaStages>
__global__ void qwen35_gemm_w4a16_bf16_qkv_sched_kernel(
    const __nv_bfloat16* activation, const int32_t* weight_packed,
    const __nv_bfloat16* weight_scale, const int32_t* weight_zero_point,
    __nv_bfloat16* output) {
  static_assert(Rows == 64 || Rows == 128);
  static_assert(ActStages == 1 || ActStages == 2);
  static_assert(MetaStages == 1 || MetaStages == 2);
  constexpr int kIn = 5120, kOut = 10240, kGroups = 160, kPacked = 640;
  constexpr int kThreads = Rows * 2;
  constexpr int kWarps = Rows / 16;
  const int out_base = blockIdx.x * Rows;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int l4 = lane % 4, jn = lane / 4;
  const int row0 = warp * 16 + jn, row1 = row0 + 8;
  const int k0 = 2 * l4, k1 = k0 + 8, a_col = k0;
  __shared__ __align__(16) __nv_bfloat16 s_act[ActStages][128];
  __shared__ uint32_t s_weight[MetaStages][Rows * 16];
  __shared__ __nv_bfloat16 s_scale[MetaStages][4 * Rows];
  __shared__ uint32_t s_zp[MetaStages][4 * (Rows / 8)];
  float c0[16] = {}, c1[16] = {};

  if constexpr (ActStages == 2) {
    if (threadIdx.x < 16) {
      const int offset = threadIdx.x * 8;
      const uint32_t dst = static_cast<uint32_t>(
          __cvta_generic_to_shared(&s_act[0][offset]));
      asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::
                   "r"(dst), "l"(activation + offset));
    }
    asm volatile("cp.async.commit_group;\n" ::);
  } else {
    for (int idx = threadIdx.x; idx < 128; idx += kThreads)
      s_act[0][idx] = activation[idx];
  }
  for (int idx = threadIdx.x; idx < Rows * 16; idx += kThreads)
    s_weight[0][idx] = static_cast<uint32_t>(weight_packed[
        static_cast<int64_t>(out_base + idx / 16) * kPacked + idx % 16]);
  for (int idx = threadIdx.x; idx < 4 * Rows; idx += kThreads)
    s_scale[0][idx] = weight_scale[
        static_cast<int64_t>(out_base + idx % Rows) * kGroups + idx / Rows];
  for (int idx = threadIdx.x; idx < 4 * (Rows / 8); idx += kThreads)
    s_zp[0][idx] = static_cast<uint32_t>(weight_zero_point[
        static_cast<int64_t>(out_base / 8 + idx % (Rows / 8)) * kGroups +
        idx / (Rows / 8)]);
  if constexpr (ActStages == 2)
    asm volatile("cp.async.wait_group 0;\n" ::);
  __syncthreads();

  constexpr int kTiles = kIn / 128;
  for (int tile_index = 0; tile_index < kTiles; ++tile_index) {
    const int tile = tile_index * 128;
    const int act_stage = ActStages == 2 ? tile_index & 1 : 0;
    const int meta_stage = MetaStages == 2 ? tile_index & 1 : 0;
    const bool has_next = tile_index + 1 < kTiles;
    const int next_tile = tile + 128;
    const int next_act_stage = ActStages == 2 ? act_stage ^ 1 : 0;
    const int next_meta_stage = MetaStages == 2 ? meta_stage ^ 1 : 0;

    if constexpr (ActStages == 2) {
      if (has_next && threadIdx.x < 16) {
        const int offset = threadIdx.x * 8;
        const uint32_t dst = static_cast<uint32_t>(
            __cvta_generic_to_shared(&s_act[next_act_stage][offset]));
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::
                     "r"(dst), "l"(activation + next_tile + offset));
      }
      if (has_next) asm volatile("cp.async.commit_group;\n" ::);
    }
    if constexpr (MetaStages == 2) {
      if (has_next) {
        for (int idx = threadIdx.x; idx < Rows * 16; idx += kThreads)
          s_weight[next_meta_stage][idx] = static_cast<uint32_t>(weight_packed[
              static_cast<int64_t>(out_base + idx / 16) * kPacked +
              next_tile / 8 + idx % 16]);
        for (int idx = threadIdx.x; idx < 4 * Rows; idx += kThreads)
          s_scale[next_meta_stage][idx] = weight_scale[
              static_cast<int64_t>(out_base + idx % Rows) * kGroups +
              next_tile / 32 + idx / Rows];
        for (int idx = threadIdx.x; idx < 4 * (Rows / 8); idx += kThreads)
          s_zp[next_meta_stage][idx] = static_cast<uint32_t>(weight_zero_point[
              static_cast<int64_t>(out_base / 8 + idx % (Rows / 8)) * kGroups +
              next_tile / 32 + idx / (Rows / 8)]);
      }
    }

    uint32_t a0[8], a2[8];
#pragma unroll
    for (int sub = 0; sub < 8; ++sub) {
      a0[sub] = a2[sub] = 0;
      if (jn == 0) {
        a0[sub] = qwen35_pack_bf16(
            s_act[act_stage][sub * 16 + a_col],
            s_act[act_stage][sub * 16 + a_col + 1]);
        a2[sub] = qwen35_pack_bf16(
            s_act[act_stage][sub * 16 + a_col + 8],
            s_act[act_stage][sub * 16 + a_col + 9]);
      }
    }
#pragma unroll
    for (int sub = 0; sub < 8; ++sub) {
      const int group = sub / 2;
      const int zp0 = (s_zp[meta_stage][group * (Rows / 8) + row0 / 8] >>
                       (jn * 4)) & 0xf;
      const int zp1 = (s_zp[meta_stage][group * (Rows / 8) + row1 / 8] >>
                       (jn * 4)) & 0xf;
      const float sc0 = __bfloat162float(
          s_scale[meta_stage][group * Rows + row0]);
      const float sc1 = __bfloat162float(
          s_scale[meta_stage][group * Rows + row1]);
      const uint32_t w00 = s_weight[meta_stage][row0 * 16 + sub * 2];
      const uint32_t w01 = s_weight[meta_stage][row0 * 16 + sub * 2 + 1];
      const uint32_t w10 = s_weight[meta_stage][row1 * 16 + sub * 2];
      const uint32_t w11 = s_weight[meta_stage][row1 * 16 + sub * 2 + 1];
      const uint32_t b00 = qwen35_pack_bf16(
          __float2bfloat16((static_cast<int>((w00 >> (k0 * 4)) & 0xf) - zp0) * sc0),
          __float2bfloat16((static_cast<int>((w00 >> ((k0 + 1) * 4)) & 0xf) - zp0) * sc0));
      const uint32_t b01 = qwen35_pack_bf16(
          __float2bfloat16((static_cast<int>((w01 >> ((k1 & 7) * 4)) & 0xf) - zp0) * sc0),
          __float2bfloat16((static_cast<int>((w01 >> (((k1 + 1) & 7) * 4)) & 0xf) - zp0) * sc0));
      const uint32_t b10 = qwen35_pack_bf16(
          __float2bfloat16((static_cast<int>((w10 >> (k0 * 4)) & 0xf) - zp1) * sc1),
          __float2bfloat16((static_cast<int>((w10 >> ((k0 + 1) * 4)) & 0xf) - zp1) * sc1));
      const uint32_t b11 = qwen35_pack_bf16(
          __float2bfloat16((static_cast<int>((w11 >> ((k1 & 7) * 4)) & 0xf) - zp1) * sc1),
          __float2bfloat16((static_cast<int>((w11 >> (((k1 + 1) & 7) * 4)) & 0xf) - zp1) * sc1));
      qwen35_mma_bf16(c0[(sub % 4) * 4], c0[(sub % 4) * 4 + 1],
                      c0[(sub % 4) * 4 + 2], c0[(sub % 4) * 4 + 3],
                      a0[sub], 0, a2[sub], 0, b00, b01);
      qwen35_mma_bf16(c1[(sub % 4) * 4], c1[(sub % 4) * 4 + 1],
                      c1[(sub % 4) * 4 + 2], c1[(sub % 4) * 4 + 3],
                      a0[sub], 0, a2[sub], 0, b10, b11);
    }

    if (has_next) {
      if constexpr (ActStages == 1 || MetaStages == 1)
        __syncthreads();
      if constexpr (ActStages == 2) {
        asm volatile("cp.async.wait_group 0;\n" ::);
      } else {
        for (int idx = threadIdx.x; idx < 128; idx += kThreads)
          s_act[0][idx] = activation[next_tile + idx];
      }
      if constexpr (MetaStages == 1) {
        for (int idx = threadIdx.x; idx < Rows * 16; idx += kThreads)
          s_weight[0][idx] = static_cast<uint32_t>(weight_packed[
              static_cast<int64_t>(out_base + idx / 16) * kPacked +
              next_tile / 8 + idx % 16]);
        for (int idx = threadIdx.x; idx < 4 * Rows; idx += kThreads)
          s_scale[0][idx] = weight_scale[
              static_cast<int64_t>(out_base + idx % Rows) * kGroups +
              next_tile / 32 + idx / Rows];
        for (int idx = threadIdx.x; idx < 4 * (Rows / 8); idx += kThreads)
          s_zp[0][idx] = static_cast<uint32_t>(weight_zero_point[
              static_cast<int64_t>(out_base / 8 + idx % (Rows / 8)) * kGroups +
              next_tile / 32 + idx / (Rows / 8)]);
      }
      __syncthreads();
    }
  }

  if (lane < 4) {
    const int col = 2 * lane;
    output[out_base + warp * 16 + col] =
        __float2bfloat16(c0[0] + c0[4] + c0[8] + c0[12]);
    output[out_base + warp * 16 + col + 1] =
        __float2bfloat16(c0[1] + c0[5] + c0[9] + c0[13]);
    output[out_base + warp * 16 + 8 + col] =
        __float2bfloat16(c1[0] + c1[4] + c1[8] + c1[12]);
    output[out_base + warp * 16 + 8 + col + 1] =
        __float2bfloat16(c1[1] + c1[5] + c1[9] + c1[13]);
  }
}

// APXINF_W4_PAIR_PREFETCH candidate. Two shared-memory stages retain the
// baseline activation and metadata values while SM80 cp.async fetches the
// next activation tile. Metadata is double-buffered with exact-type loads.
__global__ void qwen35_gemm_w4a16_bf16_tc_pair_prefetch_kernel(
    const __nv_bfloat16* activation, const int32_t* weight_packed0,
    const __nv_bfloat16* weight_scale0, const int32_t* weight_zero_point0,
    __nv_bfloat16* output0, int out_cols0, const int32_t* weight_packed1,
    const __nv_bfloat16* weight_scale1, const int32_t* weight_zero_point1,
    __nv_bfloat16* output1, int in_cols, int groups) {
  int output_block = blockIdx.x;
  const int first_blocks = out_cols0 / QWEN35_TC_OUT_TILE;
  const int32_t* weight_packed = weight_packed0;
  const __nv_bfloat16* weight_scale = weight_scale0;
  const int32_t* weight_zero_point = weight_zero_point0;
  __nv_bfloat16* output = output0;
  if (output_block >= first_blocks) {
    output_block -= first_blocks;
    weight_packed = weight_packed1;
    weight_scale = weight_scale1;
    weight_zero_point = weight_zero_point1;
    output = output1;
  }
  const int out_base = output_block * QWEN35_TC_OUT_TILE;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int l4 = lane % 4;
  const int jn = lane / 4;
  const int out_row = out_base + warp * 8 + jn;
  const int packed_cols = in_cols / 8;
  const int a_col = 2 * l4;
  const int k0 = 2 * l4;
  const int k1 = k0 + 8;
  __shared__ __align__(16) __nv_bfloat16 s_act[2][128];
  __shared__ __nv_bfloat16 s_scale[2][4 * QWEN35_TC_OUT_TILE];
  __shared__ uint32_t s_zp[2][4 * (QWEN35_TC_OUT_TILE / 8)];
  float c[16] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  if (threadIdx.x < 16) {
    const int offset = threadIdx.x * 8;
    const uint32_t dst = static_cast<uint32_t>(
        __cvta_generic_to_shared(&s_act[0][offset]));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::
                 "r"(dst), "l"(activation + offset));
  }
  asm volatile("cp.async.commit_group;\n" ::);
  for (int idx = threadIdx.x; idx < 4 * QWEN35_TC_OUT_TILE;
       idx += blockDim.x) {
    const int group_local = idx / QWEN35_TC_OUT_TILE;
    const int row_local = idx % QWEN35_TC_OUT_TILE;
    s_scale[0][idx] = weight_scale[
        static_cast<int64_t>(out_base + row_local) * groups + group_local];
  }
  for (int idx = threadIdx.x; idx < 4 * (QWEN35_TC_OUT_TILE / 8);
       idx += blockDim.x) {
    const int group_local = idx / (QWEN35_TC_OUT_TILE / 8);
    const int row_pack_local = idx % (QWEN35_TC_OUT_TILE / 8);
    s_zp[0][idx] = static_cast<uint32_t>(weight_zero_point[
        static_cast<int64_t>(out_base / 8 + row_pack_local) * groups +
        group_local]);
  }
  asm volatile("cp.async.wait_group 0;\n" ::);
  __syncthreads();
  int stage = 0;
  for (int tile128 = 0; tile128 < in_cols; tile128 += 128) {
    const int next_tile = tile128 + 128;
    const bool has_next = next_tile < in_cols;
    const int next_stage = stage ^ 1;
    if (has_next) {
      if (threadIdx.x < 16) {
        const int offset = threadIdx.x * 8;
        const uint32_t dst = static_cast<uint32_t>(
            __cvta_generic_to_shared(&s_act[next_stage][offset]));
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::
                     "r"(dst), "l"(activation + next_tile + offset));
      }
      asm volatile("cp.async.commit_group;\n" ::);
      const int next_group = next_tile / 32;
      for (int idx = threadIdx.x; idx < 4 * QWEN35_TC_OUT_TILE;
           idx += blockDim.x) {
        const int group_local = idx / QWEN35_TC_OUT_TILE;
        const int row_local = idx % QWEN35_TC_OUT_TILE;
        s_scale[next_stage][idx] = weight_scale[
            static_cast<int64_t>(out_base + row_local) * groups + next_group +
            group_local];
      }
      for (int idx = threadIdx.x; idx < 4 * (QWEN35_TC_OUT_TILE / 8);
           idx += blockDim.x) {
        const int group_local = idx / (QWEN35_TC_OUT_TILE / 8);
        const int row_pack_local = idx % (QWEN35_TC_OUT_TILE / 8);
        s_zp[next_stage][idx] = static_cast<uint32_t>(weight_zero_point[
            static_cast<int64_t>(out_base / 8 + row_pack_local) * groups +
            next_group + group_local]);
      }
    }
    uint32_t a0[8], a2[8];
#pragma unroll
    for (int sub = 0; sub < 8; sub++) {
      a0[sub] = 0;
      a2[sub] = 0;
      if (lane / 4 == 0) {
        a0[sub] = qwen35_pack_bf16(s_act[stage][sub * 16 + a_col],
                                   s_act[stage][sub * 16 + a_col + 1]);
        a2[sub] = qwen35_pack_bf16(s_act[stage][sub * 16 + a_col + 8],
                                   s_act[stage][sub * 16 + a_col + 9]);
      }
    }
#pragma unroll
    for (int sub = 0; sub < 8; sub++) {
      const int group_local = sub / 2;
      const int base_col = tile128 + sub * 16;
      const uint32_t zp_word =
          s_zp[stage][group_local * (QWEN35_TC_OUT_TILE / 8) + warp];
      const int zp = static_cast<int>((zp_word >> (jn * 4)) & 0xFu);
      const float scale = __bfloat162float(
          s_scale[stage][group_local * QWEN35_TC_OUT_TILE + warp * 8 + jn]);
      const uint32_t w0 = static_cast<uint32_t>(weight_packed[
          static_cast<int64_t>(out_row) * packed_cols + (base_col + k0) / 8]);
      const uint32_t w1 = static_cast<uint32_t>(weight_packed[
          static_cast<int64_t>(out_row) * packed_cols + (base_col + k1) / 8]);
      const int q00 = static_cast<int>((w0 >> ((k0 & 7) * 4)) & 0xFu);
      const int q01 = static_cast<int>((w0 >> (((k0 + 1) & 7) * 4)) & 0xFu);
      const int q10 = static_cast<int>((w1 >> ((k1 & 7) * 4)) & 0xFu);
      const int q11 = static_cast<int>((w1 >> (((k1 + 1) & 7) * 4)) & 0xFu);
      const uint32_t b0 = qwen35_pack_bf16(
          __float2bfloat16(static_cast<float>(q00 - zp) * scale),
          __float2bfloat16(static_cast<float>(q01 - zp) * scale));
      const uint32_t b1 = qwen35_pack_bf16(
          __float2bfloat16(static_cast<float>(q10 - zp) * scale),
          __float2bfloat16(static_cast<float>(q11 - zp) * scale));
      qwen35_mma_bf16(c[(sub % 4) * 4], c[(sub % 4) * 4 + 1],
                      c[(sub % 4) * 4 + 2], c[(sub % 4) * 4 + 3],
                      a0[sub], 0, a2[sub], 0, b0, b1);
    }
    if (has_next) asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();
    stage = next_stage;
  }
  if (lane < 4) {
    const int col = 2 * lane;
    const float r0 = c[0] + c[4] + c[8] + c[12];
    const float r1 = c[1] + c[5] + c[9] + c[13];
    output[out_base + warp * 8 + col] = __float2bfloat16(r0);
    output[out_base + warp * 8 + col + 1] = __float2bfloat16(r1);
  }
}

// Six-warp paired raw-layout decode. Each CTA covers 48 output rows as six
// independent eight-row warp tiles. A final partial CTA keeps the same MMA
// schedule and masks only rows beyond the selected projection's extent.
__global__ void qwen35_gemm_w4a16_bf16_tc_pair_6w_kernel(
    const __nv_bfloat16* activation, const int32_t* weight_packed0,
    const __nv_bfloat16* weight_scale0, const int32_t* weight_zero_point0,
    __nv_bfloat16* output0, int out_cols0, const int32_t* weight_packed1,
    const __nv_bfloat16* weight_scale1, const int32_t* weight_zero_point1,
    __nv_bfloat16* output1, int out_cols1, int in_cols, int groups) {
  int output_block = blockIdx.x;
  const int first_blocks =
      (out_cols0 + QWEN35_TC_PAIR_6W_OUT_TILE - 1) /
      QWEN35_TC_PAIR_6W_OUT_TILE;
  const int32_t* weight_packed = weight_packed0;
  const __nv_bfloat16* weight_scale = weight_scale0;
  const int32_t* weight_zero_point = weight_zero_point0;
  __nv_bfloat16* output = output0;
  int out_cols = out_cols0;
  if (output_block >= first_blocks) {
    output_block -= first_blocks;
    weight_packed = weight_packed1;
    weight_scale = weight_scale1;
    weight_zero_point = weight_zero_point1;
    output = output1;
    out_cols = out_cols1;
  }
  const int out_base = output_block * QWEN35_TC_PAIR_6W_OUT_TILE;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int l4 = lane % 4;
  const int jn = lane / 4;
  const int out_row = out_base + warp * 8 + jn;
  const bool active_row = out_row < out_cols;
  const int packed_cols = in_cols / 8;
  const int a_col = 2 * l4;
  const int k0 = 2 * l4;
  const int k1 = k0 + 8;
  __shared__ __nv_bfloat16 s_act[128];
  __shared__ __nv_bfloat16 s_scale[4 * QWEN35_TC_PAIR_6W_OUT_TILE];
  __shared__ uint32_t s_zp[4 * (QWEN35_TC_PAIR_6W_OUT_TILE / 8)];
  float c[16] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  for (int tile128 = 0; tile128 < in_cols; tile128 += 128) {
    for (int idx = threadIdx.x; idx < 128; idx += blockDim.x)
      s_act[idx] = activation[tile128 + idx];

    const int tile_group = tile128 / 32;
    for (int idx = threadIdx.x;
         idx < 4 * QWEN35_TC_PAIR_6W_OUT_TILE; idx += blockDim.x) {
      const int group_local = idx / QWEN35_TC_PAIR_6W_OUT_TILE;
      const int row_local = idx % QWEN35_TC_PAIR_6W_OUT_TILE;
      const int row = out_base + row_local;
      s_scale[idx] = row < out_cols
          ? weight_scale[static_cast<int64_t>(row) * groups + tile_group +
                         group_local]
          : __float2bfloat16(0.0f);
    }
    for (int idx = threadIdx.x;
         idx < 4 * (QWEN35_TC_PAIR_6W_OUT_TILE / 8); idx += blockDim.x) {
      const int group_local = idx / (QWEN35_TC_PAIR_6W_OUT_TILE / 8);
      const int row_pack_local = idx % (QWEN35_TC_PAIR_6W_OUT_TILE / 8);
      const int row = out_base + row_pack_local * 8;
      s_zp[idx] = row < out_cols
          ? static_cast<uint32_t>(weight_zero_point[
                static_cast<int64_t>(row / 8) * groups + tile_group +
                group_local])
          : 0u;
    }
    __syncthreads();

    uint32_t a0[8], a2[8];
#pragma unroll
    for (int sub = 0; sub < 8; sub++) {
      a0[sub] = 0;
      a2[sub] = 0;
      if (lane / 4 == 0) {
        a0[sub] = qwen35_pack_bf16(s_act[sub * 16 + a_col],
                                   s_act[sub * 16 + a_col + 1]);
        a2[sub] = qwen35_pack_bf16(s_act[sub * 16 + a_col + 8],
                                   s_act[sub * 16 + a_col + 9]);
      }
    }
#pragma unroll
    for (int sub = 0; sub < 8; sub++) {
      const int group_local = sub / 2;
      const int base_col = tile128 + sub * 16;
      const uint32_t zp_word = s_zp[
          group_local * (QWEN35_TC_PAIR_6W_OUT_TILE / 8) + warp];
      const int zp = static_cast<int>((zp_word >> (jn * 4)) & 0xFu);
      const float scale = __bfloat162float(s_scale[
          group_local * QWEN35_TC_PAIR_6W_OUT_TILE + warp * 8 + jn]);
      uint32_t w0 = 0;
      uint32_t w1 = 0;
      if (active_row) {
        w0 = static_cast<uint32_t>(weight_packed[
            static_cast<int64_t>(out_row) * packed_cols +
            (base_col + k0) / 8]);
        w1 = static_cast<uint32_t>(weight_packed[
            static_cast<int64_t>(out_row) * packed_cols +
            (base_col + k1) / 8]);
      }
      const int q00 = static_cast<int>((w0 >> ((k0 & 7) * 4)) & 0xFu);
      const int q01 =
          static_cast<int>((w0 >> (((k0 + 1) & 7) * 4)) & 0xFu);
      const int q10 = static_cast<int>((w1 >> ((k1 & 7) * 4)) & 0xFu);
      const int q11 =
          static_cast<int>((w1 >> (((k1 + 1) & 7) * 4)) & 0xFu);
      const uint32_t b0 = qwen35_pack_bf16(
          __float2bfloat16(static_cast<float>(q00 - zp) * scale),
          __float2bfloat16(static_cast<float>(q01 - zp) * scale));
      const uint32_t b1 = qwen35_pack_bf16(
          __float2bfloat16(static_cast<float>(q10 - zp) * scale),
          __float2bfloat16(static_cast<float>(q11 - zp) * scale));
      qwen35_mma_bf16(c[(sub % 4) * 4], c[(sub % 4) * 4 + 1],
                      c[(sub % 4) * 4 + 2], c[(sub % 4) * 4 + 3],
                      a0[sub], 0, a2[sub], 0, b0, b1);
    }
    __syncthreads();
  }
  if (active_row && lane < 4) {
    const int col = 2 * lane;
    const float r0 = c[0] + c[4] + c[8] + c[12];
    const float r1 = c[1] + c[5] + c[9] + c[13];
    output[out_base + warp * 8 + col] = __float2bfloat16(r0);
    output[out_base + warp * 8 + col + 1] = __float2bfloat16(r1);
  }
}

// APXINF_W4_PAIR_WARP candidate. Four loader warps stage activation,
// metadata, and the exact lane-specific B operands for eight compute warps.
// Compute warps retain the baseline sub-tile, MMA, accumulator, and reduction
// order. The opt-in ABI restricts this larger CTA to the profiled model shapes.
__global__ void qwen35_gemm_w4a16_bf16_tc_pair_warp_kernel(
    const __nv_bfloat16* activation, const int32_t* weight_packed0,
    const __nv_bfloat16* weight_scale0, const int32_t* weight_zero_point0,
    __nv_bfloat16* output0, int out_cols0, const int32_t* weight_packed1,
    const __nv_bfloat16* weight_scale1, const int32_t* weight_zero_point1,
    __nv_bfloat16* output1, int in_cols, int groups) {
  int output_block = blockIdx.x;
  const int first_blocks = out_cols0 / QWEN35_TC_OUT_TILE;
  const int32_t* weight_packed = weight_packed0;
  const __nv_bfloat16* weight_scale = weight_scale0;
  const int32_t* weight_zero_point = weight_zero_point0;
  __nv_bfloat16* output = output0;
  if (output_block >= first_blocks) {
    output_block -= first_blocks;
    weight_packed = weight_packed1;
    weight_scale = weight_scale1;
    weight_zero_point = weight_zero_point1;
    output = output1;
  }

  const int out_base = output_block * QWEN35_TC_OUT_TILE;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const bool compute_warp = warp < 8;
  const int packed_cols = in_cols / 8;
  __shared__ __nv_bfloat16 s_act[128];
  __shared__ __nv_bfloat16 s_scale[4 * QWEN35_TC_OUT_TILE];
  __shared__ uint32_t s_zp[4 * (QWEN35_TC_OUT_TILE / 8)];
  __shared__ uint32_t s_b[8][8][32][2];
  float c[16] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  for (int tile128 = 0; tile128 < in_cols; tile128 += 128) {
    if (!compute_warp) {
      const int loader = warp - 8;
      s_act[loader * 32 + lane] = activation[tile128 + loader * 32 + lane];
      const int tile_group = tile128 / 32;
      for (int idx = loader * 32 + lane;
           idx < 4 * QWEN35_TC_OUT_TILE; idx += 128) {
        const int group_local = idx / QWEN35_TC_OUT_TILE;
        const int row_local = idx % QWEN35_TC_OUT_TILE;
        s_scale[idx] = weight_scale[
            static_cast<int64_t>(out_base + row_local) * groups +
            tile_group + group_local];
      }
      if (loader == 0) {
        const int group_local = lane / (QWEN35_TC_OUT_TILE / 8);
        const int row_pack_local = lane % (QWEN35_TC_OUT_TILE / 8);
        s_zp[lane] = static_cast<uint32_t>(weight_zero_point[
            static_cast<int64_t>(out_base / 8 + row_pack_local) * groups +
            tile_group + group_local]);
      }
    }
    __syncthreads();

    if (!compute_warp) {
      const int loader = warp - 8;
#pragma unroll
      for (int assigned = 0; assigned < 2; ++assigned) {
        const int output_warp = loader + assigned * 4;
        const int l4 = lane % 4;
        const int jn = lane / 4;
        const int out_row = out_base + output_warp * 8 + jn;
        const int k0 = 2 * l4;
        const int k1 = k0 + 8;
#pragma unroll
        for (int sub = 0; sub < 8; ++sub) {
          const int group_local = sub / 2;
          const int base_col = tile128 + sub * 16;
          const uint32_t zp_word =
              s_zp[group_local * (QWEN35_TC_OUT_TILE / 8) + output_warp];
          const int zp = static_cast<int>((zp_word >> (jn * 4)) & 0xFu);
          const float scale = __bfloat162float(s_scale[
              group_local * QWEN35_TC_OUT_TILE + output_warp * 8 + jn]);
          const uint32_t w0 = static_cast<uint32_t>(weight_packed[
              static_cast<int64_t>(out_row) * packed_cols +
              (base_col + k0) / 8]);
          const uint32_t w1 = static_cast<uint32_t>(weight_packed[
              static_cast<int64_t>(out_row) * packed_cols +
              (base_col + k1) / 8]);
          const int q00 = static_cast<int>((w0 >> ((k0 & 7) * 4)) & 0xFu);
          const int q01 =
              static_cast<int>((w0 >> (((k0 + 1) & 7) * 4)) & 0xFu);
          const int q10 = static_cast<int>((w1 >> ((k1 & 7) * 4)) & 0xFu);
          const int q11 =
              static_cast<int>((w1 >> (((k1 + 1) & 7) * 4)) & 0xFu);
          s_b[output_warp][sub][lane][0] = qwen35_pack_bf16(
              __float2bfloat16(static_cast<float>(q00 - zp) * scale),
              __float2bfloat16(static_cast<float>(q01 - zp) * scale));
          s_b[output_warp][sub][lane][1] = qwen35_pack_bf16(
              __float2bfloat16(static_cast<float>(q10 - zp) * scale),
              __float2bfloat16(static_cast<float>(q11 - zp) * scale));
        }
      }
    }
    __syncthreads();

    if (compute_warp) {
      const int a_col = 2 * (lane % 4);
#pragma unroll
      for (int sub = 0; sub < 8; ++sub) {
        uint32_t a0 = 0;
        uint32_t a2 = 0;
        if (lane / 4 == 0) {
          a0 = qwen35_pack_bf16(s_act[sub * 16 + a_col],
                                s_act[sub * 16 + a_col + 1]);
          a2 = qwen35_pack_bf16(s_act[sub * 16 + a_col + 8],
                                s_act[sub * 16 + a_col + 9]);
        }
        qwen35_mma_bf16(c[(sub % 4) * 4], c[(sub % 4) * 4 + 1],
                        c[(sub % 4) * 4 + 2], c[(sub % 4) * 4 + 3],
                        a0, 0, a2, 0, s_b[warp][sub][lane][0],
                        s_b[warp][sub][lane][1]);
      }
    }
    __syncthreads();
  }
  if (compute_warp && lane < 4) {
    const int col = 2 * lane;
    const float r0 = c[0] + c[4] + c[8] + c[12];
    const float r1 = c[1] + c[5] + c[9] + c[13];
    output[out_base + warp * 8 + col] = __float2bfloat16(r0);
    output[out_base + warp * 8 + col + 1] = __float2bfloat16(r1);
  }
}

// Paired raw-layout decode with double-buffered activation and metadata
// staging. Vector loads preserve the BF16 activation bits, while alternating
// shared slots remove the post-consume overwrite barrier between K tiles.
__global__ void qwen35_gemm_w4a16_bf16_tc_pair_act_kernel(
    const __nv_bfloat16* activation, const int32_t* weight_packed0,
    const __nv_bfloat16* weight_scale0, const int32_t* weight_zero_point0,
    __nv_bfloat16* output0, int out_cols0, const int32_t* weight_packed1,
    const __nv_bfloat16* weight_scale1, const int32_t* weight_zero_point1,
    __nv_bfloat16* output1, int in_cols, int groups) {
  int output_block = blockIdx.x;
  const int first_blocks = out_cols0 / QWEN35_TC_OUT_TILE;
  const int32_t* weight_packed = weight_packed0;
  const __nv_bfloat16* weight_scale = weight_scale0;
  const int32_t* weight_zero_point = weight_zero_point0;
  __nv_bfloat16* output = output0;
  if (output_block >= first_blocks) {
    output_block -= first_blocks;
    weight_packed = weight_packed1;
    weight_scale = weight_scale1;
    weight_zero_point = weight_zero_point1;
    output = output1;
  }
  const int out_base = output_block * QWEN35_TC_OUT_TILE;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int l4 = lane % 4;
  const int jn = lane / 4;
  const int out_row = out_base + warp * 8 + jn;
  const int packed_cols = in_cols / 8;
  const int a_col = 2 * l4;
  const int k0 = 2 * l4;
  const int k1 = k0 + 8;
  __shared__ uint4 s_act_vec[2][16];
  __shared__ __nv_bfloat16 s_scale[2][4 * QWEN35_TC_OUT_TILE];
  __shared__ uint32_t s_zp[2][4 * (QWEN35_TC_OUT_TILE / 8)];
  float c[16] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  // Prime slot zero. The launch gate guarantees complete K=128 tiles and
  // group size 32, so every staged vector and metadata entry is in bounds.
  if (threadIdx.x < 16)
    s_act_vec[0][threadIdx.x] =
        reinterpret_cast<const uint4*>(activation)[threadIdx.x];
  for (int idx = threadIdx.x; idx < 4 * QWEN35_TC_OUT_TILE;
       idx += blockDim.x) {
    const int group_local = idx / QWEN35_TC_OUT_TILE;
    const int row_local = idx % QWEN35_TC_OUT_TILE;
    s_scale[0][idx] = weight_scale[
        static_cast<int64_t>(out_base + row_local) * groups + group_local];
  }
  for (int idx = threadIdx.x; idx < 4 * (QWEN35_TC_OUT_TILE / 8);
       idx += blockDim.x) {
    const int group_local = idx / (QWEN35_TC_OUT_TILE / 8);
    const int row_pack_local = idx % (QWEN35_TC_OUT_TILE / 8);
    s_zp[0][idx] = static_cast<uint32_t>(weight_zero_point[
        static_cast<int64_t>(out_base / 8 + row_pack_local) * groups +
        group_local]);
  }
  __syncthreads();

  const int tile_count = in_cols / 128;
  for (int tile = 0; tile < tile_count; ++tile) {
    const int slot = tile & 1;
    const int tile128 = tile * 128;
    const __nv_bfloat16* s_act =
        reinterpret_cast<const __nv_bfloat16*>(s_act_vec[slot]);
    uint32_t a0[8], a2[8];
#pragma unroll
    for (int sub = 0; sub < 8; sub++) {
      a0[sub] = 0;
      a2[sub] = 0;
      if (lane / 4 == 0) {
        a0[sub] = qwen35_pack_bf16(s_act[sub * 16 + a_col],
                                   s_act[sub * 16 + a_col + 1]);
        a2[sub] = qwen35_pack_bf16(s_act[sub * 16 + a_col + 8],
                                   s_act[sub * 16 + a_col + 9]);
      }
    }
#pragma unroll
    for (int sub = 0; sub < 8; sub++) {
      const int group_local = sub / 2;
      const int base_col = tile128 + sub * 16;
      const uint32_t zp_word =
          s_zp[slot][group_local * (QWEN35_TC_OUT_TILE / 8) + warp];
      const int zp = static_cast<int>((zp_word >> (jn * 4)) & 0xFu);
      const float scale = __bfloat162float(
          s_scale[slot][group_local * QWEN35_TC_OUT_TILE + warp * 8 + jn]);
      const uint32_t w0 = static_cast<uint32_t>(weight_packed[
          static_cast<int64_t>(out_row) * packed_cols + (base_col + k0) / 8]);
      const uint32_t w1 = static_cast<uint32_t>(weight_packed[
          static_cast<int64_t>(out_row) * packed_cols + (base_col + k1) / 8]);
      const int q00 = static_cast<int>((w0 >> ((k0 & 7) * 4)) & 0xFu);
      const int q01 =
          static_cast<int>((w0 >> (((k0 + 1) & 7) * 4)) & 0xFu);
      const int q10 = static_cast<int>((w1 >> ((k1 & 7) * 4)) & 0xFu);
      const int q11 =
          static_cast<int>((w1 >> (((k1 + 1) & 7) * 4)) & 0xFu);
      const uint32_t b0 = qwen35_pack_bf16(
          __float2bfloat16(static_cast<float>(q00 - zp) * scale),
          __float2bfloat16(static_cast<float>(q01 - zp) * scale));
      const uint32_t b1 = qwen35_pack_bf16(
          __float2bfloat16(static_cast<float>(q10 - zp) * scale),
          __float2bfloat16(static_cast<float>(q11 - zp) * scale));
      qwen35_mma_bf16(c[(sub % 4) * 4], c[(sub % 4) * 4 + 1],
                      c[(sub % 4) * 4 + 2], c[(sub % 4) * 4 + 3],
                      a0[sub], 0, a2[sub], 0, b0, b1);
    }

    if (tile + 1 < tile_count) {
      const int next_slot = slot ^ 1;
      const int next_tile128 = tile128 + 128;
      const int next_tile_group = next_tile128 / 32;
      if (threadIdx.x < 16)
        s_act_vec[next_slot][threadIdx.x] =
            reinterpret_cast<const uint4*>(activation + next_tile128)[threadIdx.x];
      for (int idx = threadIdx.x; idx < 4 * QWEN35_TC_OUT_TILE;
           idx += blockDim.x) {
        const int group_local = idx / QWEN35_TC_OUT_TILE;
        const int row_local = idx % QWEN35_TC_OUT_TILE;
        s_scale[next_slot][idx] = weight_scale[
            static_cast<int64_t>(out_base + row_local) * groups +
            next_tile_group + group_local];
      }
      for (int idx = threadIdx.x; idx < 4 * (QWEN35_TC_OUT_TILE / 8);
           idx += blockDim.x) {
        const int group_local = idx / (QWEN35_TC_OUT_TILE / 8);
        const int row_pack_local = idx % (QWEN35_TC_OUT_TILE / 8);
        s_zp[next_slot][idx] = static_cast<uint32_t>(weight_zero_point[
            static_cast<int64_t>(out_base / 8 + row_pack_local) * groups +
            next_tile_group + group_local]);
      }
      __syncthreads();
    }
  }
  if (lane < 4) {
    const int col = 2 * lane;
    const float r0 = c[0] + c[4] + c[8] + c[12];
    const float r1 = c[1] + c[5] + c[9] + c[13];
    output[out_base + warp * 8 + col] = __float2bfloat16(r0);
    output[out_base + warp * 8 + col + 1] = __float2bfloat16(r1);
  }
}

// Paired raw-layout candidate using the read-only cache for immutable inputs.
// Metadata staging, dequantization, MMA order, accumulator layout, reduction,
// and output routing are identical to qwen35_gemm_w4a16_bf16_tc_pair_meta_kernel.
__global__ void qwen35_gemm_w4a16_bf16_tc_pair_cache_kernel(
    const __nv_bfloat16* activation, const int32_t* weight_packed0,
    const __nv_bfloat16* weight_scale0, const int32_t* weight_zero_point0,
    __nv_bfloat16* output0, int out_cols0, const int32_t* weight_packed1,
    const __nv_bfloat16* weight_scale1, const int32_t* weight_zero_point1,
    __nv_bfloat16* output1, int in_cols, int groups) {
  int output_block = blockIdx.x;
  const int first_blocks = out_cols0 / QWEN35_TC_OUT_TILE;
  const int32_t* weight_packed = weight_packed0;
  const __nv_bfloat16* weight_scale = weight_scale0;
  const int32_t* weight_zero_point = weight_zero_point0;
  __nv_bfloat16* output = output0;
  if (output_block >= first_blocks) {
    output_block -= first_blocks;
    weight_packed = weight_packed1;
    weight_scale = weight_scale1;
    weight_zero_point = weight_zero_point1;
    output = output1;
  }
  const int out_base = output_block * QWEN35_TC_OUT_TILE;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int l4 = lane % 4;
  const int jn = lane / 4;
  const int out_row = out_base + warp * 8 + jn;
  const int packed_cols = in_cols / 8;
  const int a_col = 2 * l4;
  const int k0 = 2 * l4;
  const int k1 = k0 + 8;
  __shared__ __nv_bfloat16 s_act[128];
  __shared__ __nv_bfloat16 s_scale[4 * QWEN35_TC_OUT_TILE];
  __shared__ uint32_t s_zp[4 * (QWEN35_TC_OUT_TILE / 8)];
  float c[16] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  for (int tile128 = 0; tile128 < in_cols; tile128 += 128) {
    for (int idx = threadIdx.x; idx < 128; idx += blockDim.x)
      s_act[idx] = __ldg(activation + tile128 + idx);

    const int tile_group = tile128 / 32;
    for (int idx = threadIdx.x; idx < 4 * QWEN35_TC_OUT_TILE;
         idx += blockDim.x) {
      const int group_local = idx / QWEN35_TC_OUT_TILE;
      const int row_local = idx % QWEN35_TC_OUT_TILE;
      s_scale[idx] = __ldg(
          weight_scale + static_cast<int64_t>(out_base + row_local) * groups +
          tile_group + group_local);
    }
    for (int idx = threadIdx.x;
         idx < 4 * (QWEN35_TC_OUT_TILE / 8); idx += blockDim.x) {
      const int group_local = idx / (QWEN35_TC_OUT_TILE / 8);
      const int row_pack_local = idx % (QWEN35_TC_OUT_TILE / 8);
      s_zp[idx] = static_cast<uint32_t>(__ldg(
          weight_zero_point +
          static_cast<int64_t>(out_base / 8 + row_pack_local) * groups +
          tile_group + group_local));
    }
    __syncthreads();

    uint32_t a0[8], a2[8];
#pragma unroll
    for (int sub = 0; sub < 8; sub++) {
      a0[sub] = 0;
      a2[sub] = 0;
      if (lane / 4 == 0) {
        a0[sub] = qwen35_pack_bf16(s_act[sub * 16 + a_col],
                                   s_act[sub * 16 + a_col + 1]);
        a2[sub] = qwen35_pack_bf16(s_act[sub * 16 + a_col + 8],
                                   s_act[sub * 16 + a_col + 9]);
      }
    }
#pragma unroll
    for (int sub = 0; sub < 8; sub++) {
      const int group_local = sub / 2;
      const int base_col = tile128 + sub * 16;
      const uint32_t zp_word =
          s_zp[group_local * (QWEN35_TC_OUT_TILE / 8) + warp];
      const int zp = static_cast<int>((zp_word >> (jn * 4)) & 0xFu);
      const float scale = __bfloat162float(
          s_scale[group_local * QWEN35_TC_OUT_TILE + warp * 8 + jn]);
      const uint32_t w0 = static_cast<uint32_t>(__ldg(
          weight_packed + static_cast<int64_t>(out_row) * packed_cols +
          (base_col + k0) / 8));
      const uint32_t w1 = static_cast<uint32_t>(__ldg(
          weight_packed + static_cast<int64_t>(out_row) * packed_cols +
          (base_col + k1) / 8));
      const int q00 = static_cast<int>((w0 >> ((k0 & 7) * 4)) & 0xFu);
      const int q01 =
          static_cast<int>((w0 >> (((k0 + 1) & 7) * 4)) & 0xFu);
      const int q10 = static_cast<int>((w1 >> ((k1 & 7) * 4)) & 0xFu);
      const int q11 =
          static_cast<int>((w1 >> (((k1 + 1) & 7) * 4)) & 0xFu);
      const uint32_t b0 = qwen35_pack_bf16(
          __float2bfloat16(static_cast<float>(q00 - zp) * scale),
          __float2bfloat16(static_cast<float>(q01 - zp) * scale));
      const uint32_t b1 = qwen35_pack_bf16(
          __float2bfloat16(static_cast<float>(q10 - zp) * scale),
          __float2bfloat16(static_cast<float>(q11 - zp) * scale));
      qwen35_mma_bf16(c[(sub % 4) * 4], c[(sub % 4) * 4 + 1],
                      c[(sub % 4) * 4 + 2], c[(sub % 4) * 4 + 3],
                      a0[sub], 0, a2[sub], 0, b0, b1);
    }
    __syncthreads();
  }
  if (lane < 4) {
    const int col = 2 * lane;
    const float r0 = c[0] + c[4] + c[8] + c[12];
    const float r1 = c[1] + c[5] + c[9] + c[13];
    output[out_base + warp * 8 + col] = __float2bfloat16(r0);
    output[out_base + warp * 8 + col + 1] = __float2bfloat16(r1);
  }
}

// Paired raw-layout decode candidate using two warps to compute 16 output
// rows per CTA. Per-row dequantization, K-ordered MMA accumulation, and final
// reduction are identical to the shared-metadata paired kernel above.
__global__ void qwen35_gemm_w4a16_bf16_tc_pair_2w_kernel(
    const __nv_bfloat16* activation, const int32_t* weight_packed0,
    const __nv_bfloat16* weight_scale0, const int32_t* weight_zero_point0,
    __nv_bfloat16* output0, int out_cols0, const int32_t* weight_packed1,
    const __nv_bfloat16* weight_scale1, const int32_t* weight_zero_point1,
    __nv_bfloat16* output1, int in_cols, int groups) {
  int output_block = blockIdx.x;
  const int first_blocks = out_cols0 / QWEN35_TC_PAIR_2W_OUT_TILE;
  const int32_t* weight_packed = weight_packed0;
  const __nv_bfloat16* weight_scale = weight_scale0;
  const int32_t* weight_zero_point = weight_zero_point0;
  __nv_bfloat16* output = output0;
  if (output_block >= first_blocks) {
    output_block -= first_blocks;
    weight_packed = weight_packed1;
    weight_scale = weight_scale1;
    weight_zero_point = weight_zero_point1;
    output = output1;
  }
  const int out_base = output_block * QWEN35_TC_PAIR_2W_OUT_TILE;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int l4 = lane % 4;
  const int jn = lane / 4;
  const int out_row = out_base + warp * 8 + jn;
  const int packed_cols = in_cols / 8;
  const int a_col = 2 * l4;
  const int k0 = 2 * l4;
  const int k1 = k0 + 8;
  __shared__ __nv_bfloat16 s_act[128];
  __shared__ __nv_bfloat16 s_scale[4 * QWEN35_TC_PAIR_2W_OUT_TILE];
  __shared__ uint32_t s_zp[4 * (QWEN35_TC_PAIR_2W_OUT_TILE / 8)];
  float c[16] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  for (int tile128 = 0; tile128 < in_cols; tile128 += 128) {
    for (int idx = threadIdx.x; idx < 128; idx += blockDim.x)
      s_act[idx] = activation[tile128 + idx];

    const int tile_group = tile128 / 32;
    for (int idx = threadIdx.x; idx < 4 * QWEN35_TC_PAIR_2W_OUT_TILE;
         idx += blockDim.x) {
      const int group_local = idx / QWEN35_TC_PAIR_2W_OUT_TILE;
      const int row_local = idx % QWEN35_TC_PAIR_2W_OUT_TILE;
      s_scale[idx] = weight_scale[
          static_cast<int64_t>(out_base + row_local) * groups +
          tile_group + group_local];
    }
    for (int idx = threadIdx.x;
         idx < 4 * (QWEN35_TC_PAIR_2W_OUT_TILE / 8); idx += blockDim.x) {
      const int group_local = idx / (QWEN35_TC_PAIR_2W_OUT_TILE / 8);
      const int row_pack_local = idx % (QWEN35_TC_PAIR_2W_OUT_TILE / 8);
      s_zp[idx] = static_cast<uint32_t>(weight_zero_point[
          static_cast<int64_t>(out_base / 8 + row_pack_local) * groups +
          tile_group + group_local]);
    }
    __syncthreads();

    uint32_t a0[8], a2[8];
#pragma unroll
    for (int sub = 0; sub < 8; sub++) {
      a0[sub] = 0;
      a2[sub] = 0;
      if (lane / 4 == 0) {
        a0[sub] = qwen35_pack_bf16(s_act[sub * 16 + a_col],
                                   s_act[sub * 16 + a_col + 1]);
        a2[sub] = qwen35_pack_bf16(s_act[sub * 16 + a_col + 8],
                                   s_act[sub * 16 + a_col + 9]);
      }
    }
#pragma unroll
    for (int sub = 0; sub < 8; sub++) {
      const int group_local = sub / 2;
      const int base_col = tile128 + sub * 16;
      const uint32_t zp_word =
          s_zp[group_local * (QWEN35_TC_PAIR_2W_OUT_TILE / 8) + warp];
      const int zp = static_cast<int>((zp_word >> (jn * 4)) & 0xFu);
      const float scale = __bfloat162float(
          s_scale[group_local * QWEN35_TC_PAIR_2W_OUT_TILE + warp * 8 + jn]);
      const uint32_t w0 = static_cast<uint32_t>(weight_packed[
          static_cast<int64_t>(out_row) * packed_cols + (base_col + k0) / 8]);
      const uint32_t w1 = static_cast<uint32_t>(weight_packed[
          static_cast<int64_t>(out_row) * packed_cols + (base_col + k1) / 8]);
      const int q00 = static_cast<int>((w0 >> ((k0 & 7) * 4)) & 0xFu);
      const int q01 =
          static_cast<int>((w0 >> (((k0 + 1) & 7) * 4)) & 0xFu);
      const int q10 = static_cast<int>((w1 >> ((k1 & 7) * 4)) & 0xFu);
      const int q11 =
          static_cast<int>((w1 >> (((k1 + 1) & 7) * 4)) & 0xFu);
      const uint32_t b0 = qwen35_pack_bf16(
          __float2bfloat16(static_cast<float>(q00 - zp) * scale),
          __float2bfloat16(static_cast<float>(q01 - zp) * scale));
      const uint32_t b1 = qwen35_pack_bf16(
          __float2bfloat16(static_cast<float>(q10 - zp) * scale),
          __float2bfloat16(static_cast<float>(q11 - zp) * scale));
      qwen35_mma_bf16(c[(sub % 4) * 4], c[(sub % 4) * 4 + 1],
                      c[(sub % 4) * 4 + 2], c[(sub % 4) * 4 + 3],
                      a0[sub], 0, a2[sub], 0, b0, b1);
    }
    __syncthreads();
  }
  if (lane < 4) {
    const int col = 2 * lane;
    const float r0 = c[0] + c[4] + c[8] + c[12];
    const float r1 = c[1] + c[5] + c[9] + c[13];
    output[out_base + warp * 8 + col] = __float2bfloat16(r0);
    output[out_base + warp * 8 + col + 1] = __float2bfloat16(r1);
  }
}

// Raw-layout decode candidate that coarsens two 64-column output tiles onto
// each CTA. Every tile retains the baseline kernel's BF16 dequantization,
// K-ordered MMA sequence, accumulator layout, and final reduction.
__global__ void qwen35_gemm_w4a16_bf16_tc_persistent_kernel(
    const __nv_bfloat16* activation, const int32_t* weight_packed,
    const __nv_bfloat16* weight_scale, const int32_t* weight_zero_point,
    __nv_bfloat16* output, int in_cols, int out_cols, int groups) {
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int group_size = in_cols / groups;
  const int packed_cols = in_cols / 8;
  __shared__ __nv_bfloat16 s_act[128];
  const int a_col = 2 * (lane % 4);
  const int l4 = lane % 4;
  const int jn = lane / 4;
  const int k0 = 2 * l4;
  const int k1 = k0 + 8;
  const int output_blocks = out_cols / QWEN35_TC_OUT_TILE;

  for (int output_block = blockIdx.x; output_block < output_blocks;
       output_block += gridDim.x) {
    const int out_base = output_block * QWEN35_TC_OUT_TILE;
    const int out_row = out_base + warp * 8 + jn;
    float c[16] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                   0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int tile128 = 0; tile128 < in_cols; tile128 += 128) {
      for (int idx = threadIdx.x; idx < 128; idx += 256)
        s_act[idx] = activation[tile128 + idx];
      __syncthreads();
      uint32_t a0[8], a2[8];
#pragma unroll
      for (int sub = 0; sub < 8; sub++) {
        a0[sub] = 0;
        a2[sub] = 0;
        if (lane / 4 == 0) {
          a0[sub] = qwen35_pack_bf16(s_act[sub * 16 + a_col],
                                     s_act[sub * 16 + a_col + 1]);
          a2[sub] = qwen35_pack_bf16(s_act[sub * 16 + a_col + 8],
                                     s_act[sub * 16 + a_col + 9]);
        }
      }
#pragma unroll
      for (int sub = 0; sub < 8; sub++) {
        const int group = (tile128 + sub * 16) / group_size;
        const int base_col = tile128 + sub * 16;
        const uint32_t zp_word = static_cast<uint32_t>(weight_zero_point[
            static_cast<int64_t>(out_row / 8) * groups + group]);
        const int zp =
            static_cast<int>((zp_word >> ((out_row & 7) * 4)) & 0xFu);
        const float scale = __bfloat162float(
            weight_scale[static_cast<int64_t>(out_row) * groups + group]);
        const uint32_t w0 = static_cast<uint32_t>(weight_packed[
            static_cast<int64_t>(out_row) * packed_cols +
            (base_col + k0) / 8]);
        const uint32_t w1 = static_cast<uint32_t>(weight_packed[
            static_cast<int64_t>(out_row) * packed_cols +
            (base_col + k1) / 8]);
        const int q00 = static_cast<int>((w0 >> ((k0 & 7) * 4)) & 0xFu);
        const int q01 =
            static_cast<int>((w0 >> (((k0 + 1) & 7) * 4)) & 0xFu);
        const int q10 = static_cast<int>((w1 >> ((k1 & 7) * 4)) & 0xFu);
        const int q11 =
            static_cast<int>((w1 >> (((k1 + 1) & 7) * 4)) & 0xFu);
        const uint32_t b0 = qwen35_pack_bf16(
            __float2bfloat16(static_cast<float>(q00 - zp) * scale),
            __float2bfloat16(static_cast<float>(q01 - zp) * scale));
        const uint32_t b1 = qwen35_pack_bf16(
            __float2bfloat16(static_cast<float>(q10 - zp) * scale),
            __float2bfloat16(static_cast<float>(q11 - zp) * scale));
        qwen35_mma_bf16(c[(sub % 4) * 4], c[(sub % 4) * 4 + 1],
                        c[(sub % 4) * 4 + 2], c[(sub % 4) * 4 + 3],
                        a0[sub], 0, a2[sub], 0, b0, b1);
      }
      __syncthreads();
    }
    if (lane < 4) {
      const int col = 2 * lane;
      const float r0 = c[0] + c[4] + c[8] + c[12];
      const float r1 = c[1] + c[5] + c[9] + c[13];
      output[out_base + warp * 8 + col] = __float2bfloat16(r0);
      output[out_base + warp * 8 + col + 1] = __float2bfloat16(r1);
    }
    __syncthreads();
  }
}

// W4_REPACKED_N64_K16_V1 decode ABI. Physical indices are
// qword[n64][k16][row64][word2], scale[n64][group][row64], and
// zero_point[n64][group][rowpack8]. One subgroup leader loads the metadata
// shared by four MMA lanes; shuffles preserve the exact BF16 operands and MMA
// instruction/accumulator order of the raw compressed-tensors kernel above.
__global__ void qwen35_gemm_w4a16_bf16_tc_repacked_v1_kernel(
    const __nv_bfloat16* activation, const int32_t* weight_packed,
    const __nv_bfloat16* weight_scale, const int32_t* weight_zero_point,
    __nv_bfloat16* output, int in_cols, int out_cols, int groups) {
  const int output_block = blockIdx.x;
  const int out_base = output_block * QWEN35_TC_OUT_TILE;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int l4 = lane % 4;
  const int jn = lane / 4;
  const int out_row = out_base + warp * 8 + jn;
  const bool active = out_row < out_cols;
  const int row64 = out_row & 63;
  const int n64 = out_row / 64;
  const int k_tiles = in_cols / 16;
  const int group_size = in_cols / groups;
  __shared__ __nv_bfloat16 s_act[128];
  const int a_col = 2 * l4;
  const int k0 = 2 * l4;
  const int k1 = k0 + 8;
  float c[16] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  for (int tile128 = 0; tile128 < in_cols; tile128 += 128) {
    for (int idx = threadIdx.x; idx < 128; idx += blockDim.x)
      s_act[idx] = activation[tile128 + idx];
    __syncthreads();
    uint32_t a0[8], a2[8];
#pragma unroll
    for (int sub = 0; sub < 8; sub++) {
      a0[sub] = 0;
      a2[sub] = 0;
      if (lane / 4 == 0) {
        a0[sub] = qwen35_pack_bf16(s_act[sub * 16 + a_col],
                                   s_act[sub * 16 + a_col + 1]);
        a2[sub] = qwen35_pack_bf16(s_act[sub * 16 + a_col + 8],
                                   s_act[sub * 16 + a_col + 9]);
      }
    }
#pragma unroll
    for (int sub = 0; sub < 8; sub++) {
      const int base_col = tile128 + sub * 16;
      const int group = base_col / group_size;
      uint32_t w0 = 0;
      uint32_t w1 = 0;
      uint32_t zp_word = 0;
      uint32_t scale_bits = 0;
      if (l4 == 0 && active) {
        const int64_t word_base =
            ((static_cast<int64_t>(n64) * k_tiles + base_col / 16) * 64 +
             row64) * 2;
        w0 = static_cast<uint32_t>(weight_packed[word_base]);
        w1 = static_cast<uint32_t>(weight_packed[word_base + 1]);
        zp_word = static_cast<uint32_t>(weight_zero_point[
            (static_cast<int64_t>(n64) * groups + group) * 8 + row64 / 8]);
        scale_bits = static_cast<uint32_t>(__nv_bfloat16_raw(weight_scale[
            (static_cast<int64_t>(n64) * groups + group) * 64 + row64]).x);
      }
      w0 = __shfl_sync(0xffffffffu, w0, lane - l4);
      w1 = __shfl_sync(0xffffffffu, w1, lane - l4);
      zp_word = __shfl_sync(0xffffffffu, zp_word, lane - l4);
      scale_bits = __shfl_sync(0xffffffffu, scale_bits, lane - l4);
      const int zp = static_cast<int>((zp_word >> ((row64 & 7) * 4)) & 0xFu);
      const float scale = __bfloat162float(
          __ushort_as_bfloat16(static_cast<unsigned short>(scale_bits)));
      const int q00 = static_cast<int>((w0 >> ((k0 & 7) * 4)) & 0xFu);
      const int q01 = static_cast<int>((w0 >> (((k0 + 1) & 7) * 4)) & 0xFu);
      const int q10 = static_cast<int>((w1 >> ((k1 & 7) * 4)) & 0xFu);
      const int q11 = static_cast<int>((w1 >> (((k1 + 1) & 7) * 4)) & 0xFu);
      const uint32_t b0 = qwen35_pack_bf16(
          __float2bfloat16(static_cast<float>(q00 - zp) * scale),
          __float2bfloat16(static_cast<float>(q01 - zp) * scale));
      const uint32_t b1 = qwen35_pack_bf16(
          __float2bfloat16(static_cast<float>(q10 - zp) * scale),
          __float2bfloat16(static_cast<float>(q11 - zp) * scale));
      qwen35_mma_bf16(c[(sub % 4) * 4], c[(sub % 4) * 4 + 1],
                      c[(sub % 4) * 4 + 2], c[(sub % 4) * 4 + 3],
                      a0[sub], 0, a2[sub], 0, b0, b1);
    }
    __syncthreads();
  }
  if (active && lane < 4) {
    const int col = 2 * lane;
    const float r0 = c[0] + c[4] + c[8] + c[12];
    const float r1 = c[1] + c[5] + c[9] + c[13];
    output[out_base + warp * 8 + col] = __float2bfloat16(r0);
    output[out_base + warp * 8 + col + 1] = __float2bfloat16(r1);
  }
}

// Routes up to four independently packed projections through one decode launch.
// Each output uses the same MMA sequence as qwen35_gemm_w4a16_bf16_tc_kernel;
// the final block is predicated so small projections such as in_proj_a/b (48
// rows) do not require padding or a different reduction.
__global__ void qwen35_gemm_w4a16_bf16_tc_multi_kernel(
    const __nv_bfloat16* activation,
    const int32_t* weight_packed0, const __nv_bfloat16* weight_scale0,
    const int32_t* weight_zero_point0, __nv_bfloat16* output0, int out_cols0,
    const int32_t* weight_packed1, const __nv_bfloat16* weight_scale1,
    const int32_t* weight_zero_point1, __nv_bfloat16* output1, int out_cols1,
    const int32_t* weight_packed2, const __nv_bfloat16* weight_scale2,
    const int32_t* weight_zero_point2, __nv_bfloat16* output2, int out_cols2,
    const int32_t* weight_packed3, const __nv_bfloat16* weight_scale3,
    const int32_t* weight_zero_point3, __nv_bfloat16* output3, int out_cols3,
    int projection_count, int in_cols, int groups) {
  int output_block = blockIdx.x;
  const int blocks0 = (out_cols0 + QWEN35_TC_OUT_TILE - 1) / QWEN35_TC_OUT_TILE;
  const int blocks1 = (out_cols1 + QWEN35_TC_OUT_TILE - 1) / QWEN35_TC_OUT_TILE;
  const int blocks2 = (out_cols2 + QWEN35_TC_OUT_TILE - 1) / QWEN35_TC_OUT_TILE;
  const int32_t* weight_packed;
  const __nv_bfloat16* weight_scale;
  const int32_t* weight_zero_point;
  __nv_bfloat16* output;
  int out_cols;
  if (output_block < blocks0) {
    weight_packed = weight_packed0;
    weight_scale = weight_scale0;
    weight_zero_point = weight_zero_point0;
    output = output0;
    out_cols = out_cols0;
  } else if ((output_block -= blocks0) < blocks1) {
    weight_packed = weight_packed1;
    weight_scale = weight_scale1;
    weight_zero_point = weight_zero_point1;
    output = output1;
    out_cols = out_cols1;
  } else if ((output_block -= blocks1) < blocks2) {
    weight_packed = weight_packed2;
    weight_scale = weight_scale2;
    weight_zero_point = weight_zero_point2;
    output = output2;
    out_cols = out_cols2;
  } else {
    output_block -= blocks2;
    weight_packed = weight_packed3;
    weight_scale = weight_scale3;
    weight_zero_point = weight_zero_point3;
    output = output3;
    out_cols = out_cols3;
  }
  (void)projection_count;

  const int out_base = output_block * QWEN35_TC_OUT_TILE;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int l4 = lane % 4;
  const int jn = lane / 4;
  const int row_local = warp * 8 + jn;
  const int out_row = out_base + row_local;
  const bool active = out_row < out_cols;
  const int packed_cols = in_cols / 8;
  const int a_col = 2 * l4;
  const int k0 = 2 * l4;
  const int k1 = k0 + 8;
  __shared__ __nv_bfloat16 s_act[128];
  __shared__ uint32_t s_weight[QWEN35_TC_OUT_TILE * 16];
  __shared__ __nv_bfloat16 s_scale[4 * QWEN35_TC_OUT_TILE];
  __shared__ uint32_t s_zp[4 * (QWEN35_TC_OUT_TILE / 8)];
  float c[16] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  for (int tile128 = 0; tile128 < in_cols; tile128 += 128) {
    for (int idx = threadIdx.x; idx < 128; idx += blockDim.x)
      s_act[idx] = activation[tile128 + idx];
    const int packed_tile_base = tile128 / 8;
    for (int idx = threadIdx.x; idx < QWEN35_TC_OUT_TILE * 16;
         idx += blockDim.x) {
      const int staged_row = idx / 16;
      const int staged_word = idx % 16;
      s_weight[idx] = out_base + staged_row < out_cols
          ? static_cast<uint32_t>(weight_packed[
                static_cast<int64_t>(out_base + staged_row) * packed_cols +
                packed_tile_base + staged_word])
          : 0u;
    }
    const int tile_group = tile128 / 32;
    for (int idx = threadIdx.x; idx < 4 * QWEN35_TC_OUT_TILE;
         idx += blockDim.x) {
      const int group_local = idx / QWEN35_TC_OUT_TILE;
      const int staged_row = idx % QWEN35_TC_OUT_TILE;
      s_scale[idx] = out_base + staged_row < out_cols
          ? weight_scale[static_cast<int64_t>(out_base + staged_row) * groups +
                         tile_group + group_local]
          : __float2bfloat16(0.0f);
    }
    for (int idx = threadIdx.x;
         idx < 4 * (QWEN35_TC_OUT_TILE / 8); idx += blockDim.x) {
      const int group_local = idx / (QWEN35_TC_OUT_TILE / 8);
      const int row_pack_local = idx % (QWEN35_TC_OUT_TILE / 8);
      s_zp[idx] = out_base + row_pack_local * 8 < out_cols
          ? static_cast<uint32_t>(weight_zero_point[
                static_cast<int64_t>(out_base / 8 + row_pack_local) * groups +
                tile_group + group_local])
          : 0u;
    }
    __syncthreads();
    uint32_t a0[8], a2[8];
#pragma unroll
    for (int sub = 0; sub < 8; sub++) {
      a0[sub] = 0;
      a2[sub] = 0;
      if (lane / 4 == 0) {
        a0[sub] = qwen35_pack_bf16(s_act[sub * 16 + a_col],
                                   s_act[sub * 16 + a_col + 1]);
        a2[sub] = qwen35_pack_bf16(s_act[sub * 16 + a_col + 8],
                                   s_act[sub * 16 + a_col + 9]);
      }
    }
#pragma unroll
    for (int sub = 0; sub < 8; sub++) {
      const int group_local = sub / 2;
      const uint32_t zp_word_leader =
          l4 == 0 ? s_zp[group_local * (QWEN35_TC_OUT_TILE / 8) + warp] : 0u;
      const uint32_t zp_word =
          __shfl_sync(0xffffffffu, zp_word_leader, lane - l4);
      const int zp = static_cast<int>((zp_word >> (jn * 4)) & 0xFu);
      const float scale_leader = l4 == 0
          ? __bfloat162float(s_scale[group_local * QWEN35_TC_OUT_TILE + row_local])
          : 0.0f;
      const float scale = __shfl_sync(0xffffffffu, scale_leader, lane - l4);
      const uint32_t w0_leader =
          l4 == 0 ? s_weight[row_local * 16 + sub * 2] : 0u;
      const uint32_t w1_leader =
          l4 == 0 ? s_weight[row_local * 16 + sub * 2 + 1] : 0u;
      const uint32_t w0 = __shfl_sync(0xffffffffu, w0_leader, lane - l4);
      const uint32_t w1 = __shfl_sync(0xffffffffu, w1_leader, lane - l4);
      const int q00 = static_cast<int>((w0 >> ((k0 & 7) * 4)) & 0xFu);
      const int q01 = static_cast<int>((w0 >> (((k0 + 1) & 7) * 4)) & 0xFu);
      const int q10 = static_cast<int>((w1 >> ((k1 & 7) * 4)) & 0xFu);
      const int q11 = static_cast<int>((w1 >> (((k1 + 1) & 7) * 4)) & 0xFu);
      const uint32_t b0 = qwen35_pack_bf16(
          __float2bfloat16(static_cast<float>(q00 - zp) * scale),
          __float2bfloat16(static_cast<float>(q01 - zp) * scale));
      const uint32_t b1 = qwen35_pack_bf16(
          __float2bfloat16(static_cast<float>(q10 - zp) * scale),
          __float2bfloat16(static_cast<float>(q11 - zp) * scale));
      qwen35_mma_bf16(c[(sub % 4) * 4], c[(sub % 4) * 4 + 1],
                      c[(sub % 4) * 4 + 2], c[(sub % 4) * 4 + 3],
                      a0[sub], 0, a2[sub], 0, b0, b1);
    }
    __syncthreads();
  }
  if (active && lane < 4) {
    const int col = 2 * lane;
    const float r0 = c[0] + c[4] + c[8] + c[12];
    const float r1 = c[1] + c[5] + c[9] + c[13];
    output[out_base + warp * 8 + col] = __float2bfloat16(r0);
    output[out_base + warp * 8 + col + 1] = __float2bfloat16(r1);
  }
}

#endif  // __CUDA_ARCH__ >= 800

// ── Full attention: sigmoid(gate) · attn ───────────────────────────────────

__global__ void qwen35_sigmoid_mul_kernel(
    const __nv_bfloat16* gate, const __nv_bfloat16* x, __nv_bfloat16* out,
    int64_t count) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    out[index] = __float2bfloat16(
        __bfloat162float(x[index]) * sigmoidf_f32(__bfloat162float(gate[index])));
  }
}

// ── Full attention: causal GQA flash prefill ───────────────────────────────
//
// q: [seq, heads, head_dim] bf16 (post q_norm + rope)
// k_cache/v_cache: [n_kv_heads, max_seq_len, head_dim] bf16
// out: [seq, heads, head_dim] bf16
// One block per (s, head); blockDim.x = head_dim; QWEN35_PREFILL_WARPS warps
// split the timesteps and merge.

#define QWEN35_PREFILL_WARPS 8

__global__ void qwen35_flash_prefill_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* k_cache,
    const __nv_bfloat16* v_cache, __nv_bfloat16* out, int seq, int heads,
    int n_kv_heads, int head_dim, float scale, uint32_t start_pos,
    int max_seq_len) {
  const int s = blockIdx.x;
  const int q_head = blockIdx.y;
  const int kv_head = q_head / (heads / n_kv_heads);
  const int lane = threadIdx.x % 32;
  const int warp_id = threadIdx.x / 32;
  const int tid = threadIdx.x;
  if (tid >= head_dim) return;

  const int elems = head_dim / 32;  // dims per thread (head_dim % 32 == 0)
  const int visible = (int)start_pos + s + 1;  // causal prefix length
  const __nv_bfloat16* q_row = q + (s * heads + q_head) * head_dim;
  const __nv_bfloat16* k_base = k_cache + kv_head * max_seq_len * head_dim;
  const __nv_bfloat16* v_base = v_cache + kv_head * max_seq_len * head_dim;

  float q_reg[16];
  for (int i = 0; i < elems; i++)
    q_reg[i] = __bfloat162float(q_row[i * 32 + lane]);

  float m = -INFINITY;
  float l = 0.0f;
  float acc[16];
  for (int i = 0; i < elems; i++) acc[i] = 0.0f;

  for (int t = warp_id; t < visible; t += QWEN35_PREFILL_WARPS) {
    float dot = 0.0f;
    for (int i = 0; i < elems; i++)
      dot += q_reg[i] * __bfloat162float(k_base[t * head_dim + i * 32 + lane]);
    for (int off = 16; off > 0; off >>= 1)
      dot += __shfl_xor_sync(0xffffffff, dot, off);
    dot *= scale;
    const float m_new = fmaxf(m, dot);
    const float p = expf(dot - m_new);
    const float exp_m = expf(m - m_new);
    l = l * exp_m + p;
    for (int i = 0; i < elems; i++)
      acc[i] = acc[i] * exp_m +
               p * __bfloat162float(v_base[t * head_dim + i * 32 + lane]);
    m = m_new;
  }

  // Merge per-warp partials through shared memory.
  __shared__ float s_m[QWEN35_PREFILL_WARPS];
  __shared__ float s_l[QWEN35_PREFILL_WARPS];
  __shared__ float s_acc[QWEN35_PREFILL_WARPS][16][32];
  s_m[warp_id] = m;
  s_l[warp_id] = l;
  for (int i = 0; i < elems; i++) s_acc[warp_id][i][lane] = acc[i];
  __syncthreads();

  float m_global = -INFINITY;
  for (int w = 0; w < QWEN35_PREFILL_WARPS; w++)
    m_global = fmaxf(m_global, s_m[w]);
  float l_global = 0.0f;
  for (int w = 0; w < QWEN35_PREFILL_WARPS; w++)
    l_global += s_l[w] * expf(s_m[w] - m_global);
  for (int i = 0; i < elems; i++) {
    float acc_global = 0.0f;
    for (int w = 0; w < QWEN35_PREFILL_WARPS; w++)
      acc_global += s_acc[w][i][lane] * expf(s_m[w] - m_global);
    const float inv_l = (l_global > 0.0f) ? (1.0f / l_global) : 0.0f;
    out[(s * heads + q_head) * head_dim + i * 32 + lane] =
        __float2bfloat16(acc_global * inv_l);
  }
}

// Exact seq=1 specialization for the production 24 Q / 4 KV / 256 shape.
// Warp timestep ownership, online softmax, and merge order match the fallback.
// The result is rounded to bf16 before gating, as in flash + sigmoid_mul.
__global__ void qwen35_flash_decode_gated_256_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* k_cache,
    const __nv_bfloat16* v_cache, const __nv_bfloat16* gate,
    __nv_bfloat16* out, float scale, const uint32_t* position,
    int max_seq_len) {
  constexpr int kHeadDim = 256, kHeads = 24, kKvHeads = 4;
  constexpr int kElems = 8, kWarps = QWEN35_PREFILL_WARPS;
  const int q_head = blockIdx.x, kv_head = q_head / (kHeads / kKvHeads);
  const int lane = threadIdx.x & 31, warp_id = threadIdx.x >> 5;
  const int visible = static_cast<int>(*position) + 1;
  const int out_base = q_head * kHeadDim;
  const __nv_bfloat16* q_row = q + out_base;
  const __nv_bfloat16* k_row = k_cache +
      (static_cast<int64_t>(kv_head) * max_seq_len + warp_id) * kHeadDim + lane;
  const __nv_bfloat16* v_row = v_cache +
      (static_cast<int64_t>(kv_head) * max_seq_len + warp_id) * kHeadDim + lane;
  float q_reg[kElems], acc[kElems];
#pragma unroll
  for (int i = 0; i < kElems; ++i) {
    q_reg[i] = __bfloat162float(q_row[i * 32 + lane]);
    acc[i] = 0.0f;
  }
  float m = -INFINITY, l = 0.0f;
  for (int t = warp_id; t < visible; t += kWarps) {
    float dot = 0.0f;
#pragma unroll
    for (int i = 0; i < kElems; ++i)
      dot += q_reg[i] * __bfloat162float(k_row[i * 32]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      dot += __shfl_xor_sync(0xffffffffu, dot, off);
    dot *= scale;
    const float m_new = fmaxf(m, dot);
    const float p = expf(dot - m_new), exp_m = expf(m - m_new);
    l = l * exp_m + p;
#pragma unroll
    for (int i = 0; i < kElems; ++i)
      acc[i] = acc[i] * exp_m + p * __bfloat162float(v_row[i * 32]);
    m = m_new;
    k_row += kWarps * kHeadDim;
    v_row += kWarps * kHeadDim;
  }
  __shared__ float s_m[kWarps], s_l[kWarps];
  __shared__ float s_acc[kWarps][kElems][32], s_weight[kWarps], s_inv_l;
  if (lane == 0) { s_m[warp_id] = m; s_l[warp_id] = l; }
#pragma unroll
  for (int i = 0; i < kElems; ++i) s_acc[warp_id][i][lane] = acc[i];
  __syncthreads();
  if (threadIdx.x == 0) {
    float m_global = -INFINITY;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) m_global = fmaxf(m_global, s_m[w]);
    float l_global = 0.0f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) {
      s_weight[w] = expf(s_m[w] - m_global);
      l_global += s_l[w] * s_weight[w];
    }
    s_inv_l = l_global > 0.0f ? 1.0f / l_global : 0.0f;
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < kElems; ++i) {
    float acc_global = 0.0f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w)
      acc_global += s_acc[w][i][lane] * s_weight[w];
    const int index = out_base + i * 32 + lane;
    const __nv_bfloat16 attn = __float2bfloat16(acc_global * s_inv_l);
    out[index] = __float2bfloat16(__bfloat162float(attn) *
        sigmoidf_f32(__bfloat162float(gate[index])));
  }
}

// Exact two-pass seq=1 attention. Pass 1 preserves the original eight warp
// timestep streams but schedules them as two four-warp CTAs per head. Pass 2
constexpr int QWEN35_SPLIT_ATTN_WARPS = 8;
constexpr int QWEN35_SPLIT_ATTN_ELEMS = 8;
constexpr int QWEN35_SPLIT_ATTN_STRIDE = 2 + QWEN35_SPLIT_ATTN_ELEMS * 32;
// Iteration-33 GQA-aware exact attention. QGroup contiguous Q heads that
// belong to one KV head share one staged BF16 K/V row for each historical
// stream timestep. Every Q-head warp retains its own q registers, online
// softmax recurrence, FP32 accumulators, and existing partial slot.
template <int QGroup>
__global__ void qwen35_flash_decode_gated_256_partial_gqa_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* k_cache,
    const __nv_bfloat16* v_cache, float* partials, float scale,
    const uint32_t* position, int max_seq_len) {
  static_assert(QGroup == 2 || QGroup == 3 || QGroup == 6);
  constexpr int kHeadDim = 256, kHeads = 24, kKvHeads = 4;
  constexpr int kQPerKv = kHeads / kKvHeads;
  constexpr int kGroupsPerKv = kQPerKv / QGroup;
  const int kv_head = blockIdx.x / kGroupsPerKv;
  const int q_group = blockIdx.x % kGroupsPerKv;
  const int q_local = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int q_head = kv_head * kQPerKv + q_group * QGroup + q_local;
  const int warp_id = blockIdx.y;
  const int visible = static_cast<int>(*position) + 1;
  const int out_base = q_head * kHeadDim;
  const __nv_bfloat16* q_row = q + out_base;
  float q_reg[QWEN35_SPLIT_ATTN_ELEMS];
  float acc[QWEN35_SPLIT_ATTN_ELEMS];
#pragma unroll
  for (int i = 0; i < QWEN35_SPLIT_ATTN_ELEMS; ++i) {
    q_reg[i] = __bfloat162float(q_row[i * 32 + lane]);
    acc[i] = 0.0f;
  }
  __shared__ __align__(16) __nv_bfloat16 s_k[kHeadDim];
  __shared__ __align__(16) __nv_bfloat16 s_v[kHeadDim];
  float m = -INFINITY, l = 0.0f;
  for (int t = warp_id; t < visible; t += QWEN35_SPLIT_ATTN_WARPS) {
    const __nv_bfloat16* k_row = k_cache +
        (static_cast<int64_t>(kv_head) * max_seq_len + t) * kHeadDim;
    const __nv_bfloat16* v_row = v_cache +
        (static_cast<int64_t>(kv_head) * max_seq_len + t) * kHeadDim;
    for (int index = threadIdx.x; index < kHeadDim; index += blockDim.x) {
      s_k[index] = k_row[index];
      s_v[index] = v_row[index];
    }
    __syncthreads();
    float dot = 0.0f;
#pragma unroll
    for (int i = 0; i < QWEN35_SPLIT_ATTN_ELEMS; ++i)
      dot += q_reg[i] * __bfloat162float(s_k[i * 32 + lane]);
#pragma unroll
    for (int shift = 16; shift > 0; shift >>= 1)
      dot += __shfl_xor_sync(0xffffffffu, dot, shift);
    dot *= scale;
    const float m_new = fmaxf(m, dot);
    const float p = expf(dot - m_new);
    const float exp_m = expf(m - m_new);
    l = l * exp_m + p;
#pragma unroll
    for (int i = 0; i < QWEN35_SPLIT_ATTN_ELEMS; ++i)
      acc[i] = acc[i] * exp_m + p * __bfloat162float(s_v[i * 32 + lane]);
    m = m_new;
    __syncthreads();
  }
  float* warp_partial = partials +
      (q_head * QWEN35_SPLIT_ATTN_WARPS + warp_id) * QWEN35_SPLIT_ATTN_STRIDE;
  if (lane == 0) {
    warp_partial[0] = m;
    warp_partial[1] = l;
  }
#pragma unroll
  for (int i = 0; i < QWEN35_SPLIT_ATTN_ELEMS; ++i)
    warp_partial[2 + i * 32 + lane] = acc[i];
}

// merges warp partials in the original w=0..7 order and preserves the BF16
// attention boundary before gating.

__global__ void qwen35_flash_decode_gated_256_partial_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* k_cache,
    const __nv_bfloat16* v_cache, float* partials, float scale,
    const uint32_t* position, int max_seq_len) {
  constexpr int kHeadDim = 256, kHeads = 24, kKvHeads = 4;
  const int q_head = blockIdx.x;
  const int warp_group = blockIdx.y;
  const int local_warp = threadIdx.x >> 5;
  const int warp_id = warp_group * 4 + local_warp;
  const int lane = threadIdx.x & 31;
  const int kv_head = q_head / (kHeads / kKvHeads);
  const int visible = static_cast<int>(*position) + 1;
  const int out_base = q_head * kHeadDim;
  const __nv_bfloat16* q_row = q + out_base;
  const __nv_bfloat16* k_row = k_cache +
      (static_cast<int64_t>(kv_head) * max_seq_len + warp_id) * kHeadDim + lane;
  const __nv_bfloat16* v_row = v_cache +
      (static_cast<int64_t>(kv_head) * max_seq_len + warp_id) * kHeadDim + lane;
  float q_reg[QWEN35_SPLIT_ATTN_ELEMS];
  float acc[QWEN35_SPLIT_ATTN_ELEMS];
#pragma unroll
  for (int i = 0; i < QWEN35_SPLIT_ATTN_ELEMS; ++i) {
    q_reg[i] = __bfloat162float(q_row[i * 32 + lane]);
    acc[i] = 0.0f;
  }
  float m = -INFINITY, l = 0.0f;
  for (int t = warp_id; t < visible; t += QWEN35_SPLIT_ATTN_WARPS) {
    float dot = 0.0f;
#pragma unroll
    for (int i = 0; i < QWEN35_SPLIT_ATTN_ELEMS; ++i)
      dot += q_reg[i] * __bfloat162float(k_row[i * 32]);
#pragma unroll
    for (int shift = 16; shift > 0; shift >>= 1)
      dot += __shfl_xor_sync(0xffffffffu, dot, shift);
    dot *= scale;
    const float m_new = fmaxf(m, dot);
    const float p = expf(dot - m_new);
    const float exp_m = expf(m - m_new);
    l = l * exp_m + p;
#pragma unroll
    for (int i = 0; i < QWEN35_SPLIT_ATTN_ELEMS; ++i)
      acc[i] = acc[i] * exp_m + p * __bfloat162float(v_row[i * 32]);
    m = m_new;
    k_row += QWEN35_SPLIT_ATTN_WARPS * kHeadDim;
    v_row += QWEN35_SPLIT_ATTN_WARPS * kHeadDim;
  }
  float* warp_partial = partials +
      (q_head * QWEN35_SPLIT_ATTN_WARPS + warp_id) * QWEN35_SPLIT_ATTN_STRIDE;
  if (lane == 0) {
    warp_partial[0] = m;
    warp_partial[1] = l;
  }
#pragma unroll
  for (int i = 0; i < QWEN35_SPLIT_ATTN_ELEMS; ++i)
    warp_partial[2 + i * 32 + lane] = acc[i];
}

// Exact two-warp-CTA partial attention candidate. Each head keeps the same
// eight warp streams and partial layout; four CTAs replace two four-warp CTAs.
__global__ void qwen35_flash_decode_gated_256_partial_2w_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* k_cache,
    const __nv_bfloat16* v_cache, float* partials, float scale,
    const uint32_t* position, int max_seq_len) {
  constexpr int kHeadDim = 256, kHeads = 24, kKvHeads = 4;
  const int q_head = blockIdx.x;
  const int warp_group = blockIdx.y;
  const int local_warp = threadIdx.x >> 5;
  const int warp_id = warp_group * 2 + local_warp;
  const int lane = threadIdx.x & 31;
  const int kv_head = q_head / (kHeads / kKvHeads);
  const int visible = static_cast<int>(*position) + 1;
  const int out_base = q_head * kHeadDim;
  const __nv_bfloat16* q_row = q + out_base;
  const __nv_bfloat16* k_row = k_cache +
      (static_cast<int64_t>(kv_head) * max_seq_len + warp_id) * kHeadDim + lane;
  const __nv_bfloat16* v_row = v_cache +
      (static_cast<int64_t>(kv_head) * max_seq_len + warp_id) * kHeadDim + lane;
  float q_reg[QWEN35_SPLIT_ATTN_ELEMS];
  float acc[QWEN35_SPLIT_ATTN_ELEMS];
#pragma unroll
  for (int i = 0; i < QWEN35_SPLIT_ATTN_ELEMS; ++i) {
    q_reg[i] = __bfloat162float(q_row[i * 32 + lane]);
    acc[i] = 0.0f;
  }
  float m = -INFINITY, l = 0.0f;
  for (int t = warp_id; t < visible; t += QWEN35_SPLIT_ATTN_WARPS) {
    float dot = 0.0f;
#pragma unroll
    for (int i = 0; i < QWEN35_SPLIT_ATTN_ELEMS; ++i)
      dot += q_reg[i] * __bfloat162float(k_row[i * 32]);
#pragma unroll
    for (int shift = 16; shift > 0; shift >>= 1)
      dot += __shfl_xor_sync(0xffffffffu, dot, shift);
    dot *= scale;
    const float m_new = fmaxf(m, dot);
    const float p = expf(dot - m_new);
    const float exp_m = expf(m - m_new);
    l = l * exp_m + p;
#pragma unroll
    for (int i = 0; i < QWEN35_SPLIT_ATTN_ELEMS; ++i)
      acc[i] = acc[i] * exp_m + p * __bfloat162float(v_row[i * 32]);
    m = m_new;
    k_row += QWEN35_SPLIT_ATTN_WARPS * kHeadDim;
    v_row += QWEN35_SPLIT_ATTN_WARPS * kHeadDim;
  }
  float* warp_partial = partials +
      (q_head * QWEN35_SPLIT_ATTN_WARPS + warp_id) * QWEN35_SPLIT_ATTN_STRIDE;
  if (lane == 0) {
    warp_partial[0] = m;
    warp_partial[1] = l;
  }
#pragma unroll
  for (int i = 0; i < QWEN35_SPLIT_ATTN_ELEMS; ++i)
    warp_partial[2 + i * 32 + lane] = acc[i];
}

// Exact one-warp-CTA partial attention candidate. Eight CTAs per head map
// one-to-one to the established eight timestep streams and partial slots.
__global__ void qwen35_flash_decode_gated_256_partial_1w_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* k_cache,
    const __nv_bfloat16* v_cache, float* partials, float scale,
    const uint32_t* position, int max_seq_len) {
  constexpr int kHeadDim = 256, kHeads = 24, kKvHeads = 4;
  const int q_head = blockIdx.x;
  const int warp_id = blockIdx.y;
  const int lane = threadIdx.x;
  const int kv_head = q_head / (kHeads / kKvHeads);
  const int visible = static_cast<int>(*position) + 1;
  const int out_base = q_head * kHeadDim;
  const __nv_bfloat16* q_row = q + out_base;
  const __nv_bfloat16* k_row = k_cache +
      (static_cast<int64_t>(kv_head) * max_seq_len + warp_id) * kHeadDim + lane;
  const __nv_bfloat16* v_row = v_cache +
      (static_cast<int64_t>(kv_head) * max_seq_len + warp_id) * kHeadDim + lane;
  float q_reg[QWEN35_SPLIT_ATTN_ELEMS];
  float acc[QWEN35_SPLIT_ATTN_ELEMS];
#pragma unroll
  for (int i = 0; i < QWEN35_SPLIT_ATTN_ELEMS; ++i) {
    q_reg[i] = __bfloat162float(q_row[i * 32 + lane]);
    acc[i] = 0.0f;
  }
  float m = -INFINITY, l = 0.0f;
  for (int t = warp_id; t < visible; t += QWEN35_SPLIT_ATTN_WARPS) {
    float dot = 0.0f;
#pragma unroll
    for (int i = 0; i < QWEN35_SPLIT_ATTN_ELEMS; ++i)
      dot += q_reg[i] * __bfloat162float(k_row[i * 32]);
#pragma unroll
    for (int shift = 16; shift > 0; shift >>= 1)
      dot += __shfl_xor_sync(0xffffffffu, dot, shift);
    dot *= scale;
    const float m_new = fmaxf(m, dot);
    const float p = expf(dot - m_new);
    const float exp_m = expf(m - m_new);
    l = l * exp_m + p;
#pragma unroll
    for (int i = 0; i < QWEN35_SPLIT_ATTN_ELEMS; ++i)
      acc[i] = acc[i] * exp_m + p * __bfloat162float(v_row[i * 32]);
    m = m_new;
    k_row += QWEN35_SPLIT_ATTN_WARPS * kHeadDim;
    v_row += QWEN35_SPLIT_ATTN_WARPS * kHeadDim;
  }
  float* warp_partial = partials +
      (q_head * QWEN35_SPLIT_ATTN_WARPS + warp_id) * QWEN35_SPLIT_ATTN_STRIDE;
  if (lane == 0) {
    warp_partial[0] = m;
    warp_partial[1] = l;
  }
#pragma unroll
  for (int i = 0; i < QWEN35_SPLIT_ATTN_ELEMS; ++i)
    warp_partial[2 + i * 32 + lane] = acc[i];
}

__global__ void qwen35_flash_decode_gated_256_merge_kernel(
    const float* partials, const __nv_bfloat16* gate,
    __nv_bfloat16* out) {
  constexpr int kHeadDim = 256;
  const int q_head = blockIdx.x;
  const int lane = threadIdx.x & 31;
  const int element = threadIdx.x >> 5;
  __shared__ float s_weight[QWEN35_SPLIT_ATTN_WARPS];
  __shared__ float s_inv_l;
  const float* head_partial = partials +
      q_head * QWEN35_SPLIT_ATTN_WARPS * QWEN35_SPLIT_ATTN_STRIDE;
  if (threadIdx.x == 0) {
    float m_global = -INFINITY;
#pragma unroll
    for (int warp = 0; warp < QWEN35_SPLIT_ATTN_WARPS; ++warp)
      m_global = fmaxf(m_global,
          head_partial[warp * QWEN35_SPLIT_ATTN_STRIDE]);
    float l_global = 0.0f;
#pragma unroll
    for (int warp = 0; warp < QWEN35_SPLIT_ATTN_WARPS; ++warp) {
      const float* current = head_partial + warp * QWEN35_SPLIT_ATTN_STRIDE;
      s_weight[warp] = expf(current[0] - m_global);
      l_global += current[1] * s_weight[warp];
    }
    s_inv_l = l_global > 0.0f ? 1.0f / l_global : 0.0f;
  }
  __syncthreads();
  float acc_global = 0.0f;
#pragma unroll
  for (int warp = 0; warp < QWEN35_SPLIT_ATTN_WARPS; ++warp) {
    const float* current = head_partial + warp * QWEN35_SPLIT_ATTN_STRIDE;
    acc_global += current[2 + element * 32 + lane] * s_weight[warp];
  }
  const int index = q_head * kHeadDim + element * 32 + lane;
  const __nv_bfloat16 attention = __float2bfloat16(acc_global * s_inv_l);
  out[index] = __float2bfloat16(__bfloat162float(attention) *
      sigmoidf_f32(__bfloat162float(gate[index])));
}




// vf32 = f32(v) for the attention output GEMM: [visible, head_dim] f32.
__global__ void qwen35_v_to_f32_kernel(
    const __nv_bfloat16* v, float* vf32, int visible, int head_dim) {
  const int idx = blockIdx.x * 256 + threadIdx.x;
  if (idx < visible * head_dim)
    vf32[idx] = __bfloat162float(v[idx]);
}

// ── KV transpose for the batched attention scores GEMM ─────────────────────
// k: [visible, head_dim] bf16 -> kt: [head_dim, visible] bf16. A padded
// shared tile keeps both the cache read and transposed write coalesced.
__global__ void qwen35_transpose_kt_kernel(
    const __nv_bfloat16* k, __nv_bfloat16* kt, int visible, int head_dim) {
  __shared__ __nv_bfloat16 tile[32][33];
  const int d_in = blockIdx.x * 32 + threadIdx.x;
  for (int row = threadIdx.y; row < 32; row += blockDim.y) {
    const int t_in = blockIdx.y * 32 + row;
    if (d_in < head_dim && t_in < visible)
      tile[row][threadIdx.x] = k[t_in * head_dim + d_in];
  }
  __syncthreads();

  const int t_out = blockIdx.y * 32 + threadIdx.x;
  for (int row = threadIdx.y; row < 32; row += blockDim.y) {
    const int d_out = blockIdx.x * 32 + row;
    if (d_out < head_dim && t_out < visible)
      kt[d_out * visible + t_out] = tile[threadIdx.x][row];
  }
}

// ── GEMM-based GQA attention helpers ───────────────────────────────────────
// The scores matrix is produced by strided-batched cublas GEMMs; these
// kernels finish the softmax and the gate/scale fusion on the GPU.

// scores: [heads, seq, visible] bf16 (q@k^T, unscaled)
// l_out:  [heads, seq] f32 row sums of the softmax weights
// grid: (seq, heads); block: 256 threads sweep `visible` twice.
__global__ void qwen35_attention_softmax_rows_kernel(
    float* scores, float* l_out, int head_base, int seq, int heads,
    int visible, int row_stride, int start_pos, float scale) {
  const int s = blockIdx.x;
  const int h_local = blockIdx.y;
  const int h = head_base + h_local;
  const int tid = threadIdx.x;
  const int row = h * seq + s;
  const int local_row = h_local * seq + s;
  const int valid = start_pos + s + 1;  // causal boundary for this row
  float* row_ptr = scores + local_row * row_stride;

  // Pass 1: row max over t < valid.
  float m = -INFINITY;
  for (int t = tid; t < valid; t += 256)
    m = fmaxf(m, row_ptr[t]);
  for (int off = 16; off > 0; off >>= 1)
    m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, off));
  __shared__ float s_warp_max[8];
  if (tid % 32 == 0) s_warp_max[tid / 32] = m;
  __syncthreads();
  for (int w = 0; w < 8; w++) m = fmaxf(m, s_warp_max[w]);
  __syncthreads();

  // Pass 2: exp((x - m) * scale) in place, accumulate l; zero the
  // columns beyond the causal boundary (the pv GEMM sums all `visible`).
  float l = 0.0f;
  for (int t = tid; t < visible; t += 256) {
    if (t < valid) {
      const float p = expf((row_ptr[t] - m) * scale);
      row_ptr[t] = p;
      l += p;
    } else {
      row_ptr[t] = 0.0f;
    }
  }
  for (int off = 16; off > 0; off >>= 1)
    l += __shfl_xor_sync(0xffffffff, l, off);
  __shared__ float s_warp_l[8];
  if (tid % 32 == 0) s_warp_l[tid / 32] = l;
  __syncthreads();
  l = 0.0f;
  for (int w = 0; w < 8; w++) l += s_warp_l[w];
  if (tid == 0) l_out[row] = (l > 0.0f) ? l : 0.0f;
}

// attn_bf16 = bf16(pv / l) per row; pv/l are f32, out is [seq, heads, dim].
__global__ void qwen35_scale_out_kernel(
    const float* pv, const float* l, __nv_bfloat16* out, int seq, int heads,
    int head_dim) {
  const int s = blockIdx.x;
  const int h = blockIdx.y;
  const int d = threadIdx.x;
  const int row = h * seq + s;
  const int idx = (s * heads + h) * head_dim + d;
  const float inv_l = (l[row] > 0.0f) ? (1.0f / l[row]) : 0.0f;
  out[idx] = __float2bfloat16(pv[idx] * inv_l);
}

// Preserve the former scale_out -> sigmoid_mul BF16 boundary while avoiding
// the intermediate attention round trip and second kernel launch.
__global__ void qwen35_scale_out_gated_kernel(
    const float* pv, const float* l, const __nv_bfloat16* gate,
    __nv_bfloat16* out, int seq, int heads, int head_dim) {
  const int s = blockIdx.x;
  const int h = blockIdx.y;
  const int d = threadIdx.x;
  const int row = h * seq + s;
  const int idx = (s * heads + h) * head_dim + d;
  const float inv_l = (l[row] > 0.0f) ? (1.0f / l[row]) : 0.0f;
  const __nv_bfloat16 rounded = __float2bfloat16(pv[idx] * inv_l);
  out[idx] = __float2bfloat16(__bfloat162float(rounded) *
                              sigmoidf_f32(__bfloat162float(gate[idx])));
}



// Fused W4A16 prefill kernel. One 256-thread block computes a [32, 128]
// output tile as sixteen serial m16n16k16 tensor-core accumulators. The M
// dimension is grid-tiled, so the per-CTA shared-memory and register footprint
// is independent of the full row count (up to the adapter's CHUNK bound).
// K advances in the same 16-element MMA order as the existing exact paths,
// while each packed weight is expanded only into block-local shared memory.
// The template keeps raw compressed-tensors and repacked-v1 physical indexing
// separate without changing either baseline ABI.
#define QWEN35_PREFILL_FAST_M_TILE 32
#define QWEN35_PREFILL_FAST_N_TILE 128
#define QWEN35_PREFILL_FAST_K_TILE 32
#define QWEN35_PREFILL_NATIVE_MAX_ROWS 2048

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
template <bool Repacked>
__global__ void qwen35_gemm_w4a16_bf16_prefill_fast_kernel(
    const __nv_bfloat16* activation, const int32_t* weight_qwords,
    const __nv_bfloat16* weight_scale, const int32_t* weight_zero_point,
    __nv_bfloat16* output, int rows, int in_cols, int out_cols, int groups) {
  const int m_base = blockIdx.y * QWEN35_PREFILL_FAST_M_TILE;
  const int n_base = blockIdx.x * QWEN35_PREFILL_FAST_N_TILE;
  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int warp_m = warp / 4;
  const int warp_n = warp % 4;
  const int group_size = in_cols / groups;
  const int raw_words_per_row = in_cols / 8;
  const int repacked_k_tiles = in_cols / 16;

  __shared__ __align__(32) __nv_bfloat16
      s_activation[QWEN35_PREFILL_FAST_M_TILE * QWEN35_PREFILL_FAST_K_TILE];
  __shared__ __align__(32) __nv_bfloat16
      s_weight[QWEN35_PREFILL_FAST_N_TILE * QWEN35_PREFILL_FAST_K_TILE];
  __shared__ __align__(32) float s_accumulator[8][2][16 * 16];

  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>
      accumulator0;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>
      accumulator1;
  nvcuda::wmma::fill_fragment(accumulator0, 0.0f);
  nvcuda::wmma::fill_fragment(accumulator1, 0.0f);

  for (int k_base = 0; k_base < in_cols;
       k_base += QWEN35_PREFILL_FAST_K_TILE) {
    for (int index = tid;
         index < QWEN35_PREFILL_FAST_M_TILE * QWEN35_PREFILL_FAST_K_TILE;
         index += blockDim.x) {
      const int m = index / QWEN35_PREFILL_FAST_K_TILE;
      const int k = index % QWEN35_PREFILL_FAST_K_TILE;
      const int global_m = m_base + m;
      s_activation[index] = global_m < rows
          ? activation[static_cast<int64_t>(global_m) * in_cols + k_base + k]
          : __float2bfloat16(0.0f);
    }

    // Each iteration covers one group-32 K tile. Expanding by qword keeps
    // global loads naturally aligned and avoids the former N64/K16 launch
    // geometry that exposed too little work per block for short prefill.
    constexpr int qwords_per_tile =
        QWEN35_PREFILL_FAST_N_TILE * QWEN35_PREFILL_FAST_K_TILE / 8;
    for (int index = tid; index < qwords_per_tile; index += blockDim.x) {
      const int n_local = index / (QWEN35_PREFILL_FAST_K_TILE / 8);
      const int word_in_k = index % (QWEN35_PREFILL_FAST_K_TILE / 8);
      const int global_n = n_base + n_local;
      uint32_t qword = 0;
      int zp = 0;
      float scale = 0.0f;
      if (global_n < out_cols) {
        const int group = k_base / group_size;
        if (Repacked) {
          const int n64 = global_n / 64;
          const int row64 = global_n % 64;
          const int k16 = k_base / 16 + word_in_k / 2;
          const int word2 = word_in_k % 2;
          const int64_t qword_index =
              ((static_cast<int64_t>(n64) * repacked_k_tiles + k16) * 64 +
               row64) * 2 + word2;
          qword = static_cast<uint32_t>(weight_qwords[qword_index]);
          const uint32_t zp_word = static_cast<uint32_t>(weight_zero_point[
              (static_cast<int64_t>(n64) * groups + group) * 8 + row64 / 8]);
          zp = static_cast<int>((zp_word >> ((row64 & 7) * 4)) & 0xFu);
          scale = __bfloat162float(weight_scale[
              (static_cast<int64_t>(n64) * groups + group) * 64 + row64]);
        } else {
          qword = static_cast<uint32_t>(weight_qwords[
              static_cast<int64_t>(global_n) * raw_words_per_row +
              k_base / 8 + word_in_k]);
          const uint32_t zp_word = static_cast<uint32_t>(weight_zero_point[
              static_cast<int64_t>(global_n / 8) * groups + group]);
          zp = static_cast<int>((zp_word >> ((global_n & 7) * 4)) & 0xFu);
          scale = __bfloat162float(weight_scale[
              static_cast<int64_t>(global_n) * groups + group]);
        }
      }
#pragma unroll
      for (int element = 0; element < 8; ++element) {
        const int q = static_cast<int>((qword >> (element * 4)) & 0xFu);
        s_weight[n_local * QWEN35_PREFILL_FAST_K_TILE + word_in_k * 8 +
                 element] =
            __float2bfloat16(static_cast<float>(q - zp) * scale);
      }
    }
    __syncthreads();

#pragma unroll
    for (int k_half = 0; k_half < QWEN35_PREFILL_FAST_K_TILE; k_half += 16) {
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,
                             __nv_bfloat16, nvcuda::wmma::row_major>
          a_fragment;
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16,
                             __nv_bfloat16, nvcuda::wmma::col_major>
          b_fragment0;
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16,
                             __nv_bfloat16, nvcuda::wmma::col_major>
          b_fragment1;
      nvcuda::wmma::load_matrix_sync(
          a_fragment,
          s_activation + warp_m * 16 * QWEN35_PREFILL_FAST_K_TILE + k_half,
          QWEN35_PREFILL_FAST_K_TILE);
      const int warp_n_base = warp_n * 32;
      nvcuda::wmma::load_matrix_sync(
          b_fragment0,
          s_weight + warp_n_base * QWEN35_PREFILL_FAST_K_TILE + k_half,
          QWEN35_PREFILL_FAST_K_TILE);
      nvcuda::wmma::load_matrix_sync(
          b_fragment1,
          s_weight + (warp_n_base + 16) * QWEN35_PREFILL_FAST_K_TILE + k_half,
          QWEN35_PREFILL_FAST_K_TILE);
      nvcuda::wmma::mma_sync(
          accumulator0, a_fragment, b_fragment0, accumulator0);
      nvcuda::wmma::mma_sync(
          accumulator1, a_fragment, b_fragment1, accumulator1);
    }
    __syncthreads();
  }

  nvcuda::wmma::store_matrix_sync(
      s_accumulator[warp][0], accumulator0, 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync(
      s_accumulator[warp][1], accumulator1, 16, nvcuda::wmma::mem_row_major);
  __syncthreads();
  for (int index = tid;
       index < QWEN35_PREFILL_FAST_M_TILE * QWEN35_PREFILL_FAST_N_TILE;
       index += blockDim.x) {
    const int m = index / QWEN35_PREFILL_FAST_N_TILE;
    const int n = index % QWEN35_PREFILL_FAST_N_TILE;
    const int global_m = m_base + m;
    const int global_n = n_base + n;
    if (global_m < rows && global_n < out_cols) {
      const int owner_warp = (m / 16) * 4 + n / 32;
      const int owner_fragment = (n % 32) / 16;
      output[static_cast<int64_t>(global_m) * out_cols + global_n] =
          __float2bfloat16(s_accumulator[owner_warp][owner_fragment]
                                       [(m % 16) * 16 + n % 16]);
    }
  }
}
#endif

// Metadata-reusing raw/repacked prefill variant. It loads one scale and zero
// point per output row and group into shared memory before expanding the four
// packed qwords. Full-row calls are grid-tiled in M exactly like prefill-fast;
// activation, dequantization, MMA, and BF16 store order are unchanged.
#define QWEN35_PREFILL_PACKED_M_TILE 32
#define QWEN35_PREFILL_PACKED_N_TILE 128
#define QWEN35_PREFILL_PACKED_K_TILE 32

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
template <bool Repacked>
__global__ void qwen35_gemm_w4a16_bf16_prefill_packed_kernel(
    const __nv_bfloat16* activation, const int32_t* weight_qwords,
    const __nv_bfloat16* weight_scale, const int32_t* weight_zero_point,
    __nv_bfloat16* output, int rows, int in_cols, int out_cols, int groups) {
  const int m_base = blockIdx.y * QWEN35_PREFILL_PACKED_M_TILE;
  const int n_base = blockIdx.x * QWEN35_PREFILL_PACKED_N_TILE;
  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int warp_m = warp / 4;
  const int warp_n = warp % 4;
  const int group_size = in_cols / groups;
  const int raw_words_per_row = in_cols / 8;
  const int repacked_k_tiles = in_cols / 16;

  __shared__ __align__(32) __nv_bfloat16
      s_activation[QWEN35_PREFILL_PACKED_M_TILE * QWEN35_PREFILL_PACKED_K_TILE];
  __shared__ __align__(32) __nv_bfloat16
      s_weight[QWEN35_PREFILL_PACKED_N_TILE * QWEN35_PREFILL_PACKED_K_TILE];
  __shared__ __align__(32) __nv_bfloat16
      s_scale[QWEN35_PREFILL_PACKED_N_TILE];
  __shared__ __align__(32) uint8_t s_zero_point[QWEN35_PREFILL_PACKED_N_TILE];
  __shared__ __align__(32) float s_accumulator[8][2][16 * 16];

  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>
      accumulator0;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>
      accumulator1;
  nvcuda::wmma::fill_fragment(accumulator0, 0.0f);
  nvcuda::wmma::fill_fragment(accumulator1, 0.0f);

  for (int k_base = 0; k_base < in_cols;
       k_base += QWEN35_PREFILL_PACKED_K_TILE) {
    for (int index = tid;
         index < QWEN35_PREFILL_PACKED_M_TILE * QWEN35_PREFILL_PACKED_K_TILE;
         index += blockDim.x) {
      const int m = index / QWEN35_PREFILL_PACKED_K_TILE;
      const int k = index % QWEN35_PREFILL_PACKED_K_TILE;
      const int global_m = m_base + m;
      s_activation[index] = global_m < rows
          ? activation[static_cast<int64_t>(global_m) * in_cols + k_base + k]
          : __float2bfloat16(0.0f);
    }

    for (int n_local = tid; n_local < QWEN35_PREFILL_PACKED_N_TILE;
         n_local += blockDim.x) {
      const int global_n = n_base + n_local;
      int zp = 0;
      __nv_bfloat16 scale = __float2bfloat16(0.0f);
      if (global_n < out_cols) {
        const int group = k_base / group_size;
        if (Repacked) {
          const int n64 = global_n / 64;
          const int row64 = global_n % 64;
          const uint32_t zp_word = static_cast<uint32_t>(weight_zero_point[
              (static_cast<int64_t>(n64) * groups + group) * 8 + row64 / 8]);
          zp = static_cast<int>((zp_word >> ((row64 & 7) * 4)) & 0xFu);
          scale = weight_scale[
              (static_cast<int64_t>(n64) * groups + group) * 64 + row64];
        } else {
          const uint32_t zp_word = static_cast<uint32_t>(weight_zero_point[
              static_cast<int64_t>(global_n / 8) * groups + group]);
          zp = static_cast<int>((zp_word >> ((global_n & 7) * 4)) & 0xFu);
          scale = weight_scale[static_cast<int64_t>(global_n) * groups + group];
        }
      }
      s_zero_point[n_local] = static_cast<uint8_t>(zp);
      s_scale[n_local] = scale;
    }
    __syncthreads();

    constexpr int qwords_per_tile =
        QWEN35_PREFILL_PACKED_N_TILE * QWEN35_PREFILL_PACKED_K_TILE / 8;
    for (int index = tid; index < qwords_per_tile; index += blockDim.x) {
      const int n_local = index / (QWEN35_PREFILL_PACKED_K_TILE / 8);
      const int word_in_k = index % (QWEN35_PREFILL_PACKED_K_TILE / 8);
      const int global_n = n_base + n_local;
      uint32_t qword = 0;
      if (global_n < out_cols) {
        if (Repacked) {
          const int n64 = global_n / 64;
          const int row64 = global_n % 64;
          const int k16 = k_base / 16 + word_in_k / 2;
          const int word2 = word_in_k % 2;
          const int64_t qword_index =
              ((static_cast<int64_t>(n64) * repacked_k_tiles + k16) * 64 +
               row64) * 2 + word2;
          qword = static_cast<uint32_t>(weight_qwords[qword_index]);
        } else {
          qword = static_cast<uint32_t>(weight_qwords[
              static_cast<int64_t>(global_n) * raw_words_per_row +
              k_base / 8 + word_in_k]);
        }
      }
      const int zp = static_cast<int>(s_zero_point[n_local]);
      const float scale = __bfloat162float(s_scale[n_local]);
#pragma unroll
      for (int element = 0; element < 8; ++element) {
        const int q = static_cast<int>((qword >> (element * 4)) & 0xFu);
        s_weight[n_local * QWEN35_PREFILL_PACKED_K_TILE + word_in_k * 8 +
                 element] =
            __float2bfloat16(static_cast<float>(q - zp) * scale);
      }
    }
    __syncthreads();

#pragma unroll
    for (int k_half = 0; k_half < QWEN35_PREFILL_PACKED_K_TILE; k_half += 16) {
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,
                             __nv_bfloat16, nvcuda::wmma::row_major>
          a_fragment;
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16,
                             __nv_bfloat16, nvcuda::wmma::col_major>
          b_fragment0;
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16,
                             __nv_bfloat16, nvcuda::wmma::col_major>
          b_fragment1;
      nvcuda::wmma::load_matrix_sync(
          a_fragment,
          s_activation + warp_m * 16 * QWEN35_PREFILL_PACKED_K_TILE + k_half,
          QWEN35_PREFILL_PACKED_K_TILE);
      const int warp_n_base = warp_n * 32;
      nvcuda::wmma::load_matrix_sync(
          b_fragment0,
          s_weight + warp_n_base * QWEN35_PREFILL_PACKED_K_TILE + k_half,
          QWEN35_PREFILL_PACKED_K_TILE);
      nvcuda::wmma::load_matrix_sync(
          b_fragment1,
          s_weight + (warp_n_base + 16) * QWEN35_PREFILL_PACKED_K_TILE + k_half,
          QWEN35_PREFILL_PACKED_K_TILE);
      nvcuda::wmma::mma_sync(
          accumulator0, a_fragment, b_fragment0, accumulator0);
      nvcuda::wmma::mma_sync(
          accumulator1, a_fragment, b_fragment1, accumulator1);
    }
    __syncthreads();
  }

  nvcuda::wmma::store_matrix_sync(
      s_accumulator[warp][0], accumulator0, 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync(
      s_accumulator[warp][1], accumulator1, 16, nvcuda::wmma::mem_row_major);
  __syncthreads();
  for (int index = tid;
       index < QWEN35_PREFILL_PACKED_M_TILE * QWEN35_PREFILL_PACKED_N_TILE;
       index += blockDim.x) {
    const int m = index / QWEN35_PREFILL_PACKED_N_TILE;
    const int n = index % QWEN35_PREFILL_PACKED_N_TILE;
    const int global_m = m_base + m;
    const int global_n = n_base + n;
    if (global_m < rows && global_n < out_cols) {
      const int owner_warp = (m / 16) * 4 + n / 32;
      const int owner_fragment = (n % 32) / 16;
      output[static_cast<int64_t>(global_m) * out_cols + global_n] =
          __float2bfloat16(s_accumulator[owner_warp][owner_fragment]
                                       [(m % 16) * 16 + n % 16]);
    }
  }
}
#endif
// Physical W4_REPACKED_N64_K16_V1 prefill path. Each block computes one
// [16, 64] output tile. The packed weights are expanded only into the 2 KiB
// block-local tensor-core tile; the FP32 accumulator fragment remains live for
// the complete K dimension and the sole numerical boundary is the BF16 store.
#define QWEN35_PREFILL_W4_M 512
#define QWEN35_PREFILL_W4_N_TILE 64
#define QWEN35_PREFILL_W4_K_TILE 16

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
__global__ void qwen35_gemm_w4a16_bf16_prefill_repacked_v1_kernel(
    const __nv_bfloat16* activation, const int32_t* weight_qwords,
    const __nv_bfloat16* weight_scale, const int32_t* weight_zero_point,
    __nv_bfloat16* output, int out_cols, int in_cols, int groups) {
  const int n_tile = blockIdx.x;
  const int m_base = blockIdx.y * 16;
  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int k_subtiles = in_cols / QWEN35_PREFILL_W4_K_TILE;

  __shared__ __align__(32) __nv_bfloat16 s_activation[16 * 16];
  __shared__ __align__(32) __nv_bfloat16 s_weight[64 * 16];
  __shared__ __align__(32) float s_accumulator[4][16 * 16];

  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,
                         __nv_bfloat16, nvcuda::wmma::row_major> a_fragment;
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16,
                         __nv_bfloat16, nvcuda::wmma::col_major> b_fragment;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>
      accumulator;
  nvcuda::wmma::fill_fragment(accumulator, 0.0f);

  for (int k_subtile = 0; k_subtile < k_subtiles; ++k_subtile) {
    const int k_base = k_subtile * 16;
    for (int index = tid; index < 16 * 16; index += blockDim.x) {
      const int m = index / 16;
      const int k = index % 16;
      s_activation[index] = activation[
          static_cast<int64_t>(m_base + m) * in_cols + k_base + k];
    }

    // Exactly 128 qwords describe the [64, 16] physical subtile, so each
    // thread performs one coalesced load and expands its eight K-adjacent
    // nibbles. Scale/zp are group-major inside the same N64 tile.
    const int row = tid / 2;
    const int word_in_row = tid & 1;
    const int64_t qword_index =
        ((static_cast<int64_t>(n_tile) * k_subtiles + k_subtile) * 64 + row) *
            2 +
        word_in_row;
    const uint32_t qword = static_cast<uint32_t>(weight_qwords[qword_index]);
    const int group = k_subtile / 2;
    const int64_t scale_index =
        (static_cast<int64_t>(n_tile) * groups + group) * 64 + row;
    const int64_t zp_index =
        (static_cast<int64_t>(n_tile) * groups + group) * 8 + row / 8;
    const uint32_t zp_word =
        static_cast<uint32_t>(weight_zero_point[zp_index]);
    const int zp = static_cast<int>((zp_word >> ((row & 7) * 4)) & 0xFu);
    const float scale = __bfloat162float(weight_scale[scale_index]);
#pragma unroll
    for (int element = 0; element < 8; ++element) {
      const int q = static_cast<int>((qword >> (element * 4)) & 0xFu);
      s_weight[row * 16 + word_in_row * 8 + element] =
          __float2bfloat16(static_cast<float>(q - zp) * scale);
    }
    __syncthreads();

    nvcuda::wmma::load_matrix_sync(a_fragment, s_activation, 16);
    nvcuda::wmma::load_matrix_sync(
        b_fragment, s_weight + warp * 16 * 16, 16);
    nvcuda::wmma::mma_sync(
        accumulator, a_fragment, b_fragment, accumulator);
    __syncthreads();
  }

  nvcuda::wmma::store_matrix_sync(
      s_accumulator[warp], accumulator, 16, nvcuda::wmma::mem_row_major);
  __syncthreads();
  for (int index = tid; index < 16 * 64; index += blockDim.x) {
    const int m = index / 64;
    const int n_local = index % 64;
    const int accumulator_warp = n_local / 16;
    const int accumulator_col = n_local % 16;
    const int n = n_tile * 64 + n_local;
    if (n < out_cols) {
      output[static_cast<int64_t>(m_base + m) * out_cols + n] =
          __float2bfloat16(
              s_accumulator[accumulator_warp][m * 16 + accumulator_col]);
    }
  }
}
#endif

// Reconstructs a bounded output-row tile from the same physical layout for
// non-M512 prefill tails. This feeds the established 32 MiB dequant + cuBLAS
// path without ever reinterpreting a repacked buffer as canonical HF layout.
__global__ void qwen35_dequant_w4a16_bf16_rows_repacked_v1_kernel(
    const int32_t* weight_qwords, const __nv_bfloat16* weight_scale,
    const int32_t* weight_zero_point, __nv_bfloat16* dense, int in_cols,
    int out_cols, int groups, int row_start, int row_count) {
  const int local_row = blockIdx.x;
  if (local_row >= row_count) return;
  const int row = row_start + local_row;
  if (row >= out_cols) return;
  const int n_tile = row / 64;
  const int row_in_tile = row % 64;
  const int k_subtiles = in_cols / 16;
  const int words_per_row = in_cols / 8;
  const int64_t dense_base = static_cast<int64_t>(local_row) * in_cols;
  for (int word = threadIdx.x; word < words_per_row; word += blockDim.x) {
    const int k_subtile = word / 2;
    const int word_in_row = word & 1;
    const int group = k_subtile / 2;
    const int64_t qword_index =
        ((static_cast<int64_t>(n_tile) * k_subtiles + k_subtile) * 64 +
         row_in_tile) * 2 + word_in_row;
    const uint32_t qword = static_cast<uint32_t>(weight_qwords[qword_index]);
    const float scale = __bfloat162float(weight_scale[
        (static_cast<int64_t>(n_tile) * groups + group) * 64 + row_in_tile]);
    const uint32_t zp_word = static_cast<uint32_t>(weight_zero_point[
        (static_cast<int64_t>(n_tile) * groups + group) * 8 + row_in_tile / 8]);
    const int zp =
        static_cast<int>((zp_word >> ((row_in_tile & 7) * 4)) & 0xFu);
    __nv_bfloat16 values[8];
#pragma unroll
    for (int element = 0; element < 8; ++element) {
      const int q = static_cast<int>((qword >> (element * 4)) & 0xFu);
      values[element] =
          __float2bfloat16(static_cast<float>(q - zp) * scale);
    }
    *reinterpret_cast<float4*>(dense + dense_base + word * 8) =
        *reinterpret_cast<const float4*>(values);
  }
}
