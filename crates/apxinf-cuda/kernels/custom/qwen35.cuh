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
  return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float siluf_f32(float x) {
  return x / (1.0f + __expf(-x));
}

__device__ __forceinline__ float softplusf_f32(float x) {
  return x > 20.0f ? x : __logf(1.0f + __expf(x));
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
    const float decay = __expf(-__expf(a_log_h) * softplusf_f32(a_h + dt_bias_h));
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

// ── Fast row-major W4A16 dequant (prefill path) ────────────────────────────
//
// One block per output row; no per-element integer division. Each thread
// unpacks whole int32 words (8 nibbles) and stores 8 bf16 values vectorized.

__global__ void qwen35_dequant_w4a16_bf16_row_kernel(
    const int32_t* weight_packed, const __nv_bfloat16* weight_scale,
    const int32_t* weight_zero_point, __nv_bfloat16* dense, int in_cols,
    int out_cols, int groups) {
  const int row = blockIdx.x;
  if (row >= out_cols) return;
  const int packed_cols = (in_cols + 7) / 8;
  const int group_size = (in_cols + groups - 1) / groups;
  const int zp_row = row / 8;
  const int zp_shift = (row & 7) * 4;
  const int64_t row_base = static_cast<int64_t>(row) * packed_cols;
  const int64_t out_base = static_cast<int64_t>(row) * in_cols;
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
    const float p = __expf(dot - m_new);
    const float exp_m = __expf(m - m_new);
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
    l_global += s_l[w] * __expf(s_m[w] - m_global);
  for (int i = 0; i < elems; i++) {
    float acc_global = 0.0f;
    for (int w = 0; w < QWEN35_PREFILL_WARPS; w++)
      acc_global += s_acc[w][i][lane] * __expf(s_m[w] - m_global);
    const float inv_l = (l_global > 0.0f) ? (1.0f / l_global) : 0.0f;
    out[(s * heads + q_head) * head_dim + i * 32 + lane] =
        __float2bfloat16(acc_global * inv_l);
  }
}
