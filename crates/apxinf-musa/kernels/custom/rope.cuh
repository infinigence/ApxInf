#pragma once

// Derived in part from `crates/apxinf-cuda/kernels/custom/` (Apache-2.0,
// Copyright 2026 apxinf contributors); the differences are explained below.

// Rotary position embedding, a port of
// `crates/apxinf-cuda/kernels/custom/fused.cuh`'s `qkv_rope_bf16_kernel`.
//
// One kernel serves both entry points: `rope::split_qkv_apply_bf16` writes q, k
// and v into fresh buffers, `rope::apply_q_write_kv_bf16` writes k and v into
// the first `tokens` rows of a caller-owned cache at `kv_output_offset`. They
// differ in where the two destinations point and in nothing else, which is why
// CUDA has one launcher behind both and why this ABI takes the offset.

// The bf16 value `value` rounds to, as an f32.
//
// The round trip is the arithmetic, not a storage detail: the reference rounds
// at each of these points and the recorded tensors depend on it. See the chain
// described on the kernel below.
__device__ __forceinline__ float round_bf16(float value) {
  return __bfloat162float(__float2bfloat16(value));
}

// The rope frequency table: `powf(theta, -pair / half_dim)` for every pair.
//
// One block is plenty -- `half_dim` is at most 512 -- and it is launched only
// when `(theta, half_dim)` changes, so the cost does not appear in a pass.
__global__ void rope_frequency_kernel(float* out, int half_dim, float theta) {
  const int pair = blockIdx.x * blockDim.x + threadIdx.x;
  if (pair < half_dim) {
    out[pair] = powf(theta, -static_cast<float>(pair) / half_dim);
  }
}

// One half-split pair, rotated with the reference's rounding chain: each
// product rounds to bf16, then the difference or sum rounds again.
//
// `__fmul_rn`/`__fsub_rn`/`__fadd_rn` rather than bare operators, because the
// compiler is otherwise free to contract `first * cosine - second * sine` into
// one FMA -- which skips exactly the rounding the bf16 intermediate exists to
// impose.
__device__ __forceinline__ void rope_pair_bf16(float first, float second,
                                               float cosine, float sine,
                                               __mt_bfloat16* destination,
                                               int half_dim, int pair) {
  const float first_cosine = round_bf16(__fmul_rn(first, cosine));
  const float second_sine = round_bf16(__fmul_rn(second, sine));
  const float second_cosine = round_bf16(__fmul_rn(second, cosine));
  const float first_sine = round_bf16(__fmul_rn(first, sine));
  destination[pair] = __float2bfloat16(__fsub_rn(first_cosine, second_sine));
  destination[half_dim + pair] =
      __float2bfloat16(__fadd_rn(second_cosine, first_sine));
}

// Split a packed `[tokens, (q_heads + 2 * kv_heads) * head_dim]` projection,
// add an optional per-column bias, rotate Q and K, and write all three parts.
//
// # The rounding chain is the contract
//
// This is **not** the arithmetic the CUDA kernel computes, and the difference is
// deliberate. The reference tensors were produced by the eager bridge
// (`_rope_and_split` in the torch bridge that recorded them; it is not part of
// this tree), which follows
// Gemma: the bias is added in f32, the result is rounded to bf16, `cos`/`sin`
// are rounded to bf16 *before* the rotation, and the rotation is then bf16
// arithmetic. CUDA's kernel keeps everything in f32 and rounds once at the end,
// which is more accurate and therefore different -- measured against the
// recorded q/k it disagrees on 35.6% of elements. So the rounds below are not to
// be simplified into a single conversion at the end.
//
// # Layout
//
// Pairs are half-split -- `(i, i + head_dim / 2)` -- not interleaved. The
// frequency is `theta^(-pair / half_dim)`, read from the table described above
// (the CUDA kernel and the bridge evaluate it inline; the values are the same),
// and the
// angle is built in f32 from the integer position just as the bridge builds it
// from an f32 arange. V is not rotated; it only takes the bias.
//
// # Launch
//
// `grid(tokens, q_heads + 2 * kv_heads)`, `block(head_dim / 2)`. One thread per
// pair, so `blockDim.x == half_dim` exactly; the bounds check is for the
// launcher's benefit rather than the kernel's.
__global__ void qkv_rope_bf16_kernel(const __mt_bfloat16* qkv,
                                     const __mt_bfloat16* bias,
                                     __mt_bfloat16* q, __mt_bfloat16* k,
                                     __mt_bfloat16* v, int q_heads, int kv_heads,
                                     int head_dim, const float* frequency,
                                     int position_offset,
                                     int kv_output_offset) {
  const int token = blockIdx.x;
  const int projection_head = blockIdx.y;
  const int half_dim = head_dim / 2;
  const int pair = threadIdx.x;
  if (pair >= half_dim) return;

  const int q_width = q_heads * head_dim;
  const int kv_width = kv_heads * head_dim;
  const int fused_width = q_width + 2 * kv_width;
  const int position = position_offset + token;
  // `frequency[pair]` is `powf(theta, -pair / half_dim)`, evaluated once into a
  // table instead of once per (token, head, pair).
  //
  // It depends only on `pair`, so the grid above recomputed the same 128 values
  // 5220 times on the prefix shape. Measured with `rope_variants.cu`, that one
  // call was **80% of this kernel**: 0.935 -> 0.184 ms per call at `[522, ...]`.
  // The table is filled by `rope_frequency_kernel` below, on the device, from
  // the same `powf` the kernel used to call -- so the values are identical and
  // this changes no numbers. A host-side table would not be: `math.cuh`'s note
  // about device and host transcendentals differing applies here too.
  const float rate = frequency[pair];
  const float cosine = round_bf16(cosf(static_cast<float>(position) * rate));
  const float sine = round_bf16(sinf(static_cast<float>(position) * rate));

  // The packed projection and its bias share one layout -- q | k | v, each
  // segment `heads * head_dim` wide -- so one offset serves both.
  int segment;
  if (projection_head < q_heads) {
    segment = projection_head * head_dim;
  } else if (projection_head < q_heads + kv_heads) {
    segment = q_width + (projection_head - q_heads) * head_dim;
  } else {
    segment =
        q_width + kv_width + (projection_head - q_heads - kv_heads) * head_dim;
  }

  // The flat offsets are 64-bit even though every shape PI0.5 reaches fits in
  // 32: `tokens` is only bounded by the tensor a caller passes, and at the
  // action expert's width a token count of ~840k makes a 32-bit `source`
  // negative -- a read before `qkv` rather than a wrong answer. The sibling
  // kernels in this port (`add_position`, `embedding`, `geglu`,
  // `qkv_split_bias`) all widen the same way. The CUDA original does not.
  //
  // `kv_output_offset + token` below stays 32-bit on purpose: that sum is a
  // row index into the cache, so it is bounded by an allocation that exists --
  // passing 2^31 would need a 32 GB cache. The products above have no such
  // bound, which is the difference.
  const int64_t source = static_cast<int64_t>(token) * fused_width + segment;
  float first = __bfloat162float(qkv[source + pair]);
  float second = __bfloat162float(qkv[source + half_dim + pair]);
  if (bias != nullptr) {
    first += __bfloat162float(bias[segment + pair]);
    second += __bfloat162float(bias[segment + half_dim + pair]);
  }

  if (projection_head < q_heads) {
    // Q is per-token output: `[tokens, q_heads, head_dim]`, no cache involved.
    const int64_t destination =
        (static_cast<int64_t>(token) * q_heads + projection_head) * head_dim;
    rope_pair_bf16(round_bf16(first), round_bf16(second), cosine, sine,
                   q + destination, half_dim, pair);
    return;
  }

  const bool is_key = projection_head < q_heads + kv_heads;
  const int head = is_key ? projection_head - q_heads
                          : projection_head - q_heads - kv_heads;
  const int64_t destination =
      (static_cast<int64_t>(kv_output_offset + token) * kv_heads + head) * head_dim;
  __mt_bfloat16* target = is_key ? k : v;
  if (is_key) {
    rope_pair_bf16(round_bf16(first), round_bf16(second), cosine, sine,
                   target + destination, half_dim, pair);
  } else {
    // V is carried through unrotated: the bias add rounds once and nothing else
    // happens to it, exactly as `_write(t["v"], v_block)` does.
    target[destination + pair] = __float2bfloat16(first);
    target[destination + half_dim + pair] = __float2bfloat16(second);
  }
}
