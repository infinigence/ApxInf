#pragma once

// Derived in part from `crates/apxinf-cuda/kernels/custom/` (Apache-2.0,
// Copyright 2026 apxinf contributors); the differences are explained below.

// Attention-side kernels, a port of the parts of
// `crates/apxinf-cuda/kernels/custom/{fused,attention}.cuh` this port reaches.
//
// The split below lives in CUDA's `fused.cuh` (as `qkv_split_bias_bf16_kernel`)
// while its caller is `attention::split_qkv_bias_bf16`, and
// the port's own notes recorded it under
// `attention.cuh` for that reason. Which file a CUDA kernel sits in is not a
// contract; this crate groups by physical operation, so the split is here with
// the attention operators rather than in `fused.cuh` with the fused ones.
//
// The attention operator itself is **not here**. `attention::mqa_bf16` runs the
// MP22 assembly in `kernels/asm/` — a different kind of artifact, assembled by
// `llvm-mc` and loaded as a driver module rather than compiled by `mcc` into
// this archive. What lives in this file is the QKV split, which shares the
// facade's attention module but not the assembly's mechanism, and
// `attention::mha_bf16`, whose 16 heads of dimension 72 are a shape that
// assembly family does not cover.

// ── Vision attention's data-movement helpers ────────────────────────────────
//
// The vision attention runs on mublas rather than on the kernel above: its two
// contractions are `[T,72] @ [72,T]` and `[T,T] @ [T,72]`, which are GEMM
// shapes, and a hand-written kernel lost to the GEMM by 3.9x
// when it was measured. Two kernels remain here, for
// what the GEMMs cannot do themselves. One of them -- the permute -- has three
// entry points, because the operands it moves do not all agree on a type; see
// the launchers in `adapters/custom_kernels.cu`.
//
// `permute_heads_kernel` exists because of a stride, not a layout: the
// model holds Q/K/V token-major as `[tokens, heads, 72]`, where stepping one
// head is 72 elements and stepping one batch is `tokens * 72`. A strided batch
// takes a single stride, so one of the two axes would have to be wrong. The
// permuted `[batch * heads, tokens_per_batch, 72]` makes both contiguous.
// It runs in both directions -- the same index map, applied to whichever side
// is the source.
template <typename In, typename Out>
__global__ void permute_heads_kernel(const In* in, Out* out, int tokens,
                                     int heads, int dim, int tokens_per_batch,
                                     int to_head_major) {
  // One block per token-major row. A row is `heads * dim` contiguous elements
  // that scatter into `heads` runs of `dim` in the other layout, so reading it
  // in order keeps the loads coalesced and leaves only the stores strided by
  // row. The obvious index-parallel formulation does the opposite -- coalesced
  // stores, loads strided by `heads * dim` -- and runs an order of magnitude
  // slower than the memory system allows.
  //
  // The head and the intra-head offset are separate loop variables rather than
  // `i / dim` and `i % dim`: the destination of a run of `dim` elements is one
  // contiguous span, so the divide bought nothing and cost a software routine's
  // worth of instructions per element on M1000.
  // One warp per head. The head loop that this replaced walked the heads
  // inside a 256-thread block, so with `dim` 72 only 72 of those 256 lanes were
  // ever doing work; a flat index over `heads * dim` -- every lane busy, head
  // from a division -- measured no better (and neither did giving each block
  // several rows). A warp per head keeps all 32 lanes busy with no division,
  // and each lane moves `ceil(dim/32)` elements. `permute_fix.cu`: 0.143 ->
  // 0.094 ms per call on the vision shape, 1.5x.
  const int64_t row = blockIdx.x;
  const int batch = static_cast<int>(row / tokens_per_batch);
  const int token = static_cast<int>(row % tokens_per_batch);
  const int width = heads * dim;
  const int64_t row_base = row * width;
  const int warps = blockDim.x >> 5;
  const int lane = threadIdx.x & 31;
  for (int head = threadIdx.x >> 5; head < heads; head += warps) {
    const int64_t head_major =
        (static_cast<int64_t>(batch * heads + head) * tokens_per_batch + token) * dim;
    const int64_t token_major = row_base + static_cast<int64_t>(head) * dim;
    for (int d = lane; d < dim; d += 32) {
      if (to_head_major) {
        out[head_major + d] = static_cast<Out>(in[token_major + d]);
      } else {
        out[token_major + d] = static_cast<Out>(in[head_major + d]);
      }
    }
  }
}

// The block size `permute_heads_kernel` is launched with, at all of its entry
// points. It is deliberately not `kThreads`: a row is `heads * dim` elements
// scattered into `heads` runs of `dim`, and the kernel keeps one warp per head,
// so the warps a larger block adds bring scheduling rather than lanes. On the
// vision shape `[512, 16, 72]`: 64 threads 0.1029 ms, 128 0.0855, 256
// (`kThreads`) 0.1016, 512 0.1347 (the port's row-block-size sweep).
constexpr int kPermuteThreads = 128;

// Row softmax with the max subtracted, in f32 end to end -- the same contract
// the eager bridge's `torch.softmax` over f32 scores had, so the two differ
// only in summation order. **One warp per row**, striding the columns.
//
// This used to be one 256-thread block per row with shared-memory reduction
// trees: one element per thread, and ~18 `__syncthreads()` per row over 8192
// rows. `softmax_variants.cu` measured three shapes of the same work --
//
//   block tree, 256 threads/row   0.866 ms   29.1 GB/s
//   warp shuffle, 256 threads/row 1.012 ms   24.9 GB/s
//   one warp per row, no barriers 0.331 ms   76.0 GB/s
//
// -- so the barriers were not the whole cost; giving each lane eight elements
// and dropping shared memory altogether was. The maximum is unchanged
// (`fmaxf` is commutative and associative, so its reduction order does not
// matter); only the denominator's summation order differs, which is the one
// thing the contract above already leaves free.
__global__ void row_softmax_f32_kernel(float* data, int rows, int cols) {
  const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (row >= rows) return;
  float* target = data + static_cast<int64_t>(row) * cols;

  float local_max = -3.402823466e38f;
  for (int col = lane; col < cols; col += 32) local_max = fmaxf(local_max, target[col]);
  for (int offset = 16; offset > 0; offset >>= 1) {
    local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, offset));
  }
  const float maximum = local_max;

  float local_sum = 0.0f;
  for (int col = lane; col < cols; col += 32) local_sum += expf(target[col] - maximum);
  for (int offset = 16; offset > 0; offset >>= 1) {
    local_sum += __shfl_xor_sync(0xffffffffu, local_sum, offset);
  }
  const float inverse = 1.0f / local_sum;

  for (int col = lane; col < cols; col += 32) {
    target[col] = expf(target[col] - maximum) * inverse;
  }
}

// Split a packed `[tokens, 3 * projection_width]` projection into contiguous
// q | k | v segments, adding an optional per-column bias, rounded once.
//
// One block per token, because the three outputs are three different buffers:
// the segment a column belongs to decides where it goes, and keeping a whole
// token in one block makes that a branch rather than a race.
__global__ void qkv_split_bias_bf16_kernel(const __mt_bfloat16* qkv,
                                           const __mt_bfloat16* bias,
                                           __mt_bfloat16* q, __mt_bfloat16* k,
                                           __mt_bfloat16* v, int projection_width) {
  const int token = blockIdx.x;
  const int fused_width = 3 * projection_width;
  for (int col = threadIdx.x; col < fused_width; col += blockDim.x) {
    float value = __bfloat162float(qkv[static_cast<int64_t>(token) * fused_width + col]);
    if (bias != nullptr) value += __bfloat162float(bias[col]);
    if (col < projection_width) {
      q[static_cast<int64_t>(token) * projection_width + col] =
          __float2bfloat16(value);
    } else if (col < 2 * projection_width) {
      k[static_cast<int64_t>(token) * projection_width + col - projection_width] =
          __float2bfloat16(value);
    } else {
      v[static_cast<int64_t>(token) * projection_width + col - 2 * projection_width] =
          __float2bfloat16(value);
    }
  }
}

// Vectorized form of the above: eight columns per thread in one 16-byte load and
// one 16-byte store.
//
// `[512, 3456]` -- the vision tower's QKV split -- measures 0.1638 ms (43.3
// GB/s) scalar with the bias and 0.1327 (53.4) without it, so a fifth of the
// scalar kernel is the bias test alone. `HAS_BIAS` is therefore a compile-time
// parameter, like the register-cached kernels in `fused.cuh`. Requires
// `projection_width % 8 == 0`, which is what keeps a lane inside one segment:
// the three-way segment branch then runs once per eight columns rather than
// once per column, and never straddles a `q | k | v` boundary. The launcher
// checks that and the pointer alignment.
//
// The arithmetic and the single BF16 rounding are unchanged from the scalar
// kernel, so the two are bit-identical.
template <bool HAS_BIAS>
__global__ void qkv_split_bias_bf16_vector8_kernel(
    const __mt_bfloat16* qkv, const __mt_bfloat16* bias, __mt_bfloat16* q,
    __mt_bfloat16* k, __mt_bfloat16* v, int projection_width) {
  const int token = blockIdx.x;
  const int64_t fused_width = 3LL * projection_width;
  const int64_t token_base = static_cast<int64_t>(token) * projection_width;
  const int width = static_cast<int>(fused_width / 8);
  for (int lane = threadIdx.x; lane < width; lane += blockDim.x) {
    const int64_t column = static_cast<int64_t>(lane) * 8;
    const uint4 qkv_raw =
        *reinterpret_cast<const uint4*>(qkv + static_cast<int64_t>(token) * fused_width +
                                       column);
    uint4 bias_raw;
    if (HAS_BIAS) bias_raw = *reinterpret_cast<const uint4*>(bias + column);
    const __mt_bfloat16* packed_qkv =
        reinterpret_cast<const __mt_bfloat16*>(&qkv_raw);
    const __mt_bfloat16* packed_bias =
        reinterpret_cast<const __mt_bfloat16*>(&bias_raw);
    uint4 output_raw;
    __mt_bfloat16* packed_output = reinterpret_cast<__mt_bfloat16*>(&output_raw);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float value = __bfloat162float(packed_qkv[i]);
      if (HAS_BIAS) value += __bfloat162float(packed_bias[i]);
      packed_output[i] = __float2bfloat16(value);
    }
    if (column < projection_width) {
      *reinterpret_cast<uint4*>(q + token_base + column) = output_raw;
    } else if (column < 2 * projection_width) {
      *reinterpret_cast<uint4*>(k + token_base + column - projection_width) =
          output_raw;
    } else {
      *reinterpret_cast<uint4*>(v + token_base + column - 2 * projection_width) =
          output_raw;
    }
  }
}
