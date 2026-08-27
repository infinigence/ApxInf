#pragma once

// Copyright 2026 apxinf contributors.
// Pure CUDA operators grouped by physical operation; launch policy lives under adapters/.

// ── Argmax over [vocab] bf16 logits → u32 token id ─────────────────────────
// Exact CPU-greedy semantics: strict `>`, lowest-index finite ties, NaNs never
// replace a candidate, and an all-NaN/-inf row returns zero.
struct ArgmaxBf16Pair { float value; uint32_t index; };

__device__ __forceinline__ ArgmaxBf16Pair argmax_bf16_choose(
    ArgmaxBf16Pair current, ArgmaxBf16Pair candidate)
{
    if (candidate.value > current.value ||
        (candidate.value == current.value && candidate.index < current.index))
        return candidate;
    return current;
}

__device__ __forceinline__ ArgmaxBf16Pair argmax_bf16_warp_reduce(
    ArgmaxBf16Pair candidate)
{
    for (int offset = 16; offset > 0; offset >>= 1) {
        ArgmaxBf16Pair other = {
            __shfl_xor_sync(0xffffffff, candidate.value, offset),
            __shfl_xor_sync(0xffffffff, candidate.index, offset),
        };
        candidate = argmax_bf16_choose(candidate, other);
    }
    return candidate;
}

__global__ void argmax_bf16_partials_kernel(
    const __nv_bfloat16* logits, uint32_t n, ArgmaxBf16Pair* partials)
{
    uint32_t tid = threadIdx.x;
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + tid;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    ArgmaxBf16Pair candidate = {-INFINITY, 0};
    for (; i < n; i += stride) {
        float value = __bfloat162float(logits[i]);
        if (value > candidate.value) candidate = {value, (uint32_t)i};
    }
    candidate = argmax_bf16_warp_reduce(candidate);
    __shared__ ArgmaxBf16Pair warp_best[32];
    uint32_t warp = tid / 32;
    uint32_t lane = tid % 32;
    if (lane == 0) warp_best[warp] = candidate;
    __syncthreads();
    if (warp == 0) {
        candidate = tid < (blockDim.x + 31) / 32
            ? warp_best[tid] : ArgmaxBf16Pair{-INFINITY, 0};
        candidate = argmax_bf16_warp_reduce(candidate);
        if (lane == 0) partials[blockIdx.x] = candidate;
    }
}

// Single-launch exact reduction. Every block publishes one partial before
// incrementing `arrivals`; the last block observes all partials and writes the
// selected token. The counter is returned to zero for the next stream-ordered
// invocation, so its device address remains stable across decode steps.
__global__ void argmax_bf16_single_launch_kernel(
    const __nv_bfloat16* logits, uint32_t n, ArgmaxBf16Pair* partials,
    uint32_t* arrivals, uint32_t* out)
{
    uint32_t tid = threadIdx.x;
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + tid;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    ArgmaxBf16Pair candidate = {-INFINITY, 0};
    for (; i < n; i += stride) {
        float value = __bfloat162float(logits[i]);
        if (value > candidate.value) candidate = {value, (uint32_t)i};
    }
    candidate = argmax_bf16_warp_reduce(candidate);

    __shared__ ArgmaxBf16Pair warp_best[32];
    __shared__ bool is_last;
    uint32_t warp = tid / 32;
    uint32_t lane = tid % 32;
    if (lane == 0) warp_best[warp] = candidate;
    __syncthreads();
    if (warp == 0) {
        candidate = tid < (blockDim.x + 31) / 32
            ? warp_best[tid] : ArgmaxBf16Pair{-INFINITY, 0};
        candidate = argmax_bf16_warp_reduce(candidate);
        if (lane == 0) {
            partials[blockIdx.x] = candidate;
            __threadfence();
            is_last = atomicInc(arrivals, gridDim.x - 1) == gridDim.x - 1;
        }
    }
    __syncthreads();
    if (!is_last || warp != 0) return;

    candidate = {-INFINITY, 0};
    for (uint32_t partial = lane; partial < gridDim.x; partial += 32)
        candidate = argmax_bf16_choose(candidate, partials[partial]);
    candidate = argmax_bf16_warp_reduce(candidate);
    if (lane == 0) {
        *out = candidate.index;
        *arrivals = 0;
    }
}

// Fast finalize for the bounded partial count used by greedy decode. Each lane
// scans a strided subset before the warp reduction, so the launch remains one
// warp without assuming a particular vocabulary size.
__global__ void argmax_bf16_finalize_warp_kernel(
    const ArgmaxBf16Pair* partials, uint32_t count, uint32_t* out)
{
    uint32_t lane = threadIdx.x;
    ArgmaxBf16Pair candidate = {-INFINITY, 0};
    for (uint32_t i = lane; i < count; i += 32)
        candidate = argmax_bf16_choose(candidate, partials[i]);
    candidate = argmax_bf16_warp_reduce(candidate);
    if (lane == 0) *out = candidate.index;
}

__global__ void argmax_bf16_finalize_kernel(
    const ArgmaxBf16Pair* partials, uint32_t count, uint32_t* out)
{
    uint32_t tid = threadIdx.x;
    ArgmaxBf16Pair candidate = tid < count
        ? partials[tid] : ArgmaxBf16Pair{-INFINITY, 0};
    candidate = argmax_bf16_warp_reduce(candidate);
    __shared__ ArgmaxBf16Pair warp_best[32];
    uint32_t warp = tid / 32;
    uint32_t lane = tid % 32;
    if (lane == 0) warp_best[warp] = candidate;
    __syncthreads();
    if (warp == 0) {
        candidate = tid < (blockDim.x + 31) / 32
            ? warp_best[tid] : ArgmaxBf16Pair{-INFINITY, 0};
        candidate = argmax_bf16_warp_reduce(candidate);
        if (lane == 0) *out = candidate.index;
    }
}




