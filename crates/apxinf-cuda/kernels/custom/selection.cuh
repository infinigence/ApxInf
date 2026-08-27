#pragma once

// Copyright 2026 apxinf contributors.
// Pure CUDA operators grouped by physical operation; launch policy lives under adapters/.

// ── Argmax over [vocab] bf16 logits → u32 token id ─────────────────────────
//
// One block, many threads. Strided load and warp-shuffle max reduction carry
// both value and index. Equal values select the lowest index, matching the
// canonical CPU scan's strict `>` update rule, including signed-zero ties.
// NaNs never replace the initial -infinity value. Writes the winning index to
// `out` (typically a host-mapped u32, so the CPU reads it zero-copy).
__global__ void argmax_bf16_kernel(
    const __nv_bfloat16* logits, uint32_t n, uint32_t* out)
{
    uint32_t tid = threadIdx.x;
    float best_v = -INFINITY;
    uint32_t best_i = 0;
    for (uint32_t i = tid; i < n; i += blockDim.x) {
        float v = __bfloat162float(logits[i]);
        if (v > best_v) { best_v = v; best_i = i; }
    }
    // Warp reduce: keep the larger value, then the lower index on a tie.
    for (int off = 16; off > 0; off >>= 1) {
        float other_v = __shfl_xor_sync(0xffffffff, best_v, off);
        uint32_t other_i = __shfl_xor_sync(0xffffffff, best_i, off);
        if (other_v > best_v || (other_v == best_v && other_i < best_i)) {
            best_v = other_v;
            best_i = other_i;
        }
    }
    uint32_t warp_id = tid / 32;
    uint32_t lane = tid % 32;
    __shared__ float warp_best_v[32];
    __shared__ uint32_t warp_best_i[32];
    if (lane == 0) {
        warp_best_v[warp_id] = best_v;
        warp_best_i[warp_id] = best_i;
    }
    __syncthreads();
    if (warp_id == 0) {
        uint32_t warp_count = (blockDim.x + 31) / 32;
        best_v = tid < warp_count ? warp_best_v[tid] : -INFINITY;
        best_i = tid < warp_count ? warp_best_i[tid] : 0;
        for (int off = 16; off > 0; off >>= 1) {
            float other_v = __shfl_xor_sync(0xffffffff, best_v, off);
            uint32_t other_i = __shfl_xor_sync(0xffffffff, best_i, off);
            if (other_v > best_v || (other_v == best_v && other_i < best_i)) {
                best_v = other_v;
                best_i = other_i;
            }
        }
        if (lane == 0) *out = best_i;
    }
}

struct ArgmaxPair {
    float value;
    uint32_t index;
};

constexpr uint32_t APXINF_ARGMAX_PARTIAL_BLOCKS = 128;

__device__ __forceinline__ bool argmax_pair_better(
    float value, uint32_t index, float best_value, uint32_t best_index)
{
    return value > best_value || (value == best_value && index < best_index);
}

__global__ void argmax_bf16_partials_kernel(
    const __nv_bfloat16* logits, uint32_t n, ArgmaxPair* partials)
{
    uint32_t lane = threadIdx.x & 31;
    uint32_t warp = threadIdx.x >> 5;
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    float best_value = -INFINITY;
    uint32_t best_index = 0;
    for (uint32_t i = index; i < n; i += stride) {
        float value = __bfloat162float(logits[i]);
        if (value > best_value) {
            best_value = value;
            best_index = i;
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        float value = __shfl_xor_sync(0xffffffff, best_value, offset);
        uint32_t i = __shfl_xor_sync(0xffffffff, best_index, offset);
        if (argmax_pair_better(value, i, best_value, best_index)) {
            best_value = value;
            best_index = i;
        }
    }
    __shared__ float warp_values[32];
    __shared__ uint32_t warp_indices[32];
    if (lane == 0) {
        warp_values[warp] = best_value;
        warp_indices[warp] = best_index;
    }
    __syncthreads();
    if (warp == 0) {
        uint32_t warps = (blockDim.x + 31) / 32;
        best_value = threadIdx.x < warps ? warp_values[threadIdx.x] : -INFINITY;
        best_index = threadIdx.x < warps ? warp_indices[threadIdx.x] : 0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            float value = __shfl_xor_sync(0xffffffff, best_value, offset);
            uint32_t i = __shfl_xor_sync(0xffffffff, best_index, offset);
            if (argmax_pair_better(value, i, best_value, best_index)) {
                best_value = value;
                best_index = i;
            }
        }
        if (lane == 0) partials[blockIdx.x] = {best_value, best_index};
    }
}

__global__ void argmax_pair_final_kernel(
    const ArgmaxPair* partials, uint32_t count, uint32_t* output)
{
    uint32_t lane = threadIdx.x & 31;
    uint32_t warp = threadIdx.x >> 5;
    float best_value = -INFINITY;
    uint32_t best_index = 0;
    for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
        ArgmaxPair candidate = partials[i];
        if (argmax_pair_better(
                candidate.value, candidate.index, best_value, best_index)) {
            best_value = candidate.value;
            best_index = candidate.index;
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        float value = __shfl_xor_sync(0xffffffff, best_value, offset);
        uint32_t i = __shfl_xor_sync(0xffffffff, best_index, offset);
        if (argmax_pair_better(value, i, best_value, best_index)) {
            best_value = value;
            best_index = i;
        }
    }
    __shared__ float warp_values[32];
    __shared__ uint32_t warp_indices[32];
    if (lane == 0) {
        warp_values[warp] = best_value;
        warp_indices[warp] = best_index;
    }
    __syncthreads();
    if (warp == 0) {
        uint32_t warps = (blockDim.x + 31) / 32;
        best_value = threadIdx.x < warps ? warp_values[threadIdx.x] : -INFINITY;
        best_index = threadIdx.x < warps ? warp_indices[threadIdx.x] : 0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            float value = __shfl_xor_sync(0xffffffff, best_value, offset);
            uint32_t i = __shfl_xor_sync(0xffffffff, best_index, offset);
            if (argmax_pair_better(value, i, best_value, best_index)) {
                best_value = value;
                best_index = i;
            }
        }
        if (lane == 0) *output = best_index;
    }
}


