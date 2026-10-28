#pragma once

// Copyright 2026 apxinf contributors.
// X-VLA port operators: image normalization, convolution im2col, depthwise
// convolution, exact-GELU and sigmoid activations. Launch policy lives under
// adapters/ (see adapters/xvla_adapter.cu).

#include <cuda_bf16.h>
#include <cstdint>

// ── RGB u8 HWC → normalized BF16 token-major [H*W, 3] ─────────────────────

__global__ void xvla_rgb_u8_to_normalized_tokens_bf16_kernel(
    const unsigned char* rgb, __nv_bfloat16* output, int height, int width,
    float mean0, float mean1, float mean2, float std0, float std1, float std2)
{
    int64_t gid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t count = (int64_t)height * width;
    if (gid >= count) return;
    const unsigned char* pixel = rgb + gid * 3;
    __nv_bfloat16* out = output + gid * 3;
    const float scale = 1.0f / 255.0f;
    out[0] = __float2bfloat16(((float)pixel[0] * scale - mean0) / std0);
    out[1] = __float2bfloat16(((float)pixel[1] * scale - mean1) / std1);
    out[2] = __float2bfloat16(((float)pixel[2] * scale - mean2) / std2);
}

// ── im2col: token-major [H*W, C] input → [OH*OW, C*KH*KW] columns ─────────
//
// Column index for row = oy*out_w + ox is ((c*kh + ky)*kw + kx); out-of-bounds
// taps are zero (same convention as a zero-padded convolution).

__global__ void xvla_im2col_bf16_kernel(
    const __nv_bfloat16* input, __nv_bfloat16* columns, int channels,
    int height, int width, int kh, int kw, int stride_h, int stride_w,
    int pad_h, int pad_w, int out_h, int out_w)
{
    int64_t count = (int64_t)out_h * out_w * channels * kh * kw;
    int64_t gid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    const int kx = (int)(gid % kw);
    int64_t rest = gid / kw;
    const int ky = (int)(rest % kh);
    rest /= kh;
    const int channel = (int)(rest % channels);
    const int64_t row = rest / channels;
    const int ox = (int)(row % out_w);
    const int oy = (int)(row / out_w);
    const int iy = oy * stride_h - pad_h + ky;
    const int ix = ox * stride_w - pad_w + kx;
    float value = 0.0f;
    if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
        value = __bfloat162float(
            input[((int64_t)iy * width + ix) * channels + channel]);
    }
    columns[gid] = __float2bfloat16(value);
}

// ── Depthwise 3×3, stride 1, pad 1, token-major [H*W, C] ─────────────────

__global__ void xvla_dwconv3x3_bf16_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    const __nv_bfloat16* bias, __nv_bfloat16* output, int channels,
    int height, int width)
{
    int64_t count = (int64_t)height * width * channels;
    int64_t gid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    const int channel = (int)(gid % channels);
    const int64_t position = gid / channels;
    const int x = (int)(position % width);
    const int y = (int)(position / width);
    float accumulator = bias != nullptr ? __bfloat162float(bias[channel]) : 0.0f;
#pragma unroll
    for (int ky = 0; ky < 3; ++ky) {
        const int iy = y + ky - 1;
        if (iy < 0 || iy >= height) continue;
#pragma unroll
        for (int kx = 0; kx < 3; ++kx) {
            const int ix = x + kx - 1;
            if (ix < 0 || ix >= width) continue;
            const float value = __bfloat162float(
                input[((int64_t)iy * width + ix) * channels + channel]);
            const float tap = __bfloat162float(weight[channel * 9 + ky * 3 + kx]);
            accumulator = fmaf(value, tap, accumulator);
        }
    }
    output[gid] = __float2bfloat16(accumulator);
}

// ── Exact (erf) GELU and sigmoid ──────────────────────────────────────────

__global__ void xvla_gelu_erf_bf16_kernel(
    const __nv_bfloat16* input, __nv_bfloat16* output, int64_t count)
{
    int64_t gid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    const float x = __bfloat162float(input[gid]);
    output[gid] =
        __float2bfloat16(0.5f * x * (1.0f + erff(x * 0.70710678118654752440f)));
}

__global__ void xvla_sigmoid_bf16_kernel(
    const __nv_bfloat16* input, __nv_bfloat16* output, int64_t count)
{
    int64_t gid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    const float x = __bfloat162float(input[gid]);
    output[gid] = __float2bfloat16(1.0f / (1.0f + expf(-x)));
}

// ── Packed action tokens for the flow head ────────────────────────────────
//
// row = [action (dim_action) | proprio (dim_action) | time (dim_time)] with
// the two gripper columns of both action and proprio blocks zeroed, matching
// `EE6DActionSpace.preprocess`.

__global__ void xvla_pack_action_tokens_bf16_kernel(
    const __nv_bfloat16* action, const __nv_bfloat16* proprio,
    const __nv_bfloat16* time, __nv_bfloat16* output, int rows,
    int dim_action, int dim_time, int gripper0, int gripper1)
{
    const int total = dim_action * 2 + dim_time;
    int64_t count = (int64_t)rows * total;
    int64_t gid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    const int column = (int)(gid % total);
    const int64_t row = gid / total;
    const __nv_bfloat16 zero = __float2bfloat16(0.0f);
    if (column < dim_action) {
        output[gid] =
            (column == gripper0 || column == gripper1)
                ? zero
                : action[row * dim_action + column];
    } else if (column < dim_action * 2) {
        const int index = column - dim_action;
        output[gid] =
            (index == gripper0 || index == gripper1) ? zero : proprio[index];
    } else {
        output[gid] = time[column - dim_action * 2];
    }
}

// ── Sigmoid over two columns (ee6d gripper postprocess) ───────────────────

__global__ void xvla_sigmoid_columns_bf16_kernel(
    const __nv_bfloat16* input, __nv_bfloat16* output, int rows, int columns,
    int index0, int index1)
{
    int64_t count = (int64_t)rows * columns;
    int64_t gid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    const int column = (int)(gid % columns);
    if (column == index0 || column == index1) {
        const float value = __bfloat162float(input[gid]);
        output[gid] = __float2bfloat16(1.0f / (1.0f + expf(-value)));
    } else {
        output[gid] = input[gid];
    }
}

// ── Unbounded row gather (window partition needs duplicated/zero rows) ────

__global__ void xvla_gather_rows_bf16_kernel(
    const __nv_bfloat16* input, const unsigned int* indices,
    __nv_bfloat16* output, int64_t out_rows, int columns)
{
    int64_t count = out_rows * columns;
    int64_t gid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    const int64_t row = gid / columns;
    const int column = (int)(gid % columns);
    const unsigned int source = indices[row];
    output[gid] = input[(int64_t)source * columns + column];
}
