#pragma once
#include "../../include/apxinf_cuda/volume_types.h"
#include <cuda_bf16.h>
namespace apxinf::volume::kernels {
// Materialize only asymmetric padding; common symmetric padding remains in the
// convolution descriptor. History is the logical prefix of the current volume.
__global__ void conv3d_input(apxinf_unfold3d_spec_t s, int64_t pt, int64_t ph,
    int64_t pw, const __nv_bfloat16* input, const __nv_bfloat16* history,
    __nv_bfloat16* output, int64_t count) {
  for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
       i < count; i += int64_t(blockDim.x) * gridDim.x) {
    int64_t j = i, c = j % s.input[3]; j /= s.input[3];
    int64_t w = j % pw; j /= pw;
    int64_t h = j % ph, t = j / ph;
    t -= s.padding_before[0] - min(s.padding_before[0], s.padding_after[0]);
    h -= s.padding_before[1] - min(s.padding_before[1], s.padding_after[1]);
    w -= s.padding_before[2] - min(s.padding_before[2], s.padding_after[2]);
    __nv_bfloat16 value = __float2bfloat16(0.f);
    if (t >= 0 && t < s.input[0] + s.history_frames && h >= 0 && h < s.input[1] && w >= 0 && w < s.input[2]) {
      const bool old = t < s.history_frames;
      if (!old) t -= s.history_frames;
      value = (old ? history : input)[((t * s.input[1] + h) * s.input[2] + w) * s.input[3] + c];
    }
    output[i] = value;
  }
}
// Eight contiguous BF16 channels per transfer. The adapter proves all source
// addresses are16-byte aligned and total padded elements fit signed32 bits.
__global__ void conv3d_input_vec8(int frames, int height, int width, int channel_vectors,
    int history_frames, int padded_height, int padded_width,
    int before_t, int before_h, int before_w,
    const uint4* input, const uint4* history, uint4* output, int count) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < count; i += blockDim.x * gridDim.x) {
    int j = i, c = j % channel_vectors; j /= channel_vectors;
    int w = j % padded_width; j /= padded_width;
    int h = j % padded_height, t = j / padded_height;
    t -= before_t; h -= before_h; w -= before_w;
    uint4 value = {0, 0, 0, 0};
    if (t >= 0 && t < frames + history_frames && h >= 0 && h < height && w >= 0 && w < width) {
      const bool old = t < history_frames;
      if (!old) t -= history_frames;
      value = (old ? history : input)[((t * height + h) * width + w) * channel_vectors + c];
    }
    output[i] = value;
  }
}
__global__ void conv3d_weight(const __nv_bfloat16* source, __nv_bfloat16* target,
    int64_t k, int64_t n) {
  for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
       i < k * n; i += int64_t(blockDim.x) * gridDim.x)
    target[i] = source[(i % k) * n + i / k];
}
}
