#pragma once
#include <cuda_bf16.h>
#include "../../include/apxinf_cuda/volume_types.h"

namespace apxinf::volume::kernels {
// Pure bit copies, including exact BF16 zero padding; no arithmetic rounding.
__global__ void unfold3d(apxinf_unfold3d_spec_t s, const __nv_bfloat16* input,
                       const __nv_bfloat16* history, __nv_bfloat16* output,
                       int64_t columns) {
  const int64_t count = s.rows * columns;
  for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
       i < count; i += int64_t(gridDim.x) * blockDim.x) {
    int64_t k = i % columns;
    const int64_t c = k % s.input[3]; k /= s.input[3];
    const int64_t kw = k % s.kernel[2]; k /= s.kernel[2];
    const int64_t kh = k % s.kernel[1]; k /= s.kernel[1];
    const int64_t kt = k;
    int64_t row = s.row_start + i / columns;
    const int64_t ow = row % s.output[2]; row /= s.output[2];
    const int64_t oh = row % s.output[1]; row /= s.output[1];
    const int64_t t = row * s.stride[0] + kt - s.padding_before[0];
    const int64_t h = oh * s.stride[1] + kh - s.padding_before[1];
    const int64_t w = ow * s.stride[2] + kw - s.padding_before[2];
    __nv_bfloat16 value = __float2bfloat16(0.0f);
    if (t >= 0 && t < s.input[0] + s.history_frames &&
        h >= 0 && h < s.input[1] && w >= 0 && w < s.input[2]) {
      const bool cached = t < s.history_frames;
      const int64_t local_t = cached ? t : t - s.history_frames;
      const int64_t offset = ((local_t * s.input[1] + h) * s.input[2] + w) * s.input[3] + c;
      value = cached ? history[offset] : input[offset];
    }
    output[i] = value;
  }
}
}  // namespace apxinf::volume::kernels
