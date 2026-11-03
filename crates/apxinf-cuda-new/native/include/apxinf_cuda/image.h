#pragma once
#include "gemm_types.h"

// Fixed, allocation-free HWC image primitives. Every tensor is contiguous.
// 0: im2col [OH*OW,KH*KW*C], 1: space-to-depth [H/f*W/f,C*f*f]
// 2: grouped space/time average of a single frame left-padded with zeros
// 3: row L2 normalization, then scale and per-channel gamma (dtype rounding)
// 4: select leading output_channels and standardize (x-mean)/std to F32
typedef struct {
  uint32_t version, semantic, dtype, output_dtype;
  uint32_t height, width, channels;
  uint32_t output_height, output_width, output_channels;
  uint32_t kernel, stride, padding, temporal_factor;
} apxinf_image_spec_t;

typedef struct {
  const void *input, *parameter, *parameter2;
  void *output;
  float scale, epsilon;
  apxinf_cuda_stream_t stream;
} apxinf_image_bindings_t;

// Fixed u8 -> I32 conversion, concatenating view images along width. Layout
// 0 is [V,H,W,C], 1 is [V,C,H,W]; output is flat [H,V*W,C].
typedef struct {
  uint32_t version, layout, views, height, width, channels;
} apxinf_image_rgb_indices_spec_t;
typedef struct {
  const uint8_t* input;
  int32_t* output;
  apxinf_cuda_stream_t stream;
} apxinf_image_rgb_indices_bindings_t;

#ifdef __cplusplus
extern "C" {
#endif
apxinf_status_t apxinf_image_launch(apxinf_runtime_t runtime,
    const apxinf_image_spec_t* spec, const apxinf_image_bindings_t* bindings);
apxinf_status_t apxinf_image_rgb_indices_launch(apxinf_runtime_t runtime,
    const apxinf_image_rgb_indices_spec_t* spec,
    const apxinf_image_rgb_indices_bindings_t* bindings);
#ifdef __cplusplus
}
#endif
