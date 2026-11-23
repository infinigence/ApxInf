#pragma once

#include <stdint.h>

#include "gemm_types.h"

/* Two-dimensional convolution in channel-last (NHWC) layout.
 *
 *   input  [batch, in_h, in_w, in_channels]
 *   weight [out_channels, in_channels, kernel_h, kernel_w]
 *   bias   [out_channels]
 *   output [batch, out_h, out_w, out_channels]
 *
 * groups == 1 and dilation == 1 are implied. out_h / out_w follow PyTorch:
 *   out_h = (in_h + 2*pad_h - kernel_h) / stride_h + 1   (floor)
 *   out_w = (in_w + 2*pad_w - kernel_w) / stride_w + 1   (floor)
 */
#define APXINF_CONV2D_SPEC_VERSION 1u

typedef struct {
  uint32_t version;
  uint32_t dtype;
  uint32_t has_bias;
  uint32_t input_alignment;
  uint32_t weight_alignment;
  uint32_t bias_alignment;
  uint32_t output_alignment;
  uint32_t reserved;
  int64_t batch;
  int64_t in_h;
  int64_t in_w;
  int64_t in_channels;
  int64_t out_channels;
  int64_t kernel_h;
  int64_t kernel_w;
  int64_t stride_h;
  int64_t stride_w;
  int64_t pad_h;
  int64_t pad_w;
  int64_t out_h;
  int64_t out_w;
} apxinf_conv2d_spec_t;

typedef struct {
  const void* input;
  const void* weight;
  const void* bias;
  void* output;
  apxinf_cuda_stream_t stream;
} apxinf_conv2d_bindings_t;
