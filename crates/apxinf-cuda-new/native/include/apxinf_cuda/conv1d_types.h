#pragma once

#include <stdint.h>

#include "gemm_types.h"

/* One-dimensional convolution family in channel-last layout.
 *
 *   CONV1D           input [batch, in_length, in_channels]
 *                    weight [out_channels, in_channels / groups, kernel]
 *                    bias [out_channels]
 *                    output [batch, out_length, out_channels]
 *
 *   CONV_TRANSPOSE1D input [batch, in_length, in_channels]
 *                    weight [in_channels, out_channels / groups, kernel]
 *                    bias [out_channels]
 *                    output [batch, out_length, out_channels]
 *
 * out_length follows the PyTorch formulas:
 *   conv1d:           (L + 2*padding - dilation*(kernel-1) - 1) / stride + 1
 *   conv_transpose1d: (L - 1)*stride - 2*padding + dilation*(kernel-1)
 *                     + output_padding + 1
 */
typedef enum {
  APXINF_CONV1D_SEMANTIC_CONV1D = 0,
  APXINF_CONV1D_SEMANTIC_CONV_TRANSPOSE1D = 1,
} apxinf_conv1d_semantic_t;

#define APXINF_CONV1D_SPEC_VERSION 1u

typedef struct {
  uint32_t version;
  uint32_t semantic;
  uint32_t dtype;
  uint32_t has_bias;
  uint32_t input_alignment;
  uint32_t weight_alignment;
  uint32_t bias_alignment;
  uint32_t output_alignment;
  int64_t batch;
  int64_t in_length;
  int64_t in_channels;
  int64_t out_channels;
  int64_t kernel;
  int64_t stride;
  int64_t padding;
  int64_t dilation;
  int64_t groups;
  int64_t output_padding;
  int64_t out_length;
} apxinf_conv1d_spec_t;

typedef struct {
  const void* input;
  const void* weight;
  const void* bias;
  void* output;
  apxinf_cuda_stream_t stream;
} apxinf_conv1d_bindings_t;
