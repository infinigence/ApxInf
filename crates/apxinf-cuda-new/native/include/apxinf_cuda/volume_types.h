#pragma once
#include "gemm_types.h"

#define APXINF_UNFOLD3D_SPEC_VERSION 1u
typedef struct {
  uint32_t version, reserved;
  int64_t input[4];
  int64_t history_frames;
  int64_t kernel[3], stride[3], padding_before[3], padding_after[3], output[3];
  int64_t row_start, rows;
} apxinf_unfold3d_spec_t;

typedef struct {
  const void* input;
  const void* history;
  void* output;
  apxinf_cuda_stream_t stream;
} apxinf_unfold3d_bindings_t;

// BF16 cross-correlation: THWC input/history, immutable [KT*KH*KW*Cin,Cout]
// weights, THWC output. FP32 accumulation and one BF16 output rounding.
#define APXINF_CONV3D_SPEC_VERSION 1u
typedef struct {
  apxinf_unfold3d_spec_t geometry;
  int64_t output_channels;
} apxinf_conv3d_spec_t;
typedef struct {
  apxinf_unfold3d_bindings_t volume;
  const void* weight;
} apxinf_conv3d_bindings_t;
typedef struct {
  uint64_t workspace_limit;
  uint32_t version, online_tune, allow_fallback, reserved;
  const char* cache_dir;
} apxinf_conv3d_policy_t;
