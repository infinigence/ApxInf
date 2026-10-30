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
