#pragma once
#include "runtime.h"
#include <stdint.h>

#define APXINF_CONVOLUTION_SPEC_VERSION 1u
// Valid (unpadded), ungrouped 3D cross-correlation. Contiguous NCDHW input,
// OIDHW filter, NODHW output; F16/BF16 storage and F32 accumulation.
typedef struct {
  uint32_t version;
  uint32_t dtype;
  int64_t input[5];
  int64_t weight[5];
  int64_t output[5];
  int64_t stride[3];
  uint32_t alignment[3];
} apxinf_convolution_spec_t;
typedef struct {
  const void* input;
  const void* weight;
  void* output;
  void* workspace;
  uint64_t workspace_bytes;
  apxinf_cuda_stream_t stream;
} apxinf_convolution_bindings_t;
typedef struct apxinf_convolution_execution* apxinf_convolution_execution_t;
#ifdef __cplusplus
extern "C" {
#endif
// Zero means headers were unavailable at build time or cuDNN 9 cannot load.
uint64_t apxinf_convolution_cudnn_version(void);
apxinf_status_t apxinf_convolution_prepare(apxinf_runtime_t,
    const apxinf_convolution_spec_t*, const apxinf_convolution_bindings_t*,
    int32_t configuration, apxinf_convolution_execution_t*);
apxinf_status_t apxinf_convolution_enqueue(apxinf_convolution_execution_t);
const char* apxinf_convolution_summary(apxinf_convolution_execution_t);
int32_t apxinf_convolution_configuration_count(apxinf_convolution_execution_t);
void apxinf_convolution_destroy(apxinf_convolution_execution_t);
#ifdef __cplusplus
}
#endif
