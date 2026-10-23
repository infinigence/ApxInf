// Copyright 2026 ApxInf contributors.
// Fixed implementation for this operator family; ported without numerical changes.
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace {
#include "../../kernels/custom/gdn_policy.h"
#include "../../kernels/custom/math.cuh"
#include "../../kernels/custom/reduction.cuh"
#include "../../kernels/custom/pillow_bicubic_u8.cuh"
#include "../../kernels/custom/linear_attention.cuh"
#include "../../kernels/custom/gdn_raw_inverse_f1.cuh"
#include "../../kernels/custom/gdn_chunk_state_wmma.cuh"
#include "../../kernels/custom/gdn_chunk_gemm_tri.cuh"
__global__ void sinusoidal_embedding_bf16_kernel(const float* positions,__nv_bfloat16* out,
    int dim,float scale,float frequency_step) {
  const int row=blockIdx.x,half=dim/2;
  for(int i=threadIdx.x;i<half;i+=blockDim.x) {
    const float frequency=expf(__fmul_rn(-frequency_step,static_cast<float>(i)));
    const float angle=__fmul_rn(__fmul_rn(scale,positions[row]),frequency);
    const int64_t base=static_cast<int64_t>(row)*dim;
    out[base+i]=__float2bfloat16(sinf(angle));
    out[base+half+i]=__float2bfloat16(cosf(angle));
  }
}


}  // namespace

extern "C" cudaError_t apxinf_hybrid_fourier_features_bf16(
    const void* waypoints, const void* freqs, void* out,
    int rows, int point_dim, int num_features, cudaStream_t stream) {
  if (waypoints == nullptr || freqs == nullptr || out == nullptr ||
      rows <= 0 || point_dim <= 0 || num_features <= 0) {
    return cudaErrorInvalidValue;
  }
  fourier_features_bf16_kernel<<<rows, 64, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(waypoints),
      static_cast<const __nv_bfloat16*>(freqs),
      static_cast<__nv_bfloat16*>(out), point_dim, num_features);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_hybrid_sinusoidal_embedding_bf16(const void* positions,void* output,
    int rows,int dim,float scale,float frequency_step,cudaStream_t stream) {
  if(!positions||!output||rows<=0||dim<=0||dim%2||!std::isfinite(scale)||!std::isfinite(frequency_step))
    return cudaErrorInvalidValue;
  sinusoidal_embedding_bf16_kernel<<<rows,128,0,stream>>>(
      (const float*)positions,(__nv_bfloat16*)output,dim,scale,frequency_step);
  return cudaGetLastError();
}
