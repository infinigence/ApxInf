#include "../../kernels/embedding/fixed_bf16.cuh"

extern "C" cudaError_t apxinf_preprocess_fourier_features_bf16(
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

extern "C" cudaError_t apxinf_embedding_sinusoidal_embedding_bf16(const void* positions,void* output,
    int rows,int dim,float scale,float frequency_step,cudaStream_t stream) {
  if(!positions||!output||rows<=0||dim<=0||dim%2||!std::isfinite(scale)||!std::isfinite(frequency_step))
    return cudaErrorInvalidValue;
  sinusoidal_embedding_bf16_kernel<<<rows,128,0,stream>>>(
      (const float*)positions,(__nv_bfloat16*)output,dim,scale,frequency_step);
  return cudaGetLastError();
}
