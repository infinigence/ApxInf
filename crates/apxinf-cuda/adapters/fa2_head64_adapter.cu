// Head-64 noncausal BF16 specialization of the existing licensed FA2 provider.
// PyTorch's reference build uses separate multiply/add in the softmax exponent.
#define UNFUSE_FMA
#include "flash_attn/namespace_config.h"
#include "flash_attn/flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

using Head64Traits = Flash_fwd_kernel_traits<64, 64, 256, 4, false, false, cutlass::bfloat16_t>;

template<bool Split>
cudaError_t launch_head64(Flash_fwd_params& params, cudaStream_t stream) {
  auto kernel = &flash_fwd_splitkv_kernel<Head64Traits, false, false, false, false, true, false, Split, false>;
  cudaStreamCaptureStatus capture;
  auto status = cudaStreamIsCapturing(stream, &capture);
  if (status != cudaSuccess) return status;
  if (Head64Traits::kSmemSize >= 48 * 1024 && capture == cudaStreamCaptureStatusNone) {
    status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Head64Traits::kSmemSize);
    if (status != cudaSuccess) return status;
  }
  dim3 grid((params.seqlen_q + 63) / 64, Split ? params.num_splits : params.b,
            Split ? params.b * params.h : params.h);
  kernel<<<grid, Head64Traits::kNThreads, Head64Traits::kSmemSize, stream>>>(params);
  return cudaGetLastError();
}

int run_bf16_head64_splitkv(Flash_fwd_params& params, cudaStream_t stream) {
  if (params.d != 64 || params.is_causal || params.num_splits < 1 || params.num_splits > 128)
    return static_cast<int>(cudaErrorInvalidValue);
  if (params.num_splits == 1) return static_cast<int>(launch_head64<false>(params, stream));
  auto status = launch_head64<true>(params, stream);
  if (status != cudaSuccess) return static_cast<int>(status);
  dim3 grid((params.b * params.h * params.seqlen_q + 7) / 8);
#define COMBINE(LOG) flash_fwd_splitkv_combine_kernel<Head64Traits, 8, LOG, true><<<grid, Head64Traits::kNThreads, 0, stream>>>(params)
  if (params.num_splits <= 2) { COMBINE(1); }
  else if (params.num_splits <= 4) { COMBINE(2); }
  else if (params.num_splits <= 8) { COMBINE(3); }
  else if (params.num_splits <= 16) { COMBINE(4); }
  else if (params.num_splits <= 32) { COMBINE(5); }
  else if (params.num_splits <= 64) { COMBINE(6); }
  else { COMBINE(7); }
#undef COMBINE
  return static_cast<int>(cudaGetLastError());
}

} // namespace FLASH_NAMESPACE
