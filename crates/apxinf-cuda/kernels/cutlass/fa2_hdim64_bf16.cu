// ApxInf-owned BF16 head-dimension-64 dispatch over the unmodified upstream
// FlashAttention-2 templates. The official 128x128 tile remains the fallback.

#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>

#include "flash_attn/flash.h"
#include "flash_attn/flash_fwd_launch_template.h"
#include "flash_attn/namespace_config.h"

namespace apxinf::cuda::cutlass_ops {

bool use_mha_fwd_hdim64_bf16_apx(
    const FLASH_NAMESPACE::Flash_fwd_params& params) {
  int device = 0;
  cudaDeviceProp properties{};
  return cudaGetDevice(&device) == cudaSuccess &&
      cudaGetDeviceProperties(&properties, device) == cudaSuccess &&
      properties.major == 8 && properties.minor == 7 &&
      params.seqlen_q == params.seqlen_k;
}

void run_mha_fwd_hdim64_bf16_apx(
    FLASH_NAMESPACE::Flash_fwd_params& params, cudaStream_t stream) {
  if (params.seqlen_q <= 5120) {
    FLASH_NAMESPACE::run_flash_fwd<
        Flash_fwd_kernel_traits<
            64, 128, 64, 4, false, false, cutlass::bfloat16_t>,
        false, false>(params, stream);
  } else {
    FLASH_NAMESPACE::run_flash_fwd<
        Flash_fwd_kernel_traits<
            64, 128, 128, 4, false, false, cutlass::bfloat16_t>,
        false, false>(params, stream);
  }
}

}  // namespace apxinf::cuda::cutlass_ops
