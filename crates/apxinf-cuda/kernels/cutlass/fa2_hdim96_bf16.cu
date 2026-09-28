// ApxInf-owned optional BF16 head-dimension-96 dispatch over the unmodified
// upstream FlashAttention-2 templates. The public/default 128x64 tile remains
// unchanged; GR00T may explicitly select this 64x64 tile for measured SM87 or
// SM110 Q=41, H=32, D=48 production shapes.

#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>

#include "flash_attn/flash.h"
#include "flash_attn/flash_fwd_launch_template.h"
#include "flash_attn/namespace_config.h"

namespace apxinf::cuda::cutlass_ops {

void run_mha_fwd_hdim96_bm64_bf16_apx(
    FLASH_NAMESPACE::Flash_fwd_params& params, cudaStream_t stream) {
  FLASH_NAMESPACE::run_flash_fwd<
      Flash_fwd_kernel_traits<
          96, 64, 64, 4, false, false, cutlass::bfloat16_t>,
      false, false>(params, stream);
}

}  // namespace apxinf::cuda::cutlass_ops
