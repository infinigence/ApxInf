// ApxInf-owned direct-E4M3 instantiation of the vendored FA2 forward template.

#include "flash_attn/namespace_config.h"
#include "flash_attn/flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template <>
void run_mha_fwd_<cutlass::half_t, 256, false>(
    Flash_fwd_params& params, cudaStream_t stream) {
  run_mha_fwd_hdim256<cutlass::half_t, false>(params, stream);
}

}  // namespace FLASH_NAMESPACE
