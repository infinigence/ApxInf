// ApxInf-owned FA2 instantiations not shipped by the upstream BF16 source.

#include "flash_attn/namespace_config.h"
#include "flash_attn/flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template <>
void run_mha_fwd_<cutlass::half_t, 128, false>(
    Flash_fwd_params& params, cudaStream_t stream) {
  run_mha_fwd_hdim128<cutlass::half_t, false>(params, stream);
}

template <>
void run_mha_fwd_<cutlass::bfloat16_t, 128, true>(
    Flash_fwd_params& params, cudaStream_t stream) {
  run_mha_fwd_hdim128<cutlass::bfloat16_t, true>(params, stream);
}

}  // namespace FLASH_NAMESPACE
