// ApxInf-owned FP16 FA2 instantiation for the SigLIP head dimension.

#include "flash_attn/namespace_config.h"
#include "flash_attn/flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template <>
void run_mha_fwd_<cutlass::half_t, 96, false>(
    Flash_fwd_params& params, cudaStream_t stream) {
  run_mha_fwd_hdim96<cutlass::half_t, false>(params, stream);
}

}  // namespace FLASH_NAMESPACE
