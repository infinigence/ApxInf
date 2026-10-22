// ApxInf-owned FP16 split-KV FA2 instantiations for decode attention.
// Non-causal only: the F16 pipeline gates FA2 to APXINF_ATTENTION_MASK_NONE.

#include "flash_attn/namespace_config.h"
#include "flash_attn/flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template void run_mha_fwd_splitkv_dispatch<cutlass::half_t, 128, false>(
    Flash_fwd_params& params, cudaStream_t stream);
template void run_mha_fwd_splitkv_dispatch<cutlass::half_t, 256, false>(
    Flash_fwd_params& params, cudaStream_t stream);

}  // namespace FLASH_NAMESPACE
