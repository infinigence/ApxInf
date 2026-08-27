// Copyright 2026 apxinf contributors.
// Local causal BF16 forward instantiations for the vendored FA2 subset.

#include "fa2/flash_attn/namespace_config.h"
#include "fa2/flash_attn/flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template <>
void run_mha_fwd_<cutlass::bfloat16_t, 96, true>(
    Flash_fwd_params& params, cudaStream_t stream) {
  run_mha_fwd_hdim96<cutlass::bfloat16_t, true>(params, stream);
}

template <>
void run_mha_fwd_<cutlass::bfloat16_t, 256, true>(
    Flash_fwd_params& params, cudaStream_t stream) {
  run_mha_fwd_hdim256<cutlass::bfloat16_t, true>(params, stream);
}

}  // namespace FLASH_NAMESPACE
