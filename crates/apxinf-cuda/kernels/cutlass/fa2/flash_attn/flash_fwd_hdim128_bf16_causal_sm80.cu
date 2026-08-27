// Copyright (c) 2024, Tri Dao.
// Minimal BF16 head-dim-128 causal specialization for SM80-family builds.
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template<>
void run_mha_fwd_<cutlass::bfloat16_t, 128, true>(
    Flash_fwd_params &params, cudaStream_t stream) {
    using KernelTraits = Flash_fwd_kernel_traits<
        128, 64, 64, 4, false, false, cutlass::bfloat16_t>;
    run_flash_fwd<KernelTraits, false, true>(params, stream);
}

} // namespace FLASH_NAMESPACE
