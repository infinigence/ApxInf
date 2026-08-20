// Copyright (c) 2024, Tri Dao.
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template<>
void run_mha_fwd_<cutlass::half_t, 256, false>(
    Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim256<cutlass::half_t, false>(params, stream);
}

} // namespace FLASH_NAMESPACE
