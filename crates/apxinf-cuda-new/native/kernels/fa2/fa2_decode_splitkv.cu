#include "flash_attn/flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

void run_decode_splitkv_bf16_hdim256(Flash_fwd_params& params, cudaStream_t stream) {
  run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 256, false>(params, stream);
}

}
