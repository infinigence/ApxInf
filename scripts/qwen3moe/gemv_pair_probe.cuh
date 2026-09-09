#pragma once
#define MARLIN_NAMESPACE_NAME apxinf_decode_pair
#include "../../crates/apxinf-cuda/kernels/marlin/csrc/moe/marlin_moe_wna16/kernel.h"
#include "../../crates/apxinf-cuda/kernels/marlin/csrc/quantization/gptq_marlin/dequant.h"
#include "../../crates/apxinf-cuda/kernels/custom/w4a16_pair.cuh"
