// Copyright 2026 ApxInf contributors.
#pragma once

#include <cuda_runtime_api.h>

namespace apxinf::cuda_new::attention {

int split_query_and_gate(const void* fused, void* query, void* gate,
                         int tokens, int heads, int head_dim,
                         cudaStream_t stream);
int apply_output_gate(void* data, const void* gate, long long count,
                      cudaStream_t stream);

}  // namespace apxinf::cuda_new::attention
