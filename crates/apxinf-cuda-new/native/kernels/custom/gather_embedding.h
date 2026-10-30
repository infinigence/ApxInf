// Copyright 2026 ApxInf contributors.
#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>

namespace apxinf::cuda_new::gather {

int embedding_gather(const void* table, const int32_t* ids, void* output,
                     int tokens, int hidden, int vocab, cudaStream_t stream);

}  // namespace apxinf::cuda_new::gather
