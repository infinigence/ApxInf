// Copyright 2026 ApxInf contributors.
#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>

namespace apxinf::cuda::model_ops {

// Gather rows from a BF16 embedding table: out[t, :] = table[ids[t], :].
//
// Decode touches one row of a 248320 x 5120 table, so the whole table stays
// resident and only the gathered rows move.
int embedding_gather(const void* table, const int32_t* ids, void* output,
                     int tokens, int hidden, int vocab, cudaStream_t stream);

// Index of the largest logit in a single BF16 row.
//
// Reducing on device avoids copying 248320 logits to the host per token,
// which at BF16 is ~486 KiB of PCIe/unified traffic for one integer.
int argmax_bf16(const void* logits, int32_t* index, int count,
                cudaStream_t stream);

}  // namespace apxinf::cuda::model_ops
