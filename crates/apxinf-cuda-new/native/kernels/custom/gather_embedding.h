// Copyright 2026 ApxInf contributors.
#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>

namespace apxinf::cuda_new::gather_embedding {

// Gather rows from a BF16 embedding table: out[t, :] = table[ids[t], :].
//
// Decode touches one row of a 248320 x 5120 table, so the whole table stays
// resident and only the gathered rows move.
//
// NOT the same semantic as GatherSemantic::EmbeddingLookup, despite the name
// overlap recorded in doc/cuda-new-design.md section 5.2. That kernel always
// multiplies by sqrt(cols) -- Gemma folds the scale into the lookup and its
// comment states the scale is not optional -- while this one is a plain copy.
// The index dtype differs too (int32 here, uint32 there), as does out-of-range
// behaviour: this clamps to row 0, that one writes zeros. Unifying them means
// adding a scale field and picking one index dtype, not deleting one.
int embedding_gather(const void* table, const int32_t* ids, void* output,
                     int tokens, int hidden, int vocab, cudaStream_t stream);

}  // namespace apxinf::cuda_new::gather_embedding
