#pragma once

#include "gather_types.h"

#ifdef __cplusplus
extern "C" {
#endif

apxinf_status_t apxinf_gather_launch(
    apxinf_runtime_t runtime, const apxinf_gather_spec_t* spec,
    const apxinf_gather_bindings_t* bindings);

/* Plain BF16 embedding lookup: out[t, :] = table[ids[t], :].

   NOT the EMBEDDING_LOOKUP semantic above, despite the name. That one always
   multiplies by sqrt(cols) -- Gemma folds the scale into the lookup, and
   dropping it silently destroys the model output -- while this is a plain copy.
   The index dtype differs (int32 vs uint32) and so does out-of-range handling
   (clamp to row 0 vs write zeros). Kernel in
   kernels/custom/gather_embedding.cu. */
apxinf_status_t apxinf_gather_embedding(const void* table,
                                              const void* ids, void* output,
                                              int64_t tokens, int64_t hidden,
                                              int64_t vocab,
                                              apxinf_cuda_stream_t stream);

#ifdef __cplusplus
}
#endif
