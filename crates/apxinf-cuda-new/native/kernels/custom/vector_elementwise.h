// Copyright 2026 ApxInf contributors.
#pragma once

#include <cstdlib>
#include <cstring>

#include <cstdint>

namespace apxinf::cuda_new::vector_elementwise {

// Opt-in predicate for the 16-byte-vector elementwise fast paths.
//
// Shared by `pointwise_ops::add_bf16` and
// `quantization_ops::quantize_fp8_per_tensor`: both have a uint4 variant behind
// the same env flag, and both need the same legality test, so the predicate
// lives here rather than being written twice. It is the only thing those two
// families have in common -- the kernels themselves share nothing.
//
// The pointer checks are not redundant with the flag: the uint4 reinterpret
// casts in those kernels require 16-byte alignment, and a count that is not a
// multiple of 8 has no tail handling in the vector variants.
inline bool enabled(const void* source, const void* destination,
                    long long count) {
  const char* flag = std::getenv("APXINF_QWEN38_VECTOR_ELEMENTWISE");
  return flag != nullptr && std::strcmp(flag, "1") == 0 && count % 8 == 0 &&
         reinterpret_cast<uintptr_t>(source) % 16 == 0 &&
         reinterpret_cast<uintptr_t>(destination) % 16 == 0;
}

}  // namespace apxinf::cuda_new::vector_elementwise
