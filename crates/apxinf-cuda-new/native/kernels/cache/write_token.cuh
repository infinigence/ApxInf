// Copyright 2026 ApxInf contributors.
// Device-position KV token write kernel, ported verbatim from the legacy
// family: copy one input token row into the cache row named by `meta[1]`. A
// row at or past the capacity writes nothing, mirroring the clamp the decode
// attention applies to the same device-side metadata.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kTokenWriteThreads = 256;

__global__ void cache_write_token_kernel(const uint16_t* input,
                                         uint16_t* cache,
                                         const uint32_t* meta,
                                         int64_t capacity,
                                         int64_t elements) {
  const uint32_t row = meta[1];
  if (row >= capacity) return;
  for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
       i < elements; i += int64_t(blockDim.x) * gridDim.x)
    cache[static_cast<int64_t>(row) * elements + i] = input[i];
}

}  // namespace
