#pragma once

#ifdef APXINF_GEMM_CUTLASS_SM89
__global__ void pack_adjacent_bf16(const __nv_bfloat16* source,
                                   __nv_bfloat16* destination,
                                   int64_t rows,
                                   int64_t columns) {
  const int64_t half = columns / 2;
  for (int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       index < rows * columns;
       index += int64_t(gridDim.x) * blockDim.x) {
    const int64_t row = index / columns;
    const int64_t column = index % columns;
    const int64_t pair = column < half ? column : column - half;
    const int64_t lane = column < half ? 0 : 1;
    destination[row * columns + 2 * pair + lane] = source[index];
  }
}
#endif

#ifdef APXINF_GEMM_CUTLASS_SM87_W8A8
__global__ void transpose_w8a8_weight(const int8_t* source,
                                      int8_t* destination,
                                      int64_t rows,
                                      int64_t columns) {
  for (int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       index < rows * columns;
       index += int64_t(gridDim.x) * blockDim.x) {
    const int64_t row = index / columns;
    const int64_t column = index % columns;
    destination[column * rows + row] = source[index];
  }
}
#endif
