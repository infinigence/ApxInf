#include <cuda_runtime.h>

#define MARLIN_NAMESPACE_NAME apxinf_marlin
#include "../kernels/marlin/kernel.h"
#include "../kernels/marlin/marlin_template.h"
#include "../kernels/marlin/repack.cuh"

namespace {

__global__ void transpose_i32_kernel(const uint32_t* input, uint32_t* output,
                                     int rows, int columns) {
  __shared__ uint32_t tile[32][33];
  int input_column = blockIdx.x * 32 + threadIdx.x;
  int input_row = blockIdx.y * 32 + threadIdx.y;
  for (int offset = 0; offset < 32; offset += blockDim.y) {
    if (input_column < columns && input_row + offset < rows)
      tile[threadIdx.y + offset][threadIdx.x] =
          input[static_cast<int64_t>(input_row + offset) * columns +
                input_column];
  }
  __syncthreads();
  int output_column = blockIdx.y * 32 + threadIdx.x;
  int output_row = blockIdx.x * 32 + threadIdx.y;
  for (int offset = 0; offset < 32; offset += blockDim.y) {
    if (output_column < rows && output_row + offset < columns)
      output[static_cast<int64_t>(output_row + offset) * rows +
             output_column] = tile[threadIdx.x][threadIdx.y + offset];
  }
}

__device__ __constant__ int kScalePermutation[64] = {
    0,  8,  16, 24, 32, 40, 48, 56, 1,  9,  17, 25, 33, 41, 49, 57,
    2,  10, 18, 26, 34, 42, 50, 58, 3,  11, 19, 27, 35, 43, 51, 59,
    4,  12, 20, 28, 36, 44, 52, 60, 5,  13, 21, 29, 37, 45, 53, 61,
    6,  14, 22, 30, 38, 46, 54, 62, 7,  15, 23, 31, 39, 47, 55, 63};
__device__ __constant__ int kZeroInterleave[8] = {0, 2, 4, 6, 1, 3, 5, 7};

__global__ void transform_scales_bf16_kernel(
    const __nv_bfloat16* original, __nv_bfloat16* transformed,
    int output_dim, int groups) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t count = static_cast<int64_t>(output_dim) * groups;
  if (index >= count) return;
  int64_t source = (index / 64) * 64 + kScalePermutation[index & 63];
  int source_group = source / output_dim;
  int source_output = source % output_dim;
  transformed[index] = original[static_cast<int64_t>(source_output) * groups +
                                source_group];
}

__global__ void transform_zero_u4_kernel(
    const uint32_t* original, uint32_t* transformed,
    int output_dim, int groups) {
  int64_t packed_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t packed_count = static_cast<int64_t>(groups) * output_dim / 8;
  if (packed_index >= packed_count) return;
  int group = packed_index / (output_dim / 8);
  int packed_output = packed_index % (output_dim / 8);
  uint32_t word = 0;
#pragma unroll
  for (int nibble = 0; nibble < 8; ++nibble) {
    int64_t final_index =
        static_cast<int64_t>(group) * output_dim + packed_output * 8 + nibble;
    int64_t interleaved = (final_index / 8) * 8 + kZeroInterleave[nibble];
    int64_t source =
        (interleaved / 64) * 64 + kScalePermutation[interleaved & 63];
    int source_group = source / output_dim;
    int source_output = source % output_dim;
    uint32_t source_word = original[
        static_cast<int64_t>(source_output / 8) * groups + source_group];
    uint32_t value = (source_word >> ((source_output & 7) * 4)) & 15;
    word |= value << (nibble * 4);
  }
  transformed[packed_index] = word;
}

template <int kThreadMBlocks, int kThreadNBlocks, int kThreadKBlocks,
          bool kMBlockSize8, int kThreads>
cudaError_t launch_marlin_bf16_u4_group32(
    const void* activation, const void* repacked_weight,
    const void* permuted_scales, const void* permuted_zero_points,
    void* output, void* reduce_workspace, void* locks, int rows,
    int output_dim, int input_dim, cudaStream_t stream) {
  int device = 0;
  int multiprocessors = 0;
  int max_shared_memory = 0;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) return status;
  status = cudaDeviceGetAttribute(
      &multiprocessors, cudaDevAttrMultiProcessorCount, device);
  if (status != cudaSuccess) return status;
  status = cudaDeviceGetAttribute(
      &max_shared_memory, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  if (status != cudaSuccess) return status;

  auto kernel = apxinf_marlin::Marlin<
      vllm::kBFloat16.id(), vllm::kU4.id(), vllm::kBFloat16.id(),
      vllm::kBFloat16.id(), kThreads, kThreadMBlocks, kThreadNBlocks,
      kThreadKBlocks, kMBlockSize8, 4, 2, false>;
  status = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_memory);
  if (status != cudaSuccess) return status;
  kernel<<<multiprocessors, kThreads, max_shared_memory, stream>>>(
      static_cast<const int4*>(activation),
      static_cast<const int4*>(repacked_weight),
      static_cast<int4*>(output),
      static_cast<int4*>(reduce_workspace),
      nullptr,
      nullptr,
      static_cast<const int4*>(permuted_scales),
      nullptr,
      static_cast<const int4*>(permuted_zero_points),
      nullptr,
      input_dim / 32,
      rows,
      output_dim,
      input_dim,
      input_dim,
      static_cast<int*>(locks),
      false,
      false,
      true,
      max_shared_memory);
  return cudaGetLastError();
}

}  // namespace

extern "C" cudaError_t apxinf_static_marlin_repack_u4(
    const void* original_output_major, void* transposed_workspace,
    void* repacked_weight, int output_dim, int input_dim,
    cudaStream_t stream) {
  if (original_output_major == nullptr || transposed_workspace == nullptr ||
      repacked_weight == nullptr || output_dim <= 0 || output_dim % 64 != 0 ||
      input_dim <= 0 || input_dim % 16 != 0) {
    return cudaErrorInvalidValue;
  }
  int packed_input = input_dim / 8;
  dim3 transpose_block(32, 8, 1);
  dim3 transpose_grid((packed_input + 31) / 32,
                      (output_dim + 31) / 32, 1);
  transpose_i32_kernel<<<transpose_grid, transpose_block, 0, stream>>>(
      static_cast<const uint32_t*>(original_output_major),
      static_cast<uint32_t*>(transposed_workspace), output_dim, packed_input);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) return status;

  int device = 0;
  int multiprocessors = 0;
  int max_shared_memory = 0;
  status = cudaGetDevice(&device);
  if (status != cudaSuccess) return status;
  status = cudaDeviceGetAttribute(
      &multiprocessors, cudaDevAttrMultiProcessorCount, device);
  if (status != cudaSuccess) return status;
  status = cudaDeviceGetAttribute(
      &max_shared_memory, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  if (status != cudaSuccess) return status;
  auto kernel = apxinf_marlin::gptq_marlin_repack_kernel<256, 4, false, false>;
  status = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_memory);
  if (status != cudaSuccess) return status;
  kernel<<<multiprocessors, 256, max_shared_memory, stream>>>(
      static_cast<const uint32_t*>(transposed_workspace), nullptr,
      static_cast<uint32_t*>(repacked_weight), input_dim, output_dim);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_marlin_transform_scales_zero_u4_group32(
    const void* original_scales, const void* original_zero_points,
    void* transformed_scales, void* transformed_zero_points,
    int output_dim, int input_dim, cudaStream_t stream) {
  if (original_scales == nullptr || original_zero_points == nullptr ||
      transformed_scales == nullptr || transformed_zero_points == nullptr ||
      output_dim <= 0 || output_dim % 64 != 0 || input_dim <= 0 ||
      input_dim % 32 != 0) {
    return cudaErrorInvalidValue;
  }
  int groups = input_dim / 32;
  int64_t scale_count = static_cast<int64_t>(groups) * output_dim;
  int64_t zero_count = scale_count / 8;
  constexpr int threads = 256;
  transform_scales_bf16_kernel<<<(scale_count + threads - 1) / threads,
                                  threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(original_scales),
      static_cast<__nv_bfloat16*>(transformed_scales), output_dim, groups);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  transform_zero_u4_kernel<<<(zero_count + threads - 1) / threads,
                              threads, 0, stream>>>(
      static_cast<const uint32_t*>(original_zero_points),
      static_cast<uint32_t*>(transformed_zero_points), output_dim, groups);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_marlin_bf16_u4_group32(
    const void* activation, const void* repacked_weight,
    const void* permuted_scales, const void* permuted_zero_points,
    void* output, void* reduce_workspace, void* locks, int rows,
    int output_dim, int input_dim, cudaStream_t stream) {
  if (activation == nullptr || repacked_weight == nullptr ||
      permuted_scales == nullptr || permuted_zero_points == nullptr ||
      output == nullptr || reduce_workspace == nullptr || locks == nullptr ||
      rows < 1 || rows > 64 || output_dim <= 0 || input_dim <= 0) {
    return cudaErrorInvalidValue;
  }
  if (rows <= 8) {
    if (output_dim % 128 != 0 || input_dim % 128 != 0)
      return cudaErrorInvalidValue;
    return launch_marlin_bf16_u4_group32<1, 8, 8, true, 256>(
        activation, repacked_weight, permuted_scales, permuted_zero_points,
        output, reduce_workspace, locks, rows, output_dim, input_dim, stream);
  }
  if (rows <= 16) {
    if (output_dim % 128 != 0 || input_dim % 128 != 0)
      return cudaErrorInvalidValue;
    return launch_marlin_bf16_u4_group32<1, 8, 8, false, 256>(
        activation, repacked_weight, permuted_scales, permuted_zero_points,
        output, reduce_workspace, locks, rows, output_dim, input_dim, stream);
  }
  if (output_dim % 256 != 0 || input_dim % 64 != 0)
    return cudaErrorInvalidValue;
  if (rows <= 32)
    return launch_marlin_bf16_u4_group32<2, 16, 4, false, 256>(
        activation, repacked_weight, permuted_scales, permuted_zero_points,
        output, reduce_workspace, locks, rows, output_dim, input_dim, stream);
  if (rows <= 48)
    return launch_marlin_bf16_u4_group32<3, 16, 4, false, 256>(
        activation, repacked_weight, permuted_scales, permuted_zero_points,
        output, reduce_workspace, locks, rows, output_dim, input_dim, stream);
  return launch_marlin_bf16_u4_group32<4, 16, 4, false, 256>(
      activation, repacked_weight, permuted_scales, permuted_zero_points,
      output, reduce_workspace, locks, rows, output_dim, input_dim, stream);
}
