/*
 * SPDX-License-Identifier: Apache-2.0
 * Standalone ApxInf adapter derived from vLLM v0.27.1 Marlin.
 * Upstream Marlin copyright/license notices remain in kernels/marlin sources.
 *
 * Raw ABI layout `marlin_awq_u4_g32_v1`:
 * - AWQ input repack: int32 [padded_k, padded_n / 8].
 * - packed output: int32 [padded_k / 16, padded_n * 2].
 * - scales: BF16 [padded_k / 32, padded_n], upstream Marlin permutation.
 * - zero_points: packed U4 [padded_k / 32, padded_n / 8], upstream permutation.
 * - activation/output: BF16 [m,padded_k] and [m,padded_n].
 * - workspace: `sms` int32 values, reset exactly once before every launch.
 */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <type_traits>
#include <cstdlib>
#include <cstring>

#include "../kernels/marlin/kernel.h"
#include "../kernels/marlin/marlin_template.h"

namespace marlin {

template <int threads>
__global__ void awq_u4_repack_kernel(const uint32_t* __restrict__ input,
                                     uint32_t* __restrict__ output,
                                     int size_k, int size_n) {
  constexpr int pack_factor = 8;
  constexpr int target_tile_n = tile_n_size;
  constexpr int target_tile_k = tile_k_size;
  const int k_tiles = size_k / target_tile_k;
  const int n_tiles = size_n / target_tile_n;
  const int64_t total = static_cast<int64_t>(k_tiles) * n_tiles * 128;
  for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < total; linear += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int tile_elem = linear % 128;
    const int tile = linear / 128;
    const int n_tile = tile % n_tiles;
    const int k_tile = tile / n_tiles;
    const int th = tile_elem / 4;
    const int warp = tile_elem % 4;
    const int tc_col = th / 4;
    const int tc_row = (th % 4) * 2;
    constexpr int tc_offsets[4] = {0, 1, 8, 9};
    const int cur_n = warp * 16 + tc_col;
    const int cur_n_packed = cur_n / pack_factor;
    const int cur_n_pos = cur_n % pack_factor;
    constexpr int undo_pack[8] = {0, 4, 1, 5, 2, 6, 3, 7};
    uint32_t vals[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int k = k_tile * 16 + tc_row + tc_offsets[i];
      const int base_n = n_tile * 8;
      const uint32_t q0 = input[k * (size_n / 8) + base_n + cur_n_packed];
      const uint32_t q1 = input[k * (size_n / 8) + base_n + cur_n_packed + 1];
      vals[i] = (q0 >> (undo_pack[cur_n_pos] * 4)) & 15U;
      vals[4 + i] = (q1 >> (undo_pack[cur_n_pos] * 4)) & 15U;
    }
    constexpr int pack_idx[8] = {0, 2, 4, 6, 1, 3, 5, 7};
    uint32_t result = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) result |= vals[pack_idx[i]] << (i * 4);
    output[tile * 128 + th * 4 + warp] = result;
  }
}

// Reverse the persistent Marlin permutations into a canonical
// compressed-tensors row tile.  The output is deliberately still U4: the
// established raw row dequantizer remains the single owner of BF16 rounding.
template <int threads>
__global__ void awq_u4_inverse_raw_rows_kernel(
    const uint32_t* __restrict__ packed,
    const __nv_bfloat16* __restrict__ scales,
    const uint32_t* __restrict__ zero_points,
    uint32_t* __restrict__ raw_qweight,
    __nv_bfloat16* __restrict__ raw_scales,
    uint32_t* __restrict__ raw_zero_points, int logical_k, int padded_n,
    int source_n_offset, int row_start, int row_count) {
  const int local_row = blockIdx.x;
  const int groups = logical_k / 32;
  const int words_per_row = logical_k / 8;
  const int n_tiles = padded_n / tile_n_size;

  if (local_row < row_count) {
    const int source_n = source_n_offset + row_start + local_row;
    const int n_tile = source_n / tile_n_size;
    const int n_local = source_n % tile_n_size;
    const int warp = n_local / 16;
    const int packed_thread_base = (n_local % 8) * 4;
    const int n_half = n_local % 16 >= 8 ? 4 : 0;
    for (int word = threadIdx.x; word < words_per_row; word += threads) {
      const int k_base = word * 8;
      uint32_t result = 0;
#pragma unroll
      for (int element = 0; element < 8; ++element) {
        const int k = k_base + element;
        const int k_tile = k / tile_k_size;
        const int k_local = k % tile_k_size;
        const int packed_thread = packed_thread_base + (k_local % 8) / 2;
        const int source = n_half + (k_local >= 8 ? 2 : 0) + (k_local & 1);
        const int q_nibble = (source & 1) * 4 + source / 2;
        const int64_t packed_index =
            (static_cast<int64_t>(k_tile) * n_tiles + n_tile) * 128 +
            packed_thread * 4 + warp;
        const uint32_t q = (packed[packed_index] >> (q_nibble * 4)) & 15U;
        result |= q << (element * 4);
      }
      raw_qweight[static_cast<int64_t>(local_row) * words_per_row + word] =
          result;
    }
    for (int group = threadIdx.x; group < groups; group += threads) {
      // get_scale_perms is an 8x8 transpose and therefore self-inverse.
      const int scale_pos = (n_local % 8) * 8 + n_local / 8;
      const int64_t scale_index =
          static_cast<int64_t>(group) * padded_n + n_tile * 64 + scale_pos;
      raw_scales[static_cast<int64_t>(local_row) * groups + group] =
          scales[scale_index];
    }
  }

  // Canonical zero points pack eight consecutive rows into each word.  A
  // block may therefore also own one local eight-row pack independently of
  // its qweight/scale work above.
  const int zp_row_pack = blockIdx.x;
  const int zp_row_packs = (row_count + 7) / 8;
  if (zp_row_pack < zp_row_packs) {
    for (int group = threadIdx.x; group < groups; group += threads) {
      uint32_t result = 0;
#pragma unroll
      for (int lane = 0; lane < 8; ++lane) {
        const int tile_row = zp_row_pack * 8 + lane;
        if (tile_row < row_count) {
          const int source_n = source_n_offset + row_start + tile_row;
          const int n_tile = source_n / tile_n_size;
          const int n_local = source_n % tile_n_size;
          const int scale_pos = (n_local % 8) * 8 + n_local / 8;
          const int zp_word_in_block = scale_pos / 8;
          const int interleaved = scale_pos % 8;
          const int zp_nibble = (interleaved & 1) * 4 + interleaved / 2;
          const int64_t zp_index =
              static_cast<int64_t>(group) * (padded_n / 8) + n_tile * 8 +
              zp_word_in_block;
          result |= ((zero_points[zp_index] >> (zp_nibble * 4)) & 15U)
                    << (lane * 4);
        }
      }
      raw_zero_points[static_cast<int64_t>(zp_row_pack) * groups + group] =
          result;
    }
  }
}

// Direct exact BF16 inverse used by the production prefill path. One lane
// writes sixteen contiguous K values and reuses one group scale/zero point.
template <int threads>
__global__ void awq_u4_dequant_bf16_vec16_kernel(
    const uint32_t* __restrict__ packed,
    const __nv_bfloat16* __restrict__ scales,
    const uint32_t* __restrict__ zero_points,
    __nv_bfloat16* __restrict__ dense, int logical_n, int logical_k,
    int padded_n, int source_n_offset) {
  const int k_vectors = logical_k / 16;
  const int64_t total = static_cast<int64_t>(logical_n) * k_vectors;
  const int n_tiles = padded_n / tile_n_size;
  for (int64_t linear = static_cast<int64_t>(blockIdx.x) * threads + threadIdx.x;
       linear < total;
       linear += static_cast<int64_t>(threads) * gridDim.x) {
    const int n = static_cast<int>(linear / k_vectors);
    const int k = static_cast<int>(linear - static_cast<int64_t>(n) * k_vectors) * 16;
    const int source_n = source_n_offset + n;
    const int n_tile = source_n / tile_n_size;
    const int n_local = source_n % tile_n_size;
    const int k_tile = k / tile_k_size;
    const int warp = n_local / 16;
    const int packed_thread = (n_local % 8) * 4;
    const int64_t packed_base =
        (static_cast<int64_t>(k_tile) * n_tiles + n_tile) * 128 +
        packed_thread * 4 + warp;
    const int low_nibble = n_local % 16 >= 8 ? 2 : 0;
    const int group = k / 32;
    const int scale_pos = (n_local % 8) * 8 + n_local / 8;
    const int64_t scale_index =
        static_cast<int64_t>(group) * padded_n + n_tile * 64 + scale_pos;
    const float scale = __bfloat162float(scales[scale_index]);
    const int zp_word_in_block = scale_pos / 8;
    const int interleaved = scale_pos % 8;
    const int zp_nibble = (interleaved & 1) * 4 + interleaved / 2;
    const int64_t zp_index =
        static_cast<int64_t>(group) * (padded_n / 8) + n_tile * 8 +
        zp_word_in_block;
    const int zero_point = static_cast<int>(
        (zero_points[zp_index] >> (zp_nibble * 4)) & 15U);
    union alignas(16) DenseVector {
      __nv_bfloat16 values[16];
      uint4 vectors[2];
    } result;
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
      const uint32_t qword = packed[packed_base + pair * 4];
      const int q0 = static_cast<int>((qword >> (low_nibble * 4)) & 15U);
      const int q1 = static_cast<int>((qword >> ((low_nibble + 4) * 4)) & 15U);
      const int q8 = static_cast<int>((qword >> ((low_nibble + 1) * 4)) & 15U);
      const int q9 = static_cast<int>((qword >> ((low_nibble + 5) * 4)) & 15U);
      result.values[pair * 2] =
          __float2bfloat16(static_cast<float>(q0 - zero_point) * scale);
      result.values[pair * 2 + 1] =
          __float2bfloat16(static_cast<float>(q1 - zero_point) * scale);
      result.values[8 + pair * 2] =
          __float2bfloat16(static_cast<float>(q8 - zero_point) * scale);
      result.values[8 + pair * 2 + 1] =
          __float2bfloat16(static_cast<float>(q9 - zero_point) * scale);
    }
    uint4* output = reinterpret_cast<uint4*>(
        dense + static_cast<int64_t>(n) * logical_k + k);
    output[0] = result.vectors[0];
    output[1] = result.vectors[1];
  }
}
using Kernel = void (*)(MARLIN_KERNEL_PARAMS);

static Kernel select_kernel(int mb, bool m8, int threads, int nk, int nn) {
#define PICK(MB, M8, T, NK, NN)                                                \
  if (mb == MB && m8 == M8 && threads == T && nk == NK && nn == NN)           \
    return Marlin<vllm::kBFloat16.id(), vllm::kU4.id(),                       \
                  vllm::kBFloat16.id(), vllm::kBFloat16.id(), T, MB, NN, NK,  \
                  M8, 4, 2, false>
  PICK(1, true, 256, 8, 8); PICK(1, true, 128, 4, 8); PICK(1, true, 128, 8, 4);
  PICK(1, false, 256, 8, 8); PICK(1, false, 128, 4, 8); PICK(1, false, 128, 8, 4);
  PICK(2, false, 256, 4, 16); PICK(2, false, 128, 4, 8); PICK(2, false, 128, 8, 4);
  PICK(3, false, 256, 4, 16); PICK(3, false, 128, 4, 8); PICK(3, false, 128, 8, 4);
  PICK(4, false, 256, 4, 16); PICK(4, false, 128, 4, 8); PICK(4, false, 128, 8, 4);
#undef PICK
  return nullptr;
}

static int shared_bytes(int mb, int tk, int tn) {
  constexpr int stages = 4;
  const int sh_a = stages * mb * 16 * tk * 2;
  const int sh_b = stages * tk * tn / 8 * 4;
  const int sh_red = mb * 16 * (tn + 8) * 2;
  const int largest = sh_b > sh_red ? sh_b : sh_red;
  const int combined = (sh_b < sh_red ? sh_b : sh_red) + tn * 2;
  const int tmp = largest > combined ? largest : combined;
  const int sh_scales = (tk / 32) * tn * 2 * stages;
  const int sh_zp = sh_scales / 4;
  return tmp + sh_a + sh_scales + sh_zp;
}
struct DeviceLaunchState {
  int device = -1;
  int sms = 0;
  int capability = 0;
  int max_smem = 0;
  Kernel prepared[12]{};
  int prepared_smem[12]{};
  int prepared_count = 0;
  int m1_shape = 0;  // 0=K128xN128 default, 1=K64xN128 opt-in, 2=K128xN64
};

static thread_local DeviceLaunchState launch_state;

static cudaError_t get_launch_state(DeviceLaunchState** result) {
  int device = 0;
  cudaError_t err = cudaGetDevice(&device);
  if (err != cudaSuccess) return err;
  if (launch_state.device != device) {
    int major = 0, minor = 0;
    if ((err = cudaDeviceGetAttribute(&launch_state.sms,
                                      cudaDevAttrMultiProcessorCount,
                                      device)) != cudaSuccess ||
        (err = cudaDeviceGetAttribute(&major,
                                      cudaDevAttrComputeCapabilityMajor,
                                      device)) != cudaSuccess ||
        (err = cudaDeviceGetAttribute(&minor,
                                      cudaDevAttrComputeCapabilityMinor,
                                      device)) != cudaSuccess ||
        (err = cudaDeviceGetAttribute(
             &launch_state.max_smem,
             cudaDevAttrMaxSharedMemoryPerBlockOptin, device)) != cudaSuccess)
      return err;
    launch_state.device = device;
    launch_state.capability = major * 10 + minor;
    launch_state.prepared_count = 0;
    const char* m1_shape = std::getenv("APXINF_MARLIN_M1_SHAPE");
    launch_state.m1_shape =
        m1_shape && std::strcmp(m1_shape, "k64n128") == 0
            ? 1
            : (m1_shape && std::strcmp(m1_shape, "k128n64") == 0 ? 2 : 0);
  }
  *result = &launch_state;
  return cudaSuccess;
}

static cudaError_t prepare_kernel(DeviceLaunchState* state, Kernel kernel,
                                  int smem) {
  for (int i = 0; i < state->prepared_count; ++i)
    if (state->prepared[i] == kernel && state->prepared_smem[i] >= smem)
      return cudaSuccess;
  cudaError_t err = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  if (err == cudaSuccess && state->prepared_count < 12) {
    const int i = state->prepared_count++;
    state->prepared[i] = kernel;
    state->prepared_smem[i] = smem;
  }
  return err;
}

// Persistent batch-one dispatch. Weights and workspace keep stable VRAM
// addresses; the final CTA of every sliced reduction restores its lock to
// zero, so only the pre-decode transition must initialize the lock slab.
static cudaError_t launch_gemm_m1(
    const void* activation, const void* packed, const void* scales,
    const void* zero_points, void* output, void* workspace, int padded_n,
    int padded_k, int sms, DeviceLaunchState* state, cudaStream_t stream) {
  constexpr int mb = 1;
  constexpr bool m8 = true;
  int tk = 128, tn = 128, threads = 256;
  if (state->m1_shape == 1) {
    // K64xN128 opt-in: halves dynamic shared memory but sits on a frozen-
    // reference near-tie at 1K under current clocks; kept off by default.
    tk = 64; tn = 128; threads = 128;
  } else if (state->m1_shape == 2) {
    tk = 128; tn = 64; threads = 128;
  } else {
    tk = 128; tn = 128; threads = 256;
  }
  const int smem = shared_bytes(mb, tk, tn);
  auto kernel = select_kernel(mb, m8, threads, tk / 16, tn / 16);
  if (!kernel || padded_n % tn || padded_k % tk ||
      smem > state->max_smem - 512)
    return cudaErrorNotSupported;
  cudaError_t err = prepare_kernel(state, kernel, smem);
  if (err != cudaSuccess) return err;
  kernel<<<sms, threads, smem, stream>>>(
      static_cast<const int4*>(activation),
      static_cast<const int4*>(packed), static_cast<int4*>(output), nullptr,
      nullptr, nullptr, static_cast<const int4*>(scales), nullptr,
      static_cast<const int4*>(zero_points), nullptr, padded_k / 32, 1,
      padded_n, padded_k, padded_k, static_cast<int*>(workspace), false, false,
      false, smem);
  return cudaGetLastError();
}

static cudaError_t launch_gemm(
    const void* activation, const void* packed, const void* scales,
    const void* zero_points, void* output, void* workspace, int m,
    int padded_n, int padded_k, int sms, DeviceLaunchState* state,
    cudaStream_t stream) {
  if (m == 1) {
    return launch_gemm_m1(activation, packed, scales, zero_points, output,
                          workspace, padded_n, padded_k, sms, state, stream);
  }
  cudaError_t err = cudaSuccess;
  auto* a = static_cast<const int4*>(activation);
  auto* c = static_cast<int4*>(output);
  int rest = m;
  int max_thread_m_blocks = 4;
  while (rest) {
    const int par_count = rest / (max_thread_m_blocks * 16);
    const int split = par_count > 0
                          ? par_count * (max_thread_m_blocks * 16)
                          : rest;
    const int mb_raw = (split + 15) / 16;
    const int mb = mb_raw < max_thread_m_blocks ? mb_raw : max_thread_m_blocks;
    const bool m8 = split <= 8;
    int tk = mb > 1 ? 64 : 128;
    int tn = mb > 1 ? 256 : 128;
    int threads = 256;
    if (padded_n % tn || padded_k % tk ||
        shared_bytes(mb, tk, tn) > state->max_smem - 512) {
      tk = 64;
      tn = 128;
      threads = 128;
    }
    if (padded_n % tn || padded_k % tk ||
        shared_bytes(mb, tk, tn) > state->max_smem - 512) {
      tk = 128;
      tn = 64;
      threads = 128;
    }
    const int grid_work = (padded_n / tn) *
                          ((split + mb * 16 - 1) / (mb * 16)) * 4;
    if (grid_work <= sms && padded_k % 128 == 0 &&
        shared_bytes(mb, 128, 64) <= state->max_smem) {
      tk = 128;
      tn = 64;
      threads = 128;
    }
    const int smem = shared_bytes(mb, tk, tn);
    auto kernel = select_kernel(mb, m8, threads, tk / 16, tn / 16);
    if (!kernel || padded_n % tn || padded_k % tk ||
        smem > state->max_smem - 512) {
      if (max_thread_m_blocks > 1) {
        --max_thread_m_blocks;
        continue;
      }
      return cudaErrorNotSupported;
    }
    if ((err = prepare_kernel(state, kernel, smem)) != cudaSuccess) return err;
    if ((err = cudaMemsetAsync(workspace, 0,
                               static_cast<size_t>(sms) * sizeof(int),
                               stream)) != cudaSuccess)
      return err;
    kernel<<<sms, threads, smem, stream>>>(
        a, static_cast<const int4*>(packed), c, nullptr, nullptr, nullptr,
        static_cast<const int4*>(scales), nullptr,
        static_cast<const int4*>(zero_points), nullptr, padded_k / 32, split,
        padded_n, padded_k, padded_k, static_cast<int*>(workspace), false, false,
        false, smem);
    if ((err = cudaGetLastError()) != cudaSuccess) return err;
    a += static_cast<int64_t>(split) * padded_k / 8;
    c += static_cast<int64_t>(split) * padded_n / 8;
    rest -= split;
  }
  return cudaSuccess;
}

}  // namespace marlin

extern "C" cudaError_t apxinf_marlin_awq_u4_g32_v1_repack(
    const void* awq_qweight, void* marlin_qweight, int padded_k, int padded_n,
    cudaStream_t stream) {
  if (!awq_qweight || !marlin_qweight || padded_k <= 0 || padded_n <= 0 ||
      padded_k % 16 || padded_n % 64) return cudaErrorInvalidValue;
  int device = 0, sms = 0;
  cudaError_t err = cudaGetDevice(&device);
  if (err != cudaSuccess) return err;
  err = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  if (err != cudaSuccess) return err;
  marlin::awq_u4_repack_kernel<256><<<sms, 256, 0, stream>>>(
      static_cast<const uint32_t*>(awq_qweight),
      static_cast<uint32_t*>(marlin_qweight), padded_k, padded_n);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_marlin_awq_u4_g32_v1_inverse_raw_rows(
    const void* packed, const void* scales, const void* zero_points,
    void* raw_qweight, void* raw_scales, void* raw_zero_points,
    int logical_k, int padded_n, int padded_k, int source_n_offset,
    int row_start, int row_count, cudaStream_t stream) {
  if (!packed || !scales || !zero_points || !raw_qweight || !raw_scales ||
      !raw_zero_points || logical_k <= 0 || logical_k % 32 ||
      logical_k > padded_k || padded_k % 128 || padded_n <= 0 ||
      padded_n % 64 || source_n_offset < 0 || source_n_offset > padded_n ||
      row_start < 0 || row_start > padded_n - source_n_offset ||
      row_count <= 0 || row_count > padded_n - source_n_offset - row_start ||
      (reinterpret_cast<uintptr_t>(packed) & 15U) ||
      (reinterpret_cast<uintptr_t>(raw_qweight) & 15U) ||
      (reinterpret_cast<uintptr_t>(raw_scales) & 1U) ||
      (reinterpret_cast<uintptr_t>(raw_zero_points) & 3U))
    return cudaErrorInvalidValue;
  constexpr int threads = 256;
  marlin::awq_u4_inverse_raw_rows_kernel<threads>
      <<<row_count, threads, 0, stream>>>(
          static_cast<const uint32_t*>(packed),
          static_cast<const __nv_bfloat16*>(scales),
          static_cast<const uint32_t*>(zero_points),
          static_cast<uint32_t*>(raw_qweight),
          static_cast<__nv_bfloat16*>(raw_scales),
          static_cast<uint32_t*>(raw_zero_points), logical_k, padded_n,
          source_n_offset, row_start, row_count);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_marlin_awq_u4_g32_v1_dequant_bf16(
    const void* packed, const void* scales, const void* zero_points,
    void* dense, int logical_n, int logical_k, int padded_n, int padded_k,
    int source_n_offset, cudaStream_t stream) {
  if (!packed || !scales || !zero_points || !dense || logical_n <= 0 ||
      logical_k <= 0 || logical_k % 16 || logical_k > padded_k ||
      padded_k % 128 || padded_n <= 0 || padded_n % 64 ||
      source_n_offset < 0 || source_n_offset > padded_n - logical_n ||
      (reinterpret_cast<uintptr_t>(packed) & 15U) ||
      (reinterpret_cast<uintptr_t>(dense) & 15U))
    return cudaErrorInvalidValue;
  constexpr int threads = 256;
  const int64_t total = static_cast<int64_t>(logical_n) * (logical_k / 16);
  const int blocks = static_cast<int>(
      (total + threads - 1) / threads < 4096
          ? (total + threads - 1) / threads
          : 4096);
  marlin::awq_u4_dequant_bf16_vec16_kernel<threads><<<blocks, threads, 0, stream>>>(
      static_cast<const uint32_t*>(packed),
      static_cast<const __nv_bfloat16*>(scales),
      static_cast<const uint32_t*>(zero_points),
      static_cast<__nv_bfloat16*>(dense), logical_n, logical_k, padded_n,
      source_n_offset);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_marlin_awq_u4_g32_v1_gemm_bf16(
    const void* activation, const void* packed, const void* scales,
    const void* zero_points, void* output, void* workspace, int m,
    int logical_n, int logical_k, int padded_n, int padded_k, int sms,
    cudaStream_t stream) {
  if (!activation || !packed || !scales || !zero_points || !output ||
      !workspace || m < 1 || m > 512 || logical_n < 1 || logical_k < 1 ||
      logical_n > padded_n || logical_k > padded_k || padded_n % 64 ||
      padded_k % 128 || sms < 1 ||
      (reinterpret_cast<uintptr_t>(activation) & 15U) ||
      (reinterpret_cast<uintptr_t>(packed) & 15U) ||
      (reinterpret_cast<uintptr_t>(output) & 15U)) return cudaErrorInvalidValue;
  marlin::DeviceLaunchState* state = nullptr;
  cudaError_t err = marlin::get_launch_state(&state);
  if (err != cudaSuccess) return err;
  if (sms > state->sms || state->capability < 80)
    return cudaErrorNotSupported;
  return marlin::launch_gemm(
      activation, packed, scales, zero_points, output, workspace, m, padded_n,
      padded_k, sms, state, stream);
}

struct apxinf_marlin_awq_u4_g32_v1_projection {
  const void* packed;
  const void* scales;
  const void* zero_points;
  void* output;
  void* padded_output;
  int logical_n;
  int padded_n;
};

extern "C" cudaError_t apxinf_marlin_awq_u4_g32_v1_gemm_batch_bf16(
    const void* activation,
    const apxinf_marlin_awq_u4_g32_v1_projection* projections,
    int projection_count, void* workspace, int m, int logical_k,
    int padded_k, int sms, cudaStream_t stream) {
  if (!activation || !projections || !workspace || projection_count < 2 ||
      projection_count > 4 || m < 1 || m > 512 || logical_k < 1 ||
      logical_k > padded_k || padded_k % 128 || sms < 1 ||
      (reinterpret_cast<uintptr_t>(activation) & 15U))
    return cudaErrorInvalidValue;
  marlin::DeviceLaunchState* state = nullptr;
  cudaError_t err = marlin::get_launch_state(&state);
  if (err != cudaSuccess) return err;
  if (sms > state->sms || state->capability < 80)
    return cudaErrorNotSupported;
  for (int i = 0; i < projection_count; ++i) {
    const auto& projection = projections[i];
    void* launch_output = projection.logical_n == projection.padded_n
                              ? projection.output
                              : projection.padded_output;
    if (!projection.packed || !projection.scales || !projection.zero_points ||
        !projection.output || !launch_output || projection.logical_n < 1 ||
        projection.logical_n > projection.padded_n ||
        projection.padded_n % 64 ||
        (reinterpret_cast<uintptr_t>(projection.packed) & 15U) ||
        (reinterpret_cast<uintptr_t>(launch_output) & 15U))
      return cudaErrorInvalidValue;
    err = marlin::launch_gemm(
        activation, projection.packed, projection.scales,
        projection.zero_points, launch_output, workspace, m,
        projection.padded_n, padded_k, sms, state, stream);
    if (err != cudaSuccess) return err;
    if (projection.logical_n != projection.padded_n) {
      err = cudaMemcpy2DAsync(
          projection.output,
          static_cast<size_t>(projection.logical_n) * sizeof(__nv_bfloat16),
          launch_output,
          static_cast<size_t>(projection.padded_n) * sizeof(__nv_bfloat16),
          static_cast<size_t>(projection.logical_n) * sizeof(__nv_bfloat16), m,
          cudaMemcpyDeviceToDevice, stream);
      if (err != cudaSuccess) return err;
    }
  }
  return cudaSuccess;
}

#include "../kernels/marlin/marlin_awq_u4_g32_kernels.cuh"
