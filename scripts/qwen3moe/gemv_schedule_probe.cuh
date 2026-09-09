#pragma once
// Experimental scheduling/occupancy probe; production kernel remains unchanged.
#include "../../crates/apxinf-cuda/kernels/custom/w4a16_magic.cuh"
template<bool blocked=false, bool pdl=false, bool persistent=false, int load_unroll=16, int min_blocks=1>
__global__ void __launch_bounds__(256, min_blocks) w4a16_gemv_schedule_kernel(
    const __nv_bfloat16* __restrict__ x, int64_t x_slot_stride,
    const int32_t* __restrict__ qweight, const int32_t* __restrict__ qzeros,
    const half* __restrict__ scales, const int32_t* __restrict__ expert_ids,
    int64_t stride_q, int64_t stride_z, int64_t stride_s,
    const float* __restrict__ slot_scale, float* __restrict__ partial,
    int rows, int packed_cols, int group_size, int rows_per_block,
    int schedule_splits=0, int schedule_slots=0) {
  static_assert(load_unroll >= 4 && load_unroll % 4 == 0, "blocked loads need groups of four words");
  static_assert(!(persistent && pdl), "persistent scheduling needs a separately measured PDL policy");
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if constexpr(pdl) {
    // Dense expert selection is static. Only immutable weight addresses may
    // be touched before the producer has finished the activation/router data.
    // Routed expert IDs deliberately remain unread until the dependency sync.
    if(expert_ids==nullptr && blockIdx.x*32+(threadIdx.x&31)<packed_cols) {
      int lane=threadIdx.x&31,warp=threadIdx.x>>5;
      int begin=blockIdx.y*rows_per_block+warp*(rows_per_block/8);
      int end=min(rows,begin+min(16,rows_per_block/8));
      const int32_t* weights=qweight+int64_t(blockIdx.z)*stride_q;
      for(int k=begin;k<end;k+=(blocked?4:1)) {
        int64_t address;
        if constexpr(blocked) address=((int64_t(blockIdx.x)*(rows/4)+k/4)*32+lane)*4+(k&3);
        else address=int64_t(k)*packed_cols+blockIdx.x*32+lane;
        asm volatile("prefetch.global.L2 [%0];" :: "l"(weights+address));
      }
    }
    cudaGridDependencySynchronize();
    cudaTriggerProgrammaticLaunchCompletion();
  }
#elif defined(__CUDA_ARCH__)
  static_assert(!pdl, "programmatic dependent launch requires SM90 or newer");
#endif
  extern __shared__ float smem[];
  float* xs = smem;                       // [rows_per_block]
  float* red = smem + rows_per_block;     // [8 warps][256]

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  // Static scheduling only changes which CTA owns a task. Each task retains
  // its original K split, warp reduction, output address, and accumulation.
  const int n_tiles = (packed_cols * 8 + 255) / 256;
  const int task_count = persistent ? n_tiles * schedule_splits * schedule_slots : 1;
  const int first_task = persistent ? int(blockIdx.x) : 0;
  const int task_stride = persistent ? int(gridDim.x) : 1;
  for (int task = first_task; task < task_count; task += task_stride) {
  const int tile = persistent ? task % n_tiles : int(blockIdx.x);
  const int splits = persistent ? schedule_splits : int(gridDim.y);
  const int split = persistent ? (task / n_tiles) % splits : int(blockIdx.y);
  const int slot = persistent ? task / (n_tiles * splits) : int(blockIdx.z);
  const int cols = packed_cols * 8;
  const int expert = expert_ids == nullptr ? slot : expert_ids[slot];
  const int32_t* task_qweight = qweight + static_cast<int64_t>(expert) * stride_q;
  const int32_t* task_qzeros = qzeros + static_cast<int64_t>(expert) * stride_z;
  const half* task_scales = scales + static_cast<int64_t>(expert) * stride_s;
  const __nv_bfloat16* task_x = x + static_cast<int64_t>(slot) * x_slot_stride;

  const int k_begin = split * rows_per_block;
  const int k_end = min(rows, k_begin + rows_per_block);
  for (int k = k_begin + threadIdx.x; k < k_end; k += blockDim.x) {
    xs[k - k_begin] = __bfloat162float(task_x[k]);
  }
  __syncthreads();

  const int j = tile * 32 + lane;
  const bool active = j < packed_cols;
  const int rows_per_warp = rows_per_block / 8;
  const int kw_begin = k_begin + warp * rows_per_warp;
  const int kw_end = min(k_end, kw_begin + rows_per_warp);

  float total[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    total[i] = 0.0f;
  }
  const int32_t* wp = blocked ? task_qweight + (int64_t(tile)*(rows/4)*32+lane)*4
                              : (active ? task_qweight + j : task_qweight);
  auto load_word = [&](int k) {
    if constexpr(blocked) return wp[int64_t(k/4)*128+(k&3)];
    else return wp[int64_t(k)*packed_cols];
  };

  // Walk one quantization group at a time. Within a group the scale and zero
  // point are constant, so the affine correction s*(acc - z*sum(x)) leaves the
  // inner loop entirely and the body reduces to 16
  // *independent* loads followed by their FMAs.
  //
  // The unroll is the point, not the arithmetic — see 16 above.
  // The old loop tested the group boundary inside the k loop, which both
  // serialized the loads and inhibited unrolling.
  for (int k0 = kw_begin; k0 < kw_end;) {
    const int g = k0 / group_size;
    const int k1 = min(kw_end, (g + 1) * group_size);

    float zero[8];
    float scale[8];
    if (active) {
      const int32_t z = task_qzeros[static_cast<int64_t>(g) * packed_cols + j];
      const uint4 packed_scales = *reinterpret_cast<const uint4*>(
          task_scales + static_cast<int64_t>(g) * cols + j * 8);
      const half* s = reinterpret_cast<const half*>(&packed_scales);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        zero[i] = static_cast<float>((z >> (4 * i)) & 0xF);
        scale[i] = __half2float(s[awq_nibble_column(i)]);
      }
    } else {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        zero[i] = 0.0f;
        scale[i] = 0.0f;
      }
    }

    float acc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      acc[i] = 0.0f;
    }
    float xsum = 0.0f;

    int k = k0;
    for (; k + load_unroll <= k1; k += load_unroll) {
      // Issued together: nothing below consumes a value until all of them are
      // in flight, which is the whole point of the unroll.
      int32_t q[load_unroll];
      float xv[load_unroll];
      if constexpr(blocked) {
        if(active && (k&3)==0) {
#pragma unroll
          for(int u=0;u<load_unroll;u+=4) {
            int4 batch=*reinterpret_cast<const int4*>(wp+int64_t((k+u)/4)*128);
            q[u]=batch.x;q[u+1]=batch.y;q[u+2]=batch.z;q[u+3]=batch.w;
          }
        } else {
#pragma unroll
          for(int u=0;u<load_unroll;++u)q[u]=active?load_word(k+u):0;
        }
      } else {
#pragma unroll
        for(int u=0;u<load_unroll;++u)q[u]=active?load_word(k+u):0;
      }
#pragma unroll
      for(int u=0;u<load_unroll;++u)xv[u]=xs[k+u-k_begin];
#pragma unroll
      for (int u = 0; u < load_unroll; ++u) {
        xsum += xv[u];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          acc[i] += xv[u] * awq_magic_float(uint32_t(q[u]) >> (4 * i));
        }
      }
    }
    for (; k < k1; ++k) {
      const float xv = xs[k - k_begin];
      const int32_t q = active ? load_word(k) : 0;
      xsum += xv;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        acc[i] += xv * awq_magic_float(uint32_t(q) >> (4 * i));
      }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      total[i] += scale[i] * (acc[i] - zero[i] * xsum);
    }
    k0 = k1;
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    red[warp * 256 + lane * 8 + awq_nibble_column(i)] = total[i];
  }
  __syncthreads();
  const int column = threadIdx.x;  // 0..255 within the tile
  float sum = 0.0f;
#pragma unroll
  for (int w = 0; w < 8; ++w) {
    sum += red[w * 256 + column];
  }
  const int n = tile * 256 + column;
  if (n < cols) {
    if (slot_scale != nullptr) sum *= slot_scale[slot];
    partial[(static_cast<int64_t>(slot) * splits + split) * cols + n] = sum;
  }
  if constexpr (persistent) __syncthreads();  // Shared staging is reused by the next task.
  }
}

// out[n] = sum_{c < count} partial[c][n], as BF16.
