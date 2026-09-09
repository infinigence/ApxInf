#pragma once
// Exact paired INT4 conversion reuses the vendored Marlin dequant helper.
template<bool blocked=false, bool pdl=false, int fixed_group=0, bool vector_red=false>
__global__ void __launch_bounds__(256) w4a16_gemv_pair_kernel(
    const __nv_bfloat16* __restrict__ x, int64_t x_slot_stride,
    const int32_t* __restrict__ qweight, const int32_t* __restrict__ qzeros,
    const half* __restrict__ scales, const int32_t* __restrict__ expert_ids,
    int64_t stride_q, int64_t stride_z, int64_t stride_s,
    const float* __restrict__ slot_scale, float* __restrict__ partial,
    int rows, int packed_cols, int group_size_arg, int rows_per_block) {
  const int group_size = fixed_group ? fixed_group : group_size_arg;
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
  extern __shared__ __align__(16) float smem[];
  float* xs = smem;                       // [rows_per_block]
  float* red = smem + rows_per_block;     // [8 warps][256]

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int slot = blockIdx.z;
  const int split = blockIdx.y;
  const int cols = packed_cols * 8;
  const int expert = expert_ids == nullptr ? slot : expert_ids[slot];
  qweight += static_cast<int64_t>(expert) * stride_q;
  qzeros += static_cast<int64_t>(expert) * stride_z;
  scales += static_cast<int64_t>(expert) * stride_s;
  x += static_cast<int64_t>(slot) * x_slot_stride;

  const int k_begin = split * rows_per_block;
  const int k_end = min(rows, k_begin + rows_per_block);
  for (int k = k_begin + threadIdx.x; k < k_end; k += blockDim.x) {
    xs[k - k_begin] = __bfloat162float(x[k]);
  }
  __syncthreads();

  const int j = blockIdx.x * 32 + lane;
  const bool active = j < packed_cols;
  const int rows_per_warp = rows_per_block / 8;
  const int kw_begin = k_begin + warp * rows_per_warp;
  const int kw_end = min(k_end, kw_begin + rows_per_warp);

  float total[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    total[i] = 0.0f;
  }
  const int32_t* wp = blocked ? qweight + (int64_t(blockIdx.x)*(rows/4)*32+lane)*4
                              : (active ? qweight + j : qweight);
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
      const int32_t z = qzeros[static_cast<int64_t>(g) * packed_cols + j];
      const uint4 packed_scales = *reinterpret_cast<const uint4*>(
          scales + static_cast<int64_t>(g) * cols + j * 8);
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
    for (; k + 16 <= k1; k += 16) {
      // Issued together: nothing below consumes a value until all of them are
      // in flight, which is the whole point of the unroll.
      int32_t q[16];
      float xv[16];
      if constexpr(blocked) {
        if(active && (k&3)==0) {
#pragma unroll
          for(int u=0;u<16;u+=4) {
            int4 batch=*reinterpret_cast<const int4*>(wp+int64_t((k+u)/4)*128);
            q[u]=batch.x;q[u+1]=batch.y;q[u+2]=batch.z;q[u+3]=batch.w;
          }
        } else {
#pragma unroll
          for(int u=0;u<16;++u)q[u]=active?load_word(k+u):0;
        }
      } else {
#pragma unroll
        for(int u=0;u<16;++u)q[u]=active?load_word(k+u):0;
      }
#pragma unroll
      for(int u=0;u<16;++u)xv[u]=xs[k+u-k_begin];
#pragma unroll
      for (int u = 0; u < 16; ++u) {
        xsum += xv[u];
#pragma unroll
        for (int pair = 0; pair < 2; ++pair) {
          half2 halves[2];
          apxinf_decode_pair::dequant<half2, vllm::kU4.id(), false>(uint32_t(q[u]) >> (8 * pair), halves);
          float2 lo=__half22float2(halves[0]),hi=__half22float2(halves[1]);
          acc[2*pair] += xv[u]*lo.x;acc[2*pair+4] += xv[u]*lo.y;
          acc[2*pair+1] += xv[u]*hi.x;acc[2*pair+5] += xv[u]*hi.y;
        }
      }
    }
    for (; k < k1; ++k) {
      const float xv = xs[k - k_begin];
      const int32_t q = active ? load_word(k) : 0;
      xsum += xv;
#pragma unroll
      for (int pair = 0; pair < 2; ++pair) {
        half2 halves[2];
        apxinf_decode_pair::dequant<half2, vllm::kU4.id(), false>(uint32_t(q) >> (8 * pair), halves);
        float2 lo=__half22float2(halves[0]),hi=__half22float2(halves[1]);
        acc[2*pair] += xv*lo.x;acc[2*pair+4] += xv*lo.y;
        acc[2*pair+1] += xv*hi.x;acc[2*pair+5] += xv*hi.y;
      }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      total[i] += scale[i] * (acc[i] - zero[i] * xsum);
    }
    k0 = k1;
  }
  if constexpr(vector_red) {
    // rows_per_block is a multiple of eight, so each float4 is aligned.
    float4* row=reinterpret_cast<float4*>(red+warp*256+lane*8);
    row[0]=make_float4(total[0],total[4],total[1],total[5]);
    row[1]=make_float4(total[2],total[6],total[3],total[7]);
  } else {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    red[warp * 256 + lane * 8 + awq_nibble_column(i)] = total[i];
  }
  }
  __syncthreads();
  const int column = threadIdx.x;  // 0..255 within the tile
  float sum = 0.0f;
#pragma unroll
  for (int w = 0; w < 8; ++w) {
    sum += red[w * 256 + column];
  }
  const int n = blockIdx.x * 256 + column;
  if (n < cols) {
    if (slot_scale != nullptr) sum *= slot_scale[slot];
    partial[(static_cast<int64_t>(slot) * gridDim.y + split) * cols + n] = sum;
  }
}
