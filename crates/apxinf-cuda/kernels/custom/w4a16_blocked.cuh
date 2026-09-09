#pragma once
// AWQ qweight layout [expert, N/256, K/4, 32 packed columns, 4 K words].
// Adjacent lanes load adjacent int4 vectors; nibble accumulation order stays unchanged.
__global__ void w4a16_blocked_repack_kernel(const int32_t* source,int4* output,int rows,int packed_cols,int experts) {
  int64_t per_expert=int64_t(rows/4)*packed_cols,total=per_expert*experts;
  for(int64_t index=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;index<total;index+=int64_t(gridDim.x)*blockDim.x) {
    int64_t expert=index/per_expert,local=index%per_expert;
    int lane=local%32,k4=(local/32)%(rows/4),tile=local/(int64_t(rows/4)*32);
    int64_t offset=expert*int64_t(rows)*packed_cols+int64_t(k4*4)*packed_cols+tile*32+lane;
    int4 value;value.x=source[offset];value.y=source[offset+packed_cols];value.z=source[offset+2*packed_cols];value.w=source[offset+3*packed_cols];
    output[index]=value;
  }
}
__global__ void __launch_bounds__(256) w4a16_gemv_blocked_partial_kernel(
    const __nv_bfloat16* __restrict__ x, int64_t x_slot_stride,
    const int32_t* __restrict__ qweight, const int32_t* __restrict__ qzeros,
    const half* __restrict__ scales, const int32_t* __restrict__ expert_ids,
    int64_t stride_q, int64_t stride_z, int64_t stride_s,
    const float* __restrict__ slot_scale, float* __restrict__ partial,
    int rows, int packed_cols, int group_size, int rows_per_block) {
  extern __shared__ float smem[];
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
  const int32_t* wp = qweight + (int64_t(blockIdx.x)*(rows/4)*32+lane)*4;
  auto load_word = [&](int k) { return wp[int64_t(k/4)*128+(k&3)]; };

  // Walk one quantization group at a time. Within a group the scale and zero
  // point are constant, so the affine correction s*(acc - z*sum(x)) leaves the
  // inner loop entirely and the body reduces to W4A16_GEMV_UNROLL
  // *independent* loads followed by their FMAs.
  //
  // The unroll is the point, not the arithmetic — see W4A16_GEMV_UNROLL above.
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
    for (; k + W4A16_GEMV_UNROLL <= k1; k += W4A16_GEMV_UNROLL) {
      // Issued together: nothing below consumes a value until all of them are
      // in flight, which is the whole point of the unroll.
      int32_t q[W4A16_GEMV_UNROLL];
      float xv[W4A16_GEMV_UNROLL];
      if(active && (k&3)==0) {
#pragma unroll
        for(int u=0;u<W4A16_GEMV_UNROLL;u+=4) {
          int4 batch=*reinterpret_cast<const int4*>(wp+int64_t((k+u)/4)*128);
          q[u]=batch.x;q[u+1]=batch.y;q[u+2]=batch.z;q[u+3]=batch.w;
        }
      } else {
#pragma unroll
        for(int u=0;u<W4A16_GEMV_UNROLL;++u)q[u]=active?load_word(k+u):0;
      }
#pragma unroll
      for(int u=0;u<W4A16_GEMV_UNROLL;++u)xv[u]=xs[k+u-k_begin];
#pragma unroll
      for (int u = 0; u < W4A16_GEMV_UNROLL; ++u) {
        xsum += xv[u];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          acc[i] += xv[u] * static_cast<float>((q[u] >> (4 * i)) & 0xF);
        }
      }
    }
    for (; k < k1; ++k) {
      const float xv = xs[k - k_begin];
      const int32_t q = active ? load_word(k) : 0;
      xsum += xv;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        acc[i] += xv * static_cast<float>((q >> (4 * i)) & 0xF);
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
  const int n = blockIdx.x * 256 + column;
  if (n < cols) {
    if (slot_scale != nullptr) sum *= slot_scale[slot];
    partial[(static_cast<int64_t>(slot) * gridDim.y + split) * cols + n] = sum;
  }
}
