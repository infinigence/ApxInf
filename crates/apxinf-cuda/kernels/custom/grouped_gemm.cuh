#pragma once
#include <mma.h>

// SM80 BF16 tensor-core baseline: one CTA per (expert, M64, N128) tile.
// Host supplies int32 triples [expert, first grouped row, exclusive row end].
// Packed AWQ weights are expanded only in shared memory, never in global.
__global__ void w4a16_grouped_bf16_kernel(
    const __nv_bfloat16* __restrict__ input,
    const int32_t* __restrict__ qweight, const int32_t* __restrict__ qzeros,
    const half* __restrict__ scales, const int32_t* __restrict__ tiles,
    __nv_bfloat16* __restrict__ output, int k_size, int n_size,
    int group_size, int64_t stride_q, int64_t stride_z, int64_t stride_s) {
#if __CUDA_ARCH__ >= 800
  using namespace nvcuda;
  __shared__ __align__(32) unsigned char storage[32768];
  auto* a = reinterpret_cast<__nv_bfloat16*>(storage);
  auto* b = a + 64 * 72;
  auto* result = reinterpret_cast<float*>(storage);
  const int expert = tiles[blockIdx.x * 3];
  const int row_start = tiles[blockIdx.x * 3 + 1];
  const int row_end = tiles[blockIdx.x * 3 + 2];
  const int col_start = blockIdx.y * 128;
  const int warp = threadIdx.x / 32;
  const int wm = (warp / 2) * 32, wn = (warp % 2) * 64;
  qweight += expert * stride_q;
  qzeros += expert * stride_z;
  scales += expert * stride_s;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][4];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) wmma::fill_fragment(acc[i][j], 0.0f);
  for (int kb = 0; kb < k_size; kb += 64) {
    uint4 av[4];
#pragma unroll
    for (int u=0;u<4;++u) {
      const int p=threadIdx.x+u*128;
      const int row=row_start+p/8, k=kb+(p%8)*8;
      av[u]=row<row_end?*reinterpret_cast<const uint4*>(input+int64_t(row)*k_size+k):make_uint4(0,0,0,0);
    }
#pragma unroll
    for (int u=0;u<4;++u) {
      const int p=threadIdx.x+u*128;
      *reinterpret_cast<uint4*>(a+(p/8)*72+(p%8)*8)=av[u];
    }
    const int col=col_start+(threadIdx.x%16)*8;
    const int group=kb/128;
    uint32_t qwords[8];
    uint32_t zero=0;uint4 scale_words=make_uint4(0,0,0,0);
    if(col<n_size) {
      zero=qzeros[int64_t(group)*(n_size/8)+col/8];
      scale_words=*reinterpret_cast<const uint4*>(scales+int64_t(group)*n_size+col);
    }
#pragma unroll
    for(int u=0;u<8;++u) {
      int k=kb+threadIdx.x/16+u*8;
      qwords[u]=col<n_size?qweight[int64_t(k)*(n_size/8)+col/8]:0;
    }
    const half* scale_values=reinterpret_cast<const half*>(&scale_words);
#pragma unroll
    for(int u=0;u<8;++u) {
      __align__(16) __nv_bfloat16 expanded[8];
#pragma unroll
      for(int nib=0;nib<8;++nib) {
        int c=((nib&3)<<1)|(nib>>2);
        float value=float((qwords[u]>>(4*nib))&15)-float((zero>>(4*nib))&15);
        expanded[c]=__float2bfloat16(value*__half2float(scale_values[c]));
      }
      *reinterpret_cast<uint4*>(b+(threadIdx.x/16+u*8)*136+(threadIdx.x%16)*8)=*reinterpret_cast<const uint4*>(expanded);
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < 64; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af[2];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bf[4];
#pragma unroll
      for (int i = 0; i < 2; ++i) wmma::load_matrix_sync(af[i], a + (wm + i * 16) * 72 + kk, 72);
#pragma unroll
      for (int j = 0; j < 4; ++j) wmma::load_matrix_sync(bf[j], b + kk * 136 + wn + j * 16, 136);
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j)
      wmma::store_matrix_sync(result + (wm + i * 16) * 128 + wn + j * 16,
                              acc[i][j], 128, wmma::mem_row_major);
  __syncthreads();
  for (int p = threadIdx.x; p < 64 * 128; p += 128) {
    const int row = row_start + p / 128, col = col_start + p % 128;
    if (row < row_end && col < n_size) output[int64_t(row) * n_size + col] = __float2bfloat16(result[p]);
  }
#endif
}
