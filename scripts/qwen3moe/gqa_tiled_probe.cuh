#pragma once
#include "../../crates/apxinf-cuda/kernels/custom/gqa_decode.cuh"

// BF16 tensor-core QK scores, FP32 softmax and PV accumulation. Q has eight
// real rows (padded to 16); each CTA owns one KV head and one sequence split.
template<int tile_tokens>
__global__ void gqa_decode_tiled_bf16_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    float* partial, const uint32_t* position, int capacity, int splits,
    int kv_heads, float scale) {
#if __CUDA_ARCH__ >= 800
  using namespace nvcuda;
  static_assert(tile_tokens == 32 || tile_tokens == 64 || tile_tokens == 128);
  extern __shared__ __align__(32) unsigned char workspace[];
  auto queries=reinterpret_cast<__nv_bfloat16*>(workspace);
  auto keys=queries+16*136;
  auto values=keys+tile_tokens*136;
  auto scores=reinterpret_cast<float*>(values+tile_tokens*128);
  auto probabilities_hi=reinterpret_cast<__nv_bfloat16*>(scores+16*tile_tokens);
  auto probabilities_lo=probabilities_hi+16*(tile_tokens+8);
  auto products=reinterpret_cast<float*>(probabilities_lo+16*(tile_tokens+8));
  const int lane=threadIdx.x%32,warp=threadIdx.x/32;
  const int kv=blockIdx.x,split=blockIdx.y,head=kv*8+warp;
  const int length=int(min(uint64_t(*position)+1,uint64_t(capacity))),span=(length+splits-1)/splits;
  const int begin=min(split*span,length),end=min(begin+span,length);
  for(int i=threadIdx.x;i<16*128;i+=256)
    queries[(i/128)*136+i%128]=i/128<8?q[kv*8*128+i]:__float2bfloat16(0);
  __syncthreads();
  float acc[4]={0,0,0,0},maximum=-INFINITY,total=0;
  for(int tile=begin;tile<end;tile+=tile_tokens) {
    const int count=min(tile_tokens,end-tile);
    for(int i=threadIdx.x;i<tile_tokens*128;i+=256) {
      const int t=i/128,d=i%128;
      const int64_t offset=(int64_t(kv)*capacity+tile+t)*128+d;
      keys[t*136+d]=t<count?k[offset]:__float2bfloat16(0);
      values[i]=t<count?v[offset]:__float2bfloat16(0);
    }
    __syncthreads();
    if(warp<tile_tokens/16) {
      wmma::fragment<wmma::accumulator,16,16,16,float> c;
      wmma::fill_fragment(c,0.f);
#pragma unroll
      for(int kk=0;kk<128;kk+=16) {
        wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> a;
        wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> b;
        wmma::load_matrix_sync(a,queries+kk,136);
        wmma::load_matrix_sync(b,keys+warp*16*136+kk,136);
        wmma::mma_sync(c,a,b,c);
      }
      wmma::store_matrix_sync(scores+warp*16,c,tile_tokens,wmma::mem_row_major);
    }
    __syncthreads();
    float dots[tile_tokens/32],tile_max=-INFINITY;
#pragma unroll
    for(int i=0;i<tile_tokens/32;++i) {
      dots[i]=lane+i*32<count?scores[warp*tile_tokens+lane+i*32]*scale:-INFINITY;
      tile_max=fmaxf(tile_max,dots[i]);
    }
#pragma unroll
    for(int delta=16;delta>0;delta/=2)tile_max=fmaxf(tile_max,__shfl_xor_sync(0xffffffff,tile_max,delta));
    float next=fmaxf(maximum,tile_max),alpha=expf(maximum-next),tile_sum=0;
#pragma unroll
    for(int i=0;i<tile_tokens/32;++i) {
      float p=expf(dots[i]-next);tile_sum+=p;
      auto hi=__float2bfloat16(p);
      probabilities_hi[warp*(tile_tokens+8)+lane+i*32]=hi;
      probabilities_lo[warp*(tile_tokens+8)+lane+i*32]=__float2bfloat16(p-__bfloat162float(hi));
      probabilities_hi[(warp+8)*(tile_tokens+8)+lane+i*32]=__float2bfloat16(0);
      probabilities_lo[(warp+8)*(tile_tokens+8)+lane+i*32]=__float2bfloat16(0);
    }
#pragma unroll
    for(int delta=16;delta>0;delta/=2)tile_sum+=__shfl_xor_sync(0xffffffff,tile_sum,delta);
    __syncthreads();
    total=total*alpha+tile_sum;
    wmma::fragment<wmma::accumulator,16,16,16,float> pv;
    wmma::fill_fragment(pv,0.f);
#pragma unroll
    for(int kk=0;kk<tile_tokens;kk+=16) {
      wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> ph,pl;
      wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> b;
      wmma::load_matrix_sync(ph,probabilities_hi+kk,tile_tokens+8);
      wmma::load_matrix_sync(pl,probabilities_lo+kk,tile_tokens+8);
      wmma::load_matrix_sync(b,values+kk*128+warp*16,128);
      wmma::mma_sync(pv,pl,b,pv);
      wmma::mma_sync(pv,ph,b,pv);
    }
    wmma::store_matrix_sync(products+warp*16,pv,128,wmma::mem_row_major);
    __syncthreads();
#pragma unroll
    for(int d=0;d<4;++d)acc[d]=acc[d]*alpha+products[warp*128+lane+d*32];
    maximum=next;
    __syncthreads();
  }
  float* out=partial+(int64_t(split)*kv_heads*8+head)*130;
  if(lane==0){out[128]=maximum;out[129]=total;}
#pragma unroll
  for(int d=0;d<4;++d)out[lane+d*32]=acc[d];
#endif
}
