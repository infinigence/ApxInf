#pragma once
#include <mma.h>

// BF16 tensor-core QK scores, FP32 softmax and PV accumulation. Q has eight
// real rows (padded to 16); each CTA owns one KV head and one sequence split.
__global__ void gqa_decode_partial_bf16_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    float* partial, const uint32_t* position, int capacity, int splits,
    int kv_heads, float scale) {
#if __CUDA_ARCH__ >= 800
  using namespace nvcuda;
  __shared__ __align__(32) __nv_bfloat16 queries[16*136], keys[32*136], values[32*128];
  __shared__ __align__(32) float scores[16*32];
  __shared__ __align__(32) __nv_bfloat16 probabilities_hi[16*40], probabilities_lo[16*40];
  __shared__ __align__(32) float products[16*128];
  const int lane=threadIdx.x%32,warp=threadIdx.x/32;
  const int kv=blockIdx.x,split=blockIdx.y,head=kv*8+warp;
  const int length=int(min(uint64_t(*position)+1,uint64_t(capacity))),span=(length+splits-1)/splits;
  const int begin=min(split*span,length),end=min(begin+span,length);
  for(int i=threadIdx.x;i<16*128;i+=256)
    queries[(i/128)*136+i%128]=i/128<8?q[kv*8*128+i]:__float2bfloat16(0);
  __syncthreads();
  float acc[4]={0,0,0,0},maximum=-INFINITY,total=0;
  for(int tile=begin;tile<end;tile+=32) {
    const int count=min(32,end-tile);
    for(int i=threadIdx.x;i<32*128;i+=256) {
      const int t=i/128,d=i%128;
      const int64_t offset=(int64_t(kv)*capacity+tile+t)*128+d;
      keys[t*136+d]=t<count?k[offset]:__float2bfloat16(0);
      values[i]=t<count?v[offset]:__float2bfloat16(0);
    }
    __syncthreads();
    if(warp<2) {
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
      wmma::store_matrix_sync(scores+warp*16,c,32,wmma::mem_row_major);
    }
    __syncthreads();
    float dot=lane<count?scores[warp*32+lane]*scale:-INFINITY,tile_max=dot;
#pragma unroll
    for(int delta=16;delta>0;delta/=2)tile_max=fmaxf(tile_max,__shfl_xor_sync(0xffffffff,tile_max,delta));
    float next=fmaxf(maximum,tile_max),alpha=expf(maximum-next),p=expf(dot-next),tile_sum=p;
    auto hi = __float2bfloat16(p);
    probabilities_hi[warp*40+lane]=hi;
    probabilities_lo[warp*40+lane]=__float2bfloat16(p-__bfloat162float(hi));
    probabilities_hi[(warp+8)*40+lane]=__float2bfloat16(0);
    probabilities_lo[(warp+8)*40+lane]=__float2bfloat16(0);
#pragma unroll
    for(int delta=16;delta>0;delta/=2)tile_sum+=__shfl_xor_sync(0xffffffff,tile_sum,delta);
    __syncthreads();
    total=total*alpha+tile_sum;
    wmma::fragment<wmma::accumulator,16,16,16,float> pv;
    wmma::fill_fragment(pv,0.f);
#pragma unroll
    for(int kk=0;kk<32;kk+=16) {
      wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> ph,pl;
      wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> b;
      wmma::load_matrix_sync(ph,probabilities_hi+kk,40);
      wmma::load_matrix_sync(pl,probabilities_lo+kk,40);
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

// BF16 tensor-core QK scores, FP32 softmax and PV accumulation. Q has eight
// real rows (padded to 16); each CTA owns one KV head and one sequence split.
template<bool BalancedTiles=false>
__global__ void gqa_decode_vector_bf16_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    float* partial, const uint32_t* position, int capacity, int splits,
    int kv_heads, float scale) {
#if __CUDA_ARCH__ >= 800
  using namespace nvcuda;
  __shared__ __align__(32) __nv_bfloat16 queries[16*136], keys[32*136], values[32*128];
  __shared__ __align__(32) float scores[16*32];
  __shared__ __align__(32) __nv_bfloat16 probabilities_hi[16*40], probabilities_lo[16*40];
  __shared__ __align__(32) float products[16*128];
  const int lane=threadIdx.x%32,warp=threadIdx.x/32;
  const int kv=blockIdx.x,split=blockIdx.y,head=kv*8+warp;
  const int length=int(min(uint64_t(*position)+1,uint64_t(capacity))),span=(length+splits-1)/splits;
  const int begin=min(split*span,length),end=min(begin+span,length);
  // Balance whole 32-token tiles after the short-context range. This avoids
  // assigning almost every split a mostly empty tile just past boundaries.
  int balanced_begin=begin,balanced_end=end;
  if constexpr (BalancedTiles) {
    if(length>512) {
      const int tiles=(length+31)/32,per=tiles/splits,extra=tiles%splits;
      balanced_begin=min((split*per+min(split,extra))*32,length);
      balanced_end=min(balanced_begin+(per+(split<extra))*32,length);
    }
  }
  for(int i=threadIdx.x;i<16*16;i+=256) {
    const int headrow=i/16,d=(i%16)*8;
    uint4 data={0,0,0,0};
    if(headrow<8)data=*reinterpret_cast<const uint4*>(q+kv*8*128+headrow*128+d);
    *reinterpret_cast<uint4*>(queries+headrow*136+d)=data;
  }
  __syncthreads();
  float acc[4]={0,0,0,0},maximum=-INFINITY,total=0;
  for(int tile=balanced_begin;tile<balanced_end;tile+=32) {
    const int count=min(32,balanced_end-tile);
    for(int i=threadIdx.x;i<32*16;i+=256) {
      const int t=i/16,d=(i%16)*8;
      uint4 key4={0,0,0,0},value4={0,0,0,0};
      if(t<count) {
        const int64_t offset=(int64_t(kv)*capacity+tile+t)*128+d;
        key4=*reinterpret_cast<const uint4*>(k+offset);
        value4=*reinterpret_cast<const uint4*>(v+offset);
      }
      *reinterpret_cast<uint4*>(keys+t*136+d)=key4;
      *reinterpret_cast<uint4*>(values+t*128+d)=value4;
    }
    __syncthreads();
    if(warp<2) {
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
      wmma::store_matrix_sync(scores+warp*16,c,32,wmma::mem_row_major);
    }
    __syncthreads();
    float dot=lane<count?scores[warp*32+lane]*scale:-INFINITY,tile_max=dot;
#pragma unroll
    for(int delta=16;delta>0;delta/=2)tile_max=fmaxf(tile_max,__shfl_xor_sync(0xffffffff,tile_max,delta));
    float next=fmaxf(maximum,tile_max),alpha=expf(maximum-next),p=expf(dot-next),tile_sum=p;
    auto hi = __float2bfloat16(p);
    probabilities_hi[warp*40+lane]=hi;
    probabilities_lo[warp*40+lane]=__float2bfloat16(p-__bfloat162float(hi));
    probabilities_hi[(warp+8)*40+lane]=__float2bfloat16(0);
    probabilities_lo[(warp+8)*40+lane]=__float2bfloat16(0);
#pragma unroll
    for(int delta=16;delta>0;delta/=2)tile_sum+=__shfl_xor_sync(0xffffffff,tile_sum,delta);
    __syncthreads();
    total=total*alpha+tile_sum;
    wmma::fragment<wmma::accumulator,16,16,16,float> pv;
    wmma::fill_fragment(pv,0.f);
#pragma unroll
    for(int kk=0;kk<32;kk+=16) {
      wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> ph,pl;
      wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> b;
      wmma::load_matrix_sync(ph,probabilities_hi+kk,40);
      wmma::load_matrix_sync(pl,probabilities_lo+kk,40);
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

__global__ void gqa_decode_combine_bf16_kernel(const float* partial,
    __nv_bfloat16* output,int heads,int splits) {
  const int head=blockIdx.x,d=threadIdx.x;
  float maximum=-INFINITY;
  for(int s=0;s<splits;++s)maximum=fmaxf(maximum,partial[(int64_t(s)*heads+head)*130+128]);
  float sum=0,acc=0;
  for(int s=0;s<splits;++s) {
    const float* p=partial+(int64_t(s)*heads+head)*130;
    // Empty splits have maximum=-inf and total=0.
    float factor=p[129]>0?expf(p[128]-maximum):0;
    sum+=factor*p[129];acc+=factor*p[d];
  }
  output[head*128+d]=__float2bfloat16(sum>0?acc/sum:0);
}
