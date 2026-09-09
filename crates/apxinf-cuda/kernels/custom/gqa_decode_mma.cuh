#pragma once
#include "gqa_decode.cuh"

// Eight real heads occupy the N=8 axis of m16n8k16 for both K*Q^T
// and V^T*P^T. This removes padded heads while retaining FP32 softmax
// and high/low BF16 probabilities. One CTA owns a KV head/sequence split.
__device__ __forceinline__ uint32_t gqa_bf16_pair(__nv_bfloat16 lo,__nv_bfloat16 hi) {
  return uint32_t(__bfloat16_as_ushort(lo)) | (uint32_t(__bfloat16_as_ushort(hi))<<16);
}
__device__ __forceinline__ void gqa_mma_16x8(const uint32_t* a,const uint32_t* b,float* c) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
    : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
    : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
}

template<int tile_tokens>
__global__ void gqa_decode_mma_bf16_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    float* partial, const uint32_t* position, int capacity, int splits,
    int kv_heads, float scale) {
#if __CUDA_ARCH__ >= 800
  static_assert(tile_tokens == 32 || tile_tokens == 64 || tile_tokens == 128);
  extern __shared__ __align__(32) unsigned char workspace[];
  auto queries=reinterpret_cast<__nv_bfloat16*>(workspace);
  auto keys=queries+8*136;
  auto values=keys+tile_tokens*136;
  auto scores=reinterpret_cast<float*>(values+tile_tokens*128);
  auto probabilities_hi=reinterpret_cast<__nv_bfloat16*>(scores+8*(tile_tokens+4));
  auto probabilities_lo=probabilities_hi+8*(tile_tokens+8);
  auto products=reinterpret_cast<float*>(probabilities_lo+8*(tile_tokens+8));
  const int lane=threadIdx.x%32,warp=threadIdx.x/32;
  const int kv=blockIdx.x,split=blockIdx.y,head=kv*8+warp;
  const int length=int(min(uint64_t(*position)+1,uint64_t(capacity))),span=(length+splits-1)/splits;
  const int begin=min(split*span,length),end=min(begin+span,length);
  for(int i=threadIdx.x;i<8*128;i+=256)
    queries[(i/128)*136+i%128]=q[kv*8*128+i];
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
      float c[4]={0,0,0,0};
      const int row=warp*16+lane/4,col=(lane%4)*2;
#pragma unroll
      for(int kk=0;kk<128;kk+=16) {
        uint32_t a[4],b[2];
        a[0]=*reinterpret_cast<const uint32_t*>(keys+row*136+kk+col);
        a[1]=*reinterpret_cast<const uint32_t*>(keys+(row+8)*136+kk+col);
        a[2]=*reinterpret_cast<const uint32_t*>(keys+row*136+kk+col+8);
        a[3]=*reinterpret_cast<const uint32_t*>(keys+(row+8)*136+kk+col+8);
        b[0]=*reinterpret_cast<const uint32_t*>(queries+(lane/4)*136+kk+col);
        b[1]=*reinterpret_cast<const uint32_t*>(queries+(lane/4)*136+kk+col+8);
        gqa_mma_16x8(a,b,c);
      }
      scores[col*(tile_tokens+4)+row]=c[0];
      scores[(col+1)*(tile_tokens+4)+row]=c[1];
      scores[col*(tile_tokens+4)+row+8]=c[2];
      scores[(col+1)*(tile_tokens+4)+row+8]=c[3];
    }
    __syncthreads();
    float dots[tile_tokens/32],tile_max=-INFINITY;
#pragma unroll
    for(int i=0;i<tile_tokens/32;++i) {
      dots[i]=lane+i*32<count?scores[warp*(tile_tokens+4)+lane+i*32]*scale:-INFINITY;
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
    }
#pragma unroll
    for(int delta=16;delta>0;delta/=2)tile_sum+=__shfl_xor_sync(0xffffffff,tile_sum,delta);
    __syncthreads();
    total=total*alpha+tile_sum;
    float pv[4]={0,0,0,0};
    const int drow=warp*16+lane/4,tcol=(lane%4)*2;
#pragma unroll
    for(int kk=0;kk<tile_tokens;kk+=16) {
      uint32_t a[4],bh[2],bl[2];
      a[0]=gqa_bf16_pair(values[(kk+tcol)*128+drow],values[(kk+tcol+1)*128+drow]);
      a[1]=gqa_bf16_pair(values[(kk+tcol)*128+drow+8],values[(kk+tcol+1)*128+drow+8]);
      a[2]=gqa_bf16_pair(values[(kk+tcol+8)*128+drow],values[(kk+tcol+9)*128+drow]);
      a[3]=gqa_bf16_pair(values[(kk+tcol+8)*128+drow+8],values[(kk+tcol+9)*128+drow+8]);
      bh[0]=*reinterpret_cast<const uint32_t*>(probabilities_hi+(lane/4)*(tile_tokens+8)+kk+tcol);
      bh[1]=*reinterpret_cast<const uint32_t*>(probabilities_hi+(lane/4)*(tile_tokens+8)+kk+tcol+8);
      bl[0]=*reinterpret_cast<const uint32_t*>(probabilities_lo+(lane/4)*(tile_tokens+8)+kk+tcol);
      bl[1]=*reinterpret_cast<const uint32_t*>(probabilities_lo+(lane/4)*(tile_tokens+8)+kk+tcol+8);
      gqa_mma_16x8(a,bl,pv);gqa_mma_16x8(a,bh,pv);
    }
    products[tcol*132+drow]=pv[0];products[(tcol+1)*132+drow]=pv[1];
    products[tcol*132+drow+8]=pv[2];products[(tcol+1)*132+drow+8]=pv[3];
    __syncthreads();
#pragma unroll
    for(int d=0;d<4;++d)acc[d]=acc[d]*alpha+products[warp*132+lane+d*32];
    maximum=next;
    __syncthreads();
  }
  float* out=partial+(int64_t(split)*kv_heads*8+head)*130;
  if(lane==0){out[128]=maximum;out[129]=total;}
#pragma unroll
  for(int d=0;d<4;++d)out[lane+d*32]=acc[d];
#endif
}
