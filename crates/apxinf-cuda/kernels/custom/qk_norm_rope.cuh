#pragma once

// Match the core RoPE contraction: low uses FMA(low, cos, -high*sin);
// high uses FMA(high, cos, low*sin).
// Leaving contraction to the compiler changes BF16 results near cancellation.
__device__ __forceinline__ __nv_bfloat162 qk_rope_pair(float low,float high,float cosine,float sine) {
  return __floats2bfloat162_rn(__fmaf_rn(low,cosine,-__fmul_rn(high,sine)),
                             __fmaf_rn(high,cosine,__fmul_rn(low,sine)));
}

// One warp per head row, preserving the four original 32-element norm
// reductions and the BF16 rounding boundaries before RoPE and FP16 casting.
template<bool cached_f16=false>
__global__ void qk_norm_rope_append_f16_kernel(
    const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,
    const __nv_bfloat16* qw,const __nv_bfloat16* kw,
    half* oq,half* ok,half* ov,__nv_bfloat16* cache_k,__nv_bfloat16* cache_v,
    int tokens,int qheads,int kvheads,int capacity,int offset,float eps,const float* rope,
    const int32_t* used_k=nullptr) {
  if constexpr(cached_f16) {
    offset=*used_k-tokens;
    if(offset<0 || offset>capacity-tokens)return;
  }
  int row=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;
  if(row>=tokens*(qheads+kvheads))return;
  int token=row/(qheads+kvheads),head=row%(qheads+kvheads);
  bool is_q=head<qheads;int h=is_q?head:head-qheads,heads=is_q?qheads:kvheads;
  const __nv_bfloat16* input=(is_q?q:k)+(int64_t(token)*heads+h)*128;
  const __nv_bfloat16* weight=is_q?qw:kw;
  float x[4],sum[4];
#pragma unroll
  for(int i=0;i<4;++i) { x[i]=__bfloat162float(input[lane+i*32]);sum[i]=x[i]*x[i]; }
#pragma unroll
  for(int delta=16;delta>0;delta/=2)
#pragma unroll
    for(int i=0;i<4;++i)sum[i]+=__shfl_xor_sync(0xffffffff,sum[i],delta);
  float square_sum=(sum[0]+sum[2])+(sum[1]+sum[3]);
  // The original norm broadcasts lane zero; other XOR lanes can differ by an ULP.
  square_sum=__shfl_sync(0xffffffff,square_sum,0);
  float inv=rsqrtf(square_sum/128.f+eps);
#pragma unroll
  for(int i=0;i<4;++i)x[i]=__bfloat162float(__float2bfloat16(x[i]*inv*__bfloat162float(weight[lane+i*32])));
  half* out=(is_q?oq:ok)+(cached_f16 && !is_q ? int64_t(h)*capacity+offset+token : int64_t(token)*heads+h)*128;
#pragma unroll
  for(int i=0;i<2;++i) {
    int pair=lane+i*32;
    int64_t index=(int64_t(offset+token)*64+pair)*2;
    float cosine=rope[index],sine=rope[index+1];
    __nv_bfloat16 low=__float2bfloat16(__fmaf_rn(x[i],cosine,-__fmul_rn(x[i+2],sine)));
    __nv_bfloat16 high=__float2bfloat16(__fmaf_rn(x[i+2],cosine,__fmul_rn(x[i],sine)));
    out[pair]=__float2half(__bfloat162float(low));out[pair+64]=__float2half(__bfloat162float(high));
    if(!is_q) {
      int64_t base=(int64_t(h)*capacity+offset+token)*128;
      cache_k[base+pair]=low;cache_k[base+pair+64]=high;
    }
  }
  if(!is_q) {
    int64_t source=(int64_t(token)*kvheads+h)*128,cache=(int64_t(h)*capacity+offset+token)*128;
#pragma unroll
    for(int i=0;i<4;++i) {
      int d=lane+i*32;__nv_bfloat16 value=v[source+d];
      ov[(cached_f16?cache:source)+d]=__float2half(__bfloat162float(value));cache_v[cache+d]=value;
    }
  }
}

// Shared across every layer; preserves the original float pow/sin/cos path.
__global__ void rope_table_f32_kernel(float* table,int positions,float theta) {
  for(int index=blockIdx.x*blockDim.x+threadIdx.x;index<positions*64;index+=gridDim.x*blockDim.x) {
    int position=index/64,pair=index%64;
    float frequency=1.f/powf(theta,2.f*float(pair)/128.f);
    float angle=float(position)*frequency;
    table[index*2]=cosf(angle);table[index*2+1]=sinf(angle);
  }
}

// Decode composition: sum split-K QKV in the original order, round to BF16,
// normalize Q/K, apply RoPE, and append K/V without intermediate launches.
__global__ void qkv_partial_norm_rope_cache_bf16_kernel(const float* partial,
    const __nv_bfloat16* qw,const __nv_bfloat16* kw,__nv_bfloat16* oq,
    __nv_bfloat16* cache_k,__nv_bfloat16* cache_v,const uint32_t* position,
    int qheads,int kvheads,int capacity,int splits,float eps,const float* rope) {
  int head=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;
  if(head>=qheads+kvheads || *position>=uint32_t(capacity))return;
  int pos=*position,cols=(qheads+2*kvheads)*128;
  bool is_q=head<qheads;int h=is_q?head:head-qheads;
  const __nv_bfloat16* weight=is_q?qw:kw;
  float x[4],sum[4];
#pragma unroll
  for(int i=0;i<4;++i) {
    float total=0;
    for(int split=0;split<splits;++split)total+=partial[int64_t(split)*cols+head*128+lane+i*32];
    x[i]=__bfloat162float(__float2bfloat16(total));sum[i]=x[i]*x[i];
  }
#pragma unroll
  for(int delta=16;delta>0;delta/=2)
#pragma unroll
    for(int i=0;i<4;++i)sum[i]+=__shfl_xor_sync(0xffffffff,sum[i],delta);
  float square_sum=(sum[0]+sum[2])+(sum[1]+sum[3]);
  // The original norm broadcasts lane zero; other XOR lanes can differ by an ULP.
  square_sum=__shfl_sync(0xffffffff,square_sum,0);
  float inv=rsqrtf(square_sum/128.f+eps);
#pragma unroll
  for(int i=0;i<4;++i)x[i]=__bfloat162float(__float2bfloat16(x[i]*inv*__bfloat162float(weight[lane+i*32])));
  __nv_bfloat16* out=is_q?oq+head*128:cache_k+(int64_t(h)*capacity+pos)*128;
#pragma unroll
  for(int i=0;i<2;++i) {
    int pair=lane+i*32;int64_t index=(int64_t(pos)*64+pair)*2;
    float cosine=rope[index],sine=rope[index+1];
    __nv_bfloat162 rotated=qk_rope_pair(x[i],x[i+2],cosine,sine);
    out[pair]=rotated.x;out[pair+64]=rotated.y;
  }
  if(!is_q) {
#pragma unroll
    for(int i=0;i<4;++i) {
      int d=lane+i*32;float total=0;
      for(int split=0;split<splits;++split)total+=partial[int64_t(split)*cols+(qheads+kvheads+h)*128+d];
      cache_v[(int64_t(h)*capacity+pos)*128+d]=__float2bfloat16(total);
    }
  }
}
