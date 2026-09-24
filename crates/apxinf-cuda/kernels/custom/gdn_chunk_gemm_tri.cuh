#pragma once
// Fixed-shape W/U specialization. The safe form checks finite operands and
// the lower-triangular A produced by gdn_block_inverse64_kernel; arbitrary
// dense or nonfinite inputs use the same ordered dense FMA loop.
// Preserve increasing-m FP32 FMA order. Replay skipped zeros when an
// accumulator is negative zero so signed-zero behavior matches the dense path.
// Positive zero plus either signed zero is positive zero under fmaf RN;
// with finite operands checked by SAFE, it needs no zero-FMA replay.
template <bool TRI, bool STAGE_A, bool SAFE=false, typename KInput=float, int TILE=4, bool DIRECT_V=false>
__global__ __launch_bounds__(256, 4) void gdn_chunk_gemm_tri_kernel(
    const float* a, const __nv_bfloat16* v, const KInput* k,
    const float* beta, const float* g, __nv_bfloat16* u,
    __nv_bfloat16* w, int seq_pad, int seq=0, int conv_dim=0, int v_offset=0) {
  static_assert(!SAFE || STAGE_A, "safe triangle dispatch requires staged A");
  static_assert(TILE == 4 || TILE == 8, "supported W/U tile sizes are 4 and 8");
  constexpr int C=64, D=128;
  const int tid=threadIdx.x;
  __shared__ int triangular_safe;
  if constexpr(SAFE){ if(tid==0) triangular_safe=1; __syncthreads(); }
  const long long tok=(long long)blockIdx.y*seq_pad+blockIdx.x*C;
  const long long ab=((long long)blockIdx.y*gridDim.x+blockIdx.x)*C*C;
  const long long ob=((long long)blockIdx.y*gridDim.x+blockIdx.x)*C*D;
  extern __shared__ float memory[];
  auto* vb=reinterpret_cast<__nv_bfloat16*>(memory);
  auto* kb=vb+C*D;
  float* sa=reinterpret_cast<float*>(kb+C*D);
  for(int p=tid;p<C*D;p+=256){
    int m=p/D,j=p%D;
    float b=beta[tok+m];
    __nv_bfloat16 vv;
    if constexpr(DIRECT_V) {
      const int token=blockIdx.x*C+m;
      vv=token<seq ? v[(long long)token*conv_dim+v_offset+blockIdx.y*D+j]
                   : __ushort_as_bfloat16(0);
    } else {
      vv=v[(tok+m)*D+j];
    }
    vb[p]=__float2bfloat16(__bfloat162float(vv)*b);
    float z=__bfloat162float(__float2bfloat16(gdn_qk_widen(k[(tok+m)*D+j])*b));
    kb[p]=__float2bfloat16(z*gdn_exp2_approx(g[tok+m]));
    if constexpr(SAFE) if(!isfinite(__bfloat162float(vb[p])) || !isfinite(__bfloat162float(kb[p]))) atomicExch(&triangular_safe,0);
  }
  if constexpr(STAGE_A) for(int p=tid;p<C*C;p+=256){
    float av=a[ab+p]; sa[p]=av;
    if constexpr(SAFE) if(!isfinite(av) || (p/C<p%C && av!=0.0f)) atomicExch(&triangular_safe,0);
  }
  __syncthreads();
  const bool triangular=TRI && (!SAFE || triangular_safe);
  for(int base=tid;base<C*D;base+=256*TILE){
    int row0=base/D,j=base%D;
    float au[TILE]={},aw[TILE]={};
    if constexpr(TILE == 8 && sizeof(KInput) == sizeof(__nv_bfloat16)) {
      if(triangular){
        // For TILE8 the first row is row0; all eight rows consume m=0..row0.
        for(int m=0;m<=row0;++m){
          float bv=__bfloat162float(vb[m*D+j]);
          float bk=__bfloat162float(kb[m*D+j]);
          #pragma unroll
          for(int s=0;s<TILE;++s){
            int row=row0+s*2;
            float av=STAGE_A?sa[row*C+m]:a[ab+row*C+m];
            au[s]=fmaf(av,bv,au[s]);
            aw[s]=fmaf(av,bk,aw[s]);
          }
        }
        // Fixed two-m tails: each pair removes one output row. Each remaining
        // accumulator still sees strictly increasing m through its own row.
        #pragma unroll
        for(int delta=1;delta<=2*(TILE-1);++delta){
          const int m=row0+delta;
          float bv=__bfloat162float(vb[m*D+j]);
          float bk=__bfloat162float(kb[m*D+j]);
          #pragma unroll
          for(int s=0;s<TILE;++s){
            if(delta<=2*s){
              int row=row0+s*2;
              float av=STAGE_A?sa[row*C+m]:a[ab+row*C+m];
              au[s]=fmaf(av,bv,au[s]);
              aw[s]=fmaf(av,bk,aw[s]);
            }
          }
        }
      }else{
        // Keep SAFE's dense/nonfinite fallback in its original increasing-m order.
        for(int m=0;m<C;++m){
          float bv=__bfloat162float(vb[m*D+j]);
          float bk=__bfloat162float(kb[m*D+j]);
          #pragma unroll
          for(int s=0;s<TILE;++s){
            int row=row0+s*2;
            float av=STAGE_A?sa[row*C+m]:a[ab+row*C+m];
            au[s]=fmaf(av,bv,au[s]);
            aw[s]=fmaf(av,bk,aw[s]);
          }
        }
      }
    } else {
      int limit=triangular ? row0+(TILE-1)*2+1 : C;
      for(int m=0;m<limit;++m){
        float bv=__bfloat162float(vb[m*D+j]);
        float bk=__bfloat162float(kb[m*D+j]);
        #pragma unroll
        for(int s=0;s<TILE;++s){
          int row=row0+s*2;
          if(!triangular || m<=row){
            float av=STAGE_A?sa[row*C+m]:a[ab+row*C+m];
            au[s]=fmaf(av,bv,au[s]);
            aw[s]=fmaf(av,bk,aw[s]);
          }
        }
      }
    }
    #pragma unroll
    for(int s=0;s<TILE;++s){
      int row=row0+s*2;
      if(triangular){
        if(__float_as_uint(au[s])==0x80000000u || __float_as_uint(aw[s])==0x80000000u){
          bool zu=__float_as_uint(au[s])==0x80000000u,zw=__float_as_uint(aw[s])==0x80000000u;
          for(int m=row+1;m<C;++m){
            float av=STAGE_A?sa[row*C+m]:a[ab+row*C+m];
            if(zu) au[s]=fmaf(av,__bfloat162float(vb[m*D+j]),au[s]);
            if(zw) aw[s]=fmaf(av,__bfloat162float(kb[m*D+j]),aw[s]);
          }
        }
      }
      u[ob+base+s*256]=__float2bfloat16(au[s]);
      w[ob+base+s*256]=__float2bfloat16(aw[s]);
    }
  }
}
