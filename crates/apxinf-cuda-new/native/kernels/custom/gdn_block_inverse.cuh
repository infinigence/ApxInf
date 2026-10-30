#pragma once
// 64x64 triangular inverse composed from 16x16 diagonal and TF32 block products.
// Algorithm follows FLA chunk_fwd.py (MIT license, FLA contributors).
// The reference issues TF32 MMA without a rounding conversion; truncate the
// mantissa before these FP32 products to preserve that input precision.
__device__ float gdn_tf32_truncate(float x) { return __uint_as_float(__float_as_uint(x)&0xffffe000u); }
__device__ void gdn_mul16_tf32(const float* a,int as,const float* b,int bs,float* out,int os) {
  int row=threadIdx.x/16,col=threadIdx.x%16;float value=0;
  for(int k=0;k<16;++k) value=fmaf(gdn_tf32_truncate(a[row*as+k]),
      gdn_tf32_truncate(b[k*bs+col]),value);
  out[row*os+col]=value;__syncthreads();
}
__global__ void gdn_block_inverse64_kernel(float* a) {
  __shared__ float raw[4096],inv[4096],t0[256],t1[256],t2[256];
  const int base=blockIdx.x*4096;
  for(int i=threadIdx.x;i<4096;i+=256)raw[i]=inv[i]=a[base+i];
  __syncthreads();
  const int group=threadIdx.x/64,j=threadIdx.x%64,start=group*16;
  for(int row=2;row<16;++row) {
    if(j<row) {
      float products[16];
      for(int k=0;k<16;++k)products[k]=k<row ? __fmul_rn(raw[(start+row)*64+start+k],inv[(start+k)*64+start+j]) : 0.f;
      for(int offset=8;offset>0;offset>>=1)
        for(int k=0;k<offset;++k)products[k]=__fadd_rn(products[k],products[k+offset]);
      inv[(start+row)*64+start+j]=__fadd_rn(raw[(start+row)*64+start+j],products[0]);
    }
    __syncthreads();
  }
  for(int i=threadIdx.x;i<64;i+=256)inv[i*64+i]=1.f;
  __syncthreads();
  // raw stores negative lower blocks, so these products already carry the minus sign.
  gdn_mul16_tf32(inv+16*64+16,64,raw+16*64,64,t0,16);
  gdn_mul16_tf32(t0,16,inv,64,inv+16*64,64);
  gdn_mul16_tf32(inv+32*64+32,64,raw+32*64+16,64,t0,16);
  gdn_mul16_tf32(t0,16,inv+16*64+16,64,inv+32*64+16,64);
  gdn_mul16_tf32(inv+48*64+48,64,raw+48*64+32,64,t0,16);
  gdn_mul16_tf32(t0,16,inv+32*64+32,64,inv+48*64+32,64);
  gdn_mul16_tf32(raw+32*64,64,inv,64,t0,16);
  gdn_mul16_tf32(raw+32*64+16,64,inv+16*64,64,t1,16);
  t0[threadIdx.x]+=t1[threadIdx.x];__syncthreads();
  gdn_mul16_tf32(inv+32*64+32,64,t0,16,inv+32*64,64);
  gdn_mul16_tf32(raw+48*64+16,64,inv+16*64+16,64,t0,16);
  gdn_mul16_tf32(raw+48*64+32,64,inv+32*64+16,64,t1,16);
  t0[threadIdx.x]+=t1[threadIdx.x];__syncthreads();
  gdn_mul16_tf32(inv+48*64+48,64,t0,16,inv+48*64+16,64);
  gdn_mul16_tf32(raw+48*64,64,inv,64,t0,16);
  gdn_mul16_tf32(raw+48*64+16,64,inv+16*64,64,t1,16);
  gdn_mul16_tf32(raw+48*64+32,64,inv+32*64,64,t2,16);
  t0[threadIdx.x]=t0[threadIdx.x]+t1[threadIdx.x]+t2[threadIdx.x];__syncthreads();
  gdn_mul16_tf32(inv+48*64+48,64,t0,16,inv+48*64,64);
  for(int i=threadIdx.x;i<4096;i+=256)a[base+i]=__bfloat162float(__float2bfloat16(inv[i]));
}
