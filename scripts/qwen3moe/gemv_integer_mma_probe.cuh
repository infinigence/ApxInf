#pragma once
#include <mma.h>

// Diagnostic only. BF16 activations multiply exactly represented integer
// (q-zero) values on tensor cores; FP32 group scales are applied afterward.
// Accumulation order differs from the production scalar GEMV.
__global__ void gemv_integer_mma_probe_kernel(
    const __nv_bfloat16* x,int64_t x_stride,
    const int32_t* qweight,const int32_t* qzeros,const half* scales,
    const int32_t* expert_ids,int64_t stride_q,int64_t stride_z,int64_t stride_s,
    const float* slot_scale,float* partial,int rows,int cols,int rows_per_block) {
#if __CUDA_ARCH__ >= 800
  using namespace nvcuda;
  __shared__ __align__(32) __nv_bfloat16 lhs[16*136],rhs[128*136];
  __shared__ __align__(32) float products[16*128];
  __shared__ int zeros[16];
  const int tid=threadIdx.x,warp=tid/32,slot=blockIdx.z,split=blockIdx.y;
  const int nb=blockIdx.x*128,pc=cols/8;
  const int expert=expert_ids?expert_ids[slot]:slot;
  qweight+=int64_t(expert)*stride_q;qzeros+=int64_t(expert)*stride_z;
  scales+=int64_t(expert)*stride_s;x+=int64_t(slot)*x_stride;
  const int begin=split*rows_per_block,end=min(rows,begin+rows_per_block);
  for(int i=tid;i<16*136;i+=256)lhs[i]=__float2bfloat16(0);
  float total=0;
  for(int group=begin/128;group*128<end;++group) {
    if(tid<128) {
      int k=group*128+tid;
      lhs[tid]=(k>=begin&&k<end)?x[k]:__float2bfloat16(0);
    }
    if(tid<16)zeros[tid]=qzeros[int64_t(group)*pc+nb/8+tid];
    __syncthreads();
    for(int i=tid;i<32*16;i+=256) {
      const int k4=i/16,jlocal=i%16,j=nb/8+jlocal;
      const int k=group*128+k4*4;
      const int64_t address=((int64_t(j/32)*(rows/4)+k/4)*32+j%32)*4;
      const int4 packed=*reinterpret_cast<const int4*>(qweight+address);
      const int words[4]={packed.x,packed.y,packed.z,packed.w};
      const int z=zeros[jlocal];
#pragma unroll
      for(int u=0;u<4;++u) {
        __align__(16) __nv_bfloat16 values[8];
#pragma unroll
        for(int nibble=0;nibble<8;++nibble) {
          int integer=((words[u]>>(4*nibble))&15)-((z>>(4*nibble))&15);
          values[awq_nibble_column(nibble)]=__float2bfloat16(float(integer));
        }
        *reinterpret_cast<uint4*>(rhs+(k4*4+u)*136+jlocal*8)=
            *reinterpret_cast<const uint4*>(values);
      }
    }
    __syncthreads();
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;
    wmma::fill_fragment(acc,0.f);
#pragma unroll
    for(int k=0;k<128;k+=16) {
      wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> a;
      wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> b;
      wmma::load_matrix_sync(a,lhs+k,136);
      wmma::load_matrix_sync(b,rhs+k*136+warp*16,136);
      wmma::mma_sync(acc,a,b,acc);
    }
    wmma::store_matrix_sync(products+warp*16,acc,128,wmma::mem_row_major);
    __syncthreads();
    if(tid<128)total+=products[tid]*__half2float(scales[int64_t(group)*cols+nb+tid]);
    __syncthreads();
  }
  if(tid<128) {
    if(slot_scale)total*=slot_scale[slot];
    partial[(int64_t(slot)*gridDim.y+split)*cols+nb+tid]=total;
  }
#endif
}
