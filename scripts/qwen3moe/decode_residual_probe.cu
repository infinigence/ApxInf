#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include "../../crates/apxinf-cuda/kernels/custom/decode_epilogues.cuh"
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));exit(1);}}while(0)
template<class T>T* upload(const std::vector<T>& x){T*p;CHECK(cudaMalloc(&p,x.size()*sizeof(T)));CHECK(cudaMemcpy(p,x.data(),x.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
// Original residual RMSNorm composition, copied unchanged except the name.
__global__ void reference_residual_norm(
    __nv_bfloat16* x_inout, const __nv_bfloat16* delta,
    const __nv_bfloat16* weight, __nv_bfloat16* output,
    uint32_t cols, uint32_t rows, float eps)
{
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    uint32_t tid = threadIdx.x;
    uint32_t offset = row * cols;

    // Shared memory: x_new[cols] in fp32 + one slot for the reduced sum.
    extern __shared__ float smem[];
    float* x_new = smem;        // [cols]
    __shared__ float s_sum;

    // Phase 1: strided load of x+delta into shared memory; each thread
    // accumulates its partial sum_sq.
    float partial = 0.0f;
    for (uint32_t i = tid; i < cols; i += blockDim.x) {
        float xv = __bfloat162float(x_inout[offset + i]);
        float dv = __bfloat162float(delta[offset + i]);
        float xn = xv + dv;
        x_new[i] = xn;
        partial += xn * xn;
    }

    // Phase 2: warp-shuffle reduction within each warp, then a small
    // shared-memory reduction across warps. Assumes blockDim.x <= 1024
    // (max 32 warps).
    for (int off = 16; off > 0; off >>= 1)
        partial += __shfl_xor_sync(0xffffffff, partial, off);
    // Partial now holds the warp sum for lane 0 of each warp.
    __shared__ float warp_sums[32];
    uint32_t warp_id = tid / 32;
    uint32_t lane = tid % 32;
    if (lane == 0) warp_sums[warp_id] = partial;
    __syncthreads();
    if (warp_id == 0) {
        float v = (tid < (blockDim.x + 31) / 32) ? warp_sums[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_xor_sync(0xffffffff, v, off);
        if (lane == 0) s_sum = v;
    }
    __syncthreads();
    float rms = rsqrtf(s_sum / (float)cols + eps);

    // Phase 3: write x_new back to x_inout and the normed output.
    for (uint32_t i = tid; i < cols; i += blockDim.x) {
        float xn = x_new[i];
        float w  = __bfloat162float(weight[i]);
        x_inout[offset + i] = __float2bfloat16(xn);
        output[offset + i]  = __float2bfloat16(xn * rms * w);
    }
}
int main(){
 for(int cols:{128,2048,4096,8192})for(int count:{1,4,8,16}){
  uint32_t seed=52;auto random=[&](){seed=seed*1664525+1013904223;return float(int(seed>>16)-32768)/32768.f;};
  std::vector<float> parts(cols*count);std::vector<__nv_bfloat16>x(cols),weight(cols),delta(cols);
  for(auto&p:parts)p=random();for(auto&p:x)p=__float2bfloat16(random());for(auto&p:weight)p=__float2bfloat16(1+random());
  for(int i=0;i<cols;++i){float s=0;for(int c=0;c<count;++c)s+=parts[c*cols+i];delta[i]=__float2bfloat16(s);}
  auto dp=upload(parts);auto dx=upload(x),rx=upload(x),dw=upload(weight),dd=upload(delta);__nv_bfloat16* out,*ref;CHECK(cudaMalloc(&out,cols*2));CHECK(cudaMalloc(&ref,cols*2));
  reference_residual_norm<<<1,256,cols*4>>>(rx,dd,dw,ref,cols,1,1e-6f);
  partial_residual_rms_bf16_kernel<<<1,256,cols*4>>>(dx,dp,dw,out,cols,1,count,1e-6f);CHECK(cudaDeviceSynchronize());
  std::vector<__nv_bfloat16> got(cols),expected(cols),gx(cols),ex(cols);
  CHECK(cudaMemcpy(got.data(),out,cols*2,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(expected.data(),ref,cols*2,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(gx.data(),dx,cols*2,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(ex.data(),rx,cols*2,cudaMemcpyDeviceToHost));
  int bad=0;for(int i=0;i<cols;++i){bad+=memcmp(&got[i],&expected[i],2)!=0;bad+=memcmp(&gx[i],&ex[i],2)!=0;}
  printf("cols=%d count=%d bit_mismatches=%d\n",cols,count,bad);if(bad)return 2;
  for(void*p:{(void*)dp,(void*)dx,(void*)rx,(void*)dw,(void*)dd,(void*)out,(void*)ref})CHECK(cudaFree(p));
 }
 return 0;
}
