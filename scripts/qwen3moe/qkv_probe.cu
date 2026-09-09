#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include "../../crates/apxinf-cuda/kernels/custom/qk_norm_rope.cuh"
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));exit(1);}}while(0)
template<class T>T* upload(const std::vector<T>& x){T*p;CHECK(cudaMalloc(&p,x.size()*sizeof(T)));CHECK(cudaMemcpy(p,x.data(),x.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
// Original composition kernels, copied unchanged except their names.
__global__ void reference_norm(
    const __nv_bfloat16* input, const __nv_bfloat16* weight, __nv_bfloat16* output,
    uint32_t cols, uint32_t rows, float eps)
{
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    uint32_t tid = threadIdx.x;
    uint32_t offset = row * cols;

    // Cache the row in fp32 shared memory so the normalize phase doesn't
    // re-read HBM. cols * sizeof(float) bytes (8 KB for cols=2048 — fits the
    // 48 KB per-block limit).
    extern __shared__ float x_buf[];
    __shared__ float s_sum;

    // Phase 1: strided load; each thread accumulates a partial sum_sq.
    float partial = 0.0f;
    for (uint32_t i = tid; i < cols; i += blockDim.x) {
        float v = __bfloat162float(input[offset + i]);
        x_buf[i] = v;
        partial += v * v;
    }

    // Phase 2: warp-shuffle reduction within each warp, then across warps.
    for (int off = 16; off > 0; off >>= 1)
        partial += __shfl_xor_sync(0xffffffff, partial, off);
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

    // Phase 3: write the normed output from shared memory.
    for (uint32_t i = tid; i < cols; i += blockDim.x) {
        float w = __bfloat162float(weight[i]);
        output[offset + i] = __float2bfloat16(x_buf[i] * rms * w);
    }
}
__global__ void reference_rope(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    uint32_t head_dim, uint32_t n_heads, uint32_t seq_len,
    float rope_theta, uint32_t pos_offset)
{
    uint32_t pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t head_idx = blockIdx.y;
    uint32_t seq_idx  = blockIdx.z;
    if (pair_idx >= head_dim / 2) return;

    uint32_t pos = seq_idx + pos_offset;
    float freq = 1.0f / powf(rope_theta, 2.0f * (float)pair_idx / (float)head_dim);
    float angle = (float)pos * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    uint32_t base = seq_idx * n_heads * head_dim + head_idx * head_dim;
    uint32_t half = head_dim / 2;
    uint32_t idx0 = base + pair_idx;
    uint32_t idx1 = base + half + pair_idx;

    float x0 = __bfloat162float(input[idx0]);
    float x1 = __bfloat162float(input[idx1]);
    output[idx0] = __float2bfloat16(x0 * cos_val - x1 * sin_val);
    output[idx1] = __float2bfloat16(x0 * sin_val + x1 * cos_val);
}
int main(){
  for(float theta:{1e6f,1e7f})for(int tokens:{1,17,128,1024})for(int offset:{0,7}){
    int capacity=tokens+offset+13;float eps=1e-6f;
    std::vector<__nv_bfloat16> q(tokens*32*128),k(tokens*4*128),v(k.size()),qw(128),kw(128);
    uint32_t seed=42;auto random=[&](){seed=seed*1664525+1013904223;return __float2bfloat16((int(seed>>16)-32768)/8192.f);};
    for(auto&x:q)x=random();for(auto&x:k)x=random();for(auto&x:v)x=random();for(auto&x:qw)x=random();for(auto&x:kw)x=random();
    auto dq=upload(q),dk=upload(k),dv=upload(v),dqw=upload(qw),dkw=upload(kw);
    __nv_bfloat16 *qn,*kn,*qr,*kr,*ck,*cv;
    CHECK(cudaMalloc(&qn,q.size()*2));CHECK(cudaMalloc(&kn,k.size()*2));CHECK(cudaMalloc(&qr,q.size()*2));CHECK(cudaMalloc(&kr,k.size()*2));
    CHECK(cudaMalloc(&ck,4*capacity*128*2));CHECK(cudaMalloc(&cv,4*capacity*128*2));CHECK(cudaMemset(ck,0,4*capacity*128*2));CHECK(cudaMemset(cv,0,4*capacity*128*2));
    half *oq,*ok,*ov;CHECK(cudaMalloc(&oq,q.size()*2));CHECK(cudaMalloc(&ok,k.size()*2));CHECK(cudaMalloc(&ov,k.size()*2));
    reference_norm<<<tokens*32,256,128*4>>>(dq,dqw,qn,128,tokens*32,eps);
    reference_norm<<<tokens*4,256,128*4>>>(dk,dkw,kn,128,tokens*4,eps);
    reference_rope<<<dim3(1,32,tokens),256>>>(qn,qr,128,32,tokens,theta,offset);
    reference_rope<<<dim3(1,4,tokens),256>>>(kn,kr,128,4,tokens,theta,offset);
    float* rope;CHECK(cudaMalloc(&rope,capacity*64*2*4));rope_table_f32_kernel<<<256,256>>>(rope,capacity,theta);
    auto launch=[&](){qk_norm_rope_append_f16_kernel<<<(tokens*36+3)/4,128>>>(dq,dk,dv,dqw,dkw,oq,ok,ov,ck,cv,tokens,32,4,capacity,offset,eps,rope);};
    launch();CHECK(cudaDeviceSynchronize());
    std::vector<__nv_bfloat16> rq(q.size()),rk(k.size()),cachek(4*capacity*128),cachev(cachek.size());
    std::vector<half> gq(q.size()),gk(k.size()),gv(k.size());
    CHECK(cudaMemcpy(rq.data(),qr,q.size()*2,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(rk.data(),kr,k.size()*2,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(gq.data(),oq,q.size()*2,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(gk.data(),ok,k.size()*2,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(gv.data(),ov,k.size()*2,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(cachek.data(),ck,cachek.size()*2,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(cachev.data(),cv,cachev.size()*2,cudaMemcpyDeviceToHost));
    int bad=0;
    for(size_t i=0;i<q.size();++i){half expected=__float2half(__bfloat162float(rq[i]));if(memcmp(&expected,&gq[i],2))++bad;}
    for(size_t i=0;i<k.size();++i){half expected=__float2half(__bfloat162float(rk[i])),ev=__float2half(__bfloat162float(v[i]));if(memcmp(&expected,&gk[i],2)||memcmp(&ev,&gv[i],2))++bad;}
    for(int h=0;h<4;++h)for(int t=0;t<capacity;++t)for(int d=0;d<128;++d){
      __nv_bfloat16 ek=__float2bfloat16(0),ev=ek;
      if(t>=offset&&t<offset+tokens){ek=rk[((t-offset)*4+h)*128+d];ev=v[((t-offset)*4+h)*128+d];}
      int i=(h*capacity+t)*128+d;if(memcmp(&ek,&cachek[i],2)||memcmp(&ev,&cachev[i],2))++bad;
    }
    cudaEvent_t a,b;CHECK(cudaEventCreate(&a));CHECK(cudaEventCreate(&b));CHECK(cudaEventRecord(a));for(int r=0;r<20;++r)launch();CHECK(cudaEventRecord(b));CHECK(cudaEventSynchronize(b));float ms;CHECK(cudaEventElapsedTime(&ms,a,b));
    printf("theta=%g tokens=%d offset=%d ms=%g bit_mismatches=%d\n",theta,tokens,offset,ms/20,bad);if(bad)return 2;
    for(void*p:{(void*)rope,(void*)dq,(void*)dk,(void*)dv,(void*)dqw,(void*)dkw,(void*)qn,(void*)kn,(void*)qr,(void*)kr,(void*)ck,(void*)cv,(void*)oq,(void*)ok,(void*)ov})CHECK(cudaFree(p));CHECK(cudaEventDestroy(a));CHECK(cudaEventDestroy(b));
  }
  return 0;
}
