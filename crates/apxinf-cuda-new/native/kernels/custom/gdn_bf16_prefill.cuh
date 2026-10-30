#pragma once
// Synchronous BF16 causal convolution, SiLU and normalized Q/K preparation.
// FP32 arithmetic, with explicit approximate intrinsics from pinned FlashInfer
// math.cuh (19f1a41e6b21f0c422d775e377b6fdf9a1fc9d23).
// Keep the multiplication by x LAST. For finite very negative x, the
// exponential may be +inf, its reciprocal becomes +0, and finite x * +0
// is -0 (finite). Never form x*exp(-x), exp/(1+exp), or inf*0 intermediates.
// -log2e*x can itself overflow to +inf at the negative BF16 extreme; that
// follows the same defined path. Very positive x yields exp=0 and x*1=x.
// FTZ may drop extremely small negative tails; the exhaustive BF16 gate
// records these differences. This is NOT bitwise-preserving arithmetic.
__device__ __forceinline__ float gdn_edge_fast_silu(float x) {
  constexpr float log2e = 1.44269504088896340736f;
  float exponential, reciprocal;
  const float exponent = -log2e * x;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(exponential) : "f"(exponent));
  const float denominator = 1.0f + exponential;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(reciprocal) : "f"(denominator));
  return x * reciprocal;
}

// First-prefill conv with Q/K normalization into the compact Edge layout.
__global__ void causal_conv_edge_fused_qkv_kernel(
    const __nv_bfloat16* x, const __nv_bfloat16* weight,
    __nv_bfloat16* q_packed, __nv_bfloat16* k_packed,
    __nv_bfloat16* v_packed,
    __nv_bfloat16* new_state, int seq, int seq_pad) {
  constexpr int TOKENS=32, CHANNELS=8192, KERNEL=4, X_STRIDE=12352;
  constexpr int QK=4096, V=4096;
  extern __shared__ __nv_bfloat16 conv_smem[];
  __nv_bfloat16* post = conv_smem + (TOKENS + KERNEL - 1) * 256;
  const int lane=static_cast<int>(threadIdx.x);
  const int channel=blockIdx.y*blockDim.x+lane;
  const int width=static_cast<int>(blockDim.x);
  const int token_base=blockIdx.x*TOKENS;
  if(token_base>=seq) return;
  const int tokens=min(TOKENS,seq-token_base);
  const int active=channel<CHANNELS;

  for(int r=0;r<tokens+KERNEL-1;++r){
    const int src=token_base-(KERNEL-1)+r;
    float value=0.0f;
    if(active && src>=0)
      value=__bfloat162float(x[static_cast<int64_t>(src)*X_STRIDE+channel]);
    conv_smem[r*width+lane]=__float2bfloat16(value);
  }
  __syncthreads();
  if(!active) return;

  float taps[8];
  for(int i=0;i<KERNEL;++i)
    taps[i]=__bfloat162float(weight[channel*KERNEL+i]);
  for(int t=0;t<tokens;++t){
    float acc=0.0f;
    for(int i=0;i<KERNEL;++i)
      acc+=__bfloat162float(conv_smem[(t+i)*width+lane])*taps[i];
    const float conv=__bfloat162float(__float2bfloat16(acc));
    const int token=token_base+t;
    const __nv_bfloat16 result=__float2bfloat16(
        gdn_edge_fast_silu(conv));
    if(channel<QK)
      post[t*width+lane]=result;
    else
      v_packed[static_cast<int64_t>(token)*V+channel-QK]=result;
  }

  // CTA-uniform branch. Post-SiLU values have already been rounded to BF16.
  // A second barrier makes the full 32x256 tile visible before remapping
  // warps to heads. No barrier is placed in the serial convolution loop.
  if (channel < QK) {
    __syncthreads();
    const int warp = lane / 32, warp_lane = lane % 32;
    for (int row = warp; row < tokens * 2; row += 8) {
      const int t = row / 2, local_head = row % 2;
      __nv_bfloat16 values[4];
      float sum = 0.0f;
      for (int j = 0; j < 4; ++j) {
        values[j] = post[t * width + local_head * 128 + warp_lane + j * 32];
        const float value = __bfloat162float(values[j]);
        sum += value * value;
      }
      for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, off);
      const float inv = rsqrtf(sum + 1e-6f);
      const int head_col = blockIdx.y * 256 + local_head * 128;
      __nv_bfloat16* output = head_col < 2048 ? q_packed : k_packed;
      const int out_col = head_col % 2048;
      for (int j = 0; j < 4; ++j)
        output[static_cast<int64_t>(token_base + t) * 2048 +
               out_col + warp_lane + j * 32] =
            __float2bfloat16(__bfloat162float(values[j]) * inv);
    }
  }

  if(token_base+tokens==seq){
    if (channel < QK) {
      __nv_bfloat16* output = channel < 2048 ? q_packed : k_packed;
      for (int token = seq; token < seq_pad; ++token)
        output[static_cast<int64_t>(token) * 2048 + channel % 2048] =
            __float2bfloat16(0.0f);
    }
    if(channel>=QK){
      for(int token=seq;token<seq_pad;++token)
        v_packed[static_cast<int64_t>(token)*V+channel-QK]=
            __float2bfloat16(0.0f);
    }
    for(int i=0;i<KERNEL;++i){
      float value=0.0f;
      if(seq+i>=KERNEL)
        value=__bfloat162float(
            x[static_cast<int64_t>(seq-KERNEL+i)*X_STRIDE+channel]);
      new_state[channel*KERNEL+i]=__float2bfloat16(value);
    }
  }
}
