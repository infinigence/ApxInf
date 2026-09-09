#pragma once
// Keep the original sum order and BF16 boundary before residual RMSNorm.
template<bool wide_loads=false, bool pdl=false, typename NormWeight=__nv_bfloat16>
__global__ void partial_residual_rms_bf16_kernel(
    __nv_bfloat16* x_inout, const float* partials,
    const NormWeight* weight, __nv_bfloat16* output,
    uint32_t cols, uint32_t rows, uint32_t count, float eps)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if constexpr(pdl) {
        cudaGridDependencySynchronize();
        cudaTriggerProgrammaticLaunchCompletion();
    }
#elif defined(__CUDA_ARCH__)
    static_assert(!pdl, "programmatic dependent launch requires SM90 or newer");
#endif
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    uint32_t tid = threadIdx.x;
    uint32_t offset = row * cols;
    partials += int64_t(row) * count * cols;

    // Shared memory: x_new[cols] in fp32 + one slot for the reduced sum.
    extern __shared__ float smem[];
    float* x_new = smem;        // [cols]
    __shared__ float s_sum;

    // Phase 1: strided load of x+delta into shared memory; each thread
    // accumulates its partial sum_sq.
    float partial = 0.0f;
    for (uint32_t i = tid; i < cols; i += blockDim.x) {
        float xv = __bfloat162float(x_inout[offset + i]);
        float sum = 0.f;
        for(uint32_t c=0;c<count;++c)sum+=partials[int64_t(c)*cols+i];
        float dv = __bfloat162float(__float2bfloat16(sum));
        float xn = xv + dv;
        x_new[i] = xn;
        if constexpr(!wide_loads) partial += xn * xn;
    }

    if constexpr(wide_loads) {
        __syncthreads();
        // Preserve the original 256-thread sum-of-squares tree even though
        // more threads cooperatively load partials and store outputs.
        if(tid<256)for(uint32_t i=tid;i<cols;i+=256){float xn=x_new[i];partial+=xn*xn;}
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
        float v = (tid < (wide_loads ? 8 : (blockDim.x + 31) / 32)) ? warp_sums[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_xor_sync(0xffffffff, v, off);
        if (lane == 0) s_sum = v;
    }
    __syncthreads();
    float rms = rsqrtf(s_sum / (float)cols + eps);

    // Phase 3: write x_new back to x_inout and the normed output.
    for (uint32_t i = tid; i < cols; i += blockDim.x) {
        float xn = x_new[i];
        float w  = static_cast<float>(weight[i]);
        x_inout[offset + i] = __float2bfloat16(xn);
        output[offset + i]  = __float2bfloat16(xn * rms * w);
    }
}

// Prefill combine + residual + RMSNorm, retaining the intermediate BF16 rounding.
template<bool input_f16=false, typename NormWeight=__nv_bfloat16>
__global__ void routed_residual_rms_bf16_kernel(
    __nv_bfloat16* x_inout, const __nv_bfloat16* y,const int32_t* slot_rows,const float* router_weights,
    const NormWeight* weight, __nv_bfloat16* output,
    uint32_t cols, uint32_t rows, uint32_t topk, float eps)
{
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    uint32_t tid = threadIdx.x;
    uint64_t offset = uint64_t(row) * cols;

    // Shared memory: x_new[cols] in fp32 + one slot for the reduced sum.
    extern __shared__ float smem[];
    float* x_new = smem;        // [cols]
    __shared__ float s_sum;

    // Phase 1: strided load of x+delta into shared memory; each thread
    // accumulates its partial sum_sq.
    float partial = 0.0f;
    for (uint32_t i = tid; i < cols; i += blockDim.x) {
        float xv = __bfloat162float(x_inout[offset + i]);
        float sum = 0.f;
        for(uint32_t route=0;route<topk;++route) {
            int64_t slot=int64_t(row)*topk+route;int64_t source=slot_rows[slot];
            float value=0.f;
            if(source>=0 && source<int64_t(rows)*topk) {
                if constexpr(input_f16) value=__bfloat162float(__float2bfloat16(__half2float(reinterpret_cast<const half*>(y)[source*cols+i])));
                else value=__bfloat162float(y[source*cols+i]);
            }
            sum+=router_weights[slot]*value;
        }
        float dv = __bfloat162float(__float2bfloat16(sum));
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
        float w  = static_cast<float>(weight[i]);
        x_inout[offset + i] = __float2bfloat16(xn);
        output[offset + i]  = __float2bfloat16(xn * rms * w);
    }
}

// FP16 expert computation, preserving BF16 boundaries around SwiGLU.
__global__ void silu_mul_rows_f16_rounded_kernel(const half* gu,half* output,int rows,int inter) {
  for(int64_t idx=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;idx<int64_t(rows)*inter;idx+=int64_t(gridDim.x)*blockDim.x) {
    int64_t row=idx/inter,col=idx%inter;
    float g=__bfloat162float(__float2bfloat16(__half2float(gu[row*2*inter+col])));
    float u=__bfloat162float(__float2bfloat16(__half2float(gu[row*2*inter+inter+col])));
    output[idx]=__float2half(__bfloat162float(__float2bfloat16((g/(1.f+expf(-g)))*u)));
  }
}

// BF16 has only 65536 bit patterns. Precomputing the FP32 SiLU value removes
// repeated exp/div work while retaining the original FP32 rounding boundary.
__global__ void silu_bf16_table_kernel(float* table) {
  int bits=blockIdx.x*blockDim.x+threadIdx.x;
  if(bits<65536){float g=__bfloat162float(__ushort_as_bfloat16(uint16_t(bits)));table[bits]=g/(1.f+expf(-g));}
}
__global__ void silu_mul_rows_f16_lut_kernel(const half* gu,half* output,
    const float* table,int rows,int inter) {
  for(int64_t idx=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;idx<int64_t(rows)*inter;idx+=int64_t(gridDim.x)*blockDim.x) {
    int64_t row=idx/inter,col=idx%inter;
    __nv_bfloat16 g=__float2bfloat16(__half2float(gu[row*2*inter+col]));
    float u=__bfloat162float(__float2bfloat16(__half2float(gu[row*2*inter+inter+col])));
    output[idx]=__float2half(__bfloat162float(__float2bfloat16(table[__bfloat16_as_ushort(g)]*u)));
  }
}

// Vectorized inter=768 specialization. Preserves every BF16 rounding boundary;
// 16-byte loads/stores and constant 32-bit indexing halve the measured full-row
// kernel time. The adapter checks alignment and bounds the signed loop counter.
__global__ void silu_mul_rows_f16_lut_768_kernel(const half* gu, half* out, const float* table, int rows) {
    union EightHalves { int4 packed; uint16_t values[8]; };
    for(int index=blockIdx.x*blockDim.x+threadIdx.x;index<rows*96;index+=blockDim.x*gridDim.x) {
        int row=index/96,col=index%96;
        EightHalves gate,up,result;
        gate.packed=reinterpret_cast<const int4*>(gu)[int64_t(row)*192+col];
        up.packed=reinterpret_cast<const int4*>(gu)[int64_t(row)*192+col+96];
#pragma unroll
        for(int i=0;i<8;++i) {
            auto g=__float2bfloat16(__half2float(__ushort_as_half(gate.values[i])));
            float u=__bfloat162float(__float2bfloat16(__half2float(__ushort_as_half(up.values[i]))));
            result.values[i]=__half_as_ushort(__float2half(__bfloat162float(__float2bfloat16(table[__bfloat16_as_ushort(g)]*u))));
        }
        reinterpret_cast<int4*>(out)[index]=result.packed;
    }
}

// Vectorized routed reduction: align each lane to eight adjacent columns,
// then preserve the original 256-thread RMS reduction through shared memory.
// Dispatch is limited to aligned 2048/4096-column rows validated by the probe.
template<bool input_f16=false, typename NormWeight=__nv_bfloat16>
__global__ void routed_residual_rms_vector_kernel(
    __nv_bfloat16* x_inout, const __nv_bfloat16* y,const int32_t* slot_rows,const float* router_weights,
    const NormWeight* weight, __nv_bfloat16* output,
    uint32_t cols, uint32_t rows, uint32_t topk, float eps)
{
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    uint32_t tid = threadIdx.x;
    uint64_t offset = uint64_t(row) * cols;

    // Shared memory: x_new[cols] in fp32 + one slot for the reduced sum.
    extern __shared__ float smem[];
    float* x_new = smem;        // [cols]
    __shared__ float s_sum;

    // Phase 1: strided load of x+delta into shared memory; each thread
    // accumulates its partial sum_sq.
    union EightValues { int4 packed; uint16_t words[8]; };
    for(uint32_t base=tid*8;base<cols;base+=blockDim.x*8) {
        float values[8]={0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
        for(uint32_t route=0;route<topk;++route) {
            int64_t slot=int64_t(row)*topk+route,source=slot_rows[slot];
            EightValues packed{};
            bool valid=source>=0 && source<int64_t(rows)*topk;
            if(valid)packed.packed=reinterpret_cast<const int4*>(y)[(source*cols+base)/8];
            float scale=router_weights[slot];
#pragma unroll
            for(int i=0;i<8;++i) {
                float value=0.f;
                if(valid) {
                    if constexpr(input_f16) value=__bfloat162float(__float2bfloat16(__half2float(__ushort_as_half(packed.words[i]))));
                    else value=__bfloat162float(__ushort_as_bfloat16(packed.words[i]));
                }
                values[i]+=scale*value;
            }
        }
        EightValues xv;xv.packed=reinterpret_cast<const int4*>(x_inout)[(offset+base)/8];
#pragma unroll
        for(int i=0;i<8;++i) {
            float dv=__bfloat162float(__float2bfloat16(values[i]));
            x_new[base+i]=__bfloat162float(__ushort_as_bfloat16(xv.words[i]))+dv;
        }
    }
    __syncthreads();
    // Reuse the original thread-to-column assignment for the RMS reduction.
    float partial=0.f;
    for(uint32_t i=tid;i<cols;i+=blockDim.x){float xn=x_new[i];partial+=xn*xn;}

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
        float w  = static_cast<float>(weight[i]);
        x_inout[offset + i] = __float2bfloat16(xn);
        output[offset + i]  = __float2bfloat16(xn * rms * w);
    }
}
