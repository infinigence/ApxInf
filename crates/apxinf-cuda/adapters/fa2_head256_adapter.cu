// BF16 head256 specialization for causal prefill and noncausal split-KV.
#define UNFUSE_FMA
#include "flash_attn/namespace_config.h"
#include "flash_attn/flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {
using Head256Traits = Flash_fwd_kernel_traits<256,64,64,4,false,false,cutlass::bfloat16_t>;

template<bool Split>
cudaError_t launch_head256_split(Flash_fwd_params& params,cudaStream_t stream) {
  auto kernel=&flash_fwd_splitkv_kernel<Head256Traits,false,false,false,false,true,false,Split,false>;
  cudaStreamCaptureStatus capture;auto status=cudaStreamIsCapturing(stream,&capture);
  if(status!=cudaSuccess)return status;
  if(capture==cudaStreamCaptureStatusNone) {
    status=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,Head256Traits::kSmemSize);
    if(status!=cudaSuccess)return status;
  }
  kernel<<<dim3((params.seqlen_q+63)/64,Split?params.num_splits:params.b,
      Split?params.b*params.h:params.h),Head256Traits::kNThreads,Head256Traits::kSmemSize,stream>>>(params);
  return cudaGetLastError();
}

int run_bf16_head256_splitkv(Flash_fwd_params& params,cudaStream_t stream) {
  if(params.d!=256||params.is_causal||params.num_splits<1||params.num_splits>128)
    return static_cast<int>(cudaErrorInvalidValue);
  if(params.num_splits==1)return static_cast<int>(launch_head256_split<false>(params,stream));
  auto status=launch_head256_split<true>(params,stream);if(status!=cudaSuccess)return static_cast<int>(status);
  dim3 grid((params.b*params.h*params.seqlen_q+3)/4);
#define COMBINE256(LOG) flash_fwd_splitkv_combine_kernel<Head256Traits,4,LOG,true><<<grid,Head256Traits::kNThreads,0,stream>>>(params)
  if(params.num_splits<=2){COMBINE256(1);}else if(params.num_splits<=4){COMBINE256(2);}
  else if(params.num_splits<=8){COMBINE256(3);}else if(params.num_splits<=16){COMBINE256(4);}
  else if(params.num_splits<=32){COMBINE256(5);}else if(params.num_splits<=64){COMBINE256(6);}
  else {COMBINE256(7);}
#undef COMBINE256
  return static_cast<int>(cudaGetLastError());
}

int run_bf16_head256_causal(Flash_fwd_params& params, cudaStream_t stream) {
  if (params.d != 256 || !params.is_causal || params.seqlen_q != params.seqlen_k)
    return static_cast<int>(cudaErrorInvalidValue);
  using Traits = Flash_fwd_kernel_traits<256,64,64,4,false,false,cutlass::bfloat16_t>;
  auto kernel = &flash_fwd_kernel<Traits,false,true,false,false,false,true,false,false>;
  cudaStreamCaptureStatus capture;
  auto status=cudaStreamIsCapturing(stream,&capture);
  if(status!=cudaSuccess)return static_cast<int>(status);
  if(capture==cudaStreamCaptureStatusNone) {
    status=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,Traits::kSmemSize);
    if(status!=cudaSuccess)return static_cast<int>(status);
  }
  kernel<<<dim3((params.seqlen_q+63)/64,params.b,params.h),Traits::kNThreads,Traits::kSmemSize,stream>>>(params);
  return static_cast<int>(cudaGetLastError());
}
}
