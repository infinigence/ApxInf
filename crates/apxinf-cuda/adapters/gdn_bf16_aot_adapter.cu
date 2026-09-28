// SPDX-License-Identifier: Apache-2.0
// BF16 first-prefill recurrent attention, backed by the pinned FI64 AOT kernel.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <atomic>
#include <mutex>
#include <cstdint>
#include "../kernels/custom/gdn_fused_qkv.cuh"
#include "flashinfer_gdn_bf16_t64_sm110_real3392.h"
namespace {
constexpr int H=16, HV=32, D=128, KEY=H*D, VALUE=HV*D;
constexpr int CONV=2*KEY+VALUE;
constexpr int AOT_ERROR_BASE=0x10000;
int32_t status(cudaError_t error) { return static_cast<int32_t>(error); }
__global__ void write_offsets(int32_t* out,int32_t seq) {
  if(threadIdx.x==0){out[0]=0;out[1]=seq;}
}
__global__ void state_vmajor_to_kmajor(const float* in,float* out) {
  int block=blockIdx.x,row=blockIdx.y*32+threadIdx.y,
      col=blockIdx.z*32+threadIdx.x;
  __shared__ float tile[32][33];
  tile[threadIdx.y][threadIdx.x]=in[size_t(block)*D*D+size_t(row)*D+col];
  __syncthreads();
  row=blockIdx.z*32+threadIdx.y;
  col=blockIdx.y*32+threadIdx.x;
  out[size_t(block)*D*D+size_t(row)*D+col]=tile[threadIdx.x][threadIdx.y];
}
__global__ void pack_ab_only_kernel(const __nv_bfloat16* zba,
                                    __nv_bfloat16* a, __nv_bfloat16* b,
                                    int seq, int seq_pad) {
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=size_t(seq_pad)*32) return;
  int token=int(i/32),head=int(i%32);
  if(token<seq){
    a[i]=zba[size_t(token)*12352+8192+4096+32+head];
    b[i]=zba[size_t(token)*12352+8192+4096+head];
  }else{
    a[i]=__float2bfloat16(0.f);
    b[i]=__float2bfloat16(0.f);
  }
}
constexpr int FLASH_SEQ=3387, FLASH_PAD=3392;
constexpr size_t FLASH_GATE_BYTES=size_t(FLASH_PAD)*HV*sizeof(float);
constexpr size_t FLASH_WORKSPACE_BYTES=10240;
constexpr size_t FLASH_SCRATCH_BYTES=2*FLASH_GATE_BYTES+FLASH_WORKSPACE_BYTES;
constexpr size_t RAW_QK_BYTES=size_t(FLASH_SEQ)*2*KEY*sizeof(__nv_bfloat16);
static_assert(FLASH_GATE_BYTES%256==0 && FLASH_SCRATCH_BYTES<=RAW_QK_BYTES);
flashinfer_gdn_bf16_t64_sm110_real3392_Kernel_Module_t flash_module{};
std::mutex flash_init_mutex;
std::atomic<int> initialized_device{-1};
std::atomic<bool> flash_ready{false};

__global__ void flash_alpha_beta(const __nv_bfloat16* a,const __nv_bfloat16* b,
                                 const float* A_log,const __nv_bfloat16* dt_bias,
                                 float* alpha,float* beta){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=FLASH_PAD*HV) return;
  int token=i/HV,head=i%HV;
  if(token>=FLASH_SEQ){alpha[i]=1.f;beta[i]=0.f;return;}
  float x=__bfloat162float(a[i])+__bfloat162float(dt_bias[head]);
  float softplus=x>20.f ? x : logf(1.f+expf(x));
  float g_log=-expf(A_log[head])*softplus;
  alpha[i]=expf(g_log);
  beta[i]=1.f/(1.f+expf(-__bfloat162float(b[i])));
}
}

extern "C" int32_t apxinf_static_gdn_flashinfer64_init() {
  int device = -1;
  cudaError_t error = cudaGetDevice(&device);
  if (error != cudaSuccess) return status(error);
  if (flash_ready.load(std::memory_order_acquire))
    return device == initialized_device.load() ? 0 : status(cudaErrorInvalidDevice);
  std::lock_guard<std::mutex> lock(flash_init_mutex);
  if (flash_ready.load(std::memory_order_relaxed))
    return device == initialized_device.load() ? 0 : status(cudaErrorInvalidDevice);
  int major = 0, minor = 0, sms = 0, devices = 0;
  if ((error = cudaGetDeviceCount(&devices)) != cudaSuccess) return status(error);
  if (devices != 1) return status(cudaErrorInvalidDevice);
  if ((error = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device)) != cudaSuccess)
    return status(error);
  if ((error = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device)) != cudaSuccess)
    return status(error);
  if ((error = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device)) != cudaSuccess)
    return status(error);
  if (major != 11 || minor != 0 || sms != 20) return status(cudaErrorInvalidDeviceFunction);
  cudaLibrary_t* library = &flash_module.module;
  struct InitArgs { cudaLibrary_t** library; cudaError_t* error; } init{&library, &error};
  _mlir_flashinfer_gdn_bf16_t64_sm110_real3392_cuda_init(reinterpret_cast<void**>(&init));
  if (error != cudaSuccess) return status(error);
  struct LoadArgs { cudaLibrary_t** library; int32_t* device; cudaError_t* error; }
      load{&library, &device, &error};
  _mlir_flashinfer_gdn_bf16_t64_sm110_real3392_cuda_load_to_device(reinterpret_cast<void**>(&load));
  if (error != cudaSuccess) {
    if (flash_module.module) cudaLibraryUnload(flash_module.module);
    flash_module.module = nullptr;
    return status(error);
  }
  if (!flash_module.module) return status(cudaErrorInvalidResourceHandle);
  initialized_device.store(device);
  flash_ready.store(true, std::memory_order_release);
  return 0;
}

extern "C" int32_t apxinf_static_gdn_flashinfer64_prefill_compact_v(
    const void* zba,const void* conv_weight,void* new_conv_state,
    const float* A_log,const void* dt_bias_bf16,
    void* raw_qk_scratch,void* q,void* k,void* v,void* a,void* b,
    float* h_output_vmajor,float* final_state_kmajor,
    void* output_pad,int32_t* cu_seqlens,int32_t seq,int32_t seq_pad,
    cudaStream_t stream){
  if(!flash_ready.load(std::memory_order_acquire)) return status(cudaErrorNotReady);
  int device=-1;
  cudaError_t error=cudaGetDevice(&device);
  if(error!=cudaSuccess) return status(error);
  if(device!=initialized_device.load(std::memory_order_relaxed))
    return status(cudaErrorInvalidDevice);
  if(seq!=FLASH_SEQ || seq_pad!=FLASH_PAD ||
     !zba || !conv_weight || !new_conv_state || !A_log || !dt_bias_bf16 ||
     !raw_qk_scratch || !q || !k || !v || !a || !b ||
     !h_output_vmajor || !final_state_kmajor || !output_pad || !cu_seqlens)
    return status(cudaErrorInvalidValue);
  if((reinterpret_cast<uintptr_t>(raw_qk_scratch)&255)!=0 ||
     h_output_vmajor==final_state_kmajor)
    return status(cudaErrorInvalidValue);
  auto* scratch=static_cast<unsigned char*>(raw_qk_scratch);
  auto* alpha=reinterpret_cast<float*>(scratch);
  auto* beta=reinterpret_cast<float*>(scratch+FLASH_GATE_BYTES);
  void* descriptor_workspace=scratch+2*FLASH_GATE_BYTES;
  constexpr int CONV_TOKENS=32,CONV_KERNEL=4;
  constexpr size_t FUSED_SHARED=(CONV_TOKENS+CONV_KERNEL-1+CONV_TOKENS)*256*
                                sizeof(__nv_bfloat16);
  causal_conv_edge_fused_qkv_kernel<<<
      dim3((FLASH_SEQ+CONV_TOKENS-1)/CONV_TOKENS,CONV/256),256,
      FUSED_SHARED,stream>>>(
      static_cast<const __nv_bfloat16*>(zba),
      static_cast<const __nv_bfloat16*>(conv_weight),
      static_cast<__nv_bfloat16*>(q),static_cast<__nv_bfloat16*>(k),
      static_cast<__nv_bfloat16*>(v),
      static_cast<__nv_bfloat16*>(new_conv_state),seq,seq_pad);
  error=cudaGetLastError();if(error!=cudaSuccess) return status(error);
  write_offsets<<<1,1,0,stream>>>(cu_seqlens,seq);
  error=cudaGetLastError();if(error!=cudaSuccess) return status(error);
  pack_ab_only_kernel<<<(FLASH_PAD*HV+255)/256,256,0,stream>>>(
      static_cast<const __nv_bfloat16*>(zba),
      static_cast<__nv_bfloat16*>(a),static_cast<__nv_bfloat16*>(b),
      seq,seq_pad);
  error=cudaGetLastError();if(error!=cudaSuccess) return status(error);
  flash_alpha_beta<<<(FLASH_PAD*HV+255)/256,256,0,stream>>>(
      static_cast<const __nv_bfloat16*>(a),
      static_cast<const __nv_bfloat16*>(b),A_log,
      static_cast<const __nv_bfloat16*>(dt_bias_bf16),alpha,beta);
  error=cudaGetLastError();if(error!=cudaSuccess) return status(error);
  flashinfer_gdn_bf16_t64_sm110_real3392_Tensor_q_t tq{q};
  flashinfer_gdn_bf16_t64_sm110_real3392_Tensor_k_t tk{k};
  flashinfer_gdn_bf16_t64_sm110_real3392_Tensor_v_t tv{v};
  flashinfer_gdn_bf16_t64_sm110_real3392_Tensor_gate_t tg{alpha};
  flashinfer_gdn_bf16_t64_sm110_real3392_Tensor_beta_t tb{beta};
  flashinfer_gdn_bf16_t64_sm110_real3392_Tensor_o_t to{output_pad};
  flashinfer_gdn_bf16_t64_sm110_real3392_Tensor_cu_seqlens_t ts{cu_seqlens};
  flashinfer_gdn_bf16_t64_sm110_real3392_Tensor_s_out_t th{h_output_vmajor};
  flashinfer_gdn_bf16_t64_sm110_real3392_Tensor_tensormap_workspace_t tw{
      descriptor_workspace};
  int32_t aot_status=cute_dsl_flashinfer_gdn_bf16_t64_sm110_real3392_wrapper(
      &flash_module,&tq,&tk,&tv,&tg,&tb,&to,&ts,&th,0,
      0.08838834764831845f,&tw,stream);
  if(aot_status!=0) return AOT_ERROR_BASE | (aot_status & 0xffff);
  state_vmajor_to_kmajor<<<dim3(HV,4,4),dim3(32,32),0,stream>>>(
      h_output_vmajor,final_state_kmajor);
  return status(cudaGetLastError());
}
