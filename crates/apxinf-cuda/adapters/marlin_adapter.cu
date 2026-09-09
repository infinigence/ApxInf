// Narrow, model-neutral AWQ group-128 adapter for the vendored Marlin kernel.
#define MARLIN_NAMESPACE_NAME apxinf_marlin
#include "../kernels/marlin/csrc/moe/marlin_moe_wna16/kernel.h"
#include "../kernels/marlin/csrc/moe/marlin_moe_wna16/marlin_template.h"
#include "../kernels/marlin/csrc/quantization/gptq_marlin/awq_marlin_repack.cu"

namespace {
constexpr auto kernel64_constants_f16 = apxinf_marlin::Marlin<half, vllm::kU4.id(),
    vllm::kFloat16.id(), 128, 4, 8, 4, false, 4, 8, false, false, true>;
constexpr auto kernel32_n128_f16 = apxinf_marlin::Marlin<half, vllm::kU4.id(),
    vllm::kFloat16.id(), 64, 2, 8, 4, false, 4, 8, false>;
constexpr auto kernel32_n256_f16 = apxinf_marlin::Marlin<half, vllm::kU4.id(),
    vllm::kFloat16.id(), 128, 2, 16, 4, false, 4, 8, false>;
constexpr auto kernel64_silu = apxinf_marlin::Marlin<half, vllm::kU4.id(),
    vllm::kFloat16.id(), 128, 4, 8, 4, false, 4, 8, false, true>;
constexpr auto kernel_silu = apxinf_marlin::Marlin<half, vllm::kU4.id(),
    vllm::kFloat16.id(), 256, 2, 16, 4, false, 4, 8, false, true>;
constexpr auto kernel64_f16 = apxinf_marlin::Marlin<half, vllm::kU4.id(),
    vllm::kFloat16.id(), 128, 4, 8, 4, false, 4, 8, false>;
constexpr auto kernel_f16 = apxinf_marlin::Marlin<half, vllm::kU4.id(),
    vllm::kFloat16.id(), 256, 2, 16, 4, false, 4, 8, false>;
constexpr auto kernel64 = apxinf_marlin::Marlin<nv_bfloat16, vllm::kU4.id(),
    vllm::kFloat16.id(), 128, 4, 8, 4, false, 4, 8, false>;
constexpr auto kernel = apxinf_marlin::Marlin<nv_bfloat16, vllm::kU4.id(),
    vllm::kFloat16.id(), 256, 2, 16, 4, false, 4, 8, false>;

// Place G64/U64 pairs in each 128-column tile, preserving AWQ nibble order.
__device__ int64_t paired_column(int64_t column,int n,bool paired) {
  if(!paired)return column;
  int c=column%n;
  return (column/n)*n+(c/128)*64+c%64+(c%128>=64?n/2:0);
}
__global__ void pair_qweight(const uint32_t* source,uint32_t* output,int64_t words,int n) {
  int packed=n/8;
  for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<words;i+=int64_t(gridDim.x)*blockDim.x) {
    int j=i%packed;
    output[i]=source[(i/packed)*packed+(j/16)*8+j%8+(j%16>=8?n/16:0)];
  }
}
__global__ void permute_scales_zeros(const half* source_s, const uint32_t* source_z,
    half* dest_s, uint32_t* dest_z, int64_t words,int n,bool paired) {
  for(int64_t word=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;word<words;word+=int64_t(gridDim.x)*blockDim.x) {
    uint32_t packed=0;
#pragma unroll
    for(int nib=0;nib<8;++nib) {
      int64_t i=word*8+nib;
      int64_t si=(i/64)*64+(i%64)/8+8*(i%8);
      dest_s[i]=source_s[paired_column(si,n,paired)];
      int64_t j=word*8+((nib&3)<<1)+(nib>>2);
      int64_t zi=(j/64)*64+(j%64)/8+8*(j%8);
      zi=paired_column(zi,n,paired);
      int shift=((zi%8)/2+(zi%2)*4)*4;
      packed|=((source_z[zi/8]>>shift)&15)<<(nib*4);
    }
    dest_z[word]=packed;
  }
}
}

extern "C" cudaError_t apxinf_marlin_prepare() {
  auto status = cudaFuncSetAttribute(kernel64_constants_f16,cudaFuncAttributeMaxDynamicSharedMemorySize,65536);
  if(status!=cudaSuccess)return status;
  status = cudaFuncSetAttribute(kernel32_n128_f16,cudaFuncAttributeMaxDynamicSharedMemorySize,36864);
  if(status!=cudaSuccess)return status;
  status = cudaFuncSetAttribute(kernel32_n256_f16,cudaFuncAttributeMaxDynamicSharedMemorySize,53248);
  if(status!=cudaSuccess)return status;
  status = cudaFuncSetAttribute(kernel64_silu,cudaFuncAttributeMaxDynamicSharedMemorySize,65536);
  if(status!=cudaSuccess)return status;
  status = cudaFuncSetAttribute(kernel_silu,cudaFuncAttributeMaxDynamicSharedMemorySize,65536);
  if(status!=cudaSuccess)return status;
  status = cudaFuncSetAttribute(kernel64_f16,cudaFuncAttributeMaxDynamicSharedMemorySize,65536);
  if(status!=cudaSuccess)return status;
  status = cudaFuncSetAttribute(kernel_f16,cudaFuncAttributeMaxDynamicSharedMemorySize,65536);
  if(status!=cudaSuccess)return status;
  status = cudaFuncSetAttribute(kernel64,cudaFuncAttributeMaxDynamicSharedMemorySize,65536);
  if(status!=cudaSuccess)return status;
  return cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,65536);
}

extern "C" cudaError_t apxinf_marlin_repack(const void* q,const void* z,const void* s,
    void* out_q,void* out_z,void* out_s,int k,int n,int experts,int fused_silu,void* scratch,cudaStream_t stream) {
  if(k<=128 || k%128 || n<=0 || n%256 || experts<=0 || (fused_silu && !scratch))return cudaErrorInvalidValue;
  int64_t stride=int64_t(k)*n/8;
  for(int e=0;e<experts;++e) {
    const uint32_t* source=static_cast<const uint32_t*>(q)+e*stride;
    if(fused_silu) {
      pair_qweight<<<256,256,0,stream>>>(source,static_cast<uint32_t*>(scratch),stride,n);
      source=static_cast<const uint32_t*>(scratch);
    }
    apxinf_marlin::awq_marlin_repack_kernel<256,4><<<14,256,8192,stream>>>(
        source,static_cast<uint32_t*>(out_q)+e*stride,k,n);
  }
  int64_t words=int64_t(experts)*(k/128)*(n/8);
  permute_scales_zeros<<<256,256,0,stream>>>(static_cast<const half*>(s),
      static_cast<const uint32_t*>(z),static_cast<half*>(out_s),
      static_cast<uint32_t*>(out_z),words,n,fused_silu!=0);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_marlin_grouped(const void* input,const void* q,
    const void* z,const void* s,const void* ids,const void* experts,const void* padded,
    void* out,void* temporary,void* locks,int m,int n,int k,int blocks,int top_k,int tile_m,int use_f16,int fused_silu,int variant,cudaStream_t stream) {
  if(variant<0 || variant>3 || (variant && (!use_f16 || fused_silu)) ||
      ((variant==1 || variant==2) && tile_m!=32) || (variant==3 && tile_m!=64))
    return cudaErrorInvalidValue;
  if((fused_silu && !use_f16) || (use_f16!=0 && use_f16!=1) || (tile_m!=32 && tile_m!=64) || top_k<=0 || m<=0 || int64_t(m)*top_k>INT32_MAX || n<=0 || n%256 || k<=128 || k%128 || blocks<=0)return cudaErrorInvalidValue;
  auto selected = use_f16 ? (fused_silu ? (tile_m==64?kernel64_silu:kernel_silu)
      : (tile_m==64?kernel64_f16:kernel_f16)) : (tile_m==64?kernel64:kernel);
  int threads=tile_m==64?128:256, shared=65536;
  if(variant==1) { selected=kernel32_n128_f16;threads=64;shared=36864; }
  if(variant==2) { selected=kernel32_n256_f16;threads=128;shared=53248; }
  if(variant==3) { selected=kernel64_constants_f16; }
  selected<<<blocks,threads,shared,stream>>>(static_cast<const int4*>(input),
      static_cast<const int4*>(q),static_cast<int4*>(out),static_cast<int4*>(temporary),
      nullptr,static_cast<const int4*>(s),nullptr,static_cast<const int4*>(z),nullptr,
      static_cast<const int32_t*>(ids),static_cast<const int32_t*>(experts),
      static_cast<const int32_t*>(padded),nullptr,top_k,false,false,k/128,m,n,k,
      static_cast<int*>(locks),false,false,true,shared);
  return cudaGetLastError();
}
