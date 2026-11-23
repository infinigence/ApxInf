// MP22 ASM BF16 NN/NT GEMM. See kernels/asm/README.md for provenance.
#include <musa.h>
#include <musa_runtime.h>
#include <cstdint>
#include <climits>
#include <map>
#include <utility>
#include "adapter_common.h"
#include "mubin_images.h"
namespace {
using namespace apxinf_adapters;
struct Image { const unsigned char* data; unsigned long size; const char* symbol; int m,n,threads; bool m_fast = false; };
// `images[0]` (NT 128_128x128) and `images[6]` (NN 128_128x128) are never
// selected: `select_nt_cfg` returns 1..=5 and `apxinf_musa_asm_encode_variant`
// returns 12..=15, so no path addresses the two base tiles. Their bytes still
// ship in `mubin_images.h`; dropping them would renumber every variant and has
// to wait for a regeneration of that header.
const Image images[] = {
    {kGemm0, kGemm0Size, "asm_bf16bf16gemm_nt_tc_128_128x128_epilogue", 128, 128, 128},
    {kGemm1, kGemm1Size, "asm_bf16bf16gemm_nt_tc_128_128x64_epilogue", 128, 64, 128},
    {kGemm2, kGemm2Size, "asm_bf16bf16gemm_nt_tc_128_32x128_epilogue", 32, 128, 128},
    {kGemm3, kGemm3Size, "asm_bf16bf16gemm_nt_tc_256_256x128_epilogue", 256, 128, 256},
    {kGemm4, kGemm4Size, "asm_bf16bf16gemm_nt_tc_256_256x64_epilogue", 256, 64, 256},
    {kGemm5, kGemm5Size, "asm_bf16bf16gemm_nt_tc_512_256x256_epilogue", 256, 256, 512},
    {kGemm6, kGemm6Size, "asm_bf16bf16gemm_nn_tc_128_128x128_epilogue", 128, 128, 128},
    {kGemm7, kGemm7Size, "asm_bf16bf16gemm_nn_tc_128_128x64_epilogue", 128, 64, 128},
    {kGemm8, kGemm8Size, "asm_bf16bf16gemm_nn_tc_128_32x128_epilogue", 32, 128, 128},
    {kGemm9, kGemm9Size, "asm_bf16bf16gemm_nn_tc_256_256x128_epilogue", 256, 128, 256},
    {kGemm10, kGemm10Size, "asm_bf16bf16gemm_nn_tc_256_256x64_epilogue", 256, 64, 256},
    {kGemm11, kGemm11Size, "asm_bf16bf16gemm_nn_tc_512_256x256_epilogue", 256, 256, 512},
    {kGemm12, kGemm12Size, "asm_bf16bf16gemm_nn_tc_128_128x128_epilogue_wide_mfast", 128, 128, 128, true},
    {kGemm13, kGemm13Size, "asm_bf16bf16gemm_nn_tc_256_128x256_epilogue_wide_mfast", 128, 256, 256, true},
    {kGemm14, kGemm14Size, "asm_bf16bf16gemm_nn_tc_128_128x128_epilogue_normal_mfast", 128, 128, 128, true},
    {kGemm15, kGemm15Size, "asm_bf16bf16gemm_nn_tc_256_256x128_epilogue_wide_nfast", 256, 128, 256, false},


};
// Modules deliberately remain loaded for captured graph lifetime. Cache by
// context as well as thread, so switching devices cannot reuse a foreign module.
thread_local std::map<std::pair<MUcontext,int>, MUfunction> functions;
MUfunction load(int variant, void* stream) {
    MUcontext ctx = nullptr;
    if (muCtxGetCurrent(&ctx) != MUSA_SUCCESS || !ctx) {
        set_error("ASM GEMM: no current MUSA context"); return nullptr;
    }
    const auto key = std::make_pair(ctx,variant);
    auto found = functions.find(key);
    if (found != functions.end()) return found->second;
    if (capturing(stream)) {
        set_error("ASM GEMM must be prepared before graph capture"); return nullptr;
    }
    const auto& image = images[variant];
    if (!image.size) { set_error("ASM BF16 GEMM image missing (requires MP22 assembler)"); return nullptr; }
    MUmodule module = nullptr;
    MUresult status = muModuleLoadData(&module, image.data);
    if (status != MUSA_SUCCESS) { set_error("ASM GEMM module load failed: %d", int(status)); return nullptr; }
    MUfunction fn = nullptr;
    status = muModuleGetFunction(&fn, module, image.symbol);
    if (status != MUSA_SUCCESS) {
        muModuleUnload(module); set_error("ASM GEMM symbol lookup failed: %d", int(status)); return nullptr;
    }
    functions.emplace(key,fn);
    return fn;
}
constexpr int kNumMps = 8;
int ceil_div_int(int a, int b) { return (a+b-1)/b; }
int select_nt_cfg(int batch, int m, int n, int k) {
  const int block_512_256x256 = ceil_div_int(m, 256) * ceil_div_int(n, 256) * batch;
  const int block_256_256x128 = ceil_div_int(m, 256) * ceil_div_int(n, 128) * batch;
  const int block_128_128x128 = ceil_div_int(m, 128) * ceil_div_int(n, 128) * batch;
  const int block_128_128x64  = ceil_div_int(m, 128) * ceil_div_int(n, 64) * batch;
  const int block_256_256x64  = ceil_div_int(m, 256) * ceil_div_int(n, 64) * batch;

  const int dispatch_round_512_256x256 = ceil_div_int(block_512_256x256, kNumMps);
  const int dispatch_round_256_256x128 = ceil_div_int(block_256_256x128, kNumMps);
  const int dispatch_round_128_128x128 = ceil_div_int(block_128_128x128, kNumMps);
  const int dispatch_round_128_128x64  = ceil_div_int(block_128_128x64, kNumMps);
  const int dispatch_round_256_256x64  = ceil_div_int(block_256_256x64, kNumMps);

  const bool is_balance_256_256x128 = block_256_256x128 % kNumMps == 0;
  const bool is_prefered_128_128x64 =
      ((dispatch_round_256_256x128 * 4 > dispatch_round_128_128x64) && (dispatch_round_256_256x128 < 2) && m >= 256) ||
      ((dispatch_round_256_256x64 * 2 > dispatch_round_128_128x64) && (dispatch_round_256_256x64 < 2)) ||
      ((n < 384) && (n % 64 == 0) && (n % 128 != 0) && (dispatch_round_512_256x256 < 4)) || m <= 128;

  const bool is_prefered_512_256x256 =
      ((((dispatch_round_512_256x256 * 2) <= dispatch_round_256_256x128) && (dispatch_round_512_256x256 >= 2)) ||
       (dispatch_round_512_256x256 >= 4)) &&
      k > 256;

  const bool is_prefered_256_256x128 = ((dispatch_round_256_256x128 * 2 <= dispatch_round_256_256x64) &&
                                        !(dispatch_round_256_256x128 == 1 && !is_balance_256_256x128)) ||
                                       ((dispatch_round_256_256x128 * 2 <= dispatch_round_128_128x128) &&
                                        !(dispatch_round_256_256x128 == 1 && !is_balance_256_256x128)) ||
                                       dispatch_round_256_256x128 >= 4;

  const bool is_prefered_256_256x64 =
      is_prefered_256_256x128 && (k <= 1024 || ((dispatch_round_256_256x64 * 2 <= dispatch_round_256_256x128) &&
                                                (dispatch_round_256_256x64 <= 4)));

  if (n < 128) {
    return 1;
  }
  if (m <= 32) {
    return 2;
  }
  if (is_prefered_128_128x64) {
    return 1;
  }
  if (is_prefered_512_256x256) {
    return 5;
  }
  if (is_prefered_256_256x64) {
    return 4;
  }
  if (is_prefered_256_256x128) {
    return 3;
  }
  return 4;
}


__global__ void transpose_weight(const uint16_t* in, uint16_t* out, int k, int n) {
    const size_t i = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i < size_t(n)*k) out[i] = in[(i%k)*n+i/k];
}
}
// Only measured encode shapes opt in. Everything else retains the old selector.
// This query is also used by Rust before looking up registered NN weights.
extern "C" int apxinf_musa_asm_encode_variant(int m, int n, int k) {
    if (m != 522) return -1;
    if (k == 2048) {
        if (n == 2560) return 12;
        if (n == 2048) return 13;
        if (n == 16384 || n == 32768) return 14;
    }
    if (k == 16384 && n == 2048) return 15;
    return -1;
}
extern "C" const char* apxinf_musa_asm_gemm_last_error() { return error_buffer(); }
extern "C" int apxinf_musa_asm_transpose(const void* in, void* out, int k, int n, void* stream) {
    if (!in || !out || k<=0 || n<=0 || int64_t(k)*n*2>INT_MAX || capturing(stream)) {
        set_error("ASM weight transpose: invalid shape or called during capture"); return -1;
    }
    if (!select_device(stream_device(stream))) {
        set_error("ASM weight transpose: cannot select stream device"); return -1;
    }
    transpose_weight<<<(size_t(n)*k+255)/256,256,0,static_cast<musaStream_t>(stream)>>>(
        static_cast<const uint16_t*>(in),static_cast<uint16_t*>(out),k,n);
    auto status=musaGetLastError();
    if (status!=musaSuccess) { set_error("ASM weight transpose: %s",musaGetErrorString(status)); return -1; }
    return 0;
}
extern "C" int apxinf_musa_asm_gemm(const void* a, const void* b, void* out,
                                    int m, int n, int k, int nt, void* stream) {
    if (!a || !b || !out || m<=0 || n<=0 || k<=0 || k%32 ||
        int64_t(m)*k*2>INT_MAX || int64_t(n)*k*2>INT_MAX || int64_t(m)*n*2>INT_MAX) {
        set_error("ASM GEMM: invalid shape or buffer exceeds int32 ABI"); return -1;
    }
    if (nt != 0 && nt != 1) { set_error("ASM GEMM: invalid layout"); return -1; }
    if (!select_device(stream_device(stream))) { set_error("ASM GEMM: cannot select stream device"); return -1; }
    const int encode_variant = nt ? -1 : apxinf_musa_asm_encode_variant(m,n,k);
    const int variant = encode_variant >= 0 ? encode_variant : select_nt_cfg(1,m,n,k)+(nt ? 0 : 6);
    auto fn=load(variant,stream);
    if (!fn) return -1;
    int out_bytes=m*n*2, a_bytes=m*k*2, b_bytes=n*k*2, zero=0, batch=1;
    int sa=m*k,sb=n*k,sc=m*n,ldb=nt ? k : n;
    float alpha=1.f,beta=0.f,gamma=0.f;
    void* nil=nullptr;
    // Full muDNN epilogue ABI: seven (pointer,int32 bytes,int32 zero)
    // triples followed by dimensions, leading dimensions, scales and strides.
    void* args[]={&out,&out_bytes,&zero,&a,&a_bytes,&zero,&b,&b_bytes,&zero,
        &nil,&zero,&zero,&nil,&zero,&zero,&nil,&zero,&zero,&nil,&zero,&zero,
        &m,&n,&k,&k,&ldb,&n,&alpha,&beta,&gamma,&batch,&sa,&sb,&sc};
    const auto& cfg=images[variant];
    const int grid_m = (m+cfg.m-1)/cfg.m, grid_n = (n+cfg.n-1)/cfg.n;
    auto status=muLaunchKernel(fn,cfg.m_fast ? grid_m : grid_n,cfg.m_fast ? grid_n : grid_m,1,
        cfg.threads,1,1,0,reinterpret_cast<MUstream>(stream),args,nullptr);
    if (status!=MUSA_SUCCESS) { set_error("ASM GEMM launch failed: %d",int(status)); return -1; }
    return 0;
}
