// Copyright 2026 ApxInf contributors. SPDX-License-Identifier: Apache-2.0
#define FLASH_NAMESPACE apxinf_cuda_new_fa2_indexed
#include "indexed_sm110.h"
#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>
#include <algorithm>
#include <cmath>
#include <memory>
#include "flash.h"
#include "apxinf_indexed_fa2.hpp"
using B=cutlass::bfloat16_t;
constexpr float kLog2E=1.4426950408889634074f;
static void fill_mapped_params(FLASH_NAMESPACE::Flash_fwd_params& params, bool is_bf16,
                 const void* q, const void* k, const void* v, void* output,
                 void* softmax_lse, int batch, int query_tokens,
                 int key_tokens, int query_heads, int kv_heads, int head_dim,
                 float softmax_scale) {
  params = {};
  params.is_bf16 = is_bf16;
  params.q_ptr = const_cast<void*>(q);
  params.k_ptr = const_cast<void*>(k);
  params.v_ptr = const_cast<void*>(v);
  params.o_ptr = output;
  params.softmax_lse_ptr = softmax_lse;

  const int64_t q_row_stride = static_cast<int64_t>(query_heads) * head_dim;
  const int64_t kv_row_stride = static_cast<int64_t>(kv_heads) * head_dim;
  params.q_batch_stride = static_cast<int64_t>(query_tokens) * q_row_stride;
  params.k_batch_stride = static_cast<int64_t>(key_tokens) * kv_row_stride;
  params.v_batch_stride = params.k_batch_stride;
  params.o_batch_stride = params.q_batch_stride;
  params.q_row_stride = q_row_stride;
  params.k_row_stride = kv_row_stride;
  params.v_row_stride = kv_row_stride;
  params.o_row_stride = q_row_stride;
  params.q_head_stride = head_dim;
  params.k_head_stride = head_dim;
  params.v_head_stride = head_dim;
  params.o_head_stride = head_dim;

  params.b = batch;
  params.h = query_heads;
  params.h_k = kv_heads;
  params.h_h_k_ratio = query_heads / kv_heads;
  params.seqlen_q = query_tokens;
  params.seqlen_k = key_tokens;
  params.seqlen_q_rounded = ((query_tokens + 127) / 128) * 128;
  params.seqlen_k_rounded = ((key_tokens + 127) / 128) * 128;
  params.d = head_dim;
  params.d_rounded = (head_dim + 31) & ~31;

  params.scale_softmax = softmax_scale;
  params.scale_softmax_log2 = softmax_scale * kLog2E;
  params.scale_softmax_rp_dropout = softmax_scale;
  params.p_dropout = 1.0f;
  params.p_dropout_in_uint8_t = 255;
  params.rp_dropout = 1.0f;

  params.is_causal = false;
  params.window_size_left = -1;
  params.window_size_right = -1;
  params.is_seqlens_k_cumulative = true;
  params.num_splits = 1;
}

namespace FLASH_NAMESPACE {
struct IndexedParams:Flash_fwd_params {const int* row_map=nullptr;int original_queries=0;int original_heads=0;};
}
using Traits=Flash_fwd_kernel_traits<128,128,64,4,false,false,B>;
__global__ void indexed_fa2_kernel(__grid_constant__ const FLASH_NAMESPACE::IndexedParams p){
 FLASH_NAMESPACE::compute_attn<Traits,false,false,false,false,false,true,false,false>(p);
}
namespace apxinf::cuda_new::indexed_attention {
namespace { struct Plan { FLASH_NAMESPACE::IndexedParams params; }; }
int prepare_mapped(const Bindings& b,void** result){
 if(result==nullptr)return -1;*result=nullptr;
 try {
  int device=0,maximum=0;
  if(cudaGetDevice(&device)!=cudaSuccess || cudaDeviceGetAttribute(&maximum,cudaDevAttrMaxSharedMemoryPerBlockOptin,device)!=cudaSuccess)return -2;
  if(Traits::kSmemSize>maximum)return -1;
  if(cudaFuncSetAttribute(indexed_fa2_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,Traits::kSmemSize)!=cudaSuccess)return -2;
  auto plan=std::make_unique<Plan>();auto& p=plan->params;
  fill_mapped_params(p,true,b.query,b.key,b.value,b.output,b.fa2_lse,b.heads,b.queries,b.keys,1,1,128,b.scale);
  p.cu_seqlens_q=b.cumulative;p.total_q=b.queries*b.heads;p.unpadded_lse=false;
  p.k_row_stride=p.v_row_stride=int64_t(b.heads)*128;p.k_batch_stride=p.v_batch_stride=128;
  p.row_map=b.indices;p.original_queries=b.queries;p.original_heads=b.heads;
  *result=plan.release();return 0;
 }catch(...){return -2;}
}
cudaError_t enqueue_mapped(void* raw,cudaStream_t stream){
 const auto& p=static_cast<Plan*>(raw)->params;
 indexed_fa2_kernel<<<dim3((p.seqlen_q+127)/128,p.b,1),Traits::kNThreads,Traits::kSmemSize,stream>>>(p);return cudaGetLastError();
}
void destroy_mapped(void* p)noexcept{delete static_cast<Plan*>(p);}
} // namespace apxinf::cuda_new::indexed_attention
