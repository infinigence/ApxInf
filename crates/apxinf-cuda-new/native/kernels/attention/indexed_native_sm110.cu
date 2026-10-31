// Copyright 2026 ApxInf contributors. SPDX-License-Identifier: Apache-2.0
// Fixed BF16/F32 reference-biased native path; see INDEXED_VENDOR.md.
#define APXINF_FUSE_ROWSUM 0
#define APXINF_KV_DESCENDING 1
#define APXINF_INDEXED_FA2_ROWSUM 1
#define APXINF_INDEXED_UNFUSED_SCALE 1
#define APXINF_INDEXED_KEY_TILE 64
#define APXINF_S_PIPE_STAGES 2
#define APXINF_1Q_NUM_REGS_SOFTMAX 128
#define APXINF_1Q_NUM_REGS_CORRECTION 80
#define APXINF_1Q_NUM_REGS_OTHER 24
#include "indexed_sm110.h"
#include <memory>
// Copyright 2025 ApxInf contributors.
// SPDX-License-Identifier: Apache-2.0
//
// Rust/FFI adaptation of an upstream SM100/SM110 CUTLASS FMHA kernel. Its pinned
// FMHA and CUTLASS headers live alongside this wrapper under kernels/cutlass.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cute/tensor.hpp"
#include "cutlass/util/packed_stride.hpp"

// The upstream example guard recognizes SM100 and SM103. CUTLASS enables the
// same kernel path for SM101 and SM110, so alias those architecture markers
// only while parsing the kernel header.
#if (defined(CUTLASS_ARCH_MMA_SM101A_ENABLED) || \
     defined(CUTLASS_ARCH_MMA_SM110A_ENABLED)) && \
    !defined(CUTLASS_ARCH_MMA_SM100A_ENABLED)
#define APXINF_FMHA_UNDEF_SM100A 1
#define CUTLASS_ARCH_MMA_SM100A_ENABLED 1
#endif
#if (defined(CUTLASS_ARCH_MMA_SM101F_ENABLED) || \
     defined(CUTLASS_ARCH_MMA_SM110F_ENABLED)) && \
    !defined(CUTLASS_ARCH_MMA_SM100F_ENABLED)
#define APXINF_FMHA_UNDEF_SM100F 1
#define CUTLASS_ARCH_MMA_SM100F_ENABLED 1
#endif
#include "kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp"
#include "apxinf_indexed_singleq_kernel.hpp"
#if defined(APXINF_FMHA_UNDEF_SM100A)
#undef CUTLASS_ARCH_MMA_SM100A_ENABLED
#undef APXINF_FMHA_UNDEF_SM100A
#endif
#if defined(APXINF_FMHA_UNDEF_SM100F)
#undef CUTLASS_ARCH_MMA_SM100F_ENABLED
#undef APXINF_FMHA_UNDEF_SM100F
#endif
#include "apxinf_indexed_mainloop.hpp"
#include "apxinf_indexed_singleq_mainloop.hpp"
#include "device/fmha.hpp"
#include "collective/sm100_fmha_fwd_epilogue_tma_warpspecialized.hpp"
#include "collective/sm100_fmha_load_tma_warpspecialized.hpp"
#include "collective/fmha_fusion.hpp"

using namespace cute;
#ifndef APXINF_INDEXED_KEY_TILE
#define APXINF_INDEXED_KEY_TILE 128
#endif
using TileShape = Shape<_256, Int<APXINF_INDEXED_KEY_TILE>, _128>;

using StrideQ = cute::tuple<int, _1, cute::tuple<cute::tuple<int, int>, int>>;
using StrideK = cute::tuple<int, _1, cute::tuple<cute::tuple<_0, int>, int>>;
using StrideV = StrideK;
using StrideO = StrideQ;
using StrideLSE = cute::tuple<_1, cute::tuple<cute::tuple<int, int>, int>>;
using ProblemShape = cute::tuple<int, int, int, cute::tuple<cute::tuple<int, int>, int>>;


namespace apxinf::cuda_new::indexed_attention {
namespace {
using Element=cutlass::bfloat16_t;
using BaselineMainloop=cutlass::fmha::indexed_collective::Sm100FmhaFwdMainloopTmaWarpspecialized<
 Element,float,float,TileShape,StrideQ,StrideK,StrideV,cutlass::fmha::collective::ResidualMask>;
using BaselineEpilogue=cutlass::fmha::collective::Sm100FmhaFwdEpilogueTmaWarpspecialized<
 Element,float,typename BaselineMainloop::TileShapePV,StrideO,StrideLSE>;
using BaselineKernel=cutlass::fmha::kernel::Sm100FmhaFwdKernelTmaWarpspecialized<
 ProblemShape,BaselineMainloop,BaselineEpilogue,cutlass::fmha::kernel::IndividualTileScheduler>;

// Both CTA ranks of a cluster must execute the same logical query tile,
// including the final partial tile. The grid is a whole number of clusters.
struct Cluster2TileScheduler {
 struct Params {dim3 grid;};
 bool valid_=true;
 CUTLASS_DEVICE explicit Cluster2TileScheduler(const Params&) {}
 template<class Shape,class Cluster,class Tile>
 static Params to_underlying_arguments(const Shape& shape,cutlass::KernelHardwareInfo,
                                       const Cluster& cluster,const Tile& tile){
  const int tiles=int(cute::ceil_div(cute::size<0>(shape),cute::size<0>(tile)));
  return {dim3(unsigned(tiles*int(cute::size<0>(cluster))),
               unsigned(cute::size<3,0>(shape)),unsigned(cute::size<3,1>(shape)))};
 }
 static dim3 get_grid_shape(const Params& p){return p.grid;}
 CUTLASS_DEVICE bool is_valid(){return valid_;}
 CUTLASS_DEVICE auto get_block_coord(){return cute::make_coord(blockIdx.x/2,cute::_0{},cute::make_coord(blockIdx.y,blockIdx.z));}
 CUTLASS_DEVICE Cluster2TileScheduler& operator++(){valid_=false;return *this;}
};
using SingleQMainloop=cutlass::fmha::indexed_singleq_collective::Sm100FmhaFwdMainloopTmaWarpspecialized<
 Element,float,float,TileShape,StrideQ,StrideK,StrideV,cutlass::fmha::collective::ResidualMask,
 cute::Shape<cute::_1,cute::_1,cute::_1>,cute::Shape<cute::_2,cute::_1,cute::_1>>;
using SingleQEpilogue=cutlass::fmha::collective::Sm100FmhaFwdEpilogueTmaWarpspecialized<
 Element,float,typename SingleQMainloop::TileShapePV,StrideO,StrideLSE,cute::false_type,1>;
using SingleQKernel=cutlass::fmha::indexed_singleq_kernel::Sm100FmhaFwdKernelTmaWarpspecialized<
 ProblemShape,SingleQMainloop,SingleQEpilogue,Cluster2TileScheduler,
 cutlass::fmha::indexed_singleq_kernel::Sm100FmhaCtxKernelWarpspecialized1QSchedule>;

struct Plan {
 virtual ~Plan()=default;
 virtual cudaError_t enqueue(cudaStream_t)=0;
};
template<class Kernel>struct Prepared final:Plan {
 using Device=cutlass::fmha::device::FMHA<Kernel>;
 Device operation;
 cudaLaunchConfig_t launch{};
 cudaLaunchAttribute cluster_attribute{};
 cudaError_t enqueue(cudaStream_t stream)override{
  auto config=launch;config.stream=stream;
  void* arguments[]={const_cast<typename Device::Params*>(&operation.params())};
  return cudaLaunchKernelExC(&config,(const void*)cutlass::device_kernel<Kernel>,arguments);
 }
};
template<class Kernel>int prepare_plan(const Bindings& b,void** result){
 using Device=typename Prepared<Kernel>::Device;
 const int dim=128,token_stride=b.heads*dim,rounded=(b.queries+127)/128*128;
 auto problem=make_tuple(b.queries,b.keys,dim,make_tuple(make_tuple(1,b.heads),1));
 StrideQ qs=make_stride(token_stride,_1{},make_stride(make_stride(dim,dim),token_stride*b.queries));
 StrideK ks=make_stride(token_stride,_1{},make_stride(make_stride(_0{},dim),token_stride*b.keys));
 StrideLSE ls=make_stride(_1{},make_stride(make_stride(rounded,rounded),rounded*b.heads));
 int device=0,sm=0,limit=0;
 if(cudaGetDevice(&device)!=cudaSuccess || cudaDeviceGetAttribute(&sm,cudaDevAttrMultiProcessorCount,device)!=cudaSuccess || cudaDeviceGetAttribute(&limit,cudaDevAttrMaxSharedMemoryPerBlockOptin,device)!=cudaSuccess)return -2;
 if(Kernel::SharedStorageSize>limit)return -1;
 typename Device::Arguments args{problem,
   {{static_cast<const Element*>(b.query),qs,static_cast<const Element*>(b.key),ks,static_cast<const Element*>(b.value),ks},b.scale,1.f,1.f,1.f,1.f},
   {static_cast<Element*>(b.output),qs,b.native_lse,ls},{device,sm}};
 args.mainloop.row_flags=b.flags;args.mainloop.flag_queries=b.queries;args.mainloop.flag_heads=b.heads;args.mainloop.flag_margin=16;
 if(Device::can_implement(args)!=cutlass::Status::kSuccess || Device::get_workspace_size(args)!=0)return -1;
 if(cudaFuncSetAttribute(cutlass::device_kernel<Kernel>,cudaFuncAttributeMaxDynamicSharedMemorySize,Kernel::SharedStorageSize)!=cudaSuccess)return -2;
 if(cudaFuncSetAttribute(cutlass::device_kernel<Kernel>,cudaFuncAttributeNonPortableClusterSizeAllowed,1)!=cudaSuccess)return -2;
 auto plan=std::make_unique<Prepared<Kernel>>();
 if(Kernel::initialize_workspace(args,nullptr,nullptr)!=cutlass::Status::kSuccess || plan->operation.update(args,nullptr)!=cutlass::Status::kSuccess)return -2;
 const dim3 cluster(cute::size<0>(typename Kernel::ClusterShape{}),
                    cute::size<1>(typename Kernel::ClusterShape{}),
                    cute::size<2>(typename Kernel::ClusterShape{}));
 plan->launch.gridDim=Device::get_grid_shape(plan->operation.params());
 plan->launch.blockDim=Kernel::get_block_shape();
 plan->launch.dynamicSmemBytes=Kernel::SharedStorageSize;
 if(cutlass::ClusterLauncher::check_cluster_dims(plan->launch.gridDim,cluster)!=cutlass::Status::kSuccess)return -1;
 plan->cluster_attribute.id=cudaLaunchAttributeClusterDimension;
 plan->cluster_attribute.val.clusterDim={cluster.x,cluster.y,cluster.z};
 plan->launch.numAttrs=1;plan->launch.attrs=&plan->cluster_attribute;
 *result=static_cast<Plan*>(plan.release());return 0;
}
} // namespace
int prepare_native(const Bindings& b,void** result,int configuration){
 if(result==nullptr)return -1;*result=nullptr;
 try {
  switch(configuration){
   case 0:return prepare_plan<BaselineKernel>(b,result);
   case 1:return prepare_plan<SingleQKernel>(b,result);
   default:return -1;
  }
 }catch(...){return -2;}
}
cudaError_t enqueue_native(void* p,cudaStream_t stream){return static_cast<Plan*>(p)->enqueue(stream);}
void destroy_native(void* p)noexcept{delete static_cast<Plan*>(p);}
// Stable per-head compaction: indices are in increasing original query order.
// Integer ballots/prefixes avoid data-dependent atomic append order entirely.
__global__ void compact_rows(const uint32_t* flags,int* counts,int* indices,int queries){
 const int head=blockIdx.x,lane=threadIdx.x%32,warp=threadIdx.x/32;
 __shared__ int warp_offsets[4];__shared__ int base;__shared__ int total;
 if(threadIdx.x==0)base=0;__syncthreads();
 for(int start=0;start<queries;start+=128){
  int query=start+threadIdx.x;bool selected=query<queries&&flags[head*queries+query]!=0;
  unsigned mask=__ballot_sync(0xffffffff,selected);
  if(lane==0)warp_offsets[warp]=__popc(mask);__syncthreads();
  if(threadIdx.x==0){int offset=0;for(int k=0;k<4;k++){int count=warp_offsets[k];warp_offsets[k]=offset;offset+=count;}total=offset;}
  __syncthreads();
  int rank=__popc(mask&((1u<<lane)-1u));
  if(selected)indices[head*queries+base+warp_offsets[warp]+rank]=query;
  __syncthreads();
  if(threadIdx.x==0)base+=total;
  __syncthreads();
 }
 if(threadIdx.x==0)counts[head]=base;
}
__global__ void prefix_counts(const int* counts,int* cumulative,int heads){
 if(threadIdx.x==0){int sum=0;cumulative[0]=0;for(int h=0;h<heads;h++){sum+=counts[h];cumulative[h+1]=sum;}}
}
cudaError_t compact(const Bindings& b,cudaStream_t stream){
 compact_rows<<<b.heads,128,0,stream>>>(b.flags,b.counts,b.indices,b.queries);
 auto status=cudaGetLastError();if(status!=cudaSuccess)return status;
 prefix_counts<<<1,1,0,stream>>>(b.counts,b.cumulative,b.heads);return cudaGetLastError();
}
} // namespace apxinf::cuda_new::indexed_attention
