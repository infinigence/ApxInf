#pragma once

// Deterministic expert-major, slot-stable M32 routing metadata. No atomics,
// host copies, or allocation; all launch dimensions depend only on capacity.
__global__ void moe_count_slots_kernel(const int32_t* ids,int* counts,int slots) {
  int count=0;
  for(int i=threadIdx.x;i<slots;i+=blockDim.x)count+=ids[i]==int(blockIdx.x);
  for(int delta=16;delta>0;delta/=2)count+=__shfl_down_sync(0xffffffff,count,delta);
  __shared__ int sums[8];
  int lane=threadIdx.x%32,warp=threadIdx.x/32;
  if(lane==0)sums[warp]=count;
  __syncthreads();
  if(warp==0){count=lane<8?sums[lane]:0;
    for(int delta=16;delta>0;delta/=2)count+=__shfl_down_sync(0xffffffff,count,delta);
    if(lane==0)counts[blockIdx.x]=count;
  }
}

__global__ void moe_scan_tiles_kernel(const int* counts,int* offsets,
    int32_t* expert_ids,int* padded_count,int experts,int tile=32) {
  __shared__ int scan[128];
  int e=threadIdx.x;
  scan[e]=e<experts?((counts[e]+tile-1)/tile)*tile:0;
  __syncthreads();
  for(int delta=1;delta<128;delta*=2){
    int add=e>=delta?scan[e-delta]:0;__syncthreads();scan[e]+=add;__syncthreads();
  }
  if(e<experts){int begin=e?scan[e-1]:0;offsets[e]=begin;
    for(int i=begin/tile;i<scan[e]/tile;++i)expert_ids[i]=e;
    if(e==experts-1){offsets[experts]=scan[e];*padded_count=scan[e];}
  }
}

__global__ void moe_scatter_slots_kernel(const int32_t* ids,const int* counts,
    const int* offsets,int32_t* sorted,int slots) {
  __shared__ int warp_counts[4];
  int e=blockIdx.x,lane=threadIdx.x%32,warp=threadIdx.x/32,seen=0;
  for(int base=0;base<slots;base+=128){
    int slot=base+threadIdx.x;bool match=slot<slots&&ids[slot]==e;
    unsigned mask=__ballot_sync(0xffffffff,match);
    if(lane==0)warp_counts[warp]=__popc(mask);
    __syncthreads();
    int prefix=0,total=0;
#pragma unroll
    for(int w=0;w<4;++w){if(w<warp)prefix+=warp_counts[w];total+=warp_counts[w];}
    if(match)sorted[offsets[e]+seen+prefix+__popc(mask&((1u<<lane)-1))]=slot;
    seen+=total;__syncthreads();
  }
  for(int i=counts[e]+threadIdx.x;i<offsets[e+1]-offsets[e];i+=128)sorted[offsets[e]+i]=slots;
}
