#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include "../../crates/apxinf-cuda/kernels/custom/moe_permutation.cuh"
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));exit(1);}}while(0)
int main(){
  for(int slots:{8,127,128,129,1024,8192,32768,65536})for(int pattern=0;pattern<3;++pattern){
    const int experts=128,max_padded=slots+experts*32;
    std::vector<int32_t> ids(slots),expected,expected_experts;
    uint32_t seed=42;for(auto&x:ids){seed=seed*1664525+1013904223;x=pattern==0?0:pattern==1?(seed>>16)%experts:(seed>>16)%7;}
    for(int e=0;e<experts;++e){int count=0;for(int i=0;i<slots;++i)if(ids[i]==e){expected.push_back(i);++count;}
      for(int i=count;i%32;++i)expected.push_back(slots);
      for(int i=0;i<(count+31)/32;++i)expected_experts.push_back(e);
    }
    int32_t *input,*sorted,*expert_ids;int *counts,*offsets,*padded;
    CHECK(cudaMalloc(&input,slots*4));CHECK(cudaMemcpy(input,ids.data(),slots*4,cudaMemcpyHostToDevice));
    CHECK(cudaMalloc(&sorted,max_padded*4));CHECK(cudaMalloc(&expert_ids,((max_padded+31)/32)*4));
    CHECK(cudaMalloc(&counts,experts*4));CHECK(cudaMalloc(&offsets,(experts+1)*4));CHECK(cudaMalloc(&padded,4));
    auto launch=[&](){moe_count_slots_kernel<<<experts,256>>>(input,counts,slots);moe_scan_tiles_kernel<<<1,128>>>(counts,offsets,expert_ids,padded,experts);moe_scatter_slots_kernel<<<experts,128>>>(input,counts,offsets,sorted,slots);};
    launch();CHECK(cudaDeviceSynchronize());int n;CHECK(cudaMemcpy(&n,padded,4,cudaMemcpyDeviceToHost));
    if(n!=int(expected.size()))return 2;
    std::vector<int32_t> got(n),got_experts(n/32);CHECK(cudaMemcpy(got.data(),sorted,n*4,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(got_experts.data(),expert_ids,n/32*4,cudaMemcpyDeviceToHost));
    if(got!=expected||got_experts!=expected_experts)return 3;
    // Replay after poisoning output, including graph capture.
    cudaStream_t stream;CHECK(cudaStreamCreate(&stream));cudaGraph_t graph;cudaGraphExec_t exec;
    CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    moe_count_slots_kernel<<<experts,256,0,stream>>>(input,counts,slots);moe_scan_tiles_kernel<<<1,128,0,stream>>>(counts,offsets,expert_ids,padded,experts);moe_scatter_slots_kernel<<<experts,128,0,stream>>>(input,counts,offsets,sorted,slots);
    CHECK(cudaStreamEndCapture(stream,&graph));CHECK(cudaGraphInstantiate(&exec,graph,0));
    CHECK(cudaMemsetAsync(sorted,0xff,max_padded*4,stream));CHECK(cudaGraphLaunch(exec,stream));CHECK(cudaStreamSynchronize(stream));
    CHECK(cudaMemcpy(got.data(),sorted,n*4,cudaMemcpyDeviceToHost));if(got!=expected)return 4;
    cudaEvent_t a,b;CHECK(cudaEventCreate(&a));CHECK(cudaEventCreate(&b));CHECK(cudaEventRecord(a,stream));for(int r=0;r<20;++r)CHECK(cudaGraphLaunch(exec,stream));CHECK(cudaEventRecord(b,stream));CHECK(cudaEventSynchronize(b));float ms;CHECK(cudaEventElapsedTime(&ms,a,b));
    printf("slots=%d pattern=%d padded=%d graph_ms=%g stable=1\n",slots,pattern,n,ms/20);
    CHECK(cudaGraphExecDestroy(exec));CHECK(cudaGraphDestroy(graph));CHECK(cudaStreamDestroy(stream));CHECK(cudaEventDestroy(a));CHECK(cudaEventDestroy(b));
    for(void*p:{(void*)input,(void*)sorted,(void*)expert_ids,(void*)counts,(void*)offsets,(void*)padded})CHECK(cudaFree(p));
  }
}
