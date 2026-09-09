// CPU softmax + stable sorting oracle, including tied BF16 router logits.
// nvcc -O3 -arch=sm_101 scripts/qwen3moe/router_probe.cu -o router_probe
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>
#include "../../crates/apxinf-cuda/kernels/custom/reduction.cuh"
#include "../../crates/apxinf-cuda/kernels/custom/selection.cuh"
#define CHECK(x) do { auto e=(x);if(e!=cudaSuccess){ \
  fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));std::exit(1); \
} } while(0)
int main() {
  int cases=0;
  for(int tokens:{1,5,33,128,257})for(int experts:{8,32,128,256})
  for(int k:{1,8,32})for(int normalize:{0,1}) {
    if(k>experts)continue;
    std::vector<__nv_bfloat16> input(tokens*experts);
    uint32_t seed=84;auto next=[&](){return seed=seed*1664525+1013904223;};
    for(int row=0;row<tokens;++row)for(int e=0;e<experts;++e) {
      float value;
      // Equal rows, exact ties with a nontrivial denominator, and random rows.
      if(row%3==0)value=0;
      else if(row%3==1)value=float((e*13)%7-3)*0.375f;
      else value=float(int(next()%1281)-640)/16.f;
      input[row*experts+e]=__float2bfloat16(value);
    }
    __nv_bfloat16* logits;int32_t* ids;float* weights;
    CHECK(cudaMalloc(&logits,input.size()*2));CHECK(cudaMalloc(&ids,tokens*k*4));
    CHECK(cudaMalloc(&weights,tokens*k*4));
    CHECK(cudaMemcpy(logits,input.data(),input.size()*2,cudaMemcpyHostToDevice));
    moe_router_topk_bf16_kernel<<<(tokens+3)/4,128>>>(logits,ids,weights,tokens,experts,k,normalize);
    CHECK(cudaDeviceSynchronize());
    std::vector<int32_t> got_ids(tokens*k);std::vector<float> got_weights(tokens*k);
    CHECK(cudaMemcpy(got_ids.data(),ids,tokens*k*4,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(got_weights.data(),weights,tokens*k*4,cudaMemcpyDeviceToHost));
    float max_error=0;
    for(int row=0;row<tokens;++row) {
      std::vector<float> probabilities(experts);std::vector<int> order(experts);
      std::iota(order.begin(),order.end(),0);float maximum=-INFINITY,total=0;
      for(int e=0;e<experts;++e)maximum=std::max(maximum,__bfloat162float(input[row*experts+e]));
      for(int e=0;e<experts;++e){probabilities[e]=std::exp(__bfloat162float(input[row*experts+e])-maximum);total+=probabilities[e];}
      for(float& p:probabilities)p/=total;
      std::stable_sort(order.begin(),order.end(),[&](int a,int b){return probabilities[a]>probabilities[b];});
      float selected=0;for(int i=0;i<k;++i)selected+=probabilities[order[i]];
      for(int i=0;i<k;++i) {
        float expected=probabilities[order[i]]/(normalize?selected:1.f);
        float got=got_weights[row*k+i];max_error=std::max(max_error,std::abs(got-expected));
        if(got_ids[row*k+i]!=order[i] || !std::isfinite(got) || std::abs(got-expected)>2e-6f) {
          fprintf(stderr,"router mismatch row=%d E=%d k=%d normalize=%d rank=%d id=%d expected=%d weight=%.9g expected=%.9g\n",
              row,experts,k,normalize,i,got_ids[row*k+i],order[i],got,expected);return 2;
        }
      }
    }
    printf("tokens=%d experts=%d k=%d normalize=%d max_weight_error=%g\n",tokens,experts,k,normalize,max_error);
    CHECK(cudaFree(logits));CHECK(cudaFree(ids));CHECK(cudaFree(weights));++cases;
  }
  printf("passed=%d\n",cases);return 0;
}
