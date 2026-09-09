#define main residual_probe_main
#include "decode_residual_probe.cu"
#undef main
__global__ void reference_silu(
    const __nv_bfloat16* __restrict__ gate_up, __nv_bfloat16* __restrict__ output,
    int rows, int inter) {
  const int64_t total = static_cast<int64_t>(rows) * inter;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total; index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int r = static_cast<int>(index / inter);
    const int i = static_cast<int>(index - static_cast<int64_t>(r) * inter);
    const __nv_bfloat16* row = gate_up + static_cast<int64_t>(r) * 2 * inter;
    const float g = __bfloat162float(row[i]);
    const float u = __bfloat162float(row[inter + i]);
    output[index] = __float2bfloat16(g / (1.0f + expf(-g)) * u);
  }
}
int main(){
 for(int rows:{1,17,128,1024}){
  int inter=768;std::vector<half>gu(rows*inter*2);std::vector<__nv_bfloat16>bg(gu.size());uint32_t seed=51;
  for(size_t i=0;i<gu.size();++i){seed=seed*1664525+1013904223;gu[i]=__float2half(float(int(seed>>16)-32768)/8192.f);bg[i]=__float2bfloat16(__half2float(gu[i]));}
  auto g=upload(gu);auto b=upload(bg);half*out;__nv_bfloat16*ref;CHECK(cudaMalloc(&out,rows*inter*2));CHECK(cudaMalloc(&ref,rows*inter*2));
  silu_mul_rows_f16_rounded_kernel<<<256,256>>>(g,out,rows,inter);reference_silu<<<256,256>>>(b,ref,rows,inter);CHECK(cudaDeviceSynchronize());
  std::vector<half>got(rows*inter);std::vector<__nv_bfloat16>expected(got.size());CHECK(cudaMemcpy(got.data(),out,got.size()*2,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(expected.data(),ref,got.size()*2,cudaMemcpyDeviceToHost));
  int bad=0;for(size_t i=0;i<got.size();++i){half value=__float2half(__bfloat162float(expected[i]));bad+=memcmp(&got[i],&value,2)!=0;}
  printf("rows=%d bit_mismatches=%d\n",rows,bad);if(bad)return 2;
  for(void*p:{(void*)g,(void*)b,(void*)out,(void*)ref})CHECK(cudaFree(p));
 }
 return 0;
}
