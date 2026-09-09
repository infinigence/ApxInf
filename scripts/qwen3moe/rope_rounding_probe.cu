// Regression for the real-model cancellation at layer 38, pair 8, position 5.
#define main prefill_probe_main
#include "qkv_probe.cu"
#undef main
__global__ void explicit_rope(const __nv_bfloat16* input,__nv_bfloat16* output,const float* table){
 int pair=threadIdx.x;if(pair>=64)return;
 auto result=qk_rope_pair(__bfloat162float(input[pair]),__bfloat162float(input[pair+64]),table[(5*64+pair)*2],table[(5*64+pair)*2+1]);
 output[pair]=result.x;output[pair+64]=result.y;
}
int main(){
 std::vector<__nv_bfloat16> input(128,__float2bfloat16(0));input[8]=__float2bfloat16(2.859375f);input[72]=__float2bfloat16(-2.25f);
 auto x=upload(input);__nv_bfloat16*out,*ref;float*table;CHECK(cudaMalloc(&out,256));CHECK(cudaMalloc(&ref,256));CHECK(cudaMalloc(&table,6*512));
 rope_table_f32_kernel<<<1,256>>>(table,6,1e7f);
 reference_rope<<<dim3(1,1,1),256>>>(x,ref,128,1,1,1e7f,5);explicit_rope<<<1,64>>>(x,out,table);CHECK(cudaDeviceSynchronize());
 std::vector<uint16_t> got(128),expected(128);CHECK(cudaMemcpy(got.data(),out,256,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(expected.data(),ref,256,cudaMemcpyDeviceToHost));
 int bad=0;for(int i=0;i<128;++i)bad+=got[i]!=expected[i];
 printf("expected=0x%04x got=0x%04x bit_mismatches=%d\n",expected[72],got[72],bad);
 for(void*p:{(void*)x,(void*)out,(void*)ref,(void*)table})CHECK(cudaFree(p));return bad || expected[72]!=0x3981 ? 2 : 0;
}
