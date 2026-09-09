// Compare the production vector SiLU specialization against the scalar LUT path.
// nvcc -O3 -arch=sm_101 scripts/qwen3moe/silu_vector_probe.cu -o silu_vector_probe
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "../../crates/apxinf-cuda/kernels/custom/decode_epilogues.cuh"
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));exit(1);}}while(0)


int main() {
    float* table;CHECK(cudaMalloc(&table,65536*4));
    silu_bf16_table_kernel<<<256,256>>>(table);
    for(int rows:{1,5,28,86,128,8192}) {
        const int inter=768;std::vector<half> input(int64_t(rows)*inter*2);
        half *gu,*original,*candidate;CHECK(cudaMalloc(&gu,input.size()*2));
        CHECK(cudaMalloc(&original,int64_t(rows)*inter*2));CHECK(cudaMalloc(&candidate,int64_t(rows)*inter*2));
        for(int pass=0;pass<(rows==86?8:1);++pass) {
            const float up_values[]={1,-1,.1f,.3f,2,1e-3f,5,-5};
            for(int row=0;row<rows;++row)for(int i=0;i<inter;++i) {
                uint16_t bits=(row*inter+i)%65536;
                // Exhaust all finite FP16 gates; cover eight independent up values.
                if((bits&0x7c00)==0x7c00)bits=0;
                input[int64_t(row)*inter*2+i]=__ushort_as_half(bits);
                input[int64_t(row)*inter*2+inter+i]=__float2half(up_values[pass]);
            }
            CHECK(cudaMemcpy(gu,input.data(),input.size()*2,cudaMemcpyHostToDevice));
            silu_mul_rows_f16_lut_kernel<<<256,256>>>(gu,original,table,rows,inter);
            silu_mul_rows_f16_lut_768_kernel<<<256,256>>>(gu,candidate,table,rows);CHECK(cudaDeviceSynchronize());
            std::vector<half> a(int64_t(rows)*inter),b(a.size());
            CHECK(cudaMemcpy(a.data(),original,a.size()*2,cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(b.data(),candidate,b.size()*2,cudaMemcpyDeviceToHost));
            if(std::memcmp(a.data(),b.data(),a.size()*2)){fprintf(stderr,"vector SiLU mismatch rows=%d pass=%d\n",rows,pass);return 2;}
        }
        printf("rows=%d bit_equal=1\n",rows);
        if(rows==8192)for(int vector:{0,1,0,1}) {
            auto launch=[&](){if(vector)silu_mul_rows_f16_lut_768_kernel<<<256,256>>>(gu,candidate,table,rows);else silu_mul_rows_f16_lut_kernel<<<256,256>>>(gu,original,table,rows,inter);};
            for(int i=0;i<10;++i)launch();
            cudaEvent_t begin,end;CHECK(cudaEventCreate(&begin));CHECK(cudaEventCreate(&end));
            CHECK(cudaEventRecord(begin));for(int i=0;i<100;++i)launch();CHECK(cudaEventRecord(end));CHECK(cudaEventSynchronize(end));
            float ms;CHECK(cudaEventElapsedTime(&ms,begin,end));printf("vector=%d us=%.3f\n",vector,ms*10);
            CHECK(cudaEventDestroy(begin));CHECK(cudaEventDestroy(end));
        }
        CHECK(cudaFree(gu));CHECK(cudaFree(original));CHECK(cudaFree(candidate));
    }
    CHECK(cudaFree(table));return 0;
}
