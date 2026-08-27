#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "../../crates/apxinf-cuda/kernels/custom/selection.cuh"

static void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(1);
    }
}

int main() {
    constexpr uint32_t kVocab = 151936;
    constexpr uint32_t kBlocks = APXINF_ARGMAX_PARTIAL_BLOCKS;
    constexpr int kIterations = 1000;
    std::vector<__nv_bfloat16> host(kVocab, __float2bfloat16(-4.0f));
    host[17] = __float2bfloat16(8.0f);
    host[150000] = __float2bfloat16(8.0f);

    __nv_bfloat16* logits = nullptr;
    ArgmaxPair* partials = nullptr;
    uint32_t* host_output = nullptr;
    uint32_t* device_output = nullptr;
    check(cudaMalloc(&logits, host.size() * sizeof(host[0])), "cudaMalloc logits");
    check(cudaMalloc(&partials, kBlocks * sizeof(ArgmaxPair)), "cudaMalloc partials");
    check(cudaHostAlloc(&host_output, sizeof(uint32_t), cudaHostAllocMapped),
          "cudaHostAlloc output");
    check(cudaHostGetDevicePointer(&device_output, host_output, 0),
          "cudaHostGetDevicePointer");
    check(cudaMemcpy(logits, host.data(), host.size() * sizeof(host[0]),
                     cudaMemcpyHostToDevice),
          "cudaMemcpy logits");

    for (int i = 0; i < 10; ++i) {
        argmax_bf16_kernel<<<1, 256>>>(logits, kVocab, device_output);
        argmax_bf16_partials_kernel<<<kBlocks, 256>>>(logits, kVocab, partials);
        argmax_pair_final_kernel<<<1, 256>>>(partials, kBlocks, device_output);
    }
    check(cudaDeviceSynchronize(), "warmup");

    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "cudaEventCreate start");
    check(cudaEventCreate(&stop), "cudaEventCreate stop");
    check(cudaEventRecord(start), "cudaEventRecord one start");
    for (int i = 0; i < kIterations; ++i)
        argmax_bf16_kernel<<<1, 256>>>(logits, kVocab, device_output);
    check(cudaEventRecord(stop), "cudaEventRecord one stop");
    check(cudaEventSynchronize(stop), "cudaEventSynchronize one");
    float one_ms = 0.0f;
    check(cudaEventElapsedTime(&one_ms, start, stop), "cudaEventElapsedTime one");
    uint32_t one_output = *host_output;

    check(cudaEventRecord(start), "cudaEventRecord two start");
    for (int i = 0; i < kIterations; ++i) {
        argmax_bf16_partials_kernel<<<kBlocks, 256>>>(logits, kVocab, partials);
        argmax_pair_final_kernel<<<1, 256>>>(partials, kBlocks, device_output);
    }
    check(cudaEventRecord(stop), "cudaEventRecord two stop");
    check(cudaEventSynchronize(stop), "cudaEventSynchronize two");
    float two_ms = 0.0f;
    check(cudaEventElapsedTime(&two_ms, start, stop), "cudaEventElapsedTime two");
    uint32_t two_output = *host_output;

    std::printf(
        "{\"vocab\":%u,\"blocks\":%u,\"iterations\":%d,"
        "\"one_block_us\":%.3f,\"two_stage_us\":%.3f,"
        "\"one_output\":%u,\"two_output\":%u}\n",
        kVocab, kBlocks, kIterations, one_ms * 1000.0f / kIterations,
        two_ms * 1000.0f / kIterations, one_output, two_output);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFreeHost(host_output);
    cudaFree(partials);
    cudaFree(logits);
    return one_output == 17 && two_output == 17 ? 0 : 2;
}
