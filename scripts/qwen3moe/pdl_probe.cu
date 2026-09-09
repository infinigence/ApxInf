// M3.2: PDL correctness and observable prologue overlap, eager and captured.
// nvcc -O3 -arch=sm_101 pdl_probe.cu -o pdl_probe
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#define CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess) { fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e)); return 1; } } while(0)
__device__ unsigned long long timestamp() {
  unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}
__global__ void producer(int* value, unsigned long long* times) {
  cudaTriggerProgrammaticLaunchCompletion();
  auto start=timestamp();
  while(timestamp()-start<1000000) {}
  *value=42; times[0]=timestamp();
}
__global__ void consumer(const int* value, int* result, unsigned long long* times) {
  times[1]=timestamp(); // independent prologue starts before waiting for data
  auto start=timestamp();
  while(timestamp()-start<100000) {}
  cudaGridDependencySynchronize();
  *result=*value; times[2]=timestamp();
}
int main() {
  cudaStream_t stream;CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
  int *value,*result;unsigned long long* times;
  CHECK(cudaMalloc(&value,sizeof(int)));CHECK(cudaMalloc(&result,sizeof(int)));CHECK(cudaMalloc(&times,3*sizeof(*times)));
  cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeProgrammaticStreamSerialization;attr.val.programmaticStreamSerializationAllowed=1;
  cudaLaunchConfig_t config{};config.gridDim=dim3(1);config.blockDim=dim3(1);config.stream=stream;config.attrs=&attr;config.numAttrs=1;
  for(int iteration=-1;iteration<6;++iteration) {
    int capture=iteration>=3;
    CHECK(cudaMemsetAsync(value,0,sizeof(int),stream));
    CHECK(cudaMemsetAsync(result,0,sizeof(int),stream));
    cudaGraph_t graph{};cudaGraphExec_t exec{};
    if(capture)CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    CHECK(cudaLaunchKernelEx(&config,producer,value,times));
    CHECK(cudaLaunchKernelEx(&config,consumer,value,result,times));
    if(capture) { CHECK(cudaStreamEndCapture(stream,&graph));CHECK(cudaGraphInstantiate(&exec,graph,0));CHECK(cudaGraphLaunch(exec,stream)); }
    CHECK(cudaStreamSynchronize(stream));
    int got;unsigned long long t[3];CHECK(cudaMemcpy(&got,result,sizeof(int),cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(t,times,sizeof(t),cudaMemcpyDeviceToHost));
    if(iteration>=0)printf("capture=%d value=%d overlap=%d producer_end=%llu consumer_start=%llu consumer_end=%llu\n",capture,got,t[1]<t[0],t[0],t[1],t[2]);
    if(got!=42 || t[2]<t[0])return 2;
    if(capture) { CHECK(cudaGraphExecDestroy(exec));CHECK(cudaGraphDestroy(graph)); }
  }
  CHECK(cudaFree(value));CHECK(cudaFree(result));CHECK(cudaFree(times));CHECK(cudaStreamDestroy(stream));
}
