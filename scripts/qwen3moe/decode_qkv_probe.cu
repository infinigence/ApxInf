#define main prefill_probe_main
#include "qkv_probe.cu"
#undef main
int main(){
 const int qh=32,kh=4,capacity=8192,cols=(qh+2*kh)*128;const float eps=1e-6f;
 for(float theta:{1e6f,1e7f})for(int splits:{1,2,4,8})for(uint32_t pos:{0u,7u,1023u,8191u}){
  uint32_t seed=51;auto random=[&](){seed=seed*1664525+1013904223;return float(int(seed>>16)-32768)/32768.f;};
  std::vector<float> parts(splits*cols);for(auto&x:parts)x=random();
  std::vector<__nv_bfloat16> summed(cols),qw(128),kw(128);
  for(int i=0;i<cols;++i){float x=0;for(int s=0;s<splits;++s)x+=parts[s*cols+i];summed[i]=__float2bfloat16(x);}
  for(auto&x:qw)x=__float2bfloat16(1+random());for(auto&x:kw)x=__float2bfloat16(1+random());
  auto dp=upload(parts);auto dqw=upload(qw),dkw=upload(kw),dx=upload(summed);auto position=upload(std::vector<uint32_t>{pos});
  __nv_bfloat16 *qn,*kn,*qr,*kr,*oq,*ck,*cv;
  CHECK(cudaMalloc(&qn,qh*128*2));CHECK(cudaMalloc(&kn,kh*128*2));CHECK(cudaMalloc(&qr,qh*128*2));CHECK(cudaMalloc(&kr,kh*128*2));CHECK(cudaMalloc(&oq,qh*128*2));
  CHECK(cudaMalloc(&ck,kh*capacity*128*2));CHECK(cudaMalloc(&cv,kh*capacity*128*2));CHECK(cudaMemset(ck,0,kh*capacity*128*2));CHECK(cudaMemset(cv,0,kh*capacity*128*2));
  float*rope;CHECK(cudaMalloc(&rope,capacity*512));rope_table_f32_kernel<<<256,256>>>(rope,capacity,theta);
  reference_norm<<<qh,256,512>>>(dx,dqw,qn,128,qh,eps);reference_norm<<<kh,256,512>>>(dx+qh*128,dkw,kn,128,kh,eps);
  reference_rope<<<dim3(1,qh,1),256>>>(qn,qr,128,qh,1,theta,pos);reference_rope<<<dim3(1,kh,1),256>>>(kn,kr,128,kh,1,theta,pos);
  cudaStream_t stream;CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));CHECK(cudaDeviceSynchronize());
  auto launch=[&](){qkv_partial_norm_rope_cache_bf16_kernel<<<(qh+kh+3)/4,128,0,stream>>>(dp,dqw,dkw,oq,ck,cv,position,qh,kh,capacity,splits,eps,rope);};
  launch();CHECK(cudaDeviceSynchronize());
  std::vector<__nv_bfloat16> rq(qh*128),rk(kh*128),gotq(rq.size()),gotk(kh*capacity*128),gotv(gotk.size());
  CHECK(cudaMemcpy(rq.data(),qr,rq.size()*2,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(rk.data(),kr,rk.size()*2,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(gotq.data(),oq,rq.size()*2,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(gotk.data(),ck,gotk.size()*2,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(gotv.data(),cv,gotv.size()*2,cudaMemcpyDeviceToHost));
  int bad=0;for(int i=0;i<qh*128;++i)bad+=memcmp(&rq[i],&gotq[i],2)!=0;
  for(int h=0;h<kh;++h)for(int t=0;t<capacity;++t)for(int d=0;d<128;++d){auto ek=__float2bfloat16(0),ev=ek;if(t==pos){ek=rk[h*128+d];ev=summed[(qh+kh+h)*128+d];}int i=(h*capacity+t)*128+d;bad+=memcmp(&ek,&gotk[i],2)!=0;bad+=memcmp(&ev,&gotv[i],2)!=0;}
  cudaGraph_t graph;cudaGraphExec_t exec;CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));launch();CHECK(cudaStreamEndCapture(stream,&graph));CHECK(cudaGraphInstantiate(&exec,graph,0,0,0));
  CHECK(cudaGraphLaunch(exec,stream));CHECK(cudaDeviceSynchronize());CHECK(cudaMemcpy(gotq.data(),oq,rq.size()*2,cudaMemcpyDeviceToHost));for(int i=0;i<qh*128;++i)bad+=memcmp(&rq[i],&gotq[i],2)!=0;
  printf("theta=%g splits=%d position=%u bit_mismatches=%d\n",theta,splits,pos,bad);if(bad)return 2;
  CHECK(cudaGraphExecDestroy(exec));CHECK(cudaGraphDestroy(graph));CHECK(cudaStreamDestroy(stream));for(void*p:{(void*)dp,(void*)dqw,(void*)dkw,(void*)dx,(void*)position,(void*)qn,(void*)kn,(void*)qr,(void*)kr,(void*)oq,(void*)ck,(void*)cv,(void*)rope})CHECK(cudaFree(p));
 }
}
