// Native muDNN operators. Descriptors and scratch belong to a device/stream;
// no torch objects, host fallback, or hot-path allocation is involved.
//
// Contract
// --------
// Every entry point returns **0 on success** and non-zero on failure, and
// `apxinf_musa_mudnn_last_error()` names the reason for the last failure on this
// thread. Two vocabularies meet in that `int`, and they are kept apart by
// construction rather than by luck: the vendor's `md::Status` values are
// whatever the enum says, and this adapter's own conditions are `kStatus*`
// below, starting at 1000. Before that, "the scratch buffer is the wrong size"
// reached Rust as the bare number 1005 with nothing anywhere to say what it
// meant -- the Rust side formats the code it is handed, so an unnamed code is an
// unreadable error.
//
// `mublas_adapter.mu` and the two FA adapters return their own status types;
// this one returns `int` because muDNN's own Run() does, and translating would
// mean inventing a mapping for codes this file does not own.
#include <mudnn.h>
#include <musa_runtime.h>
#include <cmath>
#include <map>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "adapter_common.h"

namespace {
namespace md = musa::dnn;

// This adapter's own failure codes. 1000 and up, so they cannot collide with a
// `md::Status` value, and named, so a caller reading `status 1004` can find out
// what it means.
enum {
  kStatusCaptureStatusUnreadable = 1002,
  kStatusCaptureInProgress = 1003,
  kStatusAllocationFailed = 1004,
  kStatusScratchTooSmall = 1005,
  kStatusInvalidArgument = 1006,
  kStatusUnknown = 1007,
  kStatusDeviceQueryFailed = 1008,
};

struct StatusError { int status; };
void check(md::Status status) {
  if (status != md::Status::SUCCESS) throw StatusError{int(status)};
}
bool capturing(musaStream_t stream) {
  return apxinf_adapters::capturing(stream);
}
struct Scratch {
  std::vector<void*> allocations;
  std::vector<size_t> sizes;
  size_t cursor = 0;
  musaStream_t stream;
  explicit Scratch(musaStream_t s) : stream(s) {}
  ~Scratch() { for (void* p : allocations) musaFree(p); }
  md::MemoryHandler get(size_t bytes) {
    const size_t slot = cursor++;
    if (slot >= allocations.size()) {
      if (capturing(stream)) throw StatusError{kStatusCaptureInProgress};
      void* p = nullptr;
      if (musaMalloc(&p, bytes) != musaSuccess) throw StatusError{kStatusAllocationFailed};
      allocations.push_back(p); sizes.push_back(bytes);
    } else if (bytes > sizes[slot]) {
      // A later call for the same shape wanted more scratch than the first did.
      // This used to throw and leave the shape permanently unusable, because the
      // entry is keyed on the shape and never retried -- an intermittent muDNN
      // failure that could not heal. Reallocating is what the size is for.
      //
      // Refused during capture, for the reason every other first-touch
      // allocation is: the graph baked the pointer this would replace.
      if (capturing(stream)) throw StatusError{kStatusCaptureInProgress};
      void* p = nullptr;
      if (musaMalloc(&p, bytes) != musaSuccess) throw StatusError{kStatusAllocationFailed};
      musaFree(allocations[slot]);
      allocations[slot] = p;
      sizes[slot] = bytes;
    }
    return md::MemoryHandler(allocations[slot], [](void*) {});
  }
};
struct Matmul {
  md::Tensor a, w, y;
  md::MatMul operation;
  Scratch scratch;
  Matmul(int m,int n,int k,musaStream_t stream) : scratch(stream) {
    for(auto p:{&a,&w,&y}) check(p->SetType(md::Tensor::Type::BFLOAT16));
    check(a.SetNdInfo({m,k})); check(w.SetNdInfo({k,n})); check(y.SetNdInfo({m,n}));
    check(operation.SetTranspose(false,false));
    check(operation.SetComputeMode(md::MatMul::ComputeMode::TENSOR));
    check(operation.SetAlpha(1)); check(operation.SetBeta(0));
    check(operation.SetDeterministic(true));
  }
};
#ifdef APXINF_MUSA_MP31_FP8
struct Fp8Matmul {
  struct FreeBuffer { void operator()(void* pointer) const { musaFree(pointer); } };
  md::Tensor activation, weight, output, bias, scales[4];
  md::BatchMatMul operation;
  md::MatMulLtParam parameters;
  Scratch scratch;
  std::unique_ptr<void, FreeBuffer> constants;
  Fp8Matmul(int rows, int cols, int inner, bool transpose, musaStream_t stream)
      : scratch(stream) {
    void* allocation = nullptr;
    const size_t bytes = 4 * sizeof(float) + size_t(cols) * 2;
    if (musaMalloc(&allocation, bytes) != musaSuccess)
      throw StatusError{kStatusAllocationFailed};
    constants.reset(allocation);
    const float units[] = {1, 1, 1, 1};
    if (musaMemset(allocation, 0, bytes) != musaSuccess ||
        musaMemcpy(allocation, units, sizeof(units), musaMemcpyHostToDevice) != musaSuccess)
      throw StatusError{kStatusUnknown};
    check(activation.SetType(md::Tensor::Type::FP8_E4M3));
    check(weight.SetType(md::Tensor::Type::FP8_E4M3));
    check(output.SetType(md::Tensor::Type::BFLOAT16));
    check(bias.SetType(md::Tensor::Type::BFLOAT16));
    check(activation.SetNdInfo({1, rows, inner}));
    if (transpose) check(weight.SetNdInfo({1, cols, inner}));
    else check(weight.SetNdInfo({1, inner, cols}));
    check(output.SetNdInfo({1, rows, cols}));
    check(bias.SetNdInfo({cols}));
    check(bias.SetAddr(static_cast<float*>(allocation) + 4));
    for (int index = 0; index < 4; ++index) {
      check(scales[index].SetType(md::Tensor::Type::FLOAT));
      check(scales[index].SetNdInfo({1}));
      check(scales[index].SetAddr(static_cast<float*>(allocation) + index));
    }
    check(parameters.SetScale(scales[0], scales[1], scales[2], scales[3]));
    check(parameters.SetEpilogue(md::MatMulLtParam::MatMulLtEpilogueMode::MATMULLT_EPILOGUE_DEFAULT));
    check(operation.SetTranspose(false, transpose));
    check(operation.SetComputeMode(md::BatchMatMul::ComputeMode::TENSOR));
    check(operation.SetBeta(0));
    check(operation.SetGamma(0));
  }
};
#endif
struct Flash {
  md::Tensor q,k,v,y,lse,mask,dropout;
  md::ScaledDotProductAttention operation;
  Scratch scratch;
  void* lse_buffer = nullptr;
  Flash(int batch,int heads,int tokens,int dim,musaStream_t stream) : scratch(stream) {
    for(auto p:{&q,&k,&v,&y,&mask}) check(p->SetType(md::Tensor::Type::BFLOAT16));
    for(auto p:{&q,&k,&v,&y}) check(p->SetNdInfo({batch,heads,tokens,dim},
      {tokens*heads*dim,dim,heads*dim,1}));
    check(lse.SetType(md::Tensor::Type::FLOAT)); check(lse.SetNdInfo({batch,heads,tokens}));
    check(mask.SetNdInfo({0}));
    check(dropout.SetType(md::Tensor::Type::FLOAT)); check(dropout.SetNdInfo({0}));
    check(operation.SetBatchSize(batch)); check(operation.SetHeadsNum(heads));
    check(operation.SetEmbedDim(heads*dim)); check(operation.SetScale(1./std::sqrt(double(dim))));
    check(operation.SetCausal(false)); check(operation.SetMaskMode(false));
    check(operation.SetTraining(false)); check(operation.SetDropoutP(0));
    // Pinned, like Matmul above and for the same reason: a captured graph is
    // only worth capturing if replaying it twice gives one answer twice. The
    // vendor default is not documented as reproducible, and one run of this
    // binary against a fixed fixture has disagreed with three others at the
    // sixth significant digit of the end-to-end cosine with no input differing.
    check(operation.SetIsDeterministic(true));
    // Last, so no later `check` can throw past an owned allocation. A throwing
    // constructor does not run `~Flash` (only the members'), and a rejected
    // shape re-runs the constructor on every call -- `make_unique` never
    // inserts it -- so a leak here would repeat rather than happen once.
    if(musaMalloc(&lse_buffer,size_t(batch)*heads*tokens*sizeof(float))!=musaSuccess)
      throw StatusError{kStatusAllocationFailed};
    check(lse.SetAddr(lse_buffer));
  }
  ~Flash() { if(lse_buffer) musaFree(lse_buffer); }
};

// General BF16 flash-attention descriptor with independent Q/K lengths, GQA
// head counts, causal selection and an optional packed-QKV physical layout.
// Keep it separate from `Flash`, whose established D72 path is used by PI0.5.
struct GeneralFlash {
  md::Tensor q,k,v,y,lse,mask,dropout;
  md::ScaledDotProductAttention operation;
  Scratch scratch;
  void* lse_buffer = nullptr;
  GeneralFlash(int batch,int query_heads,int kv_heads,int query_tokens,int key_tokens,
               int dim,bool causal,bool packed_qkv,musaStream_t stream) : scratch(stream) {
    for(auto p:{&q,&k,&v,&y,&mask}) check(p->SetType(md::Tensor::Type::BFLOAT16));
    check(y.SetNdInfo({batch,query_heads,query_tokens,dim},
      {query_tokens*query_heads*dim,dim,query_heads*dim,1}));
    if(packed_qkv) {
      const int token_stride=3*query_heads*dim;
      for(auto p:{&q,&k,&v}) check(p->SetNdInfo(
        {batch,query_heads,query_tokens,dim},
        {query_tokens*token_stride,dim,token_stride,1}));
    } else {
      check(q.SetNdInfo({batch,query_heads,query_tokens,dim},
        {query_tokens*query_heads*dim,dim,query_heads*dim,1}));
      for(auto p:{&k,&v}) check(p->SetNdInfo({batch,kv_heads,key_tokens,dim},
        {key_tokens*kv_heads*dim,dim,kv_heads*dim,1}));
    }
    check(lse.SetType(md::Tensor::Type::FLOAT));
    check(lse.SetNdInfo({batch,query_heads,query_tokens}));
    check(mask.SetNdInfo({0}));
    check(dropout.SetType(md::Tensor::Type::FLOAT)); check(dropout.SetNdInfo({0}));
    check(operation.SetBatchSize(batch)); check(operation.SetHeadsNum(query_heads));
    check(operation.SetEmbedDim(query_heads*dim));
    check(operation.SetScale(1./std::sqrt(double(dim))));
    check(operation.SetCausal(causal)); check(operation.SetMaskMode(false));
    check(operation.SetMaxSeqlenQ(query_tokens));
    check(operation.SetMaxSeqlenK(key_tokens));
    check(operation.SetTraining(false)); check(operation.SetDropoutP(0));
    check(operation.SetIsDeterministic(true));
    if(musaMalloc(&lse_buffer,size_t(batch)*query_heads*query_tokens*sizeof(float))!=musaSuccess)
      throw StatusError{kStatusAllocationFailed};
    check(lse.SetAddr(lse_buffer));
  }
  ~GeneralFlash() { if(lse_buffer) musaFree(lse_buffer); }
};
struct Stream {
  md::Handle handle;
#ifdef APXINF_MUSA_MP31_FP8
  std::map<std::tuple<int,int,int,bool>,std::unique_ptr<Fp8Matmul>> fp8_matmuls;
#endif
  std::map<std::tuple<int,int,int>,std::unique_ptr<Matmul>> matmuls;
  std::map<std::tuple<int,int,int,int>,std::unique_ptr<Flash>> flashes;
  std::map<std::tuple<int,int,int,int,int,int,bool,bool>,std::unique_ptr<GeneralFlash>> general_flashes;
  Stream(int device,musaStream_t stream) : handle(device) {
    check(handle.SetStream(stream)); check(handle.SetAllowTF32(false));
  }
};
thread_local std::map<std::pair<int,musaStream_t>,std::unique_ptr<Stream>> streams;
Stream& state(musaStream_t stream) {
  const int device=apxinf_adapters::stream_device(stream);
  if(device<0) throw StatusError{kStatusDeviceQueryFailed};
  auto key=std::make_pair(device,stream); auto it=streams.find(key);
  if(it==streams.end()) {
    if(capturing(stream)) throw StatusError{kStatusCaptureInProgress};
    it=streams.emplace(key,std::make_unique<Stream>(device,stream)).first;
  }
  return *it->second;
}

/// Name the reason the last call on this thread failed.
///
/// `set_error` writes the buffer; the two entry points below are the only
/// writers, and each writes on the way out. A caller that reads this after a
/// non-zero return gets a sentence; one that reads it after a zero return gets
/// whatever the previous failure said, which is why the Rust side only asks
/// when the status is non-zero.
void record_failure(int status, const char* what) {
  switch (status) {
    case kStatusDeviceQueryFailed:
      apxinf_adapters::set_error(
          "%s: the stream\'s device could not be named, so the allocation is refused "
          "rather than guessed at", what);
      break;
    case kStatusCaptureStatusUnreadable:
      apxinf_adapters::set_error(
          "%s: musaStreamIsCapturing did not answer, so whether this stream is capturing "
          "is unknown. The allocation is refused rather than guessed at", what);
      break;
    case kStatusCaptureInProgress:
      apxinf_adapters::set_error(
          "%s: the muDNN descriptors for this shape were first needed during graph capture; "
          "a captured graph bakes the pointers it recorded, so every shape must be seen once "
          "on the warmup traversal before musaStreamBeginCapture", what);
      break;
    case kStatusAllocationFailed:
      apxinf_adapters::set_error("%s: musaMalloc failed", what);
      break;
    case kStatusScratchTooSmall:
      apxinf_adapters::set_error("%s: the muDNN scratch buffer for this shape is too small", what);
      break;
    case kStatusInvalidArgument:
      apxinf_adapters::set_error("%s: a pointer was null or a dimension was not positive", what);
      break;
    case kStatusUnknown:
      apxinf_adapters::set_error("%s: an exception this adapter does not name", what);
      break;
    default:
      // A vendor `md::Status`: it has its own names, and inventing one here
      // would be a second vocabulary for the same number.
      apxinf_adapters::set_error("%s: muDNN returned status %d", what, status);
      break;
  }
}
}

extern "C" int apxinf_musa_mudnn_fp8_gemm(const void* activation, const void* weight,
    void* output, int rows, int cols, int inner, int transpose, float scale,
    musaStream_t stream) {
#ifdef APXINF_MUSA_MP31_FP8
  if (!activation || !weight || !output || rows <= 0 || cols <= 0 || inner <= 0 ||
      (transpose != 0 && transpose != 1) || !std::isfinite(scale) || scale <= 0) {
    record_failure(kStatusInvalidArgument, "muDNN FP8 GEMM");
    return kStatusInvalidArgument;
  }
  try {
    auto& current = state(stream);
    const auto key = std::make_tuple(rows, cols, inner, bool(transpose));
    auto found = current.fp8_matmuls.find(key);
    if (found == current.fp8_matmuls.end()) {
      if (capturing(stream)) throw StatusError{kStatusCaptureInProgress};
      found = current.fp8_matmuls.emplace(key,
          std::make_unique<Fp8Matmul>(rows, cols, inner, bool(transpose), stream)).first;
    }
    auto& plan = *found->second;
    check(plan.activation.SetAddr(const_cast<void*>(activation)));
    check(plan.weight.SetAddr(const_cast<void*>(weight)));
    check(plan.output.SetAddr(output));
    check(plan.operation.SetAlpha(scale));
    plan.scratch.cursor = 0;
    check(plan.operation.RunLt(current.handle, plan.output, plan.activation, plan.weight,
        plan.output, plan.bias, plan.parameters,
        [&](size_t bytes) { return plan.scratch.get(bytes); }));
    return 0;
  } catch (const StatusError& error) {
    record_failure(error.status, "muDNN FP8 GEMM");
    return error.status;
  } catch (...) {
    record_failure(kStatusUnknown, "muDNN FP8 GEMM");
    return kStatusUnknown;
  }
#else
  record_failure(kStatusInvalidArgument, "muDNN FP8 GEMM requires an MP31 build");
  return kStatusInvalidArgument;
#endif
}

extern "C" int apxinf_musa_mudnn_bf16_gemm(const void* a,const void* w,void* y,
 int m,int n,int k,musaStream_t stream) {
 if(!a||!w||!y||m<=0||n<=0||k<=0) {
   record_failure(kStatusInvalidArgument, "muDNN BF16 GEMM");
   return kStatusInvalidArgument;
 }
 try {
  auto& s=state(stream); auto key=std::make_tuple(m,n,k); auto it=s.matmuls.find(key);
  if(it==s.matmuls.end()) {
    if(capturing(stream)) {
      record_failure(kStatusCaptureInProgress, "muDNN BF16 GEMM");
      return kStatusCaptureInProgress;
    }
    it=s.matmuls.emplace(key,std::make_unique<Matmul>(m,n,k,stream)).first;
  }
  auto& p=*it->second; check(p.a.SetAddr(const_cast<void*>(a)));
  check(p.w.SetAddr(const_cast<void*>(w))); check(p.y.SetAddr(y)); p.scratch.cursor=0;
  const int status = int(p.operation.Run(s.handle,p.y,p.a,p.w,[&](size_t n){return p.scratch.get(n);}));
  if (status != 0) record_failure(status, "muDNN BF16 GEMM");
  return status;
 } catch(const StatusError& e) {
   record_failure(e.status, "muDNN BF16 GEMM");
   return e.status;
 } catch(...) {
   record_failure(kStatusUnknown, "muDNN BF16 GEMM");
   return kStatusUnknown;
 }
}

extern "C" int apxinf_musa_mudnn_flash_bf16(const void* q,const void* k,const void* v,
 void* y,int batch,int heads,int tokens,int dim,musaStream_t stream) {
 if(!q||!k||!v||!y||batch<=0||heads<=0||tokens<=0||dim<=0) {
   record_failure(kStatusInvalidArgument, "muDNN flash attention");
   return kStatusInvalidArgument;
 }
 try {
  auto& s=state(stream); auto key=std::make_tuple(batch,heads,tokens,dim); auto it=s.flashes.find(key);
  if(it==s.flashes.end()) {
    if(capturing(stream)) {
      record_failure(kStatusCaptureInProgress, "muDNN flash attention");
      return kStatusCaptureInProgress;
    }
    it=s.flashes.emplace(key,std::make_unique<Flash>(batch,heads,tokens,dim,stream)).first;
  }
  auto& p=*it->second; check(p.q.SetAddr(const_cast<void*>(q)));check(p.k.SetAddr(const_cast<void*>(k)));
  check(p.v.SetAddr(const_cast<void*>(v)));check(p.y.SetAddr(y));p.scratch.cursor=0;
  const int status = int(p.operation.RunFlash(s.handle,p.y,p.lse,p.q,p.k,p.v,p.mask,p.dropout,
    [&](size_t n){return p.scratch.get(n);}));
  if (status != 0) record_failure(status, "muDNN flash attention");
  return status;
 } catch(const StatusError& e) {
   record_failure(e.status, "muDNN flash attention");
   return e.status;
 } catch(...) {
   record_failure(kStatusUnknown, "muDNN flash attention");
   return kStatusUnknown;
 }
}

extern "C" int apxinf_musa_mudnn_general_flash_bf16(
 const void* q,const void* k,const void* v,void* y,int batch,
 int query_heads,int kv_heads,int query_tokens,int key_tokens,int dim,
 bool causal,musaStream_t stream) {
 if(!q||!k||!v||!y||batch<=0||query_heads<=0||kv_heads<=0||
    query_tokens<=0||key_tokens<=0||dim<=0||query_heads%kv_heads!=0) {
   record_failure(kStatusInvalidArgument, "muDNN general flash attention");
   return kStatusInvalidArgument;
 }
 try {
  auto& s=state(stream);
  auto key=std::make_tuple(batch,query_heads,kv_heads,query_tokens,key_tokens,dim,causal,false);
  auto it=s.general_flashes.find(key);
  if(it==s.general_flashes.end()) {
    if(capturing(stream)) {
      record_failure(kStatusCaptureInProgress, "muDNN general flash attention");
      return kStatusCaptureInProgress;
    }
    it=s.general_flashes.emplace(
      key,std::make_unique<GeneralFlash>(batch,query_heads,kv_heads,query_tokens,
                                         key_tokens,dim,causal,false,stream)).first;
  }
  auto& p=*it->second;
  check(p.q.SetAddr(const_cast<void*>(q))); check(p.k.SetAddr(const_cast<void*>(k)));
  check(p.v.SetAddr(const_cast<void*>(v))); check(p.y.SetAddr(y)); p.scratch.cursor=0;
  const int status = int(p.operation.RunFlash(s.handle,p.y,p.lse,p.q,p.k,p.v,p.mask,p.dropout,
    [&](size_t n){return p.scratch.get(n);}));
  if (status != 0) record_failure(status, "muDNN general flash attention");
  return status;
 } catch(const StatusError& e) {
   record_failure(e.status, "muDNN general flash attention");
   return e.status;
 } catch(...) {
   record_failure(kStatusUnknown, "muDNN general flash attention");
   return kStatusUnknown;
 }
}

extern "C" int apxinf_musa_mudnn_packed_qkv_flash_bf16(
 const void* qkv,void* y,int batch,int heads,int tokens,int dim,musaStream_t stream) {
 if(!qkv||!y||batch<=0||heads<=0||tokens<=0||dim<=0) {
   record_failure(kStatusInvalidArgument, "muDNN packed-QKV flash attention");
   return kStatusInvalidArgument;
 }
 try {
  auto& s=state(stream);
  auto key=std::make_tuple(batch,heads,heads,tokens,tokens,dim,false,true);
  auto it=s.general_flashes.find(key);
  if(it==s.general_flashes.end()) {
    if(capturing(stream)) {
      record_failure(kStatusCaptureInProgress, "muDNN packed-QKV flash attention");
      return kStatusCaptureInProgress;
    }
    it=s.general_flashes.emplace(
      key,std::make_unique<GeneralFlash>(batch,heads,heads,tokens,tokens,
                                         dim,false,true,stream)).first;
  }
  auto& p=*it->second;
  const auto* base=static_cast<const unsigned char*>(qkv);
  const size_t projection_bytes=size_t(heads)*size_t(dim)*sizeof(uint16_t);
  check(p.q.SetAddr(const_cast<unsigned char*>(base)));
  check(p.k.SetAddr(const_cast<unsigned char*>(base+projection_bytes)));
  check(p.v.SetAddr(const_cast<unsigned char*>(base+2*projection_bytes)));
  check(p.y.SetAddr(y)); p.scratch.cursor=0;
  const int status = int(p.operation.RunFlash(s.handle,p.y,p.lse,p.q,p.k,p.v,p.mask,p.dropout,
    [&](size_t n){return p.scratch.get(n);}));
  if (status != 0) record_failure(status, "muDNN packed-QKV flash attention");
  return status;
 } catch(const StatusError& e) {
   record_failure(e.status, "muDNN packed-QKV flash attention");
   return e.status;
 } catch(...) {
   record_failure(kStatusUnknown, "muDNN packed-QKV flash attention");
   return kStatusUnknown;
 }
}

/// Why the last call on this thread failed. A static string.
extern "C" const char* apxinf_musa_mudnn_last_error(void) {
  return apxinf_adapters::error_buffer();
}

extern "C" void apxinf_musa_mudnn_release_stream(musaStream_t stream) {
 for(auto it=streams.begin();it!=streams.end();) {
  if(it->first.second==stream) {
   apxinf_adapters::select_device(it->first.first);
   it=streams.erase(it);
  } else ++it;
 }
}
