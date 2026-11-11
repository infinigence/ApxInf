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
struct Stream {
  md::Handle handle;
  std::map<std::tuple<int,int,int>,std::unique_ptr<Matmul>> matmuls;
  std::map<std::tuple<int,int,int,int>,std::unique_ptr<Flash>> flashes;
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
