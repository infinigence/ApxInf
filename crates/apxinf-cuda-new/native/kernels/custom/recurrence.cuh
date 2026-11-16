#pragma once
#include "../../include/apxinf_cuda/recurrence.h"

namespace apxinf::recurrence::kernels {
__global__ void recurrence(apxinf_recurrence_spec_t s, apxinf_recurrence_bindings_t b) {
  for (int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x; i<s.count; i+=int64_t(gridDim.x)*blockDim.x) {
    double current=double(b.input[i]);
    double next=current;
    if (!s.reset) {
      double combined=__dadd_rn(__dmul_rn(s.b,current),__dmul_rn(s.c,b.previous[i]));
      next=__dadd_rn(__dmul_rn(s.a,b.state[i]),__dmul_rn(s.d,combined));
    }
    b.state[i]=next; b.previous[i]=s.reset?0.:current; b.output[i]=__double2float_rn(next);
  }
}
}  // namespace apxinf::recurrence::kernels
