#pragma once

#include "mutlass/arch/barrier.hpp"

namespace mate::attention::fmha {

MUTLASS_DEVICE
static void named_barrier_arrive(uint32_t barrier_id_) {
  uint32_t barrier_id = barrier_id_ + mutlass::arch::AsyncBarrier::ReservedAsyncBarrierCount;
  mutlass::arch::AsyncBarrier::arrive(barrier_id);
}

MUTLASS_DEVICE
static void named_barrier_sync(uint32_t barrier_id_) {
  uint32_t barrier_id = barrier_id_ + mutlass::arch::AsyncBarrier::ReservedAsyncBarrierCount;
  mutlass::arch::AsyncBarrier::sync(barrier_id);
}

enum class FwdNamedBarriers {
  PipelineWrapPhase0  = 0,
  PipelineWrapPhase1  = 1,
  ReuseP              = 2,
  AppendKV            = 3,
  BarrierKV           = 4,
  RotaryQ             = 5,
  NumFwdNamedBarriers = 6
};

}  // namespace mate::attention::fmha
