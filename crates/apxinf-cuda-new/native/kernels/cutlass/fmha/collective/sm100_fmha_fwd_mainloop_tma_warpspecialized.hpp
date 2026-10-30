/***************************************************************************************************
 * Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/arch/memory_sm80.h"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cute/arch/simd_sm100.hpp"
#include "cute/tensor.hpp"
#include "cute/layout.hpp"

#include "collective/fmha_common.hpp"
#include "collective/fmha_fusion.hpp"
#include "collective/sm100_fmha_load_tma_warpspecialized.hpp"

// ===========================================================================
// apxinf: macro vocabulary for the SM100 FMHA forward mainloop.
//
// EVERY knob below defaults to the UPSTREAM CUTLASS behaviour.  With no
// -DAPXINF_* on the command line this header must generate code that is
// numerically identical to the pristine upstream mainloop; that is the gate
// and it is verified by running the existing 2-stage consumer
// (crates/apxinf-cuda/adapters/cutlass_fmha_adapter.cu) and byte-comparing
// its output.  See devlocal/aot2cuda/reports/20-fmha-header-reconciliation.md.
//
// Three AOT contracts are selectable on top of the upstream default:
//   vision   (hd64,  non-causal, FA4 flash_fwd_sm100.py)
//   action   (hd256, non-causal, packed GQA, FA4 flash_fwd_sm100.py)
//   language (hd256, causal, FA4 sm100_hd256_2cta_fmha_forward.py)
// The per-operator flag sets live in devlocal/aot2cuda/harness/FINAL_FLAGS.sh.
// ===========================================================================

// --- Guards against the pre-reconciliation macro names ----------------------
// The two forks named the same concepts differently and, worse, gave some of
// them opposite defaults.  Fail loudly rather than silently selecting the
// wrong contract.
#if defined(APXINF_KV_DESCEND)
#  error "APXINF_KV_DESCEND was renamed to APXINF_KV_DESCENDING (default 0)"
#endif
#if defined(APXINF_ROWSUM_FA4)
#  error "APXINF_ROWSUM_FA4=1 is now APXINF_ACC_SCALE_HALF=0 (upstream default is 1)"
#endif
#if defined(APXINF_RCP_APPROX) || defined(APXINF_ROWSUM_RCP)
#  error "APXINF_RCP_APPROX / APXINF_ROWSUM_RCP were replaced by APXINF_FINAL_NORM (0=upstream div)"
#endif
#if defined(APXINF_LAZY_RESCALE)
#  error "APXINF_LAZY_RESCALE is implied by APXINF_RESCALE_THRESHOLD > 0 (default 0.0f)"
#endif
#if defined(APXINF_STAGE_COUNT_KV)
#  error "APXINF_STAGE_COUNT_KV was renamed to APXINF_FMHA_KV_STAGES (default 0 = upstream formula)"
#endif
#if defined(APXINF_STAGE_COUNT_Q)
#  error "APXINF_STAGE_COUNT_Q is gone; the Q stage count is size(ThreadShape{})"
#endif
#if defined(APXINF_P_FLUSH_OFFSET)
#  error "APXINF_P_FLUSH_OFFSET was replaced by the absolute APXINF_P_FLUSH_AT (default size-6)"
#endif

// --- exp2 ------------------------------------------------------------------
// UPSTREAM: ::exp2f, the correctly-rounded libdevice path.
// The AOT FA4 kernels emit MUFU.EX2 (ex2.approx.f32) instead, and for a
// deterministic subset of elements a cubic-polynomial emulation.
//
// APXINF_EX2_APPROX  0 = ::exp2f (UPSTREAM)
//                    1 = inline ex2.approx[.ftz].f32
// APXINF_EX2_FTZ     only read when APXINF_EX2_APPROX == 1.  `.ftz` is
//   REQUIRED for AOT parity: without it ptxas wraps every ex2 in a -126
//   subnormal guard (FSETP.GEU x,-126 ; FMUL x,0.5 ; MUFU.EX2 ; FMUL r,r),
//   ~560 extra instructions, FSETP.GEU 187 -> 1, precision unchanged.
//   NOTE the opposite answer inside apxinf_ex2_emulation_2 below: there the
//   FMA sequence must NOT be .ftz (the AOT SASS has zero .FTZ there).
#ifndef APXINF_EX2_APPROX
#define APXINF_EX2_APPROX 0
#endif
#ifndef APXINF_EX2_FTZ
#define APXINF_EX2_FTZ 1
#endif

CUTLASS_DEVICE float apxinf_ex2_approx(float x) {
#if APXINF_EX2_APPROX
  float r;
#if APXINF_EX2_FTZ
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
#else
  asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
#endif
  return r;
#else
  return ::exp2f(x);
#endif
}

CUTLASS_DEVICE float apxinf_rcp_approx(float x) {
  float r;
  asm("rcp.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

// FA4 utils.py::fmax forces the 3-input form of max.f32.  max is exact, so
// this cannot move a bit; it halves the rowmax reduction instruction count.
CUTLASS_DEVICE float apxinf_fmax3(float a, float b, float c) {
  float r;
  asm("max.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c));
  return r;
}

// FA4's `utils.ex2_emulation_2`: a software exp2 that spends FMA pipe instead
// of the MUFU pipe.  FA4 interleaves it with hardware ex2.approx (see
// `apxinf_use_hw_ex2`) to balance the two pipes, and the two paths do NOT
// agree bit for bit, so matching the AOT output requires emulating exactly
// the same elements the same way.
//
// Coefficients are POLY_EX2[3] from flash_attn/cute/utils.py, cross-checked
// against the constants in that file's `e2e_asm2` reference PTX
// (0f3F800000 / 0f3F31F519 / 0f3E6906A4 / 0f3D9DF09D).
// NOTE: no .ftz here.  FA4's DSL path (cute.arch.fmax / add_packed_f32x2 /
// fma) emits non-ftz ops and the AOT SASS confirms it: plain FMNMX / FADD.RM /
// FFMA with no .FTZ suffix.  Using the .ftz spelling from utils.py's unused
// e2e_asm2 helper produces FMNMX.FTZ / FADD.FTZ.RM / FFMA.FTZ and does NOT
// match.  This is the opposite answer from apxinf_ex2_approx above, which DOES
// need .ftz -- two different call sites, two different answers.
//
// Written as PTX rather than CUDA intrinsics so the rounding modes and the
// absence of .ftz do not depend on compiler flags.
CUTLASS_DEVICE void apxinf_ex2_emulation_2(float& x, float& y) {
  float ox, oy;
  asm("{\n\t"
      ".reg .f32 xc, yc, xr, yr, xrb, yrb, xf, yf, px, py;\n\t"
      ".reg .s32 xri, yri, pxi, pyi, xre, yre, oxi, oyi;\n\t"
      "max.f32    xc, %2, 0fC2FE0000;\n\t"      // fmax(x, -127.0f)
      "max.f32    yc, %3, 0fC2FE0000;\n\t"
      "add.rm.f32 xr, xc, 0f4B400000;\n\t"      // + (2^23 + 2^22), round -inf
      "add.rm.f32 yr, yc, 0f4B400000;\n\t"
      "sub.rn.f32 xrb, xr, 0f4B400000;\n\t"     // integer part as float
      "sub.rn.f32 yrb, yr, 0f4B400000;\n\t"
      "sub.rn.f32 xf, xc, xrb;\n\t"             // fractional part in [0,1)
      "sub.rn.f32 yf, yc, yrb;\n\t"
      "mov.f32 px, 0f3D9DF09D;\n\t"
      "mov.f32 py, 0f3D9DF09D;\n\t"
      "fma.rn.f32 px, xf, px, 0f3E6906A4;\n\t"
      "fma.rn.f32 py, yf, py, 0f3E6906A4;\n\t"
      "fma.rn.f32 px, px, xf, 0f3F31F519;\n\t"
      "fma.rn.f32 py, py, yf, 0f3F31F519;\n\t"
      "fma.rn.f32 px, px, xf, 0f3F800000;\n\t"
      "fma.rn.f32 py, py, yf, 0f3F800000;\n\t"
      "mov.b32 xri, xr;  mov.b32 yri, yr;\n\t"
      "mov.b32 pxi, px;  mov.b32 pyi, py;\n\t"
      "shl.b32 xre, xri, 23;  shl.b32 yre, yri, 23;\n\t"  // 2^floor(x)
      "add.s32 oxi, xre, pxi;  add.s32 oyi, yre, pyi;\n\t" // LEA in SASS
      "mov.b32 %0, oxi;  mov.b32 %1, oyi;\n\t"
      "}\n"
      : "=f"(ox), "=f"(oy) : "f"(x), "f"(y));
  x = ox;
  y = oy;
}

// Perf variant with IDENTICAL arithmetic (APXINF_EX2_HOIST_C3).  `mov.f32 px,
// 0f3D9DF09D` is loop-invariant but ptxas does not CSE across separate asm
// blocks, so a 128-element fragment paid 16 redundant MOVs; and the two
// `mov.b32 %0, oxi` result bitcasts stay real MOVs because the outputs are
// declared .f32 while the value comes from add.s32.  Passing c3 in as an
// operand and declaring the outputs u32 removes both.
CUTLASS_DEVICE float apxinf_ex2_c3_reg() {
  float c;
  asm("mov.f32 %0, 0f3D9DF09D;" : "=f"(c));
  return c;
}

CUTLASS_DEVICE void apxinf_ex2_emulation_2_c3(float& x, float& y, float c3) {
  uint32_t ox, oy;
  asm("{\n\t"
      ".reg .f32 xc, yc, xr, yr, xrb, yrb, xf, yf, px, py;\n\t"
      ".reg .s32 xri, yri, pxi, pyi, xre, yre;\n\t"
      "max.f32    xc, %2, 0fC2FE0000;\n\t"
      "max.f32    yc, %3, 0fC2FE0000;\n\t"
      "add.rm.f32 xr, xc, 0f4B400000;\n\t"
      "add.rm.f32 yr, yc, 0f4B400000;\n\t"
      "sub.rn.f32 xrb, xr, 0f4B400000;\n\t"
      "sub.rn.f32 yrb, yr, 0f4B400000;\n\t"
      "sub.rn.f32 xf, xc, xrb;\n\t"
      "sub.rn.f32 yf, yc, yrb;\n\t"
      "fma.rn.f32 px, xf, %4, 0f3E6906A4;\n\t"
      "fma.rn.f32 py, yf, %4, 0f3E6906A4;\n\t"
      "fma.rn.f32 px, px, xf, 0f3F31F519;\n\t"
      "fma.rn.f32 py, py, yf, 0f3F31F519;\n\t"
      "fma.rn.f32 px, px, xf, 0f3F800000;\n\t"
      "fma.rn.f32 py, py, yf, 0f3F800000;\n\t"
      "mov.b32 xri, xr;  mov.b32 yri, yr;\n\t"
      "mov.b32 pxi, px;  mov.b32 pyi, py;\n\t"
      "shl.b32 xre, xri, 23;  shl.b32 yre, yri, 23;\n\t"
      "add.s32 %0, xre, pxi;  add.s32 %1, yre, pyi;\n\t"
      "}\n"
      : "=r"(ox), "=r"(oy) : "f"(x), "f"(y), "f"(c3));
  x = __int_as_float(int(ox));
  y = __int_as_float(int(oy));
}

// APXINF_EX2_HOIST_C3: materialise the cubic's leading coefficient once per
// fragment through opaque asm so NVVM cannot rematerialise it at every use.
// Measured no-op on instruction count; kept because it was ablated.
#ifndef APXINF_EX2_HOIST_C3
#define APXINF_EX2_HOIST_C3 0
#endif

// FA4's `Softmax.apply_exp2_convert` picks hardware vs emulated exp2 per
// element pair, from the pair's position in the 32-element fragment.
// The schedule comes from `flash_fwd_sm100.py::_TUNING_CONFIG`, keyed on
// (is_persistent, is_causal, head_dim, is_fp8).  SASS-verified per operator:
//   vision   hd64             16 / 4 / 1  (32 poly + 225 MUFU.EX2)
//   language hd256 2-CTA      14 / 6 / 0  (36 poly +  94 MUFU.EX2)
//   action   hd256 pack-GQA   emulation DISABLED (0 poly + 257 MUFU.EX2)
// UPSTREAM has no emulation at all, i.e. RES = 0.
#ifndef APXINF_EX2_EMU_RES
#define APXINF_EX2_EMU_RES 0
#endif
#ifndef APXINF_EX2_EMU_FREQ
#define APXINF_EX2_EMU_FREQ 16
#endif
#ifndef APXINF_EX2_EMU_START_FRG
#define APXINF_EX2_EMU_START_FRG 1
#endif
CUTLASS_DEVICE constexpr bool apxinf_use_hw_ex2(int i, int frg_cnt) {
  constexpr int kEmuFreq = APXINF_EX2_EMU_FREQ;
  constexpr int kEmuRes = APXINF_EX2_EMU_RES;
  constexpr int kEmuStartFrg = APXINF_EX2_EMU_START_FRG;
  if (kEmuRes <= 0) return true;  // emulation disabled (UPSTREAM)
  int j = i / 32;
  int k = i % 32;
  return (k % kEmuFreq < kEmuFreq - kEmuRes) || (j >= frg_cnt - 1) ||
         (j < kEmuStartFrg);
}

// --- final O normalisation --------------------------------------------------
// FOUR distinct spellings, all three forks disagreed about the default:
//   0  UPSTREAM    scale_output / row_sum                  (div.rn, no guard)
//   1  vision      scale_output * rcp.approx(row_sum)      (no guard)
//   2  action      scale_output * rcp.approx(0/NaN ? 1 : row_sum)
//                  (flash_fwd_sm100.py:2752)
//   3  language    0/NaN ? 0.0f : scale_output / row_sum
//                  (sm100_hd256_2cta::correction_epilog)
#define APXINF_FINAL_NORM_DIV        0
#define APXINF_FINAL_NORM_RCP        1
#define APXINF_FINAL_NORM_RCP_GUARD  2
#define APXINF_FINAL_NORM_DIV_GUARD  3
#ifndef APXINF_FINAL_NORM
#define APXINF_FINAL_NORM APXINF_FINAL_NORM_DIV
#endif
CUTLASS_DEVICE float apxinf_normalize_scale(float scale_output, float row_sum) {
#if APXINF_FINAL_NORM == APXINF_FINAL_NORM_RCP
  return scale_output * apxinf_rcp_approx(row_sum);
#elif APXINF_FINAL_NORM == APXINF_FINAL_NORM_RCP_GUARD
  bool zero_or_nan = (row_sum == 0.0f) || (row_sum != row_sum);
  return scale_output * apxinf_rcp_approx(zero_or_nan ? 1.0f : row_sum);
#elif APXINF_FINAL_NORM == APXINF_FINAL_NORM_DIV_GUARD
  bool zero_or_nan = (row_sum == 0.0f) || (row_sum != row_sum);
  return zero_or_nan ? 0.0f : scale_output / row_sum;
#else
  return scale_output / row_sum;   // UPSTREAM
#endif
}

// --- the log-sum-exp the epilogue writes to ptr_LSE -------------------------
// UPSTREAM: `cutlass::fast_log(row_sum) + scale_softmax * row_max`, i.e.
// libdevice `::logf` (a correctly-rounded polynomial behind a CALL) plus two
// separate roundings.
//
// FA4 (`flash_fwd_sm100.py:2844-2847`):
//     LN2 = math.log(2.0)
//     lse = (row_max * softmax_scale_log2_eff
//            + (cute.math.log2(row_sum, fastmath=True) - max_offset)) * LN2
// with `max_offset = 0` for every non-8-bit dtype (`:2627`), so the
// subtraction folds away.  `cute.math.log2(..., fastmath=True)` is the
// approximate hardware log, and the ADD IS FUSED INTO THE MULTIPLY.  All
// three facts are read off the AOT object's own SASS
// (`sass/apxinf_fa4_d256_a_splitbatch_sm110.sass`, the ONLY MUFU.LG2 in the
// kernel):
//     /*a5e0*/ MUFU.LG2 R48, R48 ;
//     /*a690*/ FFMA     R65, R65, UR10, R48 ;
//     /*c030*/ FMUL     R0,  R65, 0.69314718246459960938 ;
// `0.69314718246459960938` is `float(math.log(2.0))` = 0x3F317218, and UR10 is
// `scale_softmax_log2` (host-side `softmax_scale * log2(e)`; for this geometry
// `softmax_scale = 0.0625`, a power of two, so the host rounding of
// `log2(e) * 0.0625` is identical in Python's float64 and in this header's
// `float` -- scaling by 2^-4 is exact).
//
// `.ftz` on the `lg2` CANNOT matter numerically here: `row_sum` is a sum of
// `ex2` values of which at least one is >= 1, and the only fp32 input whose
// `log2` is subnormal is within 2^-126 of 1.0.  What it changes is the
// instruction sequence -- without it ptxas wraps the MUFU in the usual
// subnormal guard
//     FSETP.GEU P1, |x|, 1.1754944e-38 ; @!P1 FMUL x, x, 16777216 ;
//     MUFU.LG2 ; @!P1 FADD r, r, -24
// which the AOT SASS does NOT have, so FA4's DSL lowering is the `.ftz` one.
// Both spellings were run: 1600/1600 bit-equal LSE at all five action shapes
// either way (report 27 S5), and `APXINF_LSE_LG2_FTZ=1` additionally drops the
// three predicated instructions and reproduces the AOT's exact three-op
// sequence.  Same question `apxinf_ex2_approx` answers, same answer -- and for
// the record the opposite of what `apxinf_ex2_emulation_2` needs.
//
// Why this is not cosmetic: the LSE does not feed O, so upstream's 1-3 fp32
// ULP of difference is invisible in every operator that does not emit it.  The
// action operator emits it, and the split-batch merge weight is
// `sigmoid(lse0 - lse1)`.  At `key_tokens = 3436` the two batches have equal
// KV length, `lse0 - lse1` collapses from ~6e-4 to ~2e-5 (median) with 92 of
// 800 rows below 1e-5 and some exactly 0, so a 1e-6 LSE error is a 5-100%
// error in the weight, and for the ~90 output elements per shape where
// `p1 == -p0` exactly it is a sign flip: 12,112 BF16 ULP at an absolute error
// of 1.9e-10.  Report 27.
#ifndef APXINF_LSE_FA4
#define APXINF_LSE_FA4 0
#endif
#ifndef APXINF_LSE_LG2_FTZ
#define APXINF_LSE_LG2_FTZ 1
#endif

CUTLASS_DEVICE float apxinf_lg2_approx(float x) {
  float r;
#if APXINF_LSE_LG2_FTZ
  asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
#else
  asm("lg2.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
#endif
  return r;
}

CUTLASS_DEVICE float apxinf_lse(float row_sum, float row_max,
                                float scale_softmax,
                                float scale_softmax_log2) {
#if APXINF_LSE_FA4
  return __fmaf_rn(row_max, scale_softmax_log2, apxinf_lg2_approx(row_sum)) *
         0.69314718246459960938f;
#else
  return cutlass::fast_log(row_sum) + scale_softmax * row_max;  // UPSTREAM
#endif
}

// --- online softmax numerics ------------------------------------------------
// APXINF_ACC_SCALE_HALF -- *** TRAP ***  the upstream default is 1, NOT 0.
//   1 (UPSTREAM, language): acc_scale = 0.5f * ex2(scale*(old-new)) and the
//     carried row_sum is seeded into BOTH lanes of the packed accumulator
//     (the 0.5 is what cancels the double count).
//   0 (vision, action): FA4 utils.py::fadd_reduce seeds lane 0 ONLY and
//     therefore uses no extra factor.
// The 1-Q-stage fork defaulted this to 0, which is the *action* contract;
// taking that default silently changes upstream.
#ifndef APXINF_ACC_SCALE_HALF
#define APXINF_ACC_SCALE_HALF 1
#endif

// APXINF_ROWSUM_QUARTERS
//   0 (UPSTREAM, vision, action) four stride-8 packed accumulators
//   1 (language) four CONTIGUOUS QUARTERS
// Upstream is ACC_SCALE_HALF=1 with ROWSUM_QUARTERS=0; language needs both 1.
#ifndef APXINF_ROWSUM_QUARTERS
#define APXINF_ROWSUM_QUARTERS 0
#endif

// APXINF_FUSE_ROWSUM
//   0 (UPSTREAM) row_sum reduced in a second pass after the exp2/convert loop
//   1 fused into the exp2/convert loop.  The four accumulators and their
//     visiting order are unchanged so this is bit-identical, but it IS a
//     reordering, so it is a knob and not unconditional (the 1-Q-stage fork
//     made it unconditional, which silently applied it to the 2-stage path).
//     Required for hd256: keeping 128 S floats live past the P pack pushed
//     the softmax warps to ~192 live registers and ~700 spill instructions.
#ifndef APXINF_FUSE_ROWSUM
#define APXINF_FUSE_ROWSUM 0
#endif
#if APXINF_ROWSUM_QUARTERS && !APXINF_FUSE_ROWSUM
#  error "APXINF_ROWSUM_QUARTERS=1 requires APXINF_FUSE_ROWSUM=1"
#endif

// APXINF_RESCALE_THRESHOLD -- FA4's lazy rescale (softmax.py::
// update_row_max_from_local, rescale_threshold).  If the running row max grew
// by less than threshold/scale_log2 the accumulator is NOT rescaled and the
// OLD max is kept; P may then exceed 1 (up to 2^threshold), which bf16
// tolerates.  UPSTREAM always rescales, i.e. threshold 0.
//   vision/action 8.0f ; language 0.0f (SASS: zero `FSETP.GE ..,-8`).
#ifndef APXINF_RESCALE_THRESHOLD
#define APXINF_RESCALE_THRESHOLD 0.0f
#endif

// APXINF_KV_DESCENDING
//   0 (UPSTREAM, language) ascending: unmasked tiles first, masked LAST
//     (sm100_hd256_2cta_fmha_forward.py:1449)
//   1 (vision, action) descending [n-1, n-2, ..., 0], masked tile FIRST
//     (flash_fwd_sm100.py:2355).  Must match the fetch order in
//     sm100_fmha_load_tma_warpspecialized.hpp.
#ifndef APXINF_KV_DESCENDING
#define APXINF_KV_DESCENDING 0
#endif

// APXINF_BIAS_NEG_FIRST  0 = UPSTREAM `-(row_max_safe * scale)`
//                        1 = FA4      `(0.0f - row_max_safe) * scale`
#ifndef APXINF_BIAS_NEG_FIRST
#define APXINF_BIAS_NEG_FIRST 0
#endif
// APXINF_CORR_UNCOND     0 = UPSTREAM, skip the O rescale when scale == 1.0f
//                        1 = FA4, rescale unconditionally
#ifndef APXINF_CORR_UNCOND
#define APXINF_CORR_UNCOND 0
#endif
// APXINF_FMAX3           0 = UPSTREAM 2-input max tree
//                        1 = FA4 utils.py::fmax_reduce 3-input max tree (exact)
#ifndef APXINF_FMAX3
#define APXINF_FMAX3 0
#endif
// APXINF_RELEASE_PIPE_COUNT  upstream 10, must be a multiple of 2
#ifndef APXINF_RELEASE_PIPE_COUNT
#define APXINF_RELEASE_PIPE_COUNT 10
#endif
// APXINF_P_FLUSH_AT  element index at which P chunk 0 is flushed to tmem.
//   UPSTREAM size-6 (too late to free anything for hd256);
//   hd256 wants size/2, which is where chunk 0 is first complete.
#ifndef APXINF_P_FLUSH_HALF
#define APXINF_P_FLUSH_HALF 0
#endif
#ifndef APXINF_P_FLUSH_AT
#if APXINF_P_FLUSH_HALF
#define APXINF_P_FLUSH_AT (decltype(size(tTMEM_LOADrS))::value / 2)
#else
#define APXINF_P_FLUSH_AT (decltype(size(tTMEM_LOADrS))::value - 6)
#endif
#endif
// APXINF_LATE_C_ACQUIRE  0 = UPSTREAM.  1 acquires the stats (V) tmem slot
//   immediately before the write instead of at the tail of the previous step.
//   Scheduling only; V contents identical.  Two-Q-stage path only.
#ifndef APXINF_LATE_C_ACQUIRE
#define APXINF_LATE_C_ACQUIRE 0
#endif
// APXINF_NO_ORDER_S  0 = UPSTREAM.  1 drops the softmax ping-pong ordering
//   barrier entirely (the single-Q-stage path drops it automatically because
//   there is no partner warpgroup).
//   🔴 NOT BIT-NEUTRAL on the two-Q-stage path, despite FA4 setting
//   `s0_s1_barrier = False` and despite report 21 §①.4 adopting `=1` as a
//   2.3% win on the strength of `100.000000%` at both vision lengths.  That
//   verification used the `±0.05` uniform stimulus, where the operator
//   degenerates to `mean(V)` (report 31 §2.3).  On a model-shaped
//   (heavy-tailed) stimulus `=1` differs from the FA4 AOT object by exactly
//   one BF16 ULP on ~3 of every 1e8 output elements, in BOTH vision AOT
//   objects, and on the real captured `vis01.qkv` tensor; `=0` is bit-exact on
//   all of them.  Report 32 §1.1.  The two softmax warpgroups share no data
//   (separate S/V/O tmem slots, separate mma_s and s_corr pipelines), so the
//   hazard is an ordering one and the barrier is load-bearing.  Do not set
//   this to 1 for a performance win without re-running
//   `aot_and_cuda_vision_attention_are_bit_identical_on_a_model_shaped_stimulus`.
#ifndef APXINF_NO_ORDER_S
#define APXINF_NO_ORDER_S 0
#endif
// APXINF_P_CHUNKS16  0 = UPSTREAM, P is stored to tmem as TWO 32-column
//   chunks (SM100_TMEM_STORE_32dp32b32x).  1 = FA4 parity,
//   `flash_fwd_sm100.py:2102-2108` uses `St32x32bOp(Repetition(16))`, i.e.
//   FOUR 16-column chunks, each retired as soon as it is complete.  Store
//   granularity only -- the stored bits and their order are identical, so this
//   is bit-neutral (verified: vision 624/2200 100.000000% with and without).
//   Worth 0.8% on vision-624 (report 21 §1.3).
#ifndef APXINF_P_CHUNKS16
#define APXINF_P_CHUNKS16 0
#endif
// APXINF_SKIP_PAD_QSTAGE  0 = UPSTREAM.  1 routes a Q stage whose every row is
//   out of range to a handshake-only `softmax_step_all_padding()`.
//   Only reachable with more than one Q stage per tile (TileShape M = 256 with
//   ThreadShape<2,1,1>): the kernel skips WHOLE tiles, so at e.g. L=624 the
//   last tile's stage 1 (rows 640..767) has zero valid rows and still pays
//   full price.  Worth 5.3% on vision-624, exactly 0 at L=2200 (whose stage 1
//   has 24 live rows).  Report 21 §1.2.
#ifndef APXINF_SKIP_PAD_QSTAGE
#define APXINF_SKIP_PAD_QSTAGE 0
#endif
// APXINF_CORR_RING2  0 = UPSTREAM, stage `D / kCorrectionTileSize` chunks of O
//   (upstream hardcodes the numerator 128; see the derivation below).
//   1 = stage exactly two and alternate.  Numerically identical (same values,
//   same order) but a large register-allocation change, so it is a knob.
//   REQUIRED for D > 128, where the upstream sizing indexes out of bounds.
//   Both head_dim-256 operators set it.  Vision does NOT: report 30 measured
//   1.0199 with it against a [1.0164, 1.0190] control band (16 rounds, repo
//   path) at identical REG/STACK and identical SASS instruction COUNT -- which
//   is also the demonstration that the instruction count is not an identity
//   test for the generated code.
#ifndef APXINF_CORR_RING2
#define APXINF_CORR_RING2 0
#endif
// APXINF_CORR_TILE  the column tile `correction_rescale` processes per
//   TMEM_LOAD / FMUL2 / TMEM_STORE round trip.  0 = UPSTREAM (16).  The loop
//   runs `D / APXINF_CORR_TILE` times, so this trades round count against the
//   live register footprint of one chunk.  Bit-neutral (same values, same
//   order; verified: vision 12509184/12509184 absmax 0 at 8 and at 16), and
//   8 needs the 8-column TMEM atoms, hence the `conditional_t` below.
//   Must divide `size<2>(TileShape)`; 8 and 16 are the values measured.
//   MEASURED (report 30, repo path, 16 rounds, ABBA, control band
//   [1.0164, 1.0190] over six replicates of identical code):
//     8               1.0186  -- neutral, NOT adopted
//     8 + CORR_RING2  1.0218  -- worse
//     16 (upstream)   1.0183/1.0187/1.0188/1.0190 (controls)
//   The vision fork measured 8 as a ~5% WIN on its standalone L=624 driver
//   (report 08 #13).  That did not transfer: through the repo path the 624
//   group is already at 1.004 and the whole gap is the 2200 group, which this
//   knob does not touch.  Re-sweep before trusting either verdict.
#ifndef APXINF_CORR_TILE
#define APXINF_CORR_TILE 0
#endif
// APXINF_DEBUG_SHOW_SO  upstream unconditionally `print(sO)` in the
//   degenerate (empty) epilogue.  That single statement is worth 60
//   CALL.ABS.NOINC/vprintf sites in the kernel image -- it is the whole of the
//   "CALL.ABS 60 -> 0" delta report 18 6.2 attributed to div.rn -> rcp.approx.
//   Default 1 keeps upstream behaviour bit-for-bit; every AOT operator sets 0.
#ifndef APXINF_DEBUG_SHOW_SO
#define APXINF_DEBUG_SHOW_SO 1
#endif
// APXINF_FMHA_KV_STAGES  0 = UPSTREAM formula (fp8 ? 4 : 3)
//   MEASURED (report 30, vision hd64, repo path, 16 rounds, ABBA, control band
//   [1.0164, 1.0190] over six replicates of the shipped 4):
//     3 -> 1.0210   4 -> controls (shipped)   5 -> 1.0234   6 -> 1.0244
//   The vision fork preferred 6 on its standalone L=624 driver (report 08 #10)
//   and then re-measured 3/4/6 as 1.122/1.129/1.123 there, i.e. 4 WORST.
//   Through the repo path 4 is the best of the four -- another sign flip; do
//   not carry either verdict across a regime change without re-sweeping.
#ifndef APXINF_FMHA_KV_STAGES
#define APXINF_FMHA_KV_STAGES 0
#endif
// APXINF_S_PIPE_STAGES  depth of the mma->softmax S pipeline.  Only read on
//   the single-Q-stage path; the two-Q-stage path always has one S slot per
//   softmax warpgroup, i.e. PipelineS/PipelineC depth 1 as upstream.
#ifndef APXINF_S_PIPE_STAGES
#define APXINF_S_PIPE_STAGES 1
#endif

// 🔴 APXINF_ROWSUM_RESCALE_RN -- operator-specific AOT arithmetic.
//    0 = upstream, contractable packed add.  1 = keep the carried-row-sum
//    multiply separate by spelling it as `mul.rn.f32`.
//
//    `cute::add(float2&,...)` (include/cute/arch/simd_sm100.hpp:45) spells
//    `add.f32x2` with NO rounding modifier, and the PTX ISA then licenses
//    ptxas to contract a feeding multiply into it: "mul/add sequences with no
//    rounding modifiers may be optimized to use fused-multiply-add".
//    `asm volatile` constrains motion, not contraction.  With the plain `*=`
//    ptxas DID contract -- the language SASS had
//        FFMA R105, R2, R140, R100     // fma(acc_scale, row_sum, P[0])
//    where the AOT's own PTX has two separately rounded operations
//        mul.f32 %r2101, %r3062, %r2100 ; add.rn.f32x2 %rd938, {%r2101,%r2101}, ..
//    (`sm100_hd256_2cta_fmha_forward.py:1772-1773` lowered through
//    `cute.arch.add_packed_f32x2`, which emits the `.rn` spelling).
//
//    One fused rounding instead of two moved `row_sum` by 1-4 fp32 ULP and was
//    the ENTIRE `aot-language-attention` 31/13873152 residual: setting 1 took
//    the strict gate to 13873152/13873152 **and** report 34's fp32-LSE
//    instrument from 904/54192 to 0/54192, on 4 seeds x 7 lengths and on both
//    exponent stimulus modes (report 35 §3).
//
//    This is not a universal "CuTe DSL" setting. The action AOT's own SASS
//    contracts `row_sum * acc_scale + first_exp` into FFMA, so action must set
//    0. On a captured Qwen-Drive tensor, setting 1 made 9/1600 row sums differ
//    and left one LSE mismatch; setting 0 gives 1600/1600 direct bit equality.
//    Language must set 1 as above. Vision independently pins 1, which its
//    model-shaped and captured-input gates have verified bit-exact.
//
//    🔴 The default must stay 0: at zero `-DAPXINF_*` this header has to
//    reproduce pristine upstream CUTLASS byte for byte, and upstream contracts.
//    Measured: flipping this knob alone moves 7 of the 25 upstream-default
//    equivalence files -- and EXACTLY the seven `.lse` ones, i.e. only
//    `row_sum`, never `O` (report 35 §5.1).  That is also the cleanest
//    independent proof of the mechanism, in pristine CUTLASS with no AOT
//    anywhere in the picture.
#ifndef APXINF_ROWSUM_RESCALE_RN
#define APXINF_ROWSUM_RESCALE_RN 0
#endif
#if APXINF_ROWSUM_RESCALE_RN
#define APXINF_ROWSUM_RESCALE(rs, sc)                                       \
  asm volatile("mul.rn.f32 %0, %0, %1;\n" : "+f"(rs) : "f"(sc))
#else
#define APXINF_ROWSUM_RESCALE(rs, sc) ((rs) *= (sc))
#endif

// No multiply feeds the remaining packed additions, so their ordinary
// `add.f32x2` cannot contract.  Keeping them outside inline asm gives ptxas
// the same scheduling freedom as the upstream path while preserving the AOT
// arithmetic above.
#define APXINF_ROWSUM_ADD(c, a, b) cute::add((c), (a), (b))

// Carried-row_sum rescale + packed-accumulator seeding, shared by the fused
// and the upstream two-pass row_sum reductions.  Expanded inside
// softmax_step(), where skip_rescale / old_row_max / row_max_safe /
// acc_scale_log2 are in scope.
//
// UPSTREAM is `(old == safe) ? 0.5f : 0.5f * exp2f(scale*(old-safe))` with the
// carried row_sum in BOTH lanes.  `0.5f * (cond ? 1.0f : ex2(..))` is the same
// value bit for bit (ex2(0) == 1.0f, and scaling by 0.5f is exact).
#if APXINF_ACC_SCALE_HALF
#define APXINF_ACC_SCALE_SEED()                                                \
    ElementQK acc_scale =                                                      \
        0.5f * ((skip_rescale || old_row_max == row_max_safe)                  \
                    ? 1.0f                                                     \
                    : apxinf_ex2_approx(scale * (old_row_max - row_max_safe)));\
    APXINF_ROWSUM_RESCALE(row_sum, acc_scale);                                 \
    float2 local_row_sum_f32x2 = make_float2(row_sum, row_sum);                \
    float2 local_row_sum_1 = make_float2(0, 0);                                \
    float2 local_row_sum_2 = make_float2(0, 0);                                \
    float2 local_row_sum_3 = make_float2(0, 0)
#else
#define APXINF_ACC_SCALE_SEED()                                                \
    ElementQK acc_scale =                                                      \
        (skip_rescale || old_row_max == row_max_safe)                          \
            ? 1.0f                                                             \
            : apxinf_ex2_approx(scale * (old_row_max - row_max_safe));         \
    APXINF_ROWSUM_RESCALE(row_sum, acc_scale);                                 \
    float2 local_row_sum_f32x2 = make_float2(row_sum, 0.0f);                   \
    float2 local_row_sum_1 = make_float2(0, 0);                                \
    float2 local_row_sum_2 = make_float2(0, 0);                                \
    float2 local_row_sum_3 = make_float2(0, 0)
#endif


namespace cutlass::fmha::collective {

using namespace cute;

template<
  class Element_,
  class ElementQK_,
  class ElementPV_,
  class TileShape_,
  class StrideQ_,
  class StrideK_,
  class StrideV_,
  class Mask_,
  // shape here is QG K H
  // and referes to the two softmax warps
  // (2, 1, 1) means that they are stacked (best for large Q since it loads the least K/V)
  // (1, 2, 1) means they sit side by side (best for small Q / large K)
  class ThreadShape = Shape<_2, _1, _1>,
  class ClusterShape_ = Shape<_1, _1, _1>,
  // Since shared memory is sufficient for FMHA, there is no need to reuse shared memory.
  class OrderLoadEpilogue = cute::false_type
>
struct Sm100FmhaFwdMainloopTmaWarpspecialized {

  using Element = Element_;
  using ElementQK = ElementQK_;
  using ElementPV = ElementPV_;
  using TileShape = TileShape_;
  using StrideQ = StrideQ_;
  using StrideK = StrideK_;
  using StrideV = StrideV_;
  using Mask = Mask_;

  // ---- apxinf: single-Q-stage support for head_dim 256 ---------------------
  // The stock collective hardwires two Q stages (ThreadShape<_2,_1,_1>): two
  // softmax warpgroups ping-pong over S0/S1 and two O accumulators. That costs
  // 2*N + 2*D tmem columns, which exceeds the 512-column SM100 budget for any N
  // once D = 256, and 2*(M*D) + 3*(N*D) bytes of smem, which exceeds the
  // 232448 B opt-in limit. FA4's hd256 kernel uses a single Q stage instead
  // (one softmax warpgroup, one O accumulator): N + D = 384 tmem columns and
  // (M + 2*N) * D * 2 B = 194.0 KiB of smem, both of which fit.
  //
  // ThreadShape<_1,_1,_1> selects that path.
  static constexpr int NumQStages = cute::size(ThreadShape{});
  static_assert(NumQStages == 1 || NumQStages == 2, "only 1 or 2 Q stages");
  static constexpr bool kSingleQStage = (NumQStages == 1);

  using ClusterShape = ClusterShape_;
  static constexpr bool kIs2Sm = cute::size(ClusterShape{}) == 2;
  static_assert(cute::size(ClusterShape{}) == 1 || kIs2Sm,
                "FMHA supports one- or two-CTA clusters");

  // The dedicated hd256 2-CTA kernel uses an M256N128K128 UMMA atom and
  // therefore stages the two K128 halves of Q independently.  NumQStages is
  // still one from the softmax/tmem point of view; this is only the smem/TMA
  // pipeline depth.
  static constexpr int StageCountQ = kIs2Sm ? 2 : NumQStages;

  // apxinf: depth of the mma -> softmax S pipeline. The stock collective has
  // one S slot per Q stage, which with a single Q stage means the MMA warp
  // cannot start QK(i+1) until softmax(i) has finished reading S(i) -- QK, the
  // softmax and PV end up strictly serialised and the MMA pipe idles for the
  // whole softmax. FA4's hd256 kernel uses `qk_acc_stage = 2`
  // (`sm100_hd256_2cta_fmha_forward.py::_setup_attributes`) with
  // `tmem_s_offset = 0 / tmem_o_offset = 256`, i.e. two 128-column S slots plus
  // the 256-column O accumulator = the full 512 tmem columns. Two S slots let
  // QK(i+1) be issued *before* PV(i), so it executes while softmax(i) runs.
  // apxinf: depth of ONE PipelineS/PipelineC object.  Upstream has one such
  // object per softmax warpgroup, each of depth 1; only the single-Q-stage
  // path (which has a single softmax warpgroup) can deepen it.
  static constexpr int NumSStages = kSingleQStage ? APXINF_S_PIPE_STAGES : 1;
  static_assert(NumSStages == 1 || NumSStages == 2, "only 1 or 2 S stages");
  static constexpr bool kMultiSStage = (NumSStages > 1);
  // Number of distinct 128-column S slots in tmem: two for the stock
  // ping-pong layout (S0/S1), and NumSStages for the single-Q-stage path.
  static constexpr int NumSSlots = kSingleQStage ? NumSStages : NumQStages;
  static constexpr int StageCountKV = kIs2Sm ? 4
      : (APXINF_FMHA_KV_STAGES > 0
          ? APXINF_FMHA_KV_STAGES
          : (sizeof(Element_) == 1 ? 4 : 3));

  using StagesQ = cutlass::gemm::collective::StageCount<StageCountQ>;
  using StagesKV = cutlass::gemm::collective::StageCount<StageCountKV>;

  using MmaSchedule = cute::conditional_t<
      kIs2Sm,
      cutlass::gemm::KernelTmaWarpSpecialized2SmSm100,
      cutlass::gemm::KernelTmaWarpSpecialized1SmSm100>;

  static const int Alignment = 128 / sizeof_bits_v<Element>;

  using TileShapeQKDefault = decltype(shape_div(TileShape{}, ThreadShape{}));
  using TileShapeQK = std::conditional_t<
      kIs2Sm,
      decltype(make_shape(get<0>(TileShapeQKDefault{}),
                          get<1>(TileShapeQKDefault{}), _128{})),
      TileShapeQKDefault>;

  // TileShapePV is the full logical O tile used by correction/epilogue.  The
  // 2-CTA MMA itself emits it as two adjacent N128 instructions.
  using TileShapePV = std::conditional_t<
      kIs2Sm,
      decltype(make_shape(get<0>(TileShapeQK{}), get<2>(TileShape{}),
                          get<1>(TileShapeQK{}))),
      decltype(select<0,2,1>(TileShapeQK{}))>;
  using TileShapePVMma = std::conditional_t<
      kIs2Sm,
      decltype(make_shape(get<0>(TileShapePV{}), _128{},
                          get<2>(TileShapePV{}))),
      TileShapePV>;

  using CollectiveMmaQK = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      Element, StrideQ, Alignment,
      Element, StrideK, Alignment,
      ElementQK,
      TileShapeQK, ClusterShape, cutlass::gemm::collective::StageCount<3> /* we change it later anyways*/,
      MmaSchedule>::CollectiveOp;

  using CollectiveMmaPV = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      // the stride for A does not matter since we do not load from smem at all
      Element, StrideK, Alignment,
      Element, decltype(select<1,0,2>(StrideV{})), Alignment,
      ElementPV,
      TileShapePV, ClusterShape, cutlass::gemm::collective::StageCount<3> /* we change it later anyways*/,
      MmaSchedule>::CollectiveOp;

  // FA4's hd256 2-CTA kernel keeps the full N256 V/TMA layout, but issues PV
  // as two M256N128 UMMA instructions.  A separate compute collective gives
  // us that instruction shape while CollectiveMmaPV remains the full-tile
  // layout used by the loader and correction epilogue.
  using CollectiveMmaPVCompute = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      Element, StrideK, Alignment,
      Element, decltype(select<1,0,2>(StrideV{})), Alignment,
      ElementPV,
      TileShapePVMma, ClusterShape,
      cutlass::gemm::collective::StageCount<3>, MmaSchedule>::CollectiveOp;

  using CtaShapeQK = typename CollectiveMmaQK::CtaShape_MNK;
  using CtaShapePV = typename CollectiveMmaPV::CtaShape_MNK;

  CUTLASS_DEVICE static uint32_t cta_mma_rank() {
    return cute::block_rank_in_cluster() %
           cute::size(typename CollectiveMmaQK::TiledMma::AtomThrID{});
  }

  template<class TensorStorageEpi>
  CUTLASS_DEVICE static auto make_epilogue_smem_tensor(TensorStorageEpi& storage) {
    using LogicalLayout = typename TensorStorageEpi::SmemLayoutO;
    return make_tensor(make_smem_ptr(storage.smem_o.data()), LogicalLayout{});
  }

  using SmemLayoutQ = decltype(unstageSmemLayout(typename CollectiveMmaQK::SmemLayoutA{}, Int<StageCountQ>{}));
  using SmemLayoutK = decltype(unstageSmemLayout(typename CollectiveMmaQK::SmemLayoutB{}, Int<StageCountKV>{}));
  using SmemLayoutV = decltype(unstageSmemLayout(
      typename CollectiveMmaPVCompute::SmemLayoutB{}, Int<StageCountKV>{}));

  // Reuse shared memory for V and O.
  static constexpr bool IsOrderLoadEpilogue = std::is_same_v<OrderLoadEpilogue, cute::true_type>;
  struct TensorStorage {
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
    union {
      cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k;
      cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> smem_v;
    };
  };

  // tmem footprint, derived rather than hardcoded. The stock code used
  // kSizeS = kSizeO = 128 which silently aliases O0/O1 once D > 128.
  static constexpr uint32_t kTmemSizeS = size<1>(TileShapeQK{});   // N columns of f32
  static constexpr uint32_t kTmemSizeO = size<2>(TileShape{});     // D columns of f32
  static constexpr uint32_t kTmemNumStages = NumSSlots;
  static constexpr uint32_t kPackedPColumns =
      size<1>(TileShapeQK{}) / 4 * sizeof(Element_);
  // The dedicated hd256 2CTA kernel stores packed P at the start of each S
  // slot and puts its two fp32 statistics immediately after P.  The stock
  // 1CTA path uses the opposite arrangement (stats at 0, P at column 32).
  // For 2CTA TS MMA, P must start at column zero so the duplicated TMEM-A
  // allocation reaches both N/2 halves of the instruction.
  static constexpr uint32_t kPBaseOffset = kIs2Sm ? 0 : 32;
  static constexpr uint32_t kStatsOffset = kIs2Sm ? kPackedPColumns : 0;

  enum class TmemAllocation : uint32_t {
    kSizeS = kTmemSizeS,
    kSizeO = kTmemSizeO,
    kSizeP = 32,
    S0 = 0,
    S1 = (kTmemNumStages - 1) * kTmemSizeS,
    V0 = S0 + kStatsOffset,  // stats storage from softmax to correction
    V1 = S1 + kStatsOffset,
    P0 = S0 + kPBaseOffset,
    P1 = S1 + kPBaseOffset,
    O0 = kTmemNumStages * kTmemSizeS,
    O1 = O0 + (NumQStages - 1) * kTmemSizeO,
    kEnd = O1 + kSizeO
  };

  static constexpr uint32_t kTmemColumns = uint32_t(TmemAllocation::kEnd);
  static_assert(kTmemColumns <= 512,
      "FMHA tmem footprint exceeds the 512-column SM100 budget; "
      "use ThreadShape<_1,_1,_1> (one Q stage) for large head_dim");
  static_assert(kPBaseOffset + kPackedPColumns <= kTmemSizeS,
      "P does not fit inside the S tmem slot");
  static_assert(kStatsOffset + 2 <= kTmemSizeS,
      "softmax statistics do not fit inside the S tmem slot");

  // indices for V0 / V1
  enum : int {
    kIdxOldRowMax = 0,
    kIdxNewRowMax = 1,
    kIdxFinalRowSum = 0,
    kIdxFinalRowMax = 1
  };

  // from load to mma warp, protects q in smem
  using PipelineQ = cutlass::PipelineTmaUmmaAsync<
    StageCountQ,
    ClusterShape,
    typename CollectiveMmaQK::AtomThrShapeMNK
  >;

  // from load to mma warp, protects k/v in smem
  using PipelineKV = cutlass::PipelineTmaUmmaAsync<
    StageCountKV,
    ClusterShape,
    typename CollectiveMmaQK::AtomThrShapeMNK
  >;

  // from mma to softmax0/1 warp, protects S in tmem
  // (not sure yet about the reverse direction)
  // there is one pipe per softmax warp, and the mma warp alternates between them
  using PipelineS = cutlass::PipelineUmmaAsync<
      NumSStages, typename CollectiveMmaQK::AtomThrShapeMNK>;

  // from softmax0/1/ to correction wg
  using PipelineC = cutlass::PipelineAsync<NumSStages>;

  // from mma to correction
  using PipelineO = cutlass::PipelineUmmaAsync<
      NumQStages, typename CollectiveMmaPV::AtomThrShapeMNK>;

  // from corr to epilogue
  using PipelineE = cutlass::PipelineAsync<2>;

  using OrderBarrierSoftmax = cutlass::OrderedSequenceBarrier<
    /*stages*/ 1, /*groups*/ 2>;

  static const int TransactionBytesLoadQ = cutlass::bits_to_bytes(
      cosize(take<0,3>(SmemLayoutQ{})) * cute::sizeof_bits_v<Element>) *
      cute::size(typename CollectiveMmaQK::TiledMma::AtomThrID{});

  static const int TransactionBytesLoadK = cutlass::bits_to_bytes(
      cosize(take<0,3>(SmemLayoutK{})) * cute::sizeof_bits_v<Element>) *
      cute::size(typename CollectiveMmaQK::TiledMma::AtomThrID{});
  static const int TransactionBytesLoadV = cutlass::bits_to_bytes(
      cosize(take<0,3>(SmemLayoutV{})) * cute::sizeof_bits_v<Element>) *
      cute::size(typename CollectiveMmaQK::TiledMma::AtomThrID{});

  static_assert(TransactionBytesLoadK == TransactionBytesLoadV, "K and V smem layouts must be of equal size");

  using Load = Sm100FmhaLoadTmaWarpspecialized<
    Element, StrideQ, StrideK, StrideV,
    CollectiveMmaQK, CollectiveMmaPVCompute,
    SmemLayoutQ, SmemLayoutK, SmemLayoutV,
    TensorStorage, PipelineQ, PipelineKV, Mask, TileShape, ClusterShape
  >;

  struct Arguments {
    typename Load::Arguments load;

    // if zero, defaults to 1/sqrt(D)
    float scale_softmax = 0.0f;

    // scaling factors to dequantize QKV
    float scale_q = 1.0f;
    float scale_k = 1.0f;
    float scale_v = 1.0f;

    // scaling factor to quantize O
    float inv_scale_o = 1.0f;
  };

  struct Params {
    typename Load::Params load;

    float scale_softmax;
    float scale_softmax_log2;

    float scale_output;
  };

  template<class ProblemShape>
  static bool can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    return true;
  }

  template<class ProblemShape>
  static Params to_underlying_arguments(
      ProblemShape const& problem_shape,
      Arguments const& args,
      void* workspace) {

    float scale_softmax = args.scale_softmax;
    if (scale_softmax == 0.0f) {
      scale_softmax = 1.0f / std::sqrt(float(get<2>(problem_shape)));
    }
    float log2_e = static_cast<float>(std::log2(std::exp(1.0)));

    return Params{
        Load::to_underlying_arguments(problem_shape, args.load, workspace),
        args.scale_q * args.scale_k * scale_softmax,
        args.scale_q * args.scale_k * log2_e * scale_softmax,
        args.scale_v * args.inv_scale_o
    };
  }

  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& params) {
      Load::prefetch_tma_descriptors(params.load);
  }

  template<class BlkCoord, class ProblemShape, class ParamsProblemShape>
  CUTLASS_DEVICE void
  load(
      BlkCoord const& blk_coord, ProblemShape const& problem_shape,
      Params const& params, ParamsProblemShape const& params_problem_shape,
      TensorStorage& storage,
      PipelineQ& pipeline_q, typename PipelineQ::PipelineState& pipeline_q_producer_state,
      PipelineKV& pipeline_kv, typename PipelineKV::PipelineState& pipeline_kv_producer_state) {

    Load load;
    load.load(blk_coord, problem_shape, params.load, params_problem_shape,
        storage,
        pipeline_q, pipeline_q_producer_state,
        pipeline_kv, pipeline_kv_producer_state);
  }

  // --- apxinf: single-Q-stage mainloop ---------------------------------------
  // One S slot and one O accumulator. The MMA warp issues, per KV tile i:
  //   QK(Q, K_i) -> S0   then   PV(P0, V_i) -> O0
  // and the correction warpgroup rescales O0 in between PV(i-1) and PV(i),
  // which is what serialises the two. The S slot is re-acquired before each PV
  // (that hands P0, which overlaps S0, back to the MMA warp) exactly as the
  // two-stage path does.
  template<class BlkCoord, class ProblemShape>
  CUTLASS_DEVICE void
  mma_single_q(
      BlkCoord const& blk_coord,
      Params const& params, ProblemShape const& problem_shape,
      TensorStorage& storage,
      PipelineQ& pipeline_q, typename PipelineQ::PipelineState& pipeline_q_consumer_state,
      PipelineKV& pipeline_kv, typename PipelineKV::PipelineState& pipeline_kv_consumer_state,
      PipelineS& pipeline_s0, typename PipelineS::PipelineState& pipeline_s0_producer_state,
      PipelineO& pipeline_corr, typename PipelineO::PipelineState& pipeline_corr_producer_state) {

    auto pipeline_q_release_state = pipeline_q_consumer_state;
    auto pipeline_kv_release_state = pipeline_kv_consumer_state;

    int mask_tile_count = Mask{}.get_trip_count(blk_coord, TileShape{}, problem_shape);

    typename CollectiveMmaQK::TiledMma mma_qk;
    ThrMMA thr_mma_qk = mma_qk.get_slice(cta_mma_rank());

    typename CollectiveMmaPVCompute::TiledMma mma_pv;
    TiledMMA mma_pv_ts = to_tiled_mma_sm100_ts(mma_pv);
    ThrMMA thr_mma_pv = mma_pv_ts.get_slice(cta_mma_rank());

    Tensor sQ = make_tensor(make_smem_ptr(storage.smem_q.data()), SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(storage.smem_k.data()), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(storage.smem_v.data()), SmemLayoutV{});

    Tensor tSrQ = thr_mma_qk.make_fragment_A(sQ);
    Tensor tSrK = thr_mma_qk.make_fragment_B(sK);

    Tensor tStS = partition_fragment_C(mma_qk, select<0,1>(TileShapeQK{}));
    Tensor cO_mma = make_identity_tensor(select<0,1>(TileShapePVMma{}));
    Tensor tOtO = decltype(mma_pv_ts)::make_fragment_C(
        thr_mma_pv.partition_C(cO_mma));

    Tensor tStS0 = tStS;
    tStS0.data() = tStS.data().get() + uint32_t(TmemAllocation::S0);
    Tensor tOtO0 = tOtO;
    tOtO0.data() = tOtO.data().get() + uint32_t(TmemAllocation::O0);

    Tensor sP = make_tensor(make_smem_ptr((Element*)nullptr),
                            typename CollectiveMmaPVCompute::SmemLayoutA{});
    Tensor tOrP = thr_mma_pv.make_fragment_A(sP)(_, _, _, _0{});
    Tensor tOrP0 = tOrP;
    tOrP0.data() = tOrP0.data().get() + uint32_t(TmemAllocation::P0);

    Tensor tOtO0_hi = tOtO0;
    tOtO0_hi.data() = tOtO0.data().get() + 128;
    Tensor tOrV = thr_mma_pv.make_fragment_B(sV);

    auto gemm_pv_zero = [&](auto const& p, int part, int v_index) {
      if constexpr (kIs2Sm) {
        if (part == 0) {
          gemm_zero_acc(mma_pv_ts, p, tOrV(_, _, _, v_index), tOtO0);
        } else {
          gemm_zero_acc(mma_pv_ts, p, tOrV(_, _, _, v_index), tOtO0_hi);
        }
      } else {
        gemm_zero_acc(mma_pv_ts, p, tOrV(_, _, _, v_index), tOtO0);
      }
    };
    auto gemm_pv_accumulate = [&](auto const& p, int part, int v_index) {
      if constexpr (kIs2Sm) {
        if (part == 0) {
          gemm_reset_zero_acc(mma_pv_ts, p, tOrV(_, _, _, v_index), tOtO0);
        } else {
          gemm_reset_zero_acc(mma_pv_ts, p, tOrV(_, _, _, v_index), tOtO0_hi);
        }
      } else {
        gemm_reset_zero_acc(mma_pv_ts, p, tOrV(_, _, _, v_index), tOtO0);
      }
    };

    // apxinf: slot-indexed views for the multi-stage S pipeline.
    auto tStS_at = [&](uint32_t slot) {
      Tensor t = tStS;
      t.data() = tStS.data().get() + uint32_t(TmemAllocation::S0)
               + slot * uint32_t(TmemAllocation::kSizeS);
      return t;
    };
    auto tOrP_at = [&](uint32_t slot) {
      Tensor t = tOrP;
      t.data() = tOrP.data().get() + uint32_t(TmemAllocation::P0)
               + slot * uint32_t(TmemAllocation::kSizeS);
      return t;
    };

    // Q (single stage)
    int q_index = pipeline_q_consumer_state.index();
    pipeline_q.consumer_wait(pipeline_q_consumer_state);
    ++pipeline_q_consumer_state;
    Tensor tSrQ0 = tSrQ(_,_,_,q_index);

    if constexpr (kIs2Sm) {
      int q_index_hi = pipeline_q_consumer_state.index();
      pipeline_q.consumer_wait(pipeline_q_consumer_state);
      ++pipeline_q_consumer_state;
      Tensor tSrQ1 = tSrQ(_,_,_,q_index_hi);

      int n_tiles = mask_tile_count;
      auto issue_qk = [&]() {
        pipeline_s0.producer_acquire(pipeline_s0_producer_state);
        Tensor tStS_i = tStS_at(uint32_t(pipeline_s0_producer_state.index()));

        int k_index = pipeline_kv_consumer_state.index();
        pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
        ++pipeline_kv_consumer_state;
        gemm_zero_acc(mma_qk, tSrQ0, tSrK(_, _, _, k_index), tStS_i);

        pipeline_kv.consumer_release(pipeline_kv_release_state);
        ++pipeline_kv_release_state;

        k_index = pipeline_kv_consumer_state.index();
        pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
        ++pipeline_kv_consumer_state;
        gemm_reset_zero_acc(mma_qk, tSrQ1,
                            tSrK(_, _, _, k_index), tStS_i);
        pipeline_s0.producer_commit(pipeline_s0_producer_state);
        ++pipeline_s0_producer_state;

        pipeline_kv.consumer_release(pipeline_kv_release_state);
        ++pipeline_kv_release_state;
      };

      auto issue_pv = [&](int tile, auto final_tile) {
        // After a look-ahead QK, the producer cursor itself names P(tile).
        // For the final PV there is no following QK, so step back one slot.
        auto s_pv = pipeline_s0_producer_state;
        if constexpr (decltype(final_tile)::value) {
          ++s_pv;
        }
        pipeline_s0.producer_acquire(s_pv);
        uint32_t p_slot = uint32_t(s_pv.index());

        int v_index = pipeline_kv_consumer_state.index();
        pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
        ++pipeline_kv_consumer_state;

        // For tile i>0 the empty O barrier is released by correction(i), and
        // correction(i) first needs the statistics produced by softmax(i).
        // QK(i) therefore has to be issued before this acquire; placing the
        // acquire at the top of the loop creates a QK(i) <-> correction(i)
        // cycle and hangs as soon as a query has two KV tiles.
        pipeline_corr.producer_acquire(pipeline_corr_producer_state);

        if (tile == 0) {
          gemm_pv_zero(tOrP_at(p_slot), 0, v_index);
        } else {
          gemm_pv_accumulate(tOrP_at(p_slot), 0, v_index);
        }

        pipeline_kv.consumer_release(pipeline_kv_release_state);
        ++pipeline_kv_release_state;

        v_index = pipeline_kv_consumer_state.index();
        pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
        ++pipeline_kv_consumer_state;
        if (tile == 0) {
          gemm_pv_zero(tOrP_at(p_slot), 1, v_index);
        } else {
          gemm_pv_accumulate(tOrP_at(p_slot), 1, v_index);
        }

        pipeline_kv.consumer_release(pipeline_kv_release_state);
        ++pipeline_kv_release_state;

        pipeline_corr.producer_commit(pipeline_corr_producer_state);
        ++pipeline_corr_producer_state;
      };

      // Match the DSL MMA schedule exactly: QK0, then QK(i+1) before PV(i),
      // and finally PV(end).  The next QK overlaps the softmax/P store for the
      // preceding tile instead of leaving the tensor pipe idle.
      issue_qk();
      CUTLASS_PRAGMA_NO_UNROLL
      for (int i = 1; i < n_tiles; ++i) {
        issue_qk();
        issue_pv(i - 1, cute::false_type{});
      }
      issue_pv(n_tiles - 1, cute::true_type{});

      pipeline_q.consumer_release(pipeline_q_release_state);
      ++pipeline_q_release_state;
      pipeline_q.consumer_release(pipeline_q_release_state);
      ++pipeline_q_release_state;

      // Match the softmax warp's trailing consumer release.
      pipeline_s0.producer_commit(pipeline_s0_producer_state);
      ++pipeline_s0_producer_state;
      return;
    }

    if constexpr (kMultiSStage) {
      // ---- two S slots: run QK one tile ahead of PV ------------------------
      // Issue order per tile i (i = 0 .. n-1):
      //   [wait V_i] [corr acq] [wait K_{i+1}] QK(i+1) -> S[(i+1)%2]
      //   [S acq == "P(i) is in tmem"] PV(i) -> O
      // QK(i+1) is issued *before* PV(i), so it executes on the tensor core
      // while the tail of softmax(i) (the ex2 loop and the P store) is still
      // running. With a single S slot the acquire for QK(i+1) could not be
      // satisfied until softmax(i) released S, which strictly serialised
      // QK -> softmax -> PV -> QK and left the MMA pipe idle for the whole
      // softmax (ncu: tensor pipe 36% for us vs 60% for the AOT).
      //
      // The KV ring must still be drained in TMA issue order K0,V0,K1,V1,...
      // so V_i is waited on before K_{i+1} even though QK(i+1) is issued first.
      //
      // apxinf, report 24: EVERY tmem hazard on this path now has an mbarrier
      // chain behind it.  There used to be one that did not, and it cost the
      // language operator two rounds (report 22 §4 / §5, SHARED-FINDINGS 20.2
      // called it "codegen-fragile" and pinned a dead `else if` to keep it
      // winning a race).  The table, with the barrier that enforces each:
      //
      //  (1) PV(i) READS P(i), which softmax(i) writes into columns
      //      [kSizeP, kSizeP + N/4*sizeof(Element)) of slot i&1.
      //      ==> `pipeline_s0.producer_acquire` below, immediately before the
      //          PV issue.  Its empty barrier is arrived by softmax(i)'s
      //          `consumer_release`, which follows the P store AND the
      //          `fence_view_async_tmem_store()` that makes it visible.
      //          THIS IS THE ONE THAT WAS MISSING.  It is a pure WAIT, on the
      //          empty barrier of the slot one round AHEAD of the one QK(i+1)
      //          writes -- i.e. the slot QK(i+2) will write, which is the slot
      //          PV(i) reads P(i) from.  No commit goes with it and the
      //          producer cursor is NOT advanced (see the note below on why
      //          that matters), so the pipeline's arrive accounting is exactly
      //          the stock one.  Before it, nothing in the MMA warp waited for
      //          softmax(i) at all: the strongest thing it knew at the PV(i)
      //          issue point was "correction(i) has read the STATS of tile i",
      //          which softmax writes BEFORE its ex2/P loop -- so PV(i) raced
      //          the P store and only won because QK(i+1) happened to be
      //          longer than the rest of softmax(i).  Measured consequence
      //          when ptxas changed the role dispatch by one dead disjunct
      //          (report 22 §4): 98.8188% bit-equal with outputs ~1e36.  And
      //          with a 2 us `__nanosleep` in front of the P store, 29.8%.
      //  (2) QK(i+1) WRITES slot (i+1)&1 == slot of tile i-1, which holds
      //      tile i-1's STATS (`V0 == S0`) that correction(i-1) must read.
      //      ==> `pipeline_corr.producer_acquire` at the top of the iteration.
      //          The correction warp cannot release the (depth-1) O
      //          accumulator for tile i without having read the stats of tile
      //          i first, so this is a real happens-before, not an ordering
      //          assumption.  (Found in round 16, SHARED-FINDINGS 十六.3.)
      //  (3) QK(i+1) WRITES slot (i+1)&1 while softmax(i-1) may still be
      //      READING S from it.
      //      ==> the `pipeline_s0.producer_acquire` that already sat in front
      //          of the QK issue: on a depth-2 pipeline the acquire at cursor
      //          p is arrived by the consumer's release at cursor p-2, i.e.
      //          softmax(i-1).  (This is also exactly why that acquire CANNOT
      //          serve hazard (1): it is one tile short.)
      //  (4) QK(i+2) WRITES slot i&1 while PV(i) may still be READING P(i).
      //      ==> `pipeline_corr.producer_acquire` of iteration i+1, which
      //          precedes the QK(i+2) issue: it waits for correction(i), which
      //          waited on the O full barrier that PV(i)'s `umma_arrive`
      //          raises on COMPLETION.  So PV(i) is complete, not merely
      //          issued.  The old comment here argued instead that "UMMA ops
      //          from one CTA complete in issue order"; that assumption is no
      //          longer load-bearing.
      //  (5) softmax(n-1) parks the FINAL row_sum/row_max in slot n&1.
      //      ==> its `final_call` consumer_wait, satisfied by the trailing
      //          commit below (a bare `umma_arrive`, i.e. all prior UMMA of
      //          this CTA has completed).
      //
      // The acquire for (1) is the ONLY addition, and it is a pure WAIT:
      // `PipelineAsync::producer_acquire` is `mbarrier.try_wait.parity` on the
      // empty barrier, which is passive and idempotent.  So it adds NO arrive
      // to either barrier and the pipeline's arrive accounting is byte for
      // byte the stock one: n+1 commits per Q tile, cursor advanced n+1 times.
      //
      // THAT LAST SENTENCE IS LOAD-BEARING, and not for the reason you would
      // guess.  The first attempt at this fix gave the acquire its own cursor
      // and paid the extra round back with one more commit here and one more
      // consumer_wait/release at the tail of `softmax()`.  The counts balanced
      // and the language operator was bit-exact -- because
      // ReverseIndividualTileScheduler gives every CTA exactly ONE Q tile.
      // With more than one Q tile per CTA it corrupted ~48% of O, because the
      // per-tile STATS tmem slot is derived from the PipelineS cursor in
      // `softmax_step` (`s_base`) but from the PipelineC cursor in
      // `correction_single_q` (`vtS_slot(pipeline_s0_c_consumer_state
      // .index())`).  Those are two different pipelines that happen to agree
      // only while both advance the same number of rounds per Q tile.  Any
      // change to the S pipeline's per-Q-tile round count silently
      // de-synchronises the correction warp's view of where the statistics
      // live.  See report 24 §4 and the probe in
      // devlocal/aot2cuda/harness/fmha_1q_persist_probe.cu.
      int n_tiles = mask_tile_count;

      // K0
      int k_index = pipeline_kv_consumer_state.index();
      pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
      ++pipeline_kv_consumer_state;

      pipeline_s0.producer_acquire(pipeline_s0_producer_state);
      gemm_zero_acc(mma_qk, tSrQ0, tSrK(_,_,_,k_index),
                    tStS_at(uint32_t(pipeline_s0_producer_state.index())));
      pipeline_s0.producer_commit(pipeline_s0_producer_state);
      ++pipeline_s0_producer_state;

      pipeline_kv.consumer_release(pipeline_kv_release_state);
      ++pipeline_kv_release_state;

      // The KV ring has exactly one K slot and one V slot, and the load warp
      // fills it strictly as K0,V0,K1,V1,... so at the top of tile i the
      // consumer cursor sits on V_i and K_{i+1} is the state right after it.
      // We wait on (and release) K_{i+1} first so that its TMA slot frees as
      // early as possible; the two slots are distinct mbarriers so releasing
      // them out of cursor order is safe.
      CUTLASS_PRAGMA_NO_UNROLL
      for (int i = 0; i < n_tiles; ++i) {
        auto st_v = pipeline_kv_consumer_state;
        auto st_k = pipeline_kv_consumer_state;
        ++st_k;
        int v_index = st_v.index();
        int kn_index = st_k.index();

        uint32_t p_slot = uint32_t(i & 1);

        // apxinf: the cursor of the slot PV(i) reads P(i) from, which is one
        // round AHEAD of the slot QK(i+1) writes.  Snapshot it at the top of
        // the iteration (where the real cursor is at i+1) so the expression is
        // the same whether or not QK(i+1) gets issued below.  Nothing is
        // committed against it -- see the note on hazard (1).
        auto s_pv = pipeline_s0_producer_state;
        ++s_pv;

        pipeline_kv.consumer_wait(st_v);

        // apxinf: this acquire must come BEFORE QK(i+1) is issued -- hazard
        // (2) in the table above.  The per-tile softmax statistics (old/new
        // row max, final row sum) live in columns 0..1 of the SAME tmem slot
        // as the S tile they came from (`V0 == S0`).  QK(i+1) targets slot
        // (i+1)&1, which is the slot that held tile i-1, so it overwrites the
        // statistics of tile i-1.  `pipeline_s0`'s acquire says nothing about
        // that: it is released by softmax as soon as it is done with S.
        //
        // `pipeline_corr` is the O accumulator pipeline (depth 1): the
        // correction warp releases it only after it has rescaled O for a tile,
        // which it cannot do before reading that tile's statistics.  So
        // acquiring it here is exactly the missing guarantee, and QK(i+1) is
        // still issued before the PV(i) gemm, which is where the overlap win
        // comes from.  (Round 16, measured: without it 73% bit-equal,
        // non-deterministic, O off by a scale factor ~1e37.)
        //
        // It is also what orders QK(i+2) after PV(i) *completes* -- hazard (4).
        pipeline_corr.producer_acquire(pipeline_corr_producer_state);

        if (i + 1 < n_tiles) {
          // K(i+1) and QK(i+1) -- issued ahead of PV(i) so that the tensor
          // core works on S(i+1) while softmax(i) is still running.
          pipeline_kv.consumer_wait(st_k);

          pipeline_s0.producer_acquire(pipeline_s0_producer_state);
          gemm_zero_acc(mma_qk, tSrQ0, tSrK(_,_,_,kn_index),
                        tStS_at(uint32_t(pipeline_s0_producer_state.index())));
          pipeline_s0.producer_commit(pipeline_s0_producer_state);
          ++pipeline_s0_producer_state;

          pipeline_kv.consumer_release(st_k);
        }

        // apxinf: hazard (1), the fix of report 24.  `s_pv` is the cursor of
        // slot (i+2)&1 == i&1 -- the slot QK(i+2) will write and the slot
        // PV(i) reads P(i) from.  Its empty barrier is arrived by softmax(i)'s
        // `consumer_release`, which comes after the P store and the
        // `fence_view_async_tmem_store()` that publishes it, so once this
        // returns, P(i) is in tmem and PV(i) is legal.
        //
        // It must stay BELOW the QK(i+1) issue (otherwise the MMA warp blocks
        // before handing the tensor core its next job, which is the whole
        // point of two S slots) and ABOVE the PV(i) issue (otherwise it gates
        // nothing).  A pure wait: no commit, no cursor advance.
        pipeline_s0.producer_acquire(s_pv);

        if (i == 0) {
          gemm_pv_zero(tOrP_at(p_slot), 0, v_index);
        }
        else {
          gemm_pv_accumulate(tOrP_at(p_slot), 0, v_index);
        }
        pipeline_corr.producer_commit(pipeline_corr_producer_state);
        ++pipeline_corr_producer_state;

        pipeline_kv.consumer_release(st_v);

        ++pipeline_kv_consumer_state;
        if (i + 1 < n_tiles) { ++pipeline_kv_consumer_state; }
      }
      pipeline_kv_release_state = pipeline_kv_consumer_state;

      pipeline_q.consumer_release(pipeline_q_release_state);
      ++pipeline_q_release_state;

      // matches the softmax warp's trailing consumer_release
      pipeline_s0.producer_acquire(pipeline_s0_producer_state);
      pipeline_s0.producer_commit(pipeline_s0_producer_state);
      ++pipeline_s0_producer_state;
      return;
    }

    // K0
    int k_index = pipeline_kv_consumer_state.index();
    pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
    ++pipeline_kv_consumer_state;

    pipeline_s0.producer_acquire(pipeline_s0_producer_state);
    gemm_zero_acc(mma_qk, tSrQ0, tSrK(_,_,_,k_index), tStS0);
    pipeline_s0.producer_commit(pipeline_s0_producer_state);
    ++pipeline_s0_producer_state;

    pipeline_kv.consumer_release(pipeline_kv_release_state);
    ++pipeline_kv_release_state;

    // V0
    int v_index = pipeline_kv_consumer_state.index();
    pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
    ++pipeline_kv_consumer_state;

    pipeline_corr.producer_acquire(pipeline_corr_producer_state);
    pipeline_s0.producer_acquire(pipeline_s0_producer_state);

    gemm_pv_zero(tOrP0, 0, v_index);

    pipeline_corr.producer_commit(pipeline_corr_producer_state);
    ++pipeline_corr_producer_state;

    pipeline_kv.consumer_release(pipeline_kv_release_state);
    ++pipeline_kv_release_state;

    mask_tile_count -= 1;
    CUTLASS_PRAGMA_NO_UNROLL
    for (; mask_tile_count > 0; mask_tile_count -= 1) {
      // Ki
      k_index = pipeline_kv_consumer_state.index();
      pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
      ++pipeline_kv_consumer_state;

      gemm_zero_acc(mma_qk, tSrQ0, tSrK(_,_,_,k_index), tStS0);

      pipeline_s0.producer_commit(pipeline_s0_producer_state);
      ++pipeline_s0_producer_state;

      pipeline_kv.consumer_release(pipeline_kv_release_state);
      ++pipeline_kv_release_state;

      // Vi
      v_index = pipeline_kv_consumer_state.index();
      pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
      ++pipeline_kv_consumer_state;

      pipeline_corr.producer_acquire(pipeline_corr_producer_state);
      pipeline_s0.producer_acquire(pipeline_s0_producer_state);

      gemm_pv_accumulate(tOrP0, 0, v_index);

      pipeline_corr.producer_commit(pipeline_corr_producer_state);
      ++pipeline_corr_producer_state;

      pipeline_kv.consumer_release(pipeline_kv_release_state);
      ++pipeline_kv_release_state;
    }

    pipeline_q.consumer_release(pipeline_q_release_state);
    ++pipeline_q_release_state;

    // matches the softmax warp's trailing consumer_release
    pipeline_s0.producer_commit(pipeline_s0_producer_state);
    ++pipeline_s0_producer_state;
  }

  template<class BlkCoord, class ProblemShape>
  CUTLASS_DEVICE auto
  mma(
      BlkCoord const& blk_coord,
      Params const& params, ProblemShape const& problem_shape,
      TensorStorage& storage,
      PipelineQ& pipeline_q, typename PipelineQ::PipelineState& pipeline_q_consumer_state,
      PipelineKV& pipeline_kv, typename PipelineKV::PipelineState& pipeline_kv_consumer_state,
      PipelineS& pipeline_s0, typename PipelineS::PipelineState& pipeline_s0_producer_state,
      PipelineS& pipeline_s1, typename PipelineS::PipelineState& pipeline_s1_producer_state,
      PipelineO& pipeline_corr, typename PipelineO::PipelineState& pipeline_corr_producer_state) {

    if constexpr (kSingleQStage) {
      mma_single_q(blk_coord, params, problem_shape, storage,
          pipeline_q, pipeline_q_consumer_state,
          pipeline_kv, pipeline_kv_consumer_state,
          pipeline_s0, pipeline_s0_producer_state,
          pipeline_corr, pipeline_corr_producer_state);
      return;
    }
    else {
    auto pipeline_q_release_state = pipeline_q_consumer_state;
    auto pipeline_kv_release_state = pipeline_kv_consumer_state;

    int mask_tile_count = Mask{}.get_trip_count(blk_coord, TileShape{}, problem_shape);

    typename CollectiveMmaQK::TiledMma mma_qk;
    ThrMMA thr_mma_qk = mma_qk.get_slice(cta_mma_rank());

    typename CollectiveMmaPV::TiledMma mma_pv;
    TiledMMA mma_pv_ts = to_tiled_mma_sm100_ts(mma_pv);
    ThrMMA thr_mma_pv = mma_pv_ts.get_slice(cta_mma_rank());

    Tensor sQ = make_tensor(make_smem_ptr(storage.smem_q.data()), SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(storage.smem_k.data()), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(storage.smem_v.data()), SmemLayoutV{});

    Tensor tSrQ = thr_mma_qk.make_fragment_A(sQ);
    Tensor tSrK = thr_mma_qk.make_fragment_B(sK);
    Tensor tOrV = thr_mma_pv.make_fragment_B(sV);

    // tmem layout is
    // S0 S1`O0 O1
    // sequential in memory, where S overlaps with P and V

    Tensor tStS = partition_fragment_C(mma_qk, select<0,1>(TileShapeQK{}));
    Tensor tOtO = partition_fragment_C(mma_pv_ts, select<0,1>(TileShapePV{}));

    Tensor tStS0 = tStS;
    tStS0.data() = tStS.data().get() + uint32_t(TmemAllocation::S0);
    Tensor tStS1 = tStS;
    tStS1.data() = tStS.data().get() + uint32_t(TmemAllocation::S1);

    Tensor tOtO0 = tOtO;
    tOtO0.data() = tOtO.data().get() + uint32_t(TmemAllocation::O0);
    Tensor tOtO1 = tOtO;
    tOtO1.data() = tOtO.data().get() + uint32_t(TmemAllocation::O1);

    Tensor sP = make_tensor(make_smem_ptr((Element*)nullptr), typename CollectiveMmaPV::SmemLayoutA{});
    Tensor tOrP = thr_mma_pv.make_fragment_A(sP)(_, _, _, _0{});  // slice out staging

    Tensor tOrP0 = tOrP;
    tOrP0.data() = tOrP0.data().get() + uint32_t(TmemAllocation::P0);
    Tensor tOrP1 = tOrP;
    tOrP1.data() = tOrP1.data().get() + uint32_t(TmemAllocation::P1);

    int k_index = 0;
    int v_index = 0;
    int q_index = 0;

    // wait for Q1
    q_index = pipeline_q_consumer_state.index();
    pipeline_q.consumer_wait(pipeline_q_consumer_state);
    ++pipeline_q_consumer_state;

    Tensor tSrQ0 = tSrQ(_,_,_,q_index);


    // wait for K1
    k_index = pipeline_kv_consumer_state.index();
    pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
    ++pipeline_kv_consumer_state;

    // gemm Q1 * K1 -> S1
    pipeline_s0.producer_acquire(pipeline_s0_producer_state);

    gemm_zero_acc(mma_qk, tSrQ0, tSrK(_,_,_,k_index), tStS0);

    pipeline_s0.producer_commit(pipeline_s0_producer_state);
    ++pipeline_s0_producer_state;

    // release K1
    if constexpr (get<1>(ThreadShape{}) > 1) {
      pipeline_kv.consumer_release(pipeline_kv_release_state);
      ++pipeline_kv_release_state;
    }

    // wait for Q2
    if constexpr (get<0>(ThreadShape{}) > 1 || get<2>(ThreadShape{}) > 1) {
      q_index = pipeline_q_consumer_state.index();
      pipeline_q.consumer_wait(pipeline_q_consumer_state);
      ++pipeline_q_consumer_state;
    }

    Tensor tSrQ1 = tSrQ(_,_,_,q_index);

    if constexpr (get<1>(ThreadShape{}) > 1) {
      k_index = pipeline_kv_consumer_state.index();
      pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
      ++pipeline_kv_consumer_state;
    }

    pipeline_s1.producer_acquire(pipeline_s1_producer_state);

    // gemm Q2 * K1 -> S2
    gemm_zero_acc(mma_qk, tSrQ1, tSrK(_,_,_,k_index), tStS1);

    pipeline_s1.producer_commit(pipeline_s1_producer_state);
    ++pipeline_s1_producer_state;

    // release K1
    pipeline_kv.consumer_release(pipeline_kv_release_state);
    ++pipeline_kv_release_state;

    // wait for V1
    v_index = pipeline_kv_consumer_state.index();
    pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
    ++pipeline_kv_consumer_state;

    // this acquire returns the ownership of all of S0 to the mma warp
    // including the P0 part
    // acquire corr first to take it out of the critical
    // path since softmax takes longer
    pipeline_corr.producer_acquire(pipeline_corr_producer_state);
    pipeline_s0.producer_acquire(pipeline_s0_producer_state);

    // gemm P1 * V1 -> O1
    gemm_zero_acc(mma_pv_ts, tOrP0, tOrV(_,_,_,v_index), tOtO0);

    pipeline_corr.producer_commit(pipeline_corr_producer_state);
    ++pipeline_corr_producer_state;

      if constexpr (get<1>(ThreadShape{}) > 1) {
      pipeline_kv.consumer_release(pipeline_kv_release_state);
      ++pipeline_kv_release_state;
    }

    mma_pv_ts.accumulate_ = UMMA::ScaleOut::Zero;

    // loop:
    mask_tile_count -= 1;
    for (; mask_tile_count > 0; mask_tile_count -= 1) {

      // wait for Ki
      k_index = (pipeline_kv_consumer_state.index());
      pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
      ++pipeline_kv_consumer_state;

      // gemm Q1 * Ki -> S1
      gemm_zero_acc(mma_qk, tSrQ0, tSrK(_,_,_,k_index), tStS0);

      pipeline_s0.producer_commit(pipeline_s0_producer_state);
      ++pipeline_s0_producer_state;

      if constexpr (get<1>(ThreadShape{}) > 1) {
        pipeline_kv.consumer_release(pipeline_kv_release_state);
        ++pipeline_kv_release_state;
      }

      // gemm P2 * V(i-1) -> O2
      if constexpr (get<1>(ThreadShape{}) > 1) {
        v_index = pipeline_kv_consumer_state.index();
        pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
        ++pipeline_kv_consumer_state;
      }

      pipeline_corr.producer_acquire(pipeline_corr_producer_state);
      pipeline_s1.producer_acquire(pipeline_s1_producer_state);

      gemm_reset_zero_acc(mma_pv_ts, tOrP1, tOrV(_,_,_,v_index), tOtO1);

      pipeline_corr.producer_commit(pipeline_corr_producer_state);
      ++pipeline_corr_producer_state;

      // release V(i-1)
      pipeline_kv.consumer_release(pipeline_kv_release_state);
      ++pipeline_kv_release_state;

      if constexpr (get<1>(ThreadShape{}) > 1) {
        k_index = (pipeline_kv_consumer_state.index());
        pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
        ++pipeline_kv_consumer_state;
      }

      // gemm Q2 * Ki -> S2
      gemm_zero_acc(mma_qk, tSrQ1, tSrK(_,_,_,k_index), tStS1);

      pipeline_s1.producer_commit(pipeline_s1_producer_state);
      ++pipeline_s1_producer_state;

      // release Ki
      pipeline_kv.consumer_release(pipeline_kv_release_state);
      ++pipeline_kv_release_state;

      // wait for Vi
      v_index = (pipeline_kv_consumer_state.index());
      pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
      ++pipeline_kv_consumer_state;

      // gemm P1 * Vi -> O1
      pipeline_corr.producer_acquire(pipeline_corr_producer_state);

      pipeline_s0.producer_acquire(pipeline_s0_producer_state);

      gemm_reset_zero_acc(mma_pv_ts, tOrP0, tOrV(_,_,_,v_index), tOtO0);

      pipeline_corr.producer_commit(pipeline_corr_producer_state);
      ++pipeline_corr_producer_state;

      if constexpr (get<1>(ThreadShape{}) > 1) {
        pipeline_kv.consumer_release(pipeline_kv_release_state);
        ++pipeline_kv_release_state;
      }
    }

    // release Q1
    pipeline_q.consumer_release(pipeline_q_release_state);
    ++pipeline_q_release_state;

    // release Q2
    if constexpr (get<0>(ThreadShape{}) > 1) {
      pipeline_q.consumer_release(pipeline_q_release_state);
      ++pipeline_q_release_state;
    }

    // wait for Vi
    if constexpr (get<1>(ThreadShape{}) > 1) {
      v_index = pipeline_kv_consumer_state.index();
      pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
      ++pipeline_kv_consumer_state;
    }

    // gemm P2 * Vi -> O2
    pipeline_corr.producer_acquire(pipeline_corr_producer_state);
    pipeline_s1.producer_acquire(pipeline_s1_producer_state);

    gemm_reset_zero_acc(mma_pv_ts, tOrP1, tOrV(_,_,_,v_index), tOtO1);

    pipeline_corr.producer_commit(pipeline_corr_producer_state);
    ++pipeline_corr_producer_state;

    // release Vi
    pipeline_kv.consumer_release(pipeline_kv_release_state);
    ++pipeline_kv_release_state;

    pipeline_s0.producer_commit(pipeline_s0_producer_state);
    ++pipeline_s0_producer_state;

    pipeline_s1.producer_commit(pipeline_s1_producer_state);
    ++pipeline_s1_producer_state;

    // T0 S00 B1, T0 S10 B1, T0 S00 B2, T0 S01 B1, T0 S10 B2, T0 S11 B1, T0 S01 B2, T1 S00 B1, T0 S11 B2, ...
    // Q1 * K1  , Q2 * K1  , S11 * V1 , Q1 * K2  , S21 * V1  , Q2 * K2 , S12 * V2 , Q1 * K3  , S22 * K2 , ...
    }  // else (two Q stages)
  }

#if APXINF_SKIP_PAD_QSTAGE
  // APXINF_SKIP_PAD_QSTAGE (report 21 §1.2, ported from the vision fork).
  // TileShape M=256 with ThreadShape<2,1,1> makes each Q tile two 128-row
  // stages, and the kernel only `continue`s when the WHOLE tile is out of
  // range.  At L=624 the last tile's stage 1 covers rows 640..767 -- ZERO valid
  // rows -- yet pays full price.  FA4 does not skip it either.
  //
  // This MUST be a separate function, not a branch inside `softmax_step`: the
  // branch form was bit-exact but 11% SLOWER, because it wrecked ptxas's
  // schedule for the whole hot loop.
  //
  // Every pipeline operation of the real step is reproduced in the same order,
  // so the handshake counts across load / mma / softmax / correction / epilogue
  // are unchanged and this cannot deadlock.  The two counts that have to match
  // (SHARED-FINDINGS 21.2, report 24 §4) are, per Q stage of T KV tiles:
  //   pipeline_s: consumer_wait = consumer_release = T + 1
  //   pipeline_c: producer_commit = T + 1, producer_acquire = T + 2
  // Note in particular the `final_call` re-acquire below -- `softmax_step` does
  // one, and omitting it here HANGS the kernel with no output at all.
  //
  // The statistics are written old == new == 0, so the correction warp's scale
  // is exactly 1.0f and it skips correction_rescale; the final row sum is 1.0f
  // so the reciprocal is finite.  P is left stale -- the MMA warp still runs
  // its PV on it -- but that O feeds only rows the epilogue's TMA predicate
  // drops.
  template<class Stage>
  CUTLASS_DEVICE void
  softmax_step_all_padding(
      Stage stage, bool final_call,
      PipelineS& pipeline_s, typename PipelineS::PipelineState& pipeline_s_consumer_state,
      PipelineC& pipeline_c, typename PipelineC::PipelineState& pipeline_c_producer_state,
      OrderBarrierSoftmax& order_s) {

    Tensor tStS = partition_fragment_C(typename CollectiveMmaQK::TiledMma{}, select<0,1>(TileShapeQK{}));
    Tensor cS = make_identity_tensor(select<0,1>(TileShapeQK{}));
    Tensor tScS = typename CollectiveMmaQK::TiledMma{}.get_slice(cta_mma_rank()).partition_C(cS);

    // Same derivation as `softmax_step`'s `s_base`.  V0 == S0 and V1 == S1, so
    // the statistics live in columns 0-1 of the S slot.
    uint32_t s_base = kMultiSStage
        ? warp_uniform(uint32_t(pipeline_s_consumer_state.index())
                       * uint32_t(TmemAllocation::kSizeS))
        : uint32_t(stage == _0{} ? TmemAllocation::S0 : TmemAllocation::S1);

    Tensor tStS_v = tStS.compose(make_layout(make_shape(_128{}, _2{})));
    tStS_v.data() = s_base + kStatsOffset;
    Tensor tScS_v = tScS.compose(make_layout(make_shape(_128{}, _2{})));

    using TMEM_STORE_V = SM100_TMEM_STORE_32dp32b2x;
    int thread_idx = threadIdx.x % (4 * cutlass::NumThreadsPerWarp);
    auto tiled_tmem_storev = make_tmem_copy(TMEM_STORE_V{}, tStS_v);
    auto thr_tmem_storev = tiled_tmem_storev.get_slice(thread_idx);
    Tensor tTMEM_STOREVtS = thr_tmem_storev.partition_D(tStS_v);
    Tensor tTMEM_STOREVcS = thr_tmem_storev.partition_S(tScS_v);

    pipeline_s.consumer_wait(pipeline_s_consumer_state);

    Tensor tTMEM_STOREVrS = make_tensor<ElementQK>(shape(tTMEM_STOREVcS));
    tTMEM_STOREVrS(kIdxOldRowMax) = ElementQK(0);
    tTMEM_STOREVrS(kIdxNewRowMax) = ElementQK(0);
    copy(tiled_tmem_storev, tTMEM_STOREVrS, tTMEM_STOREVtS);

    pipeline_c.producer_commit(pipeline_c_producer_state);
    ++pipeline_c_producer_state;

    // The partner softmax warpgroup is still running real steps and still
    // waits on this barrier; skipping the pair would park it for ever.
    if constexpr (! kSingleQStage && ! (APXINF_NO_ORDER_S)) {
      order_s.wait();
      order_s.arrive();
    }

    cutlass::arch::fence_view_async_tmem_store();

    pipeline_s.consumer_release(pipeline_s_consumer_state);
    ++pipeline_s_consumer_state;

    pipeline_c.producer_acquire(pipeline_c_producer_state);

    if (final_call) {
      // `softmax_step` re-acquires the S slot in its final step; the handshake
      // count has to match exactly or pipeline_s runs one release ahead and the
      // kernel hangs (measured: without this the harness produces no output).
      pipeline_s.consumer_wait(pipeline_s_consumer_state);

      Tensor tTMEM_STOREVtS_final = tTMEM_STOREVtS;
      if constexpr (kMultiSStage) {
        uint32_t final_base = warp_uniform(
            uint32_t(pipeline_s_consumer_state.index()) * uint32_t(TmemAllocation::kSizeS));
        tTMEM_STOREVtS_final.data() =
            tTMEM_STOREVtS_final.data().get() + final_base - s_base;
      }

      Tensor tTMEM_STOREVrS_f = make_tensor<ElementQK>(shape(tTMEM_STOREVcS));
      tTMEM_STOREVrS_f(kIdxFinalRowSum) = ElementQK(1);
      tTMEM_STOREVrS_f(kIdxFinalRowMax) = ElementQK(0);
      copy(tiled_tmem_storev, tTMEM_STOREVrS_f, tTMEM_STOREVtS_final);
    }
  }
#endif  // APXINF_SKIP_PAD_QSTAGE

  template<bool need_apply_mask, class Stage, class BlkCoord, class CoordTensor, class ProblemShape>
  CUTLASS_DEVICE auto
  softmax_step(
      float& row_max, float& row_sum,
      Stage stage, bool final_call,
      BlkCoord const& blk_coord, CoordTensor const& cS,
      Params const& params, ProblemShape const& problem_shape,
      PipelineS& pipeline_s, typename PipelineS::PipelineState& pipeline_s_consumer_state,
      PipelineC& pipeline_c, typename PipelineC::PipelineState& pipeline_c_producer_state,
      OrderBarrierSoftmax& order_s) {

    Tensor tScS = typename CollectiveMmaQK::TiledMma{}.get_slice(cta_mma_rank()).partition_C(cS);

    // apxinf: with a multi-stage S pipeline the slot is a *runtime* value (the
    // pipeline state index) instead of the compile-time Q stage. For the stock
    // layouts slot*kSizeS reproduces S0/S1 exactly.
    // On the stock path this is a compile-time constant exactly as upstream
    // (S0 / S1); only the multi-S-slot path needs a runtime, warp-uniform
    // pipeline index.
    uint32_t s_base = kMultiSStage
        ? warp_uniform(uint32_t(pipeline_s_consumer_state.index())
                       * uint32_t(TmemAllocation::kSizeS))
        : uint32_t(stage == _0{} ? TmemAllocation::S0 : TmemAllocation::S1);

    Tensor tStS = partition_fragment_C(typename CollectiveMmaQK::TiledMma{}, select<0,1>(TileShapeQK{}));
    tStS.data() = s_base;

    Tensor tStS_v = tStS.compose(make_layout(make_shape(_128{}, _2{})));
    tStS_v.data() = s_base + kStatsOffset;
    Tensor tScS_v = tScS.compose(make_layout(make_shape(_128{}, _2{})));

    auto tilePlikeFP32 = size<1>(TileShapeQK{}) / Int<sizeof(float)>{} * Int<sizeof(Element)>{};
    Tensor tStS_P = tStS.compose(make_layout(make_shape(_128{}, tilePlikeFP32)));
    tStS_P.data() = warp_uniform(s_base + kPBaseOffset);
    Tensor tScS_P = tScS.compose(make_layout(make_shape(_128{}, tilePlikeFP32)));

    // Each thread owns a single row
      using TMEM_LOAD = SM100_TMEM_LOAD_32dp32b32x; // 4x32 threads with 128 cols of 32b elem
#if APXINF_P_CHUNKS16
    // FA4 parity (flash_fwd_sm100.py:2102-2108): St32x32bOp(Repetition(16)),
    // i.e. four 16-column P chunks.  Halves the packed-bf16 registers that have
    // to stay live and spreads the stores through the exp2 loop.
    using TMEM_STORE = SM100_TMEM_STORE_32dp32b16x;
#else
    using TMEM_STORE = SM100_TMEM_STORE_32dp32b32x;  // 4x32 threads with 128 cols of 8b elem
#endif
    using TMEM_STORE_V = SM100_TMEM_STORE_32dp32b2x;   // 4x32 threads with 2 cols of 32b elem

    int thread_idx = threadIdx.x % (4 * cutlass::NumThreadsPerWarp);

    auto tiled_tmem_load = make_tmem_copy(TMEM_LOAD{}, tStS);
    auto thr_tmem_load   = tiled_tmem_load.get_slice(thread_idx);

    Tensor tTMEM_LOADtS = thr_tmem_load.partition_S(tStS);
    Tensor tTMEM_LOADcS = thr_tmem_load.partition_D(tScS);

    auto tiled_tmem_storev = make_tmem_copy(TMEM_STORE_V{}, tStS_v);
    auto thr_tmem_storev  = tiled_tmem_storev.get_slice(thread_idx);

    Tensor tTMEM_STOREVtS = thr_tmem_storev.partition_D(tStS_v);
    Tensor tTMEM_STOREVcS = thr_tmem_storev.partition_S(tScS_v);

    auto tiled_tmem_store = make_tmem_copy(TMEM_STORE{}, tStS_P);
    auto thr_tmem_store  = tiled_tmem_store.get_slice(thread_idx);

    Tensor tTMEM_STOREtS_x4 = thr_tmem_store.partition_D(tStS_P);
    tTMEM_STOREtS_x4.data() = warp_uniform(tTMEM_STOREtS_x4.data().get());
    Tensor tTMEM_STOREcS = thr_tmem_store.partition_S(tScS_P);

    // wait on tensor core pipe
    pipeline_s.consumer_wait(pipeline_s_consumer_state);

    // read all of S from tmem into reg mem
    Tensor tTMEM_LOADrS = make_tensor<ElementQK>(shape(tTMEM_LOADcS));
    copy(tiled_tmem_load, tTMEM_LOADtS, tTMEM_LOADrS);

    if constexpr (need_apply_mask) {
      Mask{}.apply_mask(tTMEM_LOADrS, tTMEM_LOADcS, problem_shape);
    }

    ElementQK old_row_max = row_max;
#if APXINF_FMAX3
    {
      // FA4 utils.py::fmax_reduce, arch >= 100 path (stride-8, four
      // accumulators, forced 3-input max).  max is exact.
      float m0 = apxinf_fmax3(row_max, tTMEM_LOADrS(0), tTMEM_LOADrS(1));
      float m1 = ::fmax(tTMEM_LOADrS(2), tTMEM_LOADrS(3));
      float m2 = ::fmax(tTMEM_LOADrS(4), tTMEM_LOADrS(5));
      float m3 = ::fmax(tTMEM_LOADrS(6), tTMEM_LOADrS(7));
      CUTLASS_PRAGMA_UNROLL
      for (int i = 8; i < size(tTMEM_LOADrS); i += 8) {
        m0 = apxinf_fmax3(m0, tTMEM_LOADrS(i+0), tTMEM_LOADrS(i+1));
        m1 = apxinf_fmax3(m1, tTMEM_LOADrS(i+2), tTMEM_LOADrS(i+3));
        m2 = apxinf_fmax3(m2, tTMEM_LOADrS(i+4), tTMEM_LOADrS(i+5));
        m3 = apxinf_fmax3(m3, tTMEM_LOADrS(i+6), tTMEM_LOADrS(i+7));
      }
      m0 = ::fmax(m0, m1);
      row_max = apxinf_fmax3(m0, m2, m3);
    }
#else
    {
      // compute rowmax
      float row_max_0 = row_max;
      float row_max_1 = row_max;
      float row_max_2 = row_max;
      float row_max_3 = row_max;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tTMEM_LOADrS); i += 4) {
        row_max_0  = ::fmax(row_max_0, tTMEM_LOADrS(i));
        row_max_1 = ::fmax(row_max_1, tTMEM_LOADrS(i+1));
        row_max_2 = ::fmax(row_max_2, tTMEM_LOADrS(i+2));
        row_max_3 = ::fmax(row_max_3, tTMEM_LOADrS(i+3));
      }
      row_max = ::fmax(row_max_0, row_max_1);
      row_max = ::fmax(row_max, row_max_2);
      row_max = ::fmax(row_max, row_max_3);
    }
#endif

    ElementQK row_max_safe = row_max == -INFINITY ? 0 : row_max;

    // FA4 parity: for 16-bit inputs `flash_fwd_sm100.py` sets
    // `rescale_threshold = 8.0`. When the new row max would grow the
    // probabilities by less than 2^8 it keeps the *lagging* max and skips the
    // accumulator rescale altogether. SASS evidence: the action kernel has one
    // `FSETP.GE ..., -8`; the language (hd256 2-CTA) kernel has ZERO, i.e. that
    // kernel always rescales, so the threshold must be configurable per
    // operator.
    constexpr ElementQK kRescaleThreshold =
        sizeof(Element) == 2 ? ElementQK(APXINF_RESCALE_THRESHOLD) : 0.0f;

    // Keep this OUT of the default build: hoisting `scale * (old - new)` above
    // the exp2 loop extends two live ranges across it, which perturbs ptxas'
    // register allocation.  APXINF_ACC_SCALE_SEED() recomputes it where
    // upstream does.
    bool skip_rescale = false;
    if constexpr (kRescaleThreshold > 0.0f) {
      ElementQK acc_scale_log2 = params.scale_softmax_log2 * (old_row_max - row_max_safe);
      skip_rescale = acc_scale_log2 >= -kRescaleThreshold;
      if (skip_rescale) {
        // Revert to the lagging max: the current tile is exponentiated against
        // the old bias and O keeps its existing scale.
        row_max = old_row_max;
        row_max_safe = old_row_max;
      }
    }

#if APXINF_LATE_C_ACQUIRE
    // Acquire the stats (V) slot as late as legally possible: immediately
    // before the write, instead of at the tail of the previous step.
    pipeline_c.producer_acquire(pipeline_c_producer_state);
#endif
    Tensor tTMEM_STOREVrS = make_tensor<ElementQK>(shape(tTMEM_STOREVcS));
    tTMEM_STOREVrS(kIdxOldRowMax) = old_row_max;
    tTMEM_STOREVrS(kIdxNewRowMax) = row_max_safe;
    copy(tiled_tmem_storev, tTMEM_STOREVrS, tTMEM_STOREVtS);

    pipeline_c.producer_commit(pipeline_c_producer_state);
    ++pipeline_c_producer_state;

    // notify correction wg that they are ready (might need addtl ordering between S0 and S1 WG's)

    ElementQK scale = params.scale_softmax_log2;
#if APXINF_BIAS_NEG_FIRST
    ElementQK minus_row_max_scale = (0.0f - row_max_safe) * scale;
#else
    ElementQK minus_row_max_scale = -(row_max_safe * scale);
#endif

    float2 scale_fp32x2 = make_float2(scale, scale);
    float2 minus_row_max_scale_fp32x2 = make_float2(minus_row_max_scale, minus_row_max_scale);

    Tensor tTMEM_STORErS_x4 = make_tensor<uint32_t>(shape(tTMEM_STOREcS));

    constexpr int kConversionsPerStep = 2;

    Tensor tTMEM_STORErS_x4_e = recast<Array<Element, kConversionsPerStep>>(tTMEM_STORErS_x4);

    NumericArrayConverter<Element, ElementQK, kConversionsPerStep> convert;

    const int kReleasePipeCount = APXINF_RELEASE_PIPE_COUNT;  // multiple of 2

#if APXINF_EX2_EMU_RES > 0
    // FA4 fragments the per-thread S row in 32-element chunks; the emulation
    // schedule is indexed off that, so the row length has to agree.
    CUTE_STATIC_ASSERT_V(size(tTMEM_LOADrS) % _32{} == _0{});
    constexpr int kFrgCnt = decltype(size(tTMEM_LOADrS))::value / 32;
#endif

    // Accumulate row_sum inside the exp2 loop below rather than in a second
    // pass over tTMEM_LOADrS. Keeping all 128 exponentiated values live until
    // after P had been packed pushed the softmax warps to ~192 live registers,
    // and __launch_bounds__(512, 1) caps ptxas at 128 -- the result was ~700
    // LDL/STL spill instructions and 97x the AOT kernel's LSU traffic.
    // The four stride-8 accumulators and their visiting order are unchanged,
    // so this is bit-identical to the two-pass form (and to FA4's
    // `utils.fadd_reduce`).
#if APXINF_FUSE_ROWSUM
    APXINF_ACC_SCALE_SEED();
#endif

    // With a single Q stage there is only one softmax warpgroup, so the
    // ping-pong ordering barrier has no partner to hand off to.
    if constexpr (! kSingleQStage && ! (APXINF_NO_ORDER_S)) {
      order_s.wait();
    }

#if APXINF_EX2_HOIST_C3 && APXINF_EX2_EMU_RES > 0
    float const apx_ex2_c3 = apxinf_ex2_c3_reg();
#endif

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(tTMEM_LOADrS); i += 2) {
      float2 in = make_float2(
        tTMEM_LOADrS(i + 0),
        tTMEM_LOADrS(i + 1)
      );
      float2 out;
      cute::fma(out, scale_fp32x2, in, minus_row_max_scale_fp32x2);
      tTMEM_LOADrS(i + 0) = out.x;
      tTMEM_LOADrS(i + 1) = out.y;

#if APXINF_EX2_EMU_RES > 0
      if (apxinf_use_hw_ex2(i, kFrgCnt)) {
        tTMEM_LOADrS(i+0) = apxinf_ex2_approx(tTMEM_LOADrS(i+0));
        tTMEM_LOADrS(i+1) = apxinf_ex2_approx(tTMEM_LOADrS(i+1));
      } else {
#if APXINF_EX2_HOIST_C3
        apxinf_ex2_emulation_2_c3(tTMEM_LOADrS(i+0), tTMEM_LOADrS(i+1), apx_ex2_c3);
#else
        apxinf_ex2_emulation_2(tTMEM_LOADrS(i+0), tTMEM_LOADrS(i+1));
#endif
      }
#else
      tTMEM_LOADrS(i+0) = apxinf_ex2_approx(tTMEM_LOADrS(i+0));
      tTMEM_LOADrS(i+1) = apxinf_ex2_approx(tTMEM_LOADrS(i+1));
#endif

      Array<ElementQK, kConversionsPerStep> in_conv;
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < kConversionsPerStep; j++) {
        in_conv[j] = tTMEM_LOADrS(i + j);
      }
      tTMEM_STORErS_x4_e[i / kConversionsPerStep] = convert(in_conv);

#if APXINF_FUSE_ROWSUM
      // Fold this pair into its accumulator. The switch keeps the
      // target in a register; an indexed array would risk becoming local.
      float2 sum_in = make_float2(tTMEM_LOADrS(i + 0), tTMEM_LOADrS(i + 1));
#if APXINF_ROWSUM_QUARTERS
      constexpr int kQuarter = decltype(size(tTMEM_LOADrS))::value / 4;
      switch (i / kQuarter) {
        case 0: APXINF_ROWSUM_ADD(local_row_sum_f32x2, local_row_sum_f32x2, sum_in); break;
        case 1: APXINF_ROWSUM_ADD(local_row_sum_1, local_row_sum_1, sum_in); break;
        case 2: APXINF_ROWSUM_ADD(local_row_sum_2, local_row_sum_2, sum_in); break;
        default: APXINF_ROWSUM_ADD(local_row_sum_3, local_row_sum_3, sum_in); break;
      }
#else
      switch (i % 8) {
        case 0: APXINF_ROWSUM_ADD(local_row_sum_f32x2, local_row_sum_f32x2, sum_in); break;
        case 2: APXINF_ROWSUM_ADD(local_row_sum_1, local_row_sum_1, sum_in); break;
        case 4: APXINF_ROWSUM_ADD(local_row_sum_2, local_row_sum_2, sum_in); break;
        default: APXINF_ROWSUM_ADD(local_row_sum_3, local_row_sum_3, sum_in); break;
      }
#endif
#endif  // APXINF_FUSE_ROWSUM

      if (i == size(tTMEM_LOADrS) - kReleasePipeCount) {
        if constexpr (! kSingleQStage && ! (APXINF_NO_ORDER_S)) {
          order_s.arrive();
        }
      }

      // Flush the first half of P to tmem as soon as it is complete. The stock
      // code did this 6 elements before the end, which is far too late to free
      // anything: with D = 256 / N = 128 the softmax warpgroup holds 128 S
      // floats plus 64 packed-bf16 P registers at once, and ptxas (capped at
      // 168 regs by __launch_bounds__(384)) spilled ~40 of them. Chunk 0 covers
      // P elements [0, 64), so it is complete once i reaches size/2.
#if APXINF_P_CHUNKS16
      // Retire each P chunk to tmem as soon as it is complete rather than
      // holding all of P to the end.  Safe because the whole S row is already
      // in registers (P aliases S in tmem).  The last chunk is stored after the
      // loop, exactly as upstream.  Bit-neutral: only the store granularity
      // changes.
      {
        constexpr int kElemsPerPChunk =
            decltype(size(tTMEM_LOADrS))::value /
            decltype(size<2>(tTMEM_STORErS_x4))::value;
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < size<2>(tTMEM_STORErS_x4) - 1; ++j) {
          if (i == (j + 1) * kElemsPerPChunk - 2) {
            copy(tiled_tmem_store, tTMEM_STORErS_x4(_, _, j), tTMEM_STOREtS_x4(_, _, j));
          }
        }
      }
#else
      if constexpr (size<2>(tTMEM_STORErS_x4) == _2{}) {
        if (i == APXINF_P_FLUSH_AT) {
          copy(tiled_tmem_store, tTMEM_STORErS_x4(_, _, 0), tTMEM_STOREtS_x4(_, _, 0));
        }
      }
#endif
    }

    // tmem_store(reg_S8) -> op_P
#if APXINF_P_CHUNKS16
    CUTE_STATIC_ASSERT_V(size<2>(tTMEM_STORErS_x4) <= _4{});
#else
    CUTE_STATIC_ASSERT_V(size<2>(tTMEM_STORErS_x4) <= _2{});
#endif
    CUTE_STATIC_ASSERT_V(size<1>(tTMEM_STORErS_x4) == _1{});
    copy(tiled_tmem_store, tTMEM_STORErS_x4(_, _, size<2>(tTMEM_STORErS_x4) - 1), tTMEM_STOREtS_x4(_, _, size<2>(tTMEM_STORErS_x4) - 1));

    cutlass::arch::fence_view_async_tmem_store();

    // notify tensor core warp that P is ready
    pipeline_s.consumer_release(pipeline_s_consumer_state);
    ++pipeline_s_consumer_state;

#if !APXINF_LATE_C_ACQUIRE
    pipeline_c.producer_acquire(pipeline_c_producer_state);
#endif

#if !APXINF_FUSE_ROWSUM
    // UPSTREAM: a second pass over tTMEM_LOADrS after the P store.
    APXINF_ACC_SCALE_SEED();

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(tTMEM_LOADrS); i += 8) {
      // row_sum += tTMEM_LOADrS(i);
      float2 in = make_float2(tTMEM_LOADrS(i), tTMEM_LOADrS(i+1));
      APXINF_ROWSUM_ADD(local_row_sum_f32x2, local_row_sum_f32x2, in);

      in = make_float2(tTMEM_LOADrS(i+2), tTMEM_LOADrS(i+2+1));
      APXINF_ROWSUM_ADD(local_row_sum_1, local_row_sum_1, in);

      in = make_float2(tTMEM_LOADrS(i+4), tTMEM_LOADrS(i+4+1));
      APXINF_ROWSUM_ADD(local_row_sum_2, local_row_sum_2, in);

      in = make_float2(tTMEM_LOADrS(i+6), tTMEM_LOADrS(i+6+1));
      APXINF_ROWSUM_ADD(local_row_sum_3, local_row_sum_3, in);
    }
#endif

    APXINF_ROWSUM_ADD(local_row_sum_f32x2, local_row_sum_f32x2, local_row_sum_1);
    APXINF_ROWSUM_ADD(local_row_sum_2, local_row_sum_2, local_row_sum_3);
    APXINF_ROWSUM_ADD(local_row_sum_f32x2, local_row_sum_f32x2, local_row_sum_2);
    float local_row_sum = local_row_sum_f32x2.x + local_row_sum_f32x2.y;

    row_sum = local_row_sum;

    if (final_call) {
      // re-acquire the S part in the final step
      pipeline_s.consumer_wait(pipeline_s_consumer_state);

#if APXINF_LATE_C_ACQUIRE
      pipeline_c.producer_acquire(pipeline_c_producer_state);
#endif

      // apxinf: pipeline_s_consumer_state has already been advanced past this
      // tile, so with a multi-stage S pipeline the final row_sum/row_max go
      // into the *next* slot -- which is exactly the slot the correction warp
      // reads at its matching (last) PipelineC state.
      Tensor tTMEM_STOREVtS_final = tTMEM_STOREVtS;
      if constexpr (kMultiSStage) {
        uint32_t final_base = warp_uniform(
            uint32_t(pipeline_s_consumer_state.index()) * uint32_t(TmemAllocation::kSizeS));
        tTMEM_STOREVtS_final.data() =
            tTMEM_STOREVtS_final.data().get() + final_base - s_base;
      }

      Tensor tTMEM_STOREVrS = make_tensor<ElementQK>(shape(tTMEM_STOREVcS));
      tTMEM_STOREVrS(kIdxFinalRowMax) = row_max;
      tTMEM_STOREVrS(kIdxFinalRowSum) = row_sum;
      copy(tiled_tmem_storev, tTMEM_STOREVrS, tTMEM_STOREVtS_final);
    }
  }

  template<class Stage, class BlkCoord, class ProblemShape>
  CUTLASS_DEVICE auto
  softmax(
      Stage stage,
      BlkCoord const& blk_coord,
      Params const& params, ProblemShape const& problem_shape,
      PipelineS& pipeline_s, typename PipelineS::PipelineState& pipeline_s_consumer_state,
      PipelineC& pipeline_c, typename PipelineC::PipelineState& pipeline_c_producer_state,
      OrderBarrierSoftmax& order_s) {

    int mask_tile_count = Mask{}.get_unmasked_trip_count(blk_coord, TileShape{}, problem_shape);

    ElementQK row_max = -INFINITY;
    ElementQK row_sum = 0;

    Tensor cS_base = make_identity_tensor(select<0,1>(TileShapeQK{}));
    auto logical_offset = make_coord(
        get<0>(blk_coord) * get<0>(TileShape{}) + (stage % get<0>(ThreadShape{})) * get<0>(TileShapeQK{}),
        0 + (stage % get<1>(ThreadShape{})) * get<1>(TileShapeQK{})
    );
    Tensor cS = domain_offset(logical_offset, cS_base);

#if !APXINF_LATE_C_ACQUIRE
    pipeline_c.producer_acquire(pipeline_c_producer_state);
#endif

#if APXINF_KV_DESCENDING
    int masked_tile_count = Mask{}.get_masked_trip_count(blk_coord, TileShape{}, problem_shape);
    const int kv_tile_cols = get<1>(ThreadShape{}) * get<1>(TileShapeQK{});
#endif

    // FA4's two hd256 kernels walk the KV tiles in OPPOSITE directions, and the
    // whole online-softmax trajectory depends on the direction:
    //   action   `flash_fwd_sm100.py:2355`  n_block = n_block_max-1-n_tile
    //            => DESCENDING, the ragged/masked tile is the FIRST iteration.
    //   language `sm100_hd256_2cta_fmha_forward.py:1449`
    //            `for step in range(start_count, end_count)` with
    //            `need_apply_mask = step == end_count - 1`
    //            => ASCENDING, the masked tile is the LAST iteration.
    // Keep this in step with the fetch order in
    // sm100_fmha_load_tma_warpspecialized.hpp.
#if APXINF_SKIP_PAD_QSTAGE
    // An all-padding Q stage: reproduce the handshakes and skip the work.
    // `get_trip_count` == unmasked + masked for every in-tree mask.
    const int apxinf_total_tiles =
        Mask{}.get_trip_count(blk_coord, TileShape{}, problem_shape);
    if (get<0>(logical_offset) >= get<0>(problem_shape)) {
      CUTLASS_PRAGMA_NO_UNROLL
      for (int tile = 0; tile < apxinf_total_tiles; ++tile) {
        softmax_step_all_padding(stage, tile == apxinf_total_tiles - 1,
                                 pipeline_s, pipeline_s_consumer_state,
                                 pipeline_c, pipeline_c_producer_state,
                                 order_s);
      }
    } else {
#endif

#if APXINF_KV_DESCENDING
    // Start at the last tile and step backwards; masked tiles come first.
    cS.data() = cS.data() + E<1>{} * (kv_tile_cols * (mask_tile_count + masked_tile_count - 1));

    CUTLASS_PRAGMA_NO_UNROLL
    for (; masked_tile_count > 0; masked_tile_count -= 1) {
      softmax_step<true /* need_apply_mask */>(
          row_max, row_sum, stage,
          (masked_tile_count == 1) && (mask_tile_count == 0),
          blk_coord, cS, params, problem_shape,
          pipeline_s, pipeline_s_consumer_state,
          pipeline_c, pipeline_c_producer_state,
          order_s
      );

      cS.data() = cS.data() + E<1>{} * (-kv_tile_cols);
    }

    CUTLASS_PRAGMA_NO_UNROLL
    for (; mask_tile_count > 0; mask_tile_count -= 1) {
      softmax_step<false /* need_apply_mask */>(
          row_max, row_sum, stage, mask_tile_count == 1,
          blk_coord, cS, params, problem_shape,
          pipeline_s, pipeline_s_consumer_state,
          pipeline_c, pipeline_c_producer_state,
          order_s
      );

      cS.data() = cS.data() + E<1>{} * (-kv_tile_cols);
    }
#else
    // Ascending (UPSTREAM): unmasked tiles first, masked (trailing) tiles
    // last.  Kept VERBATIM from upstream, including re-using mask_tile_count
    // for the masked trip count and re-deriving it inline, because hoisting it
    // into a second live variable costs the ResidualMask kernel ~130 ISETP.GE
    // + ~130 IADD3 (ptxas stops strength-reducing apply_mask's per-element
    // column compare).  Numerically identical either way; cheaper this way.
    CUTLASS_PRAGMA_NO_UNROLL
    for (; mask_tile_count > 0; mask_tile_count -= 1) {
      softmax_step<false /* need_apply_mask */>(
          row_max, row_sum, stage,
          (mask_tile_count == 1) &&
              (Mask{}.get_masked_trip_count(blk_coord, TileShape{}, problem_shape) == 0),
          blk_coord, cS, params, problem_shape,
          pipeline_s, pipeline_s_consumer_state,
          pipeline_c, pipeline_c_producer_state,
          order_s
      );

      cS.data() = cS.data() + E<1>{} * get<1>(ThreadShape{}) * get<1>(TileShapeQK{});
    }

    // Masked iterations
    mask_tile_count = Mask{}.get_masked_trip_count(blk_coord, TileShape{}, problem_shape);

    CUTLASS_PRAGMA_NO_UNROLL
    for (; mask_tile_count > 0; mask_tile_count -= 1) {
      softmax_step<true /* need_apply_mask */>(
          row_max, row_sum, stage, mask_tile_count == 1,
          blk_coord, cS, params, problem_shape,
          pipeline_s, pipeline_s_consumer_state,
          pipeline_c, pipeline_c_producer_state,
          order_s
      );

      cS.data() = cS.data() + E<1>{} * get<1>(ThreadShape{}) * get<1>(TileShapeQK{});
    }
#endif

#if APXINF_SKIP_PAD_QSTAGE
    }  // else (not an all-padding Q stage)
#endif

    pipeline_c.producer_commit(pipeline_c_producer_state);
    ++pipeline_c_producer_state;

#if !APXINF_LATE_C_ACQUIRE
    pipeline_c.producer_acquire(pipeline_c_producer_state);
#endif
    // empty step to sync against pipe s
    pipeline_s.consumer_release(pipeline_s_consumer_state);
    ++pipeline_s_consumer_state;

    // apxinf: the per-Q-tile round count of THIS pipeline is coupled to
    // PipelineC's, invisibly and at a distance: `softmax_step` puts the
    // statistics at `pipeline_s_consumer_state.index() * kSizeS` while
    // `correction_single_q` reads them at
    // `pipeline_s0_c_consumer_state.index() * kSizeS`.  The two cursors agree
    // only because both advance T+1 rounds per Q tile (T = KV tiles; the +1 is
    // the final_call re-acquire above).  Adding or removing a round on either
    // side de-synchronises them from the SECOND Q tile of a CTA onwards --
    // measured at ~48% of O corrupted, with no deadlock and nothing wrong on
    // the first Q tile.  Report 24 §4; the counting discipline is
    // SHARED-FINDINGS 21.2.
  }

  template<class Stage, class TensorO>
  CUTLASS_DEVICE auto
  correction_epilogue(
      float scale,
      Stage stage,
      TensorO const& sO_01) {

    using ElementOut = typename TensorO::value_type;

    int thread_idx = threadIdx.x % (4 * cutlass::NumThreadsPerWarp);

    Tensor sO = sO_01(_,_,stage);

    // As opposed to the softmax, we do not have enough registers here
    // to load all of the values (for tile kv = 128), so we loop
    // good values would be either 32 or 64
    const int kCorrectionTileSize = 32 / sizeof(ElementOut);

    using TMEM_LOAD = std::conditional_t<kCorrectionTileSize == 32, SM100_TMEM_LOAD_32dp32b32x, SM100_TMEM_LOAD_32dp32b16x>;  // 4x32 threads with 64 cols of 32b elem

    typename CollectiveMmaPV::TiledMma mma;
    Tensor cO = make_identity_tensor(select<0,1>(TileShapePV{}));
    Tensor tOtO = partition_fragment_C(mma, select<0,1>(TileShapePV{}));
    Tensor tOcO = mma.get_slice(cta_mma_rank()).partition_C(cO);
    Tensor tOsO = mma.get_slice(cta_mma_rank()).partition_C(sO);

    Tensor tOtO_i = logical_divide(tOtO, make_layout(make_shape(_128{}, Int<kCorrectionTileSize>{})));
    Tensor tOcO_i = logical_divide(tOcO, make_layout(make_shape(_128{}, Int<kCorrectionTileSize>{})));
    Tensor tOsO_i = logical_divide(tOsO, make_layout(make_shape(_128{}, Int<kCorrectionTileSize>{})));

    if constexpr (decltype(stage == _0{})::value) {
      tOtO_i.data() = tOtO_i.data().get() + uint32_t(TmemAllocation::O0);
    }
    else {
      static_assert(decltype(stage == _1{})::value, "stage is either 0 or 1");
      tOtO_i.data() = tOtO_i.data().get() + uint32_t(TmemAllocation::O1);
    }

    auto tiled_tmem_load = make_tmem_copy(TMEM_LOAD{}, tOtO_i(make_coord(_, _), _0{}));
    auto thr_tmem_load   = tiled_tmem_load.get_slice(thread_idx);

    Tensor tTMEM_LOADtO = thr_tmem_load.partition_S(tOtO_i(make_coord(_, _), _));
    Tensor tTMEM_LOADcO = thr_tmem_load.partition_D(tOcO_i(make_coord(_, _), _));
    Tensor tTMEM_LOADsO = thr_tmem_load.partition_D(tOsO_i(make_coord(_, _), _));

    float2 scale_f32x2 = make_float2(scale, scale);

    // loop:
    //   TMEM_LOAD, FMUL2 scale, TMEM_STORE
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < get<2>(TileShape{}) / kCorrectionTileSize; i++) {
      Tensor tTMEM_LOADtO_i = tTMEM_LOADtO(_, _0{}, _0{}, i);
      Tensor tTMEM_LOADsO_i = tTMEM_LOADsO(_, _0{}, _0{}, i);

      Tensor tTMrO = make_tensor<ElementPV>(shape(tTMEM_LOADcO(_, _0{}, _0{}, i)));

      copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMrO);

#ifndef ONLY_SOFTMAX
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size(tTMrO); j += 2) {
        float2 in = make_float2(tTMrO(j), tTMrO(j+1));
        float2 out;
        cute::mul(out, scale_f32x2, in);
        tTMrO(j) = out.x;
        tTMrO(j+1) = out.y;
      }
#endif

      constexpr int N = 4 / sizeof(ElementOut);
      NumericArrayConverter<ElementOut, ElementPV, N> convert;

      Tensor tSMrO = make_tensor_like<ElementOut>(tTMrO);

      Tensor tCs = recast<decltype(convert)::source_type>(tTMrO);
      Tensor tCd = recast<decltype(convert)::result_type>(tSMrO);

      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size(tCs); j++) {
        tCd(j) = convert.convert(tCs(j));
      }

      Tensor tSMsO_i = recast<uint32_t>(tTMEM_LOADsO_i);
      Tensor tSMrO_i = recast<uint32_t>(tSMrO);

      copy(AutoVectorizingCopyWithAssumedAlignment<128>{}, tSMrO_i, tSMsO_i);
    }

    cutlass::arch::fence_view_async_shared();
  }

  // FA4's hd256 2-CTA kernel converts TMEM and writes O directly from the
  // correction warpgroup.  Keeping that path avoids trying to express a
  // rank-sliced 2-CTA accumulator through a one-CTA TMA-store layout.
  template<class Stage, class BlkCoord, class ProblemShape, class CollectiveEpilogue>
  CUTLASS_DEVICE auto
  correction_epilogue_direct(
      float scale, Stage stage, BlkCoord const& blk_coord,
      ProblemShape const& problem_shape, CollectiveEpilogue& epilogue) {
    static_assert(kIs2Sm, "direct FMHA epilogue is only used by the 2-CTA path");
    using ElementOut = typename CollectiveEpilogue::ElementOut;

    int thread_idx = threadIdx.x % (4 * cutlass::NumThreadsPerWarp);
    // FA4 correction_epilog uses Ld32x32bOp(Repetition(32)) regardless of the
    // bf16 output type.  The earlier 32/sizeof(ElementOut) spelling selected
    // Repetition(16), which only covered alternating 64-column quarters.
    const int kCorrectionTileSize = 32;
    using TMEM_LOAD = SM100_TMEM_LOAD_32dp32b32x;

    typename CollectiveMmaPVCompute::TiledMma mma;
    TiledMMA mma_ts = to_tiled_mma_sm100_ts(mma);
    ThrMMA thr_mma = mma_ts.get_slice(cta_mma_rank());
    // FA4's correction warp works on the CTA-local epilogue tile (128x128),
    // not on another MMA-partitioned view of the cluster-wide M256 tile.
    // Partitioning the identity tensor a second time aliases coordinates and
    // lets zero lanes overwrite the second output half.
    Tensor cO = make_identity_tensor(
        make_shape(_128{}, get<1>(TileShapePVMma{})));
    Tensor cO_mma = make_identity_tensor(select<0,1>(TileShapePVMma{}));
    Tensor tOtO = decltype(mma_ts)::make_fragment_C(
        thr_mma.partition_C(cO_mma));

    auto problem_shape_o = CollectiveEpilogue::get_problem_shape_O(problem_shape);
    Tensor mO = make_tensor(make_gmem_ptr(epilogue.params.ptr_O),
                            problem_shape_o, epilogue.params.dO);
    using X = Underscore;
    Tensor gO_qdl = local_tile(
        mO, make_shape(_128{}, get<1>(TileShapePVMma{}), _1{}),
        make_coord(_, _, _), Step<_1, _1, X>{});
    float2 scale_f32x2 = make_float2(scale, scale);
    CUTLASS_PRAGMA_UNROLL
    for (int iter = 0; iter < get<2>(TileShape{}) / get<1>(TileShapePVMma{}); ++iter) {
      // Match the CuTe DSL epilogue exactly: first divide the full CTA-local
      // 128x128 output tile, then let the TMEM copy atom expose its four
      // 32-column repetitions.  Dividing by 128x32 here adds another nested
      // mode; its coordinate stride collapses to zero and alternating output
      // quarters are overwritten by the other 2SM half.
      Tensor tOtO_slice = tOtO(make_coord(_, _), _0{}, _0{});
      Tensor tOtO_i = zipped_divide(
          tOtO_slice, make_shape(_128{}, get<1>(TileShapePVMma{})));
      Tensor tOcO_i = zipped_divide(
          cO, make_shape(_128{}, get<1>(TileShapePVMma{})));
      int row_tile = get<0>(blk_coord) * int(cute::size<0>(ClusterShape{}))
                   + int(cta_mma_rank());
      Tensor gO = gO_qdl(_, _, row_tile, iter, get<2>(blk_coord));
      Tensor gO_i = zipped_divide(
          gO, make_shape(_128{}, get<1>(TileShapePVMma{})));

      uint32_t o_base = decltype(stage == _0{})::value
          ? uint32_t(TmemAllocation::O0) : uint32_t(TmemAllocation::O1);
      tOtO_i.data() = tOtO_i.data().get() + o_base
                    + uint32_t(iter * get<1>(TileShapePVMma{}));

      auto tiled_tmem_load = make_tmem_copy(TMEM_LOAD{}, tOtO_i);
      auto thr_tmem_load = tiled_tmem_load.get_slice(thread_idx);
      Tensor tTMEM_LOADtO = thr_tmem_load.partition_S(tOtO_i);
      Tensor tTMEM_LOADcO = thr_tmem_load.partition_D(tOcO_i);
      Tensor tTMEM_LOADgO = thr_tmem_load.partition_D(gO_i);

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size<1>(tTMEM_LOADtO); ++i) {
      Tensor tTMEM_LOADtO_i = tTMEM_LOADtO(_, i, _0{});
      Tensor tTMEM_LOADcO_i = tTMEM_LOADcO(_, i, _0{});
      Tensor tTMEM_LOADgO_i = tTMEM_LOADgO(_, i, _0{});
      Tensor tTMrO = make_tensor<ElementPV>(shape(tTMEM_LOADcO_i));
      copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMrO);

      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size(tTMrO); j += 2) {
        float2 in = make_float2(tTMrO(j), tTMrO(j + 1));
        float2 out;
        cute::mul(out, scale_f32x2, in);
        tTMrO(j) = out.x;
        tTMrO(j + 1) = out.y;
      }

      constexpr int N = 4 / sizeof(ElementOut);
      NumericArrayConverter<ElementOut, ElementPV, N> convert;
      Tensor tRrO = make_tensor_like<ElementOut>(tTMrO);
      Tensor tCs = recast<decltype(convert)::source_type>(tTMrO);
      Tensor tCd = recast<decltype(convert)::result_type>(tRrO);
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size(tCs); ++j) {
        tCd(j) = convert.convert(tCs(j));
      }

      Tensor src = recast<Array<ElementOut, N>>(coalesce(tRrO));
      int row_base = get<0>(blk_coord) * get<0>(TileShape{})
                   + int(cta_mma_rank()) * 128;
      auto first_coord = tTMEM_LOADcO_i(0);
      if (row_base + int(get<0>(first_coord)) <
          int(cute::size<0>(problem_shape))) {
        Tensor dst = recast<Array<ElementOut, N>>(
            coalesce(tTMEM_LOADgO_i));
        copy(AutoVectorizingCopyWithAssumedAlignment<128>{}, src, dst);
      }
    }
    }
  }

  CUTLASS_DEVICE auto
  correction_rescale(
      float scale,
      uint32_t tmem_O) {

    int thread_idx = threadIdx.x % (4 * cutlass::NumThreadsPerWarp);

    if constexpr (kIs2Sm) {
      // Match the FA4 hd256 correction epilogue: the logical D256 output is
      // two adjacent M256xN128 UMMA accumulators.  Each CTA owns a 128x128
      // slice of each accumulator, so rescale the two N128 fragments
      // independently using the same rank-sliced TMEM layout as the MMA.
      using TMEM_LOAD_2CTA = SM100_TMEM_LOAD_32dp32b16x;
      using TMEM_STORE_2CTA = SM100_TMEM_STORE_32dp32b16x;

      typename CollectiveMmaPVCompute::TiledMma mma;
      TiledMMA mma_ts = to_tiled_mma_sm100_ts(mma);
      ThrMMA thr_mma = mma_ts.get_slice(cta_mma_rank());
      Tensor cO = make_identity_tensor(
          make_shape(_128{}, get<1>(TileShapePVMma{})));
      Tensor cO_mma = make_identity_tensor(select<0,1>(TileShapePVMma{}));
      Tensor tOtO = decltype(mma_ts)::make_fragment_C(
          thr_mma.partition_C(cO_mma));
      float2 scale_f32x2 = make_float2(scale, scale);

      CUTLASS_PRAGMA_UNROLL
      for (int iter = 0;
           iter < get<2>(TileShape{}) / get<1>(TileShapePVMma{}); ++iter) {
        Tensor tOtO_slice = tOtO(make_coord(_, _), _0{}, _0{});
        Tensor tOtO_i = zipped_divide(
            tOtO_slice, make_shape(_128{}, get<1>(TileShapePVMma{})));
        Tensor tOcO_i = zipped_divide(
            cO, make_shape(_128{}, get<1>(TileShapePVMma{})));
        tOtO_i.data() = tOtO_i.data().get() + tmem_O
                      + uint32_t(iter * get<1>(TileShapePVMma{}));

        auto tiled_load = make_tmem_copy(TMEM_LOAD_2CTA{}, tOtO_i);
        auto thr_load = tiled_load.get_slice(thread_idx);
        auto tiled_store = make_tmem_copy(TMEM_STORE_2CTA{}, tOtO_i);
        auto thr_store = tiled_store.get_slice(thread_idx);
        Tensor tLoad = thr_load.partition_S(tOtO_i);
        Tensor cLoad = thr_load.partition_D(tOcO_i);
        Tensor tStore = thr_store.partition_D(tOtO_i);

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size<1>(tLoad); ++i) {
          Tensor tLoad_i = tLoad(_, i, _0{});
          Tensor tStore_i = tStore(_, i, _0{});
          Tensor rO = make_tensor<ElementPV>(shape(cLoad(_, i, _0{})));
          copy(tiled_load, tLoad_i, rO);
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < size(rO); j += 2) {
            float2 in = make_float2(rO(j), rO(j + 1));
            float2 out;
            cute::mul(out, scale_f32x2, in);
            rO(j) = out.x;
            rO(j + 1) = out.y;
          }
          copy(tiled_store, rO, tStore_i);
        }
      }
      return;
    }

    // As opposed to the softmax, we do not have enough registers here
    // to load all of the values (for tile kv = 128), so we loop
    // good values would be either 32 or 64
    // APXINF_CORR_TILE: 0 keeps upstream's 16.
    const int kCorrectionTileSize = (APXINF_CORR_TILE) == 0 ? 16 : (APXINF_CORR_TILE);
    static_assert(size<2>(TileShape{}) % ((APXINF_CORR_TILE) == 0 ? 16 : (APXINF_CORR_TILE)) == 0,
        "APXINF_CORR_TILE must divide the head dimension");

    using TMEM_LOAD = std::conditional_t<kCorrectionTileSize == 8,
        SM100_TMEM_LOAD_32dp32b8x, SM100_TMEM_LOAD_32dp32b16x>;  // 4x32 threads with 64 cols of 32b elem
    using TMEM_STORE = std::conditional_t<kCorrectionTileSize == 8,
        SM100_TMEM_STORE_32dp32b8x, SM100_TMEM_STORE_32dp32b16x>;  // 4x32 threads with 64 cols of 32b elem

    typename CollectiveMmaPV::TiledMma mma;
    Tensor cO = make_identity_tensor(select<0,1>(TileShapePV{}));
    Tensor tOtO = partition_fragment_C(mma, select<0,1>(TileShapePV{}));
    Tensor tOcO = mma.get_slice(cta_mma_rank()).partition_C(cO);

    Tensor tOtO_i = tOtO.compose(make_layout(make_shape(_128{}, Int<kCorrectionTileSize>{})));
    Tensor tOcO_i = tOcO.compose(make_layout(make_shape(_128{}, Int<kCorrectionTileSize>{})));

    tOtO_i.data() = tOtO_i.data().get() + tmem_O;

    auto tiled_tmem_load = make_tmem_copy(TMEM_LOAD{}, tOtO_i);
    auto thr_tmem_load   = tiled_tmem_load.get_slice(thread_idx);
    auto tiled_tmem_store = make_tmem_copy(TMEM_STORE{}, tOtO_i);
    auto thr_tmem_store   = tiled_tmem_store.get_slice(thread_idx);

    Tensor tTMEM_LOADtO = thr_tmem_load.partition_S(tOtO_i);
    Tensor tTMEM_LOADcO = thr_tmem_load.partition_D(tOcO_i);
    Tensor tTMEM_STOREtO = thr_tmem_store.partition_D(tOtO_i);
    Tensor tTMEM_STOREcO = thr_tmem_store.partition_S(tOcO_i);
    static_assert(shape(tTMEM_STOREcO) == shape(tTMEM_LOADcO));

    float2 scale_f32x2 = make_float2(scale, scale);

    // APXINF_CORR_RING2: the stock code sizes this `128 / kCorrectionTileSize`
    // which over-allocates for D < 128 and indexes OUT OF BOUNDS for D > 128
    // (the loop runs D / kCorrectionTileSize times).  Only two chunks are ever
    // live, so =1 stages exactly two.  Same values, same order; register
    // allocation differs, hence the knob.
    // The =0 arm derives the stage count from `TileShape` rather than the
    // hardcoded 128 -- identical for D = 128 (the only shape upstream's own
    // adapter instantiates), correct for every other D.  The indices used are
    // 0..D/tile-1 either way, so this is bit-neutral by construction.
#if APXINF_CORR_RING2
    Tensor tTMrO = make_tensor<ElementPV>(make_shape(shape(tTMEM_LOADcO), _2{}));
#define APXINF_CORR_SLOT(i) ((i) % 2)
#else
    Tensor tTMrO = make_tensor<ElementPV>(make_shape(shape(tTMEM_LOADcO),
        Int<size<2>(TileShape{}) / ((APXINF_CORR_TILE) == 0 ? 16 : (APXINF_CORR_TILE))>{}));
#define APXINF_CORR_SLOT(i) (i)
#endif

    auto copy_in = [&](int i) {
      Tensor tTMEM_LOADtO_i = tTMEM_LOADtO;
      tTMEM_LOADtO_i.data() = tTMEM_LOADtO_i.data().get() + uint32_t(i * kCorrectionTileSize);
      Tensor tTMrO_i = tTMrO(_, APXINF_CORR_SLOT(i)).compose(make_layout(shape<0>(tTMrO)));
      copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMrO_i);
    };

    auto copy_out = [&](int i) {
      Tensor tTMEM_STOREtO_i = tTMEM_STOREtO;
      tTMEM_STOREtO_i.data() = tTMEM_STOREtO_i.data().get() + uint32_t(i * kCorrectionTileSize);
      Tensor tTMrO_i = tTMrO(_, APXINF_CORR_SLOT(i)).compose(make_layout(shape<0>(tTMrO)));
      copy(tiled_tmem_store, tTMrO_i, tTMEM_STOREtO_i);
    };

    // sequence: LLMSLMSLMSS

    // loop:
    //   TMEM_LOAD, FMUL2 scale, TMEM_STORE
    copy_in(0);

    int count = get<2>(TileShape{}) / kCorrectionTileSize;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < count; i++) {
      if (i != count - 1) {
        copy_in(i+1);
      }

      Tensor tTMrO_i = tTMrO(_, APXINF_CORR_SLOT(i)).compose(make_layout(shape<0>(tTMrO)));

      if ((APXINF_CORR_UNCOND) || scale != 1.0f) {
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < size(tTMrO_i); j += 2) {
          float2 in = make_float2(tTMrO_i(j), tTMrO_i(j+1));
          float2 out;
          cute::mul(out, scale_f32x2, in);
          tTMrO_i(j) = out.x;
          tTMrO_i(j+1) = out.y;
        }
      }

      copy_out(i);
    }
  }

  // --- apxinf: single-Q-stage correction ------------------------------------
  // One accumulator, so the rescale for tile i has to land between PV(i-1) and
  // PV(i); pipeline_o serialises exactly that.
  template<
    class BlkCoord, class ProblemShape, class ParamsProblemShape,
    class TensorStorageEpi, class CollectiveEpilogue
  >
  CUTLASS_DEVICE void
  correction_single_q(
      BlkCoord const& blk_coord,
      Params const& params, ProblemShape const& problem_shape,
      ParamsProblemShape const& params_problem_shape,
      TensorStorageEpi& shared_storage_epi,
      PipelineC& pipeline_s0_c, typename PipelineC::PipelineState& pipeline_s0_c_consumer_state,
      PipelineO& pipeline_o, typename PipelineO::PipelineState& pipeline_o_consumer_state,
      PipelineE& pipeline_epi, typename PipelineE::PipelineState& pipeline_epi_producer_state,
      CollectiveEpilogue& epilogue) {

    int mask_tile_count = Mask{}.get_trip_count(blk_coord, TileShape{}, problem_shape);

    int thread_idx = threadIdx.x % (4 * cutlass::NumThreadsPerWarp);

    Tensor tStS = partition_fragment_C(typename CollectiveMmaQK::TiledMma{}, select<0,1>(TileShapeQK{}));

    Tensor cS = make_identity_tensor(select<0,1>(TileShapeQK{}));
    Tensor tScS = typename CollectiveMmaQK::TiledMma{}.get_slice(cta_mma_rank()).partition_C(cS);

    Tensor tStS_v = tStS.compose(make_layout(make_shape(_128{}, _2{})));
    Tensor tScS_v = tScS.compose(make_layout(make_shape(_128{}, _2{})));

    using TMEM_LOAD_V = SM100_TMEM_LOAD_32dp32b2x;

    auto tiled_tmem_loadv = make_tmem_copy(TMEM_LOAD_V{}, tStS_v);
    auto thr_tmem_loadv  = tiled_tmem_loadv.get_slice(thread_idx);

    Tensor tTMEM_LOADVtS = thr_tmem_loadv.partition_S(tStS_v);
    Tensor tTMEM_LOADVcS = thr_tmem_loadv.partition_D(tScS_v);

    // apxinf: the stats live in the same tmem slot as the S tile they came
    // from, so with a multi-stage S pipeline the slot is the pipeline index.
    auto vtS_slot = [&](uint32_t slot) {
      Tensor t = tTMEM_LOADVtS;
      t.data() = t.data().get() + uint32_t(TmemAllocation::V0)
               + slot * uint32_t(TmemAllocation::kSizeS);
      return t;
    };
    Tensor tTMEM_LOADVtS0 = vtS_slot(0);

    // first tile needs no rescale
    pipeline_s0_c.consumer_wait(pipeline_s0_c_consumer_state);
    pipeline_s0_c.consumer_release(pipeline_s0_c_consumer_state);
    ++pipeline_s0_c_consumer_state;

    mask_tile_count -= 1;

    CUTLASS_PRAGMA_NO_UNROLL
    for (; mask_tile_count > 0; mask_tile_count -= 1) {
      pipeline_s0_c.consumer_wait(pipeline_s0_c_consumer_state);

      Tensor tTMEM_LOADVrS = make_tensor<ElementQK>(shape(tTMEM_LOADVcS));
      copy(tiled_tmem_loadv,
           kMultiSStage ? vtS_slot(uint32_t(pipeline_s0_c_consumer_state.index())) : tTMEM_LOADVtS0,
           tTMEM_LOADVrS);

      float scale = (tTMEM_LOADVrS(kIdxOldRowMax) == tTMEM_LOADVrS(kIdxNewRowMax)) ? 1.0f : apxinf_ex2_approx(params.scale_softmax_log2 * (tTMEM_LOADVrS(kIdxOldRowMax) - tTMEM_LOADVrS(kIdxNewRowMax)));

      pipeline_o.consumer_wait(pipeline_o_consumer_state);

#if APXINF_CORR_UNCOND
      correction_rescale(scale, uint32_t(TmemAllocation::O0));
#else
      bool warp_do_correction = __any_sync(0xFFFFFFFF, scale != 1.0f);
      if (warp_do_correction) {
        correction_rescale(scale, uint32_t(TmemAllocation::O0));
      }
#endif

      cutlass::arch::fence_view_async_tmem_store();

      pipeline_o.consumer_release(pipeline_o_consumer_state);
      ++pipeline_o_consumer_state;

      pipeline_s0_c.consumer_release(pipeline_s0_c_consumer_state);
      ++pipeline_s0_c_consumer_state;
    }

    // final: read row_sum / row_max and run the output epilogue
    pipeline_s0_c.consumer_wait(pipeline_s0_c_consumer_state);

    Tensor tTMEM_LOADVrS = make_tensor<ElementQK>(shape(tTMEM_LOADVcS));
    copy(tiled_tmem_loadv,
         kMultiSStage ? vtS_slot(uint32_t(pipeline_s0_c_consumer_state.index())) : tTMEM_LOADVtS0,
         tTMEM_LOADVrS);

    pipeline_s0_c.consumer_release(pipeline_s0_c_consumer_state);
    ++pipeline_s0_c_consumer_state;

    pipeline_o.consumer_wait(pipeline_o_consumer_state);
    pipeline_epi.producer_acquire(pipeline_epi_producer_state);

    Tensor sO = make_epilogue_smem_tensor(shared_storage_epi);
    Tensor gLSE = make_tensor(make_gmem_ptr(epilogue.params.ptr_LSE), cute::make_shape(cute::size<0>(problem_shape), cute::get<3>(problem_shape)), epilogue.params.dLSE);

    if constexpr (kIs2Sm) {
      correction_epilogue_direct(
          apxinf_normalize_scale(params.scale_output,
                                 tTMEM_LOADVrS(kIdxFinalRowSum)),
          _0{}, blk_coord, problem_shape, epilogue);
    } else {
      correction_epilogue(
          apxinf_normalize_scale(params.scale_output,
                                 tTMEM_LOADVrS(kIdxFinalRowSum)),
          _0{}, sO);
    }

    if (epilogue.params.ptr_LSE != nullptr) {
      int row_idx = get<0>(tTMEM_LOADVcS(_0{})) + get<0>(TileShape{}) * get<0>(blk_coord);

      int row_offset = 0;
      if constexpr (is_variable_length_v<tuple_element_t<0, ParamsProblemShape>>) {
        row_offset = get<0>(params_problem_shape).cumulative_length[get<2,1>(blk_coord)];
      }

      ElementPV lse = apxinf_lse(tTMEM_LOADVrS(kIdxFinalRowSum),
                                 tTMEM_LOADVrS(kIdxFinalRowMax),
                                 params.scale_softmax,
                                 params.scale_softmax_log2);

      if (row_idx < int(cute::size<0>(problem_shape))) {
        gLSE(row_idx + row_offset, get<2>(blk_coord)) = lse;
      }
    }

    cutlass::arch::fence_view_async_tmem_load();

    pipeline_o.consumer_release(pipeline_o_consumer_state);
    ++pipeline_o_consumer_state;

    pipeline_epi.producer_commit(pipeline_epi_producer_state);
    ++pipeline_epi_producer_state;
  }

  template<
    class BlkCoord, class ProblemShape, class ParamsProblemShape,
    class TensorStorageEpi, class CollectiveEpilogue
  >
  CUTLASS_DEVICE auto
  correction(
      BlkCoord const& blk_coord,
      Params const& params, ProblemShape const& problem_shape,
      ParamsProblemShape const& params_problem_shape,
      TensorStorageEpi& shared_storage_epi,
      PipelineC& pipeline_s0_c, typename PipelineC::PipelineState& pipeline_s0_c_consumer_state,
      PipelineC& pipeline_s1_c, typename PipelineC::PipelineState& pipeline_s1_c_consumer_state,
      PipelineO& pipeline_o, typename PipelineO::PipelineState& pipeline_o_consumer_state,
      PipelineE& pipeline_epi, typename PipelineE::PipelineState& pipeline_epi_producer_state,
      CollectiveEpilogue& epilogue) {

    if constexpr (kSingleQStage) {
      correction_single_q(blk_coord, params, problem_shape, params_problem_shape,
          shared_storage_epi,
          pipeline_s0_c, pipeline_s0_c_consumer_state,
          pipeline_o, pipeline_o_consumer_state,
          pipeline_epi, pipeline_epi_producer_state,
          epilogue);
      return;
    }
    else {
    int mask_tile_count = Mask{}.get_trip_count(blk_coord, TileShape{}, problem_shape);

    int thread_idx = threadIdx.x % (4 * cutlass::NumThreadsPerWarp);

    Tensor tStS = partition_fragment_C(typename CollectiveMmaQK::TiledMma{}, select<0,1>(TileShapeQK{}));

    Tensor cS = make_identity_tensor(select<0,1>(TileShapeQK{}));
    Tensor tScS = typename CollectiveMmaQK::TiledMma{}.get_slice(cta_mma_rank()).partition_C(cS);

    Tensor tStS_v = tStS.compose(make_layout(make_shape(_128{}, _2{})));
    Tensor tScS_v = tScS.compose(make_layout(make_shape(_128{}, _2{})));

    using TMEM_LOAD_V = SM100_TMEM_LOAD_32dp32b2x;   // 4x32 threads with 2 cols of 32b elem

    auto tiled_tmem_loadv = make_tmem_copy(TMEM_LOAD_V{}, tStS_v);
    auto thr_tmem_loadv  = tiled_tmem_loadv.get_slice(thread_idx);

    Tensor tTMEM_LOADVtS = thr_tmem_loadv.partition_S(tStS_v);
    Tensor tTMEM_LOADVcS = thr_tmem_loadv.partition_D(tScS_v);

    Tensor tTMEM_LOADVtS0 = tTMEM_LOADVtS;
    tTMEM_LOADVtS0.data() = tTMEM_LOADVtS0.data().get() + uint32_t(TmemAllocation::V0);
    Tensor tTMEM_LOADVtS1 = tTMEM_LOADVtS;
    tTMEM_LOADVtS1.data() = tTMEM_LOADVtS1.data().get() + uint32_t(TmemAllocation::V1);

    // ignore first signal from softmax as no correction is required
    pipeline_s0_c.consumer_wait(pipeline_s0_c_consumer_state);
    pipeline_s0_c.consumer_release(pipeline_s0_c_consumer_state);
    ++pipeline_s0_c_consumer_state;

    pipeline_s1_c.consumer_wait(pipeline_s1_c_consumer_state);

    // handle the last iteration differently (i.e. tmem_load/stsm for epi)
    mask_tile_count -= 1;

    CUTLASS_PRAGMA_NO_UNROLL
    for (; mask_tile_count > 0; mask_tile_count -= 1) {

      pipeline_s0_c.consumer_wait(pipeline_s0_c_consumer_state);

      Tensor tTMEM_LOADVrS = make_tensor<ElementQK>(shape(tTMEM_LOADVcS));

      // read row_wise new global max
      copy(tiled_tmem_loadv, tTMEM_LOADVtS0, tTMEM_LOADVrS);

      // e^(scale * (old_max - new_max)
      float scale = (tTMEM_LOADVrS(kIdxOldRowMax) == tTMEM_LOADVrS(kIdxNewRowMax)) ? 1.0f : apxinf_ex2_approx(params.scale_softmax_log2 * (tTMEM_LOADVrS(kIdxOldRowMax) - tTMEM_LOADVrS(kIdxNewRowMax)));

      pipeline_o.consumer_wait(pipeline_o_consumer_state);

      bool warp_do_correction = __any_sync(0xFFFFFFFF, scale != 1.0f);
      if (warp_do_correction) {
        correction_rescale(scale, uint32_t(TmemAllocation::O0));
      }

      pipeline_s1_c.consumer_release(pipeline_s1_c_consumer_state);
      ++pipeline_s1_c_consumer_state;

      cutlass::arch::fence_view_async_tmem_store();

      pipeline_o.consumer_release(pipeline_o_consumer_state);
      ++pipeline_o_consumer_state;

      pipeline_s1_c.consumer_wait(pipeline_s1_c_consumer_state);

      copy(tiled_tmem_loadv, tTMEM_LOADVtS1, tTMEM_LOADVrS);

      scale = (tTMEM_LOADVrS(kIdxOldRowMax) == tTMEM_LOADVrS(kIdxNewRowMax)) ? 1.0f : apxinf_ex2_approx(params.scale_softmax_log2 * (tTMEM_LOADVrS(kIdxOldRowMax) - tTMEM_LOADVrS(kIdxNewRowMax)));

      pipeline_o.consumer_wait(pipeline_o_consumer_state);

      warp_do_correction = __any_sync(0xFFFFFFFF, scale != 1.0f);
      if (warp_do_correction) {
        correction_rescale(scale, uint32_t(TmemAllocation::O1));
      }

      pipeline_s0_c.consumer_release(pipeline_s0_c_consumer_state);
      ++pipeline_s0_c_consumer_state;

      cutlass::arch::fence_view_async_tmem_store();

      pipeline_o.consumer_release(pipeline_o_consumer_state);
      ++pipeline_o_consumer_state;
    }

    pipeline_s1_c.consumer_release(pipeline_s1_c_consumer_state);
    ++pipeline_s1_c_consumer_state;

    // do the final correction to O1
    // better to somehow special-case it in the loop above
    // doesn't matter for non-persistent code, but if it were
    // persistent we do not want to release O too early

    pipeline_s0_c.consumer_wait(pipeline_s0_c_consumer_state);

    // read from V0
    // read row_sum and final row_max here
    Tensor tTMEM_LOADVrS = make_tensor<ElementQK>(shape(tTMEM_LOADVcS));
    copy(tiled_tmem_loadv, tTMEM_LOADVtS0, tTMEM_LOADVrS);

    pipeline_s0_c.consumer_release(pipeline_s0_c_consumer_state);
    ++pipeline_s0_c_consumer_state;

    pipeline_o.consumer_wait(pipeline_o_consumer_state);
    pipeline_epi.producer_acquire(pipeline_epi_producer_state);
    // store to epi smem

    // loop:
    //    TMEM_LOAD
    //    FMUL2 scale = 1 / global_sum * out_quant_scale
    //    F2FP
    //    store to smem
    Tensor sO = make_epilogue_smem_tensor(shared_storage_epi);
    Tensor gLSE = make_tensor(make_gmem_ptr(epilogue.params.ptr_LSE), cute::make_shape(cute::size<0>(problem_shape), cute::get<3>(problem_shape)), epilogue.params.dLSE);

    correction_epilogue(apxinf_normalize_scale(params.scale_output, tTMEM_LOADVrS(kIdxFinalRowSum)), _0{}, sO);

    if (epilogue.params.ptr_LSE != nullptr) {
      int row_idx = get<0>(tTMEM_LOADVcS(_0{})) + get<0>(TileShape{}) * get<0>(blk_coord);

      int row_offset = 0;
      if constexpr (is_variable_length_v<tuple_element_t<0, ParamsProblemShape>>) {
        row_offset = get<0>(params_problem_shape).cumulative_length[get<2,1>(blk_coord)];
      }

      ElementPV lse = apxinf_lse(tTMEM_LOADVrS(kIdxFinalRowSum),
                                 tTMEM_LOADVrS(kIdxFinalRowMax),
                                 params.scale_softmax,
                                 params.scale_softmax_log2);

      if (row_idx < int(cute::size<0>(problem_shape))) {
        gLSE(row_idx + row_offset, get<2>(blk_coord)) = lse;
      }
    }

    cutlass::arch::fence_view_async_tmem_load();

    pipeline_o.consumer_release(pipeline_o_consumer_state);
    ++pipeline_o_consumer_state;

    pipeline_epi.producer_commit(pipeline_epi_producer_state);
    ++pipeline_epi_producer_state;

    pipeline_s1_c.consumer_wait(pipeline_s1_c_consumer_state);

    // load from V1
    copy(tiled_tmem_loadv, tTMEM_LOADVtS1, tTMEM_LOADVrS);

    pipeline_s1_c.consumer_release(pipeline_s1_c_consumer_state);
    ++pipeline_s1_c_consumer_state;

    pipeline_o.consumer_wait(pipeline_o_consumer_state);
    pipeline_epi.producer_acquire(pipeline_epi_producer_state);

    correction_epilogue(apxinf_normalize_scale(params.scale_output, tTMEM_LOADVrS(kIdxFinalRowSum)), _1{}, sO);

    if (epilogue.params.ptr_LSE != nullptr) {
      int row_idx = get<0>(tTMEM_LOADVcS(_0{})) + get<0>(TileShape{}) * get<0>(blk_coord) + get<0>(TileShapeQK{});

      ElementPV lse = apxinf_lse(tTMEM_LOADVrS(kIdxFinalRowSum),
                                 tTMEM_LOADVrS(kIdxFinalRowMax),
                                 params.scale_softmax,
                                 params.scale_softmax_log2);

      int row_offset = 0;
      if constexpr (is_variable_length_v<tuple_element_t<0, ParamsProblemShape>>) {
        row_offset = get<0>(params_problem_shape).cumulative_length[get<2,1>(blk_coord)];
      }

      if (row_idx < int(cute::size<0>(problem_shape))) {
        gLSE(row_idx + row_offset, get<2>(blk_coord)) = lse;
      }
    }

    cutlass::arch::fence_view_async_tmem_load();

    pipeline_o.consumer_release(pipeline_o_consumer_state);
    ++pipeline_o_consumer_state;

    pipeline_epi.producer_commit(pipeline_epi_producer_state);
    ++pipeline_epi_producer_state;
    }  // else (two Q stages)
  }


  template<
    class BlkCoord, class ProblemShape, class ParamsProblemShape,
    class TensorStorageEpi, class CollectiveEpilogue
  >
  CUTLASS_DEVICE auto
  correction_empty(
      BlkCoord const& blk_coord,
      Params const& params, ProblemShape const& problem_shape,
      ParamsProblemShape const& params_problem_shape,
      TensorStorageEpi& shared_storage_epi,
      PipelineE& pipeline_epi, typename PipelineE::PipelineState& pipeline_epi_producer_state,
      CollectiveEpilogue& epilogue) {

    pipeline_epi.producer_acquire(pipeline_epi_producer_state);

    Tensor sO = make_tensor(make_smem_ptr(shared_storage_epi.smem_o.data()), typename TensorStorageEpi::SmemLayoutO{});
    Tensor gLSE = make_tensor(make_gmem_ptr(epilogue.params.ptr_LSE), cute::make_shape(cute::size<0>(problem_shape), cute::get<3>(problem_shape)), epilogue.params.dLSE);
    float lse = -INFINITY;
    int thread_idx = threadIdx.x % (4 * NumThreadsPerWarp);

#define DSHOW(x) print(#x ": "); print(x); print("\n")
#if APXINF_DEBUG_SHOW_SO
    if (threadIdx.x % 128 == 0 && block0()) {
      DSHOW(sO);
    }
#endif
#if 1

    using ElementOut = typename CollectiveEpilogue::ElementOut;
    auto tiled_copy = make_cotiled_copy(
        Copy_Atom<UniversalCopy<uint32_t>, ElementOut>{},
        make_ordered_layout(make_shape(_128{}, Int<sizeof(uint32_t) / sizeof(ElementOut)>{}), Step<_1, _0>{}),
        sO.layout());

    auto thr_copy = tiled_copy.get_slice(thread_idx);
    auto tOgO = thr_copy.partition_D(sO);
    auto tOrO = make_tensor<ElementOut>(shape(tOgO(_,_,_,_0{})));
    clear(tOrO);

    copy(tiled_copy, tOrO, tOgO(_,_,_,_0{}));
#endif

    if (epilogue.params.ptr_LSE != nullptr) {
      int row_idx = thread_idx + get<0>(TileShape{}) * get<0>(blk_coord);

      int row_offset = 0;
      if constexpr (is_variable_length_v<tuple_element_t<0, ParamsProblemShape>>) {
        row_offset = get<0>(params_problem_shape).cumulative_length[get<2,1>(blk_coord)];
      }

      if (row_idx < int(cute::size<0>(problem_shape))) {
        gLSE(row_idx + row_offset, get<2>(blk_coord)) = lse;
      }
    }

    if constexpr (kSingleQStage) {
      cutlass::arch::fence_view_async_shared();
      pipeline_epi.producer_commit(pipeline_epi_producer_state);
      ++pipeline_epi_producer_state;
      return;
    }
    else {

    pipeline_epi.producer_commit(pipeline_epi_producer_state);
    ++pipeline_epi_producer_state;

    copy(tiled_copy, tOrO, tOgO(_,_,_,_1{}));
    cutlass::arch::fence_view_async_shared();
    pipeline_epi.producer_acquire(pipeline_epi_producer_state);

    if (epilogue.params.ptr_LSE != nullptr) {
      int row_idx = thread_idx + get<0>(TileShape{}) * get<0>(blk_coord) + get<0>(TileShapeQK{});

      int row_offset = 0;
      if constexpr (is_variable_length_v<tuple_element_t<0, ParamsProblemShape>>) {
        row_offset = get<0>(params_problem_shape).cumulative_length[get<2,1>(blk_coord)];
      }

      if (row_idx < int(cute::size<0>(problem_shape))) {
        gLSE(row_idx + row_offset, get<2>(blk_coord)) = lse;
      }
    }

    cutlass::arch::fence_view_async_shared();
    pipeline_epi.producer_commit(pipeline_epi_producer_state);
    ++pipeline_epi_producer_state;
    }  // else (two Q stages)
  }

};

}  // namespace cutlass::fmha::collective
