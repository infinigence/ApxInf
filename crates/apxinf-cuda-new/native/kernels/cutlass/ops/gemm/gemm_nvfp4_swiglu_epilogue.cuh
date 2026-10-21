// Copyright 2026 ApxInf contributors.
//
// Bit-level transcription of the patched dense_swiglu epilogue.
//
// Derived from the AOT object's SASS and cross-checked against
// `crates/apxinf-cuda/aot/exporters/export_qwen38_nvfp4_swiglu.py`. ApxInf code;
// it only includes CUTLASS/CuTe headers, which keep their own NVIDIA
// BSD-3-Clause notices (see `kernels/cutlass/README.md`).
// Per element the AOT does, in this exact order:
//
//   FMUL   gate_f = alpha * acc_gate        FMUL  up_f = alpha * acc_up
//   F2F.BF16.F32 + SHF.L  gate = f32(bf16_rn(gate_f))
//   FMUL + MUFU.EX2       t = ex2.approx(gate * -1.4426950216293334961)
//   FADD                  d = t + 1
//   MUFU.RCP/FCHK/FFMA..  act = div.rn.f32(gate, d)
//   F2FP.BF16.F32         up_b = bf16_rn(up_f) ; act_b = bf16_rn(act)
//   HMUL2.BF16_V2         out  = up_b * act_b   (bf16 multiply)
//
// then per 16-element vector:
//
//   FMNMX3.NAN tree       amax = max(|out_i|, 0.0f)
//   FMUL x2               pv   = norm_const * (amax * (1/6))
//   F2FP.SATFINITE.E4M3   sfc  = cvt.rn.satfinite.e4m3(pv)     -> stored
//   FMUL                  eff  = f32(sfc) * down_input_scale
//   FSETP.GT + div.rn     s    = eff > 0 ? div.rn(1, eff) : 0
//   FMNMX.NAN (min)       s    = fmin(s, FLT_MAX)
//   FMUL                  q_i  = out_i * s
//   F2FP.SATFINITE.E2M1   nibble = cvt.rn.satfinite.e2m1(q_i)
//
// norm_const = MUFU.RCP(down_input_scale) -- the approximate reciprocal, not
// an exact division; that one survived the exporter patch untouched.
#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

__device__ __forceinline__ float apx_ex2_approx(float x) {
  float r;
  asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

__device__ __forceinline__ float apx_rcp_approx(float x) {
  float r;
  asm("rcp.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

__device__ __forceinline__ float apx_div_rn(float a, float b) {
  float r;
  asm("div.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

// f32 -> bf16 (round-to-nearest-even) -> back to f32.
__device__ __forceinline__ float apx_bf16_round(float x) {
  unsigned short h;
  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(h) : "f"(x));
  unsigned int bits = (unsigned int)h << 16;
  return __int_as_float((int)bits);
}

__device__ __forceinline__ float apx_fmax_nan(float a, float b) {
  float r;
  asm("max.NaN.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

__device__ __forceinline__ float apx_fmin_nan(float a, float b) {
  float r;
  asm("min.NaN.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

// cvt.rn.satfinite.e4m3x2.f32 -- returns the single low byte.
__device__ __forceinline__ uint8_t apx_f32_to_e4m3(float x) {
  unsigned short packed;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
      : "=h"(packed) : "f"(0.0f), "f"(x));
  return (uint8_t)(packed & 0xffu);
}

__device__ __forceinline__ float apx_e4m3_to_f32(uint8_t byte) {
  unsigned short packed = (unsigned short)byte;
  unsigned int out;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(out) : "h"(packed));
  __half h = __ushort_as_half((unsigned short)(out & 0xffffu));
  return __half2float(h);
}

// cvt.rn.satfinite.e2m1x2.f32 -- lo gets `a`, hi gets `b`; returns one byte.
__device__ __forceinline__ uint8_t apx_f32x2_to_e2m1x2(float lo, float hi) {
  unsigned short packed;
  asm("{ .reg .b8 t;                              \n"
      "  cvt.rn.satfinite.e2m1x2.f32 t, %2, %1;   \n"
      "  mov.b16 %0, {t, 0};                     }\n"
      : "=h"(packed) : "f"(lo), "f"(hi));
  return (uint8_t)(packed & 0xffu);
}

// Full per-element SwiGLU. Returns the bf16-rounded f32 the kernel keeps in
// tCompute.
__device__ __forceinline__ float apx_swiglu(float acc_up, float acc_gate,
                                            float alpha) {
  const float up_f = alpha * acc_up;
  const float gate_f = alpha * acc_gate;
  const float gate = apx_bf16_round(gate_f);
  const float t = apx_ex2_approx(gate * -1.4426950216293334961f);
  const float d = t + 1.0f;
  const float act = apx_div_rn(gate, d);
  const float up_b = apx_bf16_round(up_f);
  const float act_b = apx_bf16_round(act);
  return apx_bf16_round(up_b * act_b);
}

// Pair-at-a-time SwiGLU. Identical arithmetic to apx_swiglu, but emits the
// AOT's instruction mix instead of a scalar-per-element one:
//   * F2FP.BF16.F32.PACK_AB rounds two values per instruction (3 per pair
//     instead of 6 separate F2F.BF16.F32)
//   * HMUL2.BF16_V2 performs `bf16_rn(up * act)` in one instruction. A BF16
//     multiply is exactly an f32 multiply followed by round-to-nearest-even to
//     BF16: both significands are 8 bits, so the f32 product is exact and both
//     forms round the same exact value. The AOT's SASS uses HMUL2 here too.
__device__ __forceinline__ void apx_swiglu_pair(float acc_up0, float acc_up1,
                                                float acc_gate0, float acc_gate1,
                                                float alpha, float& out0,
                                                float& out1) {
  const __nv_bfloat162 gb =
      __floats2bfloat162_rn(alpha * acc_gate0, alpha * acc_gate1);
  const float g0 = __bfloat162float(gb.x);
  const float g1 = __bfloat162float(gb.y);
  const float d0 = apx_ex2_approx(g0 * -1.4426950216293334961f) + 1.0f;
  const float d1 = apx_ex2_approx(g1 * -1.4426950216293334961f) + 1.0f;
  const __nv_bfloat162 ab =
      __floats2bfloat162_rn(apx_div_rn(g0, d0), apx_div_rn(g1, d1));
  const __nv_bfloat162 ub =
      __floats2bfloat162_rn(alpha * acc_up0, alpha * acc_up1);
  const __nv_bfloat162 ob = __hmul2(ub, ab);
  out0 = __bfloat162float(ob.x);
  out1 = __bfloat162float(ob.y);
}
