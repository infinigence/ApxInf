/* Minimal replacement for glibc's `<bits/math-vector.h>` on aarch64.
 *
 * MUSA 5.1.0's `mcc` cannot compile *any* `-x musa` translation unit against
 * this host's glibc: its device pass force-includes
 * `__clang_musa_runtime_wrapper.h`, which reaches `<cmath>` -> `<math.h>` ->
 * `<bits/math-vector.h>`. That header emits the aarch64 NEON and SVE vector
 * typedefs whenever the *compiler version* is recent enough
 * (`__glibc_clang_prereq (8, 0)`), without ever checking whether the current
 * target has those features. `mp_22` has neither, so the compile dies with
 * "'__neon_vector_type__' attribute is not supported on targets missing 'neon'
 * or 'mve'" -- including for a translation unit with no includes of its own,
 * because `mcc` supplies the include.
 *
 * The real header is guarded only by a version test inside `features.h`, so
 * neither `-D__glibc_clang_prereq(a,b)=0` nor `-march`/`-Xclang
 * -target-feature` reaches it. Shadowing the header is the available lever.
 *
 * glibc anticipates exactly this case. The tail of the real header says that
 * if vector types are unsupported in the working compiler, there is "no choice
 * but to omit vector math declarations" -- so defining neither
 * `__ADVSIMD_VEC_MATH_SUPPORTED` nor `__SVE_VEC_MATH_SUPPORTED`, which is what
 * omitting those typedefs accomplishes, is a supported outcome rather than a
 * truncation. The `__DECL_SIMD_*` macros that `<math.h>` does consume come
 * from `libm-simd-decl-stubs.h`, which is still included here from glibc.
 *
 * `build_support/host_math_shim.rs` decides whether this directory is added to
 * the include path, by trying a MUSA compile without it first. A toolchain that
 * does not need the workaround does not get it.
 */
#ifndef _MATH_VECTOR_H
#define _MATH_VECTOR_H 1

#include <bits/libm-simd-decl-stubs.h>

#endif /* _MATH_VECTOR_H */
