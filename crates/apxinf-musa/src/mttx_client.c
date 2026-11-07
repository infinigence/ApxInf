/*
 * The one translation unit that speaks MTTX for this crate.
 *
 * Why this file has to exist
 * --------------------------
 * MTTX is header-only, and its public API is emitted as `inline static` with
 * hidden visibility: `MTTX_DECLSPEC` is `MTTX_INLINE_STATIC`, and the whole
 * implementation sits behind `#pragma GCC visibility push(hidden)`. No object
 * file exports a symbol named `mttxRangePushA`, so Rust cannot declare it
 * `extern "C"` and link it. The client's own state -- `mttxGlobals_v1`, the
 * export table and the implementation function pointers the tool installs --
 * is hidden and weak for the same reason: one instance per linkage unit, and
 * this file is that unit.
 *
 * So this file includes the client header once and re-exports a stable,
 * default-visibility C ABI for `src/mttx.rs`. It links into the final artifact
 * as an ordinary object, not a shared library: the process that pushes a range
 * is the process the tool has to attach to.
 *
 * What must NOT be defined
 * ------------------------
 *   * `MTTX_NO_IMPL` -- the API would be declared and never defined.
 *   * `MTTX_DISABLE` -- every call would compile to an inline no-op and no
 *     tool would ever be loaded.
 * Both fail the same way: the build succeeds and no label is ever recorded.
 * Only `mtToolsExt.h` is included; `mttx.hpp` is the C++ wrapper and nothing
 * here is C++.
 */

#include <mttx/mtToolsExt.h>

#include <stddef.h>

/* @param name  NUL-terminated range name. MTTX reads it during the call and
 *              keeps no reference, but the caller owns the buffer.
 * @return      Non-zero when a tool took the range. */
__attribute__((visibility("default"))) int apxinf_mttx_push(const char *name)
{
    if (name == NULL)
        return 0;
    /* The tool answers MTTX_NO_PUSH_POP_TRACKING when it is attached but has
     * push/pop tracking turned off, so "not -2" is the honest test for
     * "a range is now open". */
    return mttxRangePushA(name) != MTTX_NO_PUSH_POP_TRACKING;
}

/* Closes the innermost range on this thread. Only call it after a push that
 * returned non-zero: the stack is per thread, so an unearned pop closes some
 * other region. */
__attribute__((visibility("default"))) int apxinf_mttx_pop(void)
{
    return mttxRangePop() != MTTX_NO_PUSH_POP_TRACKING;
}

/* Whether anything pushed from here reaches a profiler.
 *
 * A pure read of the callback the client installed, with no side effects: the
 * client's one-shot initialisation is what `apxinf_mttx_push` triggers, and
 * that is the only place it is triggered from. It cannot be forced from here.
 *
 * Forcing it looks tempting — the header has an `mttxInitialize` for exactly
 * that — but the call re-dispatches through the installed table a second time,
 * and the tool behind msys' own `libmttxInjection.so` does not survive being
 * initialised that way: the process dies with SIGSEGV inside the attach, after
 * the injection library has already reported success. Measured on
 * 1.8.1-e3ea15070-internal-release-1.8.
 *
 * The three answers come from the pointer's own lifecycle. Before
 * initialisation it holds the `_impl_init` thunk; initialisation then either
 * installs the tool's callback or — having found none — sets it to NULL (see
 * `mttxSetInitFunctionsToNoops` in `mttxDetail/mttxInitDefs.h`). So:
 *
 *   -1  not initialised yet; no range has been pushed in this process
 *    0  initialised with no tool attached
 *    1  a tool attached and will record what we push
 *
 * An attached tool may still have push/pop tracking off, which a push reports
 * through its own return value rather than through this one. */
__attribute__((visibility("default"))) int apxinf_mttx_state(void)
{
    mttxRangePushA_impl_fntype installed =
        MTTX_VERSIONED_IDENTIFIER(mttxGlobals).mttxRangePushA_impl_fnptr;
    if (installed == NULL)
        return 0;
    if (installed == MTTX_VERSIONED_IDENTIFIER(mttxRangePushA_impl_init))
        return -1;
    return 1;
}
