//! MUSA accelerator backend for ApxInf.
//!
//! The counterpart of `apxinf-cuda` on Moore Threads hardware. Same seam, same
//! safe model-neutral operator facade, different runtime:
//!
//! | | CUDA | MUSA |
//! |---|---|---|
//! | toolkit | nvcc | mcc |
//! | arch | `sm_<cc>` | `mp_<cc>` |
//! | runtime | `cuda*` / cudart | `musa*` / musart |
//! | BLAS | cuBLAS / cuBLASLt | mublas / mublasLt |
//! | markers | NVTX | MTTX |
//!
//! Two facts about M1000 (`mp_22`) shape everything built on top of this:
//! the hardware warp is 128 lanes while every shuffle intrinsic works on
//! 32-lane groups, and there is no tensor-core path ApxInf can use — no TMA,
//! no cluster, no tcgen05, and the MMA atom is `32x32x16`.

pub mod buffer;
pub mod context;
pub mod device_caps;
mod error;
// Not `pub`: this is the raw `extern "C"` surface, and the crate's rule is
// that `ffi` is the only module allowed to declare one. Exporting it would let
// a dependent crate call `musaFree`-class symbols and then read the contracts
// that constrain them here as advice rather than as the boundary they are.
pub(crate) mod ffi;
mod graph;
pub mod kernels;
pub mod ops;
pub mod mttx;
pub mod sampling;
pub mod stream;
pub mod transfers;
pub mod tuning;
mod workspace;

pub use buffer::{MusaBuffer, MusaDeviceAddress};
pub use context::{MusaLibraryVersions, MusaContext};
pub use device_caps::{MusaArchFamily, MusaDeviceCaps};
pub use graph::{capture, CapturedGraph};
pub use stream::MusaStream;
pub use workspace::{ExecutionSession, GraphWorkspace, OperationShape};

/// Whether this build found a MUSA toolkit. Set by `build.rs`; a build
/// without one still compiles so `cargo build --workspace` works everywhere.
pub const MUSA_AVAILABLE: bool = cfg!(apxinf_musa_available);

/// Whether this build compiled the vendored mp_31 FMHA.
///
/// Carried across the crate boundary because a `cfg` does not cross it, and
/// [`D256_ROUTE`] alone cannot say *why* it is not `"mutlass"`: the arch and the
/// toolkit version are two separate gates and only one of them is visible in the
/// route. `tests/arch_gate.rs` uses this to recompute the route from the build's
/// own inputs and compare.
pub const VENDOR_FMHA_BUILT: bool = cfg!(apxinf_musa_mutlass_fmha);

/// Whether this build linked the MTTX range-marker client.
///
/// `build.rs` sets `mttx_available` only when a `mt-tools-extension` checkout
/// was found *and* its client compiled; without it the marker surface is a
/// no-op stub. A build with the client can still run with no tool attached,
/// which [`mttx::Range`] handles at runtime — so this const answers the first
/// of the three conditions in `doc/pi05-mttx.md`, and nothing answered it
/// before.
pub const MTTX_AVAILABLE: bool = cfg!(mttx_available);

/// The offload architecture this build targeted, e.g. `mp_22`.
pub const MUSA_ARCH: &str = env!("APXINF_MUSA_ARCH");

/// The generation `kernels/asm/`'s D256 attention assembly is written for.
///
/// A build-script constant, not a literal here: `build.rs` decides whether to
/// assemble with it and `kernels::attention` decides whether to launch with it,
/// and a copy in each place could drift into "the assembly was assembled for a
/// generation the launcher never asks about".
pub const ASSEMBLY_ARCH: &str = env!("APXINF_MUSA_ASSEMBLY_ARCH");

/// The generation the vendored FMHA in `kernels/mutlass_fmha/` is written for.
///
/// Same reason as [`ASSEMBLY_ARCH`]: it is used as a match pattern on the
/// launch path and as a gate in the build script.
pub const VENDOR_FMHA_ARCH: &str = env!("APXINF_MUSA_VENDOR_FMHA_ARCH");

/// Which D256 attention kernel this build has: `"assembly"`, `"mutlass"` or
/// `"unsupported"`.
///
/// **The answer, not the inputs.** `build.rs` runs
/// `build_support::musa_arch::d256_route` — the one place the rule is written —
/// and publishes what it returned, so `kernels::attention` reads a decision
/// instead of re-deriving one. A second copy of the rule is what
/// `tests/arch_gate.rs` makes unrepresentable: it recomputes the route from the
/// build's own inputs and compares, so the two answers cannot drift apart.
pub const D256_ROUTE: &str = env!("APXINF_MUSA_D256_ROUTE");

/// The published route has to be one this crate understands.
///
/// `kernels::attention` maps the string onto an enum whose catch-all is
/// `Unsupported` — a legitimate answer, so a name from a future `build.rs` would
/// look like a missing kernel rather than like a build script and a library
/// disagreeing. Checked here, at compile time, because that is the only place
/// the two are guaranteed to meet.
const _: () = {
    if !matches!(
        D256_ROUTE.as_bytes(),
        b"assembly" | b"mutlass" | b"unsupported"
    ) {
        panic!("build.rs published a D256 route this crate does not understand");
    }
};

/// Identifies the native kernel sources, so a MUSA recipe cache can never be
/// read as a CUDA one.
pub const KERNEL_BUILD_ID: &str = env!("APXINF_KERNEL_BUILD_ID");
