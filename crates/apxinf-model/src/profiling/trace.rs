//! Profiler trace ranges — hardware-neutral named time-span markers for an
//! external profiler.
//!
//! These emit no numbers themselves; they annotate the timeline so an external
//! profiler (NVTX / Nsight Systems on CUDA) can attribute time to named
//! regions. CUDA builds route through the accelerator seam; non-CUDA builds
//! provide a no-op stub so model code can call
//! `crate::profiling::trace::range("name")` unconditionally.

#[cfg(feature = "cuda")]
pub use crate::accelerator::cuda::nvtx::{range, Range};

#[cfg(not(feature = "cuda"))]
pub struct Range;

#[cfg(not(feature = "cuda"))]
#[inline]
pub fn range(_name: &str) -> Range {
    Range
}
