//! Profiler trace ranges — hardware-neutral named time-span markers for an
//! external profiler.
//!
//! These emit no numbers themselves; they annotate the timeline so an external
//! profiler (NVTX / Nsight Systems on CUDA, MTTX / msys on MUSA, os_signpost /
//! Instruments on Apple) can attribute time to named regions. Each backend's
//! client is reached through the accelerator seam; a build with none provides
//! a no-op stub so model code can call `crate::profiling::trace::range("name")`
//! unconditionally.
//!
//! One backend is chosen for the whole process, and the marker API carries no
//! device: when several are compiled in, CUDA wins over MUSA and MUSA over
//! Apple, and only the winner's profiler sees the ranges.

#[cfg(feature = "cuda")]
pub use apxinf_cuda_new::nvtx::{range, Range};

/// MUSA speaks NVTX's protocol, so the model's ranges reach msys unchanged.
#[cfg(all(feature = "musa", not(feature = "cuda")))]
pub use crate::accelerator::musa::mttx::{range, Range};

#[cfg(all(feature = "mlx-signpost", not(any(feature = "cuda", feature = "musa"))))]
pub use crate::accelerator::mlx::signpost::{range, Range};

#[cfg(not(any(feature = "cuda", feature = "musa", feature = "mlx-signpost")))]
pub struct Range;

#[cfg(not(any(feature = "cuda", feature = "musa", feature = "mlx-signpost")))]
#[inline]
pub fn range(_name: &str) -> Range {
    Range
}
