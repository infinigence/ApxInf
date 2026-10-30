//! WallOSS vision-language-action family.
//!
//! Module owners (see `doc/reference/model-layer-architecture.md`):
//! `model/` — forward order, modality connections, and the flow schedule;
//! `model_runner/` — `VlaRuntime` implementation, preparation policy, and the
//! bounded plan cache; `weights/` — checkpoint mapping and device
//! representations; `config.rs`, `load.rs`, `backend.rs` — family config,
//! assembly, and the CUDA seam.

#[cfg(feature = "cuda")]
mod backend;
#[cfg(any(feature = "cuda", test))]
mod config;
#[cfg(any(feature = "cuda", test))]
mod fp8;
#[cfg(any(feature = "cuda", test))]
mod geometry;
#[cfg(feature = "cuda")]
mod load;
#[cfg(feature = "cuda")]
mod model;
#[cfg(feature = "cuda")]
mod model_runner;
#[cfg(any(feature = "cuda", test))]
mod schedule;
#[cfg(any(feature = "cuda", test))]
mod weights;

#[cfg(any(feature = "cuda", test))]
mod workspace;

#[cfg(feature = "cuda")]
pub(crate) use config::WallossImageProcessorConfig;
#[cfg(any(feature = "cuda", test))]
pub(crate) use config::{WallossConfig, WallossTextConfig, WallossVisionConfig};

#[cfg(feature = "cuda")]
pub(crate) fn register_builtin() {
    crate::registry::register("walloss-cuda", load::load_registered);
    crate::registry::register("wall-oss-cuda", load::load_registered);
    crate::registry::register("wall_oss_05-cuda", load::load_registered);
}
