//! Cosmos3-Edge (DROID) direct-policy model family — policy path only.
#[cfg(feature = "cuda")]
mod backend;
mod config;
#[cfg(feature = "cuda")]
mod model;
#[cfg(feature = "cuda")]
mod weights;

pub use config::Cosmos3EdgeConfig;

#[cfg(feature = "cuda")]
pub(crate) mod load;
#[cfg(feature = "cuda")]
mod model_runner;

#[cfg(feature = "cuda")]
pub(super) fn register_builtin() {
    crate::registry::register("cuda_cosmos3_edge", load::load_registered);
    crate::registry::register("cosmos3-edge", load::load_registered);
}
