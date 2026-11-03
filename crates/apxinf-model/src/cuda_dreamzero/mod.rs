//! DreamZero's joint video/action model mathematics and checkpoint ownership.
//! The BF16 path keeps the official operation boundaries, including UMT5's
//! materialized low-precision attention probabilities.
mod config;
pub use config::DreamZeroConfig;
#[cfg(feature = "cuda")]
mod backend;
#[cfg(feature = "cuda")]
pub(crate) mod load;
#[cfg(feature = "cuda")]
mod model;
#[cfg(feature = "cuda")]
mod model_runner;
mod schedule;
#[cfg(feature = "cuda")]
mod weights;

#[cfg(feature = "cuda")]
pub(super) fn register_builtin() {
    crate::registry::register("dreamzero", load::load_registered);
    crate::registry::register("dreamzero_droid", load::load_registered);
}
