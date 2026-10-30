//! Cosmos Policy's LIBERO direct-policy model family.
#[cfg(feature = "cuda")]
mod backend;
mod config;
#[cfg(feature = "cuda")]
mod model;
#[cfg(feature = "cuda")]
mod weights;

pub use config::CosmosPolicyConfig;

#[cfg(feature = "cuda")]
pub(crate) mod load;
#[cfg(feature = "cuda")]
mod model_runner;

#[cfg(feature = "cuda")]
pub(super) fn register_builtin() {
    crate::registry::register("cuda_cosmos_policy", load::load_registered);
    crate::registry::register("cosmos-policy", load::load_registered);
}
