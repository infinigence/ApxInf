//! Fast-WAM action-only inference on cuda-new.
mod backend;
mod config;
mod dit;
pub(crate) mod load;
mod model;
mod model_runner;
mod text;
mod vae;
mod weights;
pub(super) fn register_builtin() {
    crate::registry::register("fastwam", load::load_registered);
}
