//! Fast-WAM action-only inference on cuda-new.
mod backend;
mod config;
pub(crate) mod load;
mod model;
mod model_runner;
mod weights;
pub(crate) fn register_builtin() {
    crate::registry::register("cuda_fastwam", load::load_registered);
    crate::registry::register("fastwam", load::load_registered);
}
