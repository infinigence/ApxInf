mod config;
mod load;
mod model;
mod runtime;
mod weights;

pub use config::SmolVlaConfig;
pub use model::SmolVlaModel;
pub use runtime::SmolVlaModelRunner;
pub use weights::SmolVlaWeights;

pub(super) fn register_builtin() {
    crate::registry::register("smolvla", load::load_registered);
    crate::registry::register("smolvla_libero", load::load_registered);
}
