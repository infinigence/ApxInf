//! The loaded π0.5 model on a HIP device: the forward math.
//!
//! [`HipPi05Model`] owns the weights and computes the stages — vision, prefix
//! K/V, flow matching — while request-shaped state (input buffers, the
//! activation arena, prepared plans) belongs to
//! [`crate::hip_pi05::model_runner`].

mod model;

pub use model::{HipPi05Model, PrefixKvCache};
