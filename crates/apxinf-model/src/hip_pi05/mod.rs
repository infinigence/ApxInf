//! π0.5 on AMD GPUs, written directly against HIP (`apxinf_hip::native`) the
//! way `pi05/` is written against `apxinf-cuda-new`.
//!
//! A self-contained port: it owns its config ([`config`]), checkpoint reader
//! ([`weights::host`]), device weights, layer computations, inference plan and
//! loader, and imports no other model module. Agreement with the CUDA
//! implementation comes from both passing the same accuracy fixtures, not from
//! shared code. Frontend config overrides (`LoadOptions::config`, typed as
//! `pi05::Pi05Config`) are converted by `AutoModel`, outside this module, and
//! arrive here as this port's own [`Config`].
//!
//! The module layout mirrors PI0.5: `model/` owns the forward math,
//! `model_runner/` owns request validation, the plan cache, execution and
//! graph capture, `weights/` owns checkpoint mapping and the device weight
//! tree. BF16 only; see [`model_runner`] for the execution policies.

pub mod config;
mod load;
pub mod math;
mod model;
mod model_runner;
mod weights;

pub use config::Config;
pub(crate) use load::load;
pub use load::load_runner;
pub use model::{HipPi05Model, PrefixKvCache};
pub use model_runner::{HipPi05Prepared, HipPi05Runner};
