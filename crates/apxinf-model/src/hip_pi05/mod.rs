//! π0.5 on AMD GPUs, written directly against HIP (`apxinf_hip::native`) the
//! way `pi05/` is written against `apxinf-cuda-new`.
//!
//! A self-contained port: it owns its config ([`config`]), checkpoint reader
//! ([`checkpoint`]), device weights, layer computations, inference plan and
//! loader, and imports no other model module. Agreement with the CUDA
//! implementation comes from both passing the same accuracy fixtures, not from
//! shared code. The one touchpoint is the frontend contract: `LoadOptions::config`
//! is typed as `pi05::Pi05Config`, which [`config::Config::from_frontend`] reads.
//!
//! BF16 only, eager for now; see [`runner`] for what graph capture needs.

pub mod checkpoint;
pub mod config;
mod load;
mod model;
pub mod runner;
mod weights;

pub use config::Config;
pub(crate) use load::load;
pub use load::load_runner;
pub use model::{HipPi05Model, PrefixKvCache};
pub use runner::{HipPi05Prepared, HipPi05Runner};
