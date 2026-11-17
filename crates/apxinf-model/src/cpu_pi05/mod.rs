//! Self-contained FP32 CPU port of PI0.5. Config, checkpoint mapping, model math
//! and execution are owned here. Copies originated in pi05/ and hip_pi05/;
//! ports share public contracts and accuracy fixtures, never model code.
pub mod config;
mod load;
mod model;
mod runner;
mod weights;
pub use config::{Config, GemmaConfig};
pub(crate) use load::load;
