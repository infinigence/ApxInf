//! Execution lifecycle; model and weights never depend on this module.

mod prepare;
mod runner;

pub use prepare::HipPi05Prepared;
pub use runner::HipPi05Runner;
