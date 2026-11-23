//! Execution lifecycle; model and weights never depend on this module.
mod prepare;
mod runner;
pub use prepare::CapturedGraph;
pub use runner::Pi05ModelRunner;
