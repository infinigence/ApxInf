//! Execution lifecycle; model and weights never depend on this module.

mod prepare;
mod runner;
pub use prepare::HipGr00tPrepared;
pub use runner::HipGr00tRunner;
