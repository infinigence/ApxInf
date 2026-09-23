//! Qwen-Drive planning VLA: BF16 vision/backbone/expert computation and runner.
//! Text generation is internal to reasoning planning. No text/VQA deployment API.
#[cfg(feature = "cuda")]
pub(crate) fn diagnostics_enabled() -> bool {
    std::env::var_os("APXINF_QWEN_DIAG").is_some()
}
#[cfg(feature = "cuda")]
macro_rules! qdiag {
    ($($arg:tt)*) => { if $crate::qwen_drive::diagnostics_enabled() { eprintln!($($arg)*); } };
}
#[cfg(feature = "cuda")]
pub(crate) mod backend;
pub mod config;
pub mod inputs;
#[cfg(feature = "cuda")]
pub(crate) mod load;
#[cfg(feature = "cuda")]
pub(crate) mod model;
#[cfg(feature = "cuda")]
pub mod model_runner;
pub mod weights;
pub use config::QwenDriveConfig;
#[cfg(feature = "cuda")]
pub use model_runner::QwenDriveModelRunner;
