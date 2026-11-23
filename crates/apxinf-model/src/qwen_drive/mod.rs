//! Qwen-Drive planning VLA: BF16 vision/backbone/expert computation and runner.
//! Text generation is internal to reasoning planning. No text/VQA deployment API.
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
#[cfg(feature = "cuda")]
pub const FIXED_SCENE_TOKENS: usize = 3387;
pub use config::QwenDriveConfig;
#[cfg(feature = "cuda")]
pub use model_runner::QwenDriveModelRunner;
