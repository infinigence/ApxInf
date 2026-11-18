//! MiniMax-H3-Image text-to-image on the cuda-new backend: INT8 ConvRot DiT, streamed Qwen3-VL text
//! encoder and the ViT VAE decoder, behind the shared [`crate::image::ImageRuntime`] contract.
#[cfg(feature = "cuda")]
mod backend;
#[cfg(feature = "cuda")]
mod config;
#[cfg(feature = "cuda")]
pub(crate) mod load;
#[cfg(feature = "cuda")]
mod model;
#[cfg(feature = "cuda")]
mod model_runner;
mod package;
#[cfg(all(test, feature = "cuda"))]
mod tests;
#[cfg(feature = "cuda")]
mod weights;

#[cfg(feature = "cuda")]
pub use config::PDD_STEPS;

pub use package::{MANIFEST, SCHEMA};

/// Whether `dir` is a MiniMax-H3-Image package (`AutoModel` detection).
pub(crate) use package::is_package;

#[cfg(feature = "cuda")]
pub(super) fn register_builtin() {
    crate::registry::register("cuda_minimax_h3_image", load::load_registered);
    crate::registry::register("minimax-h3-image", load::load_registered);
}
