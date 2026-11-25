//! MolmoAct2 (LIBERO continuous-action checkpoint) native CUDA model family, BF16 profile.
//!
//! Observation-to-action inference: the native vision encoder, pooling connector and VLM prefill produce
//! the 36 per-layer K/V tensors, a context adapter projects them for the action expert, and the expert
//! runs ten forward Euler flow steps. The family is selected as `molmoact2` (`cuda_molmoact2`) and
//! returns normalized padded actions `[10, 32]`; the Python policy owns the prompt, the image resize,
//! state discretization and the physical action units. By default one captured whole-model CUDA graph runs
//! the traversal (prepared eager execution, the same traversal enqueued from the host, stays selectable); see
//! `doc/molmoact2.md`.

#[cfg(feature = "cuda")]
mod backend;
// Host-side modules whose only consumers are the CUDA model; they still compile (and their CPU tests run)
// without the feature, where nothing uses them.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
mod config;
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
mod geometry;
#[cfg(feature = "cuda")]
pub(crate) mod load;
#[cfg(feature = "cuda")]
pub(super) mod model;
#[cfg(feature = "cuda")]
mod model_runner;
#[cfg(feature = "cuda")]
mod weights;

pub use config::MolmoAct2Config;

#[cfg(feature = "cuda")]
pub(super) fn register_builtin() {
    crate::registry::register("cuda_molmoact2", load::load_registered);
    crate::registry::register("molmoact2", load::load_registered);
}
