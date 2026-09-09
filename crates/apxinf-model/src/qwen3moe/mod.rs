//! Qwen3-MoE (`model_type = "qwen3_moe"`) with AutoAWQ INT4 weights.
//!
//! Scope is deliberately narrow (see `qwen3-30b-a3b-goal.md`): one checkpoint
//! family, one quantization scheme, one target device class. Nothing here is
//! shared with the other model implementations in this crate, so changes stay
//! contained to this directory.
//!
//! Layout:
//!
//! * [`config`] — `config.json` schema plus the AutoAWQ quantization block.
//! * [`weights`] — checkpoint packer and device-resident tensors. Weights stay
//!   in the on-disk AWQ nibble order; only `q|k|v` and per-expert `gate|up`
//!   are concatenated on the host.
//! * [`runtime`] — the two execution paths: an eager dequant + dense-GEMM
//!   prefill and a fixed-shape, CUDA-graph capturable W4A16 GEMV decode.
//! * [`general`] — [`LlmTrait`](crate::llm_trait::LlmTrait) integration and the
//!   `AutoModel` registry factory.
//!
//! Only [`config`] builds without the `cuda` feature; everything below it
//! needs device buffers and kernels.

pub mod config;
#[cfg(feature = "cuda")]
pub mod general;
#[cfg(feature = "cuda")]
pub mod runtime;
#[cfg(feature = "cuda")]
mod trace;
#[cfg(feature = "cuda")]
pub mod weights;

pub use config::{Qwen3MoeConfig, Qwen3MoeQuantization};
#[cfg(feature = "cuda")]
pub use general::{load_qwen3moe, Qwen3Moe, DEFAULT_MAX_SEQ_LEN};
#[cfg(feature = "cuda")]
pub use runtime::Qwen3MoeRuntime;
#[cfg(feature = "cuda")]
pub use weights::{AwqLinear, Qwen3MoeLayerWeights, Qwen3MoeWeights};

/// Registry name published by this family; it matches the `model_type` string
/// that `AutoModel` reads out of `config.json`.
pub const MODEL_TYPE: &str = "qwen3_moe";

#[cfg(feature = "cuda")]
pub(crate) fn register_builtin() {
    crate::registry::register(MODEL_TYPE, general::load_qwen3moe);
}
