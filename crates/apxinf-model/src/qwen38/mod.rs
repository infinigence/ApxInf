//! Qwen3.8-27B-NVFP4 on Jetson Thor (sm_110).
//!
//! 64 layers: 48 Gated DeltaNet (linear attention) + 16 full attention, mixed
//! NVFP4/FP8/BF16 from a ModelOpt checkpoint. Module owners follow
//! `doc/model-layer-architecture.md`:
//!
//! - [`config`] — checkpoint shape constants;
//! - [`weights`] — checkpoint interpretation, packing and the device weight
//!   tree;
//! - [`model`] — prefill/decode dataflow and the scratch it binds;
//! - [`model_runner`] — CUDA-graph capture, replay decode and session reset;
//! - [`backend`] — the family CUDA seam (upload/zero/readback/views);
//! - [`llm`] — the [`crate::LlmTrait`] wrapper `Qwen38`.
//!
//! Numeric contracts: the quantizers match vLLM's encodings bit for bit
//! (RNE FP4, SATFINITE E4M3, double-rounded SwiGLU — see the
//! `qwen38_*_quant_contract` test suites in `apxinf-cuda-new`), and the
//! FlashInfer GDN scan is shadow-checked against the reference chunked scan.

#[cfg(feature = "cuda")]
mod backend;
#[cfg(feature = "cuda")]
mod config;
#[cfg(feature = "cuda")]
mod llm;
#[cfg(feature = "cuda")]
mod model;
#[cfg(feature = "cuda")]
mod model_runner;
#[cfg(feature = "cuda")]
mod weights;

#[cfg(feature = "cuda")]
pub use config::VOCAB;
#[cfg(feature = "cuda")]
pub use llm::Qwen38;
