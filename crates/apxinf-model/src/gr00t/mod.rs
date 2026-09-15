//! NVIDIA GR00T model contracts.
//!
//! The N1.7 checkpoint uses a Qwen3-VL backbone and a flow-matching action
//! head.  Keep model-specific configuration and math here; CUDA operators
//! remain model-neutral under `apxinf-cuda`.

#[cfg(any(feature = "cuda", test))]
mod action_weights;
#[cfg(any(feature = "cuda", test))]
mod backbone;
#[cfg(feature = "cuda")]
mod backend;
#[cfg(feature = "cuda")]
mod bf16_executor;
#[cfg(feature = "cuda")]
mod bf16_runtime;
#[cfg(feature = "cuda")]
mod bf16_weights;
#[cfg(feature = "cuda")]
mod calibration;
#[cfg(any(feature = "cuda", test))]
mod config;
#[cfg(feature = "cuda")]
mod device_weights;
#[cfg(feature = "cuda")]
mod executor;
#[cfg(feature = "cuda")]
mod fp8_executor;
#[cfg(feature = "cuda")]
mod fp8_runtime;
#[cfg(feature = "cuda")]
mod fp8_weights;
#[cfg(any(feature = "cuda", test))]
mod geometry;
#[cfg(feature = "cuda")]
mod int8_executor;
#[cfg(feature = "cuda")]
mod int8_runtime;
#[cfg(feature = "cuda")]
mod int8_weights;
#[cfg(any(feature = "cuda", test))]
mod math;
#[cfg(feature = "cuda")]
mod vla_runtime;
#[cfg(any(feature = "cuda", test))]
mod weights;

#[cfg(any(feature = "cuda", test))]
pub(crate) use config::Gr00tConfig;

#[cfg(feature = "cuda")]
pub(crate) fn register_builtin() {
    crate::registry::register("gr00t-cuda", vla_runtime::load_registered);
    crate::registry::register("gr00t_n1_7-cuda", vla_runtime::load_registered);
    crate::registry::register("Gr00tN1d7-cuda", vla_runtime::load_registered);
}
