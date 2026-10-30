pub(crate) mod contracts;
mod fixed;
mod gemm;
mod gemm_bias;
mod gemm_bias_residual;
pub(crate) mod gemm_execution;
mod gemm_geglu;
mod gemm_gelu;
mod gemm_swiglu;
mod gemv;
pub(crate) mod nvfp4_scales;
mod observer;

pub use contracts::{
    GemmArgs, GemmNumerics, GemmPolicy, GemmQuantization, GemmWeightLayout, WeightVersion,
};
pub use fixed::*;
pub use gemm::gemm;
pub use gemm_bias::{gemm_bias, GemmBiasArgs};
pub use gemm_bias_residual::{gemm_bias_residual, GemmBiasResidualArgs};
pub use gemm_geglu::{gemm_geglu, GemmGegluArgs};
pub use gemm_gelu::{gemm_bias_gelu, GemmBiasGeluArgs};
pub use gemm_swiglu::{
    gemm_swiglu, pack_swiglu_weight_host, BlockQuantizedSwiGluArgs, GemmSwiGluArgs,
};
pub use gemv::{gemv, gemv_bias, GemvArgs, GemvBiasArgs, GemvBlockScales};
pub use observer::{install_gemm_observer, GemmActivationObserver, GemmObserverGuard};
