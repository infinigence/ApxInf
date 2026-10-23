pub(crate) mod contracts;
mod fixed;
mod gemm;
mod gemm_bias;
mod gemm_bias_residual;
pub(crate) mod gemm_execution;
mod gemm_geglu;
mod gemm_gelu;
mod gemm_swiglu;
mod nvfp4_scales;

pub use contracts::{GemmArgs, GemmPolicy, GemmQuantization, WeightVersion};
pub use fixed::*;
pub use gemm::gemm;
pub use gemm_bias::{gemm_bias, GemmBiasArgs};
pub use gemm_bias_residual::{gemm_bias_residual, GemmBiasResidualArgs};
pub use gemm_geglu::{gemm_geglu, GemmGegluArgs};
pub use gemm_gelu::{gemm_bias_gelu, GemmBiasGeluArgs};
pub use gemm_swiglu::{
    gemm_swiglu_pair_interleaved_256, pack_bf16_swiglu_weight_host, GemmSwiGluArgs,
};
pub use nvfp4_scales::{
    nvfp4_gemm_swiglu, nvfp4_pack_block_scales, nvfp4_quantize_activation, nvfp4_quantize_rms_norm,
    nvfp4_quantize_swiglu, nvfp4_scale_buffer_bytes, ScaleLayout,
};
