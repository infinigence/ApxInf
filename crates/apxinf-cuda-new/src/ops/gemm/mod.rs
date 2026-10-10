pub(crate) mod contracts;
mod gemm;
mod gemm_bias;
mod gemm_bias_residual;
pub(crate) mod gemm_execution;
mod gemm_geglu;
mod gemm_gelu;
mod nvfp4_scales;

pub use contracts::{GemmArgs, GemmPolicy, GemmQuantization, WeightVersion};
pub use gemm::gemm;
pub use gemm_bias::{gemm_bias, GemmBiasArgs};
pub use gemm_bias_residual::{gemm_bias_residual, GemmBiasResidualArgs};
pub use gemm_geglu::{gemm_geglu, GemmGegluArgs};
pub use gemm_gelu::{gemm_bias_gelu, GemmBiasGeluArgs};
pub use nvfp4_scales::{
    nvfp4_pack_block_scales, nvfp4_quantize_activation, nvfp4_quantize_rms_norm,
    nvfp4_quantize_swiglu, nvfp4_scale_buffer_bytes, ScaleLayout,
};
