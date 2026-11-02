//! Semantic BF16 operators, shaped after `apxinf-cuda-new`'s `ops` surface.
//!
//! PI0.5's device tree is written against these names, so the contract is
//! upstream's rather than this crate's: the same Args fields, the same
//! in-place output convention, the same semantics. What differs is the
//! implementation — each operator unpacks its Args and calls the native kernel
//! facade below, and an operator or variant this port cannot serve reports
//! itself by name instead of approximating.
//!
//! Only the names PI0.5's BF16 path imports are mirrored flat. The W8A8 route
//! has its own module ([`w8a8`]) rather than a [`GemmQuantization`] arm: its
//! scale vectors are FP16 with a tile-ABI pad, which the f32 contract cannot
//! carry without a conversion on every GEMM. FP8/NVFP4, GDN and the
//! packed-quantization family are not part of this surface.

mod attention;
mod cache;
mod gather;
mod gemm;
mod norm;
mod pointwise;
mod rope;
pub mod w8a8;

pub use attention::{attention, AttentionArgs, AttentionMask, AttentionPolicy};
pub use cache::{concat_rows, reserve_prefix};
pub use gather::{gather, GatherArgs, GatherSemantic, PatchGeometry as GatherPatchGeometry};
pub use gemm::{gemm, gemm_geglu, GemmArgs, GemmGegluArgs, GemmPolicy, GemmQuantization, WeightVersion};
pub use norm::{
    adaptive_rms_norm, ada_gate_residual_rms_norm, bias_residual, bias_residual_layer_norm,
    bias_residual_rms_norm, layer_norm, rms_norm, AdaGateResidualRmsNormArgs,
    AdaptiveRmsNormArgs, BiasResidualArgs, BiasResidualLayerNormArgs, BiasResidualRmsNormArgs,
    LayerNormArgs, RmsNormArgs,
};
pub use pointwise::{pointwise, PointwiseActivation, PointwiseArgs, PointwiseSemantic};
pub use rope::{rope, RopeArgs, RopeSemantic};

pub use crate::workspace::{ExecutionSession, GraphWorkspace};

use apxinf_core::{Error, Result};

/// Run a fixed-shape forward pass that may prepare native executions.
pub fn prepare_with_session<T>(
    session: &ExecutionSession,
    operation: impl FnOnce() -> Result<T>,
) -> Result<T> {
    crate::workspace::prepare_with_session(session, operation)
}

/// Run the same prepared forward pass without preparing anything new.
pub fn with_session<T>(
    session: &ExecutionSession,
    operation: impl FnOnce() -> Result<T>,
) -> Result<T> {
    crate::workspace::with_session(session, operation)
}

/// Reject an output scale this backend cannot honour.
///
/// Every operator here writes BF16. A scale other than 1.0 only makes sense for
/// a quantized output, and silently ignoring it would store unscaled values
/// that every consumer then reads as if they had been scaled.
fn require_unit_output_scale(operation: &str, output_scale: f32) -> Result<()> {
    if output_scale != 1.0 {
        return Err(Error::Other(format!(
            "MUSA {operation} writes BF16 only; output_scale {output_scale} needs a quantized output"
        )));
    }
    Ok(())
}
