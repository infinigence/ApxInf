//! Tiled, causal volume lowering. The model chooses convolution order; this
//! operator only maps THWC neighborhoods to rows for an ordinary GEMM.
mod contracts;
mod conv3d;
pub use conv3d::{Conv3dArgs, Conv3dPolicy, PreparedConv3d};
mod execution;

pub use contracts::{Unfold3dArgs, Unfold3dGeometry};
pub use execution::PreparedUnfold3d;

#[cfg(test)]
pub(crate) const DOC_IDS: &[&str] = &["Unfold3d", "Conv3d"];
