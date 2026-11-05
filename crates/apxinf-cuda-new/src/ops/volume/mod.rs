//! Tiled, causal volume lowering. The model chooses convolution order; this
//! operator only maps THWC neighborhoods to rows for an ordinary GEMM.
mod contracts;
mod conv3d;
// The bare lowering wrapper is a contract-test surface; production reaches the
// C ABI through Conv3d only.
#[cfg(test)]
mod execution;

pub use conv3d::{Conv3dArgs, Conv3dPolicy, PreparedConv3d};
pub use contracts::Unfold3dGeometry;
// `Conv3d` owns the only production path to this lowering. The bare unfold
// wrapper exists for contract tests only, so it is not part of the public or
// crate-wide surface.
#[cfg(test)]
pub(crate) use contracts::Unfold3dArgs;
#[cfg(test)]
pub(crate) use execution::PreparedUnfold3d;

#[cfg(test)]
pub(crate) const DOC_IDS: &[&str] = &["Unfold3d", "Conv3d"];
