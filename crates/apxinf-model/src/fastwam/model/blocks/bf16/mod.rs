//! BF16 block composition, including the declared F32 residual/projection paths.
mod dit;
mod ops;
mod prepare;
mod text;
mod vae;
pub(in crate::fastwam) use dit::sigma;
pub(in crate::fastwam) use dit::Expert;
