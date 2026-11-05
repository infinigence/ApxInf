mod candidates;
pub(crate) mod contracts;
pub(crate) mod launch;
mod norm;

pub use contracts::{BiasOrder, NormNumerics};
pub use norm::{
    fused_add_norm, norm, residual_add, Affine, FusedAddNormArgs, NormArgs, NormKind, QuantOut,
    ResidualAddArgs, ResidualGate,
};
