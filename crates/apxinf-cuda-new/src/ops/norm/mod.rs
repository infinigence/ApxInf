pub(crate) mod contracts;
mod head;
pub(crate) mod launch;
mod norm;

pub use head::head_rms_norm;

#[cfg(test)]
pub(crate) use norm::{fused_add_norm_spec, norm_spec, residual_add_spec};
pub use norm::{
    fused_add_norm, norm, residual_add, Affine, BiasOrder, FusedAddNormArgs, NormArgs, NormKind,
    QuantOut, ResidualAddArgs,
};
