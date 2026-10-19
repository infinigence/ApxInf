pub(crate) mod contracts;
pub(crate) mod launch;
mod pointwise;

pub use contracts::{PointwiseActivation, PointwiseArgs, PointwiseSemantic};
pub use pointwise::pointwise;

/// Catalog identifiers of the rounded arithmetic public semantics.
#[cfg(test)]
pub const ROUNDED_DOC_IDS: &[&str] = &[
    "pointwise-multiply",
    "pointwise-add",
    "pointwise-broadcast-multiply",
    "pointwise-broadcast-add",
    "pointwise-row-multiply",
    "pointwise-euler-rounded",
    "pointwise-gelu-rounded",
];
