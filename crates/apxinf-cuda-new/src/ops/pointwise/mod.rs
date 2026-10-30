pub(crate) mod contracts;
mod elementwise;
mod exact;
mod fixed;
pub(crate) mod launch;
mod pointwise;

pub use contracts::{PointwiseActivation, PointwiseArgs, PointwiseSemantic};
pub use elementwise::swiglu_into;
pub use exact::gelu_exact;
pub use fixed::*;
pub use pointwise::pointwise;

/// Catalog identifiers of the binary arithmetic public semantics.
#[cfg(test)]
pub const BINARY_DOC_IDS: &[&str] = &[
    "pointwise-multiply",
    "pointwise-broadcast-multiply",
    "pointwise-row-multiply",
];
