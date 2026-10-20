pub(crate) mod contracts;
mod decode;
pub(crate) mod launch;
mod multimodal;
mod rope;

pub use contracts::{RopeArgs, RopeSemantic};
pub use decode::{decode_rope, DecodeRopeArgs};
pub use multimodal::{multimodal_rope, split_qkv_axial_rope, AxialRopeArgs, MultimodalRopeArgs};
pub use rope::rope;

mod pairwise;
pub use pairwise::{pairwise_rope, PairwiseRopeArgs, PairwiseRopeTable};

#[cfg(test)]
pub const PAIRWISE_DOC_IDS: &[&str] = &["pairwise-rope"];
