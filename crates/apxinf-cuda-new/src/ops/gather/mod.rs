pub(crate) mod contracts;
mod embedding;
mod gather;
pub(crate) mod launch;

pub use contracts::{EmbeddingScale, GatherArgs, GatherSemantic, PatchGeometry};
pub use embedding::embedding_gather;
pub use gather::gather;

pub(crate) mod layout;
pub use layout::{permute, resize_bicubic, unfold3d, unfold3d_blocked, Unfold3dGeometry};
