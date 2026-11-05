pub(crate) mod contracts;
mod embedding;
mod gather;
pub(crate) mod launch;

pub use contracts::{EmbeddingScale, GatherArgs, GatherSemantic, PatchGeometry};
pub use embedding::embedding_gather;
pub use gather::gather;
