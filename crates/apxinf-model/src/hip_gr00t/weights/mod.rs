//! Checkpoint structure (`host`) and the device-resident weight tree (`bf16`).

mod bf16;
pub mod host;
pub use bf16::{Embodiment, EmbodimentSource, HipGr00tWeights};
pub use bf16::{DitBlock, Linear, Merger, Norm, TextLayer, VisionBlock, VlBlock};
pub use host::Gr00tWeights;
