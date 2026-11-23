//! Ascend L3 operator layer (absorbed from the former separate L3 crate).
//!
//! This is the L3 runtime used by the non-`dyn Backend` model path
//! (`ascend_pi05`). It mirrors `apxinf_cuda_new`'s public API: typed L3
//! operator contracts (`ops::*`), RI-graph capture (`graph`),
//! execution-session workspace pooling (`workspace`), a session-aware
//! device buffer (`buffer`), an L3 context (`context`), and L3
//! host<->device transfer / sampling helpers (`transfers`, `sampling`).
//! The base layer (ACL FFI, `AscendBackend`, `AscendContext`, low-level
//! buffer/transfers) lives at the crate root.

pub mod buffer;
pub mod context;
pub mod graph;
pub mod ops;
pub mod sampling;
pub mod transfers;
pub mod workspace;

pub use buffer::AscendBuffer;
pub use context::AscendNewContext;
pub use graph::{capture, CapturedGraph};
pub use workspace::ExecutionSession;

// Re-export diagnostics from the base layer so the former L3 crate's `*`
// callers map 1:1 to `crate::l3::*`.
pub use crate::acl_tensor::{reset_diagnostics, print_diagnostics};
pub use crate::transfers::{reset_h2d_diagnostics, print_h2d_diagnostics};

// Re-export types used by PI0.5's backend.rs seam.
pub use ops::{
    AttentionArgs, AttentionMask, AttentionPolicy,
    GemmArgs, GemmGegluArgs, GemmPolicy, GemmQuantization, WeightVersion,
    GatherArgs, GatherPatchGeometry, GatherSemantic,
    LayerNormArgs, RmsNormArgs, AdaptiveRmsNormArgs,
    BiasResidualArgs, BiasResidualRmsNormArgs, BiasResidualLayerNormArgs,
    AdaGateResidualArgs, AdaGateResidualRmsNormArgs, BiasThenResidualArgs,
    PointwiseArgs, PointwiseActivation, PointwiseSemantic,
    RopeArgs, RopeSemantic,
};
