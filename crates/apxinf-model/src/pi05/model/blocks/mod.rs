//! PI0.5 semantic backbones and layers, specialized at model construction.
//! Blocks own precision/layout/fusion details and fixed weights. They do not
//! capture graphs, cache requests, or run the complete model schedule.

pub(super) mod bf16;
pub(super) mod fp8_static;
pub(super) mod int8_dynamic;

pub(super) use bf16::backbone::Bf16Blocks;
pub use bf16::backbone::Bf16PrefixKvCache;
pub(super) use fp8_static::backbone::Fp8StaticBlocks;
pub use fp8_static::backbone::Fp8StaticPrefixKvCache;
pub(super) use int8_dynamic::backbone::Int8DynamicBlocks;
pub use int8_dynamic::backbone::Int8DynamicPrefixKvCache;

use crate::pi05::{backend::DeviceBuffer, Pi05Config};
use apxinf_core::{Result, Tensor};

/// Internal, statically dispatched seam. The associated state types retain
/// each Block's physical representation without exposing dtype tests to Model.
pub trait Blocks {
    type Prefix;
    /// Per-timestep, per-layer adaptive RMSNorm and residual modulation tensors,
    /// derived from the time conditioning rather than the request observation.
    type StepModulation;
    fn config(&self) -> &Pi05Config;
    /// `native` denotes the Block's already materialized input representation.
    fn vision(&self, patches: &Tensor, native: bool) -> Result<Tensor>;
    fn embed_prefix(&self, vision: &Tensor, ids: &DeviceBuffer, count: usize) -> Result<Tensor>;
    fn prefix(&self, input: &Tensor) -> Result<Self::Prefix>;
    fn prepare_modulation(&self, embeddings: &[Tensor]) -> Result<Vec<Self::StepModulation>>;
    /// Preserve execution ordering: BF16/dynamic INT8 precompute eager modulation; FP8
    /// computes them per step after prefix processing. Capture precomputes all.
    fn eager_modulation(&self, embeddings: &[Tensor]) -> Result<Option<Vec<Self::StepModulation>>>;
    fn step(
        &self,
        state: &Tensor,
        embedding: &Tensor,
        prefix: &Self::Prefix,
        dt: f32,
    ) -> Result<Tensor>;
    fn step_with_modulation(
        &self,
        state: &Tensor,
        modulation: &Self::StepModulation,
        prefix: &Self::Prefix,
        dt: f32,
    ) -> Result<Tensor>;
}
