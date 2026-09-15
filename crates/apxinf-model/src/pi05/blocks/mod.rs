//! PI0.5 semantic backbones and layers, specialized at model construction.
//! Blocks own precision/layout/fusion details and fixed weights. They do not
//! capture graphs, cache requests, or run the complete model schedule.

pub(super) mod bf16;
pub(super) mod fp8;
pub(super) mod w8a8;

pub use bf16::backbone::Bf16PrefixKvCache;
pub(super) use bf16::backbone::{Bf16Blocks, Bf16StepStyles};
pub use fp8::backbone::PrefixKvCache;
pub(super) use fp8::backbone::{Fp8Blocks, Pi05StepStyles};
pub use w8a8::backbone::Int8PrefixKvCache;
pub(super) use w8a8::backbone::{Int8StepStyles, W8A8Blocks};

use super::{backend::DeviceBuffer, Pi05Config};
use apxinf_core::{Result, Tensor};

/// Internal, statically dispatched seam. The associated state types retain
/// each Block's physical representation without exposing dtype tests to Network.
pub(super) trait Blocks {
    type Prefix;
    type Styles;
    fn config(&self) -> &Pi05Config;
    /// `native` denotes the Block's already materialized input representation.
    fn vision(&self, patches: &Tensor, native: bool) -> Result<Tensor>;
    fn embed_prefix(&self, vision: &Tensor, ids: &DeviceBuffer, count: usize) -> Result<Tensor>;
    fn prefix(&self, input: &Tensor) -> Result<Self::Prefix>;
    fn prepare_styles(&self, embeddings: &[Tensor]) -> Result<Vec<Self::Styles>>;
    /// Preserve execution ordering: BF16/W8A8 precompute eager styles; FP8
    /// computes them per step after prefix processing. Capture precomputes all.
    fn eager_styles(&self, embeddings: &[Tensor]) -> Result<Option<Vec<Self::Styles>>>;
    fn step(
        &self,
        state: &Tensor,
        embedding: &Tensor,
        prefix: &Self::Prefix,
        dt: f32,
    ) -> Result<Tensor>;
    fn step_with_styles(
        &self,
        state: &Tensor,
        styles: &Self::Styles,
        prefix: &Self::Prefix,
        dt: f32,
    ) -> Result<Tensor>;
}
