//! PI0.5 Ascend semantic backbones and layers (BF16 only).

pub(super) mod bf16;

pub(super) use bf16::backbone::Bf16Blocks;
pub use bf16::backbone::Bf16PrefixKvCache;

use crate::ascend_pi05::{backend, backend::DeviceBuffer, Pi05Config};
use apxinf_core::{Result, Tensor};

#[derive(Clone, Default)]
pub(in crate::ascend_pi05) struct L3Policies {
    pub gemm: backend::ops::GemmPolicy,
    pub attention: backend::ops::AttentionPolicy,
}

impl L3Policies {
    pub(in crate::ascend_pi05) fn for_recipe(
        _cache_dir: Option<String>,
        _online_tune: bool,
    ) -> Self {
        // Ascend has no autotuning — always use defaults
        Self::default()
    }
}

/// Internal, statically dispatched seam.
pub trait Blocks {
    type Prefix;
    type StepModulation;
    fn config(&self) -> &Pi05Config;
    fn vision(&self, patches: &Tensor, native: bool) -> Result<Tensor>;
    fn embed_prefix(&self, vision: &Tensor, ids: &DeviceBuffer, count: usize) -> Result<Tensor>;
    fn prefix(&self, input: &Tensor) -> Result<Self::Prefix>;
    /// Build the prefix KV cache with an optional RoPE position offset override.
    ///
    /// When `position_offset` is `Some(n)`, the action expert's RoPE positions
    /// start at `n` instead of the prefix's total row count. This aligns with
    /// LeRobot's `sum(prefix_pad_masks)` semantics.
    fn prefix_with_offset(
        &self,
        input: &Tensor,
        position_offset: Option<usize>,
    ) -> Result<Self::Prefix>;
    fn prepare_modulation(&self, embeddings: &[Tensor]) -> Result<Vec<Self::StepModulation>>;
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
