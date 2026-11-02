//! PI0.5 semantic backbones and layers, specialized at model construction.
//! Blocks own precision/layout/fusion details and fixed weights. They do not
//! capture graphs, cache requests, or run the complete model schedule.

/// Forward the inherent methods of one precision's Blocks type to the trait.
///
/// Both precisions implement the same five delegations; expanding them here
/// keeps the trait's shape and the two impls from drifting apart without
/// flattening the differences that live below them.
macro_rules! impl_blocks {
    ($blocks:ty, $prefix:ty, $modulation:ty) => {
        impl super::super::Blocks for $blocks {
            type Prefix = $prefix;
            type StepModulation = $modulation;

            fn config(&self) -> &crate::musa_pi05::Pi05Config {
                &self.config
            }

            fn vision(&self, patches: &apxinf_core::Tensor, native: bool) -> apxinf_core::Result<apxinf_core::Tensor> {
                let _ = native;
                self.encode_vision(patches)
            }

            fn embed_prefix(
                &self,
                vision: &apxinf_core::Tensor,
                ids: &crate::musa_pi05::backend::DeviceBuffer,
                count: usize,
            ) -> apxinf_core::Result<apxinf_core::Tensor> {
                self.embed_prefix(vision, ids, count)
            }

            fn prefix(&self, input: &apxinf_core::Tensor) -> apxinf_core::Result<Self::Prefix> {
                self.prefix_forward(input)
            }

            fn compute_modulation(
                &self,
                embeddings: &[apxinf_core::Tensor],
            ) -> apxinf_core::Result<Vec<Self::StepModulation>> {
                self.compute_all_modulation(embeddings)
            }

            fn step_with_modulation(
                &self,
                state: &apxinf_core::Tensor,
                modulation: &Self::StepModulation,
                prefix: &Self::Prefix,
                dt: f32,
            ) -> apxinf_core::Result<apxinf_core::Tensor> {
                self.denoise_step_with_modulation(state, modulation, prefix, dt)
            }
        }
    };
}

pub(super) mod bf16;
pub(super) mod int8_dynamic;

pub(super) use bf16::backbone::Bf16Blocks;
pub(super) use int8_dynamic::backbone::Int8DynamicBlocks;
pub use bf16::backbone::Bf16PrefixKvCache;
pub use int8_dynamic::backbone::Int8DynamicPrefixKvCache;

use crate::musa_pi05::backend::{
    self,
    ops::{
        adaptive_rms_norm, bias_residual, bias_residual_layer_norm, bias_residual_rms_norm,
        layer_norm, AdaptiveRmsNormArgs, BiasResidualArgs, BiasResidualLayerNormArgs,
        BiasResidualRmsNormArgs, LayerNormArgs,
    },
    Context, DeviceBuffer,
};
use crate::musa_pi05::Pi05Config;
use apxinf_core::{DType, Error, Result, Shape, Tensor};

#[derive(Clone, Default)]
pub(in crate::musa_pi05) struct L3Policies {
    pub gemm: backend::ops::GemmPolicy,
    pub attention: backend::ops::AttentionPolicy,
}

impl L3Policies {
    pub(in crate::musa_pi05) fn for_recipe(cache_dir: Option<String>, online_tune: bool) -> Self {
        Self {
            gemm: backend::ops::GemmPolicy {
                cache_dir: cache_dir.clone(),
                online_tune,
                ..backend::ops::GemmPolicy::default()
            },
            attention: backend::ops::AttentionPolicy {
                cache_dir,
                online_tune,
                ..backend::ops::AttentionPolicy::default()
            },
        }
    }

    #[cfg(test)]
    pub(in crate::musa_pi05) fn snapshot(&self) -> L3PolicySnapshot {
        L3PolicySnapshot {
            gemm_cache_dir: self.gemm.cache_dir.clone(),
            gemm_online_tune: self.gemm.online_tune,
            attention_cache_dir: self.attention.cache_dir.clone(),
            attention_online_tune: self.attention.online_tune,
        }
    }
}

#[cfg(test)]
#[derive(Debug, PartialEq, Eq)]
pub(in crate::musa_pi05) struct L3PolicySnapshot {
    pub gemm_cache_dir: Option<String>,
    pub gemm_online_tune: bool,
    pub attention_cache_dir: Option<String>,
    pub attention_online_tune: bool,
}

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
    /// Project the flow schedule into every step's modulation.
    ///
    /// The inputs are constants of the model — the schedule comes from config,
    /// the projection weights from the checkpoint — so what this returns is a
    /// precomputed constant, not a request cache, and it cannot go stale.
    /// `Pi05Model::new` calls it exactly once, before any traversal, and that
    /// placement is what makes the result usable inside a captured graph:
    /// built inside a traversal it would land in that traversal's arena, the
    /// arena is reset at the head of the next one, and a graph that captured
    /// those addresses would replay against someone else's outputs. The model
    /// holding the result is what keeps those addresses valid for the life of
    /// the graph.
    fn compute_modulation(&self, embeddings: &[Tensor]) -> Result<Vec<Self::StepModulation>>;
    fn step_with_modulation(
        &self,
        state: &Tensor,
        modulation: &Self::StepModulation,
        prefix: &Self::Prefix,
        dt: f32,
    ) -> Result<Tensor>;
}

// ── The arithmetic both precisions share ─────────────────────────────────
//
// LayerNorm, adaptive RMSNorm and the three bias+residual forms are identical
// in BF16 and INT8: the precision lives in the linear layers, not in these
// normalizations. They live here so the two block files carry only what
// actually differs.

pub(super) struct QkvTensors {
    pub(super) q: Tensor,
    pub(super) k: Tensor,
    pub(super) v: Tensor,
}

pub(super) trait QkvViews {
    fn key_2d(&self, tokens: usize, head_dim: usize) -> Result<Tensor>;
    fn value_2d(&self, tokens: usize, head_dim: usize) -> Result<Tensor>;
}

impl QkvViews for QkvTensors {
    fn key_2d(&self, tokens: usize, head_dim: usize) -> Result<Tensor> {
        self.k.reshape(vec![tokens, head_dim])
    }

    fn value_2d(&self, tokens: usize, head_dim: usize) -> Result<Tensor> {
        self.v.reshape(vec![tokens, head_dim])
    }
}

pub(super) struct ResidualNormTensors {
    pub(super) hidden: Tensor,
    pub(super) normalized: Tensor,
}

/// The first `elements` BF16 values of `tensor`, as a 1-D view.
pub(super) fn bf16_vector_prefix(tensor: &Tensor, elements: usize) -> Result<Tensor> {
    let bytes = elements
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("PI0.5 BF16 vector size overflow".into()))?;
    DeviceBuffer::from_tensor(tensor)
        .and_then(|buffer| buffer.view(0, bytes))
        .and_then(|buffer| buffer.as_tensor(Shape::new(vec![elements]), DType::BF16))
        .map_err(Error::Other)
}

pub(super) fn layer_bf16(
    ctx: &Context,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let mut output = ctx.allocate_output(input.shape().clone(), DType::BF16)?;
    layer_norm(
        ctx,
        LayerNormArgs::new(input, weight, bias, &mut output, eps),
    )?;
    Ok(output)
}

pub(super) fn adaptive_rms_bf16(
    ctx: &Context,
    input: &Tensor,
    style: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let cols = input
        .shape()
        .dims()
        .last()
        .copied()
        .ok_or_else(|| Error::Other("PI0.5 adaptive RMS input has no columns".into()))?;
    let style = bf16_vector_prefix(style, 2 * cols)?;
    let mut output = ctx.allocate_output(input.shape().clone(), DType::BF16)?;
    adaptive_rms_norm(
        ctx,
        AdaptiveRmsNormArgs::new(input, &style, &mut output, eps),
    )?;
    Ok(output)
}

pub(super) fn bias_residual_bf16(
    ctx: &Context,
    projection: &Tensor,
    bias: Option<&Tensor>,
    residual: &Tensor,
) -> Result<Tensor> {
    let mut hidden = ctx.allocate_output(projection.shape().clone(), DType::BF16)?;
    bias_residual(
        ctx,
        BiasResidualArgs::new(projection, bias, residual, &mut hidden),
    )?;
    Ok(hidden)
}

pub(super) fn bias_residual_rms_bf16(
    ctx: &Context,
    projection: &Tensor,
    bias: Option<&Tensor>,
    residual: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<ResidualNormTensors> {
    let mut hidden = ctx.allocate_output(projection.shape().clone(), DType::BF16)?;
    let mut normalized = ctx.allocate_output(projection.shape().clone(), DType::BF16)?;
    bias_residual_rms_norm(
        ctx,
        BiasResidualRmsNormArgs::new(
            projection,
            bias,
            residual,
            weight,
            &mut hidden,
            &mut normalized,
            eps,
        ),
    )?;
    Ok(ResidualNormTensors { hidden, normalized })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn bias_residual_layer_bf16(
    ctx: &Context,
    projection: &Tensor,
    projection_bias: Option<&Tensor>,
    residual: &Tensor,
    norm_weight: &Tensor,
    norm_bias: &Tensor,
    eps: f32,
) -> Result<ResidualNormTensors> {
    let mut hidden = ctx.allocate_output(projection.shape().clone(), DType::BF16)?;
    let mut normalized = ctx.allocate_output(projection.shape().clone(), DType::BF16)?;
    bias_residual_layer_norm(
        ctx,
        BiasResidualLayerNormArgs::new(
            projection,
            projection_bias,
            residual,
            norm_weight,
            norm_bias,
            &mut hidden,
            &mut normalized,
            eps,
        ),
    )?;
    Ok(ResidualNormTensors { hidden, normalized })
}
