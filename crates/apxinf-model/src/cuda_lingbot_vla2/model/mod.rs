//! Request-level composition: vision -> prefix embedding -> language K/V -> ten flow steps.
//!
//! The model owns forward order and the flow schedule. Execution state (plans, graphs, inputs) belongs to
//! `model_runner`.

pub(super) mod expert;
pub(super) mod language;
pub(super) mod util;
pub(super) mod vision;

use crate::cuda_lingbot_vla2::backend::{ops, Context};
use crate::cuda_lingbot_vla2::config::*;
use crate::cuda_lingbot_vla2::weights::model_weights::LingbotWeights;
use apxinf_core::{DType, Result, Tensor};
use expert::predict_velocity;
use language::KvCaches;
use util::*;

/// Row offsets of the pieces of the prefix for one prompt length.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PrefixLayout {
    pub prompt_tokens: usize,
    pub total: usize,
}

impl PrefixLayout {
    pub fn new(prompt_tokens: usize) -> Self {
        Self {
            prompt_tokens,
            total: prefix_len(prompt_tokens),
        }
    }
    /// First row of view `v`'s block `[vision_start, 64 image tokens, vision_end]`.
    pub fn view_start(&self, view: usize) -> usize {
        view * IMAGE_TOKENS_PER_VIEW
    }
    /// First image-token row of view `v` (after its `vision_start`).
    pub fn view_tokens(&self, view: usize) -> usize {
        self.view_start(view) + 1
    }
    pub fn prompt_start(&self) -> usize {
        NUM_VIEWS * IMAGE_TOKENS_PER_VIEW
    }
    pub fn query_start(&self) -> usize {
        self.prompt_start() + self.prompt_tokens
    }
    pub fn visual_rows(&self) -> [usize; NUM_VIEWS] {
        [
            self.view_tokens(0),
            self.view_tokens(1),
            self.view_tokens(2),
        ]
    }
}

/// Inputs bound to a prepared request. Everything here is a fixed-address device tensor.
pub(crate) struct Inputs<'a> {
    /// BF16 `[768, 1536]` patch rows of the three views.
    pub patches: &'a Tensor,
    /// BF16 `[prompt_tokens + 2, 2560]`: embeddings of the prompt ids followed by `vision_start`, `vision_end`.
    pub token_embeds: &'a Tensor,
    /// BF16 `[1, 64]` zero-padded state.
    pub state: &'a Tensor,
    /// BF16 `[50, 64]` zero-padded initial noise.
    pub noise: &'a Tensor,
}

/// Fixed constants of a prepared plan: prefix rotary tables and suffix positions.
pub(crate) struct PlanTables {
    /// U32 `[prefix, 3]` mRoPE positions of the prefix tokens (valid tokens only).
    pub prefix_positions: crate::cuda_lingbot_vla2::backend::DeviceBuffer,
    /// I32 `[51, 3]` suffix mRoPE positions (all three axes equal).
    pub suffix_positions: Tensor,
}

/// Build the prefix `[total, 2560]`: per view `[vision_start | 64 image tokens | vision_end]`, then the prompt,
/// then the 16 learned query rows.
fn build_prefix(
    ctx: &Context,
    w: &LingbotWeights,
    layout: &PrefixLayout,
    image_tokens: &Tensor,
    inputs: &Inputs<'_>,
) -> Result<Tensor> {
    let prefix = alloc(ctx, &[layout.total, VLM_HIDDEN], DType::BF16)?;
    let n = layout.prompt_tokens;
    let start_row = rows(inputs.token_embeds, n, 1, VLM_HIDDEN)?;
    let end_row = rows(inputs.token_embeds, n + 1, 1, VLM_HIDDEN)?;
    for view in 0..NUM_VIEWS {
        let at = layout.view_start(view);
        ops::copy_tensor(ctx, &start_row, &rows(&prefix, at, 1, VLM_HIDDEN)?)?;
        ops::copy_tensor(
            ctx,
            &rows(
                image_tokens,
                view * MERGED_PER_VIEW,
                MERGED_PER_VIEW,
                VLM_HIDDEN,
            )?,
            &rows(&prefix, at + 1, MERGED_PER_VIEW, VLM_HIDDEN)?,
        )?;
        ops::copy_tensor(
            ctx,
            &end_row,
            &rows(&prefix, at + 1 + MERGED_PER_VIEW, 1, VLM_HIDDEN)?,
        )?;
    }
    ops::copy_tensor(
        ctx,
        &rows(inputs.token_embeds, 0, n, VLM_HIDDEN)?,
        &rows(&prefix, layout.prompt_start(), n, VLM_HIDDEN)?,
    )?;
    ops::copy_tensor(
        ctx,
        &w.queries.rows,
        &rows(
            &prefix,
            layout.query_start(),
            PREFIX_QUERY_TOKENS,
            VLM_HIDDEN,
        )?,
    )?;
    Ok(prefix)
}

/// Full request: returns the normalised action chunk `[50, 64]` BF16 (columns >= 55 are zero).
pub(crate) fn infer(
    ctx: &Context,
    w: &LingbotWeights,
    layout: &PrefixLayout,
    tables: &PlanTables,
    inputs: &Inputs<'_>,
    caches: &KvCaches,
    gemm: &ops::GemmPolicy,
    attention: &ops::AttentionPolicy,
) -> Result<Tensor> {
    let vis = vision::forward(ctx, &w.vision, inputs.patches, gemm, attention)?;
    let prefix = build_prefix(ctx, w, layout, &vis.tokens, inputs)?;
    language::forward(
        ctx,
        &w.vlm,
        &prefix,
        &tables.prefix_positions,
        &layout.visual_rows(),
        &vis.deepstack,
        caches,
        gemm,
        attention,
    )?;
    // `state_proj` has M = 1 and K = 64, a shape whose cuBLASLt custom-epilogue candidate fails on sm_89
    // ("cuBLAS status 14"). The state token is computed as one row of a block with the 50 action rows
    // instead; the extra rows are zeros, so the result is identical.
    let state_row = {
        let padded = alloc(ctx, &[ACTION_HORIZON + 1, PADDED_DIM], DType::BF16)?;
        ops::copy_tensor(ctx, inputs.state, &rows(&padded, 0, 1, PADDED_DIM)?)?;
        let projected = dense(ctx, &padded, &w.expert.state_proj, gemm)?;
        rows(&projected, 0, 1, EXPERT_HIDDEN)?
    };
    let mut x = inputs.noise.clone();
    let mut next = alloc(ctx, x.shape().dims(), DType::BF16)?;
    for step in 0..w.expert.times.len() {
        let v = predict_velocity(
            ctx,
            &w.expert,
            step,
            &state_row,
            &x,
            &tables.suffix_positions,
            caches,
            gemm,
            attention,
        )?;
        // x_{t+dt} = x_t + dt * v, dt < 0
        let mut args = ops::PointwiseArgs::new(ops::PointwiseSemantic::EulerUpdate, &x, &mut next);
        args.secondary = Some(&v);
        args.dt = w.expert.dt;
        ops::pointwise(ctx, args)?;
        std::mem::swap(&mut x, &mut next);
        if step == 0 {
            // `x` aliased the caller's noise buffer on the first step; never write into it afterwards.
            next = alloc(ctx, x.shape().dims(), DType::BF16)?;
        }
    }
    Ok(x)
}
