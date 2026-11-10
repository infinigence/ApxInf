//! The loaded GR00T N1.7 model on a HIP device.
//!
//! Owns everything that outlives a request: device weights, the static
//! timestep conditioning, the host-side position table the vision tower
//! interpolates per shape, and the per-embodiment adapters.
//!
//! Request-shaped state (arenas, input buffers, prepared positions) belongs to
//! [`super::runner`], not here.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use apxinf_core::{Error, Result, Tensor};
use apxinf_hip::native::HipArena;
use apxinf_hip::HipContext;

use super::action_head::{self, DitBackbone, DitConditioning};
use super::backbone::{self, PreparedBackbone};
use super::checkpoint::action_head::Gr00tActionHeadWeights;
use super::checkpoint::Gr00tWeights;
use super::config::{Gr00tConfig, Qwen3VLConfig};
use super::stages::{NoStages, StageSink};
use super::vision::{self, PreparedVision};
use super::weights::{Embodiment, HipGr00tWeights};

/// Backs the static timestep conditioning for the model's lifetime: 4 steps
/// x (32 block modulations + 1 output modulation + one `[horizon, width]` time
/// embedding). Separate from any request arena, and never reset.
const CONSTANT_ARENA_BYTES: usize = 64 << 20;

pub struct HipGr00tModel {
    ctx: Arc<HipContext>,
    config: Gr00tConfig,
    backbone_config: Qwen3VLConfig,
    weights: HipGr00tWeights,
    conditioning: DitConditioning,
    /// Backs `conditioning`, so it must outlive it and must never be reset.
    constants: HipArena,
    /// `[num_position_embeddings, hidden]` on the host: the vision tower
    /// interpolates it per grid with HF's BF16 rounding points.
    vision_pos_embed: Tensor,
    /// The `[embodiment, in, out]` adapters, on the host. All 32 embodiments
    /// are 621 MiB in BF16, too much to upload blindly, and the request picks
    /// one -- so they stay here and the selected one is uploaded on demand.
    categories: Gr00tActionHeadWeights,
    selected: RefCell<Option<(usize, Rc<Embodiment>)>>,
}

impl HipGr00tModel {
    pub fn new(
        ctx: Arc<HipContext>,
        config: Gr00tConfig,
        backbone_config: Qwen3VLConfig,
        host: Gr00tWeights,
        ) -> Result<Self> {
        let constants = HipArena::new(&ctx, CONSTANT_ARENA_BYTES)?;
        // Embodiment 0 only to satisfy the constructor; `embodiment()` selects
        // the requested one and this copy is dropped immediately.
        let (weights, _) =
            HipGr00tWeights::from_checkpoint(&host, &config, &backbone_config, 0, &ctx)?;
        let conditioning = DitConditioning::new(&ctx, &config, &weights, &constants)?;
        Ok(Self {
            ctx,
            config,
            backbone_config,
            weights,
            conditioning,
            constants,
            vision_pos_embed: host.backbone_vision.pos_embed.clone(),
            categories: host.action_head,
            selected: RefCell::new(None),
        })
    }

    pub fn context(&self) -> &Arc<HipContext> {
        &self.ctx
    }

    pub fn config(&self) -> &Gr00tConfig {
        &self.config
    }

    pub fn backbone_config(&self) -> &Qwen3VLConfig {
        &self.backbone_config
    }

    /// The uploaded weights, for callers that drive one stage at a time --
    /// a benchmark attributing latency, or a stage dump. Inference itself
    /// goes through [`Self::infer`], which owns the stage order.
    pub fn weights(&self) -> &HipGr00tWeights {
        &self.weights
    }

    /// The requested embodiment's adapters, uploading them on first use.
    ///
    /// One deployment uses one embodiment, so this is normally a single
    /// upload. A request that switches embodiment uploads the new set and
    /// drops the old one's allocations, so alternating costs bandwidth but
    /// does not grow memory.
    pub fn embodiment(&self, id: usize) -> Result<Rc<Embodiment>> {
        if let Some((cached, weights)) = self.selected.borrow().as_ref() {
            if *cached == id {
                return Ok(Rc::clone(weights));
            }
        }
        let weights = Rc::new(super::weights::select_embodiment(
            &self.categories,
            id,
            &self.config,
            &self.ctx,
        )?);
        *self.selected.borrow_mut() = Some((id, Rc::clone(&weights)));
        Ok(weights)
    }

    /// Per-shape host work: the interpolated vision positions, the mRoPE
    /// positions and the row indices the vision output scatters into.
    pub fn prepare_shape(
        &self,
        patches: usize,
        grid_thw: &[[u32; 3]],
        token_ids: &[u32],
        attention_mask: &[u8],
    ) -> Result<(PreparedVision, PreparedBackbone)> {
        let vision = vision::prepare(
            &self.ctx,
            &self.backbone_config,
            &self.vision_pos_embed,
            grid_thw,
            patches,
        )?;
        let backbone = backbone::prepare(
            &self.ctx,
            &self.backbone_config,
            token_ids,
            attention_mask,
            grid_thw,
        )?;
        Ok((vision, backbone))
    }

    /// One inference, every activation from `arena`.
    #[allow(clippy::too_many_arguments)]
    pub fn infer(
        &self,
        arena: &HipArena,
        pixel_values: &Tensor,
        state: &Tensor,
        noise: &Tensor,
        vision_plan: &PreparedVision,
        backbone_plan: &PreparedBackbone,
        embodiment: &Embodiment,
        rows: &action_head::DitRowIndices,
    ) -> Result<Tensor> {
        self.infer_observed(
            arena,
            pixel_values,
            state,
            noise,
            vision_plan,
            backbone_plan,
            embodiment,
            rows,
            &mut NoStages,
        )
    }

    /// [`infer`](Self::infer), reporting the reference's stage names.
    #[allow(clippy::too_many_arguments)]
    pub fn infer_observed(
        &self,
        arena: &HipArena,
        pixel_values: &Tensor,
        state: &Tensor,
        noise: &Tensor,
        vision_plan: &PreparedVision,
        backbone_plan: &PreparedBackbone,
        embodiment: &Embodiment,
        rows: &action_head::DitRowIndices,
        stages: &mut dyn StageSink,
    ) -> Result<Tensor> {
        let vision_output = vision::forward_observed(
            &self.ctx,
            &self.backbone_config,
            &self.weights,
            arena,
            pixel_values,
            vision_plan,
            stages,
        )?;
        let hidden = backbone::forward_observed(
            &self.ctx,
            &self.backbone_config,
            &self.weights,
            arena,
            &vision_output,
            backbone_plan,
            stages,
        )?;
        stages.stage("backbone_features_raw", &hidden)?;
        let adapted = backbone::forward_adapter_observed(
            &self.ctx,
            &self.config,
            &self.weights,
            arena,
            &hidden,
            stages,
        )?;
        stages.stage("backbone_features_post_adapter", &adapted)?;

        let split = action_head::split_backbone(
            &self.ctx,
            arena,
            &adapted,
            rows,
        )?;
        self.flow(arena, &split, state, noise, embodiment, stages)
    }

    fn flow(
        &self,
        arena: &HipArena,
        backbone: &DitBackbone,
        state: &Tensor,
        noise: &Tensor,
        embodiment: &Embodiment,
        stages: &mut dyn StageSink,
    ) -> Result<Tensor> {
        action_head::forward_observed(
            &self.ctx,
            &self.config,
            &self.weights,
            embodiment,
            &self.conditioning,
            arena,
            backbone,
            state,
            noise,
            stages,
        )
    }

    /// Activation arena size for one request of this shape.
    ///
    /// An audited bound, not an estimate: every `arena.tensor` in the forward
    /// path is counted here in order, with the arena's 256-byte alignment. An
    /// arena that is one allocation short fails the run outright, so this
    /// walks the same sequence the code does and a reviewer can diff the two.
    pub fn arena_bytes(&self, patches: usize, tokens: usize) -> usize {
        let mut total = 0usize;
        let mut add = |elements: usize| total += (elements * 2).next_multiple_of(256);

        let vc = &self.backbone_config.vision;
        let (vision_hidden, vision_inner) = (vc.hidden_size, vc.intermediate_size);
        let merge_sq = vc.spatial_merge_size * vc.spatial_merge_size;

        // Vision: patch embedding and the positioned residual.
        add(patches * vision_hidden); // gemm output
        add(patches * vision_hidden); // biased
        add(patches * vision_hidden); // + position table
        for _ in 0..vc.depth {
            add(patches * vision_hidden); // norm1
            add(patches * 3 * vision_hidden); // qkv
            for _ in 0..3 {
                add(patches * vision_hidden); // q, k, v as heads
            }
            add(patches * vision_hidden); // attention output
            add(patches * vision_hidden); // proj
            add(patches * vision_hidden); // residual
            add(patches * vision_hidden); // norm2
            add(patches * vision_inner); // fc1
            add(patches * vision_inner); // activated
            add(patches * vision_hidden); // fc2
            add(patches * vision_hidden); // block output
        }
        // The three deepstack states are kept past their block.
        add(vc.deepstack_visual_indexes.len() * patches * vision_hidden);
        // Primary merger: norm over `hidden`, then the grouped MLP.
        let merged_rows = patches / merge_sq.max(1);
        let merged_width = vision_hidden * merge_sq;
        add(patches * vision_hidden); // primary merger normalizes pre-shuffle
        for index in 0..(1 + vc.deepstack_visual_indexes.len()) {
            if index > 0 {
                add(merged_rows * merged_width); // deepstack normalizes post-shuffle
            }
            add(merged_rows * merged_width); // fc1
            add(merged_rows * merged_width); // activated
            add(merged_rows * vc.out_hidden_size); // fc2
            add(merged_rows * vc.out_hidden_size); // biased
        }

        // Text stack.
        let text = &self.backbone_config.text;
        let (width, head_dim) = (text.hidden_size, text.head_dim);
        add(tokens * width); // token embedding
        for _ in 0..text.n_layers {
            add(tokens * width); // attn norm
            add(tokens * text.n_heads * head_dim); // q
            add(tokens * text.n_kv_heads * head_dim); // k
            add(tokens * text.n_kv_heads * head_dim); // v
            add(tokens * text.n_heads * head_dim); // q norm
            add(tokens * text.n_kv_heads * head_dim); // k norm
            add(tokens * text.n_heads * head_dim); // attention
            add(tokens * width); // output projection
            add(tokens * width); // residual
            add(tokens * width); // ffn norm
            add(tokens * text.intermediate_size); // gate
            add(tokens * text.intermediate_size); // up
            add(tokens * text.intermediate_size); // gated
            add(tokens * width); // down
            add(tokens * width); // layer output
        }

        // VLLN adapter and its VL self-attention blocks.
        let vl_width = self.config.backbone_embedding_dim;
        add(tokens * vl_width);
        let vl_layers = self
            .config
            .vl_self_attention
            .as_ref()
            .filter(|_| self.config.use_vl_self_attention)
            .map_or(0, |vl| vl.num_layers);
        for _ in 0..vl_layers {
            add(tokens * vl_width); // norm1
            for _ in 0..3 {
                add(tokens * vl_width); // raw projection
                add(tokens * vl_width); // biased
            }
            add(tokens * vl_width); // attention
            add(tokens * vl_width); // output projection
            add(tokens * vl_width); // residual
            add(tokens * vl_width); // norm3
            add(tokens * 4 * vl_width); // ffn in
            add(tokens * 4 * vl_width); // activated
            add(tokens * vl_width); // ffn out
            add(tokens * vl_width); // block output
        }
        // The two gathered backbone row groups together cover every token.
        add(tokens * vl_width);

        // Action head: the state token, then the flow loop.
        let dit_width = self.config.input_embedding_dim;
        let (horizon, action_dim) = (self.config.action_horizon, self.config.max_action_dim);
        let rows = horizon + 1;
        let state_inner = 1024;
        add(state_inner);
        add(state_inner);
        add(dit_width);
        add(dit_width);
        add(horizon * action_dim); // the action chunk
        for _ in 0..self.config.num_inference_timesteps {
            add(horizon * action_dim); // euler output
            add(horizon * dit_width); // W1
            add(horizon * dit_width); // biased
            add(horizon * 2 * dit_width); // concat with the time embedding
            add(horizon * dit_width); // W2
            add(horizon * dit_width); // activated
            add(horizon * dit_width); // W3
            add(horizon * dit_width); // + position
            add(rows * dit_width); // state prepended
            for _ in 0..self.config.diffusion.num_layers {
                add(rows * dit_width); // adaptive norm
                add(rows * dit_width); // raw q
                add(rows * dit_width); // biased q
                // K and V are projected from the attention *source*, which for
                // a cross-attention block is a backbone row group -- the image
                // group alone is wider than the 41 state/action rows. Bounding
                // by `tokens` covers every group and the self-attention case.
                for _ in 0..2 {
                    add(tokens * dit_width); // raw k/v
                    add(tokens * dit_width); // biased k/v
                }
                add(rows * dit_width); // attention
                add(rows * dit_width); // output projection
                add(rows * dit_width); // residual
                add(rows * dit_width); // ffn norm
                add(rows * 4 * dit_width); // ffn in
                add(rows * 4 * dit_width); // activated
                add(rows * dit_width); // ffn out
                add(rows * dit_width); // block output
            }
            add(rows * dit_width); // final adaptive norm
            add(rows * self.config.hidden_size); // output projection
            add(rows * self.config.hidden_size); // biased
            add(rows * state_inner); // decoder in
            add(rows * state_inner); // activated
            add(rows * action_dim); // decoder out
            add(rows * action_dim); // biased
            add(horizon * action_dim); // velocity
        }
        total
    }
}

impl std::fmt::Debug for HipGr00tModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HipGr00tModel")
            .field("select_layer", &self.config.select_layer)
            .field("dit_blocks", &self.config.diffusion.num_layers)
            .field("flow_steps", &self.conditioning.steps())
            .field("constants_used_mib", &(self.constants.used() >> 20))
            .finish()
    }
}

/// Reject a model whose geometry the port cannot serve, before any upload.
pub fn validate_servable(config: &Gr00tConfig) -> Result<()> {
    if !config.use_alternate_vl_dit {
        return Err(Error::Contract(
            "GR00T on HIP implements AlternateVLDiT only; this checkpoint disables it",
        ));
    }
    if config.num_inference_timesteps == 0 {
        return Err(Error::Contract(
            "GR00T needs at least one inference timestep",
        ));
    }
    Ok(())
}
