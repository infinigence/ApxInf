//! GR00T's semantic blocks: the vision tower, the Qwen3-VL backbone, the
//! VLLN adapter and the DiT action head. Blocks own precision, fusion and
//! fixed weights; they do not capture graphs, cache requests, or run the
//! complete model schedule.
//!
//! The [`Blocks`] trait is the statically dispatched seam the model sees:
//! construction and dispatch of a precision variant stay in
//! [`super::super::model::model`], and everything above it -- runner, capture,
//! workspaces, input binding -- goes through the model interface only.

pub mod action_head;
pub mod backbone;
pub mod dit_source;
pub mod qkv;
pub mod vision;

use std::sync::Arc;

use apxinf_core::{DType, Result, Tensor};
use apxinf_hip::native::HipArena;
use apxinf_hip::HipContext;

use crate::hip_gr00t::config::{Gr00tConfig, Qwen3VLConfig};
use crate::hip_gr00t::weights::HipGr00tWeights;

use action_head::{DitConditioning, DitRowIndices};
use backbone::PreparedBackbone;
use vision::PreparedVision;
pub use vision::VisionOutput;

/// Everything a prepared shape needs: the per-shape host preparation and the
/// per-shape device tables, bound once by the model and replayed by the
/// runner. An opaque handle from the runner's point of view -- it never
/// reaches the block internals.
pub struct PreparedBlocks {
    pub(super) vision: PreparedVision,
    pub(super) backbone: PreparedBackbone,
    pub(super) rows: DitRowIndices,
}

impl PreparedBlocks {
    /// Load this request's token IDs into the plan's fixed buffer. The only
    /// per-request value in a prepared shape: two prompts of the same length
    /// with their image placeholders in the same places share a plan.
    pub fn refill_tokens(&self, token_ids: &[u32]) -> Result<()> {
        self.backbone.refill_tokens(token_ids)
    }
}

/// Resource requirements for one shape, supplied by the blocks. Execution
/// owns allocation; this description allocates nothing.
pub struct WorkspaceRequirements {
    pub arena_bytes: usize,
}

/// The GR00T blocks seam. One implementation today (BF16); the trait keeps
/// construction and dispatch of further precision variants in `model/model.rs`
/// instead of leaking into the runner.
pub trait Blocks {
    fn context(&self) -> &Arc<HipContext>;
    fn config(&self) -> &Gr00tConfig;
    fn backbone_config(&self) -> &Qwen3VLConfig;
    /// The uploaded, device-resident weights.
    fn weights(&self) -> &HipGr00tWeights;
    /// Static timestep conditioning, precomputed at load for the model's
    /// lifetime. Owns its backing arena.
    fn conditioning(&self) -> &DitConditioning;

    /// Bind every allocation- and dispatch-relevant shape of one request
    /// shape: the interpolated vision positions, the mRoPE positions, the
    /// token and image-row tables, and the DiT cross-attention row groups.
    fn prepare_shape(
        &self,
        patches: usize,
        grid_thw: &[[u32; 3]],
        token_ids: &[u32],
        attention_mask: &[u8],
    ) -> Result<PreparedBlocks>;

    /// The activation arena requirement for one request of this shape. An
    /// audited bound, not an estimate: every `arena.tensor` in the forward
    /// path is counted here, with the arena's 256-byte alignment.
    fn workspace_requirements(&self, patches: usize, tokens: usize) -> Result<WorkspaceRequirements>;

    /// The vision tower alone, for a benchmark attributing latency by stage.
    fn vision_stage(
        &self,
        arena: &HipArena,
        pixel_values: &Tensor,
        vision: &PreparedVision,
    ) -> Result<VisionOutput>;

    /// The Qwen3-VL text backbone alone.
    fn backbone_stage(
        &self,
        arena: &HipArena,
        vision_output: &VisionOutput,
        backbone: &PreparedBackbone,
    ) -> Result<Tensor>;

    /// The VLLN adapter and VL self-attention blocks alone.
    fn adapter_stage(&self, arena: &HipArena, hidden: &Tensor) -> Result<Tensor>;

    /// The DiT action head alone: split, flow loop, decode.
    fn action_stage(
        &self,
        arena: &HipArena,
        adapted: &Tensor,
        state: &Tensor,
        noise: &Tensor,
        rows: &DitRowIndices,
        embodiment: &crate::hip_gr00t::weights::Embodiment,
    ) -> Result<Tensor>;

    /// One inference, every activation from `arena`. `blocks` is what
    /// [`Self::prepare_shape`] returned for this shape.
    fn infer(
        &self,
        arena: &HipArena,
        pixel_values: &Tensor,
        state: &Tensor,
        noise: &Tensor,
        blocks: &PreparedBlocks,
        embodiment: &crate::hip_gr00t::weights::Embodiment,
    ) -> Result<Tensor>;
}

/// The BF16 blocks: the only precision GR00T ships on HIP (gfx1151 has no FP8
/// tensor cores).
pub struct Bf16Blocks {
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
}

impl Bf16Blocks {
    pub fn new(
        ctx: Arc<HipContext>,
        config: Gr00tConfig,
        backbone_config: Qwen3VLConfig,
        weights: &crate::hip_gr00t::weights::host::Gr00tWeights,
    ) -> Result<Self> {
        const CONSTANT_ARENA_BYTES: usize = 64 << 20;
        let constants = HipArena::new(&ctx, CONSTANT_ARENA_BYTES)?;
        let (device_weights, _) = crate::hip_gr00t::weights::HipGr00tWeights::from_checkpoint(
            &weights,
            &config,
            &backbone_config,
            // Embodiment 0 only to satisfy the constructor; the runner owns
            // embodiment selection now and uploads the requested one, so
            // this copy is dropped immediately.
            0,
            &ctx,
        )?;
        let conditioning =
            DitConditioning::new(&ctx, &config, &device_weights, &constants)?;
        let vision_pos_embed = weights.backbone_vision.pos_embed.clone();
        Ok(Self {
            ctx,
            config,
            backbone_config,
            weights: device_weights,
            conditioning,
            constants,
            vision_pos_embed,
        })
    }
}

impl Blocks for Bf16Blocks {
    fn context(&self) -> &Arc<HipContext> {
        &self.ctx
    }

    fn config(&self) -> &Gr00tConfig {
        &self.config
    }

    fn backbone_config(&self) -> &Qwen3VLConfig {
        &self.backbone_config
    }

    fn weights(&self) -> &HipGr00tWeights {
        &self.weights
    }

    fn conditioning(&self) -> &DitConditioning {
        &self.conditioning
    }

    fn prepare_shape(
        &self,
        patches: usize,
        grid_thw: &[[u32; 3]],
        token_ids: &[u32],
        attention_mask: &[u8],
    ) -> Result<PreparedBlocks> {
        let vision_plan = vision::prepare(
            &self.ctx,
            &self.backbone_config,
            &self.vision_pos_embed,
            grid_thw,
            patches,
        )?;
        let backbone_plan = backbone::prepare(
            &self.ctx,
            &self.backbone_config,
            token_ids,
            attention_mask,
            grid_thw,
        )?;
        let groups = super::token_groups::build_backbone_token_groups(
            token_ids,
            attention_mask,
            self.backbone_config.image_token_id,
        )?;
        let rows = DitRowIndices::new(&self.ctx, &groups.non_image, &groups.image)?;
        Ok(PreparedBlocks {
            vision: vision_plan,
            backbone: backbone_plan,
            rows,
        })
    }

    fn workspace_requirements(&self, patches: usize, tokens: usize) -> Result<WorkspaceRequirements> {
        Ok(WorkspaceRequirements {
            arena_bytes: workspace_bytes(&self.config, &self.backbone_config, patches, tokens),
        })
    }

    fn vision_stage(
        &self,
        arena: &HipArena,
        pixel_values: &Tensor,
        vision: &PreparedVision,
    ) -> Result<VisionOutput> {
        vision::forward(
            &self.ctx,
            &self.backbone_config,
            &self.weights,
            arena,
            pixel_values,
            vision,
        )
    }

    fn backbone_stage(
        &self,
        arena: &HipArena,
        vision_output: &VisionOutput,
        backbone: &PreparedBackbone,
    ) -> Result<Tensor> {
        backbone::forward(
            &self.ctx,
            &self.backbone_config,
            &self.weights,
            arena,
            vision_output,
            backbone,
        )
    }

    fn adapter_stage(&self, arena: &HipArena, hidden: &Tensor) -> Result<Tensor> {
        backbone::forward_adapter(&self.ctx, &self.config, &self.weights, arena, hidden)
    }

    fn action_stage(
        &self,
        arena: &HipArena,
        adapted: &Tensor,
        state: &Tensor,
        noise: &Tensor,
        rows: &DitRowIndices,
        embodiment: &crate::hip_gr00t::weights::Embodiment,
    ) -> Result<Tensor> {
        let split = action_head::split_backbone(&self.ctx, arena, adapted, rows)?;
        action_head::forward(
            &self.ctx,
            &self.config,
            &self.weights,
            embodiment,
            &self.conditioning,
            arena,
            &split,
            state,
            noise,
        )
    }

    fn infer(
        &self,
        arena: &HipArena,
        pixel_values: &Tensor,
        state: &Tensor,
        noise: &Tensor,
        blocks: &PreparedBlocks,
        embodiment: &crate::hip_gr00t::weights::Embodiment,
    ) -> Result<Tensor> {
        let vision_output = vision::forward(
            &self.ctx,
            &self.backbone_config,
            &self.weights,
            arena,
            pixel_values,
            &blocks.vision,
        )?;
        let hidden = backbone::forward(
            &self.ctx,
            &self.backbone_config,
            &self.weights,
            arena,
            &vision_output,
            &blocks.backbone,
        )?;
        let adapted = backbone::forward_adapter(
            &self.ctx,
            &self.config,
            &self.weights,
            arena,
            &hidden,
        )?;
        let split = action_head::split_backbone(&self.ctx, arena, &adapted, &blocks.rows)?;
        action_head::forward(
            &self.ctx,
            &self.config,
            &self.weights,
            embodiment,
            &self.conditioning,
            arena,
            &split,
            state,
            noise,
        )
    }
}

/// The audited activation-arena bound for one request of this shape: every
/// `arena.tensor` in the forward path, in order, with the arena's 256-byte
/// alignment. An arena that is one allocation short fails the run outright,
/// so this walks the same sequence the code does and a reviewer can diff the
/// two.
fn workspace_bytes(
    config: &Gr00tConfig,
    backbone_config: &Qwen3VLConfig,
    patches: usize,
    tokens: usize,
) -> usize {
    let _ = DType::BF16; // element width is 2 bytes, kept for readability below
    let mut total = 0usize;
    let mut add = |elements: usize| total += (elements * 2).next_multiple_of(256);

    let vc = &backbone_config.vision;
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
    let text = &backbone_config.text;
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
    let vl_width = config.backbone_embedding_dim;
    add(tokens * vl_width);
    let vl_layers = config
        .vl_self_attention
        .as_ref()
        .filter(|_| config.use_vl_self_attention)
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
    let dit_width = config.input_embedding_dim;
    let (horizon, action_dim) = (config.action_horizon, config.max_action_dim);
    let rows = horizon + 1;
    let state_inner = 1024;
    add(state_inner);
    add(state_inner);
    add(dit_width);
    add(dit_width);
    add(horizon * action_dim); // the action chunk
    for _ in 0..config.num_inference_timesteps {
        add(horizon * action_dim); // euler output
        add(horizon * dit_width); // W1
        add(horizon * dit_width); // biased
        add(horizon * 2 * dit_width); // concat with the time embedding
        add(horizon * dit_width); // W2
        add(horizon * dit_width); // activated
        add(horizon * dit_width); // W3
        add(horizon * dit_width); // + position
        add(rows * dit_width); // state prepended
        for _ in 0..config.diffusion.num_layers {
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
        add(rows * config.hidden_size); // output projection
        add(rows * config.hidden_size); // biased
        add(rows * state_inner); // decoder in
        add(rows * state_inner); // activated
        add(rows * action_dim); // decoder out
        add(rows * action_dim); // biased
        add(horizon * action_dim); // velocity
    }
    total
}

