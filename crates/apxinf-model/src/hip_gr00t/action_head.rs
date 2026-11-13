//! GR00T's flow-matching DiT action head on HIP, BF16 eager.
//!
//! Four Euler steps over a 32-block DiT. Each block is adaptively normalized
//! by the timestep, then attends either to the backbone (alternating between
//! its non-image and image rows) or to itself, per
//! [`super::geometry::dit_attention_source`].
//!
//! The timestep conditioning is **static**: nothing in it depends on the
//! observation, so all four steps' modulations are computed once at load time
//! and only indexed at inference. That is also what makes the DiT a fixed
//! profile for a future graph capture.

use std::sync::Arc;

use apxinf_core::{DType, Error, Result, Tensor};
use apxinf_hip::native::{self, Activation, AttentionGeometry, HipArena, PreparedRowIndices};
use apxinf_hip::HipContext;

use super::config::Gr00tConfig;
use super::geometry::Gr00tDitAttentionSource;
use super::math::{action_timestep_embedding, dit_timestep_projection, flow_schedule};
use super::stages::{NoStages, StageSink};
use super::weights::{DitBlock, Embodiment, HipGr00tWeights, Linear};

/// Everything about the four timesteps that the observation cannot change.
///
/// Built once per loaded model. The modulations each cost three small GEMMs
/// and a SiLU, so computing them per request would be pure waste -- and would
/// also put host-independent work inside a graph capture region.
pub struct DitConditioning {
    /// `[step][block]` of `[2 * width]`, laid out `[scale, shift]`.
    block_modulations: Vec<Vec<Tensor>>,
    /// `[step]` of `[2 * width]` for the final adaptive norm.
    output_modulations: Vec<Tensor>,
    /// `[step]` of `[horizon, width]`, the action encoder's time input.
    time_embeddings: Vec<Tensor>,
    /// `[step]`, the Euler step size.
    deltas: Vec<f32>,
    /// The DiT's post-attention LayerNorm has no learnable affine, so it runs
    /// as [`native::layer_norm`] against ones and zeros.
    unit_weight: Tensor,
    zero_bias: Tensor,
}

impl DitConditioning {
    pub fn steps(&self) -> usize {
        self.deltas.len()
    }

    /// Precompute the static conditioning. `arena` keeps the results for the
    /// model's lifetime, so it must not be the per-request arena.
    pub fn new(
        ctx: &Arc<HipContext>,
        cfg: &Gr00tConfig,
        weights: &HipGr00tWeights,
        arena: &HipArena,
    ) -> Result<Self> {
        let width = cfg.input_embedding_dim;
        let schedule = flow_schedule(cfg.num_inference_timesteps, cfg.num_timestep_buckets)?;
        let steps = schedule.len();

        let ones = vec![half::bf16::ONE; width];
        let unit_weight = upload(ctx, arena, &Tensor::from_bf16(vec![width], &ones)?)?;
        let zero_bias = upload(ctx, arena, &Tensor::zeros(vec![width], DType::BF16))?;

        let mut block_modulations = Vec::with_capacity(steps);
        let mut output_modulations = Vec::with_capacity(steps);
        let mut time_embeddings = Vec::with_capacity(steps);
        let mut deltas = Vec::with_capacity(steps);

        // Scratch for the three projections; freed with `scratch` at the end.
        let scratch = HipArena::new(ctx, 8 << 20)?;
        for step in &schedule {
            deltas.push(step.delta);

            // The action encoder's time input: one embedding repeated across
            // the horizon, so the concatenation downstream is a plain copy.
            let embedding = action_timestep_embedding(step.discrete_timestep, width)?;
            let repeated: Vec<half::bf16> = (0..cfg.action_horizon)
                .flat_map(|_| embedding.iter().copied())
                .map(half::bf16::from_f32)
                .collect();
            time_embeddings.push(upload(
                ctx,
                arena,
                &Tensor::from_bf16(vec![cfg.action_horizon, width], &repeated)?,
            )?);

            // SiLU-MLP over the sinusoidal projection, then SiLU again --
            // the DiT's shared conditioning vector for this step.
            let projection: Vec<half::bf16> = dit_timestep_projection(step.discrete_timestep, 256)?
                .into_iter()
                .map(half::bf16::from_f32)
                .collect();
            let projection = upload(
                ctx,
                &scratch,
                &Tensor::from_bf16(vec![1, 256], &projection)?,
            )?;
            let inner = weights.timestep_in.weight.shape().dims()[1];
            let h = scratch.tensor(vec![1, inner], DType::BF16)?;
            native::gemm(ctx, &projection, &weights.timestep_in.weight, &h)?;
            let activated = scratch.tensor(vec![1, inner], DType::BF16)?;
            native::bias_activation(
                ctx,
                &h,
                weights.timestep_in.bias.as_ref(),
                Activation::Silu,
                &activated,
            )?;
            let timestep = scratch.tensor(vec![1, width], DType::BF16)?;
            native::gemm(ctx, &activated, &weights.timestep_out.weight, &timestep)?;
            let conditioning = scratch.tensor(vec![1, width], DType::BF16)?;
            native::bias_activation(
                ctx,
                &timestep,
                weights.timestep_out.bias.as_ref(),
                Activation::Silu,
                &conditioning,
            )?;

            let modulate = |linear: &Linear| -> Result<Tensor> {
                let raw = scratch.tensor(vec![1, 2 * width], DType::BF16)?;
                native::gemm(ctx, &conditioning, &linear.weight, &raw)?;
                let out = arena.tensor(vec![2 * width], DType::BF16)?;
                native::bias_activation(
                    ctx,
                    &raw,
                    linear.bias.as_ref(),
                    Activation::None,
                    &out.reshape(vec![1, 2 * width])?,
                )?;
                Ok(out)
            };
            block_modulations.push(
                weights
                    .dit_blocks
                    .iter()
                    .map(|block| modulate(&block.adaptive_norm))
                    .collect::<Result<Vec<_>>>()?,
            );
            output_modulations.push(modulate(&weights.output_modulation)?);
        }
        Ok(Self {
            block_modulations,
            output_modulations,
            time_embeddings,
            deltas,
            unit_weight,
            zero_bias,
        })
    }
}

/// The backbone rows the DiT's alternating cross-attention reads.
pub struct DitBackbone {
    pub non_image: Tensor,
    pub image: Tensor,
}

/// Every cross-attending block's K and V, projected once for the whole flow.
///
/// A DiT block cross-attends to a backbone row group, and the Euler loop does
/// not change those rows -- only the action tokens evolve. Projecting K and V
/// inside the loop therefore recomputes the same values on every step. With 16
/// of 32 blocks cross-attending, that is 96 of 128 projections doing work
/// already done.
///
/// Indexed by block, with `None` for the blocks that attend to their own
/// normalized state instead; those do change per step and stay in the loop.
pub struct CrossAttentionCache {
    per_block: Vec<Option<(Tensor, Tensor)>>,
}

impl CrossAttentionCache {
    fn build(
        ctx: &HipContext,
        cfg: &Gr00tConfig,
        weights: &HipGr00tWeights,
        arena: &HipArena,
        backbone: &DitBackbone,
    ) -> Result<Self> {
        let width = cfg.input_embedding_dim;
        let (heads, head_dim) = (
            cfg.diffusion.num_attention_heads,
            cfg.diffusion.attention_head_dim,
        );
        let mut per_block = Vec::with_capacity(weights.dit_blocks.len());
        for (index, block) in weights.dit_blocks.iter().enumerate() {
            let source = match super::geometry::dit_attention_source(cfg, index)? {
                Gr00tDitAttentionSource::NonImageBackbone => &backbone.non_image,
                Gr00tDitAttentionSource::ImageBackbone => &backbone.image,
                // Self-attention reads the block's own normalized state, which
                // the flow loop rewrites every step.
                Gr00tDitAttentionSource::StateActionSelf => {
                    per_block.push(None);
                    continue;
                }
                Gr00tDitAttentionSource::FullBackbone => {
                    return Err(Error::Contract(
                        "GR00T on HIP requires AlternateVLDiT; a full-backbone DiT block is \
                         unsupported",
                    ))
                }
            };
            let rows = source.shape().dims()[0];
            let project = |linear: &Linear| -> Result<Tensor> {
                let biased = arena.tensor(vec![rows, width], DType::BF16)?;
                native::gemm_bias_activation(
                    ctx,
                    source,
                    &linear.weight,
                    linear.bias.as_ref(),
                    Activation::None,
                    &biased,
                )?;
                biased.reshape(vec![rows, heads, head_dim])
            };
            per_block.push(Some((project(&block.k)?, project(&block.v)?)));
        }
        Ok(Self { per_block })
    }

    fn get(&self, block: usize) -> Option<&(Tensor, Tensor)> {
        self.per_block.get(block).and_then(Option::as_ref)
    }
}

/// Device-resident row indices for [`split_backbone`], built once per plan.
///
/// The two row groups are part of the plan key -- which rows are image
/// placeholders is a function of the token layout and the grid -- so the index
/// buffers are plan-invariant and belong here rather than in the forward path.
///
/// Keeping them out of the forward path is also what lets the forward be
/// captured into a HIP graph: allocating device memory during capture is
/// illegal, and a host-to-device copy captured from a `Vec` that is rebuilt
/// every call would replay from a pointer that no longer exists.
pub struct DitRowIndices {
    non_image: PreparedRowIndices,
    image: PreparedRowIndices,
    non_image_rows: usize,
    image_rows: usize,
}

impl DitRowIndices {
    /// The two row groups partition the adapted matrix's rows, so the bound
    /// they are validated against is their total; an out-of-range row in
    /// either group would gather from past the matrix into the next arena
    /// allocation.
    pub fn new(ctx: &Arc<HipContext>, non_image: &[usize], image: &[usize]) -> Result<Self> {
        let bound = non_image.len() + image.len();
        Ok(Self {
            non_image: PreparedRowIndices::new(ctx, non_image, bound)?,
            image: PreparedRowIndices::new(ctx, image, bound)?,
            non_image_rows: non_image.len(),
            image_rows: image.len(),
        })
    }
}

/// Split the adapted backbone features into the two row groups, on device.
pub fn split_backbone(
    ctx: &Arc<HipContext>,
    arena: &HipArena,
    adapted: &Tensor,
    indices: &DitRowIndices,
) -> Result<DitBackbone> {
    let width = adapted.shape().dims()[1];
    let gather = |rows: usize, ids: &PreparedRowIndices| -> Result<Tensor> {
        let out = arena.tensor(vec![rows, width], DType::BF16)?;
        native::gather_rows(ctx, adapted, ids, &out)?;
        Ok(out)
    };
    Ok(DitBackbone {
        non_image: gather(indices.non_image_rows, &indices.non_image)?,
        image: gather(indices.image_rows, &indices.image)?,
    })
}

/// Encode the robot state into the DiT's leading token. One `[1, width]` row.
pub fn encode_state(
    ctx: &HipContext,
    arena: &HipArena,
    embodiment: &Embodiment,
    state: &Tensor,
) -> Result<Tensor> {
    let inner = embodiment.state_encoder_in.weight.shape().dims()[1];
    let width = embodiment.state_encoder_out.weight.shape().dims()[1];
    let h = arena.tensor(vec![1, inner], DType::BF16)?;
    native::gemm(ctx, state, &embodiment.state_encoder_in.weight, &h)?;
    let activated = arena.tensor(vec![1, inner], DType::BF16)?;
    native::bias_activation(
        ctx,
        &h,
        embodiment.state_encoder_in.bias.as_ref(),
        Activation::Relu,
        &activated,
    )?;
    let raw = arena.tensor(vec![1, width], DType::BF16)?;
    native::gemm(ctx, &activated, &embodiment.state_encoder_out.weight, &raw)?;
    let out = arena.tensor(vec![1, width], DType::BF16)?;
    native::bias_activation(
        ctx,
        &raw,
        embodiment.state_encoder_out.bias.as_ref(),
        Activation::None,
        &out,
    )?;
    Ok(out)
}

/// Integrate the flow from `noise` to the action chunk.
///
/// `state` is `[1, state_input_dim]` and `noise` is `[horizon,
/// max_action_dim]`, both already on the device. Returns `[horizon,
/// max_action_dim]` in the normalized action domain.
#[allow(clippy::too_many_arguments)]
pub fn forward(
    ctx: &HipContext,
    cfg: &Gr00tConfig,
    weights: &HipGr00tWeights,
    embodiment: &Embodiment,
    conditioning: &DitConditioning,
    arena: &HipArena,
    backbone: &DitBackbone,
    state: &Tensor,
    noise: &Tensor,
) -> Result<Tensor> {
    forward_observed(
        ctx,
        cfg,
        weights,
        embodiment,
        conditioning,
        arena,
        backbone,
        state,
        noise,
        &mut NoStages,
    )
}

/// [`forward`], reporting the reference's stage names to `stages`.
#[allow(clippy::too_many_arguments)]
pub fn forward_observed(
    ctx: &HipContext,
    cfg: &Gr00tConfig,
    weights: &HipGr00tWeights,
    embodiment: &Embodiment,
    conditioning: &DitConditioning,
    arena: &HipArena,
    backbone: &DitBackbone,
    state: &Tensor,
    noise: &Tensor,
    stages: &mut dyn StageSink,
) -> Result<Tensor> {
    let (horizon, action_dim) = (cfg.action_horizon, cfg.max_action_dim);
    if noise.shape().dims() != [horizon, action_dim] {
        return Err(Error::ShapeMismatch {
            expected: format!("[{horizon}, {action_dim}]"),
            got: noise.shape().to_string(),
        });
    }
    if weights.dit_blocks.len() != conditioning.block_modulations[0].len() {
        return Err(Error::Other(format!(
            "GR00T has {} DiT blocks but the conditioning was built for {}",
            weights.dit_blocks.len(),
            conditioning.block_modulations[0].len()
        )));
    }

    let state_features = encode_state(ctx, arena, embodiment, state)?;
    stages.stage("state_features", &state_features)?;

    let mut actions = arena.tensor(vec![horizon, action_dim], DType::BF16)?;
    native::copy_rows(ctx, noise, &actions, 0)?;

    // Cross-attention K and V, once for the whole flow rather than once per
    // step. Their input is a backbone row group, which the Euler loop does not
    // touch, so the four steps were projecting identical values four times:
    // 16 of the 32 blocks cross-attend, and at two projections each that is 96
    // of the 128 calls doing work already done.
    let cross = CrossAttentionCache::build(ctx, cfg, weights, arena, backbone)?;

    for step in 0..conditioning.steps() {
        stages.stage(&format!("step_{step}_action_in"), &actions)?;
        let velocity = flow_step(
            ctx,
            cfg,
            weights,
            embodiment,
            conditioning,
            arena,
            backbone,
            &cross,
            &state_features,
            &actions,
            step,
            stages,
        )?;
        stages.stage(&format!("step_{step}_velocity"), &velocity)?;
        let next = arena.tensor(vec![horizon, action_dim], DType::BF16)?;
        native::euler_update(ctx, &actions, &velocity, conditioning.deltas[step], &next)?;
        actions = next;
        stages.stage(&format!("step_{step}_action_out"), &actions)?;
    }
    stages.stage("final_action", &actions)?;
    Ok(actions)
}

/// One Euler step: encode the current actions, run the DiT, decode a velocity.
#[allow(clippy::too_many_arguments)]
fn flow_step(
    ctx: &HipContext,
    cfg: &Gr00tConfig,
    weights: &HipGr00tWeights,
    embodiment: &Embodiment,
    conditioning: &DitConditioning,
    arena: &HipArena,
    backbone: &DitBackbone,
    cross: &CrossAttentionCache,
    state_features: &Tensor,
    actions: &Tensor,
    step: usize,
    stages: &mut dyn StageSink,
) -> Result<Tensor> {
    let (horizon, action_dim) = (cfg.action_horizon, cfg.max_action_dim);
    let width = cfg.input_embedding_dim;
    let eps = cfg.diffusion.norm_eps;
    let (heads, head_dim) = (
        cfg.diffusion.num_attention_heads,
        cfg.diffusion.attention_head_dim,
    );

    // Action encoder: W1, concat the step's time embedding, W2 with SiLU, W3.
    let encoded = arena.tensor(vec![horizon, width], DType::BF16)?;
    native::gemm(ctx, actions, &embodiment.action_encoder_in.weight, &encoded)?;
    let biased = arena.tensor(vec![horizon, width], DType::BF16)?;
    native::bias_activation(
        ctx,
        &encoded,
        embodiment.action_encoder_in.bias.as_ref(),
        Activation::None,
        &biased,
    )?;
    let joined = arena.tensor(vec![horizon, 2 * width], DType::BF16)?;
    native::copy_columns(ctx, &biased, &joined, 0)?;
    native::copy_columns(ctx, &conditioning.time_embeddings[step], &joined, width)?;

    let timed = arena.tensor(vec![horizon, width], DType::BF16)?;
    native::gemm(ctx, &joined, &embodiment.action_encoder_time.weight, &timed)?;
    let activated = arena.tensor(vec![horizon, width], DType::BF16)?;
    native::bias_activation(
        ctx,
        &timed,
        embodiment.action_encoder_time.bias.as_ref(),
        Activation::Silu,
        &activated,
    )?;
    let projected = arena.tensor(vec![horizon, width], DType::BF16)?;
    native::gemm(
        ctx,
        &activated,
        &embodiment.action_encoder_out.weight,
        &projected,
    )?;
    let action_features = arena.tensor(vec![horizon, width], DType::BF16)?;
    // The learned position embedding covers the action tokens only; the state
    // token is prepended afterwards and deliberately has none.
    match &weights.action_position_embedding {
        Some(position) => native::add_position(
            ctx,
            &projected,
            embodiment.action_encoder_out.bias.as_ref(),
            position,
            horizon,
            &action_features,
        )?,
        None => native::bias_activation(
            ctx,
            &projected,
            embodiment.action_encoder_out.bias.as_ref(),
            Activation::None,
            &action_features,
        )?,
    }
    stages.stage(&format!("step_{step}_action_features"), &action_features)?;

    let tokens = horizon + 1;
    let mut hidden = arena.tensor(vec![tokens, width], DType::BF16)?;
    native::copy_rows(ctx, state_features, &hidden, 0)?;
    native::copy_rows(ctx, &action_features, &hidden, 1)?;
    // The reference numbers the DiT states from its *input*, so block `i`'s
    // output is `dit_hidden_{i + 1}`. Matching that numbering is what lets the
    // 33 reference tensors per step be compared without an offset table --
    // and an offset here is indistinguishable from a real error, since a
    // block's output correlates well with its neighbour's.
    stages.stage(&format!("step_{step}_dit_hidden_0"), &hidden)?;

    // The first block normalizes its own input; every later one is handed the
    // normalization its predecessor already produced, because that pass is
    // fused into the predecessor's residual add.
    let mut normalized = arena.tensor(vec![tokens, width], DType::BF16)?;
    native::ada_layer_norm(
        ctx,
        &hidden,
        &conditioning.block_modulations[step][0],
        eps,
        &normalized,
    )?;

    for (index, block) in weights.dit_blocks.iter().enumerate() {
        let source = match super::geometry::dit_attention_source(cfg, index)? {
            Gr00tDitAttentionSource::NonImageBackbone => &backbone.non_image,
            Gr00tDitAttentionSource::ImageBackbone => &backbone.image,
            Gr00tDitAttentionSource::StateActionSelf => &normalized,
            Gr00tDitAttentionSource::FullBackbone => return Err(Error::Contract(
                "GR00T on HIP requires AlternateVLDiT; a full-backbone DiT block is unsupported",
            )),
        };

        let next_modulation = conditioning.block_modulations[step].get(index + 1);
        let (next_hidden, next_normed) = dit_block(
            ctx,
            arena,
            block,
            conditioning,
            &hidden,
            &normalized,
            source,
            cross.get(index),
            heads,
            head_dim,
            eps,
            next_modulation,
        )?;
        hidden = next_hidden;
        if let Some(next) = next_normed {
            normalized = next;
        }
        stages.stage(&format!("step_{step}_dit_hidden_{}", index + 1), &hidden)?;
    }

    // Final adaptive norm, at its own epsilon, then the shared projection.
    let normalized = arena.tensor(vec![tokens, width], DType::BF16)?;
    native::ada_layer_norm(
        ctx,
        &hidden,
        &conditioning.output_modulations[step],
        OUTPUT_NORM_EPS,
        &normalized,
    )?;
    let out_width = weights.output_projection.weight.shape().dims()[1];
    let raw = arena.tensor(vec![tokens, out_width], DType::BF16)?;
    native::gemm(ctx, &normalized, &weights.output_projection.weight, &raw)?;
    let dit_output = arena.tensor(vec![tokens, out_width], DType::BF16)?;
    native::bias_activation(
        ctx,
        &raw,
        weights.output_projection.bias.as_ref(),
        Activation::None,
        &dit_output,
    )?;
    stages.stage(&format!("step_{step}_dit_output"), &dit_output)?;

    // Action decoder, then drop the state row: the velocity covers the action
    // tokens only.
    let inner = embodiment.action_decoder_in.weight.shape().dims()[1];
    let h = arena.tensor(vec![tokens, inner], DType::BF16)?;
    native::gemm(ctx, &dit_output, &embodiment.action_decoder_in.weight, &h)?;
    let activated = arena.tensor(vec![tokens, inner], DType::BF16)?;
    native::bias_activation(
        ctx,
        &h,
        embodiment.action_decoder_in.bias.as_ref(),
        Activation::Relu,
        &activated,
    )?;
    let decoded = arena.tensor(vec![tokens, action_dim], DType::BF16)?;
    native::gemm(
        ctx,
        &activated,
        &embodiment.action_decoder_out.weight,
        &decoded,
    )?;
    let biased = arena.tensor(vec![tokens, action_dim], DType::BF16)?;
    native::bias_activation(
        ctx,
        &decoded,
        embodiment.action_decoder_out.bias.as_ref(),
        Activation::None,
        &biased,
    )?;
    let velocity = arena.tensor(vec![horizon, action_dim], DType::BF16)?;
    native::copy_row_range(ctx, &biased, 1, horizon, &velocity, 0)?;
    Ok(velocity)
}

/// The final adaptive norm uses a tighter epsilon than the blocks do.
const OUTPUT_NORM_EPS: f32 = 1e-6;

/// One DiT block: attention off `normalized`, residual, non-affine LayerNorm,
/// GELU feed-forward, residual.
///
/// `gr00t/` caches each block's cross-attention K and V across the four steps,
/// since the backbone source does not change. That is a step the performance
/// milestone can take; it changes no value, and leaving it out keeps this path
/// to one shape of code for self- and cross-attention alike.
#[allow(clippy::too_many_arguments)]
/// One DiT block. Returns its hidden state and, when `next_modulation` is
/// given, that state already normalized under the following block's
/// modulation.
///
/// The second output is why the block takes the next block's modulation: a
/// block ends with `bias + residual` and the next begins by normalizing the
/// result, and those two are one kernel. The alternative is to leave them
/// apart and pay a launch per block per flow step -- 128 of them, measured at
/// about 2.2 us each. Returning `None` for the last block keeps the final
/// adaptive norm, which runs at its own epsilon, where it was.
fn dit_block(
    ctx: &HipContext,
    arena: &HipArena,
    block: &DitBlock,
    conditioning: &DitConditioning,
    hidden: &Tensor,
    normalized: &Tensor,
    source: &Tensor,
    // This block's K and V when they were projected ahead of the flow loop;
    // `None` for a self-attending block, whose source changes every step.
    cached_kv: Option<&(Tensor, Tensor)>,
    heads: usize,
    head_dim: usize,
    eps: f32,
    next_modulation: Option<&Tensor>,
) -> Result<(Tensor, Option<Tensor>)> {
    let dims = normalized.shape().dims();
    let (tokens, width) = (dims[0], dims[1]);
    let kv_tokens = source.shape().dims()[0];
    if heads * head_dim != width {
        return Err(Error::Other(format!(
            "GR00T DiT declares {heads} x {head_dim} but the embedding is {width} wide"
        )));
    }

    let project = |linear: &Linear, input: &Tensor, rows: usize| -> Result<Tensor> {
        let raw = arena.tensor(vec![rows, width], DType::BF16)?;
        let biased = arena.tensor(vec![rows, width], DType::BF16)?;
        // Fused: where the recipe routes this shape to hipBLASLt, the bias and
        // activation are applied inside the GEMM epilogue, on the F32 accumulator.
        // Where it does not, this falls back to exactly the two calls it replaces.
        native::gemm_bias_activation(
            ctx,
            input,
            &linear.weight,
            linear.bias.as_ref(),
            Activation::None,
            &biased,
        )?;
        biased.reshape(vec![rows, heads, head_dim])
    };
    // A cross-attending block has only Q to project: its K and V are cached
    // outside the flow loop. A self-attending block recomputes all three from
    // the same normalized state, which is where the fused and serial modes
    // differ.
    let projection = match cached_kv {
        Some((k, v)) => super::qkv::QkvProjection::Separate(
            project(&block.q, normalized, tokens)?,
            k.clone(),
            v.clone(),
        ),
        None => super::qkv::project_qkv(
            ctx, arena, block, normalized, tokens, width, heads, head_dim,
        )?,
    };

    let attended = arena.tensor(vec![tokens, heads, head_dim], DType::BF16)?;
    let geometry = AttentionGeometry {
        batch: 1,
        q_tokens: tokens,
        q_heads: heads,
        kv_heads: heads,
        head_dim,
        kv_tokens,
        causal_query_start: None,
    };
    match &projection {
        super::qkv::QkvProjection::Separate(q, k, v) => {
            native::attention(ctx, q, k, v, &geometry, &attended)?
        }
        super::qkv::QkvProjection::Packed(packed) => {
            // The self-attending block projects K and V from the same
            // `normalized` rows it queries, so the packed sections share the
            // width and every row is a key.
            native::attention_qkv_packed(ctx, packed, width, width, &geometry, &attended)?
        }
    };

    let output = arena.tensor(vec![tokens, width], DType::BF16)?;
    native::gemm(
        ctx,
        &attended.reshape(vec![tokens, width])?,
        &block.output.weight,
        &output,
    )?;

    // Residual, then the DiT's affine-free LayerNorm over it.
    let residual = arena.tensor(vec![tokens, width], DType::BF16)?;
    let ffn_normed = arena.tensor(vec![tokens, width], DType::BF16)?;
    native::bias_residual_layer_norm(
        ctx,
        &output,
        block.output.bias.as_ref(),
        hidden,
        &conditioning.unit_weight,
        &conditioning.zero_bias,
        eps,
        &residual,
        &ffn_normed,
    )?;

    let inner = block.ffn_in.weight.shape().dims()[1];
    let activated = arena.tensor(vec![tokens, inner], DType::BF16)?;
    // GR00T's DiT config defaults `activation_fn` to `"gelu-approximate"` and
    // the released checkpoint does not override it, which diffusers maps to
    // `GELU(approximate="tanh")`. Reading `GELU`'s own signature default
    // (`approximate="none"`) instead of the call site gives the wrong variant.
    // hipBLASLt's GELU epilogue is the tanh form, measured bit for bit, which
    // is why this one may fuse and the merger's exact-erf GELU may not.
    // Fused: where the recipe routes this shape to hipBLASLt, the bias and
    // activation are applied inside the GEMM epilogue, on the F32 accumulator.
    // Where it does not, this falls back to exactly the two calls it replaces.
    native::gemm_bias_activation(
        ctx,
        &ffn_normed,
        &block.ffn_in.weight,
        block.ffn_in.bias.as_ref(),
        Activation::GeluTanh,
        &activated,
    )?;
    let down = arena.tensor(vec![tokens, width], DType::BF16)?;
    native::gemm(ctx, &activated, &block.ffn_out.weight, &down)?;
    let out = arena.tensor(vec![tokens, width], DType::BF16)?;
    // Fused with the next block's adaptive norm where there is one. Bit-exact
    // against the separate pair -- `hidden` is rounded and read back in the
    // same place, which is what the normalization consumes.
    match next_modulation {
        Some(modulation) => {
            let next_normed = arena.tensor(vec![tokens, width], DType::BF16)?;
            native::bias_residual_ada_layer_norm(
                ctx,
                &down,
                block.ffn_out.bias.as_ref(),
                &residual,
                modulation,
                eps,
                &out,
                &next_normed,
            )?;
            Ok((out, Some(next_normed)))
        }
        None => {
            native::bias_residual(ctx, &down, block.ffn_out.bias.as_ref(), &residual, &out)?;
            Ok((out, None))
        }
    }
}

/// Copy a host tensor into a fresh arena slot.
fn upload(ctx: &HipContext, arena: &HipArena, host: &Tensor) -> Result<Tensor> {
    let device = arena.tensor(host.shape().dims().to_vec(), host.dtype())?;
    native::upload(ctx, host, &device)?;
    Ok(device)
}
