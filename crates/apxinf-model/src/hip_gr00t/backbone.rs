//! The Qwen3-VL text stack on HIP, truncated to `select_layer`, BF16 eager.
//!
//! Turns token ids plus the vision tower's output into the hidden state GR00T
//! feeds its action head. Two things make this more than a plain decoder:
//!
//! - **mRoPE**: positions are `(t, h, w)` triples, and an image placeholder run
//!   gets 2D grid coordinates rather than a running index, so the whole image
//!   occupies one position span.
//! - **Deepstack**: the vision tower's three extra embeddings are *added* into
//!   the image rows after text layers 0, 1 and 2.
//!
//! The stack deliberately stops before the language model's final RMSNorm:
//! GR00T reads `hidden_states[-1]`, which Qwen captures before that norm. The
//! action head's own LayerNorm comes next, in the adapter.

use std::sync::Arc;

use apxinf_core::{DType, Error, Result, Tensor};
use apxinf_hip::native::{self, Activation, AttentionGeometry, HipArena, HipDeviceBytes};
use apxinf_hip::HipContext;

use super::config::{Gr00tConfig, Qwen3VLConfig};
use super::stages::{NoStages, StageSink};
use super::vision::VisionOutput;
use super::weights::{HipGr00tWeights, Norm, TextLayer, VlBlock};

/// Per-request host work: the mRoPE positions and the row indices the vision
/// output scatters into, uploaded once.
/// Every LayerNorm in GR00T's action head, VLLN included.
const ADAPTER_EPS: f32 = 1e-5;

pub struct PreparedBackbone {
    /// The only per-request value here: two prompts of the same length with
    /// their image placeholders in the same places share a plan, so this
    /// buffer must be refilled on every run. See [`Self::refill_tokens`].
    token_ids: HipDeviceBytes,
    positions: HipDeviceBytes,
    image_indices: HipDeviceBytes,
    tokens: usize,
    image_rows: usize,
}

impl PreparedBackbone {
    pub fn tokens(&self) -> usize {
        self.tokens
    }

    pub fn image_rows(&self) -> usize {
        self.image_rows
    }

    /// Load this request's token IDs into the plan's fixed buffer.
    ///
    /// **This must run on every inference, not only when the plan is built.**
    /// A plan is keyed by shape -- token *count*, grid, embodiment, and which
    /// rows are image placeholders -- and the processor pads every prompt to
    /// the same length, so two different instructions reuse one plan. Leaving
    /// the buffer at whatever the first request wrote makes the model answer
    /// the first prompt forever: no error, plausible actions, and a robot
    /// that confidently performs the previous task.
    ///
    /// The mRoPE `positions` and `image_indices` beside it are *not* refilled,
    /// and must not be: both are functions of the plan key alone. Positions
    /// come from where the image placeholder runs sit and from the grid;
    /// `image_indices` is exactly `groups.image`. Two requests that share a
    /// key therefore share both.
    pub fn refill_tokens(&self, token_ids: &[u32]) -> Result<()> {
        if token_ids.len() != self.tokens {
            return Err(Error::DataLengthMismatch {
                expected: self.tokens,
                got: token_ids.len(),
            });
        }
        self.token_ids.upload_u32(token_ids)
    }
}

/// Validate the request and upload everything the device path indexes by.
pub fn prepare(
    ctx: &Arc<HipContext>,
    cfg: &Qwen3VLConfig,
    token_ids: &[u32],
    attention_mask: &[u8],
    grid_thw: &[[u32; 3]],
) -> Result<PreparedBackbone> {
    let groups = super::geometry::build_backbone_token_groups(
        token_ids,
        attention_mask,
        cfg.image_token_id,
    )?;
    let positions = multimodal_rope_positions(
        cfg.image_token_id,
        cfg.vision.spatial_merge_size,
        token_ids,
        grid_thw,
    )?;

    let upload = |values: &[u32]| -> Result<HipDeviceBytes> {
        let bytes = HipDeviceBytes::new(ctx, values.len() * std::mem::size_of::<u32>())?;
        bytes.upload_u32(values)?;
        Ok(bytes)
    };
    let indices: Vec<u32> = groups
        .image
        .iter()
        .map(|&index| u32::try_from(index).map_err(|_| Error::Contract("row index overflow")))
        .collect::<Result<_>>()?;
    Ok(PreparedBackbone {
        token_ids: upload(token_ids)?,
        positions: upload(&positions)?,
        image_indices: upload(&indices)?,
        tokens: token_ids.len(),
        image_rows: groups.image.len(),
    })
}

/// Run the text stack. Returns the hidden state before the final RMSNorm --
/// the reference's `backbone_features_raw`.
pub fn forward(
    ctx: &HipContext,
    cfg: &Qwen3VLConfig,
    weights: &HipGr00tWeights,
    arena: &HipArena,
    vision: &VisionOutput,
    prepared: &PreparedBackbone,
) -> Result<Tensor> {
    forward_observed(ctx, cfg, weights, arena, vision, prepared, &mut NoStages)
}

/// [`forward`], reporting each layer's output to `stages`.
#[allow(clippy::too_many_arguments)]
pub fn forward_observed(
    ctx: &HipContext,
    cfg: &Qwen3VLConfig,
    weights: &HipGr00tWeights,
    arena: &HipArena,
    vision: &VisionOutput,
    prepared: &PreparedBackbone,
    stages: &mut dyn StageSink,
) -> Result<Tensor> {
    let text = &cfg.text;
    let (tokens, hidden_size) = (prepared.tokens, text.hidden_size);

    if vision.primary.shape().dims()[0] != prepared.image_rows {
        return Err(Error::Other(format!(
            "GR00T has {} image token positions but the vision tower produced {} rows",
            prepared.image_rows,
            vision.primary.shape().dims()[0]
        )));
    }

    // Token embeddings, unscaled: Qwen3-VL reads the table directly, unlike
    // pi05's Gemma backbone which multiplies by sqrt(width).
    let mut hidden = arena.tensor(vec![tokens, hidden_size], DType::BF16)?;
    native::embedding(
        ctx,
        &weights.token_embedding,
        &prepared.token_ids,
        native::EmbeddingScale::None,
        &hidden,
    )?;
    // The image placeholder rows are replaced, not added to.
    native::scatter_rows(
        ctx,
        &hidden,
        &prepared.image_indices,
        &vision.primary,
        false,
    )?;
    stages.stage("embedded", &hidden)?;

    for (index, layer) in weights.text_layers.iter().enumerate() {
        hidden = text_layer(ctx, cfg, arena, layer, &hidden, prepared, tokens)?;
        // Qwen3-VL injects the deepstack embeddings into the first layers, one
        // per captured vision state, by position rather than by the vision
        // block they came from.
        if let Some(deepstack) = vision.deepstack.get(index) {
            native::scatter_rows(ctx, &hidden, &prepared.image_indices, deepstack, true)?;
        }
        stages.stage(&format!("text_layer_{index:02}"), &hidden)?;
    }
    Ok(hidden)
}

#[allow(clippy::too_many_arguments)]
fn text_layer(
    ctx: &HipContext,
    cfg: &Qwen3VLConfig,
    arena: &HipArena,
    layer: &TextLayer,
    input: &Tensor,
    prepared: &PreparedBackbone,
    tokens: usize,
) -> Result<Tensor> {
    let text = &cfg.text;
    let (hidden_size, head_dim) = (text.hidden_size, text.head_dim);
    let (heads, kv_heads) = (text.n_heads, text.n_kv_heads);
    let eps = text.rms_norm_eps;

    let normed = arena.tensor(vec![tokens, hidden_size], DType::BF16)?;
    native::rms_norm(ctx, input, &layer.attn_norm, eps, &normed)?;

    // Q/K/V are separate projections without bias, and each is RMS-normalized
    // per head before the rotation -- Qwen3's QK-norm.
    let q = arena.tensor(vec![tokens, heads * head_dim], DType::BF16)?;
    let k = arena.tensor(vec![tokens, kv_heads * head_dim], DType::BF16)?;
    let v = arena.tensor(vec![tokens, kv_heads * head_dim], DType::BF16)?;
    native::gemm(ctx, &normed, &layer.q.weight, &q)?;
    native::gemm(ctx, &normed, &layer.k.weight, &k)?;
    native::gemm(ctx, &normed, &layer.v.weight, &v)?;

    // Q/K are separate projections without bias, and each is RMS-normalized
    // per head before the rotation -- Qwen3's QK-norm -- with the rotation in
    // the same kernel: cuda-new's fused rms_norm_apply_mrope_qk. The norm
    // rounds its head to BF16 before rotating, exactly the rounding the
    // separate rms_norm store performed, so the rotated values are
    // bit-identical to the two-kernel sequence.
    let q_rotated = arena.tensor(vec![tokens, heads, head_dim], DType::BF16)?;
    let k_rotated = arena.tensor(vec![tokens, kv_heads, head_dim], DType::BF16)?;
    native::rms_norm_apply_mrope_qk(
        ctx,
        &q.reshape(vec![tokens * heads, head_dim])?,
        &layer.q_norm,
        &k.reshape(vec![tokens * kv_heads, head_dim])?,
        &layer.k_norm,
        &prepared.positions,
        heads,
        kv_heads,
        head_dim,
        eps,
        text.rope_theta,
        text.mrope_section,
        &q_rotated,
        &k_rotated,
    )?;
    let q_heads = q_rotated;
    let k_heads = k_rotated;

    // Causal prefill: query row `i` sees keys `[0, i]`.
    let attended = arena.tensor(vec![tokens, heads, head_dim], DType::BF16)?;
    native::attention(
        ctx,
        &q_heads,
        &k_heads,
        &v.reshape(vec![tokens, kv_heads, head_dim])?,
        &AttentionGeometry {
            batch: 1,
            q_tokens: tokens,
            q_heads: heads,
            kv_heads,
            head_dim,
            kv_tokens: tokens,
            causal_query_start: Some(0),
        },
        &attended,
    )?;

    let projected = arena.tensor(vec![tokens, hidden_size], DType::BF16)?;
    native::gemm(
        ctx,
        &attended.reshape(vec![tokens, heads * head_dim])?,
        &layer.output.weight,
        &projected,
    )?;

    // Residual and the pre-FFN norm in one pass, both biasless in Qwen3.
    let residual = arena.tensor(vec![tokens, hidden_size], DType::BF16)?;
    let ffn_normed = arena.tensor(vec![tokens, hidden_size], DType::BF16)?;
    native::bias_residual_rms_norm(
        ctx,
        &projected,
        None,
        input,
        &layer.ffn_norm,
        eps,
        &residual,
        &ffn_normed,
    )?;

    // SwiGLU with gate and up kept apart, as the checkpoint stores them.
    let inner = layer.gate.weight.shape().dims()[1];
    let gate = arena.tensor(vec![tokens, inner], DType::BF16)?;
    let up = arena.tensor(vec![tokens, inner], DType::BF16)?;
    native::gemm(ctx, &ffn_normed, &layer.gate.weight, &gate)?;
    native::gemm(ctx, &ffn_normed, &layer.up.weight, &up)?;
    let gated = arena.tensor(vec![tokens, inner], DType::BF16)?;
    native::silu_mul(ctx, &gate, &up, &gated)?;
    let down = arena.tensor(vec![tokens, hidden_size], DType::BF16)?;
    native::gemm(ctx, &gated, &layer.down.weight, &down)?;
    let out = arena.tensor(vec![tokens, hidden_size], DType::BF16)?;
    native::bias_residual(ctx, &down, None, &residual, &out)?;
    Ok(out)
}

/// GR00T's VLLN adapter: a LayerNorm over the backbone features, then the
/// action head's VL self-attention blocks. The reference's
/// `backbone_features_post_adapter`.
///
/// The released LIBERO checkpoint sets `use_vl_self_attention` and ships four
/// blocks, so this is on the acceptance path, not an optional extra.
pub fn forward_adapter(
    ctx: &HipContext,
    cfg: &Gr00tConfig,
    weights: &HipGr00tWeights,
    arena: &HipArena,
    hidden: &Tensor,
) -> Result<Tensor> {
    forward_adapter_observed(ctx, cfg, weights, arena, hidden, &mut NoStages)
}

/// [`forward_adapter`], reporting each block's output to `stages`.
pub fn forward_adapter_observed(
    ctx: &HipContext,
    cfg: &Gr00tConfig,
    weights: &HipGr00tWeights,
    arena: &HipArena,
    hidden: &Tensor,
    stages: &mut dyn StageSink,
) -> Result<Tensor> {
    let dims = hidden.shape().dims().to_vec();
    let mut current = match &weights.backbone_layer_norm {
        Some(norm) => {
            let out = arena.tensor(dims.clone(), DType::BF16)?;
            layer_norm_into(ctx, norm, hidden, ADAPTER_EPS, &out)?;
            stages.stage("vlln", &out)?;
            out
        }
        None => hidden.clone(),
    };
    if weights.vl_self_attention.is_empty() {
        return Ok(current);
    }
    let vl = cfg.vl_self_attention.as_ref().ok_or(Error::Contract(
        "GR00T has VL self-attention weights but no config for them",
    ))?;
    for (index, block) in weights.vl_self_attention.iter().enumerate() {
        current = vl_block(
            ctx,
            arena,
            block,
            &current,
            vl.num_attention_heads,
            vl.attention_head_dim,
        )?;
        stages.stage(&format!("vl_self_attention_{index}"), &current)?;
    }
    Ok(current)
}

/// GR00T's `BasicTransformerBlock`: biased non-causal self-attention, then a
/// plain (not gated) GELU feed-forward.
///
/// `SelfAttentionTransformer` defaults `activation_fn` to
/// `"gelu-approximate"`, which diffusers' `FeedForward` maps to
/// `GELU(approximate="tanh")`. The released checkpoint's
/// `vl_self_attention_cfg` carries no `activation_fn`, so that default stands.
/// Only the Qwen vision patch merger -- `nn.GELU()` hardcoded in
/// `transformers` -- wants the exact error function.
fn vl_block(
    ctx: &HipContext,
    arena: &HipArena,
    block: &VlBlock,
    input: &Tensor,
    heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let dims = input.shape().dims();
    let (tokens, width) = (dims[0], dims[1]);
    if heads * head_dim != width {
        return Err(Error::Other(format!(
            "GR00T VL self-attention declares {heads} x {head_dim} but the backbone is {width} wide"
        )));
    }

    let normed = arena.tensor(vec![tokens, width], DType::BF16)?;
    layer_norm_into(ctx, &block.attn_norm, input, ADAPTER_EPS, &normed)?;

    // Q, K and V are separate biased projections here, not one packed matrix.
    let project = |linear: &super::weights::Linear| -> Result<Tensor> {
        let biased = arena.tensor(vec![tokens, width], DType::BF16)?;
    // Fused: where the recipe routes this shape to hipBLASLt, the bias and
    // activation are applied inside the GEMM epilogue, on the F32 accumulator.
    // Where it does not, this falls back to exactly the two calls it replaces.
        native::gemm_bias_activation(
            ctx,
            &normed,
            &linear.weight,
            linear.bias.as_ref(),
            Activation::None,
            &biased,
        )?;
        biased.reshape(vec![tokens, heads, head_dim])
    };
    let (q, k, v) = (project(&block.q)?, project(&block.k)?, project(&block.v)?);

    let attended = arena.tensor(vec![tokens, heads, head_dim], DType::BF16)?;
    native::attention(
        ctx,
        &q,
        &k,
        &v,
        &AttentionGeometry {
            batch: 1,
            q_tokens: tokens,
            q_heads: heads,
            kv_heads: heads,
            head_dim,
            kv_tokens: tokens,
            causal_query_start: None,
        },
        &attended,
    )?;

    let projected = arena.tensor(vec![tokens, width], DType::BF16)?;
    native::gemm(
        ctx,
        &attended.reshape(vec![tokens, width])?,
        &block.output.weight,
        &projected,
    )?;
    let residual = arena.tensor(vec![tokens, width], DType::BF16)?;
    let ffn_normed = arena.tensor(vec![tokens, width], DType::BF16)?;
    let ffn_norm_bias = block
        .ffn_norm
        .bias
        .as_ref()
        .ok_or(Error::Contract("GR00T VL self-attention norm3 needs a bias"))?;
    native::bias_residual_layer_norm(
        ctx,
        &projected,
        block.output.bias.as_ref(),
        input,
        &block.ffn_norm.weight,
        ffn_norm_bias,
        ADAPTER_EPS,
        &residual,
        &ffn_normed,
    )?;

    let inner = block.ffn_in.weight.shape().dims()[1];
    let activated = arena.tensor(vec![tokens, inner], DType::BF16)?;
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
    native::bias_residual(ctx, &down, block.ffn_out.bias.as_ref(), &residual, &out)?;
    Ok(out)
}

fn layer_norm_into(
    ctx: &HipContext,
    norm: &Norm,
    input: &Tensor,
    eps: f32,
    out: &Tensor,
) -> Result<()> {
    let bias = norm
        .bias
        .as_ref()
        .ok_or(Error::Contract("GR00T adapter LayerNorm needs a bias"))?;
    native::layer_norm(ctx, input, &norm.weight, bias, eps, out)
}

/// Qwen3-VL mRoPE positions as `(t, h, w)` per token.
///
/// Text tokens advance all three axes together. An image placeholder run takes
/// the grid's merged coordinates, so the whole image occupies a single span of
/// `max(t, h, w)` positions rather than one per patch -- that span is what
/// keeps the text after an image at the position the checkpoint was trained
/// with.
fn multimodal_rope_positions(
    image_token_id: u32,
    spatial_merge_size: usize,
    token_ids: &[u32],
    grid_thw: &[[u32; 3]],
) -> Result<Vec<u32>> {
    let merge = u32::try_from(spatial_merge_size)
        .map_err(|_| Error::Contract("Qwen3-VL spatial merge size overflow"))?;
    let mut positions = Vec::with_capacity(token_ids.len() * 3);
    let mut start = 0usize;
    let mut image_index = 0usize;
    let mut next = 0u32;
    let overflow = || Error::Contract("Qwen3-VL position overflow");
    loop {
        let found =
            (start..token_ids.len()).find(|index| token_ids[*index] == image_token_id);
        let Some(image_start) = found else {
            for _ in start..token_ids.len() {
                positions.extend_from_slice(&[next; 3]);
                next = next.checked_add(1).ok_or_else(overflow)?;
            }
            break;
        };
        for _ in start..image_start {
            positions.extend_from_slice(&[next; 3]);
            next = next.checked_add(1).ok_or_else(overflow)?;
        }
        let [temporal, height, width] = *grid_thw.get(image_index).ok_or_else(|| {
            Error::Other(format!("missing Qwen3-VL grid for image {image_index}"))
        })?;
        if merge == 0 || height % merge != 0 || width % merge != 0 {
            return Err(Error::Other(format!(
                "Qwen3-VL grid [{temporal}, {height}, {width}] is incompatible with merge {merge}"
            )));
        }
        image_index += 1;
        let (merged_h, merged_w) = (height / merge, width / merge);
        for time in 0..temporal {
            for row in 0..merged_h {
                for column in 0..merged_w {
                    positions.extend_from_slice(&[next + time, next + row, next + column]);
                }
            }
        }
        let image_tokens = temporal
            .checked_mul(merged_h)
            .and_then(|value| value.checked_mul(merged_w))
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| Error::Contract("Qwen3-VL image token count overflow"))?;
        let image_end = image_start
            .checked_add(image_tokens)
            .ok_or_else(|| Error::Contract("Qwen3-VL image token range overflow"))?;
        let placeholders_match = token_ids
            .get(image_start..image_end)
            .is_some_and(|run| run.iter().all(|token| *token == image_token_id));
        if !placeholders_match {
            return Err(Error::Other(format!(
                "Qwen3-VL image {} placeholder run does not match its grid [{temporal}, \
                 {height}, {width}]",
                image_index - 1
            )));
        }
        let span = temporal
            .saturating_sub(1)
            .max(merged_h.saturating_sub(1))
            .max(merged_w.saturating_sub(1))
            .checked_add(1)
            .ok_or_else(overflow)?;
        next = next.checked_add(span).ok_or_else(overflow)?;
        start = image_end;
    }
    if image_index != grid_thw.len() || positions.len() != token_ids.len() * 3 {
        return Err(Error::Other(format!(
            "Qwen3-VL token/grid mismatch: consumed {image_index}/{} grids and produced {}/{} \
             positions",
            grid_thw.len(),
            positions.len(),
            token_ids.len() * 3
        )));
    }
    Ok(positions)
}

#[cfg(test)]
mod tests {
    use super::*;

    const IMAGE: u32 = 7;
    const MERGE: usize = 2;

    fn positions(token_ids: &[u32], grid_thw: &[[u32; 3]]) -> Result<Vec<u32>> {
        multimodal_rope_positions(IMAGE, MERGE, token_ids, grid_thw)
    }

    #[test]
    fn text_tokens_advance_all_three_axes_together() {
        assert_eq!(
            positions(&[1, 2, 3], &[]).unwrap(),
            vec![0, 0, 0, 1, 1, 1, 2, 2, 2]
        );
    }

    #[test]
    fn an_image_run_takes_grid_coordinates_and_one_position_span() {
        // One 4x4 grid at merge 2 is 2x2 = 4 placeholders spanning 2
        // positions, so the text token after it sits at 2 and not at 4. That
        // span is what keeps post-image text where the checkpoint expects it.
        assert_eq!(
            positions(&[IMAGE, IMAGE, IMAGE, IMAGE, 9], &[[1, 4, 4]]).unwrap(),
            vec![
                0, 0, 0, // (t, h=0, w=0)
                0, 0, 1, // (t, h=0, w=1)
                0, 1, 0, // (t, h=1, w=0)
                0, 1, 1, // (t, h=1, w=1)
                2, 2, 2, // the text token after the image
            ]
        );
    }

    #[test]
    fn a_grid_that_disagrees_with_the_placeholder_run_is_refused() {
        // The grid asks for 4 placeholders but only 3 are present.
        let error = positions(&[IMAGE, IMAGE, IMAGE, 9], &[[1, 4, 4]])
            .unwrap_err()
            .to_string();
        assert!(error.contains("placeholder run"), "{error}");
        // An unconsumed grid is caught by the tail check rather than ignored.
        let error = positions(&[9, 9], &[[1, 4, 4]]).unwrap_err().to_string();
        assert!(error.contains("consumed 0/1"), "{error}");
        // A grid incompatible with the merge size is refused, not rounded.
        let error = positions(&[IMAGE; 4], &[[1, 3, 4]]).unwrap_err().to_string();
        assert!(error.contains("incompatible with merge"), "{error}");
    }

    #[test]
    fn the_fixture_shape_produces_one_position_per_token() {
        // 149 tokens with two 16x16 grids: 2 x 8 x 8 = 128 image placeholders
        // in one contiguous run, which is what the LIBERO fixture holds.
        let mut tokens = vec![9u32; 10];
        tokens.extend(std::iter::repeat_n(IMAGE, 128));
        tokens.extend([9u32; 11]);
        assert_eq!(tokens.len(), 149);
        let got = positions(&tokens, &[[1, 16, 16], [1, 16, 16]]).unwrap();
        assert_eq!(got.len(), 149 * 3);
        // The run starts at 10, each image spans 8 positions, and the second
        // image continues from where the first ended rather than restarting.
        assert_eq!(&got[10 * 3..10 * 3 + 3], &[10, 10, 10]);
        assert_eq!(&got[74 * 3..74 * 3 + 3], &[18, 18, 18]);
        assert_eq!(&got[138 * 3..138 * 3 + 3], &[26, 26, 26]);
    }
}
