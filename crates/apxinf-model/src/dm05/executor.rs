//! Correctness-first native BF16 DM05 layer composition.

use apxinf_core::{DType, Error, Result, Shape, Tensor};

use super::backend::{kernels, Context, DeviceBuffer};
use super::{
    AttentionSegment, DeviceActionLayer, DeviceDm05Weights, DeviceLanguageLayer, DeviceLinear,
    DeviceMlp, DeviceVisionBlock, Dm05Config, Dm05RopeConfig,
};
use kernels::attention::AttentionMask;

pub struct PrefixLayerOutput {
    pub hidden: Tensor,
    pub key: Tensor,
    pub value: Tensor,
}

pub struct ActionStyles<'a> {
    pub attention: &'a Tensor,
    pub mlp: &'a Tensor,
}

fn linear(ctx: &Context, input: &Tensor, weights: &DeviceLinear) -> Result<Tensor> {
    let output = kernels::gemm::bf16(ctx, input, &weights.weight)?;
    kernels::elementwise::bias_bf16(ctx, &output, weights.bias.as_ref())
}

fn gemma_rms(ctx: &Context, input: &Tensor, raw_weight: &Tensor, eps: f32) -> Result<Tensor> {
    kernels::norm::rms_affine_bf16(ctx, input, raw_weight, eps, 1.0)
}

fn qk_norm_rope(
    ctx: &Context,
    input: &Tensor,
    raw_weight: &Tensor,
    cosine: &Tensor,
    sine: &Tensor,
    heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<Tensor> {
    let tokens = input.shape().dims()[0];
    let matrix = input.reshape(vec![tokens * heads, head_dim])?;
    let normalized =
        gemma_rms(ctx, &matrix, raw_weight, eps)?.reshape(vec![tokens, heads, head_dim])?;
    kernels::rope::apply_precomputed_bf16(ctx, &normalized, cosine, sine)
}

fn mlp(ctx: &Context, input: &Tensor, weights: &DeviceMlp) -> Result<Tensor> {
    let gate = linear(ctx, input, &weights.gate)?;
    let gate = kernels::activation::bias_gelu_bf16(ctx, &gate, None)?;
    let up = linear(ctx, input, &weights.up)?;
    let activated = kernels::elementwise::mul(ctx, &gate, &up)?;
    linear(ctx, &activated, &weights.down)
}

pub(crate) fn row_view(tensor: &Tensor, start: usize, end: usize) -> Result<Tensor> {
    let dims = tensor.shape().dims();
    if dims.len() < 2 || start >= end || end > dims[0] {
        return Err(Error::Other(format!(
            "DM05 row view [{start}..{end}] is invalid for {dims:?}"
        )));
    }
    let row_elements = dims[1..]
        .iter()
        .try_fold(1usize, |value, dimension| value.checked_mul(*dimension))
        .ok_or_else(|| Error::Other("DM05 row-view element count overflow".into()))?;
    let row_bytes = row_elements
        .checked_mul(tensor.dtype().size_in_bytes())
        .ok_or_else(|| Error::Other("DM05 row-view byte count overflow".into()))?;
    let buffer = DeviceBuffer::from_tensor(tensor).map_err(Error::Cuda)?;
    let view = buffer
        .view(start * row_bytes, (end - start) * row_bytes)
        .map_err(Error::Cuda)?;
    let mut shape = dims.to_vec();
    shape[0] = end - start;
    view.as_tensor(Shape::new(shape), tensor.dtype())
        .map_err(Error::Cuda)
}

fn concatenate(ctx: &Context, values: &[Tensor]) -> Result<Tensor> {
    let mut values = values.iter();
    let first = values
        .next()
        .ok_or_else(|| Error::Other("DM05 cannot concatenate an empty tensor list".into()))?;
    values.try_fold(first.clone(), |current, value| {
        let current_dims = current.shape().dims();
        let value_dims = value.shape().dims();
        if current_dims.len() != value_dims.len() || current_dims[1..] != value_dims[1..] {
            return Err(Error::Other(format!(
                "DM05 concat shape mismatch: {current_dims:?} versus {value_dims:?}"
            )));
        }
        let cols: usize = current_dims[1..].iter().product();
        let current = current.reshape(vec![current_dims[0], cols])?;
        let value = value.reshape(vec![value_dims[0], cols])?;
        let output = kernels::elementwise::concat_rows_bf16(ctx, &current, &value)?;
        let mut shape = current_dims.to_vec();
        shape[0] += value_dims[0];
        output.reshape(shape)
    })
}

pub fn vision_layer(
    ctx: &Context,
    weights: &DeviceVisionBlock,
    input: &Tensor,
    tokens_per_view: usize,
    heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<Tensor> {
    let normalized =
        kernels::norm::layer_bf16(ctx, input, &weights.norm1.weight, &weights.norm1.bias, eps)?;
    let q = linear(ctx, &normalized, &weights.q)?.reshape(vec![
        input.shape().dims()[0],
        heads,
        head_dim,
    ])?;
    let k = linear(ctx, &normalized, &weights.k)?.reshape(vec![
        input.shape().dims()[0],
        heads,
        head_dim,
    ])?;
    let v = linear(ctx, &normalized, &weights.v)?.reshape(vec![
        input.shape().dims()[0],
        heads,
        head_dim,
    ])?;
    let attention = kernels::attention::mha_bf16(ctx, &q, &k, &v, tokens_per_view)?
        .reshape(vec![input.shape().dims()[0], heads * head_dim])?;
    let projected = linear(ctx, &attention, &weights.output)?;
    let hidden = kernels::elementwise::add(ctx, input, &projected)?;
    let normalized = kernels::norm::layer_bf16(
        ctx,
        &hidden,
        &weights.norm2.weight,
        &weights.norm2.bias,
        eps,
    )?;
    let feedforward = linear(ctx, &normalized, &weights.fc1)?;
    let feedforward = kernels::activation::bias_gelu_bf16(ctx, &feedforward, None)?;
    let feedforward = linear(ctx, &feedforward, &weights.fc2)?;
    kernels::elementwise::add(ctx, &hidden, &feedforward)
}

pub fn encode_vision(
    ctx: &Context,
    config: &Dm05Config,
    weights: &DeviceDm05Weights,
    pool_matrix: &Tensor,
    patches: &Tensor,
) -> Result<Tensor> {
    if patches.dtype() != DType::BF16
        || patches.shape().dims()
            != [
                config.num_views * config.patches_per_view(),
                3 * config.vision.patch_size * config.vision.patch_size,
            ]
    {
        return Err(Error::Other(format!(
            "DM05 patches must be BF16 [{},{}], got {} {:?}",
            config.num_views * config.patches_per_view(),
            3 * config.vision.patch_size * config.vision.patch_size,
            patches.dtype(),
            patches.shape().dims()
        )));
    }
    let projected = linear(ctx, patches, &weights.patch_embedding)?;
    let mut hidden = kernels::embedding::add_position_bf16(
        ctx,
        &projected,
        None,
        &weights.position_embedding,
        config.patches_per_view(),
    )?;
    for layer in &weights.vision_layers {
        hidden = vision_layer(
            ctx,
            layer,
            &hidden,
            config.patches_per_view(),
            config.vision.num_heads,
            config.vision.head_dim,
            config.vision.layer_norm_eps,
        )?;
    }
    hidden = kernels::norm::layer_bf16(
        ctx,
        &hidden,
        &weights.vision_post_norm.weight,
        &weights.vision_post_norm.bias,
        config.vision.layer_norm_eps,
    )?;
    let pooled = kernels::gemm::bf16(ctx, pool_matrix, &hidden)?;
    let pooled = gemma_rms(
        ctx,
        &pooled,
        &weights.projector_norm.raw_weight,
        config.vision.layer_norm_eps,
    )?;
    kernels::gemm::bf16(ctx, &pooled, &weights.projector)
}

pub fn merge_prefix(
    ctx: &Context,
    config: &Dm05Config,
    weights: &DeviceDm05Weights,
    token_ids: &DeviceBuffer,
    host_token_ids: &[u32],
    image_tokens: &Tensor,
) -> Result<Tensor> {
    let [(first_start, first_end), (second_start, second_end)] =
        config.validate_prefix_tokens(host_token_ids)?;
    let scale = half::bf16::from_f32((config.language.width as f32).sqrt()).to_f32();
    let language = kernels::embedding::lookup_scaled_bf16(
        ctx,
        &weights.token_embedding,
        token_ids,
        host_token_ids.len(),
        scale,
    )?;
    if image_tokens.shape().dims() != [config.image_tokens(), config.language.width] {
        return Err(Error::Other(format!(
            "DM05 projected image tokens must be [{},{}], got {:?}",
            config.image_tokens(),
            config.language.width,
            image_tokens.shape().dims()
        )));
    }
    let first_image = row_view(image_tokens, 0, config.vision.pooled_tokens_per_image)?;
    let second_image = row_view(
        image_tokens,
        config.vision.pooled_tokens_per_image,
        config.image_tokens(),
    )?;
    concatenate(
        ctx,
        &[
            row_view(&language, 0, first_start)?,
            first_image,
            row_view(&language, first_end, second_start)?,
            second_image,
            row_view(&language, second_end, host_token_ids.len())?,
        ],
    )
}

fn segmented_prefix_attention(
    ctx: &Context,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    segments: &[AttentionSegment],
) -> Result<Tensor> {
    let outputs = segments
        .iter()
        .map(|segment| {
            let query = row_view(q, segment.query_start, segment.query_end)?;
            let key = row_view(k, 0, segment.key_end)?;
            let value = row_view(v, 0, segment.key_end)?;
            kernels::attention::gqa_bf16(
                ctx,
                &query,
                &key,
                &value,
                if segment.causal {
                    AttentionMask::Causal
                } else {
                    AttentionMask::None
                },
            )
        })
        .collect::<Result<Vec<_>>>()?;
    concatenate(ctx, &outputs)
}

#[allow(clippy::too_many_arguments)]
pub fn language_layer(
    ctx: &Context,
    config: &Dm05Config,
    weights: &DeviceLanguageLayer,
    input: &Tensor,
    segments: &[AttentionSegment],
    cosine: &Tensor,
    sine: &Tensor,
    compute_tail: bool,
) -> Result<PrefixLayerOutput> {
    let text = &config.language;
    let tokens = input.shape().dims()[0];
    let normalized = gemma_rms(
        ctx,
        input,
        &weights.input_norm.raw_weight,
        text.rms_norm_eps,
    )?;
    let q = linear(ctx, &normalized, &weights.attention.q)?.reshape(vec![
        tokens,
        text.num_heads,
        text.head_dim,
    ])?;
    let k = linear(ctx, &normalized, &weights.attention.k)?.reshape(vec![
        tokens,
        text.num_kv_heads,
        text.head_dim,
    ])?;
    let value = linear(ctx, &normalized, &weights.attention.v)?.reshape(vec![
        tokens,
        text.num_kv_heads,
        text.head_dim,
    ])?;
    let query = qk_norm_rope(
        ctx,
        &q,
        &weights.attention.q_norm.raw_weight,
        cosine,
        sine,
        text.num_heads,
        text.head_dim,
        text.rms_norm_eps,
    )?;
    let key = qk_norm_rope(
        ctx,
        &k,
        &weights.attention.k_norm.raw_weight,
        cosine,
        sine,
        text.num_kv_heads,
        text.head_dim,
        text.rms_norm_eps,
    )?;
    if !compute_tail {
        return Ok(PrefixLayerOutput {
            hidden: input.clone(),
            key,
            value,
        });
    }
    let attention = segmented_prefix_attention(ctx, &query, &key, &value, segments)?
        .reshape(vec![tokens, text.num_heads * text.head_dim])?;
    let projected = linear(ctx, &attention, &weights.attention.output)?;
    let projected = gemma_rms(
        ctx,
        &projected,
        &weights.post_attention_norm.raw_weight,
        text.rms_norm_eps,
    )?;
    let hidden = kernels::elementwise::add(ctx, input, &projected)?;
    let normalized = gemma_rms(
        ctx,
        &hidden,
        &weights.pre_feedforward_norm.raw_weight,
        text.rms_norm_eps,
    )?;
    let feedforward = mlp(ctx, &normalized, &weights.mlp)?;
    let feedforward = gemma_rms(
        ctx,
        &feedforward,
        &weights.post_feedforward_norm.raw_weight,
        text.rms_norm_eps,
    )?;
    let hidden = kernels::elementwise::add(ctx, &hidden, &feedforward)?;
    Ok(PrefixLayerOutput { hidden, key, value })
}

#[allow(clippy::too_many_arguments)]
pub fn action_layer(
    ctx: &Context,
    config: &Dm05Config,
    weights: &DeviceActionLayer,
    input: &Tensor,
    styles: ActionStyles<'_>,
    prefix_key: &Tensor,
    prefix_value: &Tensor,
    cosine: &Tensor,
    sine: &Tensor,
) -> Result<Tensor> {
    let action = &config.action_expert;
    let tokens = input.shape().dims()[0];
    let normalized =
        kernels::norm::adaptive_rms_bf16(ctx, input, styles.attention, action.rms_norm_eps)?;
    let q = linear(ctx, &normalized, &weights.attention.q)?.reshape(vec![
        tokens,
        action.num_heads,
        action.head_dim,
    ])?;
    let k = linear(ctx, &normalized, &weights.attention.k)?.reshape(vec![
        tokens,
        action.num_kv_heads,
        action.head_dim,
    ])?;
    let v = linear(ctx, &normalized, &weights.attention.v)?.reshape(vec![
        tokens,
        action.num_kv_heads,
        action.head_dim,
    ])?;
    let query = qk_norm_rope(
        ctx,
        &q,
        &weights.attention.q_norm.raw_weight,
        cosine,
        sine,
        action.num_heads,
        action.head_dim,
        action.rms_norm_eps,
    )?;
    let key = qk_norm_rope(
        ctx,
        &k,
        &weights.attention.k_norm.raw_weight,
        cosine,
        sine,
        action.num_kv_heads,
        action.head_dim,
        action.rms_norm_eps,
    )?;
    let prefix_tokens = prefix_key.shape().dims()[0];
    let combined_key = concatenate(ctx, &[prefix_key.clone(), key])?;
    let combined_value = concatenate(ctx, &[prefix_value.clone(), v])?;
    let attention = kernels::attention::gqa_bf16(
        ctx,
        &query,
        &combined_key,
        &combined_value,
        AttentionMask::None,
    )?
    .reshape(vec![tokens, action.num_heads * action.head_dim])?;
    if combined_key.shape().dims()[0] != prefix_tokens + tokens {
        return Err(Error::Other("DM05 suffix K/V concatenation drifted".into()));
    }
    let projected = linear(ctx, &attention, &weights.attention.output)?;
    let projected = gemma_rms(
        ctx,
        &projected,
        &weights.post_attention_norm.raw_weight,
        action.rms_norm_eps,
    )?;
    let hidden = kernels::elementwise::mul_style_gate_bf16(ctx, &projected, styles.attention)?;
    let hidden = kernels::elementwise::add(ctx, input, &hidden)?;
    let normalized =
        kernels::norm::adaptive_rms_bf16(ctx, &hidden, styles.mlp, action.rms_norm_eps)?;
    let feedforward = mlp(ctx, &normalized, &weights.mlp)?;
    let feedforward = gemma_rms(
        ctx,
        &feedforward,
        &weights.post_feedforward_norm.raw_weight,
        action.rms_norm_eps,
    )?;
    let feedforward = kernels::elementwise::mul_style_gate_bf16(ctx, &feedforward, styles.mlp)?;
    let hidden = kernels::elementwise::add(ctx, &hidden, &feedforward)?;
    Ok(hidden)
}

pub fn modulation_style(
    ctx: &Context,
    conditioning: &Tensor,
    weights: &DeviceLinear,
) -> Result<Tensor> {
    linear(ctx, conditioning, weights)?.reshape(vec![weights.weight.shape().dims()[1]])
}

pub fn action_projection(ctx: &Context, input: &Tensor, weights: &DeviceLinear) -> Result<Tensor> {
    linear(ctx, input, weights)
}

pub fn rope_kind_for_layer(config: &Dm05Config, layer: usize) -> Result<Dm05RopeConfig> {
    config.language.rope_for_layer(layer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_view_rejects_cpu_tensors_before_touching_cuda_storage() {
        let tensor = Tensor::zeros((4, 3), DType::BF16);
        assert!(row_view(&tensor, 1, 3).is_err());
        assert!(row_view(&tensor, 2, 2).is_err());
    }
}
