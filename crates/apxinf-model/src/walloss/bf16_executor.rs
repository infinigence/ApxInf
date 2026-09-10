//! Native-BF16 action entry, projection, and solver update.

use apxinf_core::{Error, Result, Tensor};

use super::backend::{kernels, Context, DeviceBuffer};
use super::WallossActionWeights;
use super::{
    DeviceVisionGeometry, WallossDynamicFp8LayerWeights, WallossDynamicFp8VisionBlockWeights,
    WallossDynamicFp8VisionWeights, WallossLayerWeights, WallossTextConfig,
    WallossVisionBlockWeights, WallossVisionConfig, WallossVisionWeights,
};

#[derive(Clone, Copy)]
pub(super) enum MatrixRef<'a> {
    Bf16(&'a Tensor),
    DynamicFp8(&'a super::DynamicFp8LinearWeights),
}

pub(super) trait TransformerWeights {
    fn input_norm(&self) -> &Tensor;
    fn post_attention_norm(&self) -> &Tensor;
    fn qkv(&self) -> MatrixRef<'_>;
    fn qkv_bias(&self) -> &Tensor;
    fn output(&self) -> MatrixRef<'_>;
    fn gate_up(&self) -> MatrixRef<'_>;
    fn down(&self) -> MatrixRef<'_>;
}

pub(super) trait VisionBlockWeights {
    fn input_norm(&self) -> &Tensor;
    fn post_attention_norm(&self) -> &Tensor;
    fn qkv(&self) -> MatrixRef<'_>;
    fn qkv_bias(&self) -> &Tensor;
    fn output(&self) -> MatrixRef<'_>;
    fn output_bias(&self) -> &Tensor;
    fn gate_up(&self) -> MatrixRef<'_>;
    fn gate_up_bias(&self) -> &Tensor;
    fn down(&self) -> MatrixRef<'_>;
    fn down_bias(&self) -> &Tensor;
}

pub(super) trait VisionTowerWeights {
    type Block: VisionBlockWeights;
    fn patch_projection(&self) -> MatrixRef<'_>;
    fn blocks(&self) -> &[Self::Block];
    fn merger_norm(&self) -> &Tensor;
    fn merger_hidden(&self) -> MatrixRef<'_>;
    fn merger_hidden_bias(&self) -> &Tensor;
    fn merger_output(&self) -> MatrixRef<'_>;
    fn merger_output_bias(&self) -> &Tensor;
}

fn linear(context: &Context, input: &Tensor, weights: MatrixRef<'_>) -> Result<Tensor> {
    match weights {
        MatrixRef::Bf16(weight) => kernels::gemm::bf16(context, input, weight),
        MatrixRef::DynamicFp8(weight) => dynamic_linear(context, input, weight),
    }
}

fn dynamic_linear(
    context: &Context,
    input: &Tensor,
    weight: &super::DynamicFp8LinearWeights,
) -> Result<Tensor> {
    let input_shape = input.shape().dims();
    if input_shape.len() != 2 || input_shape[1] != weight.input_features {
        return Err(Error::Other(format!(
            "dynamic FP8 linear expects input width {}, got {:?}",
            weight.input_features, input_shape
        )));
    }
    let padded_input_features = weight.weight.shape().dims()[1];
    let activation = kernels::quantization::quantize_rows_bf16_e4m3_padded(
        context,
        input,
        padded_input_features,
    )?;
    dynamic_linear_prequantized(context, &activation, weight, false)
}

fn dynamic_linear_prequantized(
    context: &Context,
    activation: &kernels::quantization::DynamicFp8Tensor,
    weight: &super::DynamicFp8LinearWeights,
    keep_padded_output: bool,
) -> Result<Tensor> {
    let activation_shape = activation.values.shape().dims();
    let weight_shape = weight.weight.shape().dims();
    if activation_shape.len() != 2
        || weight_shape.len() != 2
        || activation_shape[1] != weight_shape[1]
    {
        return Err(Error::Other(format!(
            "prequantized dynamic FP8 linear shape mismatch: {:?} @ {:?}",
            activation_shape, weight_shape
        )));
    }
    let output = kernels::gemm::gemm_fp8_dynamic_bf16(
        context,
        &activation.values,
        &activation.scales,
        weight.as_kernel_view(),
        None,
    )?;
    if keep_padded_output {
        Ok(output)
    } else {
        kernels::quantization::slice_columns_bf16(context, &output, weight.output_features)
    }
}

fn qkv_after_rms(
    context: &Context,
    input: &Tensor,
    norm_weight: &Tensor,
    eps: f32,
    weights: MatrixRef<'_>,
) -> Result<Tensor> {
    match weights {
        MatrixRef::Bf16(weight) => {
            let normalized = kernels::norm::rms_bf16(context, input, norm_weight, eps)?;
            kernels::gemm::bf16(context, &normalized, weight)
        }
        MatrixRef::DynamicFp8(weight) => {
            let normalized = kernels::norm::rms_quantize_rows_bf16_e4m3(
                context,
                input,
                norm_weight,
                eps,
                weight.weight.shape().dims()[1],
            )?;
            dynamic_linear_prequantized(context, &normalized, weight, false)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn dynamic_residual_norm(
    context: &Context,
    projected: &Tensor,
    output_bias: Option<&Tensor>,
    input: &Tensor,
    post_attention_norm: &Tensor,
    eps: f32,
    output_cols: usize,
) -> Result<(Tensor, kernels::quantization::DynamicFp8Tensor)> {
    let fused = kernels::fused::bias_residual_rms_quantize_rows_bf16_e4m3(
        context,
        projected,
        output_bias,
        input,
        post_attention_norm,
        eps,
        output_cols,
    )?;
    Ok((fused.hidden, fused.normalized))
}

fn dynamic_swiglu(
    context: &Context,
    gate_up: &Tensor,
    gate_up_bias: Option<&Tensor>,
    logical_inner: usize,
    output_cols: usize,
) -> Result<kernels::quantization::DynamicFp8Tensor> {
    kernels::activation::swiglu_quantize_rows_bf16_e4m3(
        context,
        gate_up,
        gate_up_bias,
        logical_inner,
        output_cols,
    )
}

#[allow(clippy::too_many_arguments)]
fn residual_mlp(
    context: &Context,
    attention: &Tensor,
    input: &Tensor,
    output: MatrixRef<'_>,
    output_bias: Option<&Tensor>,
    post_attention_norm: &Tensor,
    gate_up: MatrixRef<'_>,
    gate_up_bias: Option<&Tensor>,
    down: MatrixRef<'_>,
    down_bias: Option<&Tensor>,
    eps: f32,
) -> Result<Tensor> {
    match (output, gate_up, down) {
        (MatrixRef::Bf16(output), MatrixRef::Bf16(gate_up), MatrixRef::Bf16(down)) => {
            let projected = kernels::gemm::bf16(context, attention, output)?;
            let fused = kernels::fused::bias_residual_rms_bf16(
                context,
                &projected,
                output_bias,
                input,
                post_attention_norm,
                eps,
            )?;
            let gate_up = kernels::gemm::bf16(context, &fused.normalized, gate_up)?;
            let gate_up = match gate_up_bias {
                Some(bias) => kernels::elementwise::bias_bf16(context, &gate_up, Some(bias))?,
                None => gate_up,
            };
            let activated = kernels::activation::swiglu_bf16(context, &gate_up)?;
            let down = kernels::gemm::bf16(context, &activated, down)?;
            kernels::fused::bias_residual_bf16(context, &down, down_bias, &fused.hidden)
        }
        (
            MatrixRef::DynamicFp8(output),
            MatrixRef::DynamicFp8(gate_up),
            MatrixRef::DynamicFp8(down),
        ) => {
            let projected = dynamic_linear(context, attention, output)?;
            let (hidden, normalized) = dynamic_residual_norm(
                context,
                &projected,
                output_bias,
                input,
                post_attention_norm,
                eps,
                gate_up.weight.shape().dims()[1],
            )?;
            let gate_up_output = dynamic_linear_prequantized(context, &normalized, gate_up, true)?;
            if gate_up.output_features % 2 != 0
                || down.input_features != gate_up.output_features / 2
            {
                return Err(Error::Other(format!(
                    "dynamic SwiGLU dimensions disagree: gate/up={}, down input={}",
                    gate_up.output_features, down.input_features
                )));
            }
            let activated = dynamic_swiglu(
                context,
                &gate_up_output,
                gate_up_bias,
                gate_up.output_features / 2,
                down.weight.shape().dims()[1],
            )?;
            let down = dynamic_linear_prequantized(context, &activated, down, false)?;
            kernels::fused::bias_residual_bf16(context, &down, down_bias, &hidden)
        }
        (MatrixRef::DynamicFp8(output), MatrixRef::Bf16(gate_up), MatrixRef::Bf16(down)) => {
            let projected = dynamic_linear(context, attention, output)?;
            let fused = kernels::fused::bias_residual_rms_bf16(
                context,
                &projected,
                output_bias,
                input,
                post_attention_norm,
                eps,
            )?;
            let gate_up = kernels::gemm::bf16(context, &fused.normalized, gate_up)?;
            let gate_up = match gate_up_bias {
                Some(bias) => kernels::elementwise::bias_bf16(context, &gate_up, Some(bias))?,
                None => gate_up,
            };
            let activated = kernels::activation::swiglu_bf16(context, &gate_up)?;
            let down = kernels::gemm::bf16(context, &activated, down)?;
            kernels::fused::bias_residual_bf16(context, &down, down_bias, &fused.hidden)
        }
        (MatrixRef::Bf16(output), MatrixRef::DynamicFp8(gate_up), MatrixRef::DynamicFp8(down)) => {
            let projected = kernels::gemm::bf16(context, attention, output)?;
            let (hidden, normalized) = dynamic_residual_norm(
                context,
                &projected,
                output_bias,
                input,
                post_attention_norm,
                eps,
                gate_up.weight.shape().dims()[1],
            )?;
            let gate_up_output = dynamic_linear_prequantized(context, &normalized, gate_up, true)?;
            if gate_up.output_features % 2 != 0
                || down.input_features != gate_up.output_features / 2
            {
                return Err(Error::Other(format!(
                    "dynamic SwiGLU dimensions disagree: gate/up={}, down input={}",
                    gate_up.output_features, down.input_features
                )));
            }
            let activated = dynamic_swiglu(
                context,
                &gate_up_output,
                gate_up_bias,
                gate_up.output_features / 2,
                down.weight.shape().dims()[1],
            )?;
            let down = dynamic_linear_prequantized(context, &activated, down, false)?;
            kernels::fused::bias_residual_bf16(context, &down, down_bias, &hidden)
        }
        _ => Err(Error::Other(
            "walloss transformer matrix precisions must be uniform within one layer".into(),
        )),
    }
}

impl TransformerWeights for WallossLayerWeights {
    fn input_norm(&self) -> &Tensor {
        &self.input_norm
    }
    fn post_attention_norm(&self) -> &Tensor {
        &self.post_attention_norm
    }
    fn qkv(&self) -> MatrixRef<'_> {
        MatrixRef::Bf16(&self.qkv)
    }
    fn qkv_bias(&self) -> &Tensor {
        &self.qkv_bias
    }
    fn output(&self) -> MatrixRef<'_> {
        MatrixRef::Bf16(&self.output)
    }
    fn gate_up(&self) -> MatrixRef<'_> {
        MatrixRef::Bf16(&self.gate_up)
    }
    fn down(&self) -> MatrixRef<'_> {
        MatrixRef::Bf16(&self.down)
    }
}

impl TransformerWeights for WallossDynamicFp8LayerWeights {
    fn input_norm(&self) -> &Tensor {
        &self.input_norm
    }
    fn post_attention_norm(&self) -> &Tensor {
        &self.post_attention_norm
    }
    fn qkv(&self) -> MatrixRef<'_> {
        MatrixRef::DynamicFp8(&self.qkv)
    }
    fn qkv_bias(&self) -> &Tensor {
        &self.qkv_bias
    }
    fn output(&self) -> MatrixRef<'_> {
        MatrixRef::DynamicFp8(&self.output)
    }
    fn gate_up(&self) -> MatrixRef<'_> {
        MatrixRef::DynamicFp8(&self.gate_up)
    }
    fn down(&self) -> MatrixRef<'_> {
        MatrixRef::DynamicFp8(&self.down)
    }
}

impl VisionBlockWeights for WallossVisionBlockWeights {
    fn input_norm(&self) -> &Tensor {
        &self.input_norm
    }
    fn post_attention_norm(&self) -> &Tensor {
        &self.post_attention_norm
    }
    fn qkv(&self) -> MatrixRef<'_> {
        MatrixRef::Bf16(&self.qkv)
    }
    fn qkv_bias(&self) -> &Tensor {
        &self.qkv_bias
    }
    fn output(&self) -> MatrixRef<'_> {
        MatrixRef::Bf16(&self.output)
    }
    fn output_bias(&self) -> &Tensor {
        &self.output_bias
    }
    fn gate_up(&self) -> MatrixRef<'_> {
        MatrixRef::Bf16(&self.gate_up)
    }
    fn gate_up_bias(&self) -> &Tensor {
        &self.gate_up_bias
    }
    fn down(&self) -> MatrixRef<'_> {
        MatrixRef::Bf16(&self.down)
    }
    fn down_bias(&self) -> &Tensor {
        &self.down_bias
    }
}

impl VisionBlockWeights for WallossDynamicFp8VisionBlockWeights {
    fn input_norm(&self) -> &Tensor {
        &self.input_norm
    }
    fn post_attention_norm(&self) -> &Tensor {
        &self.post_attention_norm
    }
    fn qkv(&self) -> MatrixRef<'_> {
        MatrixRef::DynamicFp8(&self.qkv)
    }
    fn qkv_bias(&self) -> &Tensor {
        &self.qkv_bias
    }
    fn output(&self) -> MatrixRef<'_> {
        MatrixRef::DynamicFp8(&self.output)
    }
    fn output_bias(&self) -> &Tensor {
        &self.output_bias
    }
    fn gate_up(&self) -> MatrixRef<'_> {
        MatrixRef::DynamicFp8(&self.gate_up)
    }
    fn gate_up_bias(&self) -> &Tensor {
        &self.gate_up_bias
    }
    fn down(&self) -> MatrixRef<'_> {
        MatrixRef::DynamicFp8(&self.down)
    }
    fn down_bias(&self) -> &Tensor {
        &self.down_bias
    }
}

impl VisionTowerWeights for WallossVisionWeights {
    type Block = WallossVisionBlockWeights;
    fn patch_projection(&self) -> MatrixRef<'_> {
        MatrixRef::Bf16(&self.patch_projection)
    }
    fn blocks(&self) -> &[Self::Block] {
        &self.blocks
    }
    fn merger_norm(&self) -> &Tensor {
        &self.merger_norm
    }
    fn merger_hidden(&self) -> MatrixRef<'_> {
        MatrixRef::Bf16(&self.merger_hidden)
    }
    fn merger_hidden_bias(&self) -> &Tensor {
        &self.merger_hidden_bias
    }
    fn merger_output(&self) -> MatrixRef<'_> {
        MatrixRef::Bf16(&self.merger_output)
    }
    fn merger_output_bias(&self) -> &Tensor {
        &self.merger_output_bias
    }
}

impl VisionTowerWeights for WallossDynamicFp8VisionWeights {
    type Block = WallossDynamicFp8VisionBlockWeights;
    fn patch_projection(&self) -> MatrixRef<'_> {
        MatrixRef::DynamicFp8(&self.patch_projection)
    }
    fn blocks(&self) -> &[Self::Block] {
        &self.blocks
    }
    fn merger_norm(&self) -> &Tensor {
        &self.merger_norm
    }
    fn merger_hidden(&self) -> MatrixRef<'_> {
        MatrixRef::DynamicFp8(&self.merger_hidden)
    }
    fn merger_hidden_bias(&self) -> &Tensor {
        &self.merger_hidden_bias
    }
    fn merger_output(&self) -> MatrixRef<'_> {
        MatrixRef::DynamicFp8(&self.merger_output)
    }
    fn merger_output_bias(&self) -> &Tensor {
        &self.merger_output_bias
    }
}

pub struct LanguageLayerOutput {
    pub hidden: Tensor,
    pub key: Tensor,
    pub value: Tensor,
}

pub struct ActionLayerOutput {
    pub hidden: Tensor,
}

pub struct PrefixCache {
    pub keys: Vec<Tensor>,
    pub values: Vec<Tensor>,
    pub prefix_tokens: usize,
    pub cache_tokens: usize,
}

pub(super) fn language_prefix<W: TransformerWeights>(
    context: &Context,
    config: &WallossTextConfig,
    weights: &[W],
    token_embedding: &Tensor,
    token_ids: &DeviceBuffer,
    vision_tokens: &Tensor,
    vision_row_map: &DeviceBuffer,
    position_ids: &DeviceBuffer,
    tokens: usize,
    cache_tokens: usize,
) -> Result<PrefixCache> {
    let embedded = kernels::embedding::lookup(context, token_embedding, token_ids, tokens)?;
    let mut hidden =
        kernels::elementwise::replace_rows_bf16(context, &embedded, vision_tokens, vision_row_map)?;
    let mut keys = Vec::with_capacity(weights.len());
    let mut values = Vec::with_capacity(weights.len());
    for layer in weights {
        let output = language_layer(context, config, layer, &hidden, position_ids, cache_tokens)?;
        hidden = output.hidden;
        keys.push(output.key);
        values.push(output.value);
    }
    Ok(PrefixCache {
        keys,
        values,
        prefix_tokens: tokens,
        cache_tokens,
    })
}

fn language_layer<W: TransformerWeights>(
    context: &Context,
    config: &WallossTextConfig,
    weights: &W,
    input: &Tensor,
    position_ids: &DeviceBuffer,
    cache_tokens: usize,
) -> Result<LanguageLayerOutput> {
    let tokens = input.shape().dims()[0];
    let qkv = qkv_after_rms(
        context,
        input,
        weights.input_norm(),
        config.rms_norm_eps,
        weights.qkv(),
    )?;
    let qkv = kernels::attention::split_gqa_qkv_mrope_cache_bf16(
        context,
        &qkv,
        Some(weights.qkv_bias()),
        position_ids,
        config.num_attention_heads,
        config.num_kv_heads,
        config.head_dim,
        config.rope_theta,
        config.mrope_section,
        cache_tokens,
        None,
    )?;
    let attention = kernels::attention::causal_gqa_bf16(context, &qkv.q, &qkv.k, &qkv.v, tokens)?
        .reshape(vec![tokens, config.num_attention_heads * config.head_dim])?;
    let hidden = residual_mlp(
        context,
        &attention,
        input,
        weights.output(),
        None,
        weights.post_attention_norm(),
        weights.gate_up(),
        None,
        weights.down(),
        None,
        config.rms_norm_eps,
    )?;
    Ok(LanguageLayerOutput {
        hidden,
        key: qkv.k,
        value: qkv.v,
    })
}

#[allow(clippy::too_many_arguments)]
fn action_layer<W: TransformerWeights>(
    context: &Context,
    config: &WallossTextConfig,
    weights: &W,
    input: &Tensor,
    prefix_key: &Tensor,
    prefix_value: &Tensor,
    prefix_tokens: usize,
    cache_tokens: usize,
    action_position_ids: &DeviceBuffer,
) -> Result<ActionLayerOutput> {
    let action_tokens = input.shape().dims()[0];
    let qkv = qkv_after_rms(
        context,
        input,
        weights.input_norm(),
        config.rms_norm_eps,
        weights.qkv(),
    )?;
    let qkv = kernels::attention::split_gqa_qkv_mrope_cache_bf16(
        context,
        &qkv,
        Some(weights.qkv_bias()),
        action_position_ids,
        config.num_attention_heads,
        config.num_kv_heads,
        config.head_dim,
        config.rope_theta,
        config.mrope_section,
        cache_tokens,
        Some((prefix_key, prefix_value, prefix_tokens)),
    )?;
    let attention =
        kernels::attention::causal_gqa_bf16(context, &qkv.q, &qkv.k, &qkv.v, cache_tokens)?
            .reshape(vec![
                action_tokens,
                config.num_attention_heads * config.head_dim,
            ])?;
    let hidden = residual_mlp(
        context,
        &attention,
        input,
        weights.output(),
        None,
        weights.post_attention_norm(),
        weights.gate_up(),
        None,
        weights.down(),
        None,
        config.rms_norm_eps,
    )?;
    Ok(ActionLayerOutput { hidden })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn action_stack<W: TransformerWeights>(
    context: &Context,
    config: &WallossTextConfig,
    transformer_weights: &[W],
    action_weights: &WallossActionWeights,
    final_norm: &Tensor,
    prefix: &PrefixCache,
    noisy_action: &Tensor,
    dof_mask: &Tensor,
    time_embedding: &Tensor,
    action_position_ids: &DeviceBuffer,
) -> Result<Tensor> {
    if transformer_weights.len() != prefix.keys.len()
        || transformer_weights.len() != prefix.values.len()
    {
        return Err(Error::Other(
            "walloss action stack and prefix cache depth differ".into(),
        ));
    }
    let mut hidden = action_embedding(
        context,
        action_weights,
        noisy_action,
        dof_mask,
        time_embedding,
    )?;
    for (index, layer) in transformer_weights.iter().enumerate() {
        hidden = action_layer(
            context,
            config,
            layer,
            &hidden,
            &prefix.keys[index],
            &prefix.values[index],
            prefix.prefix_tokens,
            prefix.cache_tokens,
            action_position_ids,
        )?
        .hidden;
    }
    let hidden = kernels::norm::rms_bf16(context, &hidden, final_norm, config.rms_norm_eps)?;
    velocity(context, action_weights, &hidden)
}

#[allow(clippy::too_many_arguments)]
fn vision_layer<W: VisionBlockWeights>(
    context: &Context,
    config: &WallossVisionConfig,
    weights: &W,
    input: &Tensor,
    position_ids: &DeviceBuffer,
    attention_offsets: &DeviceBuffer,
    host_attention_offsets: &[u32],
    segments: usize,
    max_segment_tokens: usize,
) -> Result<Tensor> {
    let tokens = input.shape().dims()[0];
    let head_dim = config.hidden_size / config.num_heads;
    let qkv = qkv_after_rms(
        context,
        input,
        weights.input_norm(),
        config.rms_norm_eps,
        weights.qkv(),
    )?;
    let qkv = kernels::attention::split_vision_qkv_rope_bf16(
        context,
        &qkv,
        Some(weights.qkv_bias()),
        position_ids,
        config.num_heads,
        head_dim,
        config.rope_theta,
    )?;
    let attention = kernels::attention::segmented_mha_bf16(
        context,
        &qkv.q,
        &qkv.k,
        &qkv.v,
        attention_offsets,
        host_attention_offsets,
        segments,
        max_segment_tokens,
    )?
    .reshape(vec![tokens, config.hidden_size])?;
    residual_mlp(
        context,
        &attention,
        input,
        weights.output(),
        Some(weights.output_bias()),
        weights.post_attention_norm(),
        weights.gate_up(),
        Some(weights.gate_up_bias()),
        weights.down(),
        Some(weights.down_bias()),
        config.rms_norm_eps,
    )
}

pub(super) fn vision_tower<W: VisionTowerWeights>(
    context: &Context,
    config: &WallossVisionConfig,
    weights: &W,
    geometry: &DeviceVisionGeometry,
    patches: &Tensor,
) -> Result<Tensor> {
    let ordered_patches = kernels::elementwise::gather_rows_bf16_device_indices(
        context,
        patches,
        &geometry.patch_order,
        geometry.patch_order.len() / std::mem::size_of::<u32>(),
    )?;
    let mut hidden = linear(context, &ordered_patches, weights.patch_projection())?;
    for (layer_index, block) in weights.blocks().iter().enumerate() {
        let full_attention = config.full_attention_blocks.contains(&layer_index);
        let (offsets, host_offsets, segments, max_tokens) = if full_attention {
            (
                &geometry.full_offsets,
                geometry.host_full_offsets.as_slice(),
                geometry.full_segments,
                geometry.max_full_tokens,
            )
        } else {
            (
                &geometry.window_offsets,
                geometry.host_window_offsets.as_slice(),
                geometry.window_segments,
                geometry.max_window_tokens,
            )
        };
        hidden = vision_layer(
            context,
            config,
            block,
            &hidden,
            &geometry.position_ids,
            offsets,
            host_offsets,
            segments,
            max_tokens,
        )?;
    }
    vision_merger(context, config, weights, &hidden, &geometry.reverse_indices)
}

fn vision_merger<W: VisionTowerWeights>(
    context: &Context,
    config: &WallossVisionConfig,
    weights: &W,
    input: &Tensor,
    reverse_indices: &DeviceBuffer,
) -> Result<Tensor> {
    let merge_unit = config.spatial_merge_size * config.spatial_merge_size;
    let tokens = input.shape().dims()[0];
    if tokens % merge_unit != 0 {
        return Err(Error::Other(format!(
            "walloss vision tokens {tokens} are not divisible by merge unit {merge_unit}"
        )));
    }
    let merged_tokens = tokens / merge_unit;
    let normalized =
        kernels::norm::rms_bf16(context, input, weights.merger_norm(), config.rms_norm_eps)?
            .reshape(vec![merged_tokens, merge_unit * config.hidden_size])?;
    let hidden = linear(context, &normalized, weights.merger_hidden())?;
    let hidden =
        kernels::activation::bias_gelu_bf16(context, &hidden, Some(weights.merger_hidden_bias()))?;
    let output = linear(context, &hidden, weights.merger_output())?;
    let output =
        kernels::elementwise::bias_bf16(context, &output, Some(weights.merger_output_bias()))?;
    kernels::elementwise::gather_rows_bf16_device_indices(
        context,
        &output,
        reverse_indices,
        merged_tokens,
    )
}

pub fn action_embedding(
    context: &Context,
    weights: &WallossActionWeights,
    noisy_action: &Tensor,
    dof_mask: &Tensor,
    time_embedding: &Tensor,
) -> Result<Tensor> {
    if noisy_action.shape() != dof_mask.shape() {
        return Err(Error::Other(format!(
            "walloss action and degree-of-freedom mask shapes differ: {:?} vs {:?}",
            noisy_action.shape().dims(),
            dof_mask.shape().dims()
        )));
    }
    let action = kernels::gemm::bf16(context, noisy_action, &weights.noisy_action_projection)?;
    let mask = kernels::gemm::bf16(context, dof_mask, &weights.dof_projection)?;
    let action = kernels::elementwise::add(context, &action, &mask)?;

    let action_part = kernels::gemm::bf16(context, &action, &weights.action_projection)?;
    let time_part = kernels::gemm::bf16(context, time_embedding, &weights.time_projection)?;
    let fused = kernels::elementwise::add(context, &action_part, &time_part)?;
    let activated = kernels::activation::silu(context, &fused)?;
    kernels::gemm::bf16(context, &activated, &weights.action_embedding_projection)
}

pub fn velocity(
    context: &Context,
    weights: &WallossActionWeights,
    action_hidden: &Tensor,
) -> Result<Tensor> {
    kernels::gemm::bf16(context, action_hidden, &weights.velocity_projection)
}

pub fn solver_update(
    context: &Context,
    state: &Tensor,
    velocity: &Tensor,
    dt: f32,
) -> Result<Tensor> {
    kernels::elementwise::euler_update_bf16(context, state, velocity, dt)
}
