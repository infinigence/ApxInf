//! Native-BF16 action entry, projection, and solver update.

use apxinf_core::{DType, Error, Result, Shape, Tensor};

use crate::walloss::backend::{ops, Context, DeviceBuffer};
use crate::walloss::geometry::DeviceVisionGeometry;
use crate::walloss::weights::{
    DynamicFp8LinearWeights, WallossActionWeights, WallossDynamicFp8LayerWeights,
    WallossDynamicFp8VisionBlockWeights, WallossDynamicFp8VisionWeights, WallossLayerWeights,
    WallossVisionBlockWeights, WallossVisionWeights,
};
use crate::walloss::{WallossTextConfig, WallossVisionConfig};

#[derive(Clone, Copy)]
pub(super) enum MatrixRef<'a> {
    Bf16(&'a Tensor),
    DynamicFp8(&'a DynamicFp8LinearWeights),
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

fn plain_gemm(context: &Context, input: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let input_shape = input.shape().dims();
    let weight_shape = weight.shape().dims();
    if input_shape.len() != 2 || weight_shape.len() != 2 || input_shape[1] != weight_shape[0] {
        return Err(Error::Other(format!(
            "WallOSS GEMM shape mismatch: {input_shape:?} @ {weight_shape:?}"
        )));
    }
    let mut output = context.allocate_output(
        Shape::new(vec![input_shape[0], weight_shape[1]]),
        DType::BF16,
    )?;
    ops::gemm::gemm(
        context,
        ops::gemm::GemmArgs::new(input, weight, &mut output),
    )?;
    Ok(output)
}

fn add_bias(context: &Context, input: &Tensor, bias: &Tensor) -> Result<Tensor> {
    bias_activation(context, input, Some(bias), ops::PointwiseActivation::None)
}

fn bias_activation(
    context: &Context,
    input: &Tensor,
    bias: Option<&Tensor>,
    activation: ops::PointwiseActivation,
) -> Result<Tensor> {
    let mut output = context.allocate_output(input.shape().clone(), input.dtype())?;
    let mut args =
        ops::PointwiseArgs::new(ops::PointwiseSemantic::BiasActivation, input, &mut output);
    args.bias = bias;
    args.activation = activation;
    ops::pointwise(context, args)?;
    Ok(output)
}

fn activation(
    context: &Context,
    input: &Tensor,
    activation: ops::ElementwiseActivation,
) -> Result<Tensor> {
    let output = context.allocate_output(input.shape().clone(), input.dtype())?;
    ops::elementwise_activation(context, input, &output, activation)?;
    Ok(output)
}

fn euler_update(context: &Context, state: &Tensor, velocity: &Tensor, dt: f32) -> Result<Tensor> {
    let mut output = context.allocate_output(state.shape().clone(), state.dtype())?;
    let mut args = ops::PointwiseArgs::new(ops::PointwiseSemantic::EulerUpdate, state, &mut output);
    args.secondary = Some(velocity);
    args.dt = dt;
    ops::pointwise(context, args)?;
    Ok(output)
}

fn embedding_lookup(
    context: &Context,
    table: &Tensor,
    ids: &DeviceBuffer,
    tokens: usize,
) -> Result<Tensor> {
    let dims = table.shape().dims();
    if dims.len() != 2 {
        return Err(Error::Other(
            "WallOSS embedding table must be rank 2".into(),
        ));
    }
    let mut output = context.allocate_output(Shape::new(vec![tokens, dims[1]]), table.dtype())?;
    let mut args = ops::GatherArgs::new(ops::GatherSemantic::EmbeddingLookup, table, &mut output);
    args.ids = Some(ids);
    args.vocab_size = dims[0];
    args.embedding_scale = ops::EmbeddingScale::None;
    ops::gather(context, args)?;
    Ok(output)
}

fn linear(context: &Context, input: &Tensor, weights: MatrixRef<'_>) -> Result<Tensor> {
    match weights {
        MatrixRef::Bf16(weight) => plain_gemm(context, input, weight),
        MatrixRef::DynamicFp8(weight) => dynamic_linear(context, input, weight),
    }
}

fn causal_gqa(
    context: &Context,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    valid_key_tokens: usize,
) -> Result<Tensor> {
    let q = query.shape().dims();
    let k = key.shape().dims();
    if q.len() != 3
        || k.len() != 3
        || value.shape() != key.shape()
        || q.contains(&0)
        || k.contains(&0)
        || q[2] != k[2]
        || q[1] % k[1] != 0
        || valid_key_tokens < q[0]
        || valid_key_tokens > k[0]
    {
        return Err(Error::Other(
            "WALL-OSS causal GQA shape or key range mismatch".into(),
        ));
    }
    let query = query.reshape(vec![1, q[0], q[1], q[2]])?;
    let key = key.reshape(vec![1, k[0], k[1], k[2]])?;
    let value = value.reshape(vec![1, k[0], k[1], k[2]])?;
    let mut output = context.allocate_output(query.shape().clone(), query.dtype())?;
    let mut args = ops::KvCacheAttentionArgs::new(&query, &key, &value, &mut output);
    args.valid_key_tokens = valid_key_tokens;
    args.query_start = valid_key_tokens - q[0];
    args.numerics = ops::AttentionNumerics::Flexible;
    ops::kv_cache_attention(context, args)?;
    output.reshape(q.to_vec())
}

#[allow(clippy::too_many_arguments)]
fn segmented_mha(
    context: &Context,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    offsets: &DeviceBuffer,
    host_offsets: &[u32],
    segments: usize,
    max_segment_tokens: usize,
) -> Result<Tensor> {
    if segments == 0
        || host_offsets.len() != segments + 1
        || max_segment_tokens == 0
        || host_offsets.windows(2).any(|bounds| {
            bounds[1] < bounds[0] || (bounds[1] - bounds[0]) as usize > max_segment_tokens
        })
    {
        return Err(Error::Other(
            "WALL-OSS segmented attention metadata mismatch".into(),
        ));
    }
    let mut output = context.allocate_output(query.shape().clone(), query.dtype())?;
    let mut args =
        ops::SegmentedAttentionArgs::new(query, key, value, &mut output, offsets, host_offsets);
    args.numerics = ops::AttentionNumerics::Flexible;
    ops::segmented_attention(context, args)?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn split_gqa_qkv_mrope_cache(
    context: &Context,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    position_ids: &DeviceBuffer,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    theta: f32,
    sections: [usize; 3],
    cache_tokens: usize,
    caches: Option<(&Tensor, &Tensor, usize)>,
) -> Result<(Tensor, Tensor, Tensor)> {
    let dims = qkv.shape().dims();
    let q_width = q_heads
        .checked_mul(head_dim)
        .ok_or_else(|| Error::Other("WALL-OSS QKV width overflow".into()))?;
    let kv_width = kv_heads
        .checked_mul(head_dim)
        .ok_or_else(|| Error::Other("WALL-OSS KV width overflow".into()))?;
    if dims.len() != 2
        || dims[0] == 0
        || dims[1] != q_width + 2 * kv_width
        || q_heads == 0
        || kv_heads == 0
        || q_heads % kv_heads != 0
        || head_dim == 0
    {
        return Err(Error::Other("WALL-OSS GQA QKV mRoPE shape mismatch".into()));
    }
    let tokens = dims[0];
    let (mut key, mut value, cache_offset) = match caches {
        Some((key, value, offset)) => (key.clone(), value.clone(), offset),
        None => {
            let shape = Shape::new(vec![cache_tokens, kv_heads, head_dim]);
            (
                context.allocate_output(shape.clone(), DType::BF16)?,
                context.allocate_output(shape, DType::BF16)?,
                0,
            )
        }
    };
    let mut query =
        context.allocate_output(Shape::new(vec![tokens, q_heads, head_dim]), DType::BF16)?;
    let positions = position_ids
        .as_tensor(Shape::new(vec![tokens, 3]), DType::I32)
        .map_err(Error::Cuda)?;
    ops::split_qkv_mrope_cache(
        context,
        ops::MropeCacheArgs {
            qkv,
            bias,
            positions: &positions,
            query: &mut query,
            key_cache: &mut key,
            value_cache: &mut value,
            theta,
            sections,
            cache_offset,
        },
    )?;
    Ok((query, key, value))
}

fn split_vision_qkv_rope(
    context: &Context,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    position_ids: &DeviceBuffer,
    heads: usize,
    head_dim: usize,
    theta: f32,
) -> Result<(Tensor, Tensor, Tensor)> {
    let dims = qkv.shape().dims();
    if dims.len() != 2
        || dims[0] == 0
        || heads == 0
        || head_dim == 0
        || dims[1] != 3 * heads * head_dim
    {
        return Err(Error::Other(
            "WALL-OSS vision QKV RoPE shape mismatch".into(),
        ));
    }
    let shape = Shape::new(vec![dims[0], heads, head_dim]);
    let mut query = context.allocate_output(shape.clone(), DType::BF16)?;
    let mut key = context.allocate_output(shape.clone(), DType::BF16)?;
    let mut value = context.allocate_output(shape, DType::BF16)?;
    let positions = position_ids
        .as_tensor(Shape::new(vec![dims[0], 2]), DType::I32)
        .map_err(Error::Cuda)?;
    ops::split_qkv_axial_rope(
        context,
        ops::AxialRopeArgs {
            qkv,
            bias,
            positions: &positions,
            query: &mut query,
            key: &mut key,
            value: &mut value,
            theta,
            numerics: ops::AxialRopeNumerics::RoundedInput,
        },
    )?;
    Ok((query, key, value))
}

fn dynamic_linear(
    context: &Context,
    input: &Tensor,
    weight: &DynamicFp8LinearWeights,
) -> Result<Tensor> {
    let input_shape = input.shape().dims();
    if input_shape.len() != 2 || input_shape[1] != weight.input_features {
        return Err(Error::Other(format!(
            "dynamic FP8 linear expects input width {}, got {:?}",
            weight.input_features, input_shape
        )));
    }
    let padded_input_features = weight.padded_input_features();
    let activation = ops::quantization::quantize_rows(context, input, Some(padded_input_features))?;
    dynamic_linear_prequantized(context, &activation, weight, false)
}

fn dynamic_linear_prequantized(
    context: &Context,
    activation: &ops::quantization::DynamicFp8Tensor,
    weight: &DynamicFp8LinearWeights,
    keep_padded_output: bool,
) -> Result<Tensor> {
    let activation_shape = activation.values.shape().dims();
    let weight_shape = weight.weight.shape().dims();
    if activation_shape.len() != 2
        || weight_shape.len() != 2
        || activation_shape[1] != weight_shape[0]
    {
        return Err(Error::Other(format!(
            "prequantized dynamic FP8 linear shape mismatch: {:?} @ KN {:?}",
            activation_shape, weight_shape
        )));
    }
    let mut output = context.allocate_output(
        Shape::new(vec![activation_shape[0], weight_shape[1]]),
        DType::BF16,
    )?;
    let gemm = ops::gemm::GemmArgs::row_channel_quantized(
        &activation.values,
        &activation.scales,
        &weight.weight,
        &weight.channel_scales,
        &mut output,
    );
    ops::gemm::gemm(context, gemm)?;
    // Most WallOSS projection widths are already 16-aligned. In that case
    // the GEMM output already has the logical width, and launching a separate
    // slice/copy kernel only adds graph nodes and replay latency.
    if keep_padded_output || output.shape().dims()[1] == weight.output_features {
        Ok(output)
    } else {
        ops::quantization::slice_columns(context, &output, weight.output_features)
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
            let normalized = context.allocate_output(input.shape().clone(), input.dtype())?;
            ops::norm::norm(
                context,
                ops::NormArgs::rms(input, norm_weight, &normalized, eps),
            )?;
            plain_gemm(context, &normalized, weight)
        }
        MatrixRef::DynamicFp8(weight) => {
            let rows = input.shape().dims()[0];
            let output_cols = weight.padded_input_features();
            let values =
                context.allocate_output(Shape::new(vec![rows, output_cols]), DType::F8E4M3)?;
            let scales = context.allocate_output(Shape::new(vec![rows]), DType::F32)?;
            ops::norm::norm(
                context,
                ops::norm::NormArgs {
                    input,
                    output: None,
                    kind: ops::norm::NormKind::Rms,
                    affine: ops::norm::Affine::Weight {
                        weight: norm_weight,
                        bias: None,
                        offset: false,
                    },
                    quantized: Some(ops::norm::QuantOut::RowwiseE4m3 {
                        values: &values,
                        scales: &scales,
                    }),
                    numerics: ops::norm::NormNumerics::Flexible,
                    eps,
                },
            )?;
            let normalized = ops::quantization::DynamicFp8Tensor { values, scales };
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
) -> Result<(Tensor, ops::quantization::DynamicFp8Tensor)> {
    let dims = projected.shape().dims();
    let rows = dims[0];
    let cols = dims[1];
    let hidden = context.allocate_output(Shape::new(vec![rows, cols]), DType::BF16)?;
    let values = context.allocate_output(Shape::new(vec![rows, output_cols]), DType::F8E4M3)?;
    let scales = context.allocate_output(Shape::new(vec![rows]), DType::F32)?;
    ops::norm::fused_add_norm(
        context,
        ops::norm::FusedAddNormArgs {
            input: projected,
            bias: output_bias,
            residual: input,
            gate: None,
            hidden: &hidden,
            normalized: None,
            kind: ops::norm::NormKind::Rms,
            affine: ops::norm::Affine::Weight {
                weight: post_attention_norm,
                bias: None,
                offset: false,
            },
            quantized: Some(ops::norm::QuantOut::RowwiseE4m3 {
                values: &values,
                scales: &scales,
            }),
            numerics: ops::norm::NormNumerics::Flexible,
            bias_order: ops::norm::BiasOrder::SingleRound,
            eps,
        },
    )?;
    Ok((
        hidden,
        ops::quantization::DynamicFp8Tensor { values, scales },
    ))
}

fn dynamic_swiglu(
    context: &Context,
    gate_up: &Tensor,
    gate_up_bias: Option<&Tensor>,
    logical_inner: usize,
    output_cols: usize,
) -> Result<ops::quantization::DynamicFp8Tensor> {
    ops::pointwise::swiglu_quantize_rows(context, gate_up, gate_up_bias, logical_inner, output_cols)
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
            let projected = plain_gemm(context, attention, output)?;
            let hidden = context.allocate_output(projected.shape().clone(), DType::BF16)?;
            let normalized = context.allocate_output(projected.shape().clone(), DType::BF16)?;
            ops::fused_add_norm(
                context,
                ops::FusedAddNormArgs {
                    input: &projected,
                    bias: output_bias,
                    residual: input,
                    gate: None,
                    hidden: &hidden,
                    normalized: Some(&normalized),
                    kind: ops::NormKind::Rms,
                    affine: ops::Affine::Weight {
                        weight: post_attention_norm,
                        bias: None,
                        offset: false,
                    },
                    quantized: None,
                    numerics: ops::NormNumerics::Flexible,
                    bias_order: ops::BiasOrder::SingleRound,
                    eps,
                },
            )?;
            let gate_up = plain_gemm(context, &normalized, gate_up)?;
            let gate_up = match gate_up_bias {
                Some(bias) => add_bias(context, &gate_up, bias)?,
                None => gate_up,
            };
            let activated = ops::pointwise::swiglu(
                context,
                &gate_up,
                ops::pointwise::SwiGluRounding::FullPrecision,
            )?;
            let down = plain_gemm(context, &activated, down)?;
            let output = context.allocate_output(down.shape().clone(), DType::BF16)?;
            ops::residual_add(
                context,
                ops::ResidualAddArgs {
                    input: &down,
                    bias: down_bias,
                    residual: &hidden,
                    gate: None,
                    hidden: &output,
                    bias_order: ops::BiasOrder::SingleRound,
                },
            )?;
            Ok(output)
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
                gate_up.padded_input_features(),
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
                down.padded_input_features(),
            )?;
            let down = dynamic_linear_prequantized(context, &activated, down, false)?;
            let output = context.allocate_output(down.shape().clone(), DType::BF16)?;
            ops::residual_add(
                context,
                ops::ResidualAddArgs {
                    input: &down,
                    bias: down_bias,
                    residual: &hidden,
                    gate: None,
                    hidden: &output,
                    bias_order: ops::BiasOrder::SingleRound,
                },
            )?;
            Ok(output)
        }
        (MatrixRef::DynamicFp8(output), MatrixRef::Bf16(gate_up), MatrixRef::Bf16(down)) => {
            let projected = dynamic_linear(context, attention, output)?;
            let hidden = context.allocate_output(projected.shape().clone(), DType::BF16)?;
            let normalized = context.allocate_output(projected.shape().clone(), DType::BF16)?;
            ops::fused_add_norm(
                context,
                ops::FusedAddNormArgs {
                    input: &projected,
                    bias: output_bias,
                    residual: input,
                    gate: None,
                    hidden: &hidden,
                    normalized: Some(&normalized),
                    kind: ops::NormKind::Rms,
                    affine: ops::Affine::Weight {
                        weight: post_attention_norm,
                        bias: None,
                        offset: false,
                    },
                    quantized: None,
                    numerics: ops::NormNumerics::Flexible,
                    bias_order: ops::BiasOrder::SingleRound,
                    eps,
                },
            )?;
            let gate_up = plain_gemm(context, &normalized, gate_up)?;
            let gate_up = match gate_up_bias {
                Some(bias) => add_bias(context, &gate_up, bias)?,
                None => gate_up,
            };
            let activated = ops::pointwise::swiglu(
                context,
                &gate_up,
                ops::pointwise::SwiGluRounding::FullPrecision,
            )?;
            let down = plain_gemm(context, &activated, down)?;
            let output = context.allocate_output(down.shape().clone(), DType::BF16)?;
            ops::residual_add(
                context,
                ops::ResidualAddArgs {
                    input: &down,
                    bias: down_bias,
                    residual: &hidden,
                    gate: None,
                    hidden: &output,
                    bias_order: ops::BiasOrder::SingleRound,
                },
            )?;
            Ok(output)
        }
        (MatrixRef::Bf16(output), MatrixRef::DynamicFp8(gate_up), MatrixRef::DynamicFp8(down)) => {
            let projected = plain_gemm(context, attention, output)?;
            let (hidden, normalized) = dynamic_residual_norm(
                context,
                &projected,
                output_bias,
                input,
                post_attention_norm,
                eps,
                gate_up.padded_input_features(),
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
                down.padded_input_features(),
            )?;
            let down = dynamic_linear_prequantized(context, &activated, down, false)?;
            let output = context.allocate_output(down.shape().clone(), DType::BF16)?;
            ops::residual_add(
                context,
                ops::ResidualAddArgs {
                    input: &down,
                    bias: down_bias,
                    residual: &hidden,
                    gate: None,
                    hidden: &output,
                    bias_order: ops::BiasOrder::SingleRound,
                },
            )?;
            Ok(output)
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
    let embedded = embedding_lookup(context, token_embedding, token_ids, tokens)?;
    let mut hidden =
        ops::elementwise::replace_rows(context, &embedded, vision_tokens, vision_row_map)?;
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
    let (query, key, value) = split_gqa_qkv_mrope_cache(
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
    let attention = causal_gqa(context, &query, &key, &value, tokens)?
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
    Ok(LanguageLayerOutput { hidden, key, value })
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
    let (query, key, value) = split_gqa_qkv_mrope_cache(
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
    let attention = causal_gqa(context, &query, &key, &value, cache_tokens)?.reshape(vec![
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
    )
    .map_err(|e| Error::Other(format!("action_embedding: {e}")))?;
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
        )
        .map_err(|e| Error::Other(format!("action_layer {index}: {e}")))?
        .hidden;
    }
    let normalized = context.allocate_output(hidden.shape().clone(), hidden.dtype())?;
    ops::norm::norm(
        context,
        ops::NormArgs::rms(&hidden, final_norm, &normalized, config.rms_norm_eps),
    )
    .map_err(|e| Error::Other(format!("action final rms: {e}")))?;
    velocity(context, action_weights, &normalized)
        .map_err(|e| Error::Other(format!("velocity: {e}")))
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
    let (query, key, value) = split_vision_qkv_rope(
        context,
        &qkv,
        Some(weights.qkv_bias()),
        position_ids,
        config.num_heads,
        head_dim,
        config.rope_theta,
    )?;
    let attention = segmented_mha(
        context,
        &query,
        &key,
        &value,
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
    let ordered_patches = ops::elementwise::gather_rows(
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
    let normalized = context.allocate_output(input.shape().clone(), input.dtype())?;
    ops::norm::norm(
        context,
        ops::NormArgs::rms(
            input,
            weights.merger_norm(),
            &normalized,
            config.rms_norm_eps,
        ),
    )?;
    let normalized = normalized.reshape(vec![merged_tokens, merge_unit * config.hidden_size])?;
    let hidden = linear(context, &normalized, weights.merger_hidden())?;
    let hidden = bias_activation(
        context,
        &hidden,
        Some(weights.merger_hidden_bias()),
        ops::pointwise::PointwiseActivation::Gelu,
    )?;
    let output = linear(context, &hidden, weights.merger_output())?;
    let output = add_bias(context, &output, weights.merger_output_bias())?;
    ops::elementwise::gather_rows(context, &output, reverse_indices, merged_tokens)
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
    let action = plain_gemm(context, noisy_action, &weights.noisy_action_projection)
        .map_err(|e| Error::Other(format!("noisy_action_projection gemm: {e}")))?;
    let mask = plain_gemm(context, dof_mask, &weights.dof_projection)
        .map_err(|e| Error::Other(format!("dof_projection gemm: {e}")))?;
    let action_sum = context.allocate_output(action.shape().clone(), action.dtype())?;
    ops::elementwise::add(context, &action, &mask, &action_sum)
        .map_err(|e| Error::Other(format!("action+mask add: {e}")))?;

    let action_part = plain_gemm(context, &action_sum, &weights.action_projection)
        .map_err(|e| Error::Other(format!("action_projection gemm: {e}")))?;
    let time_part = plain_gemm(context, time_embedding, &weights.time_projection)
        .map_err(|e| Error::Other(format!("time_projection gemm: {e}")))?;
    let fused = context.allocate_output(action_part.shape().clone(), action_part.dtype())?;
    ops::elementwise::add(context, &action_part, &time_part, &fused)
        .map_err(|e| Error::Other(format!("fused add: {e}")))?;
    let activated = activation(context, &fused, ops::ElementwiseActivation::Silu)
        .map_err(|e| Error::Other(format!("silu: {e}")))?;
    plain_gemm(context, &activated, &weights.action_embedding_projection)
        .map_err(|e| Error::Other(format!("action_embedding_projection gemm: {e}")))
}

pub fn velocity(
    context: &Context,
    weights: &WallossActionWeights,
    action_hidden: &Tensor,
) -> Result<Tensor> {
    plain_gemm(context, action_hidden, &weights.velocity_projection)
}

pub fn solver_update(
    context: &Context,
    state: &Tensor,
    velocity: &Tensor,
    dt: f32,
) -> Result<Tensor> {
    euler_update(context, state, velocity, dt)
}
