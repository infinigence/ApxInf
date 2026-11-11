//! Fused-op reference BF16 blocks for PI0.5 (Ascend).
//!
//! Mirrors "bf16.rs" but swaps the two residual+norm helpers to use
//! Fused aclnn ops matching the torch_npu op distribution:
//!
//! - "aclnnInplaceAddRmsNorm" (1 op) replaces "aclnnInplaceAdd" +
//!   "aclnnAddRmsNorm" (2 ops) in the post-attention / post-MLP path.
//! - "aclnnAddLayerNorm" (1 op) replaces "aclnnAdd" + "aclnnLayerNorm"
//!   (2 ops) in the vision-encoder residual path.
//!
//! All other ops (attention, rope, split, pointwise silu/mul, gemm,
//! geglu) are identical to "bf16.rs". Selected via "ModelVariant::Bf16Ref".

use crate::ascend_pi05::backend::{
    ops::{
        attention as l3_attention, concat_rows as l3_concat_rows, gather as l3_gather,
        adaptive_rms_norm as l3_adaptive_rms_norm, bias_residual as l3_bias_residual,
        add_layer_norm as l3_add_layer_norm,
        bias_residual_layer_norm as l3_bias_residual_layer_norm,
        inplace_add_rms_norm_zero_copy as l3_inplace_add_rms_norm_zero_copy,
        bias_residual_rms_norm as l3_bias_residual_rms_norm, gemm as l3_gemm,
        gemm_bias as l3_gemm_bias, gemm_geglu as l3_gemm_geglu, layer_norm as l3_layer_norm,
        pointwise as l3_pointwise, rms_norm as l3_rms_norm,
        reserve_prefix as l3_reserve_prefix, rope as l3_rope, AttentionArgs, GatherArgs,
        AdaptiveRmsNormArgs, AddLayerNormArgs, BiasResidualArgs, BiasResidualLayerNormArgs,
        BiasResidualRmsNormArgs, GatherPatchGeometry, GatherSemantic,
        GemmArgs, GemmBiasArgs, GemmGegluArgs, LayerNormArgs, PointwiseActivation, PointwiseArgs,
        PointwiseSemantic, RmsNormArgs, RopeArgs, RopeSemantic, WeightVersion,
    },
    Context, DeviceBuffer,
};
use apxinf_core::{DType, Error, Result, Shape, Tensor};

use super::L3Policies;
use crate::ascend_pi05::{
    Bf16DeviceActionLayer, Bf16DeviceLanguageLayer, Bf16DeviceVisionBlock, Bf16LinearWeights,
    GemmaVariantConfig,
};

pub struct Bf16RefLanguageLayerOutput {
    pub hidden: Tensor,
    pub key: Tensor,
    pub value: Tensor,
}

pub struct Bf16RefActionLayerOutput {
    pub hidden: Tensor,
    pub next_normalized: Tensor,
}

/// Intermediate tensors captured inside the first transformer block for
/// precision alignment debugging. All tensors remain on device; the caller
/// is responsible for copying to host.
///
/// Field names mirror the dump keys used in captures:
///   - normalized: hidden after adaptive_rms_norm (pre-QKV GEMM)
///   - qkv: QKV GEMM raw output [seq, (q+2kv)*head_dim]
///   - q_post_rope: Q after RoPE [seq, q_heads, head_dim]
///   - k_post_rope: K (sliced from cache) after RoPE [seq, kv_heads, head_dim]
///   - v: V (sliced from cache, no RoPE) [seq, kv_heads, head_dim]
///   - attn_out: MQA/FAS output [seq, q_heads, head_dim]
///   - attn_proj: attention output projection (GEMM) [seq, width]
///   - post_attn_hidden: hidden after adaptive gate + residual + RMS (mlp input residual)
///   - post_attn_norm: normalized hidden fed to MLP GEMM
///   - geglu_out: GeGLU activated [seq, mlp_dim]
///   - mlp_proj: down projection output [seq, width]
///   - mlp_hidden: final hidden after second adaptive gate + residual + RMS
///   - mlp_next_norm: final normalized (next layer's pre-norm input)
pub struct Bf16ActionLayerInternals {
    pub normalized: Tensor,
    pub qkv: Tensor,
    pub q_post_rope: Tensor,
    pub k_post_rope: Tensor,
    pub v: Tensor,
    pub attn_out: Tensor,
    pub attn_proj: Tensor,
    pub post_attn_hidden: Tensor,
    pub post_attn_norm: Tensor,
    pub geglu_out: Tensor,
    pub mlp_proj: Tensor,
    pub mlp_hidden: Tensor,
    pub mlp_next_norm: Tensor,
}

struct QkvTensors {
    q: Tensor,
    k: Tensor,
    v: Tensor,
}

struct ResidualNormTensors {
    hidden: Tensor,
    normalized: Tensor,
}

fn rms_bf16(ctx: &Context, input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let mut output = ctx.allocate_output(input.shape().clone(), DType::BF16)?;
    l3_rms_norm(ctx, RmsNormArgs::new(input, weight, &mut output, eps))?;
    Ok(output)
}

/// Extract raw device pointer from a tensor (for direct aclnn FFI calls).
fn device_ptr_2d(tensor: &Tensor) -> usize {
    match tensor.storage() {
        apxinf_core::Storage::Gpu { handle, .. } => handle.ptr(),
        _ => 0,
    }
}

/// Element-wise mul: `out = a * b` via `aclnnMul`.
/// Matches torch_npu graph `Mul` op for gate multiplication.
fn mul_bf16(ctx: &Context, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let out = ctx.allocate_output(a.shape().clone(), DType::BF16)?;
    let b_shape = b.shape();
    let b_reshaped = if b_shape.dims().len() == 1 && a.shape().dims().len() == 2 {
        b.reshape(Shape::new(vec![1, b_shape.dims()[0]]))?
    } else {
        b.clone()
    };
    crate::ascend_pi05::backend::ops::aclnn_mul(
        device_ptr_2d(a),
        device_ptr_2d(&b_reshaped),
        device_ptr_2d(&out),
        a.shape(),
        b_reshaped.shape(),
        DType::BF16,
        ctx.raw_stream(),
    )?;
    Ok(out)
}

/// Gelu (tanh approximation): `out = gelu(input)` via `aclnnGelu`.
/// Matches torch_npu graph `Gelu` op for MLP activation.
fn gelu_bf16(ctx: &Context, input: &Tensor) -> Result<Tensor> {
    let out = ctx.allocate_output(input.shape().clone(), DType::BF16)?;
    crate::ascend_pi05::backend::ops::aclnn_gelu(
        device_ptr_2d(input),
        device_ptr_2d(&out),
        input.shape(),
        DType::BF16,
        ctx.raw_stream(),
    )?;
    Ok(out)
}

/// Element-wise add: `out = a + alpha * b` via `aclnnAdd`.
/// Matches torch_npu graph `Add` op for residual addition.
fn add_bf16(ctx: &Context, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let out = ctx.allocate_output(a.shape().clone(), DType::BF16)?;
    crate::ascend_pi05::backend::ops::aclnn_add(
        device_ptr_2d(a),
        device_ptr_2d(b),
        device_ptr_2d(&out),
        a.shape(),
        b.shape(),
        1.0,
        DType::BF16,
        ctx.raw_stream(),
    )?;
    Ok(out)
}

/// Scalar mul: `out = input * scalar` via `aclnnMuls`.
/// Matches torch_npu graph `Muls` op for velocity * dt in euler update.
fn muls_bf16(ctx: &Context, input: &Tensor, scalar: f32) -> Result<Tensor> {
    let out = ctx.allocate_output(input.shape().clone(), DType::BF16)?;
    crate::ascend_pi05::backend::ops::aclnn_muls(
        device_ptr_2d(input),
        device_ptr_2d(&out),
        input.shape(),
        scalar,
        DType::BF16,
        ctx.raw_stream(),
    )?;
    Ok(out)
}

fn layer_bf16(
    ctx: &Context,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let mut output = ctx.allocate_output(input.shape().clone(), DType::BF16)?;
    l3_layer_norm(
        ctx,
        LayerNormArgs::new(input, weight, bias, &mut output, eps),
    )?;
    Ok(output)
}

fn adaptive_rms_bf16(
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
    l3_adaptive_rms_norm(
        ctx,
        AdaptiveRmsNormArgs::new(input, &style, &mut output, eps),
    )?;
    Ok(output)
}

fn bias_activation_bf16(
    ctx: &Context,
    input: &Tensor,
    bias: Option<&Tensor>,
    activation: PointwiseActivation,
) -> Result<Tensor> {
    let mut output = ctx.allocate_output(input.shape().clone(), DType::BF16)?;
    let mut args = PointwiseArgs::new(PointwiseSemantic::BiasActivation, input, &mut output);
    args.bias = bias;
    args.activation = activation;
    l3_pointwise(ctx, args)?;
    Ok(output)
}

fn euler_update_bf16(
    ctx: &Context,
    state: &Tensor,
    velocity: &Tensor,
    dt: f32,
) -> Result<Tensor> {
    // torch_npu graph: Muls(velocity, dt) -> Add(state, scaled_vel).
    // Split into 2 ops to match graph op sequence (no fused EulerUpdate).
    let scaled = muls_bf16(ctx, velocity, dt)?;
    add_bf16(ctx, state, &scaled)
}

fn add_position_bf16(
    ctx: &Context,
    projection: &Tensor,
    bias: Option<&Tensor>,
    position: &Tensor,
    tokens_per_view: usize,
) -> Result<Tensor> {
    let mut output = ctx.allocate_output(projection.shape().clone(), DType::BF16)?;
    let mut args = GatherArgs::new(GatherSemantic::BiasPosition, projection, &mut output);
    args.bias = bias;
    args.position = Some(position);
    args.tokens_per_view = tokens_per_view;
    l3_gather(ctx, args)?;
    Ok(output)
}

fn gemm_bf16(
    ctx: &Context,
    policies: &L3Policies,
    activation: &Tensor,
    weight: &Tensor,
) -> Result<Tensor> {
    let a = activation.shape().dims();
    let b = weight.shape().dims();
    if activation.dtype() != DType::BF16
        || weight.dtype() != DType::BF16
        || a.len() != 2
        || b.len() != 2
        || a[1] != b[0]
    {
        return Err(Error::Other(format!(
            "PI0.5 BF16 GEMM shape/dtype mismatch: {} {a:?} @ {} {b:?}",
            activation.dtype(),
            weight.dtype()
        )));
    }
    let mut output = ctx.allocate_output(Shape::new(vec![a[0], b[1]]), DType::BF16)?;
    let mut args = GemmArgs::new(activation, weight, &mut output);
    args.policy = policies.gemm.clone();
    l3_gemm(ctx, args)?;
    Ok(output)
}

/// Fused GEMM + bias: `out = activation @ weight + bias`.
///
/// Under `AclnnFusedMatmul` policy, bias is fused into the matmul kernel
/// (1 launch instead of matmul + aclnnAdd). Falls back to 2-launch
/// `aclnnMatmul + aclnnAdd` under `AclnnMatmul` policy.
fn gemm_bias_bf16(
    ctx: &Context,
    policies: &L3Policies,
    activation: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
) -> Result<Tensor> {
    let a = activation.shape().dims();
    let b = weight.shape().dims();
    if activation.dtype() != DType::BF16
        || weight.dtype() != DType::BF16
        || bias.dtype() != DType::BF16
        || a.len() != 2
        || b.len() != 2
        || a[1] != b[0]
    {
        return Err(Error::Other(format!(
            "PI0.5 BF16 GEMM+bias shape/dtype mismatch: {} {a:?} @ {} {b:?} + {} {:?}",
            activation.dtype(),
            weight.dtype(),
            bias.dtype(),
            bias.shape().dims()
        )));
    }
    let mut output = ctx.allocate_output(Shape::new(vec![a[0], b[1]]), DType::BF16)?;
    let mut gemm = GemmArgs::new(activation, weight, &mut output);
    gemm.policy = policies.gemm.clone();
    l3_gemm_bias(ctx, GemmBiasArgs { gemm, bias })?;
    Ok(output)
}

fn gemm_geglu_bf16(
    ctx: &Context,
    policies: &L3Policies,
    activation: &Tensor,
    canonical_weight: &Tensor,
) -> Result<Tensor> {
    let a = activation.shape().dims();
    let b = canonical_weight.shape().dims();
    if activation.dtype() != DType::BF16
        || canonical_weight.dtype() != DType::BF16
        || a.len() != 2
        || b.len() != 2
        || a[1] != b[0]
        || b[1] % 2 != 0
    {
        return Err(Error::Other(format!(
            "PI0.5 BF16 GEMM+GeGLU shape/dtype mismatch: {} {a:?} @ {} {b:?}",
            activation.dtype(),
            canonical_weight.dtype()
        )));
    }
    let mut output = ctx.allocate_output(Shape::new(vec![a[0], b[1] / 2]), DType::BF16)?;
    let mut gemm = GemmArgs::new(activation, canonical_weight, &mut output)
        .with_immutable_weight(WeightVersion::new(1));
    gemm.policy = policies.gemm.clone();
    l3_gemm_geglu(ctx, GemmGegluArgs { gemm })?;
    Ok(output)
}

fn mqa_bf16(
    ctx: &Context,
    policies: &L3Policies,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    key_tokens: usize,
) -> Result<Tensor> {
    let q = query.shape().dims();
    let k = key.shape().dims();
    if query.dtype() != DType::BF16
        || key.dtype() != DType::BF16
        || value.dtype() != DType::BF16
        || q.len() != 3
        || value.shape() != key.shape()
        || !matches!(k.len(), 2 | 3)
        || k[k.len() - 1] != q[2]
    {
        return Err(Error::Other("PI0.5 BF16 MQA shape/dtype mismatch".into()));
    }
    let kv_heads = if k.len() == 2 { 1 } else { k[1] };
    let available_tokens = k[0];
    if key_tokens == 0 || key_tokens > available_tokens || q[1] % kv_heads != 0 {
        return Err(Error::Other("PI0.5 BF16 MQA token/head mismatch".into()));
    }
    let kv_elements = key_tokens
        .checked_mul(kv_heads)
        .and_then(|count| count.checked_mul(q[2]))
        .ok_or_else(|| Error::Other("PI0.5 BF16 MQA size overflow".into()))?;
    let kv_bytes = kv_elements
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("PI0.5 BF16 MQA byte-size overflow".into()))?;
    let key = DeviceBuffer::from_tensor(key)
        .and_then(|buffer| buffer.view(0, kv_bytes))
        .and_then(|buffer| {
            buffer.as_tensor(
                Shape::new(vec![1, key_tokens, kv_heads, q[2]]),
                DType::BF16,
            )
        })
        .map_err(|e| Error::Other(format!("MQA key view: {e}")))?;
    let value = DeviceBuffer::from_tensor(value)
        .and_then(|buffer| buffer.view(0, kv_bytes))
        .and_then(|buffer| {
            buffer.as_tensor(
                Shape::new(vec![1, key_tokens, kv_heads, q[2]]),
                DType::BF16,
            )
        })
        .map_err(|e| Error::Other(format!("MQA value view: {e}")))?;
    let query4 = query.reshape(Shape::new(vec![1, q[0], q[1], q[2]]))?;
    let mut output = ctx.allocate_output(query4.shape().clone(), DType::BF16)?;
    let mut args = AttentionArgs::new(&query4, &key, &value, &mut output);
    args.policy = policies.attention.clone();
    l3_attention(ctx, args)?;
    output.reshape(query.shape().clone())
}

fn mha_bf16(
    ctx: &Context,
    policies: &L3Policies,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    tokens_per_batch: usize,
) -> Result<Tensor> {
    let shape = query.shape().dims();
    if query.dtype() != DType::BF16
        || key.dtype() != DType::BF16
        || value.dtype() != DType::BF16
        || shape.len() != 3
        || key.shape() != query.shape()
        || value.shape() != query.shape()
        || tokens_per_batch == 0
        || shape[0] % tokens_per_batch != 0
    {
        return Err(Error::Other("PI0.5 BF16 MHA shape/dtype mismatch".into()));
    }
    let batch = shape[0] / tokens_per_batch;
    let shape4 = Shape::new(vec![batch, tokens_per_batch, shape[1], shape[2]]);
    let query4 = query.reshape(shape4.clone())?;
    let key4 = key.reshape(shape4.clone())?;
    let value4 = value.reshape(shape4.clone())?;
    let mut output = ctx.allocate_output(shape4, DType::BF16)?;
    let mut args = AttentionArgs::new(&query4, &key4, &value4, &mut output);
    args.policy = policies.attention.clone();
    l3_attention(ctx, args)?;
    output.reshape(query.shape().clone())
}

fn split_qkv_bias_bf16(
    ctx: &Context,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    heads: usize,
    head_dim: usize,
) -> Result<QkvTensors> {
    let shape = qkv.shape().dims();
    if qkv.dtype() != DType::BF16
        || shape.len() != 2
        || shape[1] != 3 * heads * head_dim
    {
        return Err(Error::Other(
            "PI0.5 BF16 vision QKV shape/dtype mismatch".into(),
        ));
    }
    let output_shape = Shape::new(vec![shape[0], heads, head_dim]);
    let mut q = ctx.allocate_output(output_shape.clone(), DType::BF16)?;
    let mut k = ctx.allocate_output(output_shape.clone(), DType::BF16)?;
    let mut v = ctx.allocate_output(output_shape, DType::BF16)?;
    l3_rope(
        ctx,
        RopeArgs {
            semantic: RopeSemantic::SplitQkvBias,
            qkv,
            bias,
            q: &mut q,
            k: &mut k,
            v: &mut v,
            q_heads: heads,
            kv_heads: heads,
            head_dim,
            theta: 1.0,
            position_offset: 0,
            kv_output_offset: 0,
        },
    )?;
    Ok(QkvTensors { q, k, v })
}

#[allow(clippy::too_many_arguments)]
fn split_qkv_rope_bf16(
    ctx: &Context,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    theta: f32,
    position_offset: usize,
) -> Result<QkvTensors> {
    let shape = qkv.shape().dims();
    let expected_width = (q_heads + 2 * kv_heads)
        .checked_mul(head_dim)
        .ok_or_else(|| Error::Other("PI0.5 BF16 QKV width overflow".into()))?;
    if qkv.dtype() != DType::BF16 || shape.len() != 2 || shape[1] != expected_width {
        return Err(Error::Other(
            "PI0.5 BF16 rotary QKV shape/dtype mismatch".into(),
        ));
    }
    let mut q = ctx.allocate_output(
        Shape::new(vec![shape[0], q_heads, head_dim]),
        DType::BF16,
    )?;
    let kv_shape = Shape::new(vec![shape[0], kv_heads, head_dim]);
    let mut k = ctx.allocate_output(kv_shape.clone(), DType::BF16)?;
    let mut v = ctx.allocate_output(kv_shape, DType::BF16)?;
    l3_rope(
        ctx,
        RopeArgs {
            semantic: RopeSemantic::SplitQkvRope,
            qkv,
            bias,
            q: &mut q,
            k: &mut k,
            v: &mut v,
            q_heads,
            kv_heads,
            head_dim,
            theta,
            position_offset,
            kv_output_offset: 0,
        },
    )?;
    Ok(QkvTensors { q, k, v })
}

#[allow(clippy::too_many_arguments)]
fn split_qkv_rope_into_cache_bf16(
    ctx: &Context,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    theta: f32,
    position_offset: usize,
    key_cache: &Tensor,
    value_cache: &Tensor,
    cache_offset: usize,
) -> Result<Tensor> {
    let shape = qkv.shape().dims();
    let expected_width = (q_heads + 2 * kv_heads)
        .checked_mul(head_dim)
        .ok_or_else(|| Error::Other("PI0.5 BF16 QKV width overflow".into()))?;
    if qkv.dtype() != DType::BF16 || shape.len() != 2 || shape[1] != expected_width {
        return Err(Error::Other(
            "PI0.5 BF16 cached rotary QKV shape/dtype mismatch".into(),
        ));
    }
    let mut q = ctx.allocate_output(
        Shape::new(vec![shape[0], q_heads, head_dim]),
        DType::BF16,
    )?;
    let mut key_cache = key_cache.clone();
    let mut value_cache = value_cache.clone();
    l3_rope(
        ctx,
        RopeArgs {
            semantic: RopeSemantic::SplitQkvRope,
            qkv,
            bias,
            q: &mut q,
            k: &mut key_cache,
            v: &mut value_cache,
            q_heads,
            kv_heads,
            head_dim,
            theta,
            position_offset,
            kv_output_offset: cache_offset,
        },
    )?;
    Ok(q)
}

fn bf16_vector_prefix(tensor: &Tensor, elements: usize) -> Result<Tensor> {
    let bytes = elements
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("PI0.5 BF16 vector size overflow".into()))?;
    DeviceBuffer::from_tensor(tensor)
        .and_then(|buffer| buffer.view(0, bytes))
        .and_then(|buffer| buffer.as_tensor(Shape::new(vec![elements]), DType::BF16))
        .map_err(|e| Error::Other(format!("bf16_vector_prefix: {e}")))
}

fn bias_residual_bf16(
    ctx: &Context,
    projection: &Tensor,
    bias: Option<&Tensor>,
    residual: &Tensor,
) -> Result<Tensor> {
    let mut hidden = ctx.allocate_output(projection.shape().clone(), DType::BF16)?;
    l3_bias_residual(
        ctx,
        BiasResidualArgs::new(projection, bias, residual, &mut hidden),
    )?;
    Ok(hidden)
}

/// Unfused bias + residual + RMS norm (fallback path).
///
/// The fused path uses `gemm_bias_bf16` + `inplace_add_rms_bf16` instead.
/// Kept as a fallback for callers where bias cannot be fused into the GEMM
/// (e.g. when the projection comes from a non-GEMM source).
#[allow(dead_code)]
fn bias_residual_rms_bf16(
    ctx: &Context,
    projection: &Tensor,
    bias: Option<&Tensor>,
    residual: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<ResidualNormTensors> {
    let mut hidden = ctx.allocate_output(projection.shape().clone(), DType::BF16)?;
    let mut normalized = ctx.allocate_output(projection.shape().clone(), DType::BF16)?;
    l3_bias_residual_rms_norm(
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

/// Fused residual + RMS norm via `aclnnInplaceAddRmsNorm`.
///
/// `projection_with_bias` is modified in place: `projection_with_bias += residual`.
/// Then `normalized = rms_norm(projection_with_bias, weight)`.
///
/// Replaces the 2-op path (`aclnnAddRmsNorm` which already fused add+norm
/// but needs a separate bias-add before it) when used together with
/// `gemm_bias_bf16`: `gemm_bias` (1 op, bias fused into matmul) +
/// `inplace_add_rms_bf16` (1 op) = 2 ops total, vs the original 3 ops
/// (matmul + inplace_add bias + AddRmsNorm).
///
/// Zero-copy variant: `projection_with_bias` becomes `normalized` (x1Ref is
/// overwritten in place by the operator), and `residual` becomes `hidden`
/// (x2Ref is overwritten with the residual sum). No `aclnnInplaceCopy` (d2d)
/// ops are emitted — saves ~754 d2d ops per inference vs the legacy wrapper.
///
/// Callers MUST NOT read `residual` after this call (its device memory is
/// overwritten). All current call sites satisfy this: `residual` is either
/// `input` (last use) or `&fused.hidden` (moved into the new `fused`).
fn inplace_add_rms_bf16(
    ctx: &Context,
    projection_with_bias: Tensor,
    residual: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<ResidualNormTensors> {
    // x2 clone is zero-copy (Arc refcount on GpuStorageHandle).
    let x2_owned = residual.clone();
    let (normalized, hidden) = l3_inplace_add_rms_norm_zero_copy(
        ctx,
        projection_with_bias,
        x2_owned,
        weight,
        eps,
    )?;
    Ok(ResidualNormTensors { hidden, normalized })
}

#[allow(clippy::too_many_arguments)]
fn bias_residual_layer_bf16(
    ctx: &Context,
    projection: &Tensor,
    projection_bias: Option<&Tensor>,
    residual: &Tensor,
    norm_weight: &Tensor,
    norm_bias: &Tensor,
    eps: f32,
) -> Result<ResidualNormTensors> {
    // Fused path: hidden = projection + projection_bias + residual via the
    // existing bias_residual helper, then layer_norm. The aclnnAddLayerNorm
    // fusion is not applicable here because bias_residual already performed
    // the add; a separate layer_norm op is correct and graph-safe.
    let mut hidden = ctx.allocate_output(projection.shape().clone(), DType::BF16)?;
    let mut normalized = ctx.allocate_output(projection.shape().clone(), DType::BF16)?;
    l3_bias_residual_layer_norm(
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

/// Fused add + layer norm via `aclnnAddLayerNorm`.
///
/// `hidden = projection + residual; normalized = layer_norm(hidden, weight, bias)`.
/// Single op (1 kernel launch) replaces the 3-op `bias_residual_layer_bf16`
/// (add + inplace_add + layer_norm). Requires projection bias to be fused
/// into the upstream GEMM via `gemm_bias_bf16`.
fn add_layer_norm_bf16(
    ctx: &Context,
    projection: &Tensor,
    residual: &Tensor,
    norm_weight: &Tensor,
    norm_bias: &Tensor,
    eps: f32,
) -> Result<ResidualNormTensors> {
    let mut hidden = ctx.allocate_output(projection.shape().clone(), DType::BF16)?;
    let mut normalized = ctx.allocate_output(projection.shape().clone(), DType::BF16)?;
    l3_add_layer_norm(
        ctx,
        AddLayerNormArgs {
            x1: projection,
            x2: residual,
            weight: norm_weight,
            bias: norm_bias,
            norm_bias: None,
            normalized: &mut normalized,
            hidden: Some(&mut hidden),
            eps,
        },
    )?;
    Ok(ResidualNormTensors { hidden, normalized })
}

fn adaptive_gate_residual_rms_bf16(
    ctx: &Context,
    projection: &Tensor,
    residual: &Tensor,
    gate_style: &Tensor,
    norm_style: &Tensor,
    eps: f32,
) -> Result<ResidualNormTensors> {
    let cols = projection
        .shape()
        .dims()
        .last()
        .copied()
        .ok_or_else(|| Error::Other("PI0.5 AdaRMS input has no columns".into()))?;
    let norm_style = bf16_vector_prefix(norm_style, 2 * cols)?;
    let mut hidden = ctx.allocate_output(projection.shape().clone(), DType::BF16)?;
    let mut normalized = ctx.allocate_output(projection.shape().clone(), DType::BF16)?;
    crate::ascend_pi05::backend::ops::ada_gate_residual_rms_norm(
        ctx,
        crate::ascend_pi05::backend::ops::AdaGateResidualRmsNormArgs::new(
            projection,
            residual,
            &norm_style,
            gate_style,
            &mut hidden,
            &mut normalized,
            eps,
        ),
    )?;
    Ok(ResidualNormTensors { hidden, normalized })
}

#[allow(clippy::too_many_arguments)]
pub fn language_layer_bf16(
    ctx: &Context,
    config: GemmaVariantConfig,
    weights: &Bf16DeviceLanguageLayer,
    input: &Tensor,
    compute_tail: bool,
    position_offset: usize,
    rms_eps: f32,
    rope_theta: f32,
) -> Result<Bf16RefLanguageLayerOutput> {
    language_layer_bf16_with_policies(
        ctx,
        &L3Policies::default(),
        config,
        weights,
        input,
        compute_tail,
        position_offset,
        rms_eps,
        rope_theta,
    )
}

#[allow(clippy::too_many_arguments)]
fn language_layer_bf16_with_policies(
    ctx: &Context,
    policies: &L3Policies,
    config: GemmaVariantConfig,
    weights: &Bf16DeviceLanguageLayer,
    input: &Tensor,
    compute_tail: bool,
    position_offset: usize,
    rms_eps: f32,
    rope_theta: f32,
) -> Result<Bf16RefLanguageLayerOutput> {
    let normalized = rms_bf16(ctx, input, &weights.input_norm_scale, rms_eps)?;
    let qkv = gemm_bf16(ctx, policies, &normalized, &weights.qkv.weight)?;
    let qkv = split_qkv_rope_bf16(
        ctx,
        &qkv,
        weights.qkv.bias.as_ref(),
        config.num_heads,
        config.num_kv_heads,
        config.head_dim,
        rope_theta,
        position_offset,
    )?;
    let tokens = input.shape().dims()[0];
    if !compute_tail {
        return Ok(Bf16RefLanguageLayerOutput {
            hidden: input.clone(),
            key: qkv.key_2d(tokens, config.head_dim)?,
            value: qkv.value_2d(tokens, config.head_dim)?,
        });
    }
    let attention = mqa_bf16(ctx, policies, &qkv.q, &qkv.k, &qkv.v, tokens)?
        .reshape(vec![tokens, config.num_heads * config.head_dim])?;
    // Fused path: gemm_bias (1 op, bias fused into matmul) +
    // inplace_add_rms (1 op) = 2 ops vs original 3 ops
    // (matmul + inplace_add bias + AddRmsNorm).
    let projected = match weights.output.bias.as_ref() {
        Some(bias) => gemm_bias_bf16(ctx, policies, &attention, &weights.output.weight, bias)?,
        None => gemm_bf16(ctx, policies, &attention, &weights.output.weight)?,
    };
    let fused = inplace_add_rms_bf16(
        ctx,
        projected,
        input,
        &weights.post_attention_norm_scale,
        rms_eps,
    )?;
    let activated = gemm_geglu_bf16(ctx, policies, &fused.normalized, &weights.gate_up.weight)?;
    // Fused path: gemm_bias (1 op) + bias_residual without bias (1 op,
    // just residual add) = 2 ops vs original 3 ops (matmul + bias add +
    // residual add). The down projection bias is fused into gemm_bias; the
    // residual add is a plain aclnnAdd (no norm here).
    let projected = match weights.down.bias.as_ref() {
        Some(bias) => gemm_bias_bf16(ctx, policies, &activated, &weights.down.weight, bias)?,
        None => gemm_bf16(ctx, policies, &activated, &weights.down.weight)?,
    };
    let hidden = bias_residual_bf16(ctx, &projected, None, &fused.hidden)?;
    Ok(Bf16RefLanguageLayerOutput {
        hidden,
        key: qkv.key_2d(tokens, config.head_dim)?,
        value: qkv.value_2d(tokens, config.head_dim)?,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn action_layer_bf16(
    ctx: &Context,
    config: GemmaVariantConfig,
    weights: &Bf16DeviceActionLayer,
    input: &Tensor,
    attention_normalized: Option<&Tensor>,
    attention_scale_weight: &Tensor,
    attention_gate: &Tensor,
    post_attention_scale_weight: &Tensor,
    mlp_gate: &Tensor,
    next_scale_weight: &Tensor,
    qkv_bias: &Tensor,
    gate_proj_bias: &Tensor,
    up_proj_bias: &Tensor,
    gate_weight: &Tensor,
    up_weight: &Tensor,
    prefix_k: &Tensor,
    prefix_v: &Tensor,
    position_offset: usize,
    kv_cache_offset: usize,
    kv_key_count: usize,
    rms_eps: f32,
    rope_theta: f32,
) -> Result<Bf16RefActionLayerOutput> {
    action_layer_bf16_with_policies(
        ctx,
        &L3Policies::default(),
        config,
        weights,
        input,
        attention_normalized,
        attention_scale_weight,
        attention_gate,
        post_attention_scale_weight,
        mlp_gate,
        next_scale_weight,
        qkv_bias,
        gate_proj_bias,
        up_proj_bias,
        gate_weight,
        up_weight,
        prefix_k,
        prefix_v,
        position_offset,
        kv_cache_offset,
        kv_key_count,
        rms_eps,
        rope_theta,
    )
}

/// Action layer matching torch_npu 31ms graph op sequence:
/// 1. RmsNorm(x, scale_weight) — first block only; later blocks reuse
///    previous block's InplaceAddRmsNorm normalized output
/// 2. MatMul(QKV) + bias (shift folded)
/// 3. SplitV → ApplyRotaryPosEmb → FAS (cross-attn)
/// 4. MatMul(output proj)
/// 5. Mul(attn_out, gate) — _gated_residual
/// 6. InplaceAddRmsNorm(scaled, residual, post_scale_weight) → hidden + normed
/// 7. MatMul(gate_proj) + bias → Gelu → MatMul(up_proj) + bias → Mul(gelu, up)
/// 8. MatMul(down_proj)
/// 9. Mul(mlp_out, gate) — _gated_residual
/// 10. InplaceAddRmsNorm(scaled, residual, next_scale_weight) → hidden + next_normed
#[allow(clippy::too_many_arguments)]
fn action_layer_bf16_with_policies(
    ctx: &Context,
    policies: &L3Policies,
    config: GemmaVariantConfig,
    weights: &Bf16DeviceActionLayer,
    input: &Tensor,
    attention_normalized: Option<&Tensor>,
    attention_scale_weight: &Tensor,
    attention_gate: &Tensor,
    post_attention_scale_weight: &Tensor,
    mlp_gate: &Tensor,
    next_scale_weight: &Tensor,
    qkv_bias: &Tensor,
    gate_proj_bias: &Tensor,
    up_proj_bias: &Tensor,
    gate_weight: &Tensor,
    up_weight: &Tensor,
    prefix_k: &Tensor,
    prefix_v: &Tensor,
    position_offset: usize,
    kv_cache_offset: usize,
    kv_key_count: usize,
    rms_eps: f32,
    rope_theta: f32,
) -> Result<Bf16RefActionLayerOutput> {
    // 1. Pre-attn norm: RmsNorm(x, scale_weight=1+scale)
    //    For first block: RmsNorm(input, attention_scale_weight)
    //    For later blocks: attention_normalized is already the normed output
    let normalized = match attention_normalized {
        Some(value) => value.clone(),
        None => rms_bf16(ctx, input, attention_scale_weight, rms_eps)?,
    };

    // 2. QKV GEMM with folded shift bias: MatMulV3(normed, qkv_weight, qkv_bias)
    let qkv = gemm_bias_bf16(ctx, policies, &normalized, &weights.qkv.weight, qkv_bias)?;

    // 3. Split QKV + RoPE into cache
    let q = split_qkv_rope_into_cache_bf16(
        ctx,
        &qkv,
        None, // bias already folded into qkv_bias via gemm_bias
        config.num_heads,
        config.num_kv_heads,
        config.head_dim,
        rope_theta,
        position_offset,
        prefix_k,
        prefix_v,
        kv_cache_offset,
    )?;



    // 4. Cross-attention via FAS
    let attention = mqa_bf16(
        ctx,
        policies,
        &q,
        prefix_k,
        prefix_v,
        kv_key_count,
    )?
    .reshape(vec![
        input.shape().dims()[0],
        config.num_heads * config.head_dim,
    ])?;

    // 5. Output projection: MatMul(attn, output_weight)
    let projected = gemm_bf16(ctx, policies, &attention, &weights.output.weight)?;

    // 6. Mul(projected, attention_gate) — _gated_residual step 1
    let gated = mul_bf16(ctx, &projected, attention_gate)?;

    // 7. InplaceAddRmsNorm(gated, input, post_attention_scale_weight)
    //    → hidden (residual), normalized (for MLP)
    let fused = inplace_add_rms_bf16(
        ctx,
        gated,
        input,
        post_attention_scale_weight,
        rms_eps,
    )?;


    // 8. MLP: gate_proj + Gelu + up_proj + Mul + down_proj
    //    MatMul(gate_proj) with folded shift bias
    let gate_out = gemm_bias_bf16(ctx, policies, &fused.normalized, gate_weight, gate_proj_bias)?;
    // Gelu(gate_out)
    let gelu_out = gelu_bf16(ctx, &gate_out)?;
    // MatMul(up_proj) with folded shift bias
    let up_out = gemm_bias_bf16(ctx, policies, &fused.normalized, up_weight, up_proj_bias)?;
    // Mul(gelu(gate), up)
    let activated = mul_bf16(ctx, &gelu_out, &up_out)?;
    // MatMul(down_proj)
    let projected = gemm_bf16(ctx, policies, &activated, &weights.down.weight)?;

    // 9. Mul(projected, mlp_gate) — _gated_residual step 2
    let gated = mul_bf16(ctx, &projected, mlp_gate)?;

    // 10. InplaceAddRmsNorm(gated, fused.hidden, next_scale_weight)
    //     → hidden (residual), next_normalized (for next layer)
    let fused = inplace_add_rms_bf16(
        ctx,
        gated,
        &fused.hidden,
        next_scale_weight,
        rms_eps,
    )?;


    Ok(Bf16RefActionLayerOutput {
        hidden: fused.hidden,
        next_normalized: fused.normalized,
    })
}

/// Variant of [`action_layer_bf16_with_policies`] that captures every
/// intermediate tensor inside the block. Used only for precision alignment
/// diagnostics on step.0 action_layer.0 (the first transformer block).
#[allow(clippy::too_many_arguments)]
fn action_layer_bf16_with_policies_dump(
    ctx: &Context,
    policies: &L3Policies,
    config: GemmaVariantConfig,
    weights: &Bf16DeviceActionLayer,
    input: &Tensor,
    attention_normalized: Option<&Tensor>,
    attention_scale_weight: &Tensor,
    attention_gate: &Tensor,
    post_attention_scale_weight: &Tensor,
    mlp_gate: &Tensor,
    next_scale_weight: &Tensor,
    qkv_bias: &Tensor,
    gate_proj_bias: &Tensor,
    up_proj_bias: &Tensor,
    gate_weight: &Tensor,
    up_weight: &Tensor,
    prefix_k: &Tensor,
    prefix_v: &Tensor,
    position_offset: usize,
    kv_cache_offset: usize,
    kv_key_count: usize,
    rms_eps: f32,
    rope_theta: f32,
) -> Result<(Bf16RefActionLayerOutput, Bf16ActionLayerInternals)> {
    let normalized = match attention_normalized {
        Some(value) => value.clone(),
        None => rms_bf16(ctx, input, attention_scale_weight, rms_eps)?,
    };
    let normalized_capture = normalized.clone();
    let qkv = gemm_bias_bf16(ctx, policies, &normalized, &weights.qkv.weight, qkv_bias)?;
    let qkv_capture = qkv.clone();
    let q = split_qkv_rope_into_cache_bf16(
        ctx,
        &qkv,
        None,
        config.num_heads,
        config.num_kv_heads,
        config.head_dim,
        rope_theta,
        position_offset,
        prefix_k,
        prefix_v,
        kv_cache_offset,
    )?;
    let seq_len = input.shape().dims()[0];
    let kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim;
    let k_post_rope = slice_kv_rows_bf16(prefix_k, kv_cache_offset, seq_len, kv_heads, head_dim)?;
    let v = slice_kv_rows_bf16(prefix_v, kv_cache_offset, seq_len, kv_heads, head_dim)?;
    let attention = mqa_bf16(
        ctx,
        policies,
        &q,
        prefix_k,
        prefix_v,
        kv_key_count,
    )?
    .reshape(vec![
        input.shape().dims()[0],
        config.num_heads * config.head_dim,
    ])?;
    let q_post_rope = q.clone();
    let attn_out = attention.clone();
    let projected = gemm_bf16(ctx, policies, &attention, &weights.output.weight)?;
    let attn_proj = projected.clone();
    let gated = mul_bf16(ctx, &projected, attention_gate)?;
    let fused = inplace_add_rms_bf16(
        ctx,
        gated,
        input,
        post_attention_scale_weight,
        rms_eps,
    )?;
    let post_attn_hidden = fused.hidden.clone();
    let post_attn_norm = fused.normalized.clone();
    let gate_out = gemm_bias_bf16(ctx, policies, &fused.normalized, gate_weight, gate_proj_bias)?;
    let gelu_out = gelu_bf16(ctx, &gate_out)?;
    let up_out = gemm_bias_bf16(ctx, policies, &fused.normalized, up_weight, up_proj_bias)?;
    let activated = mul_bf16(ctx, &gelu_out, &up_out)?;
    let geglu_out = activated.clone();
    let projected = gemm_bf16(ctx, policies, &activated, &weights.down.weight)?;
    let mlp_proj = projected.clone();
    let gated = mul_bf16(ctx, &projected, mlp_gate)?;
    let fused = inplace_add_rms_bf16(
        ctx,
        gated,
        &fused.hidden,
        next_scale_weight,
        rms_eps,
    )?;
    Ok((
        Bf16RefActionLayerOutput {
            hidden: fused.hidden.clone(),
            next_normalized: fused.normalized.clone(),
        },
        Bf16ActionLayerInternals {
            normalized: normalized_capture,
            qkv: qkv_capture,
            q_post_rope,
            k_post_rope,
            v,
            attn_out,
            attn_proj,
            post_attn_hidden,
            post_attn_norm,
            geglu_out,
            mlp_proj,
            mlp_hidden: fused.hidden,
            mlp_next_norm: fused.normalized,
        },
    ))
}

/// Slice `seq_len` rows starting at `offset` from a KV cache tensor.
/// Cache is 2D [cache_rows, kv_heads * head_dim]; output is 3D
/// [seq_len, kv_heads, head_dim]. The returned tensor shares storage with
/// `cache` (a view), so callers must dump it before any subsequent write to
/// the same cache region. In our dump path, this is safe because each layer
/// has its own KV cache (`prefix.keys[index]`), and we only dump layer 0
/// before any other layer writes to its own cache.
fn slice_kv_rows_bf16(
    cache: &Tensor,
    offset: usize,
    seq_len: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let dims = cache.shape().dims();
    if dims.len() != 2 || dims[1] != kv_heads * head_dim {
        return Err(Error::Other(format!(
            "slice_kv_rows_bf16: cache shape mismatch, got {:?}, expected [_, {}]",
            dims,
            kv_heads * head_dim
        )));
    }
    if offset + seq_len > dims[0] {
        return Err(Error::Other(format!(
            "slice_kv_rows_bf16: offset+seq_len exceeds cache rows: {}+{} > {}",
            offset, seq_len, dims[0]
        )));
    }
    let row_bytes = kv_heads * head_dim * DType::BF16.size_in_bytes();
    let start_byte = offset * row_bytes;
    let bytes = seq_len * row_bytes;
    DeviceBuffer::from_tensor(cache)
        .and_then(|buffer| buffer.view(start_byte, bytes))
        .and_then(|buffer| {
            buffer.as_tensor(
                Shape::new(vec![seq_len, kv_heads, head_dim]),
                DType::BF16,
            )
        })
        .map_err(|e| Error::Other(format!("slice_kv_rows_bf16: {e}")))
}

pub fn vision_patch_embed_bf16(
    ctx: &Context,
    weights: &Bf16LinearWeights,
    position_embedding: &Tensor,
    patches: &Tensor,
    patches_per_view: usize,
) -> Result<Tensor> {
    vision_patch_embed_bf16_with_policies(
        ctx,
        &L3Policies::default(),
        weights,
        position_embedding,
        patches,
        patches_per_view,
    )
}

fn vision_patch_embed_bf16_with_policies(
    ctx: &Context,
    policies: &L3Policies,
    weights: &Bf16LinearWeights,
    position_embedding: &Tensor,
    patches: &Tensor,
    patches_per_view: usize,
) -> Result<Tensor> {
    let projection = gemm_bf16(ctx, policies, patches, &weights.weight)?;
    add_position_bf16(
        ctx,
        &projection,
        weights.bias.as_ref(),
        position_embedding,
        patches_per_view,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn vision_layer_bf16(
    ctx: &Context,
    weights: &Bf16DeviceVisionBlock,
    input: &Tensor,
    patches_per_view: usize,
    heads: usize,
    head_dim: usize,
    layer_norm_eps: f32,
) -> Result<Tensor> {
    vision_layer_bf16_with_policies(
        ctx,
        &L3Policies::default(),
        weights,
        input,
        patches_per_view,
        heads,
        head_dim,
        layer_norm_eps,
    )
}

#[allow(clippy::too_many_arguments)]
fn vision_layer_bf16_with_policies(
    ctx: &Context,
    policies: &L3Policies,
    weights: &Bf16DeviceVisionBlock,
    input: &Tensor,
    patches_per_view: usize,
    heads: usize,
    head_dim: usize,
    layer_norm_eps: f32,
) -> Result<Tensor> {
    let normalized = layer_bf16(
        ctx,
        input,
        &weights.norm1.weight,
        &weights.norm1.bias,
        layer_norm_eps,
    )?;
    let qkv = gemm_bf16(ctx, policies, &normalized, &weights.qkv.weight)?;
    let qkv = split_qkv_bias_bf16(ctx, &qkv, weights.qkv.bias.as_ref(), heads, head_dim)?;
    let attention = mha_bf16(ctx, policies, &qkv.q, &qkv.k, &qkv.v, patches_per_view)?
        .reshape(vec![input.shape().dims()[0], heads * head_dim])?;
    // Note: aclnnAddLayerNorm fusion attempted but produces NaN in graph
    // capture mode. Using the 3-op bias_residual_layer path for now.
    let projection = gemm_bf16(ctx, policies, &attention, &weights.output.weight)?;
    let fused = bias_residual_layer_bf16(
        ctx,
        &projection,
        weights.output.bias.as_ref(),
        input,
        &weights.norm2.weight,
        &weights.norm2.bias,
        layer_norm_eps,
    )?;
    let activation = gemm_bf16(ctx, policies, &fused.normalized, &weights.fc1.weight)?;
    let activation = bias_activation_bf16(
        ctx,
        &activation,
        weights.fc1.bias.as_ref(),
        PointwiseActivation::Gelu,
    )?;
    let projection = gemm_bf16(ctx, policies, &activation, &weights.fc2.weight)?;
    bias_residual_bf16(ctx, &projection, weights.fc2.bias.as_ref(), &fused.hidden)
}

trait QkvViews {
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

// ── Backbone: owns weights and runs the full model schedule ───────────

pub(in crate::ascend_pi05::model) mod backbone {
    use super::*;
    use crate::ascend_pi05::backend::{Context, DeviceBuffer};
    use crate::ascend_pi05::weights::*;
    use crate::ascend_pi05::Pi05Config;
    use apxinf_core::{DType, Error, Result, Tensor};
    use std::sync::Arc;

    /// Extract the raw device pointer from a tensor (for aclnn FFI calls).
    fn device_ptr_of(tensor: &Tensor) -> usize {
        match tensor.storage() {
            apxinf_core::Storage::Gpu { handle, .. } => handle.ptr(),
            _ => 0,
        }
    }

    pub struct Bf16RefPrefixKvCache {
        pub keys: Vec<Tensor>,
        pub values: Vec<Tensor>,
        pub tokens: usize,
        /// RoPE position offset for the action expert.
        /// Defaults to `tokens` (total prefix rows) in original mode.
        /// When `valid_prefix_tokens` is provided via `VlaMetadata`, this is
        /// set to that value to align with LeRobot's masked-token semantics.
        pub position_offset: usize,
    }

    /// Per-step modulation tables for the action expert, matching the
    /// torch_npu 31ms graph layout.
    ///
    ///  (= 1 + scale) is the RmsNorm gamma.  is the
    /// runtime element-wise gate.  is **not** stored here — it is
    /// folded into the downstream Linear biases (, ,
    /// , ) during precompute, eliminating the
    /// runtime  op (matching torch_npu's
    /// ).
    pub struct Bf16RefStepModulation {
        /// Per-layer pre-attn RmsNorm gamma = 1 + input_scale. [hidden]
        pub attention_scale_weight: Vec<Tensor>,
        /// Per-layer attn output gate. [hidden]
        pub attention_gate: Vec<Tensor>,
        /// Per-layer post-attn InplaceAddRmsNorm weight = 1 + post_scale. [hidden]
        pub post_attention_scale_weight: Vec<Tensor>,
        /// Per-layer MLP output gate. [hidden]
        pub mlp_gate: Vec<Tensor>,
        /// Final norm RmsNorm gamma = 1 + final_scale. [hidden]
        pub final_norm_scale_weight: Tensor,
        /// Per-layer QKV bias = qkv_weight^T @ input_shift (+ orig qkv bias).
        /// Folded shift eliminates runtime Add(normed, shift).
        pub qkv_bias: Vec<Tensor>,
        /// Per-layer gate_proj bias = gate_weight^T @ post_shift.
        pub gate_proj_bias: Vec<Tensor>,
        /// Per-layer up_proj bias = up_weight^T @ post_shift.
        pub up_proj_bias: Vec<Tensor>,
        /// action_out bias = action_out_weight^T @ final_shift + orig bias.
        pub action_out_bias: Tensor,
    }

    pub struct Bf16RefBlocks {
        pub(in crate::ascend_pi05::model) backend: Arc<Context>,
        pub(in crate::ascend_pi05::model) config: Arc<Pi05Config>,
        pub(in crate::ascend_pi05::model) weights: Arc<Bf16Weights>,
        pub(in crate::ascend_pi05::model) policies: L3Policies,
    }

    impl Bf16RefBlocks {
        pub(in crate::ascend_pi05) fn new(
            backend: Arc<Context>,
            config: Arc<Pi05Config>,
            weights: Arc<Bf16Weights>,
            policies: L3Policies,
        ) -> Result<Self> {
            config.validate()?;
            if weights.vision_layers.len() != config.vision_depth
                || weights.language_layers.len() != config.language.depth
                || weights.action_layers.len() != config.action_expert.depth
            {
                return Err(Error::Other(
                    "π0.5 BF16 device weight depth mismatch".into(),
                ));
            }
            Ok(Self {
                backend,
                config,
                weights,
                policies,
            })
        }

        fn ctx(&self) -> &Context {
            &self.backend
        }

        pub fn encode_vision_with_dump(
            &self,
            patches: &Tensor,
        ) -> Result<(Tensor, Vec<Tensor>)> {
            if patches.dtype() != DType::BF16 {
                return Err(Error::DTypeMismatch {
                    expected: DType::BF16,
                    got: patches.dtype(),
                });
            }
            let mut hidden = vision_patch_embed_bf16_with_policies(
                self.ctx(),
                &self.policies,
                &self.weights.patch_embedding,
                &self.weights.position_embedding,
                patches,
                self.config.patches_per_view(),
            )?;
            let mut layer_outputs = Vec::with_capacity(self.config.vision_depth + 1);
            layer_outputs.push(hidden.clone());
            for layer in &self.weights.vision_layers {
                hidden = vision_layer_bf16_with_policies(
                    self.ctx(),
                    &self.policies,
                    layer,
                    &hidden,
                    self.config.patches_per_view(),
                    self.config.vision_heads,
                    self.config.vision_head_dim,
                    self.config.layer_norm_eps,
                )?;
                layer_outputs.push(hidden.clone());
            }
            let hidden = layer_bf16(
                self.ctx(),
                &hidden,
                &self.weights.vision_post_norm.weight,
                &self.weights.vision_post_norm.bias,
                self.config.layer_norm_eps,
            )?;
            // Fused path: gemm_bias (1 op) vs gemm + bias_activation (2 ops).
            let output = match self.weights.multimodal_projector.bias.as_ref() {
                Some(bias) => gemm_bias_bf16(
                    self.ctx(),
                    &self.policies,
                    &hidden,
                    &self.weights.multimodal_projector.weight,
                    bias,
                )?,
                None => gemm_bf16(
                    self.ctx(),
                    &self.policies,
                    &hidden,
                    &self.weights.multimodal_projector.weight,
                )?,
            };
            Ok((output, layer_outputs))
        }

        pub fn prefix_forward_with_dump(
            &self,
            prefix: &Tensor,
            position_offset: Option<usize>,
        ) -> Result<(Bf16RefPrefixKvCache, Vec<Tensor>)> {
            let mut hidden = prefix.clone();
            let mut keys = Vec::with_capacity(self.config.language.depth);
            let mut values = Vec::with_capacity(self.config.language.depth);
            let mut layer_outputs = Vec::with_capacity(self.config.language.depth);
            for (index, layer) in self.weights.language_layers.iter().enumerate() {
                let output = language_layer_bf16_with_policies(
                    self.ctx(),
                    &self.policies,
                    self.config.language,
                    layer,
                    &hidden,
                    index + 1 < self.config.language.depth,
                    0,
                    self.config.rms_norm_eps,
                    self.config.rope_theta,
                )?;
                hidden = output.hidden;
                layer_outputs.push(hidden.clone());
                let cache_rows = prefix.shape().dims()[0] + self.config.action_horizon;
                keys.push(l3_reserve_prefix(self.ctx(), &output.key, cache_rows)?);
                values.push(l3_reserve_prefix(self.ctx(), &output.value, cache_rows)?);
            }
            let total = prefix.shape().dims()[0];
            Ok((
                Bf16RefPrefixKvCache {
                    keys,
                    values,
                    tokens: total,
                    position_offset: position_offset.unwrap_or(total),
                },
                layer_outputs,
            ))
        }

        pub fn denoise_step_with_modulation_and_dump(
            &self,
            state: &Tensor,
            modulation: &Bf16RefStepModulation,
            prefix: &Bf16RefPrefixKvCache,
            dt: f32,
        ) -> Result<(Tensor, Vec<Tensor>, Option<Bf16ActionLayerInternals>)> {
            if prefix.keys.len() != self.config.action_expert.depth
                || prefix.values.len() != self.config.action_expert.depth
                || modulation.attention_scale_weight.len() != self.config.action_expert.depth
                || modulation.post_attention_scale_weight.len() != self.config.action_expert.depth
            {
                return Err(Error::Other(
                    "PI0.5 BF16 prefix/modulation depth mismatch".into(),
                ));
            }
            let mut hidden = match self.weights.action_in.bias.as_ref() {
                Some(bias) => gemm_bias_bf16(
                    self.ctx(),
                    &self.policies,
                    state,
                    &self.weights.action_in.weight,
                    bias,
                )?,
                None => gemm_bf16(
                    self.ctx(),
                    &self.policies,
                    state,
                    &self.weights.action_in.weight,
                )?,
            };
            let mut layer_outputs = Vec::with_capacity(self.config.action_expert.depth + 1);
            layer_outputs.push(hidden.clone());
            let mut attention_normalized = None;
            let mut first_block_internals: Option<Bf16ActionLayerInternals> = None;
            for index in 0..self.config.action_expert.depth {
                let layer = &self.weights.action_layers[index];
                let next_scale_weight = if index + 1 < self.config.action_expert.depth {
                    &modulation.attention_scale_weight[index + 1]
                } else {
                    &modulation.final_norm_scale_weight
                };
                let gate_w = &layer.gate;
                let up_w = &layer.up;
                if index == 0 {
                    let (output, internals) = action_layer_bf16_with_policies_dump(
                        self.ctx(),
                        &self.policies,
                        self.config.action_expert,
                        layer,
                        &hidden,
                        attention_normalized.as_ref(),
                        &modulation.attention_scale_weight[index],
                        &modulation.attention_gate[index],
                        &modulation.post_attention_scale_weight[index],
                        &modulation.mlp_gate[index],
                        next_scale_weight,
                        &modulation.qkv_bias[index],
                        &modulation.gate_proj_bias[index],
                        &modulation.up_proj_bias[index],
                        &gate_w.weight,
                        &up_w.weight,
                        &prefix.keys[index],
                        &prefix.values[index],
                        prefix.position_offset,
                        prefix.tokens,
                        prefix.tokens + self.config.action_horizon,
                        self.config.rms_norm_eps,
                        self.config.rope_theta,
                    )?;
                    hidden = output.hidden;
                    layer_outputs.push(hidden.clone());
                    attention_normalized = Some(output.next_normalized);
                    first_block_internals = Some(internals);
                } else {
                    let output = action_layer_bf16_with_policies(
                        self.ctx(),
                        &self.policies,
                        self.config.action_expert,
                        layer,
                        &hidden,
                        attention_normalized.as_ref(),
                        &modulation.attention_scale_weight[index],
                        &modulation.attention_gate[index],
                        &modulation.post_attention_scale_weight[index],
                        &modulation.mlp_gate[index],
                        next_scale_weight,
                        &modulation.qkv_bias[index],
                        &modulation.gate_proj_bias[index],
                        &modulation.up_proj_bias[index],
                        &gate_w.weight,
                        &up_w.weight,
                        &prefix.keys[index],
                        &prefix.values[index],
                        prefix.position_offset,
                        prefix.tokens,
                        prefix.tokens + self.config.action_horizon,
                        self.config.rms_norm_eps,
                        self.config.rope_theta,
                    )?;
                    hidden = output.hidden;
                    layer_outputs.push(hidden.clone());
                    attention_normalized = Some(output.next_normalized);
                }
            }
            let hidden = attention_normalized.ok_or_else(|| {
                Error::Other("PI0.5 action expert must contain at least one layer".into())
            })?;
            let velocity = gemm_bias_bf16(
                self.ctx(),
                &self.policies,
                &hidden,
                &self.weights.action_out.weight,
                &modulation.action_out_bias,
            )?;
            let next_state = euler_update_bf16(self.ctx(), state, &velocity, dt)?;
            Ok((next_state, layer_outputs, first_block_internals))
        }

        pub fn encode_vision(&self, patches: &Tensor) -> Result<Tensor> {
            if patches.dtype() != DType::BF16 {
                return Err(Error::DTypeMismatch {
                    expected: DType::BF16,
                    got: patches.dtype(),
                });
            }
            let mut hidden = vision_patch_embed_bf16_with_policies(
                self.ctx(),
                &self.policies,
                &self.weights.patch_embedding,
                &self.weights.position_embedding,
                patches,
                self.config.patches_per_view(),
            )?;
            for layer in &self.weights.vision_layers {
                hidden = vision_layer_bf16_with_policies(
                    self.ctx(),
                    &self.policies,
                    layer,
                    &hidden,
                    self.config.patches_per_view(),
                    self.config.vision_heads,
                    self.config.vision_head_dim,
                    self.config.layer_norm_eps,
                )?;
            }
            let hidden = layer_bf16(
                self.ctx(),
                &hidden,
                &self.weights.vision_post_norm.weight,
                &self.weights.vision_post_norm.bias,
                self.config.layer_norm_eps,
            )?;
            // Fused path: gemm_bias (1 op) vs gemm + bias_activation (2 ops).
            match self.weights.multimodal_projector.bias.as_ref() {
                Some(bias) => gemm_bias_bf16(
                    self.ctx(),
                    &self.policies,
                    &hidden,
                    &self.weights.multimodal_projector.weight,
                    bias,
                ),
                None => gemm_bf16(
                    self.ctx(),
                    &self.policies,
                    &hidden,
                    &self.weights.multimodal_projector.weight,
                ),
            }
        }

        pub fn embed_prefix(
            &self,
            vision_tokens: &Tensor,
            token_ids: &DeviceBuffer,
            token_count: usize,
        ) -> Result<Tensor> {
            if token_count == 0 || token_count > self.config.max_token_len {
                return Err(Error::Other(format!(
                    "π0.5 token count must be in 1..={}, got {token_count}",
                    self.config.max_token_len
                )));
            }
            let width = self.weights.token_embedding.shape().dims()[1];
            let mut language = self
                .ctx()
                .allocate_output(Shape::new(vec![token_count, width]), DType::BF16)?;
            let mut args = GatherArgs::new(
                GatherSemantic::EmbeddingLookup,
                &self.weights.token_embedding,
                &mut language,
            );
            args.ids = Some(token_ids);
            args.vocab_size = self.weights.token_embedding.shape().dims()[0];
            l3_gather(self.ctx(), args)?;
            l3_concat_rows(self.ctx(), vision_tokens, &language)
        }

        pub fn prefix_forward(
            &self,
            prefix: &Tensor,
            position_offset: Option<usize>,
        ) -> Result<Bf16RefPrefixKvCache> {
            let mut hidden = prefix.clone();
            let mut keys = Vec::with_capacity(self.config.language.depth);
            let mut values = Vec::with_capacity(self.config.language.depth);
            for (index, layer) in self.weights.language_layers.iter().enumerate() {
                let output = language_layer_bf16_with_policies(
                    self.ctx(),
                    &self.policies,
                    self.config.language,
                    layer,
                    &hidden,
                    index + 1 < self.config.language.depth,
                    0,
                    self.config.rms_norm_eps,
                    self.config.rope_theta,
                )?;
                hidden = output.hidden;
                let cache_rows = prefix.shape().dims()[0] + self.config.action_horizon;
                keys.push(l3_reserve_prefix(self.ctx(), &output.key, cache_rows)?);
                values.push(l3_reserve_prefix(self.ctx(), &output.value, cache_rows)?);
            }
            let total = prefix.shape().dims()[0];
            Ok(Bf16RefPrefixKvCache {
                keys,
                values,
                tokens: total,
                position_offset: position_offset.unwrap_or(total),
            })
        }

        fn conditioning(&self, time_embedding: &Tensor) -> Result<Tensor> {
            let hidden = gemm_bf16(
                self.ctx(),
                &self.policies,
                time_embedding,
                &self.weights.time_mlp_in.weight,
            )?;
            let hidden = bias_activation_bf16(
                self.ctx(),
                &hidden,
                self.weights.time_mlp_in.bias.as_ref(),
                PointwiseActivation::Silu,
            )?;
            let output = gemm_bf16(
                self.ctx(),
                &self.policies,
                &hidden,
                &self.weights.time_mlp_out.weight,
            )?;
            bias_activation_bf16(
                self.ctx(),
                &output,
                self.weights.time_mlp_out.bias.as_ref(),
                PointwiseActivation::Silu,
            )
        }

        /// Compute modulation vector = dense(cond) + bias, then reshape to 1D.
        /// Output shape: [3 * hidden] (scale, shift, gate concatenated).
        fn modulation_raw(&self, conditioning: &Tensor, weights: &Bf16LinearWeights) -> Result<Tensor> {
            let projected = gemm_bf16(self.ctx(), &self.policies, conditioning, &weights.weight)?;
            let modulation = bias_activation_bf16(
                self.ctx(),
                &projected,
                weights.bias.as_ref(),
                PointwiseActivation::None,
            )?;
            modulation.reshape(vec![modulation.numel()])
        }

        /// Split a 1D modulation [3*hidden] into (scale, shift, gate) each [hidden].
        fn split_modulation(&self, modulation: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
            let cols = modulation.shape().dims()[0];
            if cols % 3 != 0 {
                return Err(Error::Other(format!(
                    "modulation length {} not divisible by 3", cols
                )));
            }
            let hidden = cols / 3;
            let bytes = hidden * DType::BF16.size_in_bytes();
            let scale = bf16_vector_prefix(modulation, hidden)?;
            let shift = DeviceBuffer::from_tensor(modulation)
                .and_then(|b| b.view(bytes, bytes))
                .and_then(|b| b.as_tensor(Shape::new(vec![hidden]), DType::BF16))
                .map_err(|e| Error::Other(format!("split_modulation shift: {e}")))?;
            let gate = DeviceBuffer::from_tensor(modulation)
                .and_then(|b| b.view(2 * bytes, bytes))
                .and_then(|b| b.as_tensor(Shape::new(vec![hidden]), DType::BF16))
                .map_err(|e| Error::Other(format!("split_modulation gate: {e}")))?;
            Ok((scale, shift, gate))
        }

        /// Compute scale_weight = 1 + scale via aclnnAdds.
        fn add_one_bf16(&self, input: &Tensor) -> Result<Tensor> {
            let out = self.ctx().allocate_output(input.shape().clone(), DType::BF16)?;
            crate::ascend_pi05::backend::ops::aclnn_adds(
                device_ptr_of(input),
                device_ptr_of(&out),
                input.shape(),
                1.0,
                DType::BF16,
                self.ctx().raw_stream(),
            )?;
            Ok(out)
        }

        /// Fold shift into Linear bias: bias = shift @ weight [+ orig_bias].
        /// shift: [hidden], weight: [in, out] -> bias: [out]
        fn fold_shift_to_bias(
            &self,
            shift: &Tensor,
            weight: &Tensor,
            orig_bias: Option<&Tensor>,
        ) -> Result<Tensor> {
            let shift_2d = shift.reshape(Shape::new(vec![1, shift.shape().dims()[0]]))?;
            let folded = gemm_bf16(self.ctx(), &self.policies, &shift_2d, weight)?;
            let bias = folded.reshape(Shape::new(vec![folded.shape().dims()[1]]))?;
            if let Some(orig) = orig_bias {
                let out = self.ctx().allocate_output(bias.shape().clone(), DType::BF16)?;
                crate::ascend_pi05::backend::ops::aclnn_add(
                    device_ptr_of(&bias),
                    device_ptr_of(orig),
                    device_ptr_of(&out),
                    bias.shape(),
                    orig.shape(),
                    1.0,
                    DType::BF16,
                    self.ctx().raw_stream(),
                )?;
                Ok(out)
            } else {
                Ok(bias)
            }
        }

        fn prepare_step_modulation(&self, time_embedding: &Tensor) -> Result<Bf16RefStepModulation> {
            let conditioning = self.conditioning(time_embedding)?;
            let depth = self.config.action_expert.depth;
            let mut attention_scale_weight = Vec::with_capacity(depth);
            let mut attention_gate = Vec::with_capacity(depth);
            let mut post_attention_scale_weight = Vec::with_capacity(depth);
            let mut mlp_gate = Vec::with_capacity(depth);
            let mut qkv_bias = Vec::with_capacity(depth);
            let mut gate_proj_bias = Vec::with_capacity(depth);
            let mut up_proj_bias = Vec::with_capacity(depth);
            for layer in &self.weights.action_layers {
                let raw = self.modulation_raw(&conditioning, &layer.input_modulation)?;
                let (scale, shift, gate) = self.split_modulation(&raw)?;
                attention_scale_weight.push(self.add_one_bf16(&scale)?);
                attention_gate.push(gate);
                qkv_bias.push(self.fold_shift_to_bias(
                    &shift, &layer.qkv.weight, layer.qkv.bias.as_ref())?);

                let raw = self.modulation_raw(&conditioning, &layer.post_attention_modulation)?;
                let (scale, shift, gate) = self.split_modulation(&raw)?;
                post_attention_scale_weight.push(self.add_one_bf16(&scale)?);
                mlp_gate.push(gate);
                let gate_w = &layer.gate;
                let up_w = &layer.up;
                gate_proj_bias.push(self.fold_shift_to_bias(
                    &shift, &gate_w.weight, gate_w.bias.as_ref())?);
                up_proj_bias.push(self.fold_shift_to_bias(
                    &shift, &up_w.weight, up_w.bias.as_ref())?);
            }
            let raw = self.modulation_raw(&conditioning, &self.weights.action_final_modulation)?;
            let (scale, shift, _gate) = self.split_modulation(&raw)?;
            let final_norm_scale_weight = self.add_one_bf16(&scale)?;
            let action_out_bias = self.fold_shift_to_bias(
                &shift, &self.weights.action_out.weight,
                self.weights.action_out.bias.as_ref())?;
            Ok(Bf16RefStepModulation {
                attention_scale_weight,
                attention_gate,
                post_attention_scale_weight,
                mlp_gate,
                final_norm_scale_weight,
                qkv_bias,
                gate_proj_bias,
                up_proj_bias,
                action_out_bias,
            })
        }

        pub(super) fn prepare_all_modulation(
            &self,
            time_embeddings: &[Tensor],
        ) -> Result<Vec<Bf16RefStepModulation>> {
            let steps = time_embeddings.len();
            if steps != self.config.num_flow_steps {
                return Err(Error::Other(format!(
                    "π0.5 expected {} timestep embeddings, got {}",
                    self.config.num_flow_steps,
                    steps
                )));
            }
            // Stack the [1,width] per-step embeddings into a single
            // [steps,width] tensor so all conditioning / modulation /
            // fold gemms run as batchN (M=steps) instead of repeated batch=1
            // matmuls — matching torch_npu's MatMulV3 which has zero batch=1
            // gemms. The per-step outputs are recovered via zero-copy row
            // views, so `Bf16RefStepModulation` layout is unchanged and the
            // computation is mathematically identical to the per-step path.
            let mut embs = time_embeddings[0].clone();
            for e in &time_embeddings[1..] {
                embs = l3_concat_rows(self.ctx(), &embs, e)?;
            }
            // Batched conditioning: [steps, width] -> [steps, width].
            let cond = self.conditioning_batched(&embs)?;
            let depth = self.config.action_expert.depth;
            let hidden = embs.shape().dims()[1];

            // Per-layer, per-step accumulation.
            let mut attention_scale_weight = vec![Vec::with_capacity(depth); steps];
            let mut attention_gate = vec![Vec::with_capacity(depth); steps];
            let mut post_attention_scale_weight = vec![Vec::with_capacity(depth); steps];
            let mut mlp_gate = vec![Vec::with_capacity(depth); steps];
            let mut qkv_bias = vec![Vec::with_capacity(depth); steps];
            let mut gate_proj_bias = vec![Vec::with_capacity(depth); steps];
            let mut up_proj_bias = vec![Vec::with_capacity(depth); steps];
            let mut final_norm_scale_weight = vec![Vec::with_capacity(1); steps];
            let mut action_out_bias = vec![Vec::with_capacity(1); steps];

            for layer in &self.weights.action_layers {
                // input modulation raw [steps, 3*hidden]; recover each step's
                // row ([3*hidden], row-major contiguous) then use the 1D split.
                let raw = self.modulation_raw_batched(&cond, &layer.input_modulation)?;
                for s in 0..steps {
                    let row = self.row_tensor(&raw, s, vec![3 * hidden])?;
                    let (scale, _shift, gate) = self.split_modulation(&row)?;
                    let scale_w = self.add_one_bf16(&scale)?;
                    attention_scale_weight[s].push(scale_w);
                    attention_gate[s].push(gate);
                }
                let qkv_bs = self.fold_shift_batched(
                    &raw,
                    &layer.qkv.weight,
                    layer.qkv.bias.as_ref(),
                    hidden,
                )?;
                for s in 0..steps {
                    qkv_bias[s].push(qkv_bs[s].clone());
                }

                // post-attention modulation raw [steps, 3*hidden]
                let raw = self.modulation_raw_batched(&cond, &layer.post_attention_modulation)?;
                for s in 0..steps {
                    let row = self.row_tensor(&raw, s, vec![3 * hidden])?;
                    let (scale, _shift, gate) = self.split_modulation(&row)?;
                    let scale_w = self.add_one_bf16(&scale)?;
                    post_attention_scale_weight[s].push(scale_w);
                    mlp_gate[s].push(gate);
                }
                let gate_w = &layer.gate;
                let up_w = &layer.up;
                let gate_bs = self.fold_shift_batched(
                    &raw, &gate_w.weight, gate_w.bias.as_ref(), hidden)?;
                let up_bs = self.fold_shift_batched(
                    &raw, &up_w.weight, up_w.bias.as_ref(), hidden)?;
                for s in 0..steps {
                    gate_proj_bias[s].push(gate_bs[s].clone());
                    up_proj_bias[s].push(up_bs[s].clone());
                }
            }
            // final + action_out (per-step independent)
            let raw = self.modulation_raw_batched(&cond, &self.weights.action_final_modulation)?;
            for s in 0..steps {
                let row = self.row_tensor(&raw, s, vec![3 * hidden])?;
                let (scale, _shift, _gate) = self.split_modulation(&row)?;
                let scale_w = self.add_one_bf16(&scale)?;
                final_norm_scale_weight[s].push(scale_w);
            }
            let out_bs = self.fold_shift_batched(
                &raw,
                &self.weights.action_out.weight,
                self.weights.action_out.bias.as_ref(),
                hidden,
            )?;
            for s in 0..steps {
                action_out_bias[s].push(out_bs[s].clone());
            }

            let mut modulations = Vec::with_capacity(steps);
            for s in 0..steps {
                modulations.push(Bf16RefStepModulation {
                    attention_scale_weight: std::mem::take(&mut attention_scale_weight[s]),
                    attention_gate: std::mem::take(&mut attention_gate[s]),
                    post_attention_scale_weight: std::mem::take(
                        &mut post_attention_scale_weight[s],
                    ),
                    mlp_gate: std::mem::take(&mut mlp_gate[s]),
                    final_norm_scale_weight: std::mem::take(&mut final_norm_scale_weight[s])
                        .remove(0),
                    qkv_bias: std::mem::take(&mut qkv_bias[s]),
                    gate_proj_bias: std::mem::take(&mut gate_proj_bias[s]),
                    up_proj_bias: std::mem::take(&mut up_proj_bias[s]),
                    action_out_bias: std::mem::take(&mut action_out_bias[s]).remove(0),
                });
            }
            Ok(modulations)
        }

        // ── Batched (batchN) modulation helpers ──────────────────────────
        // These compute the same math as `conditioning` / `modulation_raw` /
        // `split_modulation` / `add_one_bf16` / `fold_shift_to_bias` but with
        // M = num_flow_steps (10) rows at once, eliminating the batch=1 gemm
        // chain that torch_npu does not have in its graph.

        fn conditioning_batched(&self, embs: &Tensor) -> Result<Tensor> {
            let hidden = gemm_bf16(self.ctx(), &self.policies, embs, &self.weights.time_mlp_in.weight)?;
            let hidden = bias_activation_bf16(
                self.ctx(),
                &hidden,
                self.weights.time_mlp_in.bias.as_ref(),
                PointwiseActivation::Silu,
            )?;
            let output = gemm_bf16(self.ctx(), &self.policies, &hidden, &self.weights.time_mlp_out.weight)?;
            bias_activation_bf16(
                self.ctx(),
                &output,
                self.weights.time_mlp_out.bias.as_ref(),
                PointwiseActivation::Silu,
            )
        }

        /// Batch modulation raw: [steps, hidden] @ [hidden, 3*hidden] -> [steps, 3*hidden].
        fn modulation_raw_batched(
            &self,
            conditioning: &Tensor,
            weights: &Bf16LinearWeights,
        ) -> Result<Tensor> {
            let projected = gemm_bf16(self.ctx(), &self.policies, conditioning, &weights.weight)?;
            bias_activation_bf16(
                self.ctx(),
                &projected,
                weights.bias.as_ref(),
                PointwiseActivation::None,
            )
        }

        /// Zero-copy view of row `r` of a contiguous [R, C] tensor.
        fn row_tensor(
            &self,
            tensor: &Tensor,
            r: usize,
            shape_dims: Vec<usize>,
        ) -> Result<Tensor> {
            let dims = tensor.shape().dims();
            let row_bytes = dims[1] * DType::BF16.size_in_bytes();
            DeviceBuffer::from_tensor(tensor)
                .and_then(|b| b.view(r * row_bytes, row_bytes))
                .and_then(|b| b.as_tensor(Shape::new(shape_dims.clone()), DType::BF16))
                .map_err(|e| Error::Other(format!("row_tensor: {e}")))
        }

        /// Batched fold of a `modulation_raw` output `[steps, 3*hidden]` into the
        /// per-step shift-folded bias vectors, replacing 10 batch=1 gemms with a
        /// single batch=steps gemm (mathematically identical).
        ///
        /// 1. Extract the contiguous shift column `[steps, hidden]` via aclnnSlice
        ///    (same op already used for QKV split; exact copy, no arithmetic).
        /// 2. `folded = shift_batched @ weight` -> `[steps, out]` (one gemm).
        /// 3. Per step: `row_tensor` -> `[out]`, then 1D add `orig_bias` (unchanged
        ///    from the validated per-step math) -> exactly the previous numerics.
        ///
        /// Returns `steps` bias tensors, each `[out]`, in step order.
        fn fold_shift_batched(
            &self,
            raw: &Tensor,
            weight: &Tensor,
            orig_bias: Option<&Tensor>,
            hidden: usize,
        ) -> Result<Vec<Tensor>> {
            let dims = raw.shape().dims();
            let steps = dims[0];
            let out = weight.shape().dims()[1];
            // 1) contiguous shift column [steps, hidden]
            let shift_batched = self.ctx().allocate_output(
                Shape::new(vec![steps, hidden]),
                DType::BF16,
            )?;
            crate::ascend_pi05::backend::ops::aclnn_slice_2d(
                device_ptr_of(raw),
                raw.shape(),
                device_ptr_of(&shift_batched),
                steps,
                hidden,
                hidden as i64,
                DType::BF16,
                self.ctx().raw_stream(),
            )?;
            // 2) single batched fold gemm [steps, out]
            let folded = gemm_bf16(self.ctx(), &self.policies, &shift_batched, weight)?;
            // 3) per-step row + (optional) 1D bias add
            let mut out_bias = Vec::with_capacity(steps);
            for s in 0..steps {
                let bias = self.row_tensor(&folded, s, vec![out])?;
                let bias = match orig_bias {
                    Some(orig) => {
                        let out_t = self
                            .ctx()
                            .allocate_output(bias.shape().clone(), DType::BF16)?;
                        crate::ascend_pi05::backend::ops::aclnn_add(
                            device_ptr_of(&bias),
                            device_ptr_of(orig),
                            device_ptr_of(&out_t),
                            bias.shape(),
                            orig.shape(),
                            1.0,
                            DType::BF16,
                            self.ctx().raw_stream(),
                        )?;
                        out_t
                    }
                    None => bias,
                };
                out_bias.push(bias);
            }
            Ok(out_bias)
        }

        fn denoise_step_with_modulation(
            &self,
            state: &Tensor,
            modulation: &Bf16RefStepModulation,
            prefix: &Bf16RefPrefixKvCache,
            dt: f32,
        ) -> Result<Tensor> {
            if prefix.keys.len() != self.config.action_expert.depth
                || prefix.values.len() != self.config.action_expert.depth
                || modulation.attention_scale_weight.len() != self.config.action_expert.depth
                || modulation.post_attention_scale_weight.len() != self.config.action_expert.depth
            {
                return Err(Error::Other(
                    "PI0.5 BF16 prefix/modulation depth mismatch".into(),
                ));
            }
            let mut hidden = match self.weights.action_in.bias.as_ref() {
                Some(bias) => gemm_bias_bf16(
                    self.ctx(),
                    &self.policies,
                    state,
                    &self.weights.action_in.weight,
                    bias,
                )?,
                None => gemm_bf16(
                    self.ctx(),
                    &self.policies,
                    state,
                    &self.weights.action_in.weight,
                )?,
            };
            let mut attention_normalized = None;
            for index in 0..self.config.action_expert.depth {
                let layer = &self.weights.action_layers[index];
                let next_scale_weight = if index + 1 < self.config.action_expert.depth {
                    &modulation.attention_scale_weight[index + 1]
                } else {
                    &modulation.final_norm_scale_weight
                };
                let gate_w = &layer.gate;
                let up_w = &layer.up;
                let output = action_layer_bf16_with_policies(
                    self.ctx(),
                    &self.policies,
                    self.config.action_expert,
                    layer,
                    &hidden,
                    attention_normalized.as_ref(),
                    &modulation.attention_scale_weight[index],
                    &modulation.attention_gate[index],
                    &modulation.post_attention_scale_weight[index],
                    &modulation.mlp_gate[index],
                    next_scale_weight,
                    &modulation.qkv_bias[index],
                    &modulation.gate_proj_bias[index],
                    &modulation.up_proj_bias[index],
                    &gate_w.weight,
                    &up_w.weight,
                    &prefix.keys[index],
                    &prefix.values[index],
                    prefix.position_offset,
                    prefix.tokens,
                    prefix.tokens + self.config.action_horizon,
                    self.config.rms_norm_eps,
                    self.config.rope_theta,
                )?;
                hidden = output.hidden;
                attention_normalized = Some(output.next_normalized);
            }
            let hidden = attention_normalized.ok_or_else(|| {
                Error::Other("PI0.5 action expert must contain at least one layer".into())
            })?;
            let velocity = gemm_bias_bf16(
                self.ctx(),
                &self.policies,
                &hidden,
                &self.weights.action_out.weight,
                &modulation.action_out_bias,
            )?;
            let updated = euler_update_bf16(self.ctx(), state, &velocity, dt)?;
            Ok(updated)
        }

        pub fn denoise_step(
            &self,
            state: &Tensor,
            time_embedding: &Tensor,
            prefix: &Bf16RefPrefixKvCache,
            dt: f32,
        ) -> Result<Tensor> {
            let modulation = self.prepare_step_modulation(time_embedding)?;
            self.denoise_step_with_modulation(state, &modulation, prefix, dt)
        }
    }

    impl super::super::Blocks for Bf16RefBlocks {
        type Prefix = Bf16RefPrefixKvCache;
        type StepModulation = Bf16RefStepModulation;
        fn config(&self) -> &Pi05Config {
            &self.config
        }
        fn vision(&self, patches: &Tensor, native: bool) -> Result<Tensor> {
            let _ = native;
            self.encode_vision(patches)
        }
        fn embed_prefix(&self, vision: &Tensor, ids: &DeviceBuffer, count: usize) -> Result<Tensor> {
            self.embed_prefix(vision, ids, count)
        }
        fn prefix(&self, input: &Tensor) -> Result<Self::Prefix> {
            self.prefix_forward(input, None)
        }
        fn prefix_with_offset(
            &self,
            input: &Tensor,
            position_offset: Option<usize>,
        ) -> Result<Self::Prefix> {
            self.prefix_forward(input, position_offset)
        }
        fn prepare_modulation(&self, embeddings: &[Tensor]) -> Result<Vec<Self::StepModulation>> {
            self.prepare_all_modulation(embeddings)
        }
        fn eager_modulation(
            &self,
            embeddings: &[Tensor],
        ) -> Result<Option<Vec<Self::StepModulation>>> {
            self.prepare_all_modulation(embeddings).map(Some)
        }
        fn step(
            &self,
            state: &Tensor,
            embedding: &Tensor,
            prefix: &Self::Prefix,
            dt: f32,
        ) -> Result<Tensor> {
            self.denoise_step(state, embedding, prefix, dt)
        }
        fn step_with_modulation(
            &self,
            state: &Tensor,
            modulation: &Self::StepModulation,
            prefix: &Self::Prefix,
            dt: f32,
        ) -> Result<Tensor> {
            self.denoise_step_with_modulation(state, modulation, prefix, dt)
        }
    }
}

impl crate::ascend_pi05::model::PrepareBlocks for backbone::Bf16RefBlocks {
    fn backend(&self) -> &std::sync::Arc<crate::ascend_pi05::backend::Context> {
        &self.backend
    }
    fn workspace_requirements(
        &self,
        _tokens: usize,
    ) -> apxinf_core::Result<crate::ascend_pi05::model::WorkspaceRequirements> {
        // Workspace pool for aclnn operators + output tensor arena.
        // Two uses during graph capture: (1) run_op workspace — alloc +
        // rewind per op, only needs peak single-op size; (2) allocate_output
        // — accumulates without rewind, needs sum of all output tensors
        // baked into the captured graph (graph replay writes to these
        // deterministic addresses).
        // Pool is reused across inferences (model-runner-level session),
        // so aclrtMalloc(capacity) happens ONCE — no HBM fragmentation.
        // Capture-phase pool usage peaks at >4GB with the fused-op path
        // (gemm_bias + InplaceAddRmsNorm allocate larger combined workspaces
        // than the unfused matmul+inplace_add+AddRmsNorm sequence). 6GB gives
        // ample headroom so the pool never overflows (which would return Err
        // per CUDA semantics — no from_raw_ptr fallback, which leaked
        // ~5.6GB/run).
        // Pool is allocated ONCE (model-runner-level session reuse), so 6GB
        // does not cause HBM fragmentation.
        // If this returns Err in practice, increase here.
        Ok(crate::ascend_pi05::model::WorkspaceRequirements {
            bytes: 6 * 1024 * 1024 * 1024,
        })
    }
    fn raw_patch_dtype(&self) -> apxinf_core::DType {
        apxinf_core::DType::BF16
    }
    fn preprocess(
        &self,
        images: &crate::ascend_pi05::backend::DeviceBuffer,
        patches: &Tensor,
        layout: crate::ascend_pi05::AscendPi05ImageLayout,
    ) -> Result<()> {
        let mut patches = patches.clone();
        l3_gather(
            &self.backend,
            GatherArgs::rgb_to_patches(
                images,
                &mut patches,
                GatherPatchGeometry {
                    views: self.config.num_views,
                    image_size: self.config.image_size,
                    patch_size: self.config.patch_size,
                    nhwc: matches!(layout, crate::ascend_pi05::AscendPi05ImageLayout::Nhwc),
                },
            ),
        )
    }
}
