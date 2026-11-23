//! Physical Intelligence π0-FAST vision-language-action model.
//!
//! π0-FAST shares the PaliGemma backbone (SigLIP So400m/14 vision + Gemma-2B
//! text) with π0.5 but replaces flow-matching action generation with
//! autoregressive FAST action-token decoding: the same LM head emits discrete
//! action tokens that a FAST tokenizer maps back to continuous actions.
//!
//! OpenPI defines the model math and LeRobot the checkpoint contract; this
//! module keeps its own architecture code so the family stays independent of
//! `pi05`/`walloss`, composing only model-neutral kernels.

#[cfg(feature = "cuda")]
mod backend;
#[cfg(feature = "cuda")]
mod bf16_executor;
#[cfg(feature = "cuda")]
mod bf16_runtime;
mod bf16_weights;
#[cfg(feature = "cuda")]
mod calibration;
mod config;
mod device_weights;
mod fp8_calibration;
#[cfg(feature = "cuda")]
mod fp8_executor;
#[cfg(feature = "cuda")]
mod fp8_runtime;
mod fp8_weights;
mod static_bf16_weights;
mod static_fp8_weights;
#[cfg(feature = "cuda")]
mod vla_runtime;
mod weights;

#[cfg(feature = "cuda")]
pub use bf16_executor::{
    language_layer_bf16, language_layer_cached_bf16, language_layer_cached_decode_bf16,
    vision_layer_bf16, vision_patch_embed_f32_bf16, Bf16LanguageLayerOutput,
};
#[cfg(feature = "cuda")]
pub use bf16_runtime::Pi0FastBf16Runtime;
pub use bf16_weights::{bf16_to_device, f32_to_device, Bf16LinearWeights};
#[cfg(feature = "cuda")]
pub use calibration::Pi0FastCalibrationObserver;
pub use config::{Pi0FastConfig, Pi0FastLanguageConfig, Pi0FastPerformanceProfile};
pub use fp8_calibration::{
    checkpoint_identity, Pi0FastCalibrationPlan, Pi0FastFp8Calibration, LM_HEAD_SITE,
    MULTIMODAL_PROJECTOR_SITE,
};
#[cfg(feature = "cuda")]
pub use fp8_executor::{
    language_layer_cached_decode_fp8, language_layer_fp8, vision_layer_fp8, Fp8LanguageLayerOutput,
    LanguageLayerScales, Pi0FastFp8Scales, VisionLayerScales,
};
#[cfg(feature = "cuda")]
pub use fp8_runtime::Pi0FastFp8Runtime;
pub use fp8_weights::{Fp8LinearWeights, E4M3_MAX};
#[cfg(feature = "cuda")]
pub use static_bf16_weights::{
    Bf16DeviceLanguageLayer, Bf16DeviceLayerNorm, Bf16DeviceVisionBlock, StaticBf16Pi0FastWeights,
    VisionPatchEmbeddingF32,
};
#[cfg(feature = "cuda")]
pub use static_fp8_weights::{
    Fp8DeviceLanguageLayer, Fp8DeviceVisionBlock, StaticFp8Pi0FastWeights,
};
#[cfg(feature = "cuda")]
pub use vla_runtime::Pi0FastVlaRuntime;
pub use weights::{
    GemmaAttentionWeights, GemmaMlpWeights, LanguageLayerWeights, LayerNormWeights, LinearWeights,
    Pi0FastWeights, VisionBlockWeights, VisionWeights,
};

/// Device table mapping each pruned LM-head column back to a global token id.
///
/// The decode loop, the tied embedding lookup and the caller's `stop_token` all
/// speak global token ids, so the argmax kernel writes through this table
/// instead of handing back a local column index.
#[cfg(feature = "cuda")]
pub(super) fn action_head_remap(
    config: &Pi0FastConfig,
    ctx: &backend::Context,
) -> apxinf_core::Result<apxinf_core::Tensor> {
    let columns = config.action_head_columns()?;
    let bytes: Vec<u8> = columns
        .iter()
        .flat_map(|column| (*column as u32).to_ne_bytes())
        .collect();
    let buffer = backend::DeviceBuffer::alloc(bytes.len(), ctx.device_id())
        .map_err(apxinf_core::Error::Cuda)?;
    buffer
        .copy_from_host_on(&bytes, ctx.stream())
        .map_err(apxinf_core::Error::Cuda)?;
    buffer
        .as_tensor(
            apxinf_core::Shape::new(vec![columns.len()]),
            apxinf_core::DType::I32,
        )
        .map_err(apxinf_core::Error::Cuda)
}

#[cfg(feature = "cuda")]
fn attention_rank3(tensor: &apxinf_core::Tensor, what: &str) -> apxinf_core::Result<[usize; 3]> {
    let shape = tensor.shape().dims();
    if shape.len() != 3 || shape.contains(&0) {
        return Err(apxinf_core::Error::Other(format!(
            "PI0-FAST {what} requires non-empty [tokens,heads,head_dim] storage"
        )));
    }
    Ok([shape[0], shape[1], shape[2]])
}

#[cfg(feature = "cuda")]
fn attention_mqa_kv_shape(
    tensor: &apxinf_core::Tensor,
    what: &str,
) -> apxinf_core::Result<[usize; 3]> {
    let shape = tensor.shape().dims();
    match shape {
        [tokens, head_dim] if *tokens != 0 && *head_dim != 0 => {
            Ok([*tokens, 1, *head_dim])
        }
        [tokens, heads, head_dim]
            if *tokens != 0 && *heads != 0 && *head_dim != 0 =>
        {
            Ok([*tokens, *heads, *head_dim])
        }
        _ => Err(apxinf_core::Error::Other(format!(
            "PI0-FAST {what} requires non-empty [tokens,head_dim] or [tokens,heads,head_dim] storage"
        ))),
    }
}

#[cfg(feature = "cuda")]
pub(super) struct AttentionQkv {
    pub q: apxinf_core::Tensor,
    pub k: apxinf_core::Tensor,
    pub v: apxinf_core::Tensor,
}

#[cfg(feature = "cuda")]
pub(super) fn split_qkv_bias(
    ctx: &backend::Context,
    qkv: &apxinf_core::Tensor,
    bias: Option<&apxinf_core::Tensor>,
    heads: usize,
    head_dim: usize,
) -> apxinf_core::Result<AttentionQkv> {
    let dims = qkv.shape().dims();
    if dims.len() != 2
        || dims[0] == 0
        || heads == 0
        || head_dim == 0
        || dims[1] != 3 * heads * head_dim
    {
        return Err(apxinf_core::Error::Other(
            "PI0-FAST packed QKV split shape mismatch".into(),
        ));
    }
    let shape = apxinf_core::Shape::new(vec![dims[0], heads, head_dim]);
    let mut q = ctx.allocate_output(shape.clone(), qkv.dtype())?;
    let mut k = ctx.allocate_output(shape.clone(), qkv.dtype())?;
    let mut v = ctx.allocate_output(shape, qkv.dtype())?;
    backend::ops::rope(
        ctx,
        backend::ops::RopeArgs {
            semantic: backend::ops::RopeSemantic::SplitQkvBias,
            qkv,
            bias,
            q: &mut q,
            k: &mut k,
            v: &mut v,
            q_heads: heads,
            kv_heads: heads,
            head_dim,
            theta: 0.0,
            position_offset: 0,
            kv_output_offset: 0,
        },
    )?;
    Ok(AttentionQkv { q, k, v })
}

#[cfg(feature = "cuda")]
pub(super) fn attention_mqa(
    ctx: &backend::Context,
    query: &apxinf_core::Tensor,
    key: &apxinf_core::Tensor,
    value: &apxinf_core::Tensor,
    valid_key_tokens: usize,
) -> apxinf_core::Result<apxinf_core::Tensor> {
    let [query_tokens, query_heads, head_dim] = attention_rank3(query, "MQA query")?;
    // Prefix K/V are rank-3 tensors. The persistent decode cache intentionally
    // flattens its single KV head to a row-major [capacity, head_dim] matrix;
    // both layouts carry the same MQA semantic and normalize to one L3 shape.
    let [key_capacity, kv_heads, key_head_dim] = attention_mqa_kv_shape(key, "MQA key")?;
    if value.shape() != key.shape()
        || query.dtype() != apxinf_core::DType::BF16
        || key.dtype() != query.dtype()
        || value.dtype() != query.dtype()
        || kv_heads != 1
        || query_heads % kv_heads != 0
        || key_head_dim != head_dim
        || valid_key_tokens == 0
        || valid_key_tokens > key_capacity
    {
        return Err(apxinf_core::Error::Other(
            "PI0-FAST MQA contract mismatch".into(),
        ));
    }
    let query = query.reshape(vec![1, query_tokens, query_heads, head_dim])?;
    let key = key.reshape(vec![1, key_capacity, kv_heads, head_dim])?;
    let value = value.reshape(vec![1, key_capacity, kv_heads, head_dim])?;
    let mut output = ctx.allocate_output(query.shape().clone(), apxinf_core::DType::BF16)?;
    let mut args =
        backend::ops::KvCacheAttentionArgs::new(&query, &key, &value, &mut output).non_causal();
    args.valid_key_tokens = valid_key_tokens;
    args.numerics = backend::ops::AttentionNumerics::Stable;
    backend::ops::kv_cache_attention(ctx, args)?;
    output.reshape(vec![query_tokens, query_heads, head_dim])
}

#[cfg(feature = "cuda")]
pub(super) fn attention_mha(
    ctx: &backend::Context,
    query: &apxinf_core::Tensor,
    key: &apxinf_core::Tensor,
    value: &apxinf_core::Tensor,
    tokens_per_batch: usize,
) -> apxinf_core::Result<apxinf_core::Tensor> {
    let [tokens, heads, head_dim] = attention_rank3(query, "MHA query")?;
    if key.shape() != query.shape()
        || value.shape() != query.shape()
        || query.dtype() != apxinf_core::DType::BF16
        || key.dtype() != query.dtype()
        || value.dtype() != query.dtype()
        || tokens_per_batch == 0
        || tokens % tokens_per_batch != 0
    {
        return Err(apxinf_core::Error::Other(
            "PI0-FAST MHA contract mismatch".into(),
        ));
    }
    let batches = tokens / tokens_per_batch;
    let query = query.reshape(vec![batches, tokens_per_batch, heads, head_dim])?;
    let key = key.reshape(vec![batches, tokens_per_batch, heads, head_dim])?;
    let value = value.reshape(vec![batches, tokens_per_batch, heads, head_dim])?;
    let mut output = ctx.allocate_output(query.shape().clone(), apxinf_core::DType::BF16)?;
    let mut args = backend::ops::AttentionArgs::new(&query, &key, &value, &mut output);
    args.numerics = backend::ops::AttentionNumerics::Stable;
    backend::ops::attention(ctx, args)?;
    output.reshape(vec![tokens, heads, head_dim])
}

#[cfg(feature = "cuda")]
pub(crate) fn register_builtin() {
    crate::registry::register("pi0fast-cuda", vla_runtime::load_registered);
    crate::registry::register("pi0_fast-cuda", vla_runtime::load_registered);
}
