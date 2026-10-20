//! Text decision forward pass built exclusively from cuda-new L3/fixed ops.

use apxinf_core::{DType, Error, Result, Tensor};
use apxinf_cuda_new::{ops, CudaBuffer, CudaContext};

use crate::d3::backend::{copy_i32, prefix, read_bf16, view_at, zeros};
use crate::d3::vision::{forward_vision_device, D3VisionScratch};
use crate::d3::weights::bf16::{D3DeviceWeights, FullAttentionWeights, GdnWeights, MixerWeights};
use crate::d3::D3PackageConfig;

const GDN_CHUNK: usize = 64;

#[derive(Clone)]
pub(crate) struct D3Policies {
    pub(crate) gemm: ops::GemmPolicy,
    pub(crate) attention: ops::AttentionPolicy,
}

impl D3Policies {
    pub(crate) fn for_recipe(cache_dir: Option<String>, online_tune: bool) -> Self {
        Self {
            gemm: ops::GemmPolicy {
                cache_dir: cache_dir.clone(),
                online_tune,
                ..ops::GemmPolicy::default()
            },
            attention: ops::AttentionPolicy {
                cache_dir,
                online_tune,
                ..ops::AttentionPolicy::default()
            },
        }
    }
}

pub(crate) struct D3TextScratch {
    capacity: usize,
    padded: usize,
    hidden: Tensor,
    normalized: Tensor,
    projected: Tensor,
    mlp_fused: Tensor,
    mlp_activation: Tensor,
    q_fused: Tensor,
    query: Tensor,
    query_gate: Tensor,
    key: Tensor,
    value: Tensor,
    attention_out: Tensor,
    positions: Tensor,
    tokens: Tensor,
    gdn_qkv: Tensor,
    gdn_conv: Tensor,
    gdn_z: Tensor,
    gdn_a: Tensor,
    gdn_b: Tensor,
    gdn_decay: Tensor,
    gdn_beta: Tensor,
    gdn_readout: Tensor,
    gdn_gated: Tensor,
    gdn_state: Tensor,
    gdn_conv_window: Option<Tensor>,
    gdn_q16: Option<Tensor>,
    gdn_k16: Option<Tensor>,
    gdn_v16: Option<Tensor>,
    gdn_out16: Option<Tensor>,
    gdn_alpha: Option<Tensor>,
    gdn_cu_seqlens: Option<Tensor>,
    gdn_workspace: Option<Tensor>,
    last_norm_bf16: Tensor,
    last_norm_f32: Tensor,
    logits: Tensor,
}

impl D3TextScratch {
    pub(crate) fn new(
        ctx: &CudaContext,
        config: &D3PackageConfig,
        capacity: usize,
    ) -> Result<Self> {
        if capacity == 0 || capacity > config.model.text_config.max_position_embeddings {
            return Err(Error::Other(format!(
                "d3 text profile length {capacity} is outside 1..={}",
                config.model.text_config.max_position_embeddings
            )));
        }
        let text = &config.model.text_config;
        let padded = capacity.div_ceil(GDN_CHUNK) * GDN_CHUNK;
        let hidden = text.hidden_size;
        let inter = text.intermediate_size;
        let heads = text.num_attention_heads;
        let kv_heads = text.num_key_value_heads;
        let head_dim = text.head_dim;
        let gdn_k_width = text.linear_num_key_heads * text.linear_key_head_dim;
        let gdn_v_width = text.linear_num_value_heads * text.linear_value_head_dim;
        let gdn_qkv_width = 2 * gdn_k_width + gdn_v_width;
        let gdn_heads = text.linear_num_value_heads;
        let gdn_dim = text.linear_value_head_dim;
        let cake = ctx.caps().sm >= 100;
        let gdn_cu_seqlens = cake.then(|| zeros(ctx, vec![2], DType::I32)).transpose()?;
        if let Some(cu_seqlens) = &gdn_cu_seqlens {
            copy_i32(ctx, cu_seqlens, &[0, capacity as i32])?;
        }
        let workspace_bytes = ops::gdn_prefill_workspace_bytes(gdn_heads, 1);
        Ok(Self {
            capacity,
            padded,
            hidden: zeros(ctx, vec![capacity, hidden], DType::BF16)?,
            normalized: zeros(ctx, vec![capacity, hidden], DType::BF16)?,
            projected: zeros(ctx, vec![capacity, hidden], DType::BF16)?,
            mlp_fused: zeros(ctx, vec![capacity, 2 * inter], DType::BF16)?,
            mlp_activation: zeros(ctx, vec![capacity, inter], DType::BF16)?,
            q_fused: zeros(ctx, vec![capacity, heads, 2 * head_dim], DType::BF16)?,
            query: zeros(ctx, vec![capacity, heads, head_dim], DType::BF16)?,
            query_gate: zeros(ctx, vec![capacity, heads, head_dim], DType::BF16)?,
            key: zeros(ctx, vec![capacity, kv_heads, head_dim], DType::BF16)?,
            value: zeros(ctx, vec![capacity, kv_heads, head_dim], DType::BF16)?,
            attention_out: zeros(ctx, vec![capacity, heads, head_dim], DType::BF16)?,
            positions: zeros(ctx, vec![capacity], DType::I32)?,
            tokens: zeros(ctx, vec![capacity], DType::I32)?,
            gdn_qkv: zeros(ctx, vec![padded, gdn_qkv_width], DType::BF16)?,
            gdn_conv: zeros(ctx, vec![padded, gdn_qkv_width], DType::BF16)?,
            gdn_z: zeros(ctx, vec![capacity, gdn_v_width], DType::BF16)?,
            gdn_a: zeros(ctx, vec![padded, gdn_heads], DType::BF16)?,
            gdn_b: zeros(ctx, vec![padded, gdn_heads], DType::BF16)?,
            gdn_decay: zeros(ctx, vec![padded, gdn_heads], DType::F32)?,
            gdn_beta: zeros(ctx, vec![padded, gdn_heads], DType::F32)?,
            gdn_readout: zeros(ctx, vec![padded, gdn_heads, gdn_dim], DType::BF16)?,
            gdn_gated: zeros(ctx, vec![capacity, gdn_heads, gdn_dim], DType::BF16)?,
            gdn_state: zeros(ctx, vec![gdn_heads, gdn_dim, gdn_dim], DType::F32)?,
            gdn_conv_window: cake
                .then(|| {
                    zeros(
                        ctx,
                        vec![gdn_qkv_width, text.linear_conv_kernel_dim],
                        DType::F32,
                    )
                })
                .transpose()?,
            gdn_q16: cake
                .then(|| {
                    zeros(
                        ctx,
                        vec![capacity, text.linear_num_key_heads, gdn_dim],
                        DType::F16,
                    )
                })
                .transpose()?,
            gdn_k16: cake
                .then(|| {
                    zeros(
                        ctx,
                        vec![capacity, text.linear_num_key_heads, gdn_dim],
                        DType::F16,
                    )
                })
                .transpose()?,
            gdn_v16: cake
                .then(|| zeros(ctx, vec![capacity, gdn_heads, gdn_dim], DType::F16))
                .transpose()?,
            gdn_out16: cake
                .then(|| zeros(ctx, vec![capacity, gdn_heads, gdn_dim], DType::F16))
                .transpose()?,
            gdn_alpha: cake
                .then(|| zeros(ctx, vec![capacity, gdn_heads], DType::F32))
                .transpose()?,
            gdn_cu_seqlens,
            gdn_workspace: cake
                .then(|| zeros(ctx, vec![workspace_bytes.div_ceil(4)], DType::F32))
                .transpose()?,
            last_norm_bf16: zeros(ctx, vec![1, hidden], DType::BF16)?,
            last_norm_f32: zeros(ctx, vec![1, hidden], DType::F32)?,
            logits: zeros(ctx, vec![1, config.decision.codes.len()], DType::F32)?,
        })
    }

    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    pub(crate) fn logits(&self) -> &Tensor {
        &self.logits
    }
}

pub(crate) fn forward_text(
    ctx: &CudaContext,
    config: &D3PackageConfig,
    weights: &D3DeviceWeights,
    policies: &D3Policies,
    scratch: &mut D3TextScratch,
    token_ids: &[u32],
) -> Result<()> {
    write_text_inputs(ctx, config, scratch, token_ids)?;
    forward_text_device(ctx, config, weights, policies, scratch, token_ids.len())
}

pub(crate) fn write_text_inputs(
    ctx: &CudaContext,
    config: &D3PackageConfig,
    scratch: &D3TextScratch,
    token_ids: &[u32],
) -> Result<()> {
    let tokens = token_ids.len();
    if tokens == 0 || tokens > scratch.capacity {
        return Err(Error::Other(format!(
            "d3 text request length {tokens} exceeds prepared profile {}",
            scratch.capacity
        )));
    }
    let text = &config.model.text_config;
    if let Some(token) = token_ids
        .iter()
        .find(|&&token| token as usize >= text.vocab_size)
    {
        return Err(Error::Other(format!(
            "d3 token id {token} exceeds vocabulary"
        )));
    }
    let ids = token_ids.iter().map(|&id| id as i32).collect::<Vec<_>>();
    let positions = (0..tokens as i32).collect::<Vec<_>>();
    copy_i32(
        ctx,
        &prefix(&scratch.tokens, vec![tokens], DType::I32)?,
        &ids,
    )?;
    copy_i32(
        ctx,
        &prefix(&scratch.positions, vec![tokens], DType::I32)?,
        &positions,
    )?;
    Ok(())
}

pub(crate) fn forward_text_device(
    ctx: &CudaContext,
    config: &D3PackageConfig,
    weights: &D3DeviceWeights,
    policies: &D3Policies,
    scratch: &mut D3TextScratch,
    tokens: usize,
) -> Result<()> {
    let text = &config.model.text_config;
    let hidden = prefix(&scratch.hidden, vec![tokens, text.hidden_size], DType::BF16)?;
    let token_tensor = prefix(&scratch.tokens, vec![tokens], DType::I32)?;
    ops::embedding_gather(ctx, &weights.embed_tokens, &token_tensor, &hidden)?;

    forward_language_device(ctx, config, weights, policies, scratch, tokens, None)
}

pub(crate) fn forward_multimodal_device(
    ctx: &CudaContext,
    config: &D3PackageConfig,
    weights: &D3DeviceWeights,
    policies: &D3Policies,
    scratch: &mut D3TextScratch,
    vision: &mut D3VisionScratch,
    tokens: usize,
) -> Result<()> {
    let text = &config.model.text_config;
    let hidden = prefix(&scratch.hidden, vec![tokens, text.hidden_size], DType::BF16)?;
    let token_tensor = prefix(&scratch.tokens, vec![tokens], DType::I32)?;
    ops::embedding_gather(ctx, &weights.embed_tokens, &token_tensor, &hidden)?;
    forward_vision_device(ctx, config, &weights.vision, policies, vision)?;
    let row_map = CudaBuffer::from_tensor(vision.row_sources()).map_err(Error::Cuda)?;
    let hidden_input = hidden.clone();
    let mut hidden_output = hidden.clone();
    let mut replace = ops::GatherArgs::new(
        ops::GatherSemantic::ReplaceRows,
        &hidden_input,
        &mut hidden_output,
    );
    replace.replacement = Some(vision.visual_output());
    replace.row_map = Some(&row_map);
    ops::gather(ctx, replace)?;
    forward_language_device(
        ctx,
        config,
        weights,
        policies,
        scratch,
        tokens,
        Some(vision.mrope_positions()),
    )
}

fn forward_language_device(
    ctx: &CudaContext,
    config: &D3PackageConfig,
    weights: &D3DeviceWeights,
    policies: &D3Policies,
    scratch: &mut D3TextScratch,
    tokens: usize,
    mrope_positions: Option<&Tensor>,
) -> Result<()> {
    let text = &config.model.text_config;

    for (layer_index, layer) in weights.layers.iter().enumerate() {
        match layer {
            MixerWeights::FullAttention(layer) => full_attention(
                ctx,
                config,
                layer,
                policies,
                scratch,
                tokens,
                mrope_positions,
            )?,
            MixerWeights::Gdn(layer) => gdn(ctx, config, layer, policies, scratch, tokens)?,
        }
        mlp(
            ctx,
            config,
            layer_post_norm(layer),
            layer_gate_up(layer),
            layer_down(layer),
            policies,
            scratch,
            tokens,
        )?;
        if std::env::var_os("APXINF_D3_TRACE").is_some() {
            trace_last_hidden(ctx, config, scratch, tokens, layer_index)?;
        }
    }

    let last = view_at(
        &scratch.hidden,
        (tokens - 1) * text.hidden_size,
        vec![1, text.hidden_size],
        DType::BF16,
    )?;
    ops::mlp::rms_norm(
        ctx,
        &last,
        &weights.final_norm,
        &scratch.last_norm_bf16,
        text.rms_norm_eps,
    )?;
    ops::quantization(
        ctx,
        ops::QuantizationArgs::new(
            ops::QuantizationSemantic::CastBf16ToF32,
            &scratch.last_norm_bf16,
            &mut scratch.last_norm_f32,
        ),
    )?;
    gemm(
        ctx,
        &scratch.last_norm_f32,
        &weights.readout,
        &mut scratch.logits,
        &policies.gemm,
    )?;
    Ok(())
}

fn trace_last_hidden(
    ctx: &CudaContext,
    config: &D3PackageConfig,
    scratch: &D3TextScratch,
    tokens: usize,
    layer: usize,
) -> Result<()> {
    let width = config.model.text_config.hidden_size;
    let last = view_at(
        &scratch.hidden,
        (tokens - 1) * width,
        vec![1, width],
        DType::BF16,
    )?;
    let values = read_bf16(ctx, &last)?;
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let rms = (values.iter().map(|value| value * value).sum::<f32>() / values.len() as f32).sqrt();
    let max_abs = values.iter().copied().map(f32::abs).fold(0.0, f32::max);
    eprintln!(
        "D3_TRACE layer={layer} mean={mean:.9} rms={rms:.9} maxabs={max_abs:.9} first16={:?}",
        &values[..16.min(values.len())]
    );
    Ok(())
}

fn full_attention(
    ctx: &CudaContext,
    config: &D3PackageConfig,
    layer: &FullAttentionWeights,
    policies: &D3Policies,
    scratch: &mut D3TextScratch,
    tokens: usize,
    mrope_positions: Option<&Tensor>,
) -> Result<()> {
    let text = &config.model.text_config;
    let hidden = prefix(&scratch.hidden, vec![tokens, text.hidden_size], DType::BF16)?;
    let normalized = prefix(
        &scratch.normalized,
        vec![tokens, text.hidden_size],
        DType::BF16,
    )?;
    ops::mlp::rms_norm(
        ctx,
        &hidden,
        &layer.input_norm,
        &normalized,
        text.rms_norm_eps,
    )?;

    let mut q_fused = prefix(
        &scratch.q_fused,
        vec![tokens, text.num_attention_heads * 2 * text.head_dim],
        DType::BF16,
    )?;
    let mut key = prefix(
        &scratch.key,
        vec![tokens, text.num_key_value_heads * text.head_dim],
        DType::BF16,
    )?;
    let mut value = prefix(
        &scratch.value,
        vec![tokens, text.num_key_value_heads * text.head_dim],
        DType::BF16,
    )?;
    gemm(ctx, &normalized, &layer.q_w, &mut q_fused, &policies.gemm)?;
    gemm(ctx, &normalized, &layer.k_w, &mut key, &policies.gemm)?;
    gemm(ctx, &normalized, &layer.v_w, &mut value, &policies.gemm)?;

    let q_fused_heads = prefix(
        &scratch.q_fused,
        vec![tokens, text.num_attention_heads, 2 * text.head_dim],
        DType::BF16,
    )?;
    let mut query = prefix(
        &scratch.query,
        vec![tokens, text.num_attention_heads, text.head_dim],
        DType::BF16,
    )?;
    let gate = prefix(
        &scratch.query_gate,
        vec![tokens, text.num_attention_heads, text.head_dim],
        DType::BF16,
    )?;
    ops::split_query_and_gate(ctx, &q_fused_heads, &query, &gate)?;
    ops::head_rms_norm(
        ctx,
        &prefix(
            &scratch.query,
            vec![tokens * text.num_attention_heads, text.head_dim],
            DType::BF16,
        )?,
        &layer.q_norm,
        text.rms_norm_eps,
    )?;
    ops::head_rms_norm(
        ctx,
        &prefix(
            &scratch.key,
            vec![tokens * text.num_key_value_heads, text.head_dim],
            DType::BF16,
        )?,
        &layer.k_norm,
        text.rms_norm_eps,
    )?;
    let positions = prefix(&scratch.positions, vec![tokens], DType::I32)?;
    let rotary = ops::rotary_dim(text.head_dim, text.partial_rotary_factor);
    if let Some(mrope) = mrope_positions {
        ops::multimodal_rope(
            ctx,
            ops::MultimodalRopeArgs {
                data: &mut query,
                positions: mrope,
                rotary_dim: rotary,
                sections: text.rope_parameters.mrope_section,
                theta: text.rope_parameters.rope_theta,
            },
        )?;
    } else {
        ops::partial_rope(
            ctx,
            &query,
            &positions,
            rotary,
            text.rope_parameters.rope_theta,
        )?;
    }
    let mut key_heads = prefix(
        &scratch.key,
        vec![tokens, text.num_key_value_heads, text.head_dim],
        DType::BF16,
    )?;
    if let Some(mrope) = mrope_positions {
        ops::multimodal_rope(
            ctx,
            ops::MultimodalRopeArgs {
                data: &mut key_heads,
                positions: mrope,
                rotary_dim: rotary,
                sections: text.rope_parameters.mrope_section,
                theta: text.rope_parameters.rope_theta,
            },
        )?;
    } else {
        ops::partial_rope(
            ctx,
            &key_heads,
            &positions,
            rotary,
            text.rope_parameters.rope_theta,
        )?;
    }

    let query_4d = prefix(
        &scratch.query,
        vec![1, tokens, text.num_attention_heads, text.head_dim],
        DType::BF16,
    )?;
    let key_4d = prefix(
        &scratch.key,
        vec![1, tokens, text.num_key_value_heads, text.head_dim],
        DType::BF16,
    )?;
    let value_4d = prefix(
        &scratch.value,
        vec![1, tokens, text.num_key_value_heads, text.head_dim],
        DType::BF16,
    )?;
    let mut out_4d = prefix(
        &scratch.attention_out,
        vec![1, tokens, text.num_attention_heads, text.head_dim],
        DType::BF16,
    )?;
    // Decision 3.0 changes only the full-attention mask: all prompt tokens
    // attend bidirectionally. GDN layers remain their native causal scan.
    let mut attention_args = ops::AttentionArgs::new(&query_4d, &key_4d, &value_4d, &mut out_4d);
    attention_args.policy = policies.attention.clone();
    ops::attention(ctx, attention_args)?;
    let attention = prefix(
        &scratch.attention_out,
        vec![tokens, text.num_attention_heads * text.head_dim],
        DType::BF16,
    )?;
    let gate = prefix(
        &scratch.query_gate,
        vec![tokens, text.num_attention_heads * text.head_dim],
        DType::BF16,
    )?;
    ops::apply_output_gate(ctx, &attention, &gate)?;
    let mut projected = prefix(
        &scratch.projected,
        vec![tokens, text.hidden_size],
        DType::BF16,
    )?;
    gemm(ctx, &attention, &layer.o_w, &mut projected, &policies.gemm)?;
    ops::add_into(ctx, &projected, &hidden)
}

fn gdn(
    ctx: &CudaContext,
    config: &D3PackageConfig,
    layer: &GdnWeights,
    policies: &D3Policies,
    scratch: &mut D3TextScratch,
    tokens: usize,
) -> Result<()> {
    let text = &config.model.text_config;
    let hidden = prefix(&scratch.hidden, vec![tokens, text.hidden_size], DType::BF16)?;
    let normalized = prefix(
        &scratch.normalized,
        vec![tokens, text.hidden_size],
        DType::BF16,
    )?;
    ops::mlp::rms_norm(
        ctx,
        &hidden,
        &layer.input_norm,
        &normalized,
        text.rms_norm_eps,
    )?;

    let key_width = text.linear_num_key_heads * text.linear_key_head_dim;
    let value_width = text.linear_num_value_heads * text.linear_value_head_dim;
    let qkv_width = 2 * key_width + value_width;
    let mut qkv = prefix(&scratch.gdn_qkv, vec![tokens, qkv_width], DType::BF16)?;
    let mut z = prefix(&scratch.gdn_z, vec![tokens, value_width], DType::BF16)?;
    let mut a = prefix(
        &scratch.gdn_a,
        vec![tokens, text.linear_num_value_heads],
        DType::BF16,
    )?;
    let mut b = prefix(
        &scratch.gdn_b,
        vec![tokens, text.linear_num_value_heads],
        DType::BF16,
    )?;
    gemm(ctx, &normalized, &layer.qkv_w, &mut qkv, &policies.gemm)?;
    gemm(ctx, &normalized, &layer.z_w, &mut z, &policies.gemm)?;
    gemm(ctx, &normalized, &layer.a_w, &mut a, &policies.gemm)?;
    gemm(ctx, &normalized, &layer.b_w, &mut b, &policies.gemm)?;

    let decay = prefix(
        &scratch.gdn_decay,
        vec![tokens, text.linear_num_value_heads],
        DType::F32,
    )?;
    let beta = prefix(
        &scratch.gdn_beta,
        vec![tokens, text.linear_num_value_heads],
        DType::F32,
    )?;
    ops::gdn_decay_and_beta_seq(
        ctx,
        &a,
        &b,
        &layer.a_log,
        &layer.dt_bias,
        &decay,
        &beta,
        tokens,
        text.linear_num_value_heads,
    )?;

    let z_heads = prefix(
        &scratch.gdn_z,
        vec![
            tokens,
            text.linear_num_value_heads,
            text.linear_value_head_dim,
        ],
        DType::BF16,
    )?;
    let gated = prefix(
        &scratch.gdn_gated,
        vec![
            tokens,
            text.linear_num_value_heads,
            text.linear_value_head_dim,
        ],
        DType::BF16,
    )?;
    zero_tensor(ctx, &scratch.gdn_state)?;
    if let (
        Some(conv_window),
        Some(q16),
        Some(k16),
        Some(v16),
        Some(out16),
        Some(alpha),
        Some(cu_seqlens),
        Some(workspace),
    ) = (
        scratch.gdn_conv_window.as_ref(),
        scratch.gdn_q16.as_ref(),
        scratch.gdn_k16.as_ref(),
        scratch.gdn_v16.as_ref(),
        scratch.gdn_out16.as_ref(),
        scratch.gdn_alpha.as_ref(),
        scratch.gdn_cu_seqlens.as_ref(),
        scratch.gdn_workspace.as_ref(),
    ) {
        zero_tensor(ctx, conv_window)?;
        let q_rows = prefix(
            q16,
            vec![tokens, text.linear_num_key_heads, text.linear_key_head_dim],
            DType::F16,
        )?;
        let k_rows = prefix(
            k16,
            vec![tokens, text.linear_num_key_heads, text.linear_key_head_dim],
            DType::F16,
        )?;
        let v_rows = prefix(
            v16,
            vec![
                tokens,
                text.linear_num_value_heads,
                text.linear_value_head_dim,
            ],
            DType::F16,
        )?;
        let out_rows = prefix(
            out16,
            vec![
                tokens,
                text.linear_num_value_heads,
                text.linear_value_head_dim,
            ],
            DType::F16,
        )?;
        let alpha_rows = prefix(alpha, vec![tokens, text.linear_num_value_heads], DType::F32)?;
        ops::gdn_conv_prepare(
            ctx,
            &qkv,
            &layer.conv_w,
            conv_window,
            &q_rows,
            &k_rows,
            &v_rows,
            &decay,
            &alpha_rows,
            tokens,
            text.linear_num_key_heads,
            text.linear_num_value_heads,
            text.rms_norm_eps,
        )?;
        ops::gdn_prefill(
            ctx,
            &q_rows,
            &k_rows,
            &v_rows,
            &out_rows,
            &alpha_rows,
            &beta,
            cu_seqlens,
            &scratch.gdn_state,
            workspace,
            tokens,
            text.linear_num_key_heads,
            text.linear_num_value_heads,
            1,
            1.0 / (text.linear_key_head_dim as f32).sqrt(),
        )?;
        ops::gdn_gated_norm_seq_f16(
            ctx,
            &out_rows,
            &z_heads,
            &layer.gated_norm,
            &gated,
            tokens,
            text.linear_num_value_heads,
            text.linear_value_head_dim,
            text.rms_norm_eps,
        )?;
    } else {
        let conv = prefix(&scratch.gdn_conv, vec![tokens, qkv_width], DType::BF16)?;
        ops::gdn_causal_conv_forward(
            ctx,
            &qkv,
            &layer.conv_w,
            &conv,
            None,
            tokens,
            qkv_width,
            text.linear_conv_kernel_dim,
        )?;
        let fused_padded = prefix(
            &scratch.gdn_conv,
            vec![scratch.padded, qkv_width],
            DType::BF16,
        )?;
        let decay_padded = prefix(
            &scratch.gdn_decay,
            vec![scratch.padded, text.linear_num_value_heads],
            DType::F32,
        )?;
        let beta_padded = prefix(
            &scratch.gdn_beta,
            vec![scratch.padded, text.linear_num_value_heads],
            DType::F32,
        )?;
        let readout_padded = prefix(
            &scratch.gdn_readout,
            vec![
                scratch.padded,
                text.linear_num_value_heads,
                text.linear_value_head_dim,
            ],
            DType::BF16,
        )?;
        ops::gdn_chunk_scan_interleaved(
            ctx,
            &fused_padded,
            &decay_padded,
            &beta_padded,
            &readout_padded,
            &scratch.gdn_state,
            scratch.padded,
            qkv_width,
            [0, key_width, 2 * key_width],
            text.linear_num_value_heads,
            text.linear_num_key_heads,
            GDN_CHUNK,
            text.linear_key_head_dim,
        )?;
        let readout = prefix(
            &scratch.gdn_readout,
            vec![
                tokens,
                text.linear_num_value_heads,
                text.linear_value_head_dim,
            ],
            DType::BF16,
        )?;
        ops::gdn_gated_norm_seq(
            ctx,
            &readout,
            &z_heads,
            &layer.gated_norm,
            &gated,
            tokens,
            text.linear_num_value_heads,
            text.linear_value_head_dim,
            text.rms_norm_eps,
        )?;
    }
    let gated_flat = prefix(&scratch.gdn_gated, vec![tokens, value_width], DType::BF16)?;
    let mut projected = prefix(
        &scratch.projected,
        vec![tokens, text.hidden_size],
        DType::BF16,
    )?;
    gemm(
        ctx,
        &gated_flat,
        &layer.out_w,
        &mut projected,
        &policies.gemm,
    )?;
    ops::add_into(ctx, &projected, &hidden)
}

fn mlp(
    ctx: &CudaContext,
    config: &D3PackageConfig,
    norm: &Tensor,
    gate_up: &Tensor,
    down: &Tensor,
    policies: &D3Policies,
    scratch: &mut D3TextScratch,
    tokens: usize,
) -> Result<()> {
    let text = &config.model.text_config;
    let hidden = prefix(&scratch.hidden, vec![tokens, text.hidden_size], DType::BF16)?;
    let normalized = prefix(
        &scratch.normalized,
        vec![tokens, text.hidden_size],
        DType::BF16,
    )?;
    ops::mlp::rms_norm(ctx, &hidden, norm, &normalized, text.rms_norm_eps)?;
    let mut fused = prefix(
        &scratch.mlp_fused,
        vec![tokens, 2 * text.intermediate_size],
        DType::BF16,
    )?;
    gemm(ctx, &normalized, gate_up, &mut fused, &policies.gemm)?;
    let activation = prefix(
        &scratch.mlp_activation,
        vec![tokens, text.intermediate_size],
        DType::BF16,
    )?;
    ops::mlp::swiglu(ctx, &fused, &activation)?;
    let mut projected = prefix(
        &scratch.projected,
        vec![tokens, text.hidden_size],
        DType::BF16,
    )?;
    gemm(ctx, &activation, down, &mut projected, &policies.gemm)?;
    ops::add_into(ctx, &projected, &hidden)
}

fn layer_post_norm(layer: &MixerWeights) -> &Tensor {
    match layer {
        MixerWeights::FullAttention(layer) => &layer.post_norm,
        MixerWeights::Gdn(layer) => &layer.post_norm,
    }
}

fn layer_gate_up(layer: &MixerWeights) -> &Tensor {
    match layer {
        MixerWeights::FullAttention(layer) => &layer.gate_up_w,
        MixerWeights::Gdn(layer) => &layer.gate_up_w,
    }
}

fn layer_down(layer: &MixerWeights) -> &Tensor {
    match layer {
        MixerWeights::FullAttention(layer) => &layer.down_w,
        MixerWeights::Gdn(layer) => &layer.down_w,
    }
}

fn gemm(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    output: &mut Tensor,
    policy: &ops::GemmPolicy,
) -> Result<()> {
    let mut args = ops::GemmArgs::new(input, weight, output);
    args.policy = policy.clone();
    ops::gemm(ctx, args)
}

fn zero_tensor(ctx: &CudaContext, tensor: &Tensor) -> Result<()> {
    CudaBuffer::from_tensor(tensor)
        .map_err(Error::Cuda)?
        .zero_async(ctx.stream())
        .map_err(Error::Cuda)
}
