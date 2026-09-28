//! Model dataflow: prefill and decode computation over the weight tree.
//!
//! Owns layer order, projections, GDN/attention semantics and the scratch
//! buffers the computation binds. Does not own graph capture or session
//! policy -- that is `model_runner`.


use std::time::Instant;

use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda_new::{ops, CapturedGraph, CudaBuffer, CudaContext};

use super::backend::{graph_tensor_bytes, prefix, view, zeros};
use super::config::*;
use super::weights::{Fp8Weight, GdnLayer, Layer, Model, Nvfp4Weight};


/// Per-token working buffers, sized for one token of decode.
pub(crate) struct Scratch {
    pub(crate) hidden: Tensor,
    normalized: Tensor,
    fp8_activation: Tensor,
    nvfp4_activation: Tensor,
    nvfp4_scales: Tensor,
    mlp_fused: Tensor,
    mlp_activation: Tensor,
    mlp_scales: Tensor,
    mlp_out: Tensor,
    // attention
    qkv_fused: Tensor,
    query: Tensor,
    query_gate: Tensor,
    attention_out: Tensor,
    attention_fp8: Tensor,
    projected: Tensor,
    pub(crate) positions: Tensor,
    // gdn
    gdn_qkv: Tensor,
    gdn_conv: Tensor,
    gdn_z: Tensor,
    gdn_a: Tensor,
    gdn_b: Tensor,
    gdn_decay: Tensor,
    gdn_beta: Tensor,
    pub(crate) gdn_readout: Tensor,
    gdn_gated: Tensor,
    gdn_fp8: Tensor,
    pub(crate) token: Tensor,
    logits: Tensor,
    next_token: Tensor,
}

impl Scratch {
    pub(crate) fn logits(&self) -> &Tensor {
        &self.logits
    }

    pub(crate) fn next_token_host(&self) -> i32 {
        let mut id = [0u8; 4];
        CudaBuffer::from_tensor(&self.next_token)
            .unwrap()
            .copy_to_host(&mut id)
            .unwrap();
        i32::from_le_bytes(id)
    }

    pub(crate) fn new(ctx: &CudaContext) -> Scratch {
        let scale_bytes = |rows: usize, k: usize| {
            vec![ops::nvfp4_scale_buffer_bytes(rows, k, BLOCK).unwrap()]
        };
        Scratch {
            hidden: zeros(ctx, vec![1, HIDDEN], DType::BF16),
            normalized: zeros(ctx, vec![1, HIDDEN], DType::BF16),
            fp8_activation: zeros(ctx, vec![1, HIDDEN], DType::F8E4M3),
            nvfp4_activation: zeros(ctx, vec![1, HIDDEN / 2], DType::E2M1Pair),
            nvfp4_scales: zeros(ctx, scale_bytes(1, HIDDEN), DType::F8E4M3),
            mlp_fused: zeros(ctx, vec![1, 2 * INTERMEDIATE], DType::BF16),
            mlp_activation: zeros(ctx, vec![1, INTERMEDIATE / 2], DType::E2M1Pair),
            mlp_scales: zeros(ctx, scale_bytes(1, INTERMEDIATE), DType::F8E4M3),
            mlp_out: zeros(ctx, vec![1, HIDDEN], DType::BF16),
            qkv_fused: zeros(ctx, vec![1, 2 * HEADS * HEAD_DIM], DType::BF16),
            query: zeros(ctx, vec![1, HEADS, HEAD_DIM], DType::BF16),
            query_gate: zeros(ctx, vec![1, HEADS, HEAD_DIM], DType::BF16),
            attention_out: zeros(ctx, vec![1, HEADS * HEAD_DIM], DType::BF16),
            attention_fp8: zeros(ctx, vec![1, HEADS * HEAD_DIM], DType::F8E4M3),
            projected: zeros(ctx, vec![1, HIDDEN], DType::BF16),
            positions: zeros(ctx, vec![1], DType::I32),
            gdn_qkv: zeros(ctx, vec![1, QKV_WIDTH], DType::BF16),
            gdn_conv: zeros(ctx, vec![QKV_WIDTH], DType::BF16),
            gdn_z: zeros(ctx, vec![1, Z_WIDTH], DType::BF16),
            gdn_a: zeros(ctx, vec![GDN_V_HEADS], DType::BF16),
            gdn_b: zeros(ctx, vec![GDN_V_HEADS], DType::BF16),
            gdn_decay: zeros(ctx, vec![GDN_V_HEADS], DType::F32),
            gdn_beta: zeros(ctx, vec![GDN_V_HEADS], DType::F32),
            gdn_readout: zeros(ctx, vec![GDN_V_HEADS, GDN_HEAD_DIM], DType::BF16),
            gdn_gated: zeros(ctx, vec![GDN_V_HEADS, GDN_HEAD_DIM], DType::BF16),
            gdn_fp8: zeros(ctx, vec![1, Z_WIDTH], DType::F8E4M3),
            token: zeros(ctx, vec![1], DType::I32),
            logits: zeros(ctx, vec![1, VOCAB], DType::BF16),
            next_token: zeros(ctx, vec![1], DType::I32),
        }
    }
}

/// Recurrent state a GDN layer carries between tokens.
pub(crate) struct GdnState {
    pub(crate) recurrent: Tensor,
    pub(crate) conv_window: Tensor,
}

/// Key/value cache for one full-attention layer.
pub(crate) struct KvCache {
    pub(crate) keys: Tensor,
    pub(crate) values: Tensor,
}

/// Single-token FP8 projection: quantize against the checkpoint's
/// `input_scale`, then one GEMV with both per-tensor scales in alpha.
///
/// The GEMV reaches 249.7 GB/s on the qkv shape against the general GEMM's
/// 146.8 GB/s. A GEMM's tiling is built for large M and leaves half this
/// device's bandwidth on the table at M=1.
fn fp8_projection(
    ctx: &CudaContext,
    weight: &Fp8Weight,
    source: &Tensor,
    quantized: &Tensor,
    output: &mut Tensor,
) {
    ops::quantize_fp8_per_tensor(ctx, source, quantized, weight.input_scale).unwrap();
    ops::fp8_gemv(ctx, &weight.weight, quantized, output, weight.alpha).unwrap();
}

fn reuse_fp8_enabled() -> bool {
    // Validated in rounds 21/51; the shared-activation quantize is exact and
    // the env override that guarded the rollout is retired.
    true
}

fn fp8_projection_reuse_quantized(
    ctx: &CudaContext,
    weight: &Fp8Weight,
    quantized: &Tensor,
    output: &mut Tensor,
) {
    ops::fp8_gemv(ctx, &weight.weight, quantized, output, weight.alpha).unwrap();
}

/// Many-token FP8 projection: the same contract as `fp8_projection`, run as
/// a GEMM because the GEMV is built for one row.
///
/// `quantize_fp8_per_tensor` is elementwise, so it needs no change for
/// `rows > 1`; only the matmul does. Unit-scale FP8 with
/// `alpha = weight_scale * input_scale` is the contract
/// `qwen38_fp8_projection.rs` already accepts at relative L2 2.70%.
fn fp8_projection_rows(
    ctx: &CudaContext,
    weight: &Fp8Weight,
    source: &Tensor,
    quantized: &Tensor,
    output: &mut Tensor,
    rows: usize,
) {
    quantize_shared_fp8_activation(ctx, &[weight], source, quantized, rows);
    fp8_projection_prequantized(ctx, weight, quantized, output, rows);
}

/// Quantize one activation for a group of projections that all read it.
///
/// ModelOpt records `input_scale` per projection, but the GDN `qkv`/`z` pair
/// and the attention `q`/`k`/`v` triple consume the same post-norm
/// activation, and in this checkpoint their scales are bit-identical --
/// calibration saw one tensor. Quantizing per projection therefore recomputes
/// the same FP8 bytes two or three times. At 2048 tokens each repeat is
/// 10.5M elements of duplicate work, and it measured as the largest remaining
/// item in the projection stage once the GEMM itself was fixed.
///
/// The scales are asserted rather than assumed. A checkpoint that calibrated
/// them apart needs one buffer per projection, and quietly applying one
/// scale to all of them would be an accuracy fault with no symptom here.
fn quantize_shared_fp8_activation(
    ctx: &CudaContext,
    weights: &[&Fp8Weight],
    source: &Tensor,
    quantized: &Tensor,
    rows: usize,
) {
    let (first, rest) = weights.split_first().expect("no projection to quantize for");
    let k = first.weight.shape().dims()[1];
    for other in rest {
        assert_eq!(
            other.input_scale, first.input_scale,
            "projections sharing an activation must share input_scale"
        );
        assert_eq!(other.weight.shape().dims()[1], k, "shared activation width");
    }

    let source_rows = prefix(source, vec![rows, k], DType::BF16);
    let quantized_rows = prefix(quantized, vec![rows, k], DType::F8E4M3);
    ops::quantize_fp8_per_tensor(ctx, &source_rows, &quantized_rows, first.input_scale).unwrap();
}

/// One projection over an activation `quantize_shared_fp8_activation` already
/// wrote. Unit-scale FP8 with `alpha = weight_scale * input_scale` is the
/// contract `qwen38_fp8_projection.rs` accepts at relative L2 2.70%.
fn fp8_projection_prequantized(
    ctx: &CudaContext,
    weight: &Fp8Weight,
    quantized: &Tensor,
    output: &mut Tensor,
    rows: usize,
) {
    let transposed = weight
        .transposed
        .as_ref()
        .expect("prefill needs the [K, N] FP8 copy; load the model with prefill enabled");
    let k = weight.weight.shape().dims()[1];
    let n = weight.weight.shape().dims()[0];

    let quantized_rows = prefix(quantized, vec![rows, k], DType::F8E4M3);
    let mut out_rows = prefix(output, vec![rows, n], DType::BF16);
    let mut args = ops::GemmArgs::new(&quantized_rows, transposed, &mut out_rows);
    args.quantization = ops::GemmQuantization::Fp8UnitScale;
    args.alpha = weight.alpha;
    args.policy.cache_dir = gemm_cache_dir();
    ops::gemm(ctx, args).unwrap();
}

fn nvfp4_decode_gemm(ctx: &CudaContext, mut args: ops::GemmArgs<'_>) -> apxinf_core::Result<()> {
    args.policy.cache_dir = gemm_cache_dir();
    ops::gemm(ctx, args)
}

/// residual = residual + MLP(RMSNorm(residual)), with the residual being the
/// running hidden state. Taking it from `scratch` rather than as a separate
/// argument keeps the borrow disjoint.
pub(crate) fn nvfp4_mlp(
    ctx: &CudaContext,
    gate_up: &Nvfp4Weight,
    down: &Nvfp4Weight,
    norm_weight: &Tensor,
    scratch: &mut Scratch,
) {
    ops::nvfp4_quantize_rms_norm(
        ctx,
        &scratch.hidden,
        norm_weight,
        &scratch.nvfp4_activation,
        &scratch.nvfp4_scales,
        EPSILON,
        gate_up.input_scale,
        BLOCK,
        ops::ScaleLayout::GemmAtom,
    )
    .unwrap();
    nvfp4_decode_gemm(
        ctx,
        ops::GemmArgs::nvfp4(
            &scratch.nvfp4_activation,
            &scratch.nvfp4_scales,
            &gate_up.packed,
            &gate_up.scales,
            BLOCK,
            gate_up.alpha,
            &mut scratch.mlp_fused,
        ),
    )
    .unwrap();
    ops::nvfp4_quantize_swiglu(
        ctx,
        &scratch.mlp_fused,
        &scratch.mlp_activation,
        &scratch.mlp_scales,
        down.input_scale,
        BLOCK,
        ops::ScaleLayout::GemmAtom,
    )
    .unwrap();
    nvfp4_decode_gemm(
        ctx,
        ops::GemmArgs::nvfp4(
            &scratch.mlp_activation,
            &scratch.mlp_scales,
            &down.packed,
            &down.scales,
            BLOCK,
            down.alpha,
            &mut scratch.mlp_out,
        ),
    )
    .unwrap();
    ops::add_into(ctx, &scratch.mlp_out, &scratch.hidden).unwrap();
}

pub(crate) fn gdn_decode_layer(
    ctx: &CudaContext,
    gdn: &GdnLayer,
    scratch: &mut Scratch,
    state: &mut GdnState,
) {
    ops::rms_norm(ctx, &scratch.hidden, &gdn.input_norm, &scratch.normalized, EPSILON).unwrap();
    fp8_projection(
        ctx,
        &gdn.qkv,
        &scratch.normalized,
        &scratch.fp8_activation,
        &mut scratch.gdn_qkv,
    );
    fp8_projection(
        ctx,
        &gdn.z,
        &scratch.normalized,
        &scratch.fp8_activation,
        &mut scratch.gdn_z,
    );
    let qkv_flat = view(&scratch.gdn_qkv, vec![QKV_WIDTH], DType::BF16);
    ops::gdn_causal_conv_step(
        ctx,
        &state.conv_window,
        &qkv_flat,
        &gdn.conv_weight,
        &scratch.gdn_conv,
    )
    .unwrap();
    let (q, k, v) = split_gdn_qkv(ctx, &scratch.gdn_conv);
    ops::gdn_l2_normalize_heads(ctx, &q, EPSILON).unwrap();
    ops::gdn_l2_normalize_heads(ctx, &k, EPSILON).unwrap();
    bf16_matvec(ctx, &gdn.in_proj_a, &scratch.normalized, &scratch.gdn_a);
    bf16_matvec(ctx, &gdn.in_proj_b, &scratch.normalized, &scratch.gdn_b);
    ops::gdn_decay_and_beta(
        ctx,
        &scratch.gdn_a,
        &scratch.gdn_b,
        &gdn.a_log,
        &gdn.dt_bias,
        &scratch.gdn_decay,
        &scratch.gdn_beta,
    )
    .unwrap();
    ops::gdn_recurrent_step(
        ctx,
        &state.recurrent,
        &q,
        &k,
        &v,
        &scratch.gdn_decay,
        &scratch.gdn_beta,
        &scratch.gdn_readout,
        GDN_K_HEADS,
    )
    .unwrap();
    ops::gdn_gated_norm(
        ctx,
        &scratch.gdn_readout,
        &gdn_z_heads(ctx, &scratch.gdn_z),
        &gdn.norm_weight,
        &scratch.gdn_gated,
        EPSILON,
    )
    .unwrap();
    let flat = flatten(ctx, &scratch.gdn_gated, Z_WIDTH);
    fp8_projection(
        ctx,
        &gdn.out,
        &flat,
        &scratch.gdn_fp8,
        &mut scratch.projected,
    );
    ops::add_into(ctx, &scratch.projected, &scratch.hidden).unwrap();
}

pub(crate) fn assert_scan_close(label: &str, candidate: &Tensor, reference: &Tensor, dtype: DType) {
    let decode = |tensor: &Tensor| -> Vec<f64> {
        let bytes = graph_tensor_bytes(tensor);
        match dtype {
            DType::F32 => bytes.chunks_exact(4).map(|value| {
                f32::from_le_bytes(value.try_into().unwrap()) as f64
            }).collect(),
            DType::BF16 => bytes.chunks_exact(2).map(|value| {
                half::bf16::from_bits(u16::from_le_bytes(value.try_into().unwrap())).to_f32() as f64
            }).collect(),
            _ => panic!("unsupported comparison dtype"),
        }
    };
    let actual = decode(candidate);
    let expected = decode(reference);
    assert!(!actual.is_empty());
    assert_eq!(actual.len(), expected.len());
    let mut actual_norm = 0.0;
    let mut expected_norm = 0.0;
    let mut dot = 0.0;
    let mut error = 0.0;
    for (index, (&value, &baseline)) in actual.iter().zip(&expected).enumerate() {
        assert!(value.is_finite() && baseline.is_finite(),
            "{label}: nonfinite index={index} candidate={value} reference={baseline}");
        actual_norm += value * value;
        expected_norm += baseline * baseline;
        dot += value * baseline;
        error += (value - baseline) * (value - baseline);
    }
    assert!(expected_norm > 0.0 && actual_norm > 0.0, "{label}: vacuous zero comparison");
    let cosine = dot / (actual_norm * expected_norm).sqrt();
    let relative = (error / expected_norm).sqrt();
    println!("scan validation {label}: cosine={cosine:.9}, relL2={relative:.9}");
    assert!(cosine >= 0.9999 && relative <= 0.01,
        "{label}: scan mismatch, cosine={cosine}, relL2={relative}");
}

pub(crate) fn verify_finite_bf16(ctx: &CudaContext, label: &str, tensor: &Tensor) {
    if std::env::var("APXINF_QWEN38_VERIFY_FINITE").as_deref() != Ok("1") {
        return;
    }
    ctx.synchronize().unwrap();
    let bytes = graph_tensor_bytes(tensor);
    assert!(!bytes.is_empty(), "{label}: empty tensor");
    for (index, bytes) in bytes.chunks_exact(2).enumerate() {
        let value = half::bf16::from_bits(u16::from_le_bytes(bytes.try_into().unwrap())).to_f32();
        assert!(value.is_finite(), "{label}: nonfinite index={index} value={value}");
    }
    println!("finite validation {label}: {} elements", bytes.len() / 2);
}

#[allow(dead_code)] // diagnostic: session-replay bit-compare, kept for kernel debugging
pub(crate) fn prefill_snapshot(
    ctx: &CudaContext,
    prefill: &PrefillScratch,
    scratch: &Scratch,
    states: &[GdnState],
    caches: &[KvCache],
) -> Vec<(String, Vec<u8>)> {
    ctx.synchronize().unwrap();
    let mut snapshot = vec![
        ("hidden".into(), graph_tensor_bytes(&prefill.hidden)),
        ("logits".into(), graph_tensor_bytes(&scratch.logits)),
    ];
    for (index, state) in states.iter().enumerate() {
        snapshot.push((format!("gdn{index}/state"), graph_tensor_bytes(&state.recurrent)));
        snapshot.push((format!("gdn{index}/conv"), graph_tensor_bytes(&state.conv_window)));
    }
    for (index, cache) in caches.iter().enumerate() {
        snapshot.push((format!("attention{index}/key"), graph_tensor_bytes(&cache.keys)));
        snapshot.push((format!("attention{index}/value"), graph_tensor_bytes(&cache.values)));
    }
    snapshot
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_step(
    ctx: &CudaContext,
    model: &Model,
    scratch: &mut Scratch,
    gdn_states: &mut [GdnState],
    kv_caches: &mut [KvCache],
    position: usize,
) {
    decode_step_inner(ctx, model, scratch, gdn_states, kv_caches, position, None, None);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_step_inner(
    ctx: &CudaContext,
    model: &Model,
    scratch: &mut Scratch,
    gdn_states: &mut [GdnState],
    kv_caches: &mut [KvCache],
    position: usize,
    mlp_graphs: Option<&[CapturedGraph]>,
    gdn_graphs: Option<&[Option<CapturedGraph>]>,
) {
    ops::embedding_gather(ctx, &model.embedding, &scratch.token, &scratch.hidden).unwrap();

    let rotary = ops::rotary_dim(HEAD_DIM, PARTIAL_ROTARY);
    let mut gdn_index = 0usize;
    let mut attention_index = 0usize;

    // APXINF_QWEN38_LAYER_TIMING=1 splits a step across the two layer kinds.
    // It synchronizes per layer, which inflates the total -- read the ratio,
    // not the absolute.
    let timing = std::env::var("APXINF_QWEN38_LAYER_TIMING").is_ok();
    let mut attention_time = 0.0f64;
    let mut gdn_time = 0.0f64;

    for (layer_index, layer) in model.layers.iter().enumerate() {
        let layer_start = timing.then(Instant::now);
        match layer {
            Layer::Attention(attention) => {
                let cache = &mut kv_caches[attention_index];
                attention_index += 1;

                ops::rms_norm(ctx, &scratch.hidden, &attention.input_norm, &scratch.normalized, EPSILON).unwrap();
                fp8_projection(ctx, &attention.q, &scratch.normalized, &scratch.fp8_activation, &mut scratch.qkv_fused);
                let fused_heads = view(&scratch.qkv_fused, vec![1, HEADS, 2 * HEAD_DIM], DType::BF16);
                ops::split_query_and_gate(ctx, &fused_heads, &scratch.query, &scratch.query_gate).unwrap();

                // k and v project directly into this token's cache slot.
                let mut key_slot = cache_slot(&cache.keys, position, vec![1, KV_HEADS * HEAD_DIM]);
                let mut value_slot = cache_slot(&cache.values, position, vec![1, KV_HEADS * HEAD_DIM]);
                if reuse_fp8_enabled() && attention.k.input_scale == attention.q.input_scale {
                    fp8_projection_reuse_quantized(ctx, &attention.k, &scratch.fp8_activation, &mut key_slot);
                } else {
                    fp8_projection(ctx, &attention.k, &scratch.normalized, &scratch.fp8_activation, &mut key_slot);
                }
                if reuse_fp8_enabled() && attention.v.input_scale == attention.q.input_scale {
                    fp8_projection_reuse_quantized(ctx, &attention.v, &scratch.fp8_activation, &mut value_slot);
                } else {
                    fp8_projection(ctx, &attention.v, &scratch.normalized, &scratch.fp8_activation, &mut value_slot);
                }

                let key_heads = cache_slot(&cache.keys, position, vec![KV_HEADS, HEAD_DIM]);
                let query_heads = view(&scratch.query, vec![HEADS, HEAD_DIM], DType::BF16);
                ops::head_rms_norm(ctx, &query_heads, &attention.q_norm, EPSILON).unwrap();
                ops::head_rms_norm(ctx, &key_heads, &attention.k_norm, EPSILON).unwrap();

                let query_tokens = view(&scratch.query, vec![1, HEADS, HEAD_DIM], DType::BF16);
                let key_tokens = cache_slot(&cache.keys, position, vec![1, KV_HEADS, HEAD_DIM]);
                ops::partial_rope(ctx, &query_tokens, &scratch.positions, rotary, ROPE_THETA).unwrap();
                ops::partial_rope(ctx, &key_tokens, &scratch.positions, rotary, ROPE_THETA).unwrap();

                // The cache is allocated at full capacity; attention reads only
                // the tokens written so far.
                let valid = position + 1;
                // Viewed at `valid`, not at the full allocation: the FA2
                // candidate requires key_capacity == key_tokens, and a
                // capacity-shaped view fails that for every step but the last,
                // dropping decode onto the naive kernel whose cost grows with
                // KV length.
                let keys = cache_rows(&cache.keys, valid, vec![1, valid, KV_HEADS, HEAD_DIM]);
                let values = cache_rows(&cache.values, valid, vec![1, valid, KV_HEADS, HEAD_DIM]);
                let query_4d = view(&scratch.query, vec![1, 1, HEADS, HEAD_DIM], DType::BF16);
                let mut out_4d = view(&scratch.attention_out, vec![1, 1, HEADS, HEAD_DIM], DType::BF16);
                let mut args = ops::KvCacheAttentionArgs::new(&query_4d, &keys, &values, &mut out_4d);
                args.valid_key_tokens = valid;
                args.query_start = position;
            args.policy.cache_dir = attention_cache_dir();
                ops::kv_cache_attention(ctx, args).unwrap();

                let gate_flat = view(&scratch.query_gate, vec![1, HEADS * HEAD_DIM], DType::BF16);
                ops::apply_output_gate(ctx, &scratch.attention_out, &gate_flat).unwrap();
                fp8_projection(ctx, &attention.o, &scratch.attention_out, &scratch.attention_fp8, &mut scratch.projected);
                ops::add_into(ctx, &scratch.projected, &scratch.hidden).unwrap();

                if let Some(graphs) = mlp_graphs {
                    graphs[layer_index].replay().unwrap();
                } else {
                    nvfp4_mlp(ctx, &attention.gate_up, &attention.down, &attention.post_norm, scratch);
                }
            }
            Layer::Gdn(gdn) => {
                let state_index = gdn_index;
                gdn_index += 1;
                if let Some(graphs) = gdn_graphs {
                    if let Some(graph) = &graphs[layer_index] {
                        graph.replay().unwrap();
                    } else {
                        gdn_decode_layer(ctx, gdn, scratch, &mut gdn_states[state_index]);
                        nvfp4_mlp(ctx, &gdn.gate_up, &gdn.down, &gdn.post_norm, scratch);
                    }
                } else {
                    gdn_decode_layer(ctx, gdn, scratch, &mut gdn_states[state_index]);
                    if let Some(graphs) = mlp_graphs {
                        graphs[layer_index].replay().unwrap();
                    } else {
                        nvfp4_mlp(ctx, &gdn.gate_up, &gdn.down, &gdn.post_norm, scratch);
                    }
                }
            }
        }
        if let Some(start) = layer_start {
            ctx.synchronize().unwrap();
            let elapsed = start.elapsed().as_secs_f64();
            match layer {
                Layer::Attention(_) => attention_time += elapsed,
                Layer::Gdn(_) => gdn_time += elapsed,
            }
        }
        let _ = layer_index;
    }

    if timing {
        println!(
            "  layer split: attention {:7.2} ms ({} layers)   gdn {:7.2} ms ({} layers)",
            attention_time * 1e3,
            attention_index,
            gdn_time * 1e3,
            gdn_index
        );
    }

    ops::rms_norm(ctx, &scratch.hidden, &model.final_norm, &scratch.normalized, EPSILON).unwrap();
    ops::nvfp4_quantize_activation(ctx, &scratch.normalized, &scratch.nvfp4_activation, &scratch.nvfp4_scales, model.lm_head.input_scale, BLOCK, ops::ScaleLayout::GemmAtom).unwrap();
    nvfp4_decode_gemm(
        ctx,
        ops::GemmArgs::nvfp4(
            &scratch.nvfp4_activation,
            &scratch.nvfp4_scales,
            &model.lm_head.packed,
            &model.lm_head.scales,
            BLOCK,
            model.lm_head.alpha,
            &mut scratch.logits,
        ),
    )
    .unwrap();
    ops::argmax(ctx, &scratch.logits, &scratch.next_token).unwrap();
}

/// Dump per-layer hidden states for a single-token decode (token id 100,
/// position 0, empty state) so a PyTorch reference can be compared tensor by
/// tensor. Writes raw little-endian f32 files under devlocal/qwen38-nvfp4/apxdump.
fn gemm_cache_dir() -> Option<String> {
    Some(
        std::env::var("APXINF_QWEN38_GEMM_TUNE_CACHE")
            .unwrap_or_else(|_| "/tmp/apxinf-qwen38-gemm-recipes".to_string()),
    )
}

fn attention_cache_dir() -> Option<String> {
    Some(
        std::env::var("APXINF_QWEN38_TUNE_CACHE")
            .unwrap_or_else(|_| "/tmp/apxinf-qwen38-attention-recipes".to_string()),
    )
}

pub(crate) fn flatten(_ctx: &CudaContext, tensor: &Tensor, width: usize) -> Tensor {
    view(tensor, vec![1, width], DType::BF16)
}

pub(crate) fn gdn_z_heads(_ctx: &CudaContext, z: &Tensor) -> Tensor {
    view(z, vec![GDN_V_HEADS, GDN_HEAD_DIM], DType::BF16)
}

/// q, k and v live in one [10240] projection: 16*128 q, then 16*128 k, then
/// 48*128 v. The views share storage rather than copying.
pub(crate) fn split_gdn_qkv(_ctx: &CudaContext, qkv: &Tensor) -> (Tensor, Tensor, Tensor) {
    let buffer = CudaBuffer::from_tensor(qkv).unwrap();
    let element = DType::BF16.size_in_bytes();
    let q_len = GDN_K_HEADS * GDN_HEAD_DIM;
    let v_len = GDN_V_HEADS * GDN_HEAD_DIM;
    let q = buffer
        .view(0, q_len * element)
        .unwrap()
        .as_tensor(Shape::new(vec![GDN_K_HEADS, GDN_HEAD_DIM]), DType::BF16)
        .unwrap();
    let k = buffer
        .view(q_len * element, q_len * element)
        .unwrap()
        .as_tensor(Shape::new(vec![GDN_K_HEADS, GDN_HEAD_DIM]), DType::BF16)
        .unwrap();
    let v = buffer
        .view(2 * q_len * element, v_len * element)
        .unwrap()
        .as_tensor(Shape::new(vec![GDN_V_HEADS, GDN_HEAD_DIM]), DType::BF16)
        .unwrap();
    (q, k, v)
}

/// Working buffers for a whole prompt, sized to `seq_padded` rows.
///
/// Every buffer is allocated zeroed and the operators only ever write the
/// first `tokens` rows, so the padding rows stay zero for the life of the
/// run. That is what makes the chunk scan safe to call on a padded sequence:
/// `g = 0` and `beta = 0` make the gated delta rule
/// `state <- state * exp(g) + beta * (...)` the identity, so padding cannot
/// move the recurrent state. Note this cannot be arranged by zeroing `a`
/// instead -- `decay = -exp(a_log) * softplus(a + dt_bias)` is nonzero at
/// `a = 0` -- which is exactly the kind of mistake that would corrupt the
/// state silently.
pub(crate) struct PrefillScratch {
    capacity: usize,
    hidden: Tensor,
    normalized: Tensor,
    fp8_activation: Tensor,
    nvfp4_activation: Tensor,
    nvfp4_scales: Tensor,
    mlp_fused: Tensor,
    mlp_activation: Tensor,
    mlp_scales: Tensor,
    mlp_out: Tensor,
    qkv_fused: Tensor,
    query: Tensor,
    query_gate: Tensor,
    attention_out: Tensor,
    attention_fp8: Tensor,
    projected: Tensor,
    pub(crate) positions: Tensor,
    gdn_qkv: Tensor,
    gdn_conv: Tensor,
    gdn_z: Tensor,
    gdn_a: Tensor,
    gdn_b: Tensor,
    gdn_decay: Tensor,
    gdn_beta: Tensor,
    gdn_readout: Tensor,
    gdn_gated: Tensor,
    gdn_fp8: Tensor,
    pub(crate) tokens: Tensor,
    // Staging for the FlashInfer chunked scan, which takes q/k/v as separate
    // FP16 tensors and a linear-space decay.
    gdn_q16: Option<Tensor>,
    gdn_k16: Option<Tensor>,
    gdn_v16: Option<Tensor>,
    gdn_out16: Option<Tensor>,
    gdn_alpha: Option<Tensor>,
    cu_seqlens: Option<Tensor>,
    flashinfer_workspace: Option<Tensor>,
}

impl PrefillScratch {
    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    pub(crate) fn new(ctx: &CudaContext, capacity: usize) -> PrefillScratch {
        assert_eq!(capacity % CHUNK, 0, "prefill capacity must be a multiple of {CHUNK}");
        let scale_bytes =
            |rows: usize, k: usize| vec![ops::nvfp4_scale_buffer_bytes(rows, k, BLOCK).unwrap()];
        let t = capacity;
        // Prefill always runs the vendored FlashInfer chunked scan; the
        // reference scan remains available in the kernel harness for
        // precision comparison.
        let flashinfer = true;
        PrefillScratch {
            capacity,
            hidden: zeros(ctx, vec![t, HIDDEN], DType::BF16),
            normalized: zeros(ctx, vec![t, HIDDEN], DType::BF16),
            fp8_activation: zeros(ctx, vec![t, HIDDEN], DType::F8E4M3),
            nvfp4_activation: zeros(ctx, vec![t, HIDDEN / 2], DType::E2M1Pair),
            nvfp4_scales: zeros(ctx, scale_bytes(t, HIDDEN), DType::F8E4M3),
            mlp_fused: zeros(ctx, vec![t, 2 * INTERMEDIATE], DType::BF16),
            mlp_activation: zeros(ctx, vec![t, INTERMEDIATE / 2], DType::E2M1Pair),
            mlp_scales: zeros(ctx, scale_bytes(t, INTERMEDIATE), DType::F8E4M3),
            mlp_out: zeros(ctx, vec![t, HIDDEN], DType::BF16),
            qkv_fused: zeros(ctx, vec![t, 2 * HEADS * HEAD_DIM], DType::BF16),
            query: zeros(ctx, vec![t, HEADS, HEAD_DIM], DType::BF16),
            query_gate: zeros(ctx, vec![t, HEADS, HEAD_DIM], DType::BF16),
            attention_out: zeros(ctx, vec![t, HEADS * HEAD_DIM], DType::BF16),
            attention_fp8: zeros(ctx, vec![t, HEADS * HEAD_DIM], DType::F8E4M3),
            projected: zeros(ctx, vec![t, HIDDEN], DType::BF16),
            positions: zeros(ctx, vec![t], DType::I32),
            gdn_qkv: zeros(ctx, vec![t, QKV_WIDTH], DType::BF16),
            gdn_conv: zeros(ctx, vec![t, QKV_WIDTH], DType::BF16),
            gdn_z: zeros(ctx, vec![t, Z_WIDTH], DType::BF16),
            gdn_a: zeros(ctx, vec![t, GDN_V_HEADS], DType::BF16),
            gdn_b: zeros(ctx, vec![t, GDN_V_HEADS], DType::BF16),
            gdn_decay: zeros(ctx, vec![t, GDN_V_HEADS], DType::F32),
            gdn_beta: zeros(ctx, vec![t, GDN_V_HEADS], DType::F32),
            gdn_readout: zeros(ctx, vec![t, GDN_V_HEADS, GDN_HEAD_DIM], DType::BF16),
            gdn_gated: zeros(ctx, vec![t, GDN_V_HEADS, GDN_HEAD_DIM], DType::BF16),
            gdn_fp8: zeros(ctx, vec![t, Z_WIDTH], DType::F8E4M3),
            tokens: zeros(ctx, vec![t], DType::I32),
            gdn_q16: flashinfer
                .then(|| zeros(ctx, vec![t, GDN_K_HEADS, GDN_HEAD_DIM], DType::F16)),
            gdn_k16: flashinfer
                .then(|| zeros(ctx, vec![t, GDN_K_HEADS, GDN_HEAD_DIM], DType::F16)),
            gdn_v16: flashinfer
                .then(|| zeros(ctx, vec![t, GDN_V_HEADS, GDN_HEAD_DIM], DType::F16)),
            gdn_out16: flashinfer
                .then(|| zeros(ctx, vec![t, GDN_V_HEADS, GDN_HEAD_DIM], DType::F16)),
            gdn_alpha: flashinfer.then(|| zeros(ctx, vec![t, GDN_V_HEADS], DType::F32)),
            cu_seqlens: flashinfer.then(|| zeros(ctx, vec![2], DType::I32)),
            flashinfer_workspace: flashinfer.then(|| {
                let bytes = ops::flashinfer_gdn_workspace_bytes(GDN_V_HEADS, 1);
                assert!(bytes > 0, "FlashInfer workspace query returned 0");
                zeros(ctx, vec![bytes.div_ceil(4)], DType::F32)
            }),
        }
    }
}

/// `[rows, hidden] x [hidden, cols] -> [rows, cols]`, in BF16.
///
/// The row-count generalization of `bf16_matvec`; `weight` is `[K, N]` for the
/// same reason.
fn bf16_matmul_rows(
    ctx: &CudaContext,
    weight: &Tensor,
    input: &Tensor,
    output: &Tensor,
    rows: usize,
) {
    let dims = weight.shape().dims().to_vec();
    let source = prefix(input, vec![rows, dims[0]], DType::BF16);
    let mut out = prefix(output, vec![rows, dims[1]], DType::BF16);
    ops::gemm(ctx, ops::GemmArgs::new(&source, weight, &mut out)).unwrap();
}

/// The MLP over `rows` tokens: `nvfp4_mlp` with a row count.
fn nvfp4_mlp_rows(
    ctx: &CudaContext,
    gate_up: &Nvfp4Weight,
    down: &Nvfp4Weight,
    norm_weight: &Tensor,
    scratch: &mut PrefillScratch,
    rows: usize,
) {
    let hidden_rows = prefix(&scratch.hidden, vec![rows, HIDDEN], DType::BF16);
    verify_finite_bf16(ctx, "prefill MLP input", &hidden_rows);
    let activation = prefix(&scratch.nvfp4_activation, vec![rows, HIDDEN / 2], DType::E2M1Pair);
    ops::nvfp4_quantize_rms_norm(
        ctx,
        &hidden_rows,
        norm_weight,
        &activation,
        &scratch.nvfp4_scales,
        EPSILON,
        gate_up.input_scale,
        BLOCK,
        ops::ScaleLayout::GemmAtom,
    )
    .unwrap();

    let mut fused = prefix(&scratch.mlp_fused, vec![rows, 2 * INTERMEDIATE], DType::BF16);
    if std::env::var("APXINF_QWEN38_VERIFY_FINITE").as_deref() == Ok("1") {
        ctx.synchronize().unwrap();
        for (index, code) in graph_tensor_bytes(&scratch.nvfp4_scales).into_iter().enumerate() {
            assert!(code & 0x7f != 0x7f, "prefill MLP activation scale is E4M3 NaN: index={index} code={code:#x}");
        }
    }
    let mut args = ops::GemmArgs::nvfp4(
        &activation,
        &scratch.nvfp4_scales,
        &gate_up.packed,
        &gate_up.scales,
        BLOCK,
        gate_up.alpha,
        &mut fused,
    );
    args.policy.cache_dir = gemm_cache_dir();
    ops::gemm(ctx, args).unwrap();

    let mlp_activation =
        prefix(&scratch.mlp_activation, vec![rows, INTERMEDIATE / 2], DType::E2M1Pair);
    verify_finite_bf16(ctx, "prefill MLP gate-up", &fused);
    ops::nvfp4_quantize_swiglu(
        ctx,
        &fused,
        &mlp_activation,
        &scratch.mlp_scales,
        down.input_scale,
        BLOCK,
        ops::ScaleLayout::GemmAtom,
    )
    .unwrap();

    let mut mlp_out = prefix(&scratch.mlp_out, vec![rows, HIDDEN], DType::BF16);
    let mut args = ops::GemmArgs::nvfp4(
        &mlp_activation,
        &scratch.mlp_scales,
        &down.packed,
        &down.scales,
        BLOCK,
        down.alpha,
        &mut mlp_out,
    );
    args.policy.cache_dir = gemm_cache_dir();
    ops::gemm(ctx, args).unwrap();

    let residual = prefix(&scratch.hidden, vec![rows, HIDDEN], DType::BF16);
    verify_finite_bf16(ctx, "prefill MLP down", &mlp_out);
    ops::add_into(ctx, &mlp_out, &residual).unwrap();
}

/// Run a whole prompt through the model in one pass per layer.
///
/// This is the batched counterpart of `decode_step`. Attention takes all
/// `tokens` queries at once against the cache it just filled; GDN replaces the
/// per-token recurrence with the chunked scan. Both leave exactly the state
/// the single-token path would have left, so decode continues from `tokens`
/// without re-running anything.
///
/// The scan needs a multiple of `CHUNK` rows, so it reads `seq_padded`; every
/// other operator is given the true `tokens`. The padding rows are never
/// written and stay zero -- see `PrefillScratch`.
pub(crate) fn prefill_step(
    ctx: &CudaContext,
    model: &Model,
    scratch: &mut PrefillScratch,
    gdn_states: &mut [GdnState],
    kv_caches: &mut [KvCache],
    tokens: usize,
) {
    assert!(tokens > 0 && tokens <= scratch.capacity, "prompt exceeds prefill capacity");
    let seq_padded = tokens.div_ceil(CHUNK) * CHUNK;

    let token_ids = prefix(&scratch.tokens, vec![tokens], DType::I32);
    let hidden_rows = prefix(&scratch.hidden, vec![tokens, HIDDEN], DType::BF16);
    ops::embedding_gather(ctx, &model.embedding, &token_ids, &hidden_rows).unwrap();

    let rotary = ops::rotary_dim(HEAD_DIM, PARTIAL_ROTARY);
    let mut gdn_index = 0usize;
    let mut attention_index = 0usize;

    let timing = std::env::var("APXINF_QWEN38_LAYER_TIMING").is_ok();
    let mut attention_time = 0.0f64;
    let mut gdn_time = 0.0f64;

    for (layer_index, layer) in model.layers.iter().enumerate() {
        let layer_start = timing.then(Instant::now);
        verify_finite_bf16(ctx, &format!("layer{layer_index}/input"), &hidden_rows);
        match layer {
            Layer::Attention(attention) => {
                let cache = &mut kv_caches[attention_index];
                attention_index += 1;

                let hidden = prefix(&scratch.hidden, vec![tokens, HIDDEN], DType::BF16);
                let normalized = prefix(&scratch.normalized, vec![tokens, HIDDEN], DType::BF16);
                ops::rms_norm(ctx, &hidden, &attention.input_norm, &normalized, EPSILON).unwrap();

                if reuse_fp8_enabled() {
                    // q, k and v read the same post-norm activation, so it is
                    // quantized once for all three.
                    quantize_shared_fp8_activation(
                        ctx,
                        &[&attention.q, &attention.k, &attention.v],
                        &normalized,
                        &scratch.fp8_activation,
                        tokens,
                    );
                    fp8_projection_prequantized(
                        ctx,
                        &attention.q,
                        &scratch.fp8_activation,
                        &mut scratch.qkv_fused,
                        tokens,
                    );
                } else {
                    fp8_projection_rows(
                        ctx,
                        &attention.q,
                        &normalized,
                        &scratch.fp8_activation,
                        &mut scratch.qkv_fused,
                        tokens,
                    );
                }
                let fused_heads =
                    prefix(&scratch.qkv_fused, vec![tokens, HEADS, 2 * HEAD_DIM], DType::BF16);
                let query = prefix(&scratch.query, vec![tokens, HEADS, HEAD_DIM], DType::BF16);
                let query_gate =
                    prefix(&scratch.query_gate, vec![tokens, HEADS, HEAD_DIM], DType::BF16);
                ops::split_query_and_gate(ctx, &fused_heads, &query, &query_gate).unwrap();

                // Rows 0..tokens of the cache are a contiguous prefix, so k and
                // v project straight into it exactly as decode does per token.
                let mut key_rows = cache_rows(&cache.keys, tokens, vec![tokens, KV_HEADS * HEAD_DIM]);
                let mut value_rows =
                    cache_rows(&cache.values, tokens, vec![tokens, KV_HEADS * HEAD_DIM]);
                if reuse_fp8_enabled() {
                    fp8_projection_prequantized(ctx, &attention.k, &scratch.fp8_activation, &mut key_rows, tokens);
                    fp8_projection_prequantized(ctx, &attention.v, &scratch.fp8_activation, &mut value_rows, tokens);
                } else {
                    fp8_projection_rows(ctx, &attention.k, &normalized, &scratch.fp8_activation, &mut key_rows, tokens);
                    fp8_projection_rows(ctx, &attention.v, &normalized, &scratch.fp8_activation, &mut value_rows, tokens);
                }

                // Per-head RMSNorm is independent per row, so the token axis
                // folds into the head axis.
                let query_heads = prefix(&scratch.query, vec![tokens * HEADS, HEAD_DIM], DType::BF16);
                let key_heads = cache_rows(&cache.keys, tokens, vec![tokens * KV_HEADS, HEAD_DIM]);
                ops::head_rms_norm(ctx, &query_heads, &attention.q_norm, EPSILON).unwrap();
                ops::head_rms_norm(ctx, &key_heads, &attention.k_norm, EPSILON).unwrap();

                let positions = prefix(&scratch.positions, vec![tokens], DType::I32);
                // RoPE takes [tokens, heads, head_dim]; the batch axis the
                // attention call wants is added separately below.
                let query_tokens = prefix(&scratch.query, vec![tokens, HEADS, HEAD_DIM], DType::BF16);
                let key_tokens = cache_rows(&cache.keys, tokens, vec![tokens, KV_HEADS, HEAD_DIM]);
                ops::partial_rope(ctx, &query_tokens, &positions, rotary, ROPE_THETA).unwrap();
                ops::partial_rope(ctx, &key_tokens, &positions, rotary, ROPE_THETA).unwrap();
                verify_finite_bf16(ctx, &format!("layer{layer_index}/query"), &query_tokens);
                verify_finite_bf16(ctx, &format!("layer{layer_index}/key"), &key_tokens);
                verify_finite_bf16(ctx, &format!("layer{layer_index}/value"), &value_rows);

                // One call for all queries: the causal mask already places
                // query i at query_start + i, so the prompt needs no masking
                // work of its own.
                // FA2 multi-query prefill requires key_capacity == key_tokens, so the
                // cache is viewed at exactly the tokens written, not its full alloc.
                let keys = cache_rows(&cache.keys, tokens, vec![1, tokens, KV_HEADS, HEAD_DIM]);
                let values = cache_rows(&cache.values, tokens, vec![1, tokens, KV_HEADS, HEAD_DIM]);
                let query_4d = prefix(&scratch.query, vec![1, tokens, HEADS, HEAD_DIM], DType::BF16);
                let mut out_4d = prefix(&scratch.attention_out, vec![1, tokens, HEADS, HEAD_DIM], DType::BF16);
                let mut args = ops::KvCacheAttentionArgs::new(&query_4d, &keys, &values, &mut out_4d);
                args.valid_key_tokens = tokens;
                args.query_start = 0;
                args.policy.cache_dir = attention_cache_dir();
                ops::kv_cache_attention(ctx, args).unwrap();
                verify_finite_bf16(ctx, &format!("layer{layer_index}/attention"), &out_4d);

                let gate_flat = prefix(&scratch.query_gate, vec![tokens, HEADS * HEAD_DIM], DType::BF16);
                let attention_out = prefix(&scratch.attention_out, vec![tokens, HEADS * HEAD_DIM], DType::BF16);
                ops::apply_output_gate(ctx, &attention_out, &gate_flat).unwrap();
                fp8_projection_rows(ctx, &attention.o, &attention_out, &scratch.attention_fp8, &mut scratch.projected, tokens);

                let projected = prefix(&scratch.projected, vec![tokens, HIDDEN], DType::BF16);
                let residual = prefix(&scratch.hidden, vec![tokens, HIDDEN], DType::BF16);
                ops::add_into(ctx, &projected, &residual).unwrap();
                verify_finite_bf16(ctx, &format!("layer{layer_index}/attention-residual"), &residual);

                nvfp4_mlp_rows(ctx, &attention.gate_up, &attention.down, &attention.post_norm, scratch, tokens);
            }
            Layer::Gdn(gdn) => {
                let state = &mut gdn_states[gdn_index];
                gdn_index += 1;

                // APXINF_QWEN38_GDN_STAGES=1 times the stages inside a GDN
                // layer. It synchronizes between them, so the total inflates;
                // the split is what it is for.
                let stages = std::env::var("APXINF_QWEN38_GDN_STAGES").is_ok();
                let mut mark = stages.then(Instant::now);
                let mut stage = |label: &str, mark: &mut Option<Instant>| {
                    if let Some(start) = mark {
                        ctx.synchronize().unwrap();
                        let elapsed = start.elapsed().as_secs_f64() * 1e3;
                        if gdn_index == 1 {
                            println!("    gdn stage {label:<14} {elapsed:8.3} ms");
                        }
                        *mark = Some(Instant::now());
                    }
                };

                let hidden = prefix(&scratch.hidden, vec![tokens, HIDDEN], DType::BF16);
                let normalized = prefix(&scratch.normalized, vec![tokens, HIDDEN], DType::BF16);
                ops::rms_norm(ctx, &hidden, &gdn.input_norm, &normalized, EPSILON).unwrap();

                if reuse_fp8_enabled() {
                    // qkv and z read the same post-norm activation, so it is
                    // quantized once for both.
                    quantize_shared_fp8_activation(
                        ctx, &[&gdn.qkv, &gdn.z], &normalized, &scratch.fp8_activation, tokens,
                    );
                    fp8_projection_prequantized(ctx, &gdn.qkv, &scratch.fp8_activation, &mut scratch.gdn_qkv, tokens);
                    fp8_projection_prequantized(ctx, &gdn.z, &scratch.fp8_activation, &mut scratch.gdn_z, tokens);
                } else {
                    fp8_projection_rows(ctx, &gdn.qkv, &normalized, &scratch.fp8_activation, &mut scratch.gdn_qkv, tokens);
                    fp8_projection_rows(ctx, &gdn.z, &normalized, &scratch.fp8_activation, &mut scratch.gdn_z, tokens);
                }
                verify_finite_bf16(ctx, &format!("layer{layer_index}/z"), &prefix(&scratch.gdn_z, vec![tokens, Z_WIDTH], DType::BF16));
                stage("qkv+z proj", &mut mark);

                // One launch over the prompt, and it reseeds the window so the
                // first decode step continues correctly.
                let qkv_rows = prefix(&scratch.gdn_qkv, vec![tokens, QKV_WIDTH], DType::BF16);
                let conv_rows = prefix(&scratch.gdn_conv, vec![tokens, QKV_WIDTH], DType::BF16);
                // Fused conv+prepare is the validated default (round 38); the
                // reference conv still runs when a diagnostic needs its output.
                let fuse_conv_prepare = scratch.gdn_q16.is_some();
                let conv_reference_needed = std::env::var("APXINF_QWEN38_VERIFY_CAKE").is_ok()
                    || std::env::var("APXINF_QWEN38_FP16_PROBE").is_ok();
                if !fuse_conv_prepare || conv_reference_needed {
                    ops::gdn_causal_conv_forward(
                    ctx,
                    &qkv_rows,
                    &gdn.conv_weight,
                    &conv_rows,
                    Some(&state.conv_window),
                    tokens,
                    QKV_WIDTH,
                    CONV_WIDTH,
                )
                .unwrap();
                }
                stage("conv", &mut mark);

                bf16_matmul_rows(ctx, &gdn.in_proj_a, &scratch.normalized, &scratch.gdn_a, tokens);
                bf16_matmul_rows(ctx, &gdn.in_proj_b, &scratch.normalized, &scratch.gdn_b, tokens);
                // Only the real rows are written; the padding keeps g = 0 and
                // beta = 0, which the scan treats as the identity.
                let a_rows = prefix(&scratch.gdn_a, vec![tokens, GDN_V_HEADS], DType::BF16);
                let b_rows = prefix(&scratch.gdn_b, vec![tokens, GDN_V_HEADS], DType::BF16);
                let decay_rows = prefix(&scratch.gdn_decay, vec![tokens, GDN_V_HEADS], DType::F32);
                let beta_rows = prefix(&scratch.gdn_beta, vec![tokens, GDN_V_HEADS], DType::F32);
                ops::gdn_decay_and_beta_seq(
                    ctx, &a_rows, &b_rows, &gdn.a_log, &gdn.dt_bias, &decay_rows, &beta_rows,
                    tokens, GDN_V_HEADS,
                )
                .unwrap();
                stage("a/b+decay", &mut mark);

                if gdn_index == 1 && std::env::var("APXINF_QWEN38_FP16_PROBE").is_ok() {
                    // The conv output carries q, k and v; v is the widest and
                    // the one a delta-rule update accumulates into.
                    let q_w = GDN_K_HEADS * GDN_HEAD_DIM;
                    let v_w = GDN_V_HEADS * GDN_HEAD_DIM;
                    probe_fp16_range("qkv(conv)", &scratch.gdn_conv, tokens * QKV_WIDTH);
                    let v_only = CudaBuffer::from_tensor(&scratch.gdn_conv)
                        .unwrap()
                        .view(2 * q_w * 2, v_w * 2)
                        .unwrap()
                        .as_tensor(Shape::new(vec![v_w]), DType::BF16)
                        .unwrap();
                    probe_fp16_range("v(token 0)", &v_only, v_w);
                    probe_fp16_range("z(gate)", &scratch.gdn_z, tokens * Z_WIDTH);
                }

                let mut fused_readout = None;
                if let (Some(q16), Some(k16), Some(v16), Some(out16), Some(alpha),
                        Some(cu), Some(workspace)) = (
                    scratch.gdn_q16.as_ref(), scratch.gdn_k16.as_ref(),
                    scratch.gdn_v16.as_ref(), scratch.gdn_out16.as_ref(),
                    scratch.gdn_alpha.as_ref(), scratch.cu_seqlens.as_ref(),
                    scratch.flashinfer_workspace.as_ref(),
                ) {
                    let shadow_state = if std::env::var("APXINF_QWEN38_VERIFY_CAKE").is_ok() {
                        ctx.synchronize().unwrap();
                        probe_fp16_range(&format!("GDN{gdn_index}/conv"), &scratch.gdn_conv, tokens * QKV_WIDTH);
                        probe_fp16_range(&format!("GDN{gdn_index}/a"), &a_rows, tokens * GDN_V_HEADS);
                        probe_fp16_range(&format!("GDN{gdn_index}/Alog"), &gdn.a_log, GDN_V_HEADS);
                        probe_fp16_range(&format!("GDN{gdn_index}/bias"), &gdn.dt_bias, GDN_V_HEADS);
                        let decay_values = graph_tensor_bytes(&decay_rows);
                        let (mut minimum, mut maximum) = (f32::INFINITY, f32::NEG_INFINITY);
                        for (index, bytes) in decay_values.chunks_exact(4).enumerate() {
                            let value = f32::from_le_bytes(bytes.try_into().unwrap());
                            assert!(value.is_finite(), "GDN{gdn_index}: nonfinite decay index={index} value={value}");
                            minimum = minimum.min(value);
                            maximum = maximum.max(value);
                        }
                        println!("scan input GDN{gdn_index}/decay: min={minimum:e} max={maximum:e}");
                        let reference = zeros(ctx, vec![GDN_V_HEADS, GDN_HEAD_DIM, GDN_HEAD_DIM], DType::F32);
                        CudaBuffer::from_tensor(&reference).unwrap()
                            .copy_from_host(&graph_tensor_bytes(&state.recurrent)).unwrap();
                        Some(reference)
                    } else {
                        None
                    };
                    // One pass turns the interleaved BF16 projection into the
                    // separate, L2-normalized, FP16 tensors this kernel wants,
                    // plus the linear-space decay.
                    let conv = prefix(&scratch.gdn_conv, vec![tokens, QKV_WIDTH], DType::BF16);
                    let decay = prefix(&scratch.gdn_decay, vec![tokens, GDN_V_HEADS], DType::F32);
                    let q_rows = prefix(q16, vec![tokens, GDN_K_HEADS, GDN_HEAD_DIM], DType::F16);
                    let k_rows = prefix(k16, vec![tokens, GDN_K_HEADS, GDN_HEAD_DIM], DType::F16);
                    let v_rows = prefix(v16, vec![tokens, GDN_V_HEADS, GDN_HEAD_DIM], DType::F16);
                    let out_rows = prefix(out16, vec![tokens, GDN_V_HEADS, GDN_HEAD_DIM], DType::F16);
                    let alpha_rows = prefix(alpha, vec![tokens, GDN_V_HEADS], DType::F32);
                    let beta_rows2 = prefix(&scratch.gdn_beta, vec![tokens, GDN_V_HEADS], DType::F32);
                    if fuse_conv_prepare {
                        ops::gdn_conv_prepare_flashinfer(
                            ctx, &qkv_rows, &gdn.conv_weight, &state.conv_window,
                            &q_rows, &k_rows, &v_rows, &decay, &alpha_rows,
                            tokens, GDN_K_HEADS, GDN_V_HEADS, EPSILON,
                        ).unwrap();
                    } else {
                    ops::gdn_prepare_flashinfer(
                        ctx, &conv, &q_rows, &k_rows, &v_rows, &decay, &alpha_rows,
                        tokens, QKV_WIDTH, GDN_K_HEADS, GDN_V_HEADS, GDN_HEAD_DIM,
                        EPSILON,
                    )
                    .unwrap();
                    }

                    // cu_seqlens for a single prompt. No padding needed: the
                    // kernel clamps its TMA descriptors per sequence.
                    CudaBuffer::from_tensor(cu)
                        .unwrap()
                        .copy_from_host(
                            &[0i32, tokens as i32]
                                .iter()
                                .flat_map(|value| value.to_le_bytes())
                                .collect::<Vec<u8>>(),
                        )
                        .unwrap();

                    ops::flashinfer_gdn_prefill(
                        ctx, &q_rows, &k_rows, &v_rows, &out_rows, &alpha_rows,
                        &beta_rows2, cu, &state.recurrent, workspace, tokens,
                        GDN_K_HEADS, GDN_V_HEADS, 1,
                        1.0 / (GDN_HEAD_DIM as f32).sqrt(),
                    )
                    .unwrap();

                    // Widen the readout back for the gated norm that follows.
                    let readout_bf =
                        prefix(&scratch.gdn_readout, vec![tokens, GDN_V_HEADS, GDN_HEAD_DIM], DType::BF16);
                    // Fused widen+gated-norm is the validated default
                    // (round 29); shadow verification still materializes the
                    // BF16 copy it compares against.
                    let fuse_norm = true;
                    if !fuse_norm || shadow_state.is_some() {
                        ops::convert_f16_to_bf16(
                            ctx, &out_rows, &readout_bf, tokens * GDN_V_HEADS * GDN_HEAD_DIM,
                        )
                        .unwrap();
                    }
                    if fuse_norm {
                        fused_readout = Some(out_rows);
                    }
                    stage("scan(flashinfer)", &mut mark);
                    if let Some(reference_state) = shadow_state {
                        let reference_output = zeros(ctx, vec![seq_padded, GDN_V_HEADS, GDN_HEAD_DIM], DType::BF16);
                        let fused = prefix(&scratch.gdn_conv, vec![seq_padded, QKV_WIDTH], DType::BF16);
                        let decay = prefix(&scratch.gdn_decay, vec![seq_padded, GDN_V_HEADS], DType::F32);
                        let beta = prefix(&scratch.gdn_beta, vec![seq_padded, GDN_V_HEADS], DType::F32);
                        let q_width = GDN_K_HEADS * GDN_HEAD_DIM;
                        ops::gdn_chunk_scan_interleaved(ctx, &fused, &decay, &beta,
                            &reference_output, &reference_state, seq_padded, QKV_WIDTH,
                            [0, q_width, 2 * q_width], GDN_V_HEADS, GDN_K_HEADS, CHUNK, GDN_HEAD_DIM).unwrap();
                        ctx.synchronize().unwrap();
                        let output_rows = prefix(&reference_output, vec![tokens, GDN_V_HEADS, GDN_HEAD_DIM], DType::BF16);
                        assert_scan_close(&format!("GDN{gdn_index}/output"), &readout_bf, &output_rows, DType::BF16);
                        assert_scan_close(&format!("GDN{gdn_index}/state"), &state.recurrent, &reference_state, DType::F32);
                    }
                } else {

                // q, k and v stay interleaved in the conv output; the scan
                // reads them in place through its row strides.
                let fused = prefix(&scratch.gdn_conv, vec![seq_padded, QKV_WIDTH], DType::BF16);
                let g_padded = prefix(&scratch.gdn_decay, vec![seq_padded, GDN_V_HEADS], DType::F32);
                let beta_padded = prefix(&scratch.gdn_beta, vec![seq_padded, GDN_V_HEADS], DType::F32);
                let readout = prefix(&scratch.gdn_readout, vec![seq_padded, GDN_V_HEADS, GDN_HEAD_DIM], DType::BF16);
                let q_width = GDN_K_HEADS * GDN_HEAD_DIM;
                ops::gdn_chunk_scan_interleaved(
                    ctx,
                    &fused,
                    &g_padded,
                    &beta_padded,
                    &readout,
                    &state.recurrent,
                    seq_padded,
                    QKV_WIDTH,
                    [0, q_width, 2 * q_width],
                    GDN_V_HEADS,
                    GDN_K_HEADS,
                    CHUNK,
                    GDN_HEAD_DIM,
                )
                .unwrap();
                stage("scan(ours)", &mut mark);
                }

                let readout_rows = fused_readout.unwrap_or_else(||
                    prefix(&scratch.gdn_readout, vec![tokens, GDN_V_HEADS, GDN_HEAD_DIM], DType::BF16));
                let z_heads = prefix(&scratch.gdn_z, vec![tokens, GDN_V_HEADS, GDN_HEAD_DIM], DType::BF16);
                let gated = prefix(&scratch.gdn_gated, vec![tokens, GDN_V_HEADS, GDN_HEAD_DIM], DType::BF16);
                ops::gdn_gated_norm_seq(
                    ctx, &readout_rows, &z_heads, &gdn.norm_weight, &gated, tokens,
                    GDN_V_HEADS, GDN_HEAD_DIM, EPSILON,
                )
                .unwrap();

                stage("gated norm", &mut mark);
                verify_finite_bf16(ctx, &format!("layer{layer_index}/gated"), &gated);

                let flat = prefix(&scratch.gdn_gated, vec![tokens, Z_WIDTH], DType::BF16);
                fp8_projection_rows(ctx, &gdn.out, &flat, &scratch.gdn_fp8, &mut scratch.projected, tokens);
                stage("out proj", &mut mark);

                let projected = prefix(&scratch.projected, vec![tokens, HIDDEN], DType::BF16);
                let residual = prefix(&scratch.hidden, vec![tokens, HIDDEN], DType::BF16);
                verify_finite_bf16(ctx, &format!("layer{layer_index}/gdn-projected"), &projected);
                ops::add_into(ctx, &projected, &residual).unwrap();

                nvfp4_mlp_rows(ctx, &gdn.gate_up, &gdn.down, &gdn.post_norm, scratch, tokens);
                stage("mlp", &mut mark);
            }
        }
        verify_finite_bf16(ctx, &format!("layer{layer_index}/output"), &hidden_rows);
        if let Some(start) = layer_start {
            ctx.synchronize().unwrap();
            let elapsed = start.elapsed().as_secs_f64();
            match layer {
                Layer::Attention(_) => attention_time += elapsed,
                Layer::Gdn(_) => gdn_time += elapsed,
            }
        }
    }

    if timing {
        println!(
            "  prefill split: attention {:8.2} ms ({} layers)   gdn {:8.2} ms ({} layers)",
            attention_time * 1e3, attention_index, gdn_time * 1e3, gdn_index
        );
    }
}

/// Logits and greedy argmax for the last row of a finished prefill.
///
/// Only one row is needed, so the lm_head runs at M=1 over a view of that row
/// rather than over the whole prompt: the other rows would cost a
/// 248320-wide projection each and are never read.
pub(crate) fn prefill_logits(
    ctx: &CudaContext,
    model: &Model,
    prefill: &PrefillScratch,
    decode: &Scratch,
    tokens: usize,
) {
    let last = view_row(&prefill.hidden, tokens - 1, HIDDEN, DType::BF16);
    ops::rms_norm(ctx, &last, &model.final_norm, &decode.normalized, EPSILON).unwrap();
    ops::nvfp4_quantize_activation(
        ctx,
        &decode.normalized,
        &decode.nvfp4_activation,
        &decode.nvfp4_scales,
        model.lm_head.input_scale,
        BLOCK,
        ops::ScaleLayout::GemmAtom,
    )
    .unwrap();
    let mut logits = view(&decode.logits, vec![1, VOCAB], DType::BF16);
    ops::gemm(
        ctx,
        ops::GemmArgs::nvfp4(
            &decode.nvfp4_activation,
            &decode.nvfp4_scales,
            &model.lm_head.packed,
            &model.lm_head.scales,
            BLOCK,
            model.lm_head.alpha,
            &mut logits,
        ),
    )
    .unwrap();
    ops::argmax(ctx, &decode.logits, &decode.next_token).unwrap();
}

/// A view of one row of a `[rows, width]` tensor.
pub(crate) fn view_row(tensor: &Tensor, row: usize, width: usize, dtype: DType) -> Tensor {
    let element = dtype.size_in_bytes();
    CudaBuffer::from_tensor(tensor)
        .unwrap()
        .view(row * width * element, width * element)
        .unwrap()
        .as_tensor(Shape::new(vec![1, width]), dtype)
        .unwrap()
}

/// A view of the first `tokens` rows of a KV cache.
pub(crate) fn cache_rows(cache_tensor: &Tensor, tokens: usize, dims: Vec<usize>) -> Tensor {
    let element = DType::BF16.size_in_bytes();
    let span = tokens * KV_HEADS * HEAD_DIM * element;
    CudaBuffer::from_tensor(cache_tensor)
        .unwrap()
        .view(0, span)
        .unwrap()
        .as_tensor(Shape::new(dims), DType::BF16)
        .unwrap()
}

/// `[1, hidden] x [hidden, heads] -> [1, heads]`, in BF16.
///
/// `weight` must already be `[K, N]`, which is what `GemmArgs::new` requires
/// and what `load_bf16_transposed` produces. Transposing here with a `view`
/// would only relabel the shape, leaving the GEMM reading permuted elements.
pub(crate) fn bf16_matvec(ctx: &CudaContext, weight: &Tensor, input: &Tensor, output: &Tensor) {
    let dims = weight.shape().dims().to_vec();
    let mut out = view(output, vec![1, dims[1]], DType::BF16);
    ops::gemm(ctx, ops::GemmArgs::new(input, weight, &mut out)).unwrap();
}

/// A view of one token's slot in a KV cache, shaped for the projection that
/// fills it. Writing the projection straight into the cache avoids a
/// device-to-device copy per layer per token.
pub(crate) fn cache_slot(cache_tensor: &Tensor, position: usize, dims: Vec<usize>) -> Tensor {
    let element = DType::BF16.size_in_bytes();
    let stride = KV_HEADS * HEAD_DIM * element;
    CudaBuffer::from_tensor(cache_tensor)
        .unwrap()
        .view(position * stride, stride)
        .unwrap()
        .as_tensor(Shape::new(dims), DType::BF16)
        .unwrap()
}

// FP16-range probe used by the APXINF_QWEN38_FP16_PROBE diagnostic.
fn probe_fp16_range(label: &str, tensor: &Tensor, elements: usize) {
    let buffer = CudaBuffer::from_tensor(tensor).unwrap();
    let mut bytes = vec![0u8; elements * 2];
    buffer.copy_to_host(&mut bytes).unwrap();
    let mut max_abs = 0.0f32;
    let mut min_nonzero = f32::INFINITY;
    let mut nonfinite = 0usize;
    let mut subnormal = 0usize;
    for pair in bytes.chunks_exact(2) {
        let value = f32::from_bits((u16::from_le_bytes([pair[0], pair[1]]) as u32) << 16);
        if !value.is_finite() {
            nonfinite += 1;
            continue;
        }
        let magnitude = value.abs();
        if magnitude > max_abs {
            max_abs = magnitude;
        }
        if magnitude > 0.0 {
            if magnitude < min_nonzero {
                min_nonzero = magnitude;
            }
            if magnitude < 6.103_515_6e-5 {
                subnormal += 1;
            }
        }
    }
    println!(
        "  fp16 probe {label:<14} max|x|={max_abs:11.4}  min|x|={min_nonzero:.3e}  headroom={:9.1e}x  subnormal={subnormal}  nonfinite={nonfinite}",
        65504.0 / max_abs.max(1e-30)
    );
}
