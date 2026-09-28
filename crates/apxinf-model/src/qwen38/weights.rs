//! Checkpoint interpretation and the device weight tree.
//!
//! Owns checkpoint keys, packing (fused gate+up, [K, N] FP8 copies, NVFP4
//! block-scale relayout) and the per-layer weight structs. No model or
//! runner types.


use std::collections::HashMap;
use std::path::Path;

use apxinf_core::{DType, Tensor};
use apxinf_cuda_new::{ops, CudaContext};

use super::backend::{cpu_bytes, upload, zeros};
use super::config::*;


pub(crate) fn checkpoint(path: &Path) -> HashMap<String, Tensor> {
    apxinf_loader::safetensors::load_native_path(path)
        .expect("checkpoint failed to load")
        .0
}

fn scalar(tensors: &HashMap<String, Tensor>, name: &str) -> f32 {
    tensors[name].to_f32_vec().unwrap()[0]
}

/// An NVFP4 weight with its relaid-out scales and the alpha folding both
/// per-tensor scales.
pub(crate) struct Nvfp4Weight {
    pub(crate) packed: Tensor,
    pub(crate) scales: Tensor,
    pub(crate) input_scale: f32,
    pub(crate) alpha: f32,
}

fn relayout(ctx: &CudaContext, source: &Tensor, rows: usize, k: usize) -> Tensor {
    let bytes = ops::nvfp4_scale_buffer_bytes(rows, k, BLOCK).unwrap();
    let destination = zeros(ctx, vec![bytes], DType::F8E4M3);
    ops::nvfp4_pack_block_scales(ctx, source, &destination, rows, k, BLOCK).unwrap();
    destination
}

fn load_nvfp4(
    ctx: &CudaContext,
    tensors: &HashMap<String, Tensor>,
    prefix: &str,
    n: usize,
    k: usize,
) -> Nvfp4Weight {
    let packed = upload(
        ctx,
        cpu_bytes(&tensors[&format!("{prefix}.weight")]),
        vec![n, k / 2],
        DType::E2M1Pair,
    );
    let checkpoint_scales = upload(
        ctx,
        cpu_bytes(&tensors[&format!("{prefix}.weight_scale")]),
        vec![n, k / BLOCK as usize],
        DType::F8E4M3,
    );
    let input_scale = scalar(tensors, &format!("{prefix}.input_scale"));
    let weight_scale_2 = scalar(tensors, &format!("{prefix}.weight_scale_2"));
    Nvfp4Weight {
        packed,
        scales: relayout(ctx, &checkpoint_scales, n, k),
        input_scale,
        alpha: input_scale * weight_scale_2,
    }
}

/// Concatenate gate and up into one [2N, K/2] operand.
///
/// Both per-tensor scales are identical in every layer of this checkpoint, so
/// the fused GEMM is exact. Asserted rather than assumed.
fn load_fused_gate_up(
    ctx: &CudaContext,
    tensors: &HashMap<String, Tensor>,
    layer: usize,
) -> Nvfp4Weight {
    let prefix = format!("model.language_model.layers.{layer}.mlp");
    let input_scale = scalar(tensors, &format!("{prefix}.gate_proj.input_scale"));
    let weight_scale_2 = scalar(tensors, &format!("{prefix}.gate_proj.weight_scale_2"));
    assert_eq!(
        input_scale,
        scalar(tensors, &format!("{prefix}.up_proj.input_scale"))
    );
    assert_eq!(
        weight_scale_2,
        scalar(tensors, &format!("{prefix}.up_proj.weight_scale_2"))
    );

    let mut weight = Vec::new();
    weight.extend_from_slice(cpu_bytes(&tensors[&format!("{prefix}.gate_proj.weight")]));
    weight.extend_from_slice(cpu_bytes(&tensors[&format!("{prefix}.up_proj.weight")]));
    let packed = upload(
        ctx,
        &weight,
        vec![2 * INTERMEDIATE, HIDDEN / 2],
        DType::E2M1Pair,
    );

    let mut scales = Vec::new();
    scales.extend_from_slice(cpu_bytes(&tensors[&format!("{prefix}.gate_proj.weight_scale")]));
    scales.extend_from_slice(cpu_bytes(&tensors[&format!("{prefix}.up_proj.weight_scale")]));
    let checkpoint_scales = upload(
        ctx,
        &scales,
        vec![2 * INTERMEDIATE, HIDDEN / BLOCK as usize],
        DType::F8E4M3,
    );

    Nvfp4Weight {
        packed,
        scales: relayout(ctx, &checkpoint_scales, 2 * INTERMEDIATE, HIDDEN),
        input_scale,
        alpha: input_scale * weight_scale_2,
    }
}

/// An FP8 weight transposed to the GEMM's [K, N] contract, with both scalar
/// scales folded into alpha.
pub(crate) struct Fp8Weight {
    pub(crate) weight: Tensor,
    /// The same weight as `[K, N]`, built only when prefill needs it.
    ///
    /// Decode's GEMV reads the checkpoint's own `[N, K]`; `GemmArgs` is
    /// contractually `b = [K, N]` and exposes no transpose, so the batched
    /// path needs its own copy. Re-`view`ing would silently feed the GEMM a
    /// permutation -- the same trap `load_bf16_transposed` documents. Holding
    /// both costs 6.719 GiB across the checkpoint's 208 FP8 tensors, which is
    /// why it is optional rather than always built.
    pub(crate) transposed: Option<Tensor>,
    pub(crate) input_scale: f32,
    pub(crate) alpha: f32,
}

fn load_fp8(
    ctx: &CudaContext,
    tensors: &HashMap<String, Tensor>,
    prefix: &str,
    n: usize,
    k: usize,
    for_prefill: bool,
) -> Fp8Weight {
    // [N, K] is the checkpoint's own orientation and the one the GEMV reads.
    // `for_prefill` additionally builds the [K, N] copy the GEMM contract
    // requires; see the comment on `Fp8Weight::transposed`.
    let weight_scale = scalar(tensors, &format!("{prefix}.weight_scale"));
    let input_scale = scalar(tensors, &format!("{prefix}.input_scale"));
    let source = cpu_bytes(&tensors[&format!("{prefix}.weight")]);
    assert_eq!(source.len(), n * k, "{prefix}.weight is not [{n}, {k}] E4M3");

    let transposed = for_prefill.then(|| {
        let mut flipped = vec![0u8; source.len()];
        for row in 0..n {
            for column in 0..k {
                flipped[column * n + row] = source[row * k + column];
            }
        }
        upload(ctx, &flipped, vec![k, n], DType::F8E4M3)
    });

    Fp8Weight {
        weight: upload(ctx, source, vec![n, k], DType::F8E4M3),
        transposed,
        input_scale,
        alpha: weight_scale * input_scale,
    }
}

fn load_bf16(
    ctx: &CudaContext,
    tensors: &HashMap<String, Tensor>,
    name: &str,
    dims: Vec<usize>,
) -> Tensor {
    upload(ctx, cpu_bytes(&tensors[name]), dims, DType::BF16)
}

/// Load a row-major `[rows, cols]` BF16 weight and store it transposed as
/// `[cols, rows]`.
///
/// `GemmArgs::new` is contractually `b = [K, N]` row-major, and the checkpoint
/// stores these as `[N, K]`. Re-`view`ing `[N, K]` storage as `[K, N]` does not
/// transpose it -- no bytes move, so the GEMM reads a *permutation* of the
/// weight. The shape check passes and the result is finite, so the error is
/// silent. The transpose has to happen to the data, and this is the cheap place
/// to do it: 48 x 5120 BF16 per projection, once per layer at load.
///
/// The FP8 and NVFP4 weights keep their `[N, K]` orientation deliberately --
/// their GEMV reads that layout directly and never goes through `GemmArgs`.
fn load_bf16_transposed(
    ctx: &CudaContext,
    tensors: &HashMap<String, Tensor>,
    name: &str,
    rows: usize,
    cols: usize,
) -> Tensor {
    let element = DType::BF16.size_in_bytes();
    let source = cpu_bytes(&tensors[name]);
    assert_eq!(
        source.len(),
        rows * cols * element,
        "{name} is not a [{rows}, {cols}] BF16 tensor"
    );
    let mut transposed = vec![0u8; source.len()];
    for row in 0..rows {
        for column in 0..cols {
            let from = (row * cols + column) * element;
            let to = (column * rows + row) * element;
            transposed[to..to + element].copy_from_slice(&source[from..from + element]);
        }
    }
    upload(ctx, &transposed, vec![cols, rows], DType::BF16)
}

pub(crate) struct AttentionLayer {
    pub(crate) input_norm: Tensor,
    pub(crate) post_norm: Tensor,
    pub(crate) q: Fp8Weight,
    pub(crate) k: Fp8Weight,
    pub(crate) v: Fp8Weight,
    pub(crate) o: Fp8Weight,
    pub(crate) q_norm: Tensor,
    pub(crate) k_norm: Tensor,
    pub(crate) gate_up: Nvfp4Weight,
    pub(crate) down: Nvfp4Weight,
}

pub(crate) struct GdnLayer {
    pub(crate) input_norm: Tensor,
    pub(crate) post_norm: Tensor,
    pub(crate) qkv: Fp8Weight,
    pub(crate) z: Fp8Weight,
    pub(crate) out: Fp8Weight,
    pub(crate) in_proj_a: Tensor,
    pub(crate) in_proj_b: Tensor,
    pub(crate) a_log: Tensor,
    pub(crate) dt_bias: Tensor,
    pub(crate) conv_weight: Tensor,
    pub(crate) norm_weight: Tensor,
    pub(crate) gate_up: Nvfp4Weight,
    pub(crate) down: Nvfp4Weight,
}

pub(crate) enum Layer {
    Attention(Box<AttentionLayer>),
    Gdn(Box<GdnLayer>),
}

pub(crate) struct Model {
    pub(crate) embedding: Tensor,
    pub(crate) layers: Vec<Layer>,
    pub(crate) final_norm: Tensor,
    pub(crate) lm_head: Nvfp4Weight,
}

/// `for_prefill` additionally builds the [K, N] FP8 copies the batched GEMM
/// needs. It costs 6.719 GiB, so decode-only callers leave it off.
pub(crate) fn load_model(
    ctx: &CudaContext,
    tensors: &HashMap<String, Tensor>,
    for_prefill: bool,
) -> Model {
    let mut layers = Vec::with_capacity(LAYERS);
    for layer in 0..LAYERS {
        let prefix = format!("model.language_model.layers.{layer}");
        let input_norm = load_bf16(ctx, tensors, &format!("{prefix}.input_layernorm.weight"), vec![HIDDEN]);
        let post_norm = load_bf16(
            ctx,
            tensors,
            &format!("{prefix}.post_attention_layernorm.weight"),
            vec![HIDDEN],
        );
        let gate_up = load_fused_gate_up(ctx, tensors, layer);
        let down = load_nvfp4(
            ctx,
            tensors,
            &format!("{prefix}.mlp.down_proj"),
            HIDDEN,
            INTERMEDIATE,
        );

        if is_full_attention(layer) {
            layers.push(Layer::Attention(Box::new(AttentionLayer {
                input_norm,
                post_norm,
                q: load_fp8(ctx, tensors, &format!("{prefix}.self_attn.q_proj"), 2 * HEADS * HEAD_DIM, HIDDEN, for_prefill),
                k: load_fp8(ctx, tensors, &format!("{prefix}.self_attn.k_proj"), KV_HEADS * HEAD_DIM, HIDDEN, for_prefill),
                v: load_fp8(ctx, tensors, &format!("{prefix}.self_attn.v_proj"), KV_HEADS * HEAD_DIM, HIDDEN, for_prefill),
                o: load_fp8(ctx, tensors, &format!("{prefix}.self_attn.o_proj"), HIDDEN, HEADS * HEAD_DIM, for_prefill),
                q_norm: load_bf16(ctx, tensors, &format!("{prefix}.self_attn.q_norm.weight"), vec![HEAD_DIM]),
                k_norm: load_bf16(ctx, tensors, &format!("{prefix}.self_attn.k_norm.weight"), vec![HEAD_DIM]),
                gate_up,
                down,
            })));
        } else {
            layers.push(Layer::Gdn(Box::new(GdnLayer {
                input_norm,
                post_norm,
                qkv: load_fp8(ctx, tensors, &format!("{prefix}.linear_attn.in_proj_qkv"), QKV_WIDTH, HIDDEN, for_prefill),
                z: load_fp8(ctx, tensors, &format!("{prefix}.linear_attn.in_proj_z"), Z_WIDTH, HIDDEN, for_prefill),
                out: load_fp8(ctx, tensors, &format!("{prefix}.linear_attn.out_proj"), HIDDEN, Z_WIDTH, for_prefill),
                // Transposed at load to [HIDDEN, GDN_V_HEADS]: these two are
                // the only weights that reach a GEMM through `GemmArgs::new`,
                // which requires b = [K, N].
                in_proj_a: load_bf16_transposed(ctx, tensors, &format!("{prefix}.linear_attn.in_proj_a.weight"), GDN_V_HEADS, HIDDEN),
                in_proj_b: load_bf16_transposed(ctx, tensors, &format!("{prefix}.linear_attn.in_proj_b.weight"), GDN_V_HEADS, HIDDEN),
                a_log: load_bf16(ctx, tensors, &format!("{prefix}.linear_attn.A_log"), vec![GDN_V_HEADS]),
                dt_bias: load_bf16(ctx, tensors, &format!("{prefix}.linear_attn.dt_bias"), vec![GDN_V_HEADS]),
                conv_weight: load_bf16(ctx, tensors, &format!("{prefix}.linear_attn.conv1d.weight"), vec![QKV_WIDTH, CONV_WIDTH]),
                norm_weight: load_bf16(ctx, tensors, &format!("{prefix}.linear_attn.norm.weight"), vec![GDN_HEAD_DIM]),
                gate_up,
                down,
            })));
        }
    }

    Model {
        embedding: load_bf16(ctx, tensors, "model.language_model.embed_tokens.weight", vec![VOCAB, HIDDEN]),
        layers,
        final_norm: load_bf16(ctx, tensors, "model.language_model.norm.weight", vec![HIDDEN]),
        lm_head: load_nvfp4(ctx, tensors, "lm_head", VOCAB, HIDDEN),
    }
}
