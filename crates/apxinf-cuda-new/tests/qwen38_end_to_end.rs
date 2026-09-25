//! The whole Qwen3.8-27B-NVFP4 text model, forward, on one Thor.
//!
//! 64 layers: 48 Gated DeltaNet and 16 full attention, mixed NVFP4/FP8/BF16,
//! embedding through lm_head, greedy token selection. This is the first point
//! where a decode and prefill rate for the *model* exists rather than for a
//! component.
//!
//! What this establishes and what it does not: the model runs, its memory fits,
//! and the token rate is measured. It does **not** establish that the tokens
//! are the right ones. Four modelling assumptions are still unvalidated -- the
//! GDN recurrence, the rotary pairing convention, the mRoPE collapse, and the
//! conv/SiLU ordering -- and each would produce plausible output if wrong. See
//! devlocal/qwen38-nvfp4/reports/STATUS.md. Validating them needs a reference
//! engine that can execute this checkpoint, which thor-3 does not have.
//!
//! ```text
//! APXINF_QWEN38_CHECKPOINT=/path/to/Qwen3.8-27B-NVFP4 \
//!   bash crates/apxinf-cuda-new/test-new.sh \
//!     test -p apxinf-cuda --test qwen38_end_to_end --release \
//!     -- --ignored --nocapture
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda::{ops, CudaBuffer, CudaContext};

const HIDDEN: usize = 5120;
const INTERMEDIATE: usize = 17408;
const VOCAB: usize = 248320;
const LAYERS: usize = 64;
const FULL_ATTENTION_INTERVAL: usize = 4;
const BLOCK: u32 = 16;
const EPSILON: f32 = 1e-6;

// Full attention
const HEADS: usize = 24;
const KV_HEADS: usize = 4;
const HEAD_DIM: usize = 256;
const ROPE_THETA: f32 = 1.0e7;
const PARTIAL_ROTARY: f32 = 0.25;

// Gated DeltaNet
const GDN_K_HEADS: usize = 16;
const GDN_V_HEADS: usize = 48;
const GDN_HEAD_DIM: usize = 128;
const CONV_WIDTH: usize = 4;
const CHUNK: usize = 64; // gdn_chunk_scan works a chunk at a time
const QKV_WIDTH: usize = 10240; // 16*128 q + 16*128 k + 48*128 v
const Z_WIDTH: usize = 6144;

fn is_full_attention(layer: usize) -> bool {
    (layer + 1) % FULL_ATTENTION_INTERVAL == 0
}

fn checkpoint() -> HashMap<String, Tensor> {
    let path: PathBuf = std::env::var_os("APXINF_QWEN38_CHECKPOINT")
        .expect("set APXINF_QWEN38_CHECKPOINT")
        .into();
    apxinf_loader::safetensors::load_native_path(&path)
        .expect("checkpoint failed to load")
        .0
}

fn cpu_bytes(tensor: &Tensor) -> &[u8] {
    match tensor.storage() {
        apxinf_core::Storage::Cpu(data) => data,
        _ => panic!("expected a CPU tensor"),
    }
}

fn upload(ctx: &CudaContext, bytes: &[u8], dims: Vec<usize>, dtype: DType) -> Tensor {
    let buffer = CudaBuffer::alloc(bytes.len().max(1), ctx.device_id()).unwrap();
    buffer.copy_from_host(bytes).unwrap();
    buffer.as_tensor(Shape::new(dims), dtype).unwrap()
}

fn zeros(ctx: &CudaContext, dims: Vec<usize>, dtype: DType) -> Tensor {
    let bytes = dims.iter().product::<usize>() * dtype.size_in_bytes();
    let buffer = CudaBuffer::alloc(bytes.max(1), ctx.device_id()).unwrap();
    buffer.copy_from_host(&vec![0u8; bytes.max(1)]).unwrap();
    buffer.as_tensor(Shape::new(dims), dtype).unwrap()
}

fn scalar(tensors: &HashMap<String, Tensor>, name: &str) -> f32 {
    tensors[name].to_f32_vec().unwrap()[0]
}

/// An NVFP4 weight with its relaid-out scales and the alpha folding both
/// per-tensor scales.
struct Nvfp4Weight {
    packed: Tensor,
    scales: Tensor,
    input_scale: f32,
    alpha: f32,
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
struct Fp8Weight {
    weight: Tensor,
    /// The same weight as `[K, N]`, built only when prefill needs it.
    ///
    /// Decode's GEMV reads the checkpoint's own `[N, K]`; `GemmArgs` is
    /// contractually `b = [K, N]` and exposes no transpose, so the batched
    /// path needs its own copy. Re-`view`ing would silently feed the GEMM a
    /// permutation -- the same trap `load_bf16_transposed` documents. Holding
    /// both costs 6.719 GiB across the checkpoint's 208 FP8 tensors, which is
    /// why it is optional rather than always built.
    transposed: Option<Tensor>,
    input_scale: f32,
    alpha: f32,
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

struct AttentionLayer {
    input_norm: Tensor,
    post_norm: Tensor,
    q: Fp8Weight,
    k: Fp8Weight,
    v: Fp8Weight,
    o: Fp8Weight,
    q_norm: Tensor,
    k_norm: Tensor,
    gate_up: Nvfp4Weight,
    down: Nvfp4Weight,
}

struct GdnLayer {
    input_norm: Tensor,
    post_norm: Tensor,
    qkv: Fp8Weight,
    z: Fp8Weight,
    out: Fp8Weight,
    in_proj_a: Tensor,
    in_proj_b: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
    conv_weight: Tensor,
    norm_weight: Tensor,
    gate_up: Nvfp4Weight,
    down: Nvfp4Weight,
}

enum Layer {
    Attention(Box<AttentionLayer>),
    Gdn(Box<GdnLayer>),
}

struct Model {
    embedding: Tensor,
    layers: Vec<Layer>,
    final_norm: Tensor,
    lm_head: Nvfp4Weight,
}

/// `for_prefill` additionally builds the [K, N] FP8 copies the batched GEMM
/// needs. It costs 6.719 GiB, so decode-only callers leave it off.
fn load_model(
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

/// Per-token working buffers, sized for one token of decode.
struct Scratch {
    hidden: Tensor,
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
    positions: Tensor,
    // gdn
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
    token: Tensor,
    logits: Tensor,
    next_token: Tensor,
}

impl Scratch {
    fn next_token_host(&self) -> i32 {
        let mut id = [0u8; 4];
        CudaBuffer::from_tensor(&self.next_token)
            .unwrap()
            .copy_to_host(&mut id)
            .unwrap();
        i32::from_le_bytes(id)
    }

    fn new(ctx: &CudaContext) -> Scratch {
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
struct GdnState {
    recurrent: Tensor,
    conv_window: Tensor,
}

/// Key/value cache for one full-attention layer.
struct KvCache {
    keys: Tensor,
    values: Tensor,
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

/// residual = residual + MLP(RMSNorm(residual)), with the residual being the
/// running hidden state. Taking it from `scratch` rather than as a separate
/// argument keeps the borrow disjoint.
fn nvfp4_mlp(
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
    ops::gemm(
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
    ops::gemm(
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

#[allow(clippy::too_many_arguments)]
fn decode_step(
    ctx: &CudaContext,
    model: &Model,
    scratch: &mut Scratch,
    gdn_states: &mut [GdnState],
    kv_caches: &mut [KvCache],
    position: usize,
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
                fp8_projection(ctx, &attention.k, &scratch.normalized, &scratch.fp8_activation, &mut key_slot);
                fp8_projection(ctx, &attention.v, &scratch.normalized, &scratch.fp8_activation, &mut value_slot);

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

                nvfp4_mlp(ctx, &attention.gate_up, &attention.down, &attention.post_norm, scratch);
            }
            Layer::Gdn(gdn) => {
                let state = &mut gdn_states[gdn_index];
                gdn_index += 1;

                ops::rms_norm(ctx, &scratch.hidden, &gdn.input_norm, &scratch.normalized, EPSILON).unwrap();
                fp8_projection(ctx, &gdn.qkv, &scratch.normalized, &scratch.fp8_activation, &mut scratch.gdn_qkv);
                fp8_projection(ctx, &gdn.z, &scratch.normalized, &scratch.fp8_activation, &mut scratch.gdn_z);

                let qkv_flat = view(&scratch.gdn_qkv, vec![QKV_WIDTH], DType::BF16);
                ops::gdn_causal_conv_step(ctx, &state.conv_window, &qkv_flat, &gdn.conv_weight, &scratch.gdn_conv).unwrap();

                let (q, k, v) = split_gdn_qkv(ctx, &scratch.gdn_conv);
                ops::gdn_l2_normalize_heads(ctx, &q, EPSILON).unwrap();
                ops::gdn_l2_normalize_heads(ctx, &k, EPSILON).unwrap();

                bf16_matvec(ctx, &gdn.in_proj_a, &scratch.normalized, &scratch.gdn_a);
                bf16_matvec(ctx, &gdn.in_proj_b, &scratch.normalized, &scratch.gdn_b);
                ops::gdn_decay_and_beta(ctx, &scratch.gdn_a, &scratch.gdn_b, &gdn.a_log, &gdn.dt_bias, &scratch.gdn_decay, &scratch.gdn_beta).unwrap();

                ops::gdn_recurrent_step(ctx, &state.recurrent, &q, &k, &v, &scratch.gdn_decay, &scratch.gdn_beta, &scratch.gdn_readout, GDN_K_HEADS).unwrap();
                ops::gdn_gated_norm(ctx, &scratch.gdn_readout, &gdn_z_heads(ctx, &scratch.gdn_z), &gdn.norm_weight, &scratch.gdn_gated, EPSILON).unwrap();

                let flat = flatten(ctx, &scratch.gdn_gated, Z_WIDTH);
                fp8_projection(ctx, &gdn.out, &flat, &scratch.gdn_fp8, &mut scratch.projected);
                ops::add_into(ctx, &scratch.projected, &scratch.hidden).unwrap();

                nvfp4_mlp(ctx, &gdn.gate_up, &gdn.down, &gdn.post_norm, scratch);
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
    ops::gemm(
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
#[test]
#[ignore = "requires the checkpoint and a GPU; pairs with ref_dump.py"]
fn dump_layer_outputs() {
    let ctx = CudaContext::new(0).unwrap();
    let tensors = checkpoint();
    let model = load_model(&ctx, &tensors, false);
    ctx.synchronize().unwrap();
    drop(tensors);

    let out_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../devlocal/qwen38-nvfp4/apxdump");
    std::fs::create_dir_all(&out_dir).unwrap();
    let dump = |name: &str, t: &Tensor, n: usize| {
        ctx.synchronize().unwrap();
        let mut bytes = vec![0u8; n * 2];
        CudaBuffer::from_tensor(t).unwrap().copy_to_host(&mut bytes).unwrap();
        let f: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|v| half::bf16::from_bits(u16::from_le_bytes([v[0], v[1]])).to_f32())
            .collect();
        let mut raw = Vec::with_capacity(f.len() * 4);
        for x in &f { raw.extend_from_slice(&x.to_le_bytes()); }
        std::fs::write(out_dir.join(format!("{name}.f32")), raw).unwrap();
        println!("apxdump {name} n={n}");
    };

    let capacity = 64usize;
    let mut gdn_states: Vec<GdnState> = (0..LAYERS - LAYERS / FULL_ATTENTION_INTERVAL)
        .map(|_| GdnState {
            recurrent: zeros(&ctx, vec![GDN_V_HEADS, GDN_HEAD_DIM, GDN_HEAD_DIM], DType::F32),
            conv_window: zeros(&ctx, vec![QKV_WIDTH, CONV_WIDTH], DType::F32),
        })
        .collect();
    let mut kv_caches: Vec<KvCache> = (0..LAYERS / FULL_ATTENTION_INTERVAL)
        .map(|_| KvCache {
            keys: zeros(&ctx, vec![1, capacity, KV_HEADS, HEAD_DIM], DType::BF16),
            values: zeros(&ctx, vec![1, capacity, KV_HEADS, HEAD_DIM], DType::BF16),
        })
        .collect();
    let mut scratch = Scratch::new(&ctx);

    // token id 100, position 0
    CudaBuffer::from_tensor(&scratch.token).unwrap()
        .copy_from_host(&100i32.to_le_bytes()).unwrap();
    CudaBuffer::from_tensor(&scratch.positions).unwrap()
        .copy_from_host(&0i32.to_le_bytes()).unwrap();

    ops::embedding_gather(&ctx, &model.embedding, &scratch.token, &scratch.hidden).unwrap();
    ctx.synchronize().unwrap();
    dump("input_hidden", &scratch.hidden, HIDDEN);

    let rotary = ops::rotary_dim(HEAD_DIM, PARTIAL_ROTARY);
    let mut gdn_index = 0usize;
    let mut attention_index = 0usize;
    for (layer_index, layer) in model.layers.iter().enumerate() {
        let owned = format!("L{layer_index}");
        let label = Some(owned.as_str());
        run_one_layer(&ctx, layer, &mut scratch, &mut gdn_states, &mut kv_caches,
                      &mut gdn_index, &mut attention_index, rotary, 0, label);
        ctx.synchronize().unwrap();
        dump(&format!("hidden_after_layer{layer_index}"), &scratch.hidden, HIDDEN);
        if layer_index == 0 { dump("layer0_out", &scratch.hidden, HIDDEN); }
        if layer_index == 3 { dump("layer3_out", &scratch.hidden, HIDDEN); break; }
    }
    println!("APXDUMP COMPLETE -> {}", out_dir.display());
}


fn apxdump_tensor(ctx: &CudaContext, name: &str, t: &Tensor, n: usize) {
    let out_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../devlocal/qwen38-nvfp4/apxdump");
    std::fs::create_dir_all(&out_dir).unwrap();
    // Order the copy against the stream the ops actually ran on.
    ctx.synchronize().unwrap();
    let mut bytes = vec![0u8; n * 2];
    CudaBuffer::from_tensor(t).unwrap().copy_to_host(&mut bytes).unwrap();
    let f: Vec<f32> = bytes
        .chunks_exact(2)
        .map(|v| half::bf16::from_bits(u16::from_le_bytes([v[0], v[1]])).to_f32())
        .collect();
    let mut raw = Vec::with_capacity(f.len() * 4);
    for x in &f { raw.extend_from_slice(&x.to_le_bytes()); }
    std::fs::write(out_dir.join(format!("{name}.f32")), raw).unwrap();
}

/// One decoder layer, extracted from decode_step so the dump test can stop
/// after a chosen layer. Mutates scratch.hidden in place.
#[allow(clippy::too_many_arguments)]
fn run_one_layer(
    ctx: &CudaContext,
    layer: &Layer,
    scratch: &mut Scratch,
    gdn_states: &mut [GdnState],
    kv_caches: &mut [KvCache],
    gdn_index: &mut usize,
    attention_index: &mut usize,
    rotary: usize,
    position: usize,
    dump_label: Option<&str>,
) {
    match layer {
        Layer::Attention(attention) => {
            let cache = &mut kv_caches[*attention_index];
            *attention_index += 1;
            ops::rms_norm(ctx, &scratch.hidden, &attention.input_norm, &scratch.normalized, EPSILON).unwrap();
            fp8_projection(ctx, &attention.q, &scratch.normalized, &scratch.fp8_activation, &mut scratch.qkv_fused);
            let fused_heads = view(&scratch.qkv_fused, vec![1, HEADS, 2 * HEAD_DIM], DType::BF16);
            ops::split_query_and_gate(ctx, &fused_heads, &scratch.query, &scratch.query_gate).unwrap();
            let mut key_slot = cache_slot(&cache.keys, position, vec![1, KV_HEADS * HEAD_DIM]);
            let mut value_slot = cache_slot(&cache.values, position, vec![1, KV_HEADS * HEAD_DIM]);
            fp8_projection(ctx, &attention.k, &scratch.normalized, &scratch.fp8_activation, &mut key_slot);
            fp8_projection(ctx, &attention.v, &scratch.normalized, &scratch.fp8_activation, &mut value_slot);
            let key_heads = cache_slot(&cache.keys, position, vec![KV_HEADS, HEAD_DIM]);
            let query_heads = view(&scratch.query, vec![HEADS, HEAD_DIM], DType::BF16);
            ops::head_rms_norm(ctx, &query_heads, &attention.q_norm, EPSILON).unwrap();
            ops::head_rms_norm(ctx, &key_heads, &attention.k_norm, EPSILON).unwrap();
            let query_tokens = view(&scratch.query, vec![1, HEADS, HEAD_DIM], DType::BF16);
            let key_tokens = cache_slot(&cache.keys, position, vec![1, KV_HEADS, HEAD_DIM]);
            ops::partial_rope(ctx, &query_tokens, &scratch.positions, rotary, ROPE_THETA).unwrap();
            ops::partial_rope(ctx, &key_tokens, &scratch.positions, rotary, ROPE_THETA).unwrap();
            let valid = position + 1;
            // Viewed at `valid`, not at the full allocation: the FA2 candidate
            // requires key_capacity == key_tokens, and a capacity-shaped view
            // fails that for every step but the last, dropping decode onto the
            // naive kernel whose cost grows with KV length.
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
            if let Some(l) = dump_label {
                apxdump_tensor(ctx, &format!("{l}_input_norm"), &scratch.normalized, HIDDEN);
                apxdump_tensor(ctx, &format!("{l}_mixer_out"), &scratch.projected, HIDDEN);
            }
            ops::add_into(ctx, &scratch.projected, &scratch.hidden).unwrap();
            nvfp4_mlp(ctx, &attention.gate_up, &attention.down, &attention.post_norm, scratch);
        }
        Layer::Gdn(gdn) => {
            let state = &mut gdn_states[*gdn_index];
            *gdn_index += 1;
            ops::rms_norm(ctx, &scratch.hidden, &gdn.input_norm, &scratch.normalized, EPSILON).unwrap();
            fp8_projection(ctx, &gdn.qkv, &scratch.normalized, &scratch.fp8_activation, &mut scratch.gdn_qkv);
            fp8_projection(ctx, &gdn.z, &scratch.normalized, &scratch.fp8_activation, &mut scratch.gdn_z);
            let qkv_flat = view(&scratch.gdn_qkv, vec![QKV_WIDTH], DType::BF16);
            ops::gdn_causal_conv_step(ctx, &state.conv_window, &qkv_flat, &gdn.conv_weight, &scratch.gdn_conv).unwrap();
            let (q, k, v) = split_gdn_qkv(ctx, &scratch.gdn_conv);
            ops::gdn_l2_normalize_heads(ctx, &q, EPSILON).unwrap();
            ops::gdn_l2_normalize_heads(ctx, &k, EPSILON).unwrap();
            bf16_matvec(ctx, &gdn.in_proj_a, &scratch.normalized, &scratch.gdn_a);
            bf16_matvec(ctx, &gdn.in_proj_b, &scratch.normalized, &scratch.gdn_b);
            ops::gdn_decay_and_beta(ctx, &scratch.gdn_a, &scratch.gdn_b, &gdn.a_log, &gdn.dt_bias, &scratch.gdn_decay, &scratch.gdn_beta).unwrap();
            ops::gdn_recurrent_step(ctx, &state.recurrent, &q, &k, &v, &scratch.gdn_decay, &scratch.gdn_beta, &scratch.gdn_readout, GDN_K_HEADS).unwrap();
            ops::gdn_gated_norm(ctx, &scratch.gdn_readout, &gdn_z_heads(ctx, &scratch.gdn_z), &gdn.norm_weight, &scratch.gdn_gated, EPSILON).unwrap();
            let flat = flatten(ctx, &scratch.gdn_gated, Z_WIDTH);
            fp8_projection(ctx, &gdn.out, &flat, &scratch.gdn_fp8, &mut scratch.projected);
            if let Some(l) = dump_label {
                apxdump_tensor(ctx, &format!("{l}_input_norm"), &scratch.normalized, HIDDEN);
                apxdump_tensor(ctx, &format!("{l}_mixer_out"), &scratch.projected, HIDDEN);
            }
            ops::add_into(ctx, &scratch.projected, &scratch.hidden).unwrap();
            nvfp4_mlp(ctx, &gdn.gate_up, &gdn.down, &gdn.post_norm, scratch);
        }
    }
}

// --- small helpers that reinterpret existing device storage -----------------

fn capacity_of(cache: &Tensor) -> usize {
    cache.shape().dims()[1]
}

/// The tuning-recipe cache directory for attention.
///
/// Decode advances the KV length one token at a time and the attention tuning
/// key includes `key_tokens`, `key_capacity` and `query_start`, so every step
/// is a fresh key. Without a cache that means a full autotune per step -- 3
/// warm-up plus 10 timed launches per candidate -- which measured as 22 ms per
/// attention layer at 2048 KV, against ~42 us of actual KV traffic. Persisting
/// recipes lets a repeated KV length skip tuning entirely.
/// Where tuned GEMM recipes are persisted.
///
/// Unlike attention, the GEMM tuning key does not move per call: it carries
/// M, N, K and the scale *predicates*, not the scale values or the operand
/// addresses, so one prefill shape tunes once and every later call in the
/// process reports `source=memory-recipe`. The cache therefore changes
/// nothing in steady state. What it removes is the one-time tune, which at
/// these widths measured 0.4-0.7 s per distinct shape and is otherwise paid
/// again by every fresh process.
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

/// Report the magnitude range of a BF16 tensor, to decide whether an FP16
/// kernel could carry it.
///
/// FlashInfer's generic GDN prefill variant -- the one our 48 value heads and
/// group of 3 force us onto, neither being a power of two -- exists only with
/// FP16 I/O. FP16 saturates at 65504 and loses its last normal near 6.1e-5,
/// so adopting it hinges on where these activations actually sit. Measuring
/// beats assuming: the checkpoint is calibrated, and its magnitudes are not
/// obvious from the architecture.
fn probe_fp16_range(label: &str, tensor: &Tensor, elements: usize) {
    let buffer = CudaBuffer::from_tensor(tensor).unwrap();
    let mut bytes = vec![0u8; elements * 2];
    buffer.copy_to_host(&mut bytes).unwrap();

    let mut max_abs = 0.0f32;
    let mut min_nonzero = f32::INFINITY;
    let mut nonfinite = 0usize;
    let mut subnormal_in_fp16 = 0usize;
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
            // 6.104e-5 is the smallest normal FP16; below it precision decays.
            if magnitude < 6.104e-5 {
                subnormal_in_fp16 += 1;
            }
        }
    }
    let headroom = 65504.0f32 / max_abs.max(f32::MIN_POSITIVE);
    println!(
        "  fp16 probe {label:<14} max|x|={max_abs:11.4}  min|x|={min_nonzero:.3e}  \
headroom={headroom:9.1e}x  subnormal={subnormal_in_fp16}  nonfinite={nonfinite}"
    );
}

/// A view of the leading `dims` elements of a larger allocation.
///
/// `view` requires the shape to describe the whole buffer exactly, so it
/// cannot express "the first N rows". Prefill allocates every buffer at
/// `seq_padded` but hands most operators the true token count, so it needs
/// this instead.
fn prefix(tensor: &Tensor, dims: Vec<usize>, dtype: DType) -> Tensor {
    let span = dims.iter().product::<usize>() * dtype.size_in_bytes();
    CudaBuffer::from_tensor(tensor)
        .unwrap()
        .view(0, span)
        .unwrap()
        .as_tensor(Shape::new(dims), dtype)
        .unwrap()
}

fn view(tensor: &Tensor, dims: Vec<usize>, dtype: DType) -> Tensor {
    CudaBuffer::from_tensor(tensor)
        .unwrap()
        .as_tensor(Shape::new(dims), dtype)
        .unwrap()
}

fn flatten(_ctx: &CudaContext, tensor: &Tensor, width: usize) -> Tensor {
    view(tensor, vec![1, width], DType::BF16)
}

fn gdn_z_heads(_ctx: &CudaContext, z: &Tensor) -> Tensor {
    view(z, vec![GDN_V_HEADS, GDN_HEAD_DIM], DType::BF16)
}

/// q, k and v live in one [10240] projection: 16*128 q, then 16*128 k, then
/// 48*128 v. The views share storage rather than copying.
fn split_gdn_qkv(_ctx: &CudaContext, qkv: &Tensor) -> (Tensor, Tensor, Tensor) {
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
struct PrefillScratch {
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
    positions: Tensor,
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
    tokens: Tensor,
    // Staging for the FlashInfer chunked scan, which takes q/k/v as separate
    // FP16 tensors and a linear-space decay. Allocated only when that path is
    // selected; see `use_flashinfer_gdn`.
    gdn_q16: Option<Tensor>,
    gdn_k16: Option<Tensor>,
    gdn_v16: Option<Tensor>,
    gdn_out16: Option<Tensor>,
    gdn_alpha: Option<Tensor>,
    cu_seqlens: Option<Tensor>,
    flashinfer_workspace: Option<Tensor>,
}

/// Whether prefill runs the vendored FlashInfer chunked scan.
///
/// Off by default: our own scan is the one checked against the reference
/// tensors. `APXINF_QWEN38_FLASHINFER_GDN=1` selects the faster path.
fn use_flashinfer_gdn() -> bool {
    std::env::var("APXINF_QWEN38_FLASHINFER_GDN").is_ok()
}

impl PrefillScratch {
    fn new(ctx: &CudaContext, capacity: usize) -> PrefillScratch {
        assert_eq!(capacity % CHUNK, 0, "prefill capacity must be a multiple of {CHUNK}");
        let scale_bytes =
            |rows: usize, k: usize| vec![ops::nvfp4_scale_buffer_bytes(rows, k, BLOCK).unwrap()];
        let t = capacity;
        let flashinfer = use_flashinfer_gdn();
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
fn prefill_step(
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

    for layer in model.layers.iter() {
        let layer_start = timing.then(Instant::now);
        match layer {
            Layer::Attention(attention) => {
                let cache = &mut kv_caches[attention_index];
                attention_index += 1;

                let hidden = prefix(&scratch.hidden, vec![tokens, HIDDEN], DType::BF16);
                let normalized = prefix(&scratch.normalized, vec![tokens, HIDDEN], DType::BF16);
                ops::rms_norm(ctx, &hidden, &attention.input_norm, &normalized, EPSILON).unwrap();

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
                fp8_projection_prequantized(ctx, &attention.k, &scratch.fp8_activation, &mut key_rows, tokens);
                fp8_projection_prequantized(ctx, &attention.v, &scratch.fp8_activation, &mut value_rows, tokens);

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

                let gate_flat = prefix(&scratch.query_gate, vec![tokens, HEADS * HEAD_DIM], DType::BF16);
                let attention_out = prefix(&scratch.attention_out, vec![tokens, HEADS * HEAD_DIM], DType::BF16);
                ops::apply_output_gate(ctx, &attention_out, &gate_flat).unwrap();
                fp8_projection_rows(ctx, &attention.o, &attention_out, &scratch.attention_fp8, &mut scratch.projected, tokens);

                let projected = prefix(&scratch.projected, vec![tokens, HIDDEN], DType::BF16);
                let residual = prefix(&scratch.hidden, vec![tokens, HIDDEN], DType::BF16);
                ops::add_into(ctx, &projected, &residual).unwrap();

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

                // qkv and z read the same post-norm activation, so it is
                // quantized once for both.
                quantize_shared_fp8_activation(
                    ctx, &[&gdn.qkv, &gdn.z], &normalized, &scratch.fp8_activation, tokens,
                );
                fp8_projection_prequantized(ctx, &gdn.qkv, &scratch.fp8_activation, &mut scratch.gdn_qkv, tokens);
                fp8_projection_prequantized(ctx, &gdn.z, &scratch.fp8_activation, &mut scratch.gdn_z, tokens);
                stage("qkv+z proj", &mut mark);

                // One launch over the prompt, and it reseeds the window so the
                // first decode step continues correctly.
                let qkv_rows = prefix(&scratch.gdn_qkv, vec![tokens, QKV_WIDTH], DType::BF16);
                let conv_rows = prefix(&scratch.gdn_conv, vec![tokens, QKV_WIDTH], DType::BF16);
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

                if let (Some(q16), Some(k16), Some(v16), Some(out16), Some(alpha),
                        Some(cu), Some(workspace)) = (
                    scratch.gdn_q16.as_ref(), scratch.gdn_k16.as_ref(),
                    scratch.gdn_v16.as_ref(), scratch.gdn_out16.as_ref(),
                    scratch.gdn_alpha.as_ref(), scratch.cu_seqlens.as_ref(),
                    scratch.flashinfer_workspace.as_ref(),
                ) {
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
                    ops::gdn_prepare_flashinfer(
                        ctx, &conv, &q_rows, &k_rows, &v_rows, &decay, &alpha_rows,
                        tokens, QKV_WIDTH, GDN_K_HEADS, GDN_V_HEADS, GDN_HEAD_DIM,
                        EPSILON,
                    )
                    .unwrap();

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
                    ops::convert_f16_to_bf16(
                        ctx, &out_rows, &readout_bf, tokens * GDN_V_HEADS * GDN_HEAD_DIM,
                    )
                    .unwrap();
                    stage("scan(flashinfer)", &mut mark);
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

                let readout_rows = prefix(&scratch.gdn_readout, vec![tokens, GDN_V_HEADS, GDN_HEAD_DIM], DType::BF16);
                let z_heads = prefix(&scratch.gdn_z, vec![tokens, GDN_V_HEADS, GDN_HEAD_DIM], DType::BF16);
                let gated = prefix(&scratch.gdn_gated, vec![tokens, GDN_V_HEADS, GDN_HEAD_DIM], DType::BF16);
                ops::gdn_gated_norm_seq(
                    ctx, &readout_rows, &z_heads, &gdn.norm_weight, &gated, tokens,
                    GDN_V_HEADS, GDN_HEAD_DIM, EPSILON,
                )
                .unwrap();

                stage("gated norm", &mut mark);

                let flat = prefix(&scratch.gdn_gated, vec![tokens, Z_WIDTH], DType::BF16);
                fp8_projection_rows(ctx, &gdn.out, &flat, &scratch.gdn_fp8, &mut scratch.projected, tokens);
                stage("out proj", &mut mark);

                let projected = prefix(&scratch.projected, vec![tokens, HIDDEN], DType::BF16);
                let residual = prefix(&scratch.hidden, vec![tokens, HIDDEN], DType::BF16);
                ops::add_into(ctx, &projected, &residual).unwrap();

                nvfp4_mlp_rows(ctx, &gdn.gate_up, &gdn.down, &gdn.post_norm, scratch, tokens);
                stage("mlp", &mut mark);
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
    }

    if timing {
        println!(
            "  prefill split: attention {:8.2} ms ({} layers)   gdn {:8.2} ms ({} layers)",
            attention_time * 1e3, attention_index, gdn_time * 1e3, gdn_index
        );
    }
}

/// Upload the prompt into the prefill token buffer.
fn write_tokens(ctx: &CudaContext, prefill: &PrefillScratch, ids: &[u32]) {
    let _ = ctx;
    let mut raw = Vec::with_capacity(ids.len() * 4);
    for &id in ids {
        raw.extend_from_slice(&(id as i32).to_le_bytes());
    }
    CudaBuffer::from_tensor(&prefill.tokens)
        .unwrap()
        .copy_from_host(&raw)
        .unwrap();
    // Positions are simply 0..tokens for a fresh prompt.
    let mut positions = Vec::with_capacity(ids.len() * 4);
    for index in 0..ids.len() {
        positions.extend_from_slice(&(index as i32).to_le_bytes());
    }
    CudaBuffer::from_tensor(&prefill.positions)
        .unwrap()
        .copy_from_host(&positions)
        .unwrap();
}

/// Logits and greedy argmax for the last row of a finished prefill.
///
/// Only one row is needed, so the lm_head runs at M=1 over a view of that row
/// rather than over the whole prompt: the other rows would cost a
/// 248320-wide projection each and are never read.
fn prefill_logits(
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
fn view_row(tensor: &Tensor, row: usize, width: usize, dtype: DType) -> Tensor {
    let element = dtype.size_in_bytes();
    CudaBuffer::from_tensor(tensor)
        .unwrap()
        .view(row * width * element, width * element)
        .unwrap()
        .as_tensor(Shape::new(vec![1, width]), dtype)
        .unwrap()
}

/// A view of the first `tokens` rows of a KV cache.
fn cache_rows(cache_tensor: &Tensor, tokens: usize, dims: Vec<usize>) -> Tensor {
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
fn bf16_matvec(ctx: &CudaContext, weight: &Tensor, input: &Tensor, output: &Tensor) {
    let dims = weight.shape().dims().to_vec();
    let mut out = view(output, vec![1, dims[1]], DType::BF16);
    ops::gemm(ctx, ops::GemmArgs::new(input, weight, &mut out)).unwrap();
}

/// A view of one token's slot in a KV cache, shaped for the projection that
/// fills it. Writing the projection straight into the cache avoids a
/// device-to-device copy per layer per token.
fn cache_slot(cache_tensor: &Tensor, position: usize, dims: Vec<usize>) -> Tensor {
    let element = DType::BF16.size_in_bytes();
    let stride = KV_HEADS * HEAD_DIM * element;
    CudaBuffer::from_tensor(cache_tensor)
        .unwrap()
        .view(position * stride, stride)
        .unwrap()
        .as_tensor(Shape::new(dims), DType::BF16)
        .unwrap()
}

#[test]
#[ignore = "requires the 20 GiB Qwen3.8-27B-NVFP4 checkpoint and a GPU"]
fn full_model_decodes_and_is_timed() {
    let ctx = CudaContext::new(0).unwrap();

    let start = Instant::now();
    let tensors = checkpoint();
    println!("checkpoint read:     {:6.2} s", start.elapsed().as_secs_f64());

    let start = Instant::now();
    let model = load_model(&ctx, &tensors, false);
    ctx.synchronize().unwrap();
    println!("weights on device:   {:6.2} s", start.elapsed().as_secs_f64());
    drop(tensors);

    let capacity = 64usize;
    let mut gdn_states: Vec<GdnState> = (0..LAYERS - LAYERS / FULL_ATTENTION_INTERVAL)
        .map(|_| GdnState {
            recurrent: zeros(&ctx, vec![GDN_V_HEADS, GDN_HEAD_DIM, GDN_HEAD_DIM], DType::F32),
            conv_window: zeros(&ctx, vec![QKV_WIDTH, CONV_WIDTH], DType::F32),
        })
        .collect();
    let mut kv_caches: Vec<KvCache> = (0..LAYERS / FULL_ATTENTION_INTERVAL)
        .map(|_| KvCache {
            keys: zeros(&ctx, vec![1, capacity, KV_HEADS, HEAD_DIM], DType::BF16),
            values: zeros(&ctx, vec![1, capacity, KV_HEADS, HEAD_DIM], DType::BF16),
        })
        .collect();
    let mut scratch = Scratch::new(&ctx);

    // Warm up: the first step tunes every distinct GEMM shape.
    decode_step(&ctx, &model, &mut scratch, &mut gdn_states, &mut kv_caches, 0);
    ctx.synchronize().unwrap();

    // Sanity before timing. A forward pass that finishes is not the same as a
    // forward pass that computed anything: NaNs propagate silently, a dead
    // layer yields constant logits, and a broken recurrence still returns a
    // number. None of that shows up in a latency measurement.
    {
        let mut bytes = vec![0u8; VOCAB * 2];
        CudaBuffer::from_tensor(&scratch.logits)
            .unwrap()
            .copy_to_host(&mut bytes)
            .unwrap();
        let logits: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|v| half::bf16::from_bits(u16::from_le_bytes([v[0], v[1]])).to_f32())
            .collect();

        let non_finite = logits.iter().filter(|v| !v.is_finite()).count();
        let finite: Vec<f32> = logits.iter().copied().filter(|v| v.is_finite()).collect();
        let max = finite.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let min = finite.iter().copied().fold(f32::INFINITY, f32::min);
        let mean = finite.iter().sum::<f32>() / finite.len().max(1) as f32;
        let variance = finite.iter().map(|v| (v - mean).powi(2)).sum::<f32>()
            / finite.len().max(1) as f32;
        let distinct = {
            let mut sorted = finite.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted.dedup();
            sorted.len()
        };
        println!(
            "\nlogits: min={min:.3} max={max:.3} mean={mean:.3} sd={:.3}\n        \
             non-finite={non_finite}  distinct values={distinct} of {VOCAB}",
            variance.sqrt()
        );

        assert_eq!(non_finite, 0, "logits contain NaN or Inf");
        assert!(distinct > 1000, "logits are nearly constant -- a layer is dead");
        assert!(
            max.abs() < 1.0e4 && variance.sqrt() > 1.0e-3,
            "logit scale is implausible: sd={}, max={max}",
            variance.sqrt()
        );
    }

    // Greedy-decode a short run and look at what comes out. Repeating a single
    // token forever is the classic signature of a model that runs but does not
    // compute -- worth catching here rather than in a latency table.
    let mut produced = Vec::new();
    for position in 1..=8usize {
        // Autoregressive: feed the previous step's token and advance the
        // position, since decode_step reads scratch.token and scratch.positions.
        CudaBuffer::from_tensor(&scratch.token)
            .unwrap()
            .copy_from_host(&scratch.next_token_host().to_le_bytes())
            .unwrap();
        CudaBuffer::from_tensor(&scratch.positions)
            .unwrap()
            .copy_from_host(&(position as i32).to_le_bytes())
            .unwrap();
        decode_step(&ctx, &model, &mut scratch, &mut gdn_states, &mut kv_caches, position);
        ctx.synchronize().unwrap();
        produced.push(scratch.next_token_host());
    }
    println!("greedy token ids: {produced:?}");
    let unique: std::collections::HashSet<_> = produced.iter().collect();
    println!("  {} distinct of {}", unique.len(), produced.len());
    for id in &produced {
        assert!(
            *id >= 0 && (*id as usize) < VOCAB,
            "token id {id} is outside the vocabulary"
        );
    }

    let steps = 16usize;
    let start = Instant::now();
    for position in 1..=steps {
        decode_step(&ctx, &model, &mut scratch, &mut gdn_states, &mut kv_caches, position);
    }
    ctx.synchronize().unwrap();
    let per_token = start.elapsed().as_secs_f64() / steps as f64;

    println!(
        "\ndecode: {:7.2} ms/token   {:6.2} tok/s   ({steps} steps, batch 1)",
        per_token * 1e3,
        1.0 / per_token
    );

    // Where the time goes. Each phase synchronizes, so the sum exceeds the
    // pipelined total above; the point is the ratio between phases.
    macro_rules! phase {
        ($label:expr, $iterations:expr, $body:block) => {{
            $body
            ctx.synchronize().unwrap();
            let start = Instant::now();
            for _ in 0..$iterations {
                $body
            }
            ctx.synchronize().unwrap();
            let each = start.elapsed().as_secs_f64() / $iterations as f64;
            println!("  {:30} {:8.3} ms  x{:3} = {:7.2} ms",
                     $label, each * 1e3, $iterations, each * 1e3);
            each
        }};
    }

    println!("\nper-phase (synchronized, so these over-count):");
    let mut probe = Scratch::new(&ctx);
    let gdn_layer = model.layers.iter().find_map(|l| match l {
        Layer::Gdn(g) => Some(g),
        _ => None,
    }).unwrap();

    let qkv_quantized = view(&probe.fp8_activation, vec![1, HIDDEN], DType::F8E4M3);
    let mut qkv_out = view(&probe.gdn_qkv, vec![1, QKV_WIDTH], DType::BF16);
    // The GEMM contract reads [K, N]; reinterpreting the [N, K] weight gives
    // wrong numbers but the same sequential sweep, which is what is timed.
    let gemm_weight = view(&gdn_layer.qkv.weight, vec![HIDDEN, QKV_WIDTH], DType::F8E4M3);
    let qkv_each = phase!("fp8 qkv via GEMM", 50, {
        let mut args = ops::GemmArgs::new(&qkv_quantized, &gemm_weight, &mut qkv_out);
        args.quantization = ops::GemmQuantization::Fp8UnitScale;
        args.alpha = gdn_layer.qkv.alpha;
        ops::gemm(&ctx, args).unwrap();
    });

    // The decode path now uses the GEMV; keep the comparison so a regression
    // in either shows up.
    let gemv_weight = view(&gdn_layer.qkv.weight, vec![QKV_WIDTH, HIDDEN], DType::F8E4M3);
    let gemv_activation = view(&probe.fp8_activation, vec![HIDDEN], DType::F8E4M3);
    let gemv_out = view(&probe.gdn_qkv, vec![QKV_WIDTH], DType::BF16);
    let gemv_each = phase!("fp8 qkv via vectorized GEMV", 50, {
        ops::fp8_gemv(&ctx, &gemv_weight, &gemv_activation, &gemv_out,
                      gdn_layer.qkv.alpha).unwrap();
    });
    let weight_bytes = (QKV_WIDTH * HIDDEN) as f64;
    println!(
        "    GEMM {:6.1} GB/s   GEMV {:6.1} GB/s   ({:+.0}%)",
        weight_bytes / qkv_each / 1e9,
        weight_bytes / gemv_each / 1e9,
        (qkv_each / gemv_each - 1.0) * 100.0
    );

    let state = zeros(&ctx, vec![GDN_V_HEADS, GDN_HEAD_DIM, GDN_HEAD_DIM], DType::F32);
    let (gq, gk, gv) = split_gdn_qkv(&ctx, &probe.gdn_conv);
    let recurrent_each = phase!("gdn recurrent step", 50, {
        ops::gdn_recurrent_step(&ctx, &state, &gq, &gk, &gv, &probe.gdn_decay,
                                &probe.gdn_beta, &probe.gdn_readout, GDN_K_HEADS).unwrap();
    });

    let mlp_each = phase!("nvfp4 mlp block", 50, {
        nvfp4_mlp(&ctx, &gdn_layer.gate_up, &gdn_layer.down, &gdn_layer.post_norm, &mut probe);
    });

    let mut logits = zeros(&ctx, vec![1, VOCAB], DType::BF16);
    let head_activation = zeros(&ctx, vec![1, HIDDEN / 2], DType::E2M1Pair);
    let head_scales = zeros(&ctx, vec![ops::nvfp4_scale_buffer_bytes(1, HIDDEN, BLOCK).unwrap()], DType::F8E4M3);
    let head_each = phase!("lm_head", 20, {
        ops::gemm(&ctx, ops::GemmArgs::nvfp4(&head_activation, &head_scales,
                  &model.lm_head.packed, &model.lm_head.scales, BLOCK,
                  model.lm_head.alpha, &mut logits)).unwrap();
    });

    println!(
        "\n  64 MLP blocks              {:7.2} ms\n  \
         48 gdn recurrent steps     {:7.2} ms\n  \
         48 qkv projections         {:7.2} ms\n  \
         1 lm_head                  {:7.2} ms",
        mlp_each * 64.0 * 1e3,
        recurrent_each * 48.0 * 1e3,
        qkv_each * 48.0 * 1e3,
        head_each * 1e3
    );
}

// ===========================================================================
// Reference-tensor comparison
//
// Replays layer 0 (GDN) and layer 3 (full attention) against the verified
// PyTorch reference under devlocal/qwen38-nvfp4/reference-tensors, and writes
// every major intermediate as raw little-endian f32 for compare_reference.py.
//
// Three things make this different from `dump_layer_outputs`:
//
//   1. Each layer is fed the *reference's own* input hidden state, not the
//      previous layer's output. Layer 3 does not consume layer 0's output --
//      layers 1 and 2 sit between them in the real model and the reference
//      never built them -- and feeding one into the other would let an early
//      error hide where divergence actually starts.
//   2. Fusion is off. `nvfp4_quantize_rms_norm` and `nvfp4_quantize_swiglu`
//      skip a BF16 rounding that the reference performs, which is a change of
//      computational semantics and has to be measured on its own rather than
//      assumed equivalent. This test establishes the unfused baseline.
//   3. The default sequence length is 8, not 1. At position 0 mRoPE is the
//      identity and softmax over a single key is 1.0, so a seq_len=1 run
//      passes with a broken RoPE, a broken 1/sqrt(head_dim) scale or a broken
//      GQA repeat. Set APXINF_REF_SEQ=1 to get the degenerate case anyway.
// ===========================================================================

/// Tokens to drive. 8 is the primary case; see the module note above.
fn reference_sequence_length() -> usize {
    std::env::var("APXINF_REF_SEQ")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(8)
}

/// `hidden[0, t, i] = bf16(sin((t * HIDDEN + i) * 0.01))`.
///
/// The sine is evaluated in f64 and rounded to BF16 in one step. Evaluating it
/// in f32 changes 196 of 40960 elements, and going f64 -> f32 -> BF16 rounds
/// twice, so neither reproduces the reference input exactly.
fn reference_hidden(seq_len: usize) -> Vec<half::bf16> {
    (0..seq_len * HIDDEN)
        .map(|index| half::bf16::from_f64((index as f64 * 0.01).sin()))
        .collect()
}

/// Collects host copies of device intermediates, one entry per reference
/// tensor name, and writes them once at the end.
///
/// Most tensors are accumulated across tokens: the harness is a decode loop,
/// so a `[1, seq, width]` reference tensor is built one row at a time in token
/// order. Recurrent state is not -- only its final value is meaningful -- so it
/// uses `set_f32`, which replaces instead of appending.
///
/// Every read synchronizes first. `CudaBuffer::copy_to_host` is a plain
/// `cudaMemcpy`, which orders against the null stream only, while the ops run
/// on the session's stream -- so without this a tap reads whatever happens to
/// be in the buffer when the copy is issued, not the kernel's result. The
/// symptom is distinctive and worth recognising: a tap taken right after a
/// kernel comes back as zeros or stale data while the *same buffer* read two
/// taps later looks correct, because the intervening launches gave the first
/// kernel time to land. `dump_layer_outputs`'s `apxdump_tensor` has this bug.
struct RefDump {
    dir: std::path::PathBuf,
    tensors: std::collections::BTreeMap<String, Vec<f32>>,
}

impl RefDump {
    fn new(subdir: &str) -> RefDump {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../devlocal/qwen38-nvfp4")
            .join(subdir);
        std::fs::create_dir_all(&dir).unwrap();
        RefDump { dir, tensors: std::collections::BTreeMap::new() }
    }

    fn read_bf16(ctx: &CudaContext, tensor: &Tensor, count: usize) -> Vec<f32> {
        ctx.synchronize().unwrap();
        let mut bytes = vec![0u8; count * 2];
        CudaBuffer::from_tensor(tensor).unwrap().copy_to_host(&mut bytes).unwrap();
        bytes
            .chunks_exact(2)
            .map(|pair| half::bf16::from_bits(u16::from_le_bytes([pair[0], pair[1]])).to_f32())
            .collect()
    }

    fn read_f32(ctx: &CudaContext, tensor: &Tensor, count: usize) -> Vec<f32> {
        ctx.synchronize().unwrap();
        let mut bytes = vec![0u8; count * 4];
        CudaBuffer::from_tensor(tensor).unwrap().copy_to_host(&mut bytes).unwrap();
        bytes
            .chunks_exact(4)
            .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
            .collect()
    }

    fn push_bf16(&mut self, ctx: &CudaContext, name: &str, tensor: &Tensor, count: usize) {
        let values = Self::read_bf16(ctx, tensor, count);
        self.tensors.entry(name.to_string()).or_default().extend(values);
    }

    fn push_f32(&mut self, ctx: &CudaContext, name: &str, tensor: &Tensor, count: usize) {
        let values = Self::read_f32(ctx, tensor, count);
        self.tensors.entry(name.to_string()).or_default().extend(values);
    }

    fn set_f32(&mut self, ctx: &CudaContext, name: &str, tensor: &Tensor, count: usize) {
        let values = Self::read_f32(ctx, tensor, count);
        self.tensors.insert(name.to_string(), values);
    }

    fn write(&self) {
        for (name, values) in &self.tensors {
            let mut raw = Vec::with_capacity(values.len() * 4);
            for value in values {
                raw.extend_from_slice(&value.to_le_bytes());
            }
            std::fs::write(self.dir.join(format!("{name}.f32")), raw).unwrap();
        }
        println!("wrote {} tensors -> {}", self.tensors.len(), self.dir.display());
    }
}

/// MLP with fusion off: RMSNorm and SwiGLU each land in BF16 before being
/// quantized, which is the rounding the reference performs and the fused
/// quantizers skip.
///
/// `names` is (post_norm, gate_up, swiglu, down_proj, layer_out) so the same
/// body can carry layer 0's numbering and layer 3's.
#[allow(clippy::too_many_arguments)]
fn unfused_mlp(
    ctx: &CudaContext,
    gate_up: &Nvfp4Weight,
    down: &Nvfp4Weight,
    norm_weight: &Tensor,
    scratch: &mut Scratch,
    swiglu_out: &Tensor,
    dump: &mut RefDump,
    names: (&str, &str, &str, &str, &str),
) {
    ops::rms_norm(ctx, &scratch.hidden, norm_weight, &scratch.normalized, EPSILON).unwrap();
    dump.push_bf16(ctx, names.0, &scratch.normalized, HIDDEN);
    ops::nvfp4_quantize_activation(
        ctx,
        &scratch.normalized,
        &scratch.nvfp4_activation,
        &scratch.nvfp4_scales,
        gate_up.input_scale,
        BLOCK,
        ops::ScaleLayout::GemmAtom,
    )
    .unwrap();
    ops::gemm(
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
    // Gate first, then up -- the order load_fused_gate_up concatenates them in
    // and the order swiglu_kernel reads them back. Dumped as one block and
    // split on the comparison side.
    dump.push_bf16(ctx, names.1, &scratch.mlp_fused, 2 * INTERMEDIATE);

    ops::swiglu(ctx, &scratch.mlp_fused, swiglu_out).unwrap();
    dump.push_bf16(ctx, names.2, swiglu_out, INTERMEDIATE);
    ops::nvfp4_quantize_activation(
        ctx,
        swiglu_out,
        &scratch.mlp_activation,
        &scratch.mlp_scales,
        down.input_scale,
        BLOCK,
        ops::ScaleLayout::GemmAtom,
    )
    .unwrap();
    ops::gemm(
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
    dump.push_bf16(ctx, names.3, &scratch.mlp_out, HIDDEN);
    ops::add_into(ctx, &scratch.mlp_out, &scratch.hidden).unwrap();
    dump.push_bf16(ctx, names.4, &scratch.hidden, HIDDEN);
}

/// One GDN token, with a tap after every stage the reference records.
fn gdn_reference_step(
    ctx: &CudaContext,
    gdn: &GdnLayer,
    scratch: &mut Scratch,
    state: &mut GdnState,
    swiglu_out: &Tensor,
    dump: &mut RefDump,
) {
    dump.push_bf16(ctx, "l0_00_hidden_in", &scratch.hidden, HIDDEN);
    ops::rms_norm(ctx, &scratch.hidden, &gdn.input_norm, &scratch.normalized, EPSILON).unwrap();
    dump.push_bf16(ctx, "l0_01_input_layernorm", &scratch.normalized, HIDDEN);

    fp8_projection(ctx, &gdn.qkv, &scratch.normalized, &scratch.fp8_activation, &mut scratch.gdn_qkv);
    dump.push_bf16(ctx, "l0_02_in_proj_qkv", &scratch.gdn_qkv, QKV_WIDTH);
    fp8_projection(ctx, &gdn.z, &scratch.normalized, &scratch.fp8_activation, &mut scratch.gdn_z);
    dump.push_bf16(ctx, "l0_03_in_proj_z", &scratch.gdn_z, Z_WIDTH);

    bf16_matvec(ctx, &gdn.in_proj_a, &scratch.normalized, &scratch.gdn_a);
    bf16_matvec(ctx, &gdn.in_proj_b, &scratch.normalized, &scratch.gdn_b);
    dump.push_bf16(ctx, "l0_05_in_proj_a", &scratch.gdn_a, GDN_V_HEADS);
    dump.push_bf16(ctx, "l0_04_in_proj_b", &scratch.gdn_b, GDN_V_HEADS);

    let qkv_flat = view(&scratch.gdn_qkv, vec![QKV_WIDTH], DType::BF16);
    ops::gdn_causal_conv_step(ctx, &state.conv_window, &qkv_flat, &gdn.conv_weight, &scratch.gdn_conv)
        .unwrap();
    // The conv kernel folds SiLU in, so only the post-activation tap exists.
    dump.push_bf16(ctx, "l0_07_conv1d_silu", &scratch.gdn_conv, QKV_WIDTH);

    let (q, k, v) = split_gdn_qkv(ctx, &scratch.gdn_conv);
    dump.push_bf16(ctx, "l0_08_q_split", &q, GDN_K_HEADS * GDN_HEAD_DIM);
    dump.push_bf16(ctx, "l0_08_k_split", &k, GDN_K_HEADS * GDN_HEAD_DIM);
    dump.push_bf16(ctx, "l0_08_v_split", &v, GDN_V_HEADS * GDN_HEAD_DIM);

    // q and k are views into gdn_conv and the normalization is in place, so
    // the split tap above has to come first.
    ops::gdn_l2_normalize_heads(ctx, &q, EPSILON).unwrap();
    ops::gdn_l2_normalize_heads(ctx, &k, EPSILON).unwrap();
    dump.push_bf16(ctx, "l0_12_q_l2norm", &q, GDN_K_HEADS * GDN_HEAD_DIM);
    dump.push_bf16(ctx, "l0_12_k_l2norm", &k, GDN_K_HEADS * GDN_HEAD_DIM);

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
    // gdn_decay holds g, the log decay, which is what l0_10_g_log_decay is.
    dump.push_f32(ctx, "l0_10_g_log_decay", &scratch.gdn_decay, GDN_V_HEADS);
    dump.push_f32(ctx, "l0_09_beta", &scratch.gdn_beta, GDN_V_HEADS);

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
    dump.push_bf16(ctx, "l0_16_core_attn_out", &scratch.gdn_readout, GDN_V_HEADS * GDN_HEAD_DIM);
    // State is [v_heads, v_dim, k_dim] here against the reference's
    // [v_heads, k_dim, v_dim]; the comparison transposes the last two axes.
    dump.set_f32(ctx, "l0_15_recurrent_state_final",
        &state.recurrent,
        GDN_V_HEADS * GDN_HEAD_DIM * GDN_HEAD_DIM,
    );

    ops::gdn_gated_norm(
        ctx,
        &scratch.gdn_readout,
        &gdn_z_heads(ctx, &scratch.gdn_z),
        &gdn.norm_weight,
        &scratch.gdn_gated,
        EPSILON,
    )
    .unwrap();
    dump.push_bf16(ctx, "l0_17_gated_rmsnorm", &scratch.gdn_gated, Z_WIDTH);

    let flat = flatten(ctx, &scratch.gdn_gated, Z_WIDTH);
    fp8_projection(ctx, &gdn.out, &flat, &scratch.gdn_fp8, &mut scratch.projected);
    dump.push_bf16(ctx, "l0_18_out_proj", &scratch.projected, HIDDEN);
    ops::add_into(ctx, &scratch.projected, &scratch.hidden).unwrap();
    dump.push_bf16(ctx, "l0_19_residual_1", &scratch.hidden, HIDDEN);

    unfused_mlp(
        ctx,
        &gdn.gate_up,
        &gdn.down,
        &gdn.post_norm,
        scratch,
        swiglu_out,
        dump,
        (
            "l0_20_post_attention_layernorm",
            "l0_21_gate_up",
            "l0_22_swiglu",
            "l0_23_down_proj",
            "l0_24_layer_out",
        ),
    );
}

/// One full-attention token, with a tap after every stage the reference records.
#[allow(clippy::too_many_arguments)]
fn attention_reference_step(
    ctx: &CudaContext,
    attention: &AttentionLayer,
    scratch: &mut Scratch,
    cache: &mut KvCache,
    swiglu_out: &Tensor,
    rotary: usize,
    position: usize,
    dump: &mut RefDump,
) {
    dump.push_bf16(ctx, "l3_00_hidden_in", &scratch.hidden, HIDDEN);
    ops::rms_norm(ctx, &scratch.hidden, &attention.input_norm, &scratch.normalized, EPSILON).unwrap();
    dump.push_bf16(ctx, "l3_01_input_layernorm", &scratch.normalized, HIDDEN);

    fp8_projection(ctx, &attention.q, &scratch.normalized, &scratch.fp8_activation, &mut scratch.qkv_fused);
    dump.push_bf16(ctx, "l3_03_q_proj_full", &scratch.qkv_fused, 2 * HEADS * HEAD_DIM);
    let fused_heads = view(&scratch.qkv_fused, vec![1, HEADS, 2 * HEAD_DIM], DType::BF16);
    ops::split_query_and_gate(ctx, &fused_heads, &scratch.query, &scratch.query_gate).unwrap();
    dump.push_bf16(ctx, "l3_04_query_split", &scratch.query, HEADS * HEAD_DIM);
    dump.push_bf16(ctx, "l3_04_gate_split", &scratch.query_gate, HEADS * HEAD_DIM);

    let mut key_slot = cache_slot(&cache.keys, position, vec![1, KV_HEADS * HEAD_DIM]);
    let mut value_slot = cache_slot(&cache.values, position, vec![1, KV_HEADS * HEAD_DIM]);
    fp8_projection(ctx, &attention.k, &scratch.normalized, &scratch.fp8_activation, &mut key_slot);
    fp8_projection(ctx, &attention.v, &scratch.normalized, &scratch.fp8_activation, &mut value_slot);
    dump.push_bf16(ctx, "l3_05_k_proj", &key_slot, KV_HEADS * HEAD_DIM);
    dump.push_bf16(ctx, "l3_05_v_proj", &value_slot, KV_HEADS * HEAD_DIM);

    let key_heads = cache_slot(&cache.keys, position, vec![KV_HEADS, HEAD_DIM]);
    let query_heads = view(&scratch.query, vec![HEADS, HEAD_DIM], DType::BF16);
    ops::head_rms_norm(ctx, &query_heads, &attention.q_norm, EPSILON).unwrap();
    ops::head_rms_norm(ctx, &key_heads, &attention.k_norm, EPSILON).unwrap();
    dump.push_bf16(ctx, "l3_06_q_norm", &scratch.query, HEADS * HEAD_DIM);
    dump.push_bf16(ctx, "l3_06_k_norm", &key_slot, KV_HEADS * HEAD_DIM);
    // v is never normalized; the reference records it here for alignment.
    dump.push_bf16(ctx, "l3_06_v_states", &value_slot, KV_HEADS * HEAD_DIM);

    let query_tokens = view(&scratch.query, vec![1, HEADS, HEAD_DIM], DType::BF16);
    let key_tokens = cache_slot(&cache.keys, position, vec![1, KV_HEADS, HEAD_DIM]);
    ops::partial_rope(ctx, &query_tokens, &scratch.positions, rotary, ROPE_THETA).unwrap();
    ops::partial_rope(ctx, &key_tokens, &scratch.positions, rotary, ROPE_THETA).unwrap();
    dump.push_bf16(ctx, "l3_07_q_rope", &scratch.query, HEADS * HEAD_DIM);
    dump.push_bf16(ctx, "l3_07_k_rope", &key_slot, KV_HEADS * HEAD_DIM);

    let valid = position + 1;
    // Viewed at `valid`: FA2 requires key_capacity == key_tokens.
    let keys = cache_rows(&cache.keys, valid, vec![1, valid, KV_HEADS, HEAD_DIM]);
    let values = cache_rows(&cache.values, valid, vec![1, valid, KV_HEADS, HEAD_DIM]);
    let query_4d = view(&scratch.query, vec![1, 1, HEADS, HEAD_DIM], DType::BF16);
    let mut out_4d = view(&scratch.attention_out, vec![1, 1, HEADS, HEAD_DIM], DType::BF16);
    let mut args = ops::KvCacheAttentionArgs::new(&query_4d, &keys, &values, &mut out_4d);
    args.valid_key_tokens = valid;
    args.query_start = position;
    args.policy.cache_dir = attention_cache_dir();
    ops::kv_cache_attention(ctx, args).unwrap();
    dump.push_bf16(ctx, "l3_09_attn_out", &scratch.attention_out, HEADS * HEAD_DIM);

    let gate_flat = view(&scratch.query_gate, vec![1, HEADS * HEAD_DIM], DType::BF16);
    ops::apply_output_gate(ctx, &scratch.attention_out, &gate_flat).unwrap();
    dump.push_bf16(ctx, "l3_10_attn_gated", &scratch.attention_out, HEADS * HEAD_DIM);

    fp8_projection(ctx, &attention.o, &scratch.attention_out, &scratch.attention_fp8, &mut scratch.projected);
    dump.push_bf16(ctx, "l3_11_o_proj", &scratch.projected, HIDDEN);
    ops::add_into(ctx, &scratch.projected, &scratch.hidden).unwrap();
    dump.push_bf16(ctx, "l3_12_residual_1", &scratch.hidden, HIDDEN);

    unfused_mlp(
        ctx,
        &attention.gate_up,
        &attention.down,
        &attention.post_norm,
        scratch,
        swiglu_out,
        dump,
        (
            "l3_13_post_attention_layernorm",
            "l3_14_gate_up",
            "l3_15_swiglu",
            "l3_16_down_proj",
            "l3_17_layer_out",
        ),
    );
}

/// Replay layers 0 and 3 on the reference's input and dump every tap.
///
/// Pairs with devlocal/qwen38-nvfp4/scripts/compare_reference.py, which reads
/// the .f32 files this writes and reports cosine and relative L2 against
/// reference-tensors/ in the acceptance order.
#[test]
#[ignore = "requires the 20 GiB Qwen3.8-27B-NVFP4 checkpoint and a GPU"]
fn compare_against_reference_tensors() {
    let ctx = CudaContext::new(0).unwrap();
    let seq_len = reference_sequence_length();
    let tensors = checkpoint();
    let model = load_model(&ctx, &tensors, false);
    ctx.synchronize().unwrap();
    drop(tensors);

    let mut dump = RefDump::new(&format!("apxdump-ref-seq{seq_len}"));
    let hidden_input = reference_hidden(seq_len);
    let mut scratch = Scratch::new(&ctx);
    let swiglu_out = zeros(&ctx, vec![1, INTERMEDIATE], DType::BF16);
    let rotary = ops::rotary_dim(HEAD_DIM, PARTIAL_ROTARY);

    // A fresh state per layer: the reference starts both layers from an empty
    // cache and an empty recurrent state.
    let mut gdn_state = GdnState {
        recurrent: zeros(&ctx, vec![GDN_V_HEADS, GDN_HEAD_DIM, GDN_HEAD_DIM], DType::F32),
        conv_window: zeros(&ctx, vec![QKV_WIDTH, CONV_WIDTH], DType::F32),
    };
    let capacity = seq_len.max(1);
    let mut cache = KvCache {
        keys: zeros(&ctx, vec![1, capacity, KV_HEADS, HEAD_DIM], DType::BF16),
        values: zeros(&ctx, vec![1, capacity, KV_HEADS, HEAD_DIM], DType::BF16),
    };

    let gdn = match &model.layers[0] {
        Layer::Gdn(gdn) => gdn.as_ref(),
        Layer::Attention(_) => panic!("layer 0 should be a GDN layer"),
    };
    let attention = match &model.layers[3] {
        Layer::Attention(attention) => attention.as_ref(),
        Layer::Gdn(_) => panic!("layer 3 should be a full-attention layer"),
    };

    // Each token is loaded straight from the reference input, so no step
    // inherits the previous step's residual stream.
    let load_token = |scratch: &Scratch, token: usize| {
        let row = &hidden_input[token * HIDDEN..(token + 1) * HIDDEN];
        let mut bytes = Vec::with_capacity(HIDDEN * 2);
        for value in row {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        CudaBuffer::from_tensor(&scratch.hidden).unwrap().copy_from_host(&bytes).unwrap();
    };

    for token in 0..seq_len {
        load_token(&scratch, token);
        gdn_reference_step(&ctx, gdn, &mut scratch, &mut gdn_state, &swiglu_out, &mut dump);
        ctx.synchronize().unwrap();
    }

    for token in 0..seq_len {
        load_token(&scratch, token);
        CudaBuffer::from_tensor(&scratch.positions)
            .unwrap()
            .copy_from_host(&(token as i32).to_le_bytes())
            .unwrap();
        attention_reference_step(
            &ctx,
            attention,
            &mut scratch,
            &mut cache,
            &swiglu_out,
            rotary,
            token,
            &mut dump,
        );
        ctx.synchronize().unwrap();
    }

    dump.write();
    println!("REFERENCE DUMP COMPLETE seq_len={seq_len}");
}

// ===========================================================================
// Whole-model teacher-forcing (ACCEPTANCE.md §5)
//
// Drives the full 64-layer decode over fixed token sequences, feeding the
// *given* token at every position (teacher forcing -- NOT the port's own
// argmax), and dumps the full FP32 logits vector at every position for
// compare_teacher_forcing.py. Two modes:
//   fused   -- the production path (nvfp4_quantize_rms_norm / _swiglu), and the
//              fused rms_norm+quantize lm_head as in decode_step.
//   unfused -- rms_norm then nvfp4_quantize_activation, swiglu then
//              nvfp4_quantize_activation, matching what the reference computes
//              (a BF16 rounding between norm/activation and quantize).
//
// APXINF_TF_TOKENS   path to a "name id id id..." per-line token file
// APXINF_TF_OUT      output directory for <name>.logits.f32 + argmax
// APXINF_TF_FUSED    "1" => fused path, anything else => unfused (default)
// ===========================================================================

/// MLP with fusion off but no tapping: RMSNorm and SwiGLU each land in BF16
/// before quantization, matching the reference.
fn unfused_mlp_quiet(
    ctx: &CudaContext,
    gate_up: &Nvfp4Weight,
    down: &Nvfp4Weight,
    norm_weight: &Tensor,
    scratch: &mut Scratch,
    swiglu_out: &Tensor,
) {
    ops::rms_norm(ctx, &scratch.hidden, norm_weight, &scratch.normalized, EPSILON).unwrap();
    ops::nvfp4_quantize_activation(
        ctx, &scratch.normalized, &scratch.nvfp4_activation, &scratch.nvfp4_scales,
        gate_up.input_scale, BLOCK, ops::ScaleLayout::GemmAtom,
    ).unwrap();
    ops::gemm(ctx, ops::GemmArgs::nvfp4(
        &scratch.nvfp4_activation, &scratch.nvfp4_scales, &gate_up.packed, &gate_up.scales,
        BLOCK, gate_up.alpha, &mut scratch.mlp_fused,
    )).unwrap();
    ops::swiglu(ctx, &scratch.mlp_fused, swiglu_out).unwrap();
    ops::nvfp4_quantize_activation(
        ctx, swiglu_out, &scratch.mlp_activation, &scratch.mlp_scales,
        down.input_scale, BLOCK, ops::ScaleLayout::GemmAtom,
    ).unwrap();
    ops::gemm(ctx, ops::GemmArgs::nvfp4(
        &scratch.mlp_activation, &scratch.mlp_scales, &down.packed, &down.scales,
        BLOCK, down.alpha, &mut scratch.mlp_out,
    )).unwrap();
    ops::add_into(ctx, &scratch.mlp_out, &scratch.hidden).unwrap();
}

/// One decoder layer for teacher forcing, with a fused/unfused MLP switch. The
/// mixer (attention / GDN) is identical to decode_step; only the MLP quantizer
/// fusion differs.
#[allow(clippy::too_many_arguments)]
fn tf_run_layer(
    ctx: &CudaContext,
    layer: &Layer,
    scratch: &mut Scratch,
    gdn_states: &mut [GdnState],
    kv_caches: &mut [KvCache],
    gdn_index: &mut usize,
    attention_index: &mut usize,
    rotary: usize,
    position: usize,
    unfused: bool,
    swiglu_out: &Tensor,
) {
    match layer {
        Layer::Attention(attention) => {
            let cache = &mut kv_caches[*attention_index];
            *attention_index += 1;
            ops::rms_norm(ctx, &scratch.hidden, &attention.input_norm, &scratch.normalized, EPSILON).unwrap();
            fp8_projection(ctx, &attention.q, &scratch.normalized, &scratch.fp8_activation, &mut scratch.qkv_fused);
            let fused_heads = view(&scratch.qkv_fused, vec![1, HEADS, 2 * HEAD_DIM], DType::BF16);
            ops::split_query_and_gate(ctx, &fused_heads, &scratch.query, &scratch.query_gate).unwrap();
            let mut key_slot = cache_slot(&cache.keys, position, vec![1, KV_HEADS * HEAD_DIM]);
            let mut value_slot = cache_slot(&cache.values, position, vec![1, KV_HEADS * HEAD_DIM]);
            fp8_projection(ctx, &attention.k, &scratch.normalized, &scratch.fp8_activation, &mut key_slot);
            fp8_projection(ctx, &attention.v, &scratch.normalized, &scratch.fp8_activation, &mut value_slot);
            let key_heads = cache_slot(&cache.keys, position, vec![KV_HEADS, HEAD_DIM]);
            let query_heads = view(&scratch.query, vec![HEADS, HEAD_DIM], DType::BF16);
            ops::head_rms_norm(ctx, &query_heads, &attention.q_norm, EPSILON).unwrap();
            ops::head_rms_norm(ctx, &key_heads, &attention.k_norm, EPSILON).unwrap();
            let query_tokens = view(&scratch.query, vec![1, HEADS, HEAD_DIM], DType::BF16);
            let key_tokens = cache_slot(&cache.keys, position, vec![1, KV_HEADS, HEAD_DIM]);
            ops::partial_rope(ctx, &query_tokens, &scratch.positions, rotary, ROPE_THETA).unwrap();
            ops::partial_rope(ctx, &key_tokens, &scratch.positions, rotary, ROPE_THETA).unwrap();
            let valid = position + 1;
            // Viewed at `valid`, not at the full allocation: the FA2 candidate
            // requires key_capacity == key_tokens, and a capacity-shaped view
            // fails that for every step but the last, dropping decode onto the
            // naive kernel whose cost grows with KV length.
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
            if unfused {
                unfused_mlp_quiet(ctx, &attention.gate_up, &attention.down, &attention.post_norm, scratch, swiglu_out);
            } else {
                nvfp4_mlp(ctx, &attention.gate_up, &attention.down, &attention.post_norm, scratch);
            }
        }
        Layer::Gdn(gdn) => {
            let state = &mut gdn_states[*gdn_index];
            *gdn_index += 1;
            ops::rms_norm(ctx, &scratch.hidden, &gdn.input_norm, &scratch.normalized, EPSILON).unwrap();
            fp8_projection(ctx, &gdn.qkv, &scratch.normalized, &scratch.fp8_activation, &mut scratch.gdn_qkv);
            fp8_projection(ctx, &gdn.z, &scratch.normalized, &scratch.fp8_activation, &mut scratch.gdn_z);
            let qkv_flat = view(&scratch.gdn_qkv, vec![QKV_WIDTH], DType::BF16);
            ops::gdn_causal_conv_step(ctx, &state.conv_window, &qkv_flat, &gdn.conv_weight, &scratch.gdn_conv).unwrap();
            let (q, k, v) = split_gdn_qkv(ctx, &scratch.gdn_conv);
            ops::gdn_l2_normalize_heads(ctx, &q, EPSILON).unwrap();
            ops::gdn_l2_normalize_heads(ctx, &k, EPSILON).unwrap();
            bf16_matvec(ctx, &gdn.in_proj_a, &scratch.normalized, &scratch.gdn_a);
            bf16_matvec(ctx, &gdn.in_proj_b, &scratch.normalized, &scratch.gdn_b);
            ops::gdn_decay_and_beta(ctx, &scratch.gdn_a, &scratch.gdn_b, &gdn.a_log, &gdn.dt_bias, &scratch.gdn_decay, &scratch.gdn_beta).unwrap();
            ops::gdn_recurrent_step(ctx, &state.recurrent, &q, &k, &v, &scratch.gdn_decay, &scratch.gdn_beta, &scratch.gdn_readout, GDN_K_HEADS).unwrap();
            ops::gdn_gated_norm(ctx, &scratch.gdn_readout, &gdn_z_heads(ctx, &scratch.gdn_z), &gdn.norm_weight, &scratch.gdn_gated, EPSILON).unwrap();
            let flat = flatten(ctx, &scratch.gdn_gated, Z_WIDTH);
            fp8_projection(ctx, &gdn.out, &flat, &scratch.gdn_fp8, &mut scratch.projected);
            ops::add_into(ctx, &scratch.projected, &scratch.hidden).unwrap();
            if unfused {
                unfused_mlp_quiet(ctx, &gdn.gate_up, &gdn.down, &gdn.post_norm, scratch, swiglu_out);
            } else {
                nvfp4_mlp(ctx, &gdn.gate_up, &gdn.down, &gdn.post_norm, scratch);
            }
        }
    }
}

/// Embedding -> 64 layers -> final norm -> lm_head, producing logits into
/// scratch.logits. `unfused` selects the MLP and lm_head quantizer fusion.
/// When `hidden_taps` is Some, the BF16 hidden after each layer is appended to
/// taps[layer] (used to localize inter-layer drift).
#[allow(clippy::too_many_arguments)]
fn tf_decode_step(
    ctx: &CudaContext,
    model: &Model,
    scratch: &mut Scratch,
    gdn_states: &mut [GdnState],
    kv_caches: &mut [KvCache],
    position: usize,
    unfused: bool,
    swiglu_out: &Tensor,
    mut hidden_taps: Option<&mut Vec<Vec<f32>>>,
) {
    ops::embedding_gather(ctx, &model.embedding, &scratch.token, &scratch.hidden).unwrap();
    let rotary = ops::rotary_dim(HEAD_DIM, PARTIAL_ROTARY);
    let mut gdn_index = 0usize;
    let mut attention_index = 0usize;
    for (layer_index, layer) in model.layers.iter().enumerate() {
        tf_run_layer(ctx, layer, scratch, gdn_states, kv_caches,
                     &mut gdn_index, &mut attention_index, rotary, position, unfused, swiglu_out);
        if let Some(taps) = hidden_taps.as_deref_mut() {
            ctx.synchronize().unwrap();
            let mut bytes = vec![0u8; HIDDEN * 2];
            CudaBuffer::from_tensor(&scratch.hidden).unwrap().copy_to_host(&mut bytes).unwrap();
            let f: Vec<f32> = bytes.chunks_exact(2)
                .map(|p| half::bf16::from_bits(u16::from_le_bytes([p[0], p[1]])).to_f32())
                .collect();
            taps[layer_index].extend(f);
        }
    }
    ops::rms_norm(ctx, &scratch.hidden, &model.final_norm, &scratch.normalized, EPSILON).unwrap();
    // Fused vs unfused only differs in whether the normalized BF16 is
    // materialized; the lm_head GEMM is identical. decode_step uses the fused
    // rms_norm+quantize; here rms_norm already ran, so both modes quantize the
    // materialized normalized tensor -- which is exactly the unfused lm_head.
    // For the fused mode we reproduce decode_step's fused quantize instead.
    if unfused {
        ops::nvfp4_quantize_activation(
            ctx, &scratch.normalized, &scratch.nvfp4_activation, &scratch.nvfp4_scales,
            model.lm_head.input_scale, BLOCK, ops::ScaleLayout::GemmAtom,
        ).unwrap();
    } else {
        ops::nvfp4_quantize_rms_norm(
            ctx, &scratch.hidden, &model.final_norm, &scratch.nvfp4_activation,
            &scratch.nvfp4_scales, EPSILON, model.lm_head.input_scale, BLOCK,
            ops::ScaleLayout::GemmAtom,
        ).unwrap();
    }
    ops::gemm(ctx, ops::GemmArgs::nvfp4(
        &scratch.nvfp4_activation, &scratch.nvfp4_scales, &model.lm_head.packed,
        &model.lm_head.scales, BLOCK, model.lm_head.alpha, &mut scratch.logits,
    )).unwrap();
    ops::argmax(ctx, &scratch.logits, &scratch.next_token).unwrap();
}

fn tf_read_tokens(path: &str) -> Vec<(String, Vec<i32>)> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() { continue; }
        let mut it = line.split_whitespace();
        let name = it.next().unwrap().to_string();
        let ids: Vec<i32> = it.map(|t| t.parse().unwrap()).collect();
        out.push((name, ids));
    }
    out
}

fn tf_logits_f32(ctx: &CudaContext, scratch: &Scratch) -> Vec<f32> {
    ctx.synchronize().unwrap();
    let mut bytes = vec![0u8; VOCAB * 2];
    CudaBuffer::from_tensor(&scratch.logits).unwrap().copy_to_host(&mut bytes).unwrap();
    bytes.chunks_exact(2)
        .map(|p| half::bf16::from_bits(u16::from_le_bytes([p[0], p[1]])).to_f32())
        .collect()
}

#[test]
#[ignore = "requires the checkpoint and a GPU; pairs with make_reference_fullmodel.py"]
fn teacher_forcing_logits() {
    let ctx = CudaContext::new(0).unwrap();
    let tokens_path = std::env::var("APXINF_TF_TOKENS").expect("set APXINF_TF_TOKENS");
    let out_dir = std::env::var("APXINF_TF_OUT").expect("set APXINF_TF_OUT");
    let unfused = std::env::var("APXINF_TF_FUSED").map(|v| v != "1").unwrap_or(true);
    let prompts = tf_read_tokens(&tokens_path);
    std::fs::create_dir_all(&out_dir).unwrap();
    let hidden_dump = std::env::var("APXINF_TF_HIDDEN").ok();
    println!("teacher forcing: {} prompts, mode={}, out={out_dir}",
             prompts.len(), if unfused { "UNFUSED" } else { "FUSED" });

    let tensors = checkpoint();
    let model = load_model(&ctx, &tensors, false);
    ctx.synchronize().unwrap();
    drop(tensors);

    let max_seq = prompts.iter().map(|(_, ids)| ids.len()).max().unwrap();
    let capacity = max_seq.max(1);
    let swiglu_out = zeros(&ctx, vec![1, INTERMEDIATE], DType::BF16);

    for (name, ids) in &prompts {
        // Fresh state per prompt: independent sequences, no carry-over.
        let mut gdn_states: Vec<GdnState> = (0..LAYERS - LAYERS / FULL_ATTENTION_INTERVAL)
            .map(|_| GdnState {
                recurrent: zeros(&ctx, vec![GDN_V_HEADS, GDN_HEAD_DIM, GDN_HEAD_DIM], DType::F32),
                conv_window: zeros(&ctx, vec![QKV_WIDTH, CONV_WIDTH], DType::F32),
            })
            .collect();
        let mut kv_caches: Vec<KvCache> = (0..LAYERS / FULL_ATTENTION_INTERVAL)
            .map(|_| KvCache {
                keys: zeros(&ctx, vec![1, capacity, KV_HEADS, HEAD_DIM], DType::BF16),
                values: zeros(&ctx, vec![1, capacity, KV_HEADS, HEAD_DIM], DType::BF16),
            })
            .collect();
        let mut scratch = Scratch::new(&ctx);

        let mut all_logits: Vec<f32> = Vec::with_capacity(ids.len() * VOCAB);
        let mut argmax: Vec<i32> = Vec::with_capacity(ids.len());
        let want_hidden = hidden_dump.as_deref() == Some(name.as_str());
        let mut taps: Vec<Vec<f32>> = if want_hidden {
            (0..LAYERS).map(|_| Vec::new()).collect()
        } else {
            Vec::new()
        };
        for (position, &token) in ids.iter().enumerate() {
            // Teacher forcing: feed the prompt's own token, never the argmax.
            CudaBuffer::from_tensor(&scratch.token).unwrap()
                .copy_from_host(&token.to_le_bytes()).unwrap();
            CudaBuffer::from_tensor(&scratch.positions).unwrap()
                .copy_from_host(&(position as i32).to_le_bytes()).unwrap();
            let taps_ref = if want_hidden { Some(&mut taps) } else { None };
            tf_decode_step(&ctx, &model, &mut scratch, &mut gdn_states, &mut kv_caches,
                           position, unfused, &swiglu_out, taps_ref);
            all_logits.extend(tf_logits_f32(&ctx, &scratch));
            argmax.push(scratch.next_token_host());
        }

        if want_hidden {
            for (li, tap) in taps.iter().enumerate() {
                let mut raw = Vec::with_capacity(tap.len() * 4);
                for v in tap { raw.extend_from_slice(&v.to_le_bytes()); }
                std::fs::write(
                    std::path::Path::new(&out_dir).join(format!("hidden_{name}_L{li}.f32")),
                    raw,
                ).unwrap();
            }
            println!("  wrote per-layer hidden taps for {name}");
        }

        let mut raw = Vec::with_capacity(all_logits.len() * 4);
        for v in &all_logits { raw.extend_from_slice(&v.to_le_bytes()); }
        let logits_path = std::path::Path::new(&out_dir).join(format!("{name}.logits.f32"));
        std::fs::write(&logits_path, raw).unwrap();
        let argmax_str: Vec<String> = argmax.iter().map(|v| v.to_string()).collect();
        std::fs::write(
            std::path::Path::new(&out_dir).join(format!("{name}.argmax.txt")),
            argmax_str.join(" "),
        ).unwrap();
        let nonfinite = all_logits.iter().filter(|v| !v.is_finite()).count();
        println!("  {name:10} seq={:3} nonfinite={nonfinite} argmax[:8]={:?}",
                 ids.len(), &argmax[..argmax.len().min(8)]);
    }
    println!("TEACHER FORCING DUMP COMPLETE mode={} -> {out_dir}",
             if unfused { "unfused" } else { "fused" });
}

// -------------------------------------------------------------------------
// Real prompt generation driver: tokenize -> per-token prefill (building KV +
// GDN state) -> greedy decode to EOS or a token budget -> detokenize -> print.
//
// Prefill runs the whole prompt through prefill_step: attention takes all
// queries at once against the cache it just filled, and each GDN layer runs
// the chunked scan instead of the per-token recurrence. Set
// APXINF_QWEN38_PER_TOKEN_PREFILL=1 to fall back to the one-token-at-a-time
// path, which is the reference the batched path is checked against
// (qwen38_gdn_prefill.rs proves the two leave the same GDN state at cosine
// 1.0, relL2 < 1e-3).
//
//   APXINF_QWEN38_CHECKPOINT=/path/to/Qwen3.8-27B-NVFP4 \
//   APXINF_QWEN38_PROMPT="The capital of France is" \
//   APXINF_QWEN38_MAX_NEW=32 \
//     bash crates/apxinf-cuda-new/test-new.sh \
//       test -p apxinf-cuda --test qwen38_end_to_end --release \
//       -- --ignored generate_from_a_real_prompt --nocapture
// -------------------------------------------------------------------------
#[test]
#[ignore = "requires the 20 GiB Qwen3.8-27B-NVFP4 checkpoint and a GPU"]
fn generate_from_a_real_prompt() {
    use apxinf_tokenizer::Tokenizer;

    let ctx = CudaContext::new(0).unwrap();

    let ckpt: PathBuf = std::env::var_os("APXINF_QWEN38_CHECKPOINT")
        .expect("set APXINF_QWEN38_CHECKPOINT")
        .into();
    let tokenizer = Tokenizer::from_file(ckpt.join("tokenizer.json"))
        .expect("load tokenizer.json from the checkpoint directory");

    let prompt = std::env::var("APXINF_QWEN38_PROMPT")
        .unwrap_or_else(|_| "The capital of France is".to_string());
    let max_new: usize = std::env::var("APXINF_QWEN38_MAX_NEW")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);

    // APXINF_QWEN38_PROMPT_LEN=N ignores the text and builds N deterministic
    // token ids, matching a fixed-prompt-length random dataset.
    let prompt_ids: Vec<u32> = match std::env::var("APXINF_QWEN38_PROMPT_LEN") {
        Ok(n) => {
            let n: usize = n.parse().expect("PROMPT_LEN must be an integer");
            // A fixed LCG over the vocab: reproducible, spread across ids, and
            // clear of the special tokens near the top of the range.
            let mut state = 0x2545_F491_4F6C_DD1Du64;
            (0..n)
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((state >> 16) as usize % (VOCAB - 1000)) as u32
                })
                .collect()
        }
        Err(_) => tokenizer.encode(&prompt).expect("encode prompt"),
    };
    let eos = tokenizer.eos_token_id();
    println!("prompt: {prompt:?}");
    println!("prompt tokens ({}): {prompt_ids:?}", prompt_ids.len());
    println!("eos token id: {eos:?}");
    assert!(!prompt_ids.is_empty(), "empty prompt tokenization");

    // Load the model.
    let tensors = checkpoint();
    // The batched prefill projections run as GEMMs, which need the [K, N]
    // FP8 copies; see `Fp8Weight::transposed`.
    let model = load_model(
        &ctx,
        &tensors,
        std::env::var("APXINF_QWEN38_PER_TOKEN_PREFILL").is_err(),
    );
    ctx.synchronize().unwrap();
    drop(tensors);

    // KV cache large enough for prompt + generation.
    let capacity = (prompt_ids.len() + max_new + 8).next_power_of_two();
    let mut gdn_states: Vec<GdnState> = (0..LAYERS - LAYERS / FULL_ATTENTION_INTERVAL)
        .map(|_| GdnState {
            recurrent: zeros(&ctx, vec![GDN_V_HEADS, GDN_HEAD_DIM, GDN_HEAD_DIM], DType::F32),
            conv_window: zeros(&ctx, vec![QKV_WIDTH, CONV_WIDTH], DType::F32),
        })
        .collect();
    let mut kv_caches: Vec<KvCache> = (0..LAYERS / FULL_ATTENTION_INTERVAL)
        .map(|_| KvCache {
            keys: zeros(&ctx, vec![1, capacity, KV_HEADS, HEAD_DIM], DType::BF16),
            values: zeros(&ctx, vec![1, capacity, KV_HEADS, HEAD_DIM], DType::BF16),
        })
        .collect();
    let mut scratch = Scratch::new(&ctx);
    let n_prompt = prompt_ids.len();

    // APXINF_QWEN38_PER_TOKEN_PREFILL=1 keeps the original one-token-at-a-time
    // prefill. It is the reference the batched path is checked against, and
    // the fallback if the [K, N] FP8 copies do not fit.
    let batched_prefill = std::env::var("APXINF_QWEN38_PER_TOKEN_PREFILL").is_err();
    let mut prefill_scratch = batched_prefill
        .then(|| PrefillScratch::new(&ctx, n_prompt.div_ceil(CHUNK) * CHUNK));

    // Warm-up so the GEMM autotuner does not pollute the prefill timing; state
    // is reset right after.
    if let Some(prefill) = prefill_scratch.as_mut() {
        write_tokens(&ctx, prefill, &prompt_ids);
        prefill_step(&ctx, &model, prefill, &mut gdn_states, &mut kv_caches, n_prompt);
    } else {
        set_token(&ctx, &scratch, prompt_ids[0] as i32);
        set_position(&ctx, &scratch, 0);
        decode_step(&ctx, &model, &mut scratch, &mut gdn_states, &mut kv_caches, 0);
    }
    ctx.synchronize().unwrap();
    reset_state(&ctx, &mut gdn_states, &mut kv_caches);

    // --- Prefill: the whole prompt, keeping the last-position logits. --------
    let prefill_start = Instant::now();
    if let Some(prefill) = prefill_scratch.as_mut() {
        write_tokens(&ctx, prefill, &prompt_ids);
        prefill_step(&ctx, &model, prefill, &mut gdn_states, &mut kv_caches, n_prompt);
        prefill_logits(&ctx, &model, prefill, &scratch, n_prompt);
    } else {
        for (pos, &tok) in prompt_ids.iter().enumerate() {
            set_token(&ctx, &scratch, tok as i32);
            set_position(&ctx, &scratch, pos);
            decode_step(&ctx, &model, &mut scratch, &mut gdn_states, &mut kv_caches, pos);
        }
    }
    ctx.synchronize().unwrap();
    let prefill_secs = prefill_start.elapsed().as_secs_f64();
    // After prefill, scratch.next_token holds the argmax at the last prompt
    // position: the first generated token.
    let first_token = scratch.next_token_host();

    println!(
        "\nprefill[{}]: {:7.2} ms  ({} tokens, {:6.1} tok/s)  TTFT {:7.2} ms",
        if batched_prefill { "batched" } else { "per-token" },
        prefill_secs * 1e3,
        n_prompt,
        n_prompt as f64 / prefill_secs,
        prefill_secs * 1e3
    );

    // --- Greedy decode from the first generated token to EOS or the budget. ---
    let mut generated: Vec<u32> = Vec::new();
    let mut next = first_token;
    let decode_start = Instant::now();
    let mut steps = 0usize;
    for i in 0..max_new {
        if next < 0 || (next as usize) >= VOCAB {
            panic!("token id {next} outside vocabulary at step {i}");
        }
        generated.push(next as u32);
        // A fixed-length benchmark must not stop early on EOS.
        let fixed_len = std::env::var("APXINF_QWEN38_FIXED_DECODE").is_ok();
        if !fixed_len && eos.map(|e| e as i32 == next).unwrap_or(false) {
            break;
        }
        let pos = n_prompt + i;
        set_token(&ctx, &scratch, next);
        set_position(&ctx, &scratch, pos);
        decode_step(&ctx, &model, &mut scratch, &mut gdn_states, &mut kv_caches, pos);
        ctx.synchronize().unwrap();
        next = scratch.next_token_host();
        steps += 1;
    }
    let decode_secs = decode_start.elapsed().as_secs_f64();
    let per_token = if steps > 0 { decode_secs / steps as f64 } else { decode_secs };

    println!("generated token ids ({}): {generated:?}", generated.len());
    let text = tokenizer.decode(&generated).expect("detokenize generation");
    println!("\n=== GENERATED TEXT ===\n{prompt}{text}\n======================");
    println!(
        "\ndecode: {:7.3} ms/token   {:6.2} tok/s   ({steps} steps)",
        per_token * 1e3,
        if per_token > 0.0 { 1.0 / per_token } else { 0.0 }
    );

    assert!(!generated.is_empty(), "no tokens generated");
}

fn set_token(ctx: &CudaContext, scratch: &Scratch, id: i32) {
    let _ = ctx;
    CudaBuffer::from_tensor(&scratch.token)
        .unwrap()
        .copy_from_host(&id.to_le_bytes())
        .unwrap();
}

fn set_position(ctx: &CudaContext, scratch: &Scratch, pos: usize) {
    let _ = ctx;
    CudaBuffer::from_tensor(&scratch.positions)
        .unwrap()
        .copy_from_host(&(pos as i32).to_le_bytes())
        .unwrap();
}

fn reset_state(ctx: &CudaContext, gdn_states: &mut [GdnState], kv_caches: &mut [KvCache]) {
    for s in gdn_states.iter() {
        zero_tensor(ctx, &s.recurrent);
        zero_tensor(ctx, &s.conv_window);
    }
    for c in kv_caches.iter() {
        zero_tensor(ctx, &c.keys);
        zero_tensor(ctx, &c.values);
    }
    ctx.synchronize().unwrap();
}

fn zero_tensor(ctx: &CudaContext, tensor: &Tensor) {
    let _ = ctx;
    let buf = CudaBuffer::from_tensor(tensor).unwrap();
    let zeros = vec![0u8; buf.len()];
    buf.copy_from_host(&zeros).unwrap();
}
