//! Layer-by-layer numerical comparison of the Qwen3.8-27B-NVFP4 CUDA port
//! against the verified PyTorch reference.
//!
//! Replays layer 0 (Gated DeltaNet) and layer 3 (full attention) on the
//! reference's own seq_len=8 input, running the port's ops one at a time in
//! compute order, and after each writes the result as a float32 `.npy` under
//! `devlocal/qwen38-nvfp4/apxinf-tensors/seq8/` with the reference's base name.
//! `scripts/compare.py` then reports cosine and relative L2 per tensor and
//! flags the first divergence.
//!
//! Deliberately unfused: `rms_norm` + `nvfp4_quantize_activation` + `gemm`
//! rather than `nvfp4_quantize_rms_norm`, and `swiglu` + separate quantize
//! rather than `nvfp4_quantize_swiglu`. The fused kernels skip a BF16 rounding
//! the reference performs; this establishes the unfused baseline.
//!
//! ```text
//! APXINF_QWEN38_CHECKPOINT=/path/to/Qwen3.8-27B-NVFP4 \
//!   bash crates/apxinf-cuda-new/test-new.sh \
//!     test -p apxinf-cuda --release --test qwen38_layer_compare \
//!     -- --ignored --nocapture --test-threads=1
//! ```

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::path::PathBuf;

use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda::{ops, CudaBuffer, CudaContext};

const HIDDEN: usize = 5120;
const INTERMEDIATE: usize = 17408;
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
const QKV_WIDTH: usize = 10240; // 16*128 q + 16*128 k + 48*128 v
const Z_WIDTH: usize = 6144;

// ---------------------------------------------------------------------------
// checkpoint + upload
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// weight structs + loaders (mirrors qwen38_end_to_end.rs)
// ---------------------------------------------------------------------------

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
    let packed = upload(ctx, &weight, vec![2 * INTERMEDIATE, HIDDEN / 2], DType::E2M1Pair);

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

struct Fp8Weight {
    weight: Tensor,
    input_scale: f32,
    alpha: f32,
}

fn load_fp8(
    ctx: &CudaContext,
    tensors: &HashMap<String, Tensor>,
    prefix: &str,
    n: usize,
    k: usize,
) -> Fp8Weight {
    let weight_scale = scalar(tensors, &format!("{prefix}.weight_scale"));
    let input_scale = scalar(tensors, &format!("{prefix}.input_scale"));
    Fp8Weight {
        weight: upload(
            ctx,
            cpu_bytes(&tensors[&format!("{prefix}.weight")]),
            vec![n, k],
            DType::F8E4M3,
        ),
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
/// `[cols, rows]`, for the two GDN projections that reach a GEMM through
/// `GemmArgs::new` (b = [K, N]).
fn load_bf16_transposed(
    ctx: &CudaContext,
    tensors: &HashMap<String, Tensor>,
    name: &str,
    rows: usize,
    cols: usize,
) -> Tensor {
    let element = DType::BF16.size_in_bytes();
    let source = cpu_bytes(&tensors[name]);
    assert_eq!(source.len(), rows * cols * element);
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

fn load_gdn_layer(ctx: &CudaContext, tensors: &HashMap<String, Tensor>, layer: usize) -> GdnLayer {
    let prefix = format!("model.language_model.layers.{layer}");
    GdnLayer {
        input_norm: load_bf16(ctx, tensors, &format!("{prefix}.input_layernorm.weight"), vec![HIDDEN]),
        post_norm: load_bf16(ctx, tensors, &format!("{prefix}.post_attention_layernorm.weight"), vec![HIDDEN]),
        qkv: load_fp8(ctx, tensors, &format!("{prefix}.linear_attn.in_proj_qkv"), QKV_WIDTH, HIDDEN),
        z: load_fp8(ctx, tensors, &format!("{prefix}.linear_attn.in_proj_z"), Z_WIDTH, HIDDEN),
        out: load_fp8(ctx, tensors, &format!("{prefix}.linear_attn.out_proj"), HIDDEN, Z_WIDTH),
        in_proj_a: load_bf16_transposed(ctx, tensors, &format!("{prefix}.linear_attn.in_proj_a.weight"), GDN_V_HEADS, HIDDEN),
        in_proj_b: load_bf16_transposed(ctx, tensors, &format!("{prefix}.linear_attn.in_proj_b.weight"), GDN_V_HEADS, HIDDEN),
        a_log: load_bf16(ctx, tensors, &format!("{prefix}.linear_attn.A_log"), vec![GDN_V_HEADS]),
        dt_bias: load_bf16(ctx, tensors, &format!("{prefix}.linear_attn.dt_bias"), vec![GDN_V_HEADS]),
        conv_weight: load_bf16(ctx, tensors, &format!("{prefix}.linear_attn.conv1d.weight"), vec![QKV_WIDTH, CONV_WIDTH]),
        norm_weight: load_bf16(ctx, tensors, &format!("{prefix}.linear_attn.norm.weight"), vec![GDN_HEAD_DIM]),
        gate_up: load_fused_gate_up(ctx, tensors, layer),
        down: load_nvfp4(ctx, tensors, &format!("{prefix}.mlp.down_proj"), HIDDEN, INTERMEDIATE),
    }
}

fn load_attention_layer(
    ctx: &CudaContext,
    tensors: &HashMap<String, Tensor>,
    layer: usize,
) -> AttentionLayer {
    let prefix = format!("model.language_model.layers.{layer}");
    AttentionLayer {
        input_norm: load_bf16(ctx, tensors, &format!("{prefix}.input_layernorm.weight"), vec![HIDDEN]),
        post_norm: load_bf16(ctx, tensors, &format!("{prefix}.post_attention_layernorm.weight"), vec![HIDDEN]),
        q: load_fp8(ctx, tensors, &format!("{prefix}.self_attn.q_proj"), 2 * HEADS * HEAD_DIM, HIDDEN),
        k: load_fp8(ctx, tensors, &format!("{prefix}.self_attn.k_proj"), KV_HEADS * HEAD_DIM, HIDDEN),
        v: load_fp8(ctx, tensors, &format!("{prefix}.self_attn.v_proj"), KV_HEADS * HEAD_DIM, HIDDEN),
        o: load_fp8(ctx, tensors, &format!("{prefix}.self_attn.o_proj"), HIDDEN, HEADS * HEAD_DIM),
        q_norm: load_bf16(ctx, tensors, &format!("{prefix}.self_attn.q_norm.weight"), vec![HEAD_DIM]),
        k_norm: load_bf16(ctx, tensors, &format!("{prefix}.self_attn.k_norm.weight"), vec![HEAD_DIM]),
        gate_up: load_fused_gate_up(ctx, tensors, layer),
        down: load_nvfp4(ctx, tensors, &format!("{prefix}.mlp.down_proj"), HIDDEN, INTERMEDIATE),
    }
}

// ---------------------------------------------------------------------------
// views / helpers
// ---------------------------------------------------------------------------

fn view(tensor: &Tensor, dims: Vec<usize>, dtype: DType) -> Tensor {
    CudaBuffer::from_tensor(tensor)
        .unwrap()
        .as_tensor(Shape::new(dims), dtype)
        .unwrap()
}

fn capacity_of(cache: &Tensor) -> usize {
    cache.shape().dims()[1]
}

fn flatten(tensor: &Tensor, width: usize) -> Tensor {
    view(tensor, vec![1, width], DType::BF16)
}

fn gdn_z_heads(z: &Tensor) -> Tensor {
    view(z, vec![GDN_V_HEADS, GDN_HEAD_DIM], DType::BF16)
}

fn split_gdn_qkv(qkv: &Tensor) -> (Tensor, Tensor, Tensor) {
    let buffer = CudaBuffer::from_tensor(qkv).unwrap();
    let element = DType::BF16.size_in_bytes();
    let q_len = GDN_K_HEADS * GDN_HEAD_DIM;
    let v_len = GDN_V_HEADS * GDN_HEAD_DIM;
    let q = buffer.view(0, q_len * element).unwrap()
        .as_tensor(Shape::new(vec![GDN_K_HEADS, GDN_HEAD_DIM]), DType::BF16).unwrap();
    let k = buffer.view(q_len * element, q_len * element).unwrap()
        .as_tensor(Shape::new(vec![GDN_K_HEADS, GDN_HEAD_DIM]), DType::BF16).unwrap();
    let v = buffer.view(2 * q_len * element, v_len * element).unwrap()
        .as_tensor(Shape::new(vec![GDN_V_HEADS, GDN_HEAD_DIM]), DType::BF16).unwrap();
    (q, k, v)
}

fn bf16_matvec(ctx: &CudaContext, weight: &Tensor, input: &Tensor, output: &Tensor) {
    let dims = weight.shape().dims().to_vec();
    let mut out = view(output, vec![1, dims[1]], DType::BF16);
    ops::gemm(ctx, ops::GemmArgs::new(input, weight, &mut out)).unwrap();
}

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

struct GdnState {
    recurrent: Tensor,
    conv_window: Tensor,
}

struct KvCache {
    keys: Tensor,
    values: Tensor,
}

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

// ---------------------------------------------------------------------------
// .npy dumping. Accumulates rows token-major, writes float32 C-contiguous npy.
// ---------------------------------------------------------------------------

struct RefDump {
    dir: std::path::PathBuf,
    // name -> (accumulated values token-major, row width)
    tensors: BTreeMap<String, (Vec<f32>, usize)>,
    // name -> full logical shape for tensors written once (e.g. state)
    once: BTreeMap<String, (Vec<f32>, Vec<usize>)>,
}

impl RefDump {
    fn new(subdir: &str) -> RefDump {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../devlocal/qwen38-nvfp4")
            .join(subdir);
        std::fs::create_dir_all(&dir).unwrap();
        RefDump { dir, tensors: BTreeMap::new(), once: BTreeMap::new() }
    }

    fn read_bf16(ctx: &CudaContext, tensor: &Tensor, count: usize) -> Vec<f32> {
        ctx.synchronize().unwrap();
        let mut bytes = vec![0u8; count * 2];
        CudaBuffer::from_tensor(tensor).unwrap().copy_to_host(&mut bytes).unwrap();
        bytes.chunks_exact(2)
            .map(|p| half::bf16::from_bits(u16::from_le_bytes([p[0], p[1]])).to_f32())
            .collect()
    }

    fn read_f32(ctx: &CudaContext, tensor: &Tensor, count: usize) -> Vec<f32> {
        ctx.synchronize().unwrap();
        let mut bytes = vec![0u8; count * 4];
        CudaBuffer::from_tensor(tensor).unwrap().copy_to_host(&mut bytes).unwrap();
        bytes.chunks_exact(4)
            .map(|w| f32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect()
    }

    /// Append one token's row of width `width`.
    fn push_bf16(&mut self, ctx: &CudaContext, name: &str, tensor: &Tensor, width: usize) {
        let v = Self::read_bf16(ctx, tensor, width);
        let e = self.tensors.entry(name.to_string()).or_insert_with(|| (Vec::new(), width));
        assert_eq!(e.1, width, "{name} width changed between tokens");
        e.0.extend(v);
    }

    fn push_f32(&mut self, ctx: &CudaContext, name: &str, tensor: &Tensor, width: usize) {
        let v = Self::read_f32(ctx, tensor, width);
        let e = self.tensors.entry(name.to_string()).or_insert_with(|| (Vec::new(), width));
        assert_eq!(e.1, width, "{name} width changed between tokens");
        e.0.extend(v);
    }

    /// Store a full tensor once with an explicit shape (recurrent state).
    fn set_f32(&mut self, ctx: &CudaContext, name: &str, tensor: &Tensor, shape: Vec<usize>) {
        let count: usize = shape.iter().product();
        let v = Self::read_f32(ctx, tensor, count);
        self.once.insert(name.to_string(), (v, shape));
    }

    fn write_npy(path: &std::path::Path, values: &[f32], shape: &[usize]) {
        let shape_str = if shape.len() == 1 {
            format!("({},)", shape[0])
        } else {
            let parts: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
            format!("({})", parts.join(", "))
        };
        let header = format!(
            "{{'descr': '<f4', 'fortran_order': False, 'shape': {shape_str}, }}"
        );
        // magic(6) + version(2) + hlen(2) = 10 bytes preamble; pad header so
        // total preamble+header is a multiple of 64, ending in '\n'.
        let mut hbytes = header.into_bytes();
        let total = 10 + hbytes.len() + 1;
        let pad = (64 - (total % 64)) % 64;
        hbytes.extend(std::iter::repeat(b' ').take(pad));
        hbytes.push(b'\n');
        let mut out = Vec::with_capacity(10 + hbytes.len() + values.len() * 4);
        out.extend_from_slice(b"\x93NUMPY");
        out.push(1);
        out.push(0);
        out.extend_from_slice(&(hbytes.len() as u16).to_le_bytes());
        out.extend_from_slice(&hbytes);
        for x in values {
            out.extend_from_slice(&x.to_le_bytes());
        }
        std::fs::write(path, out).unwrap();
    }

    fn write(&self, seq: usize) {
        let mut n = 0;
        for (name, (values, width)) in &self.tensors {
            let rows = values.len() / width;
            assert_eq!(rows, seq, "{name} has {rows} rows, expected {seq}");
            let path = self.dir.join(format!("{name}.npy"));
            Self::write_npy(&path, values, &[seq, *width]);
            n += 1;
        }
        for (name, (values, shape)) in &self.once {
            let path = self.dir.join(format!("{name}.npy"));
            Self::write_npy(&path, values, shape);
            n += 1;
        }
        println!("wrote {} npy tensors -> {}", n, self.dir.display());
    }
}

// ---------------------------------------------------------------------------
// unfused MLP block, with taps after every reference stage.
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn unfused_mlp(
    ctx: &CudaContext,
    gate_up: &Nvfp4Weight,
    down: &Nvfp4Weight,
    norm_weight: &Tensor,
    scratch: &mut Scratch,
    swiglu_out: &Tensor,
    dump: &mut RefDump,
    names: (&str, &str, &str, &str, &str, &str),
) {
    ops::rms_norm(ctx, &scratch.hidden, norm_weight, &scratch.normalized, EPSILON).unwrap();
    dump.push_bf16(ctx, names.0, &scratch.normalized, HIDDEN);
    ops::nvfp4_quantize_activation(
        ctx, &scratch.normalized, &scratch.nvfp4_activation, &scratch.nvfp4_scales,
        gate_up.input_scale, BLOCK, ops::ScaleLayout::GemmAtom,
    ).unwrap();
    ops::gemm(ctx, ops::GemmArgs::nvfp4(
        &scratch.nvfp4_activation, &scratch.nvfp4_scales,
        &gate_up.packed, &gate_up.scales, BLOCK, gate_up.alpha, &mut scratch.mlp_fused,
    )).unwrap();
    // Fused block is [gate | up]; write the two halves under their own names so
    // compare.py needs no slicing.
    let fused = RefDump::read_bf16(ctx, &scratch.mlp_fused, 2 * INTERMEDIATE);
    {
        let gate = &fused[..INTERMEDIATE];
        let up = &fused[INTERMEDIATE..];
        let g = dump.tensors.entry(names.1.to_string()).or_insert_with(|| (Vec::new(), INTERMEDIATE));
        g.0.extend_from_slice(gate);
        let u = dump.tensors.entry(names.2.to_string()).or_insert_with(|| (Vec::new(), INTERMEDIATE));
        u.0.extend_from_slice(up);
    }

    ops::swiglu(ctx, &scratch.mlp_fused, swiglu_out).unwrap();
    dump.push_bf16(ctx, names.3, swiglu_out, INTERMEDIATE);
    ops::nvfp4_quantize_activation(
        ctx, swiglu_out, &scratch.mlp_activation, &scratch.mlp_scales,
        down.input_scale, BLOCK, ops::ScaleLayout::GemmAtom,
    ).unwrap();
    ops::gemm(ctx, ops::GemmArgs::nvfp4(
        &scratch.mlp_activation, &scratch.mlp_scales,
        &down.packed, &down.scales, BLOCK, down.alpha, &mut scratch.mlp_out,
    )).unwrap();
    dump.push_bf16(ctx, names.4, &scratch.mlp_out, HIDDEN);
    ops::add_into(ctx, &scratch.mlp_out, &scratch.hidden).unwrap();
    dump.push_bf16(ctx, names.5, &scratch.hidden, HIDDEN);
}

// ---------------------------------------------------------------------------
// scratch
// ---------------------------------------------------------------------------

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
}

impl Scratch {
    fn new(ctx: &CudaContext) -> Scratch {
        let scale_bytes = |rows: usize, k: usize| vec![ops::nvfp4_scale_buffer_bytes(rows, k, BLOCK).unwrap()];
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
        }
    }
}

// ---------------------------------------------------------------------------
// per-layer reference replay
// ---------------------------------------------------------------------------

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
    ops::gdn_causal_conv_step(ctx, &state.conv_window, &qkv_flat, &gdn.conv_weight, &scratch.gdn_conv).unwrap();
    // conv kernel folds SiLU, so only the post-activation tap exists.
    dump.push_bf16(ctx, "l0_07_conv1d_silu", &scratch.gdn_conv, QKV_WIDTH);

    let (q, k, v) = split_gdn_qkv(&scratch.gdn_conv);
    dump.push_bf16(ctx, "l0_08_q_split", &q, GDN_K_HEADS * GDN_HEAD_DIM);
    dump.push_bf16(ctx, "l0_08_k_split", &k, GDN_K_HEADS * GDN_HEAD_DIM);
    dump.push_bf16(ctx, "l0_08_v_split", &v, GDN_V_HEADS * GDN_HEAD_DIM);

    ops::gdn_l2_normalize_heads(ctx, &q, EPSILON).unwrap();
    ops::gdn_l2_normalize_heads(ctx, &k, EPSILON).unwrap();
    dump.push_bf16(ctx, "l0_12_q_l2norm", &q, GDN_K_HEADS * GDN_HEAD_DIM);
    dump.push_bf16(ctx, "l0_12_k_l2norm", &k, GDN_K_HEADS * GDN_HEAD_DIM);

    ops::gdn_decay_and_beta(ctx, &scratch.gdn_a, &scratch.gdn_b, &gdn.a_log, &gdn.dt_bias, &scratch.gdn_decay, &scratch.gdn_beta).unwrap();
    dump.push_f32(ctx, "l0_10_g_log_decay", &scratch.gdn_decay, GDN_V_HEADS);
    dump.push_f32(ctx, "l0_09_beta", &scratch.gdn_beta, GDN_V_HEADS);

    ops::gdn_recurrent_step(ctx, &state.recurrent, &q, &k, &v, &scratch.gdn_decay, &scratch.gdn_beta, &scratch.gdn_readout, GDN_K_HEADS).unwrap();
    dump.push_bf16(ctx, "l0_16_core_attn_out", &scratch.gdn_readout, GDN_V_HEADS * GDN_HEAD_DIM);
    // Port stores state [v_heads, v_dim, k_dim]; reference is [v_heads, k_dim, v_dim].
    // Stored raw here in the port's layout; compare.py transposes the reference.
    dump.set_f32(ctx, "l0_15_recurrent_state_final", &state.recurrent,
        vec![GDN_V_HEADS, GDN_HEAD_DIM, GDN_HEAD_DIM]);

    ops::gdn_gated_norm(ctx, &scratch.gdn_readout, &gdn_z_heads(&scratch.gdn_z), &gdn.norm_weight, &scratch.gdn_gated, EPSILON).unwrap();
    dump.push_bf16(ctx, "l0_17_gated_rmsnorm", &scratch.gdn_gated, Z_WIDTH);

    let flat = flatten(&scratch.gdn_gated, Z_WIDTH);
    fp8_projection(ctx, &gdn.out, &flat, &scratch.gdn_fp8, &mut scratch.projected);
    dump.push_bf16(ctx, "l0_18_out_proj", &scratch.projected, HIDDEN);
    ops::add_into(ctx, &scratch.projected, &scratch.hidden).unwrap();
    dump.push_bf16(ctx, "l0_19_residual_1", &scratch.hidden, HIDDEN);

    unfused_mlp(ctx, &gdn.gate_up, &gdn.down, &gdn.post_norm, scratch, swiglu_out, dump, (
        "l0_20_post_attention_layernorm",
        "l0_21_gate_proj",
        "l0_21_up_proj",
        "l0_22_swiglu",
        "l0_23_down_proj",
        "l0_24_layer_out",
    ));
}

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
    dump.push_bf16(ctx, "l3_06_v_states", &value_slot, KV_HEADS * HEAD_DIM);

    let query_tokens = view(&scratch.query, vec![1, HEADS, HEAD_DIM], DType::BF16);
    let key_tokens = cache_slot(&cache.keys, position, vec![1, KV_HEADS, HEAD_DIM]);
    ops::partial_rope(ctx, &query_tokens, &scratch.positions, rotary, ROPE_THETA).unwrap();
    ops::partial_rope(ctx, &key_tokens, &scratch.positions, rotary, ROPE_THETA).unwrap();
    dump.push_bf16(ctx, "l3_07_q_rope", &scratch.query, HEADS * HEAD_DIM);
    dump.push_bf16(ctx, "l3_07_k_rope", &key_slot, KV_HEADS * HEAD_DIM);

    let valid = position + 1;
    let keys = view(&cache.keys, vec![1, capacity_of(&cache.keys), KV_HEADS, HEAD_DIM], DType::BF16);
    let values = view(&cache.values, vec![1, capacity_of(&cache.values), KV_HEADS, HEAD_DIM], DType::BF16);
    let query_4d = view(&scratch.query, vec![1, 1, HEADS, HEAD_DIM], DType::BF16);
    let mut out_4d = view(&scratch.attention_out, vec![1, 1, HEADS, HEAD_DIM], DType::BF16);
    let mut args = ops::KvCacheAttentionArgs::new(&query_4d, &keys, &values, &mut out_4d);
    args.valid_key_tokens = valid;
    args.query_start = position;
    ops::kv_cache_attention(ctx, args).unwrap();
    dump.push_bf16(ctx, "l3_09_attn_out", &scratch.attention_out, HEADS * HEAD_DIM);

    let gate_flat = view(&scratch.query_gate, vec![1, HEADS * HEAD_DIM], DType::BF16);
    ops::apply_output_gate(ctx, &scratch.attention_out, &gate_flat).unwrap();
    dump.push_bf16(ctx, "l3_10_attn_gated", &scratch.attention_out, HEADS * HEAD_DIM);

    fp8_projection(ctx, &attention.o, &scratch.attention_out, &scratch.attention_fp8, &mut scratch.projected);
    dump.push_bf16(ctx, "l3_11_o_proj", &scratch.projected, HIDDEN);
    ops::add_into(ctx, &scratch.projected, &scratch.hidden).unwrap();
    dump.push_bf16(ctx, "l3_12_residual_1", &scratch.hidden, HIDDEN);

    unfused_mlp(ctx, &attention.gate_up, &attention.down, &attention.post_norm, scratch, swiglu_out, dump, (
        "l3_13_post_attention_layernorm",
        "l3_14_gate_proj",
        "l3_14_up_proj",
        "l3_15_swiglu",
        "l3_16_down_proj",
        "l3_17_layer_out",
    ));
}

// ---------------------------------------------------------------------------
// input
// ---------------------------------------------------------------------------

fn reference_sequence_length() -> usize {
    std::env::var("APXINF_REF_SEQ").ok().and_then(|v| v.parse().ok()).unwrap_or(8)
}

/// hidden[0, t, i] = bf16(sin((t * HIDDEN + i) * 0.01)), sine in f64.
fn reference_hidden(seq_len: usize) -> Vec<half::bf16> {
    (0..seq_len * HIDDEN)
        .map(|i| half::bf16::from_f64((i as f64 * 0.01).sin()))
        .collect()
}

// ---------------------------------------------------------------------------
// the test
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires the Qwen3.8-27B-NVFP4 checkpoint and a GPU; pairs with scripts/compare.py"]
fn qwen38_layer_compare() {
    let ctx = CudaContext::new(0).unwrap();
    let seq_len = reference_sequence_length();
    let tensors = checkpoint();
    let gdn = load_gdn_layer(&ctx, &tensors, 0);
    let attention = load_attention_layer(&ctx, &tensors, 3);
    ctx.synchronize().unwrap();
    drop(tensors);

    let subdir = if seq_len == 8 { "apxinf-tensors/seq8".to_string() }
                 else { format!("apxinf-tensors/seq{seq_len}") };
    let mut dump = RefDump::new(&subdir);
    let hidden_input = reference_hidden(seq_len);
    let mut scratch = Scratch::new(&ctx);
    let swiglu_out = zeros(&ctx, vec![1, INTERMEDIATE], DType::BF16);
    let rotary = ops::rotary_dim(HEAD_DIM, PARTIAL_ROTARY);

    let mut gdn_state = GdnState {
        recurrent: zeros(&ctx, vec![GDN_V_HEADS, GDN_HEAD_DIM, GDN_HEAD_DIM], DType::F32),
        conv_window: zeros(&ctx, vec![QKV_WIDTH, CONV_WIDTH], DType::F32),
    };
    let capacity = seq_len.max(1);
    let mut cache = KvCache {
        keys: zeros(&ctx, vec![1, capacity, KV_HEADS, HEAD_DIM], DType::BF16),
        values: zeros(&ctx, vec![1, capacity, KV_HEADS, HEAD_DIM], DType::BF16),
    };

    // Each token is re-seeded straight from the reference input, so no step
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
        gdn_reference_step(&ctx, &gdn, &mut scratch, &mut gdn_state, &swiglu_out, &mut dump);
        ctx.synchronize().unwrap();
    }

    for token in 0..seq_len {
        load_token(&scratch, token);
        CudaBuffer::from_tensor(&scratch.positions).unwrap()
            .copy_from_host(&(token as i32).to_le_bytes()).unwrap();
        attention_reference_step(&ctx, &attention, &mut scratch, &mut cache, &swiglu_out, rotary, token, &mut dump);
        ctx.synchronize().unwrap();
    }

    dump.write(seq_len);
    println!("QWEN38 LAYER COMPARE DUMP COMPLETE seq_len={seq_len}");
}
