//! Native one-token Qwen3.5/Qwen3.8 hybrid decoder primitives.
//!
//! This module owns the first production-shaped scheduling boundary: reusable
//! weights and workspaces for the repeating GDN,GDN,GDN,full-attention unit.
//! It intentionally does not own tokenizer, embeddings, LM head, prefill, or
//! serving policy yet.

use apxinf_core::{DType, Device, Error, Result, Tensor};
use apxinf_cuda::kernels::gdn::{
    qwen35_conv4_prepare_m8_write, qwen35_conv4_prepare_write, qwen35_gated_rmsnorm_m8_write,
    qwen35_gated_rmsnorm_write, qwen35_recurrent_m8_hybrid_write, qwen35_recurrent_m8_write,
    qwen35_recurrent_write, QWEN35_GDN_CONV_DIM as GDN_CONV_DIM,
    QWEN35_GDN_CONV_KERNEL as GDN_CONV_KERNEL,
    QWEN35_GDN_HEADS as GDN_HEADS, QWEN35_GDN_KEY_DIM as GDN_DIM,
};
use apxinf_cuda::kernels::gemm::{
    self, MarlinPreparedWeight, MarlinWorkspace, W4A16Layout, W4A16WeightView, W8A16WeightView,
};
use apxinf_cuda::kernels::{
    activation, attention, cache, qwen35_attention, qwen35_common, GraphWorkspace,
};
use apxinf_cuda::{
    transfers, CublasTranspose, CudaBuffer, CudaContext, CudaDeviceAddress, HostMappedBuffer,
};
use apxinf_loader::safetensors::{self, CheckpointManifest};

const HIDDEN: usize = 5120;
const INTERMEDIATE: usize = 17408;
const GDN_VALUE_WIDTH: usize = GDN_HEADS * GDN_DIM;
const ATTN_Q_HEADS: usize = 24;
const ATTN_KV_HEADS: usize = 4;
const ATTN_HEAD_DIM: usize = 256;
const ATTN_WIDTH: usize = ATTN_Q_HEADS * ATTN_HEAD_DIM;
const ATTN_KV_WIDTH: usize = ATTN_KV_HEADS * ATTN_HEAD_DIM;
const RMS_EPSILON: f32 = 1.0e-6;
const ATTENTION_SCALE: f32 = 1.0 / 16.0;
const PREFILL_TILE: usize = 8;
const MARLIN_PREFILL_TILE: usize = 64;
const MARLIN_PREFILL_SUBTILES: usize = MARLIN_PREFILL_TILE / PREFILL_TILE;
const LAYER_MAJOR_PREFILL_ROWS: usize = 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HybridUnitMode {
    Native,
    LayerOptimized,
    ModelOptimized,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen35PrefillMode {
    M8,
    MarlinM64,
}

impl Qwen35PrefillMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::M8 => "m8",
            Self::MarlinM64 => "marlin-m64",
        }
    }
}

impl HybridUnitMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::LayerOptimized => "layer_optimized",
            Self::ModelOptimized => "model_optimized",
        }
    }
}

struct W4Weight {
    packed: Tensor,
    scales: Tensor,
    zero_points: Tensor,
    input_dim: usize,
    output_dim: usize,
}

impl W4Weight {
    fn load(manifest: &CheckpointManifest, base: &str) -> Result<Self> {
        let (packed, scales, zero_points, input_dim, output_dim) = load_w4_cpu(manifest, base)?;
        Self::from_cpu(packed, scales, zero_points, input_dim, output_dim)
    }

    fn load_row_pair(
        manifest: &CheckpointManifest,
        first_base: &str,
        second_base: &str,
    ) -> Result<Self> {
        let (first_packed, first_scales, first_zero, input_dim, output_dim) =
            load_w4_cpu(manifest, first_base)?;
        let (second_packed, second_scales, second_zero, second_input, second_output) =
            load_w4_cpu(manifest, second_base)?;
        if second_input != input_dim || second_output != output_dim || output_dim % 8 != 0 {
            return Err(Error::Other(format!(
                "cannot concatenate W4 rows `{first_base}` and `{second_base}`"
            )));
        }
        let mut packed = first_packed.as_i32()?.to_vec();
        packed.extend_from_slice(second_packed.as_i32()?);
        let mut scales = first_scales.as_bf16()?.to_vec();
        scales.extend_from_slice(second_scales.as_bf16()?);
        let mut zero = first_zero.as_i32()?.to_vec();
        zero.extend_from_slice(second_zero.as_i32()?);
        Self::from_cpu(
            Tensor::from_i32(vec![2 * output_dim, input_dim / 8], &packed)?,
            Tensor::from_bf16(vec![2 * output_dim, input_dim / 32], &scales)?,
            Tensor::from_i32(vec![2 * output_dim / 8, input_dim / 32], &zero)?,
            input_dim,
            2 * output_dim,
        )
    }

    fn from_cpu(
        packed: Tensor,
        scales: Tensor,
        zero_points: Tensor,
        input_dim: usize,
        output_dim: usize,
    ) -> Result<Self> {
        Ok(Self {
            packed: transfers::to_cuda(&packed, 0)?,
            scales: transfers::to_cuda(&scales, 0)?,
            zero_points: transfers::to_cuda(&zero_points, 0)?,
            input_dim,
            output_dim,
        })
    }

    fn view(&self) -> W4A16WeightView<'_> {
        W4A16WeightView {
            packed_i32: &self.packed,
            scales_bf16: &self.scales,
            zero_points_i32: &self.zero_points,
            input_dim: self.input_dim,
            output_dim: self.output_dim,
            group_size: 32,
            layout: W4A16Layout::CompressedTensorsPackQuantized,
        }
    }
}

struct W8Weight {
    values: CudaBuffer,
    scales: Tensor,
    input_dim: usize,
    output_dim: usize,
}

impl W8Weight {
    fn from_bf16(weight: &Tensor) -> Result<Self> {
        if weight.device() != Device::Cpu
            || weight.dtype() != DType::BF16
            || weight.shape().dims().len() != 2
        {
            return Err(Error::Other(
                "Qwen3.5 W8 conversion requires a CPU BF16 matrix".into(),
            ));
        }
        let output_dim = weight.shape().dims()[0];
        let input_dim = weight.shape().dims()[1];
        let source = weight.as_bf16()?;
        let mut values = vec![0_u8; output_dim * input_dim];
        let mut scales = vec![0.0_f32; output_dim];
        for row in 0..output_dim {
            let row_values = &source[row * input_dim..(row + 1) * input_dim];
            let maximum = row_values
                .iter()
                .map(|value| value.to_f32().abs())
                .fold(0.0_f32, f32::max);
            let scale = (maximum / 127.0).max(1.0e-12);
            scales[row] = scale;
            for (column, value) in row_values.iter().enumerate() {
                values[row * input_dim + column] =
                    (value.to_f32() / scale).round().clamp(-128.0, 127.0) as i8 as u8;
            }
        }
        let values_gpu = CudaBuffer::alloc(values.len(), 0).map_err(Error::Cuda)?;
        values_gpu.copy_from_host(&values).map_err(Error::Cuda)?;
        Ok(Self {
            values: values_gpu,
            scales: transfers::to_cuda(&Tensor::from_f32(vec![output_dim], &scales)?, 0)?,
            input_dim,
            output_dim,
        })
    }

    fn view(&self) -> W8A16WeightView<'_> {
        W8A16WeightView {
            values_i8: &self.values,
            scales_f32: &self.scales,
            input_dim: self.input_dim,
            output_dim: self.output_dim,
        }
    }

    fn from_bf16_manifest(entry: &apxinf_loader::safetensors::TensorManifestEntry) -> Result<Self> {
        if entry.dtype != DType::BF16 || entry.shape.len() != 2 {
            return Err(Error::Other(format!(
                "Qwen3.5 streaming W8 conversion requires a BF16 matrix, got `{}` {} {:?}",
                entry.name, entry.dtype, entry.shape
            )));
        }
        let output_dim = entry.shape[0];
        let input_dim = entry.shape[1];
        if output_dim % 8 != 0 || input_dim % 8 != 0 {
            return Err(Error::Other(format!(
                "Qwen3.5 streaming W8 shape [{output_dim},{input_dim}] is unsupported"
            )));
        }
        let mut values = vec![0_u8; output_dim * input_dim];
        let mut scales = vec![0.0_f32; output_dim];
        const ROWS_PER_CHUNK: usize = 128;
        for first_row in (0..output_dim).step_by(ROWS_PER_CHUNK) {
            let row_count = ROWS_PER_CHUNK.min(output_dim - first_row);
            let chunk = safetensors::load_manifest_tensor_rows(entry, first_row, row_count)
                .map_err(Error::Other)?;
            let chunk = chunk.as_bf16()?;
            for local_row in 0..row_count {
                let row = first_row + local_row;
                let row_values = &chunk[local_row * input_dim..(local_row + 1) * input_dim];
                let maximum = row_values
                    .iter()
                    .map(|value| value.to_f32().abs())
                    .fold(0.0_f32, f32::max);
                let scale = (maximum / 127.0).max(1.0e-12);
                scales[row] = scale;
                for (column, value) in row_values.iter().enumerate() {
                    values[row * input_dim + column] =
                        (value.to_f32() / scale).round().clamp(-128.0, 127.0) as i8 as u8;
                }
            }
        }
        let values_gpu = CudaBuffer::alloc(values.len(), 0).map_err(Error::Cuda)?;
        values_gpu.copy_from_host(&values).map_err(Error::Cuda)?;
        Ok(Self {
            values: values_gpu,
            scales: transfers::to_cuda(&Tensor::from_f32(vec![output_dim], &scales)?, 0)?,
            input_dim,
            output_dim,
        })
    }
}

struct MlpWeights {
    gate_up: W4Weight,
    down: W4Weight,
}

impl MlpWeights {
    fn load(manifest: &CheckpointManifest, layer: usize) -> Result<Self> {
        let prefix = format!("model.language_model.layers.{layer}.mlp");
        Ok(Self {
            gate_up: W4Weight::load_row_pair(
                manifest,
                &format!("{prefix}.gate_proj"),
                &format!("{prefix}.up_proj"),
            )?,
            down: W4Weight::load(manifest, &format!("{prefix}.down_proj"))?,
        })
    }
}

struct GdnWeights {
    qkv: W4Weight,
    z: W4Weight,
    ab: Tensor,
    conv: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
    norm: Tensor,
    out: GdnOutputWeight,
}

enum GdnOutputWeight {
    Bf16 { weight: Tensor, w8: W8Weight },
    W4(W4Weight),
}

impl GdnWeights {
    fn load(manifest: &CheckpointManifest, layer: usize) -> Result<Self> {
        let prefix = format!("model.language_model.layers.{layer}.linear_attn");
        let a = load_tensor(manifest, &format!("{prefix}.in_proj_a.weight"))?;
        let b = load_tensor(manifest, &format!("{prefix}.in_proj_b.weight"))?;
        require_cpu_bf16_shape("GDN a", &a, &[GDN_HEADS, HIDDEN])?;
        require_cpu_bf16_shape("GDN b", &b, &[GDN_HEADS, HIDDEN])?;
        let mut ab = Vec::with_capacity(2 * GDN_HEADS * HIDDEN);
        ab.extend_from_slice(a.as_bf16()?);
        ab.extend_from_slice(b.as_bf16()?);
        let out_base = format!("{prefix}.out_proj");
        let out = if manifest.tensor(&format!("{out_base}.weight")).is_some() {
            let weight = load_tensor(manifest, &format!("{out_base}.weight"))?;
            require_cpu_bf16_shape("GDN out", &weight, &[HIDDEN, GDN_VALUE_WIDTH])?;
            GdnOutputWeight::Bf16 {
                weight: transfers::to_cuda(&weight, 0)?,
                w8: W8Weight::from_bf16(&weight)?,
            }
        } else {
            GdnOutputWeight::W4(W4Weight::load(manifest, &out_base)?)
        };
        Ok(Self {
            qkv: W4Weight::load(manifest, &format!("{prefix}.in_proj_qkv"))?,
            z: W4Weight::load(manifest, &format!("{prefix}.in_proj_z"))?,
            ab: transfers::to_cuda(&Tensor::from_bf16(vec![2 * GDN_HEADS, HIDDEN], &ab)?, 0)?,
            conv: load_gpu_bf16(manifest, &format!("{prefix}.conv1d.weight"))?,
            a_log: load_gpu_bf16(manifest, &format!("{prefix}.A_log"))?,
            dt_bias: load_gpu_bf16(manifest, &format!("{prefix}.dt_bias"))?,
            norm: load_gpu_bf16(manifest, &format!("{prefix}.norm.weight"))?,
            out,
        })
    }
}

struct AttentionWeights {
    q: W4Weight,
    k: W4Weight,
    v: W4Weight,
    o: W4Weight,
    q_norm: Tensor,
    k_norm: Tensor,
}

impl AttentionWeights {
    fn load(manifest: &CheckpointManifest, layer: usize) -> Result<Self> {
        let prefix = format!("model.language_model.layers.{layer}.self_attn");
        Ok(Self {
            q: W4Weight::load(manifest, &format!("{prefix}.q_proj"))?,
            k: W4Weight::load(manifest, &format!("{prefix}.k_proj"))?,
            v: W4Weight::load(manifest, &format!("{prefix}.v_proj"))?,
            o: W4Weight::load(manifest, &format!("{prefix}.o_proj"))?,
            q_norm: load_gpu_bf16(manifest, &format!("{prefix}.q_norm.weight"))?,
            k_norm: load_gpu_bf16(manifest, &format!("{prefix}.k_norm.weight"))?,
        })
    }
}

struct GdnState {
    conv: Tensor,
    recurrent: Tensor,
}

struct AttentionState {
    key_cache: Tensor,
    value_cache: Tensor,
}

enum Mixer {
    Gdn {
        weights: GdnWeights,
        state: GdnState,
    },
    Attention {
        weights: AttentionWeights,
        state: AttentionState,
    },
}

struct DecoderLayer {
    index: usize,
    input_norm: Tensor,
    post_attention_norm: Tensor,
    mixer: Mixer,
    mlp: MlpWeights,
}

struct GdnWorkspace {
    qkv: Tensor,
    z: Tensor,
    ab: Tensor,
    a: Tensor,
    b: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    core: Tensor,
    normalized: Tensor,
}

struct AttentionWorkspace {
    q_projection: Tensor,
    k_projection: Tensor,
    v_projection: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    gate: Tensor,
    attended: Tensor,
    gated: Tensor,
    gated_flat: Tensor,
    split: qwen35_attention::SplitCtaWorkspace,
}

struct MlpWorkspace {
    gate_up: Tensor,
    hidden: Tensor,
}

struct GdnPrefillWorkspace {
    qkv: Tensor,
    z: Tensor,
    ab: Tensor,
    a: Tensor,
    b: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    core: Tensor,
    normalized: Tensor,
}

struct AttentionPrefillWorkspace {
    q_projection: Tensor,
    k_projection: Tensor,
    v_projection: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    gate: Tensor,
    attended: Tensor,
    gated: Tensor,
}

struct MlpPrefillWorkspace {
    gate_up: Tensor,
    hidden: Tensor,
}

struct PrefillWorkspace {
    residual: Tensor,
    normalized: Tensor,
    mlp_normalized: Tensor,
    mixer_delta: Tensor,
    mlp_delta: Tensor,
    gdn: GdnPrefillWorkspace,
    attention: AttentionPrefillWorkspace,
    mlp: MlpPrefillWorkspace,
}

struct MarlinMlpPrefillWorkspace {
    gate_up: Tensor,
    hidden: Tensor,
    gate_up_weight: MarlinPreparedWeight,
    down_weight: MarlinPreparedWeight,
    kernel: MarlinWorkspace,
}

struct LayerMajorGdnWorkspace {
    qkv: Tensor,
    z: Tensor,
    normalized: Tensor,
    qkv_weight: MarlinPreparedWeight,
    z_weight: MarlinPreparedWeight,
    out_weight: MarlinPreparedWeight,
}

struct LayerMajorAttentionWorkspace {
    q_projection: Tensor,
    k_projection: Tensor,
    v_projection: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    gate: Tensor,
    attended: Tensor,
    gated: Tensor,
    q_weight: MarlinPreparedWeight,
    k_weight: MarlinPreparedWeight,
    v_weight: MarlinPreparedWeight,
    o_weight: MarlinPreparedWeight,
}

struct LayerMajorPrefillWorkspace {
    residual: Tensor,
    normalized: Tensor,
    mixer_delta: Tensor,
    gdn: LayerMajorGdnWorkspace,
    attention: LayerMajorAttentionWorkspace,
    kernel: MarlinWorkspace,
}

struct MarlinPrefillWorkspace {
    residual: Tensor,
    normalized: Tensor,
    mlp_normalized: Tensor,
    mixer_delta: Tensor,
    mlp_delta: Tensor,
    layer_major: LayerMajorPrefillWorkspace,
    mlp: MarlinMlpPrefillWorkspace,
}

struct UnitWorkspace {
    graph: GraphWorkspace,
    residual: Tensor,
    normalized: Tensor,
    mlp_normalized: Tensor,
    mixer_delta: Tensor,
    mlp_delta: Tensor,
    gdn: GdnWorkspace,
    attention: AttentionWorkspace,
    mlp: MlpWorkspace,
}

pub struct HybridUnit {
    layers: Vec<DecoderLayer>,
    next_input_norm: Tensor,
    workspace: UnitWorkspace,
    prefill: PrefillWorkspace,
    marlin_prefill: Option<MarlinPrefillWorkspace>,
    position: HostMappedBuffer,
    rope_position: HostMappedBuffer,
    prefill_positions: HostMappedBuffer,
    prefill_rope_positions: HostMappedBuffer,
    zero_conv: Tensor,
    zero_recurrent: Tensor,
    max_seq_len: usize,
}

pub struct Qwen35LmHead {
    weight: W8Weight,
    logits: Tensor,
}

impl Qwen35LmHead {
    pub fn load(manifest: &CheckpointManifest, ctx: &CudaContext) -> Result<Self> {
        if ctx.device_id() != 0 || ctx.caps().sm != 89 {
            return Err(Error::Other(
                "Qwen3.5 W8 LM head requires CUDA0/SM89".into(),
            ));
        }
        let entry = manifest
            .tensor("lm_head.weight")
            .ok_or_else(|| Error::Other("missing `lm_head.weight`".into()))?;
        let weight = W8Weight::from_bf16_manifest(entry)?;
        if weight.input_dim != HIDDEN || weight.output_dim != 248_320 {
            return Err(Error::Other(format!(
                "Qwen3.5 LM head shape is [{},{}], expected [248320,{HIDDEN}]",
                weight.output_dim, weight.input_dim
            )));
        }
        Ok(Self {
            weight,
            logits: gpu_zeros(&[1, 248_320], DType::BF16)?,
        })
    }

    pub fn forward(&self, ctx: &CudaContext, normalized_hidden: &Tensor) -> Result<()> {
        gemm::w8a16_write(ctx, normalized_hidden, self.weight.view(), &self.logits)
    }

    pub fn logits(&self) -> &Tensor {
        &self.logits
    }

    pub fn argmax_cpu(&self) -> Result<u32> {
        let logits = transfers::to_cpu(&self.logits)?.to_f32_vec()?;
        logits
            .iter()
            .enumerate()
            .max_by(|left, right| {
                left.1
                    .partial_cmp(right.1)
                    .unwrap_or(std::cmp::Ordering::Less)
            })
            .map(|(index, _)| index as u32)
            .ok_or_else(|| Error::Other("Qwen3.5 LM head produced no logits".into()))
    }
}

pub fn load_embedding_row(manifest: &CheckpointManifest, token_id: u32) -> Result<Tensor> {
    let entry = manifest
        .tensor("model.language_model.embed_tokens.weight")
        .ok_or_else(|| Error::Other("missing embedding table".into()))?;
    if entry.dtype != DType::BF16 || entry.shape != [248_320, HIDDEN] {
        return Err(Error::Other(format!(
            "Qwen3.5 embedding table must be BF16 [248320,{HIDDEN}], got {} {:?}",
            entry.dtype, entry.shape
        )));
    }
    if token_id as usize >= entry.shape[0] {
        return Err(Error::Other(format!(
            "Qwen3.5 token id {token_id} exceeds vocabulary {}",
            entry.shape[0]
        )));
    }
    safetensors::load_manifest_tensor_rows(entry, token_id as usize, 1).map_err(Error::Other)
}

impl HybridUnit {
    pub fn load_first(
        manifest: &CheckpointManifest,
        ctx: &CudaContext,
        max_seq_len: usize,
    ) -> Result<Self> {
        Self::load_range(
            manifest,
            ctx,
            max_seq_len,
            0,
            4,
            "model.language_model.layers.4.input_layernorm.weight",
            Qwen35PrefillMode::M8,
        )
    }

    pub fn load_all(
        manifest: &CheckpointManifest,
        ctx: &CudaContext,
        max_seq_len: usize,
    ) -> Result<Self> {
        Self::load_range(
            manifest,
            ctx,
            max_seq_len,
            0,
            64,
            "model.language_model.norm.weight",
            Qwen35PrefillMode::M8,
        )
    }

    pub fn load_all_with_prefill_mode(
        manifest: &CheckpointManifest,
        ctx: &CudaContext,
        max_seq_len: usize,
        prefill_mode: Qwen35PrefillMode,
    ) -> Result<Self> {
        Self::load_range(
            manifest,
            ctx,
            max_seq_len,
            0,
            64,
            "model.language_model.norm.weight",
            prefill_mode,
        )
    }

    fn load_range(
        manifest: &CheckpointManifest,
        ctx: &CudaContext,
        max_seq_len: usize,
        first_layer: usize,
        layer_count: usize,
        next_norm_name: &str,
        prefill_mode: Qwen35PrefillMode,
    ) -> Result<Self> {
        if ctx.device_id() != 0 || ctx.caps().sm != 89 || max_seq_len == 0 {
            return Err(Error::Other(
                "Qwen3.5 hybrid stack requires CUDA0/SM89 and non-zero KV capacity".into(),
            ));
        }
        if first_layer != 0 || layer_count == 0 || layer_count > 64 {
            return Err(Error::Other(
                "Qwen3.5 hybrid stack currently requires a non-empty prefix of 64 layers".into(),
            ));
        }
        let mut layers = Vec::with_capacity(layer_count);
        for layer in first_layer..first_layer + layer_count {
            let prefix = format!("model.language_model.layers.{layer}");
            let mixer = if (layer + 1) % 4 != 0 {
                Mixer::Gdn {
                    weights: GdnWeights::load(manifest, layer)?,
                    state: GdnState {
                        conv: gpu_zeros(&[GDN_CONV_DIM, GDN_CONV_KERNEL], DType::BF16)?,
                        recurrent: gpu_zeros(&[GDN_HEADS, GDN_DIM, GDN_DIM], DType::F32)?,
                    },
                }
            } else {
                Mixer::Attention {
                    weights: AttentionWeights::load(manifest, layer)?,
                    state: AttentionState {
                        key_cache: gpu_zeros(
                            &[ATTN_KV_HEADS, max_seq_len, ATTN_HEAD_DIM],
                            DType::BF16,
                        )?,
                        value_cache: gpu_zeros(
                            &[ATTN_KV_HEADS, max_seq_len, ATTN_HEAD_DIM],
                            DType::BF16,
                        )?,
                    },
                }
            };
            layers.push(DecoderLayer {
                index: layer,
                input_norm: load_gpu_bf16(manifest, &format!("{prefix}.input_layernorm.weight"))?,
                post_attention_norm: load_gpu_bf16(
                    manifest,
                    &format!("{prefix}.post_attention_layernorm.weight"),
                )?,
                mixer,
                mlp: MlpWeights::load(manifest, layer)?,
            });
        }

        let attended = gpu_zeros(&[ATTN_Q_HEADS, ATTN_HEAD_DIM], DType::BF16)?;
        let gated = gpu_zeros(&[ATTN_Q_HEADS, ATTN_HEAD_DIM], DType::BF16)?;
        let gated_flat = gated.reshape(vec![1, ATTN_WIDTH])?;
        Ok(Self {
            layers,
            next_input_norm: load_gpu_bf16(manifest, next_norm_name)?,
            workspace: UnitWorkspace {
                graph: GraphWorkspace::new(64 * 1024, ctx.device_id())?,
                residual: gpu_zeros(&[1, HIDDEN], DType::BF16)?,
                normalized: gpu_zeros(&[1, HIDDEN], DType::BF16)?,
                mlp_normalized: gpu_zeros(&[1, HIDDEN], DType::BF16)?,
                mixer_delta: gpu_zeros(&[1, HIDDEN], DType::BF16)?,
                mlp_delta: gpu_zeros(&[1, HIDDEN], DType::BF16)?,
                gdn: GdnWorkspace {
                    qkv: gpu_zeros(&[1, GDN_CONV_DIM], DType::BF16)?,
                    z: gpu_zeros(&[1, GDN_VALUE_WIDTH], DType::BF16)?,
                    ab: gpu_zeros(&[1, 2 * GDN_HEADS], DType::BF16)?,
                    a: gpu_zeros(&[1, GDN_HEADS], DType::BF16)?,
                    b: gpu_zeros(&[1, GDN_HEADS], DType::BF16)?,
                    query: gpu_zeros(&[GDN_HEADS, GDN_DIM], DType::BF16)?,
                    key: gpu_zeros(&[GDN_HEADS, GDN_DIM], DType::BF16)?,
                    value: gpu_zeros(&[GDN_HEADS, GDN_DIM], DType::BF16)?,
                    g: gpu_zeros(&[GDN_HEADS], DType::F32)?,
                    beta: gpu_zeros(&[GDN_HEADS], DType::F32)?,
                    core: gpu_zeros(&[GDN_HEADS, GDN_DIM], DType::BF16)?,
                    normalized: gpu_zeros(&[GDN_HEADS, GDN_DIM], DType::BF16)?,
                },
                attention: AttentionWorkspace {
                    q_projection: gpu_zeros(&[1, 2 * ATTN_WIDTH], DType::BF16)?,
                    k_projection: gpu_zeros(&[1, ATTN_KV_WIDTH], DType::BF16)?,
                    v_projection: gpu_zeros(&[1, ATTN_KV_WIDTH], DType::BF16)?,
                    query: gpu_zeros(&[ATTN_Q_HEADS, ATTN_HEAD_DIM], DType::BF16)?,
                    key: gpu_zeros(&[ATTN_KV_HEADS, ATTN_HEAD_DIM], DType::BF16)?,
                    value: gpu_zeros(&[ATTN_KV_HEADS, ATTN_HEAD_DIM], DType::BF16)?,
                    gate: gpu_zeros(&[ATTN_Q_HEADS, ATTN_HEAD_DIM], DType::BF16)?,
                    attended,
                    gated,
                    gated_flat,
                    split: qwen35_attention::SplitCtaWorkspace::new(ctx)?,
                },
                mlp: MlpWorkspace {
                    gate_up: gpu_zeros(&[1, 2 * INTERMEDIATE], DType::BF16)?,
                    hidden: gpu_zeros(&[1, INTERMEDIATE], DType::BF16)?,
                },
            },
            prefill: PrefillWorkspace {
                residual: gpu_zeros(&[PREFILL_TILE, HIDDEN], DType::BF16)?,
                normalized: gpu_zeros(&[PREFILL_TILE, HIDDEN], DType::BF16)?,
                mlp_normalized: gpu_zeros(&[PREFILL_TILE, HIDDEN], DType::BF16)?,
                mixer_delta: gpu_zeros(&[PREFILL_TILE, HIDDEN], DType::BF16)?,
                mlp_delta: gpu_zeros(&[PREFILL_TILE, HIDDEN], DType::BF16)?,
                gdn: GdnPrefillWorkspace {
                    qkv: gpu_zeros(&[PREFILL_TILE, GDN_CONV_DIM], DType::BF16)?,
                    z: gpu_zeros(&[PREFILL_TILE, GDN_VALUE_WIDTH], DType::BF16)?,
                    ab: gpu_zeros(&[PREFILL_TILE, 2 * GDN_HEADS], DType::BF16)?,
                    a: gpu_zeros(&[PREFILL_TILE, GDN_HEADS], DType::BF16)?,
                    b: gpu_zeros(&[PREFILL_TILE, GDN_HEADS], DType::BF16)?,
                    query: gpu_zeros(&[PREFILL_TILE, GDN_HEADS, GDN_DIM], DType::BF16)?,
                    key: gpu_zeros(&[PREFILL_TILE, GDN_HEADS, GDN_DIM], DType::BF16)?,
                    value: gpu_zeros(&[PREFILL_TILE, GDN_HEADS, GDN_DIM], DType::BF16)?,
                    g: gpu_zeros(&[PREFILL_TILE, GDN_HEADS], DType::F32)?,
                    beta: gpu_zeros(&[PREFILL_TILE, GDN_HEADS], DType::F32)?,
                    core: gpu_zeros(&[PREFILL_TILE, GDN_HEADS, GDN_DIM], DType::BF16)?,
                    normalized: gpu_zeros(&[PREFILL_TILE, GDN_HEADS, GDN_DIM], DType::BF16)?,
                },
                attention: AttentionPrefillWorkspace {
                    q_projection: gpu_zeros(&[PREFILL_TILE, 2 * ATTN_WIDTH], DType::BF16)?,
                    k_projection: gpu_zeros(&[PREFILL_TILE, ATTN_KV_WIDTH], DType::BF16)?,
                    v_projection: gpu_zeros(&[PREFILL_TILE, ATTN_KV_WIDTH], DType::BF16)?,
                    query: gpu_zeros(&[PREFILL_TILE, ATTN_Q_HEADS, ATTN_HEAD_DIM], DType::BF16)?,
                    key: gpu_zeros(&[PREFILL_TILE, ATTN_KV_HEADS, ATTN_HEAD_DIM], DType::BF16)?,
                    value: gpu_zeros(&[PREFILL_TILE, ATTN_KV_HEADS, ATTN_HEAD_DIM], DType::BF16)?,
                    gate: gpu_zeros(&[PREFILL_TILE, ATTN_Q_HEADS, ATTN_HEAD_DIM], DType::BF16)?,
                    attended: gpu_zeros(&[PREFILL_TILE, ATTN_Q_HEADS, ATTN_HEAD_DIM], DType::BF16)?,
                    gated: gpu_zeros(&[PREFILL_TILE, ATTN_Q_HEADS, ATTN_HEAD_DIM], DType::BF16)?,
                },
                mlp: MlpPrefillWorkspace {
                    gate_up: gpu_zeros(&[PREFILL_TILE, 2 * INTERMEDIATE], DType::BF16)?,
                    hidden: gpu_zeros(&[PREFILL_TILE, INTERMEDIATE], DType::BF16)?,
                },
            },
            marlin_prefill: match prefill_mode {
                Qwen35PrefillMode::M8 => None,
                Qwen35PrefillMode::MarlinM64 => Some(MarlinPrefillWorkspace {
                    residual: gpu_zeros(&[MARLIN_PREFILL_TILE, HIDDEN], DType::BF16)?,
                    normalized: gpu_zeros(&[MARLIN_PREFILL_TILE, HIDDEN], DType::BF16)?,
                    mlp_normalized: gpu_zeros(&[MARLIN_PREFILL_TILE, HIDDEN], DType::BF16)?,
                    mixer_delta: gpu_zeros(&[MARLIN_PREFILL_TILE, HIDDEN], DType::BF16)?,
                    mlp_delta: gpu_zeros(&[MARLIN_PREFILL_TILE, HIDDEN], DType::BF16)?,
                    layer_major: LayerMajorPrefillWorkspace {
                        residual: gpu_zeros(&[LAYER_MAJOR_PREFILL_ROWS, HIDDEN], DType::BF16)?,
                        normalized: gpu_zeros(&[LAYER_MAJOR_PREFILL_ROWS, HIDDEN], DType::BF16)?,
                        mixer_delta: gpu_zeros(
                            &[LAYER_MAJOR_PREFILL_ROWS, HIDDEN],
                            DType::BF16,
                        )?,
                        gdn: LayerMajorGdnWorkspace {
                            qkv: gpu_zeros(
                                &[LAYER_MAJOR_PREFILL_ROWS, GDN_CONV_DIM],
                                DType::BF16,
                            )?,
                            z: gpu_zeros(
                                &[LAYER_MAJOR_PREFILL_ROWS, GDN_VALUE_WIDTH],
                                DType::BF16,
                            )?,
                            normalized: gpu_zeros(
                                &[LAYER_MAJOR_PREFILL_ROWS, GDN_HEADS, GDN_DIM],
                                DType::BF16,
                            )?,
                            qkv_weight: MarlinPreparedWeight::new(
                                ctx,
                                HIDDEN,
                                GDN_CONV_DIM,
                            )?,
                            z_weight: MarlinPreparedWeight::new(
                                ctx,
                                HIDDEN,
                                GDN_VALUE_WIDTH,
                            )?,
                            out_weight: MarlinPreparedWeight::new(
                                ctx,
                                GDN_VALUE_WIDTH,
                                HIDDEN,
                            )?,
                        },
                        attention: LayerMajorAttentionWorkspace {
                            q_projection: gpu_zeros(
                                &[LAYER_MAJOR_PREFILL_ROWS, 2 * ATTN_WIDTH],
                                DType::BF16,
                            )?,
                            k_projection: gpu_zeros(
                                &[LAYER_MAJOR_PREFILL_ROWS, ATTN_KV_WIDTH],
                                DType::BF16,
                            )?,
                            v_projection: gpu_zeros(
                                &[LAYER_MAJOR_PREFILL_ROWS, ATTN_KV_WIDTH],
                                DType::BF16,
                            )?,
                            query: gpu_zeros(
                                &[LAYER_MAJOR_PREFILL_ROWS, ATTN_Q_HEADS, ATTN_HEAD_DIM],
                                DType::BF16,
                            )?,
                            key: gpu_zeros(
                                &[LAYER_MAJOR_PREFILL_ROWS, ATTN_KV_HEADS, ATTN_HEAD_DIM],
                                DType::BF16,
                            )?,
                            value: gpu_zeros(
                                &[LAYER_MAJOR_PREFILL_ROWS, ATTN_KV_HEADS, ATTN_HEAD_DIM],
                                DType::BF16,
                            )?,
                            gate: gpu_zeros(
                                &[LAYER_MAJOR_PREFILL_ROWS, ATTN_Q_HEADS, ATTN_HEAD_DIM],
                                DType::BF16,
                            )?,
                            attended: gpu_zeros(
                                &[LAYER_MAJOR_PREFILL_ROWS, ATTN_Q_HEADS, ATTN_HEAD_DIM],
                                DType::BF16,
                            )?,
                            gated: gpu_zeros(
                                &[LAYER_MAJOR_PREFILL_ROWS, ATTN_Q_HEADS, ATTN_HEAD_DIM],
                                DType::BF16,
                            )?,
                            q_weight: MarlinPreparedWeight::new(
                                ctx,
                                HIDDEN,
                                2 * ATTN_WIDTH,
                            )?,
                            k_weight: MarlinPreparedWeight::new(
                                ctx,
                                HIDDEN,
                                ATTN_KV_WIDTH,
                            )?,
                            v_weight: MarlinPreparedWeight::new(
                                ctx,
                                HIDDEN,
                                ATTN_KV_WIDTH,
                            )?,
                            o_weight: MarlinPreparedWeight::new(
                                ctx,
                                ATTN_WIDTH,
                                HIDDEN,
                            )?,
                        },
                        kernel: MarlinWorkspace::new(ctx)?,
                    },
                    mlp: MarlinMlpPrefillWorkspace {
                        gate_up: gpu_zeros(&[MARLIN_PREFILL_TILE, 2 * INTERMEDIATE], DType::BF16)?,
                        hidden: gpu_zeros(&[MARLIN_PREFILL_TILE, INTERMEDIATE], DType::BF16)?,
                        gate_up_weight: MarlinPreparedWeight::new(ctx, HIDDEN, 2 * INTERMEDIATE)?,
                        down_weight: MarlinPreparedWeight::new(ctx, INTERMEDIATE, HIDDEN)?,
                        kernel: MarlinWorkspace::new(ctx)?,
                    },
                }),
            },
            position: HostMappedBuffer::alloc(4, ctx.device_id()).map_err(Error::Cuda)?,
            rope_position: HostMappedBuffer::alloc(3 * 4, ctx.device_id()).map_err(Error::Cuda)?,
            prefill_positions: HostMappedBuffer::alloc(
                match prefill_mode {
                    Qwen35PrefillMode::M8 => PREFILL_TILE,
                    Qwen35PrefillMode::MarlinM64 => LAYER_MAJOR_PREFILL_ROWS,
                } * 4,
                ctx.device_id(),
            )
            .map_err(Error::Cuda)?,
            prefill_rope_positions: HostMappedBuffer::alloc(
                match prefill_mode {
                    Qwen35PrefillMode::M8 => PREFILL_TILE,
                    Qwen35PrefillMode::MarlinM64 => LAYER_MAJOR_PREFILL_ROWS,
                } * 3
                    * 4,
                ctx.device_id(),
            )
            .map_err(Error::Cuda)?,
            zero_conv: Tensor::zeros(vec![GDN_CONV_DIM, GDN_CONV_KERNEL], DType::BF16),
            zero_recurrent: Tensor::zeros(vec![GDN_HEADS, GDN_DIM, GDN_DIM], DType::F32),
            max_seq_len,
        })
    }

    pub fn reset(
        &self,
        ctx: &CudaContext,
        input: &Tensor,
        key_cache: &Tensor,
        value_cache: &Tensor,
    ) -> Result<()> {
        if input.device() != Device::Cpu
            || input.dtype() != DType::BF16
            || input.shape().dims() != [1, HIDDEN]
        {
            return Err(Error::Other(
                "Qwen3.5 hybrid unit reset input must be CPU BF16 [1,5120]".into(),
            ));
        }
        let cache_shape = [ATTN_KV_HEADS, self.max_seq_len, ATTN_HEAD_DIM];
        for (name, tensor) in [("key", key_cache), ("value", value_cache)] {
            if tensor.device() != Device::Cpu
                || tensor.dtype() != DType::BF16
                || tensor.shape().dims() != cache_shape
            {
                return Err(Error::Other(format!(
                    "Qwen3.5 hybrid unit {name} cache must be CPU BF16 {cache_shape:?}"
                )));
            }
        }
        for layer in &self.layers {
            match &layer.mixer {
                Mixer::Gdn { state, .. } => {
                    transfers::copy_cpu_to_cuda(&self.zero_conv, &state.conv)?;
                    transfers::copy_cpu_to_cuda(&self.zero_recurrent, &state.recurrent)?;
                }
                Mixer::Attention { state, .. } => {
                    transfers::copy_cpu_to_cuda(key_cache, &state.key_cache)?;
                    transfers::copy_cpu_to_cuda(value_cache, &state.value_cache)?;
                }
            }
        }
        self.set_token_input(ctx, input)?;
        ctx.synchronize().map_err(Error::Cuda)
    }

    /// Reset request-local recurrent state without clearing the full KV pool.
    /// A causal text request overwrites positions `0..prompt_len` before they
    /// become visible, so stale KV outside `valid_len` is unreachable.
    pub fn reset_text_request(&self, ctx: &CudaContext, input: &Tensor) -> Result<()> {
        for layer in &self.layers {
            if let Mixer::Gdn { state, .. } = &layer.mixer {
                transfers::copy_cpu_to_cuda(&self.zero_conv, &state.conv)?;
                transfers::copy_cpu_to_cuda(&self.zero_recurrent, &state.recurrent)?;
            }
        }
        self.set_token_input(ctx, input)?;
        ctx.synchronize().map_err(Error::Cuda)
    }

    /// Replace only the current token hidden input while preserving every
    /// recurrent and KV state owner for the next decode step.
    pub fn set_token_input(&self, ctx: &CudaContext, input: &Tensor) -> Result<()> {
        if input.device() != Device::Cpu
            || input.dtype() != DType::BF16
            || input.shape().dims() != [1, HIDDEN]
        {
            return Err(Error::Other(
                "Qwen3.5 token input must be CPU BF16 [1,5120]".into(),
            ));
        }
        transfers::copy_cpu_to_cuda(input, &self.workspace.residual)?;
        qwen35_common::rmsnorm_offset_write(
            ctx,
            &self.workspace.residual,
            &self.layers[0].input_norm,
            &self.workspace.normalized,
            RMS_EPSILON,
        )
    }

    pub fn reset_prefill8_text_request(&self, ctx: &CudaContext, input: &Tensor) -> Result<()> {
        for layer in &self.layers {
            if let Mixer::Gdn { state, .. } = &layer.mixer {
                transfers::copy_cpu_to_cuda(&self.zero_conv, &state.conv)?;
                transfers::copy_cpu_to_cuda(&self.zero_recurrent, &state.recurrent)?;
            }
        }
        self.set_prefill8_input(ctx, input)?;
        ctx.synchronize().map_err(Error::Cuda)
    }

    pub fn set_prefill8_input(&self, ctx: &CudaContext, input: &Tensor) -> Result<()> {
        if input.device() != Device::Cpu
            || input.dtype() != DType::BF16
            || input.shape().dims() != [PREFILL_TILE, HIDDEN]
        {
            return Err(Error::Other(format!(
                "Qwen3.5 prefill input must be CPU BF16 [{PREFILL_TILE},{HIDDEN}]"
            )));
        }
        transfers::copy_cpu_to_cuda(input, &self.prefill.residual)?;
        qwen35_common::rmsnorm_offset_write(
            ctx,
            &self.prefill.residual,
            &self.layers[0].input_norm,
            &self.prefill.normalized,
            RMS_EPSILON,
        )
    }

    pub fn has_marlin_prefill64(&self) -> bool {
        self.marlin_prefill.is_some()
    }

    pub fn layer_major_prefill_rows() -> usize {
        LAYER_MAJOR_PREFILL_ROWS
    }

    pub fn set_marlin_prefill64_input(&self, ctx: &CudaContext, input: &Tensor) -> Result<()> {
        let workspace = self.marlin_prefill.as_ref().ok_or_else(|| {
            Error::Other("Qwen3.5 Marlin M64 prefill workspace is not enabled".into())
        })?;
        if input.device() != Device::Cpu
            || input.dtype() != DType::BF16
            || input.shape().dims() != [MARLIN_PREFILL_TILE, HIDDEN]
        {
            return Err(Error::Other(format!(
                "Qwen3.5 Marlin prefill input must be CPU BF16 [{MARLIN_PREFILL_TILE},{HIDDEN}]"
            )));
        }
        transfers::copy_cpu_to_cuda(input, &workspace.residual)?;
        for subtile in 0..MARLIN_PREFILL_SUBTILES {
            let first = subtile * PREFILL_TILE;
            qwen35_common::rmsnorm_offset_write(
                ctx,
                &cuda_row_view(&workspace.residual, first, PREFILL_TILE)?,
                &self.layers[0].input_norm,
                &cuda_row_view(&workspace.normalized, first, PREFILL_TILE)?,
                RMS_EPSILON,
            )?;
        }
        Ok(())
    }

    pub fn set_layer_major_prefill1k_input(
        &self,
        ctx: &CudaContext,
        input: &Tensor,
    ) -> Result<()> {
        let workspace = self.marlin_prefill.as_ref().ok_or_else(|| {
            Error::Other("Qwen3.5 Marlin M64 prefill workspace is not enabled".into())
        })?;
        if input.device() != Device::Cpu
            || input.dtype() != DType::BF16
            || input.shape().dims() != [LAYER_MAJOR_PREFILL_ROWS, HIDDEN]
        {
            return Err(Error::Other(format!(
                "Qwen3.5 layer-major prefill input must be CPU BF16 [{LAYER_MAJOR_PREFILL_ROWS},{HIDDEN}]"
            )));
        }
        transfers::copy_cpu_to_cuda(input, &workspace.layer_major.residual)?;
        for first in (0..LAYER_MAJOR_PREFILL_ROWS).step_by(PREFILL_TILE) {
            qwen35_common::rmsnorm_offset_write(
                ctx,
                &cuda_row_view(&workspace.layer_major.residual, first, PREFILL_TILE)?,
                &self.layers[0].input_norm,
                &cuda_row_view(&workspace.layer_major.normalized, first, PREFILL_TILE)?,
                RMS_EPSILON,
            )?;
        }
        Ok(())
    }

    pub fn forward_prefill8(
        &self,
        ctx: &CudaContext,
        start_position: usize,
        profile: bool,
    ) -> Result<()> {
        let rope_positions = std::array::from_fn(|offset| {
            let position = (start_position + offset) as u32;
            [position, position, position]
        });
        self.forward_prefill8_with_mrope(ctx, start_position, &rope_positions, profile)
    }

    pub fn forward_prefill8_with_mrope(
        &self,
        ctx: &CudaContext,
        start_position: usize,
        rope_positions: &[[u32; 3]; PREFILL_TILE],
        profile: bool,
    ) -> Result<()> {
        if start_position
            .checked_add(PREFILL_TILE)
            .is_none_or(|end| end > self.max_seq_len)
        {
            return Err(Error::Other(format!(
                "Qwen3.5 prefill tile [{start_position}..{}) exceeds KV capacity {}",
                start_position.saturating_add(PREFILL_TILE),
                self.max_seq_len
            )));
        }
        let positions = (start_position..start_position + PREFILL_TILE)
            .map(|position| position as u32)
            .collect::<Vec<_>>();
        self.prefill_positions
            .write_u32s(&positions)
            .map_err(Error::Cuda)?;
        let rope_positions = rope_positions
            .iter()
            .flat_map(|position| position.iter().copied())
            .collect::<Vec<_>>();
        self.prefill_rope_positions
            .write_u32s(&rope_positions)
            .map_err(Error::Cuda)?;
        apxinf_cuda::kernels::with_workspace(&self.workspace.graph, || {
            self.forward_prefill8_inner(ctx, start_position, profile)
        })
    }

    pub fn forward_marlin_prefill64(
        &self,
        ctx: &CudaContext,
        start_position: usize,
        profile: bool,
    ) -> Result<()> {
        if self.marlin_prefill.is_none() {
            return Err(Error::Other(
                "Qwen3.5 Marlin M64 prefill was not enabled at model load".into(),
            ));
        }
        if start_position
            .checked_add(MARLIN_PREFILL_TILE)
            .is_none_or(|end| end > self.max_seq_len)
        {
            return Err(Error::Other(format!(
                "Qwen3.5 Marlin prefill tile [{start_position}..{}) exceeds KV capacity {}",
                start_position.saturating_add(MARLIN_PREFILL_TILE),
                self.max_seq_len
            )));
        }
        let positions = (start_position..start_position + MARLIN_PREFILL_TILE)
            .map(|position| position as u32)
            .collect::<Vec<_>>();
        self.prefill_positions
            .write_u32s(&positions)
            .map_err(Error::Cuda)?;
        let rope_positions = positions
            .iter()
            .flat_map(|position| [*position, *position, *position])
            .collect::<Vec<_>>();
        self.prefill_rope_positions
            .write_u32s(&rope_positions)
            .map_err(Error::Cuda)?;
        apxinf_cuda::kernels::with_workspace(&self.workspace.graph, || {
            self.forward_marlin_prefill64_inner(ctx, start_position, profile)
        })
    }

    pub fn forward_layer_major_prefill1k(
        &self,
        ctx: &CudaContext,
        profile: bool,
    ) -> Result<()> {
        if self.marlin_prefill.is_none() {
            return Err(Error::Other(
                "Qwen3.5 Marlin M64 prefill was not enabled at model load".into(),
            ));
        }
        if LAYER_MAJOR_PREFILL_ROWS > self.max_seq_len {
            return Err(Error::Other(format!(
                "Qwen3.5 layer-major prefill length {LAYER_MAJOR_PREFILL_ROWS} exceeds KV capacity {}",
                self.max_seq_len
            )));
        }
        let positions = (0..LAYER_MAJOR_PREFILL_ROWS)
            .map(|position| position as u32)
            .collect::<Vec<_>>();
        self.prefill_positions
            .write_u32s(&positions)
            .map_err(Error::Cuda)?;
        let rope_positions = positions
            .iter()
            .flat_map(|position| [*position, *position, *position])
            .collect::<Vec<_>>();
        self.prefill_rope_positions
            .write_u32s(&rope_positions)
            .map_err(Error::Cuda)?;
        apxinf_cuda::kernels::with_workspace(&self.workspace.graph, || {
            self.forward_layer_major_prefill1k_inner(ctx, profile)
        })
    }

    pub fn prefill_output(&self) -> &Tensor {
        &self.prefill.residual
    }

    pub fn prefill_normalized_output(&self) -> &Tensor {
        &self.prefill.normalized
    }

    pub fn marlin_prefill64_output(&self) -> Result<&Tensor> {
        self.marlin_prefill
            .as_ref()
            .map(|workspace| &workspace.residual)
            .ok_or_else(|| {
                Error::Other("Qwen3.5 Marlin M64 prefill workspace is not enabled".into())
            })
    }

    pub fn marlin_prefill64_normalized_output(&self) -> Result<&Tensor> {
        self.marlin_prefill
            .as_ref()
            .map(|workspace| &workspace.normalized)
            .ok_or_else(|| {
                Error::Other("Qwen3.5 Marlin M64 prefill workspace is not enabled".into())
            })
    }

    /// Publish the last row of an M8 prompt tile to the one-token decode
    /// workspace. This is required only when the prompt length is an exact
    /// multiple of eight so the first LM-head call sees the final prompt row.
    pub fn commit_prefill8_last(&self, ctx: &CudaContext) -> Result<()> {
        let row_bytes = HIDDEN * DType::BF16.size_in_bytes();
        let offset = (PREFILL_TILE - 1) * row_bytes;
        for (source, destination) in [
            (&self.prefill.residual, &self.workspace.residual),
            (&self.prefill.normalized, &self.workspace.normalized),
        ] {
            let source = CudaBuffer::from_tensor(source)
                .map_err(Error::Cuda)?
                .view(offset, row_bytes)
                .map_err(Error::Cuda)?;
            let destination = CudaBuffer::from_tensor(destination).map_err(Error::Cuda)?;
            destination
                .copy_from_device_async(&source, row_bytes, ctx.stream())
                .map_err(Error::Cuda)?;
        }
        Ok(())
    }

    pub fn commit_marlin_prefill64_last(&self, ctx: &CudaContext) -> Result<()> {
        let workspace = self.marlin_prefill.as_ref().ok_or_else(|| {
            Error::Other("Qwen3.5 Marlin M64 prefill workspace is not enabled".into())
        })?;
        let row_bytes = HIDDEN * DType::BF16.size_in_bytes();
        let offset = (MARLIN_PREFILL_TILE - 1) * row_bytes;
        for (source, destination) in [
            (&workspace.residual, &self.workspace.residual),
            (&workspace.normalized, &self.workspace.normalized),
        ] {
            let source = CudaBuffer::from_tensor(source)
                .map_err(Error::Cuda)?
                .view(offset, row_bytes)
                .map_err(Error::Cuda)?;
            let destination = CudaBuffer::from_tensor(destination).map_err(Error::Cuda)?;
            destination
                .copy_from_device_async(&source, row_bytes, ctx.stream())
                .map_err(Error::Cuda)?;
        }
        Ok(())
    }

    pub fn commit_layer_major_prefill1k_last(&self, ctx: &CudaContext) -> Result<()> {
        let workspace = self.marlin_prefill.as_ref().ok_or_else(|| {
            Error::Other("Qwen3.5 Marlin M64 prefill workspace is not enabled".into())
        })?;
        let row_bytes = HIDDEN * DType::BF16.size_in_bytes();
        let offset = (LAYER_MAJOR_PREFILL_ROWS - 1) * row_bytes;
        for (source, destination) in [
            (&workspace.layer_major.residual, &self.workspace.residual),
            (
                &workspace.layer_major.normalized,
                &self.workspace.normalized,
            ),
        ] {
            let source = CudaBuffer::from_tensor(source)
                .map_err(Error::Cuda)?
                .view(offset, row_bytes)
                .map_err(Error::Cuda)?;
            let destination = CudaBuffer::from_tensor(destination).map_err(Error::Cuda)?;
            destination
                .copy_from_device_async(&source, row_bytes, ctx.stream())
                .map_err(Error::Cuda)?;
        }
        Ok(())
    }

    pub fn forward(
        &self,
        ctx: &CudaContext,
        mode: HybridUnitMode,
        bucket_kv_len: usize,
        cache_position: u32,
        profile: bool,
    ) -> Result<()> {
        self.forward_with_mrope(
            ctx,
            mode,
            bucket_kv_len,
            cache_position,
            [cache_position, cache_position, cache_position],
            profile,
        )
    }

    pub fn forward_with_mrope(
        &self,
        ctx: &CudaContext,
        mode: HybridUnitMode,
        bucket_kv_len: usize,
        cache_position: u32,
        rope_position: [u32; 3],
        profile: bool,
    ) -> Result<()> {
        if bucket_kv_len == 0
            || bucket_kv_len > self.max_seq_len
            || cache_position as usize >= bucket_kv_len
        {
            return Err(Error::Other(
                "Qwen3.5 hybrid unit KV position/bucket contract mismatch".into(),
            ));
        }
        self.position
            .write_u32(cache_position)
            .map_err(Error::Cuda)?;
        self.rope_position
            .write_u32s(&rope_position)
            .map_err(Error::Cuda)?;
        apxinf_cuda::kernels::with_workspace(&self.workspace.graph, || {
            self.forward_inner(
                ctx,
                mode,
                bucket_kv_len,
                self.position.address(),
                self.rope_position.address(),
                profile,
            )
        })
    }

    pub fn output(&self) -> &Tensor {
        &self.workspace.residual
    }

    pub fn normalized_output(&self) -> &Tensor {
        &self.workspace.normalized
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    fn forward_inner(
        &self,
        ctx: &CudaContext,
        mode: HybridUnitMode,
        bucket_kv_len: usize,
        cache_position: CudaDeviceAddress,
        rope_position: CudaDeviceAddress,
        profile: bool,
    ) -> Result<()> {
        let _unit = profile.then(|| apxinf_cuda::nvtx::range("qwen35.hybrid_unit.complete"));
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let layer_range = format!("qwen35.hybrid_stack.layer{}", layer.index);
            let _layer = profile.then(|| apxinf_cuda::nvtx::range(&layer_range));
            match &layer.mixer {
                Mixer::Gdn { weights, state } => self.forward_gdn(
                    ctx,
                    weights,
                    state,
                    mode,
                    &self.workspace.normalized,
                    &self.workspace.mixer_delta,
                    profile,
                )?,
                Mixer::Attention { weights, state } => self.forward_attention(
                    ctx,
                    weights,
                    state,
                    mode,
                    &self.workspace.normalized,
                    &self.workspace.mixer_delta,
                    bucket_kv_len,
                    cache_position,
                    rope_position,
                    profile,
                )?,
            }
            qwen35_common::residual_add_rmsnorm_offset_write(
                ctx,
                &self.workspace.residual,
                &self.workspace.mixer_delta,
                &layer.post_attention_norm,
                &self.workspace.mlp_normalized,
                RMS_EPSILON,
            )?;
            self.forward_mlp(
                ctx,
                &layer.mlp,
                &self.workspace.mlp_normalized,
                &self.workspace.mlp_delta,
                profile,
            )?;
            let next_norm = if layer_index + 1 < self.layers.len() {
                &self.layers[layer_index + 1].input_norm
            } else {
                &self.next_input_norm
            };
            qwen35_common::residual_add_rmsnorm_offset_write(
                ctx,
                &self.workspace.residual,
                &self.workspace.mlp_delta,
                next_norm,
                &self.workspace.normalized,
                RMS_EPSILON,
            )?;
        }
        Ok(())
    }

    fn forward_prefill8_inner(
        &self,
        ctx: &CudaContext,
        start_position: usize,
        profile: bool,
    ) -> Result<()> {
        let _unit = profile.then(|| apxinf_cuda::nvtx::range("qwen35.prefill8.complete"));
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let layer_range = format!("qwen35.prefill8.layer{}", layer.index);
            let _layer = profile.then(|| apxinf_cuda::nvtx::range(&layer_range));
            match &layer.mixer {
                Mixer::Gdn { weights, state } => self.forward_gdn_prefill8(
                    ctx,
                    weights,
                    state,
                    &self.prefill.normalized,
                    &self.prefill.mixer_delta,
                )?,
                Mixer::Attention { weights, state } => self.forward_attention_prefill8(
                    ctx,
                    weights,
                    state,
                    &self.prefill.normalized,
                    &self.prefill.mixer_delta,
                    start_position,
                    0,
                )?,
            }
            qwen35_common::residual_add_rmsnorm_offset_write(
                ctx,
                &self.prefill.residual,
                &self.prefill.mixer_delta,
                &layer.post_attention_norm,
                &self.prefill.mlp_normalized,
                RMS_EPSILON,
            )?;
            self.forward_mlp_prefill8(
                ctx,
                &layer.mlp,
                &self.prefill.mlp_normalized,
                &self.prefill.mlp_delta,
            )?;
            let next_norm = if layer_index + 1 < self.layers.len() {
                &self.layers[layer_index + 1].input_norm
            } else {
                &self.next_input_norm
            };
            qwen35_common::residual_add_rmsnorm_offset_write(
                ctx,
                &self.prefill.residual,
                &self.prefill.mlp_delta,
                next_norm,
                &self.prefill.normalized,
                RMS_EPSILON,
            )?;
        }
        Ok(())
    }

    fn forward_marlin_prefill64_inner(
        &self,
        ctx: &CudaContext,
        start_position: usize,
        profile: bool,
    ) -> Result<()> {
        let workspace = self.marlin_prefill.as_ref().ok_or_else(|| {
            Error::Other("Qwen3.5 Marlin M64 prefill workspace is not enabled".into())
        })?;
        let _unit = profile.then(|| apxinf_cuda::nvtx::range("qwen35.prefill64.complete"));
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let layer_range = format!("qwen35.prefill64.layer{}", layer.index);
            let _layer = profile.then(|| apxinf_cuda::nvtx::range(&layer_range));
            for subtile in 0..MARLIN_PREFILL_SUBTILES {
                let first = subtile * PREFILL_TILE;
                let normalized = cuda_row_view(&workspace.normalized, first, PREFILL_TILE)?;
                let mixer_delta = cuda_row_view(&workspace.mixer_delta, first, PREFILL_TILE)?;
                match &layer.mixer {
                    Mixer::Gdn { weights, state } => {
                        self.forward_gdn_prefill8(ctx, weights, state, &normalized, &mixer_delta)?
                    }
                    Mixer::Attention { weights, state } => self.forward_attention_prefill8(
                        ctx,
                        weights,
                        state,
                        &normalized,
                        &mixer_delta,
                        start_position + first,
                        first,
                    )?,
                }
                qwen35_common::residual_add_rmsnorm_offset_write(
                    ctx,
                    &cuda_row_view(&workspace.residual, first, PREFILL_TILE)?,
                    &mixer_delta,
                    &layer.post_attention_norm,
                    &cuda_row_view(&workspace.mlp_normalized, first, PREFILL_TILE)?,
                    RMS_EPSILON,
                )?;
            }

            self.forward_mlp_marlin64(
                ctx,
                &layer.mlp,
                &workspace.mlp_normalized,
                &workspace.mlp_delta,
            )?;
            let next_norm = if layer_index + 1 < self.layers.len() {
                &self.layers[layer_index + 1].input_norm
            } else {
                &self.next_input_norm
            };
            for subtile in 0..MARLIN_PREFILL_SUBTILES {
                let first = subtile * PREFILL_TILE;
                qwen35_common::residual_add_rmsnorm_offset_write(
                    ctx,
                    &cuda_row_view(&workspace.residual, first, PREFILL_TILE)?,
                    &cuda_row_view(&workspace.mlp_delta, first, PREFILL_TILE)?,
                    next_norm,
                    &cuda_row_view(&workspace.normalized, first, PREFILL_TILE)?,
                    RMS_EPSILON,
                )?;
            }
        }
        Ok(())
    }

    fn forward_layer_major_prefill1k_inner(
        &self,
        ctx: &CudaContext,
        profile: bool,
    ) -> Result<()> {
        let workspace = self.marlin_prefill.as_ref().ok_or_else(|| {
            Error::Other("Qwen3.5 Marlin M64 prefill workspace is not enabled".into())
        })?;
        let _unit =
            profile.then(|| apxinf_cuda::nvtx::range("qwen35.prefill1k.layer_major.complete"));
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let layer_range = format!("qwen35.prefill1k.layer_major.layer{}", layer.index);
            let _layer = profile.then(|| apxinf_cuda::nvtx::range(&layer_range));
            match &layer.mixer {
                Mixer::Gdn { weights, state } => {
                    self.forward_gdn_layer_major_m64(ctx, weights, state)?
                }
                Mixer::Attention { weights, state } => {
                    self.forward_attention_layer_major_m64(ctx, weights, state)?
                }
            }
            self.prepare_mlp_marlin64_layer_major(ctx, &layer.mlp)?;
            for tile_first in (0..LAYER_MAJOR_PREFILL_ROWS).step_by(MARLIN_PREFILL_TILE) {
                qwen35_common::residual_add_rmsnorm_offset_write(
                    ctx,
                    &cuda_row_view(
                        &workspace.layer_major.residual,
                        tile_first,
                        MARLIN_PREFILL_TILE,
                    )?,
                    &cuda_row_view(
                        &workspace.layer_major.mixer_delta,
                        tile_first,
                        MARLIN_PREFILL_TILE,
                    )?,
                    &layer.post_attention_norm,
                    &workspace.mlp_normalized,
                    RMS_EPSILON,
                )?;
                self.forward_mlp_marlin64_layer_major(
                    ctx,
                    &workspace.mlp_normalized,
                    &workspace.mlp_delta,
                )?;
                let next_norm = if layer_index + 1 < self.layers.len() {
                    &self.layers[layer_index + 1].input_norm
                } else {
                    &self.next_input_norm
                };
                qwen35_common::residual_add_rmsnorm_offset_write(
                    ctx,
                    &cuda_row_view(
                        &workspace.layer_major.residual,
                        tile_first,
                        MARLIN_PREFILL_TILE,
                    )?,
                    &workspace.mlp_delta,
                    next_norm,
                    &cuda_row_view(
                        &workspace.layer_major.normalized,
                        tile_first,
                        MARLIN_PREFILL_TILE,
                    )?,
                    RMS_EPSILON,
                )?;
            }
        }
        Ok(())
    }

    fn forward_gdn_layer_major_m64(
        &self,
        ctx: &CudaContext,
        weights: &GdnWeights,
        state: &GdnState,
    ) -> Result<()> {
        let workspace = &self
            .marlin_prefill
            .as_ref()
            .ok_or_else(|| {
                Error::Other("Qwen3.5 Marlin M64 prefill workspace is not enabled".into())
            })?
            .layer_major;
        let gdn = &workspace.gdn;
        let input = &workspace.normalized;

        gdn.qkv_weight.prepare(ctx, weights.qkv.view())?;
        for first in (0..LAYER_MAJOR_PREFILL_ROWS).step_by(MARLIN_PREFILL_TILE) {
            gemm::w4a16_marlin_write(
                ctx,
                &cuda_row_view(input, first, MARLIN_PREFILL_TILE)?,
                gdn.qkv_weight.view(),
                &cuda_row_view(&gdn.qkv, first, MARLIN_PREFILL_TILE)?,
                &workspace.kernel,
            )?;
        }
        gdn.z_weight.prepare(ctx, weights.z.view())?;
        for first in (0..LAYER_MAJOR_PREFILL_ROWS).step_by(MARLIN_PREFILL_TILE) {
            gemm::w4a16_marlin_write(
                ctx,
                &cuda_row_view(input, first, MARLIN_PREFILL_TILE)?,
                gdn.z_weight.view(),
                &cuda_row_view(&gdn.z, first, MARLIN_PREFILL_TILE)?,
                &workspace.kernel,
            )?;
        }

        for first in (0..LAYER_MAJOR_PREFILL_ROWS).step_by(PREFILL_TILE) {
            self.forward_gdn_layer_major_core(
                ctx,
                weights,
                state,
                &cuda_row_view(input, first, PREFILL_TILE)?,
                &cuda_row_view(&gdn.qkv, first, PREFILL_TILE)?,
                &cuda_row_view(&gdn.z, first, PREFILL_TILE)?,
                &cuda_row_view(&gdn.normalized, first, PREFILL_TILE)?,
            )?;
        }

        match &weights.out {
            GdnOutputWeight::W4(weight) => {
                gdn.out_weight.prepare(ctx, weight.view())?;
                for first in (0..LAYER_MAJOR_PREFILL_ROWS).step_by(MARLIN_PREFILL_TILE) {
                    gemm::w4a16_marlin_write(
                        ctx,
                        &cuda_row_view(&gdn.normalized, first, MARLIN_PREFILL_TILE)?
                            .reshape(vec![MARLIN_PREFILL_TILE, GDN_VALUE_WIDTH])?,
                        gdn.out_weight.view(),
                        &cuda_row_view(
                            &workspace.mixer_delta,
                            first,
                            MARLIN_PREFILL_TILE,
                        )?,
                        &workspace.kernel,
                    )?;
                }
                Ok(())
            }
            GdnOutputWeight::Bf16 { weight, .. } => {
                for first in (0..LAYER_MAJOR_PREFILL_ROWS).step_by(PREFILL_TILE) {
                    bf16_linear_serial_rows(
                        ctx,
                        &cuda_row_view(&gdn.normalized, first, PREFILL_TILE)?
                            .reshape(vec![PREFILL_TILE, GDN_VALUE_WIDTH])?,
                        weight,
                        &cuda_row_view(&workspace.mixer_delta, first, PREFILL_TILE)?,
                        PREFILL_TILE,
                        GDN_VALUE_WIDTH,
                        HIDDEN,
                    )?;
                }
                Ok(())
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_gdn_layer_major_core(
        &self,
        ctx: &CudaContext,
        weights: &GdnWeights,
        state: &GdnState,
        input: &Tensor,
        qkv: &Tensor,
        z: &Tensor,
        normalized: &Tensor,
    ) -> Result<()> {
        let scratch = &self.prefill.gdn;
        bf16_linear_prefill_m8(
            ctx,
            input,
            &weights.ab,
            &scratch.ab,
            HIDDEN,
            2 * GDN_HEADS,
        )?;
        qwen35_conv4_prepare_m8_write(
            ctx,
            qkv,
            &weights.conv,
            &state.conv,
            &scratch.ab,
            &weights.a_log,
            &weights.dt_bias,
            &scratch.a,
            &scratch.b,
            &scratch.query,
            &scratch.key,
            &scratch.value,
            &scratch.g,
            &scratch.beta,
        )?;
        qwen35_recurrent_m8_hybrid_write(
            ctx,
            &scratch.query,
            &scratch.key,
            &scratch.value,
            &scratch.g,
            &scratch.beta,
            &state.recurrent,
            &scratch.core,
        )?;
        qwen35_gated_rmsnorm_m8_write(
            ctx,
            &scratch.core,
            &z.reshape(vec![PREFILL_TILE, GDN_HEADS, GDN_DIM])?,
            &weights.norm,
            normalized,
            RMS_EPSILON,
        )
    }

    fn forward_attention_layer_major_m64(
        &self,
        ctx: &CudaContext,
        weights: &AttentionWeights,
        state: &AttentionState,
    ) -> Result<()> {
        let workspace = &self
            .marlin_prefill
            .as_ref()
            .ok_or_else(|| {
                Error::Other("Qwen3.5 Marlin M64 prefill workspace is not enabled".into())
            })?
            .layer_major;
        let attention_workspace = &workspace.attention;
        let input = &workspace.normalized;

        attention_workspace.q_weight.prepare(ctx, weights.q.view())?;
        for first in (0..LAYER_MAJOR_PREFILL_ROWS).step_by(MARLIN_PREFILL_TILE) {
            gemm::w4a16_marlin_write(
                ctx,
                &cuda_row_view(input, first, MARLIN_PREFILL_TILE)?,
                attention_workspace.q_weight.view(),
                &cuda_row_view(
                    &attention_workspace.q_projection,
                    first,
                    MARLIN_PREFILL_TILE,
                )?,
                &workspace.kernel,
            )?;
        }
        attention_workspace.k_weight.prepare(ctx, weights.k.view())?;
        for first in (0..LAYER_MAJOR_PREFILL_ROWS).step_by(MARLIN_PREFILL_TILE) {
            gemm::w4a16_marlin_write(
                ctx,
                &cuda_row_view(input, first, MARLIN_PREFILL_TILE)?,
                attention_workspace.k_weight.view(),
                &cuda_row_view(
                    &attention_workspace.k_projection,
                    first,
                    MARLIN_PREFILL_TILE,
                )?,
                &workspace.kernel,
            )?;
        }
        attention_workspace.v_weight.prepare(ctx, weights.v.view())?;
        for first in (0..LAYER_MAJOR_PREFILL_ROWS).step_by(MARLIN_PREFILL_TILE) {
            gemm::w4a16_marlin_write(
                ctx,
                &cuda_row_view(input, first, MARLIN_PREFILL_TILE)?,
                attention_workspace.v_weight.view(),
                &cuda_row_view(
                    &attention_workspace.v_projection,
                    first,
                    MARLIN_PREFILL_TILE,
                )?,
                &workspace.kernel,
            )?;
        }

        for first in (0..LAYER_MAJOR_PREFILL_ROWS).step_by(PREFILL_TILE) {
            self.forward_attention_layer_major_core(
                ctx,
                weights,
                state,
                attention_workspace,
                first,
            )?;
        }

        attention_workspace.o_weight.prepare(ctx, weights.o.view())?;
        for first in (0..LAYER_MAJOR_PREFILL_ROWS).step_by(MARLIN_PREFILL_TILE) {
            gemm::w4a16_marlin_write(
                ctx,
                &cuda_row_view(
                    &attention_workspace.gated,
                    first,
                    MARLIN_PREFILL_TILE,
                )?
                .reshape(vec![MARLIN_PREFILL_TILE, ATTN_WIDTH])?,
                attention_workspace.o_weight.view(),
                &cuda_row_view(
                    &workspace.mixer_delta,
                    first,
                    MARLIN_PREFILL_TILE,
                )?,
                &workspace.kernel,
            )?;
        }
        Ok(())
    }

    fn forward_attention_layer_major_core(
        &self,
        ctx: &CudaContext,
        weights: &AttentionWeights,
        state: &AttentionState,
        workspace: &LayerMajorAttentionWorkspace,
        first: usize,
    ) -> Result<()> {
        let q_projection = cuda_row_view(&workspace.q_projection, first, PREFILL_TILE)?;
        let k_projection = cuda_row_view(&workspace.k_projection, first, PREFILL_TILE)?;
        let v_projection = cuda_row_view(&workspace.v_projection, first, PREFILL_TILE)?;
        let query = cuda_row_view(&workspace.query, first, PREFILL_TILE)?;
        let key = cuda_row_view(&workspace.key, first, PREFILL_TILE)?;
        let value = cuda_row_view(&workspace.value, first, PREFILL_TILE)?;
        let gate = cuda_row_view(&workspace.gate, first, PREFILL_TILE)?;
        qwen35_attention::prepare_m8_write(
            ctx,
            &q_projection,
            &k_projection,
            &v_projection,
            &weights.q_norm,
            &weights.k_norm,
            &query,
            &key,
            &value,
            &gate,
            self.prefill_rope_positions
                .address_at(first * 3 * 4, PREFILL_TILE * 3 * 4)
                .map_err(Error::Cuda)?,
        )?;

        let key_cache = CudaBuffer::from_tensor(&state.key_cache).map_err(Error::Cuda)?;
        let value_cache = CudaBuffer::from_tensor(&state.value_cache).map_err(Error::Cuda)?;
        cache::append(
            ctx,
            &key_cache,
            &key,
            ATTN_KV_HEADS,
            ATTN_HEAD_DIM,
            self.max_seq_len,
            first,
            PREFILL_TILE,
        )?;
        cache::append(
            ctx,
            &value_cache,
            &value,
            ATTN_KV_HEADS,
            ATTN_HEAD_DIM,
            self.max_seq_len,
            first,
            PREFILL_TILE,
        )?;

        let all_query = CudaBuffer::from_tensor(&workspace.query).map_err(Error::Cuda)?;
        let all_attended = CudaBuffer::from_tensor(&workspace.attended).map_err(Error::Cuda)?;
        let row_bytes = ATTN_WIDTH * DType::BF16.size_in_bytes();
        if first >= 256 {
            let tile_bytes = PREFILL_TILE * row_bytes;
            qwen35_attention::flash_split_cta_m8_buffer_write(
                ctx,
                &all_query.view(first * row_bytes, tile_bytes).map_err(Error::Cuda)?,
                &key_cache,
                &value_cache,
                &all_attended.view(first * row_bytes, tile_bytes).map_err(Error::Cuda)?,
                &self.workspace.attention.split,
                qwen35_attention::SPLIT_CTA_CANDIDATE_COUNT,
                first + PREFILL_TILE,
                self.max_seq_len,
                ATTENTION_SCALE,
                self.prefill_positions
                    .address_at(first * 4, PREFILL_TILE * 4)
                    .map_err(Error::Cuda)?,
                PREFILL_TILE,
            )?;
        } else {
            for token in first..first + PREFILL_TILE {
                let position = self
                    .prefill_positions
                    .address_at(token * 4, 4)
                    .map_err(Error::Cuda)?;
                let query_row = all_query
                    .view(token * row_bytes, row_bytes)
                    .map_err(Error::Cuda)?;
                let attended_row = all_attended
                    .view(token * row_bytes, row_bytes)
                    .map_err(Error::Cuda)?;
                let kv_len = token + 1;
                if let Some(split) = qwen35_attention::split_cta_candidate_for_bucket(kv_len) {
                    qwen35_attention::flash_split_cta_buffer_write(
                        ctx,
                        &query_row,
                        &key_cache,
                        &value_cache,
                        &attended_row,
                        &self.workspace.attention.split,
                        split,
                        kv_len,
                        self.max_seq_len,
                        ATTENTION_SCALE,
                        position,
                    )?;
                } else {
                    attention::flash_bf16_into(
                        ctx,
                        &query_row,
                        &key_cache,
                        &value_cache,
                        &attended_row,
                        ATTN_Q_HEADS,
                        ATTN_KV_HEADS,
                        ATTN_HEAD_DIM,
                        kv_len,
                        self.max_seq_len,
                        ATTENTION_SCALE,
                        position,
                    )?;
                }
            }
        }
        qwen35_attention::gate_m8_write(
            ctx,
            &cuda_row_view(&workspace.attended, first, PREFILL_TILE)?,
            &gate,
            &cuda_row_view(&workspace.gated, first, PREFILL_TILE)?,
        )
    }
    fn forward_mlp_marlin64(
        &self,
        ctx: &CudaContext,
        weights: &MlpWeights,
        input: &Tensor,
        output: &Tensor,
    ) -> Result<()> {
        let workspace = &self
            .marlin_prefill
            .as_ref()
            .ok_or_else(|| {
                Error::Other("Qwen3.5 Marlin M64 prefill workspace is not enabled".into())
            })?
            .mlp;
        workspace
            .gate_up_weight
            .prepare(ctx, weights.gate_up.view())?;
        gemm::w4a16_marlin_write(
            ctx,
            input,
            workspace.gate_up_weight.view(),
            &workspace.gate_up,
            &workspace.kernel,
        )?;
        let gate_up = CudaBuffer::from_tensor(&workspace.gate_up).map_err(Error::Cuda)?;
        let hidden = CudaBuffer::from_tensor(&workspace.hidden).map_err(Error::Cuda)?;
        let gate_up_bytes = 2 * INTERMEDIATE * DType::BF16.size_in_bytes();
        let hidden_bytes = INTERMEDIATE * DType::BF16.size_in_bytes();
        for token in 0..MARLIN_PREFILL_TILE {
            activation::silu_mul_bf16_into(
                ctx,
                &gate_up
                    .view(token * gate_up_bytes, gate_up_bytes)
                    .map_err(Error::Cuda)?,
                &hidden
                    .view(token * hidden_bytes, hidden_bytes)
                    .map_err(Error::Cuda)?,
                INTERMEDIATE,
            )?;
        }
        workspace.down_weight.prepare(ctx, weights.down.view())?;
        gemm::w4a16_marlin_write(
            ctx,
            &workspace.hidden,
            workspace.down_weight.view(),
            output,
            &workspace.kernel,
        )
    }

    fn prepare_mlp_marlin64_layer_major(
        &self,
        ctx: &CudaContext,
        weights: &MlpWeights,
    ) -> Result<()> {
        let workspace = &self
            .marlin_prefill
            .as_ref()
            .ok_or_else(|| {
                Error::Other("Qwen3.5 Marlin M64 prefill workspace is not enabled".into())
            })?
            .mlp;
        workspace
            .gate_up_weight
            .prepare(ctx, weights.gate_up.view())?;
        workspace.down_weight.prepare(ctx, weights.down.view())
    }

    fn forward_mlp_marlin64_layer_major(
        &self,
        ctx: &CudaContext,
        input: &Tensor,
        output: &Tensor,
    ) -> Result<()> {
        let workspace = &self
            .marlin_prefill
            .as_ref()
            .ok_or_else(|| {
                Error::Other("Qwen3.5 Marlin M64 prefill workspace is not enabled".into())
            })?
            .mlp;
        gemm::w4a16_marlin_write(
            ctx,
            input,
            workspace.gate_up_weight.view(),
            &workspace.gate_up,
            &workspace.kernel,
        )?;
        let gate_up = CudaBuffer::from_tensor(&workspace.gate_up).map_err(Error::Cuda)?;
        let hidden = CudaBuffer::from_tensor(&workspace.hidden).map_err(Error::Cuda)?;
        let gate_up_bytes = 2 * INTERMEDIATE * DType::BF16.size_in_bytes();
        let hidden_bytes = INTERMEDIATE * DType::BF16.size_in_bytes();
        for token in 0..MARLIN_PREFILL_TILE {
            activation::silu_mul_bf16_into(
                ctx,
                &gate_up
                    .view(token * gate_up_bytes, gate_up_bytes)
                    .map_err(Error::Cuda)?,
                &hidden
                    .view(token * hidden_bytes, hidden_bytes)
                    .map_err(Error::Cuda)?,
                INTERMEDIATE,
            )?;
        }
        gemm::w4a16_marlin_write(
            ctx,
            &workspace.hidden,
            workspace.down_weight.view(),
            output,
            &workspace.kernel,
        )
    }

    fn forward_gdn_prefill8(
        &self,
        ctx: &CudaContext,
        weights: &GdnWeights,
        state: &GdnState,
        input: &Tensor,
        output: &Tensor,
    ) -> Result<()> {
        let workspace = &self.prefill.gdn;
        gemm::w4a16_m8_write(ctx, input, weights.qkv.view(), &workspace.qkv)?;
        gemm::w4a16_m8_write(ctx, input, weights.z.view(), &workspace.z)?;
        bf16_linear_serial_rows(
            ctx,
            input,
            &weights.ab,
            &workspace.ab,
            PREFILL_TILE,
            HIDDEN,
            2 * GDN_HEADS,
        )?;
        qwen35_conv4_prepare_m8_write(
            ctx,
            &workspace.qkv,
            &weights.conv,
            &state.conv,
            &workspace.ab,
            &weights.a_log,
            &weights.dt_bias,
            &workspace.a,
            &workspace.b,
            &workspace.query,
            &workspace.key,
            &workspace.value,
            &workspace.g,
            &workspace.beta,
        )?;
        qwen35_recurrent_m8_write(
            ctx,
            &workspace.query,
            &workspace.key,
            &workspace.value,
            &workspace.g,
            &workspace.beta,
            &state.recurrent,
            &workspace.core,
        )?;
        qwen35_gated_rmsnorm_m8_write(
            ctx,
            &workspace.core,
            &workspace
                .z
                .reshape(vec![PREFILL_TILE, GDN_HEADS, GDN_DIM])?,
            &weights.norm,
            &workspace.normalized,
            RMS_EPSILON,
        )?;
        let normalized = workspace
            .normalized
            .reshape(vec![PREFILL_TILE, GDN_VALUE_WIDTH])?;
        match &weights.out {
            GdnOutputWeight::Bf16 { weight, .. } => bf16_linear_serial_rows(
                ctx,
                &normalized,
                weight,
                output,
                PREFILL_TILE,
                GDN_VALUE_WIDTH,
                HIDDEN,
            ),
            GdnOutputWeight::W4(weight) => {
                gemm::w4a16_m8_write(ctx, &normalized, weight.view(), output)
            }
        }
    }

    fn forward_attention_prefill8(
        &self,
        ctx: &CudaContext,
        weights: &AttentionWeights,
        state: &AttentionState,
        input: &Tensor,
        output: &Tensor,
        start_position: usize,
        positions_offset: usize,
    ) -> Result<()> {
        let workspace = &self.prefill.attention;
        gemm::w4a16_m8_write(ctx, input, weights.q.view(), &workspace.q_projection)?;
        gemm::w4a16_m8_write(ctx, input, weights.k.view(), &workspace.k_projection)?;
        gemm::w4a16_m8_write(ctx, input, weights.v.view(), &workspace.v_projection)?;
        qwen35_attention::prepare_m8_write(
            ctx,
            &workspace.q_projection,
            &workspace.k_projection,
            &workspace.v_projection,
            &weights.q_norm,
            &weights.k_norm,
            &workspace.query,
            &workspace.key,
            &workspace.value,
            &workspace.gate,
            self.prefill_rope_positions
                .address_at(positions_offset * 3 * 4, PREFILL_TILE * 3 * 4)
                .map_err(Error::Cuda)?,
        )?;
        let key_cache = CudaBuffer::from_tensor(&state.key_cache).map_err(Error::Cuda)?;
        let value_cache = CudaBuffer::from_tensor(&state.value_cache).map_err(Error::Cuda)?;
        cache::append(
            ctx,
            &key_cache,
            &workspace.key,
            ATTN_KV_HEADS,
            ATTN_HEAD_DIM,
            self.max_seq_len,
            start_position,
            PREFILL_TILE,
        )?;
        cache::append(
            ctx,
            &value_cache,
            &workspace.value,
            ATTN_KV_HEADS,
            ATTN_HEAD_DIM,
            self.max_seq_len,
            start_position,
            PREFILL_TILE,
        )?;
        let query = CudaBuffer::from_tensor(&workspace.query).map_err(Error::Cuda)?;
        let attended = CudaBuffer::from_tensor(&workspace.attended).map_err(Error::Cuda)?;
        let row_bytes = ATTN_WIDTH * DType::BF16.size_in_bytes();
        for token in 0..PREFILL_TILE {
            let position = self
                .prefill_positions
                .address_at((positions_offset + token) * 4, 4)
                .map_err(Error::Cuda)?;
            let kv_len = start_position + token + 1;
            let query_row = query
                .view(token * row_bytes, row_bytes)
                .map_err(Error::Cuda)?;
            let attended_row = attended
                .view(token * row_bytes, row_bytes)
                .map_err(Error::Cuda)?;
            if let Some(split) = qwen35_attention::split_cta_candidate_for_bucket(kv_len) {
                qwen35_attention::flash_split_cta_buffer_write(
                    ctx,
                    &query_row,
                    &key_cache,
                    &value_cache,
                    &attended_row,
                    &self.workspace.attention.split,
                    split,
                    kv_len,
                    self.max_seq_len,
                    ATTENTION_SCALE,
                    position,
                )?;
            } else {
                attention::flash_bf16_into(
                    ctx,
                    &query_row,
                    &key_cache,
                    &value_cache,
                    &attended_row,
                    ATTN_Q_HEADS,
                    ATTN_KV_HEADS,
                    ATTN_HEAD_DIM,
                    kv_len,
                    self.max_seq_len,
                    ATTENTION_SCALE,
                    position,
                )?;
            }
        }
        qwen35_attention::gate_m8_write(
            ctx,
            &workspace.attended,
            &workspace.gate,
            &workspace.gated,
        )?;
        gemm::w4a16_m8_write(
            ctx,
            &workspace.gated.reshape(vec![PREFILL_TILE, ATTN_WIDTH])?,
            weights.o.view(),
            output,
        )
    }

    fn forward_mlp_prefill8(
        &self,
        ctx: &CudaContext,
        weights: &MlpWeights,
        input: &Tensor,
        output: &Tensor,
    ) -> Result<()> {
        let workspace = &self.prefill.mlp;
        gemm::w4a16_m8_write(ctx, input, weights.gate_up.view(), &workspace.gate_up)?;
        let gate_up = CudaBuffer::from_tensor(&workspace.gate_up).map_err(Error::Cuda)?;
        let hidden = CudaBuffer::from_tensor(&workspace.hidden).map_err(Error::Cuda)?;
        let gate_up_bytes = 2 * INTERMEDIATE * DType::BF16.size_in_bytes();
        let hidden_bytes = INTERMEDIATE * DType::BF16.size_in_bytes();
        for token in 0..PREFILL_TILE {
            activation::silu_mul_bf16_into(
                ctx,
                &gate_up
                    .view(token * gate_up_bytes, gate_up_bytes)
                    .map_err(Error::Cuda)?,
                &hidden
                    .view(token * hidden_bytes, hidden_bytes)
                    .map_err(Error::Cuda)?,
                INTERMEDIATE,
            )?;
        }
        gemm::w4a16_m8_write(ctx, &workspace.hidden, weights.down.view(), output)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_gdn(
        &self,
        ctx: &CudaContext,
        weights: &GdnWeights,
        state: &GdnState,
        mode: HybridUnitMode,
        input: &Tensor,
        output: &Tensor,
        profile: bool,
    ) -> Result<()> {
        let workspace = &self.workspace.gdn;
        gemm::w4a16_write(ctx, input, weights.qkv.view(), &workspace.qkv)?;
        gemm::w4a16_write(ctx, input, weights.z.view(), &workspace.z)?;
        bf16_linear(
            ctx,
            input,
            &weights.ab,
            &workspace.ab,
            HIDDEN,
            2 * GDN_HEADS,
        )?;
        qwen35_conv4_prepare_write(
            ctx,
            &workspace.qkv.reshape(vec![GDN_CONV_DIM])?,
            &weights.conv,
            &state.conv,
            &workspace.ab.reshape(vec![2 * GDN_HEADS])?,
            &weights.a_log,
            &weights.dt_bias,
            &workspace.a.reshape(vec![GDN_HEADS])?,
            &workspace.b.reshape(vec![GDN_HEADS])?,
            &workspace.query,
            &workspace.key,
            &workspace.value,
            &workspace.g,
            &workspace.beta,
        )?;
        qwen35_recurrent_write(
            ctx,
            &workspace.query,
            &workspace.key,
            &workspace.value,
            &workspace.g,
            &workspace.beta,
            &state.recurrent,
            &workspace.core,
        )?;
        qwen35_gated_rmsnorm_write(
            ctx,
            &workspace.core,
            &workspace.z.reshape(vec![GDN_HEADS, GDN_DIM])?,
            &weights.norm,
            &workspace.normalized,
            RMS_EPSILON,
        )?;
        let normalized = workspace.normalized.reshape(vec![1, GDN_VALUE_WIDTH])?;
        match &weights.out {
            GdnOutputWeight::Bf16 { weight, w8 } => {
                let _range = profile.then(|| {
                    apxinf_cuda::nvtx::range(if mode == HybridUnitMode::LayerOptimized {
                        "qwen35.hybrid_unit.gdn_out_w8"
                    } else {
                        "qwen35.hybrid_unit.gdn_out_bf16"
                    })
                });
                match mode {
                    HybridUnitMode::LayerOptimized => {
                        gemm::w8a16_write(ctx, &normalized, w8.view(), output)
                    }
                    HybridUnitMode::Native | HybridUnitMode::ModelOptimized => {
                        bf16_linear(ctx, &normalized, weight, output, GDN_VALUE_WIDTH, HIDDEN)
                    }
                }
            }
            GdnOutputWeight::W4(weight) => {
                let _range =
                    profile.then(|| apxinf_cuda::nvtx::range("qwen35.hybrid_unit.gdn_out_w4"));
                gemm::w4a16_write(ctx, &normalized, weight.view(), output)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_attention(
        &self,
        ctx: &CudaContext,
        weights: &AttentionWeights,
        state: &AttentionState,
        mode: HybridUnitMode,
        input: &Tensor,
        output: &Tensor,
        bucket_kv_len: usize,
        cache_position: CudaDeviceAddress,
        rope_position: CudaDeviceAddress,
        profile: bool,
    ) -> Result<()> {
        let workspace = &self.workspace.attention;
        gemm::w4a16_write(ctx, input, weights.q.view(), &workspace.q_projection)?;
        gemm::w4a16_write(ctx, input, weights.k.view(), &workspace.k_projection)?;
        gemm::w4a16_write(ctx, input, weights.v.view(), &workspace.v_projection)?;
        qwen35_attention::prepare_write(
            ctx,
            &workspace.q_projection,
            &workspace.k_projection,
            &workspace.v_projection,
            &weights.q_norm,
            &weights.k_norm,
            &workspace.query,
            &workspace.key,
            &workspace.value,
            &workspace.gate,
            rope_position,
        )?;
        let key_cache = CudaBuffer::from_tensor(&state.key_cache).map_err(Error::Cuda)?;
        let value_cache = CudaBuffer::from_tensor(&state.value_cache).map_err(Error::Cuda)?;
        cache::append_at(
            ctx,
            DType::BF16,
            &key_cache,
            &CudaBuffer::from_tensor(&workspace.key).map_err(Error::Cuda)?,
            ATTN_KV_HEADS,
            ATTN_HEAD_DIM,
            self.max_seq_len,
            cache_position,
        )?;
        cache::append_at(
            ctx,
            DType::BF16,
            &value_cache,
            &CudaBuffer::from_tensor(&workspace.value).map_err(Error::Cuda)?,
            ATTN_KV_HEADS,
            ATTN_HEAD_DIM,
            self.max_seq_len,
            cache_position,
        )?;
        let split = (mode != HybridUnitMode::Native)
            .then(|| qwen35_attention::split_cta_candidate_for_bucket(bucket_kv_len))
            .flatten();
        let _range = profile.then(|| {
            apxinf_cuda::nvtx::range(if split.is_some() {
                "qwen35.hybrid_unit.attention_split16"
            } else {
                "qwen35.hybrid_unit.attention_incumbent"
            })
        });
        if let Some(split) = split {
            qwen35_attention::flash_split_cta_write(
                ctx,
                &workspace.query,
                &key_cache,
                &value_cache,
                &workspace.attended,
                &workspace.split,
                split,
                bucket_kv_len,
                self.max_seq_len,
                ATTENTION_SCALE,
                cache_position,
            )?;
        } else {
            attention::flash_bf16_into(
                ctx,
                &CudaBuffer::from_tensor(&workspace.query).map_err(Error::Cuda)?,
                &key_cache,
                &value_cache,
                &CudaBuffer::from_tensor(&workspace.attended).map_err(Error::Cuda)?,
                ATTN_Q_HEADS,
                ATTN_KV_HEADS,
                ATTN_HEAD_DIM,
                bucket_kv_len,
                self.max_seq_len,
                ATTENTION_SCALE,
                cache_position,
            )?;
        }
        qwen35_attention::gate_write(ctx, &workspace.attended, &workspace.gate, &workspace.gated)?;
        gemm::w4a16_write(ctx, &workspace.gated_flat, weights.o.view(), output)
    }

    fn forward_mlp(
        &self,
        ctx: &CudaContext,
        weights: &MlpWeights,
        input: &Tensor,
        output: &Tensor,
        _profile: bool,
    ) -> Result<()> {
        gemm::w4a16_write(
            ctx,
            input,
            weights.gate_up.view(),
            &self.workspace.mlp.gate_up,
        )?;
        activation::silu_mul_bf16_into(
            ctx,
            &CudaBuffer::from_tensor(&self.workspace.mlp.gate_up).map_err(Error::Cuda)?,
            &CudaBuffer::from_tensor(&self.workspace.mlp.hidden).map_err(Error::Cuda)?,
            INTERMEDIATE,
        )?;
        gemm::w4a16_write(ctx, &self.workspace.mlp.hidden, weights.down.view(), output)
    }
}

fn load_w4_cpu(
    manifest: &CheckpointManifest,
    base: &str,
) -> Result<(Tensor, Tensor, Tensor, usize, usize)> {
    let packed = load_tensor(manifest, &format!("{base}.weight_packed"))?;
    let scales = load_tensor(manifest, &format!("{base}.weight_scale"))?;
    let zero_points = load_tensor(manifest, &format!("{base}.weight_zero_point"))?;
    let shape = safetensors::read_small_i64(
        manifest
            .tensor(&format!("{base}.weight_shape"))
            .ok_or_else(|| Error::Other(format!("missing `{base}.weight_shape`")))?,
    )
    .map_err(Error::Other)?;
    if shape.len() != 2 || shape.iter().any(|dimension| *dimension <= 0) {
        return Err(Error::Other(format!(
            "invalid W4 logical shape for `{base}`: {shape:?}"
        )));
    }
    Ok((
        packed,
        scales,
        zero_points,
        shape[1] as usize,
        shape[0] as usize,
    ))
}

fn load_tensor(manifest: &CheckpointManifest, name: &str) -> Result<Tensor> {
    safetensors::load_manifest_tensor(
        manifest
            .tensor(name)
            .ok_or_else(|| Error::Other(format!("missing `{name}`")))?,
    )
    .map_err(Error::Other)
}

fn load_gpu_bf16(manifest: &CheckpointManifest, name: &str) -> Result<Tensor> {
    let tensor = load_tensor(manifest, name)?;
    if tensor.dtype() != DType::BF16 || tensor.device() != Device::Cpu {
        return Err(Error::Other(format!(
            "Qwen3.5 tensor `{name}` must be CPU BF16"
        )));
    }
    transfers::to_cuda(&tensor, 0)
}

fn require_cpu_bf16_shape(name: &str, tensor: &Tensor, shape: &[usize]) -> Result<()> {
    if tensor.device() != Device::Cpu
        || tensor.dtype() != DType::BF16
        || tensor.shape().dims() != shape
    {
        return Err(Error::Other(format!(
            "{name} must be CPU BF16 {shape:?}, got {} {:?} on {}",
            tensor.dtype(),
            tensor.shape().dims(),
            tensor.device()
        )));
    }
    Ok(())
}

fn gpu_zeros(shape: &[usize], dtype: DType) -> Result<Tensor> {
    transfers::to_cuda(&Tensor::zeros(shape.to_vec(), dtype), 0)
}

fn cuda_row_view(tensor: &Tensor, first_row: usize, rows: usize) -> Result<Tensor> {
    let dims = tensor.shape().dims();
    if tensor.device() != Device::Cuda(0)
        || dims.is_empty()
        || rows == 0
        || first_row.checked_add(rows).is_none_or(|end| end > dims[0])
    {
        return Err(Error::Other(
            "Qwen3.5 CUDA row-view contract mismatch".into(),
        ));
    }
    let row_elements = tensor.numel() / dims[0];
    let row_bytes = row_elements
        .checked_mul(tensor.dtype().size_in_bytes())
        .ok_or_else(|| Error::Other("Qwen3.5 CUDA row-view size overflow".into()))?;
    let byte_offset = first_row
        .checked_mul(row_bytes)
        .ok_or_else(|| Error::Other("Qwen3.5 CUDA row-view offset overflow".into()))?;
    let byte_len = rows
        .checked_mul(row_bytes)
        .ok_or_else(|| Error::Other("Qwen3.5 CUDA row-view length overflow".into()))?;
    let mut shape = dims.to_vec();
    shape[0] = rows;
    Ok(CudaBuffer::from_tensor(tensor)
        .map_err(Error::Cuda)?
        .view(byte_offset, byte_len)
        .map_err(Error::Cuda)?
        .into_tensor(apxinf_core::Shape::new(shape), tensor.dtype()))
}

fn bf16_linear(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    output: &Tensor,
    input_dim: usize,
    output_dim: usize,
) -> Result<()> {
    if input.dtype() != DType::BF16
        || input.shape().dims() != [1, input_dim]
        || weight.dtype() != DType::BF16
        || weight.shape().dims() != [output_dim, input_dim]
        || output.dtype() != DType::BF16
        || output.shape().dims() != [1, output_dim]
    {
        return Err(Error::Other(
            "Qwen3.5 BF16 decode linear contract mismatch".into(),
        ));
    }
    gemm::write_ex(
        ctx,
        DType::BF16,
        CublasTranspose::None,
        CublasTranspose::Transpose,
        1,
        output_dim,
        input_dim,
        1.0,
        &CudaBuffer::from_tensor(input).map_err(Error::Cuda)?,
        input_dim as i32,
        &CudaBuffer::from_tensor(weight).map_err(Error::Cuda)?,
        input_dim as i32,
        0.0,
        &CudaBuffer::from_tensor(output).map_err(Error::Cuda)?,
        output_dim as i32,
    )
}

fn bf16_linear_prefill_m8(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    output: &Tensor,
    input_dim: usize,
    output_dim: usize,
) -> Result<()> {
    if input.dtype() != DType::BF16
        || input.shape().dims() != [PREFILL_TILE, input_dim]
        || weight.dtype() != DType::BF16
        || weight.shape().dims() != [output_dim, input_dim]
        || output.dtype() != DType::BF16
        || output.shape().dims() != [PREFILL_TILE, output_dim]
    {
        return Err(Error::Other(
            "Qwen3.5 BF16 direct M8 linear contract mismatch".into(),
        ));
    }
    gemm::write_ex(
        ctx,
        DType::BF16,
        CublasTranspose::None,
        CublasTranspose::Transpose,
        PREFILL_TILE,
        output_dim,
        input_dim,
        1.0,
        &CudaBuffer::from_tensor(input).map_err(Error::Cuda)?,
        input_dim as i32,
        &CudaBuffer::from_tensor(weight).map_err(Error::Cuda)?,
        input_dim as i32,
        0.0,
        &CudaBuffer::from_tensor(output).map_err(Error::Cuda)?,
        output_dim as i32,
    )
}

fn bf16_linear_serial_rows(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    output: &Tensor,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
) -> Result<()> {
    if !(1..=PREFILL_TILE).contains(&rows)
        || input.dtype() != DType::BF16
        || input.shape().dims() != [rows, input_dim]
        || weight.dtype() != DType::BF16
        || weight.shape().dims() != [output_dim, input_dim]
        || output.dtype() != DType::BF16
        || output.shape().dims() != [rows, output_dim]
    {
        return Err(Error::Other(
            "Qwen3.5 BF16 serial-row linear contract mismatch".into(),
        ));
    }
    let input = CudaBuffer::from_tensor(input).map_err(Error::Cuda)?;
    let weight = CudaBuffer::from_tensor(weight).map_err(Error::Cuda)?;
    let output = CudaBuffer::from_tensor(output).map_err(Error::Cuda)?;
    let input_bytes = input_dim * DType::BF16.size_in_bytes();
    let output_bytes = output_dim * DType::BF16.size_in_bytes();
    for row in 0..rows {
        gemm::write_ex(
            ctx,
            DType::BF16,
            CublasTranspose::None,
            CublasTranspose::Transpose,
            1,
            output_dim,
            input_dim,
            1.0,
            &input
                .view(row * input_bytes, input_bytes)
                .map_err(Error::Cuda)?,
            input_dim as i32,
            &weight,
            input_dim as i32,
            0.0,
            &output
                .view(row * output_bytes, output_bytes)
                .map_err(Error::Cuda)?,
            output_dim as i32,
        )?;
    }
    Ok(())
}
