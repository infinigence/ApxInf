//! One BF16 Blocks implementation: vision, hybrid backbone and planning expert.
//! Layer mathematics and physical state descriptions; no graph capture policy.
use super::{GdnExecution, GdnRequest};
use crate::qwen_drive::backend::{
    kernels, transfers, Context, CublasTranspose, DeviceBuffer, RuntimeBackend,
};
use crate::qwen_drive::config::{ProjectionLayout, QwenDriveConfig};
use crate::qwen_drive::diagnostics_enabled;
use crate::qwen_drive::weights::bf16::{BackboneDeviceWeights, ExpertDeviceWeights, MixerWeights};
use apxinf_core::{DType, Device, Error, Result, Shape, Tensor};
use kernels::{activation, attention, elementwise, embedding, gemm, linear_attention as la};
use std::path::Path;
use std::sync::Arc;
const GDN_CHUNK: usize = 64;
fn gdn_stage_mark(
    ctx: &Context,
    layer_idx: usize,
    name: &str,
    since: &mut std::time::Instant,
) -> Result<()> {
    ctx.synchronize().map_err(Error::Cuda)?;
    qdiag!(
        "[qwen_drive] gdn_stage layer={} {} ms={:.2}",
        layer_idx,
        name,
        since.elapsed().as_secs_f64() * 1000.0
    );
    *since = std::time::Instant::now();
    Ok(())
}
pub(crate) struct BackboneBf16 {
    pub config: QwenDriveConfig,
    pub cuda: Arc<RuntimeBackend>,
    pub weights: BackboneDeviceWeights,
}
/// Planning expert computation; independent of backbone weights and request state.
pub(crate) struct PlannerBf16 {
    pub config: QwenDriveConfig,
    pub cuda: Arc<RuntimeBackend>,
    pub weights: ExpertDeviceWeights,
}
impl PlannerBf16 {
    pub fn prepare(&self, input: &expert::ExpertPlan<'_>) -> Result<expert::ExpertState> {
        expert::prepare(&self.config, &self.weights, self.cuda.context(), input)
    }
    pub fn step(
        &self,
        input: &expert::ExpertPlan<'_>,
        state: &expert::ExpertState,
        index: usize,
    ) -> Result<()> {
        expert::step(
            &self.config,
            &self.weights,
            self.cuda.context(),
            input,
            state,
            index,
        )
    }
    pub fn output(&self, state: expert::ExpertState) -> Tensor {
        expert::output(state)
    }
}

pub(crate) enum LayerCache {
    FullAttention {
        k: Tensor,
        v: Tensor,
    },
    Gdn {
        conv_state_a: Tensor,
        conv_state_b: Tensor,
        /// false -> a is current, b is scratch; true -> b is current.
        flip: bool,
        recurrent: Tensor,
    },
}

#[derive(Default)]
pub(crate) struct VisionState {
    pos_table_host: std::sync::OnceLock<Vec<f32>>,
    pos_embed_cache: std::sync::Mutex<Vec<(Vec<[u32; 3]>, Tensor)>>,
}

pub(crate) struct BackboneState {
    vision: std::rc::Rc<VisionState>,
    pub caches: Vec<LayerCache>,
    pub cache_len: usize,
    pub rope_delta: i64,
    pub max_seq_len: usize,
    pub last_position: i64,
    pub decode_step: Option<usize>,
}
pub(crate) fn trace_rows(name: &str, tensor: &Tensor) -> Result<()> {
    let Some(root) = std::env::var_os("APXINF_QWEN_TRACE_DIR") else {
        return Ok(());
    };
    let width = *tensor.shape().dims().last().unwrap();
    let count = tensor.numel() / width;
    // Four sampled rows cannot tell a layer that is itself nondeterministic
    // from one that merely mixes in a neighbour's unsampled row, and every
    // sequence mixer here does mix across positions. APXINF_QWEN_TRACE_FULL
    // dumps the whole tensor so the first affected layer is the real one.
    let full = std::env::var_os("APXINF_QWEN_TRACE_FULL").is_some();
    let rows = if full || count < 4 {
        (0..count).collect::<Vec<_>>()
    } else {
        vec![0, 1, 2, count - 1]
    };
    let buffer = DeviceBuffer::from_tensor(tensor).map_err(Error::Cuda)?;
    let mut bytes = Vec::new();
    for row in rows {
        let view = buffer
            .view(
                row * width * tensor.dtype().size_in_bytes(),
                width * tensor.dtype().size_in_bytes(),
            )
            .map_err(Error::Cuda)?;
        let t = view
            .as_tensor(Shape::new(vec![1, width]), tensor.dtype())
            .map_err(Error::Cuda)?;
        let values = transfers::to_cpu(&t)?
            .to_f32_vec()
            .map_err(|e| Error::Other(e.to_string()))?;
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    let root = Path::new(&root);
    std::fs::create_dir_all(root).map_err(|e| Error::Other(e.to_string()))?;
    std::fs::write(root.join(format!("{name}.f32")), bytes)
        .map_err(|e| Error::Other(e.to_string()))?;
    Ok(())
}
fn bf16_round(value: f32) -> f32 {
    half::bf16::from_f32(value).to_f32()
}

fn device_tensor(ctx: &Context, shape: &[usize], dtype: DType) -> Result<Tensor> {
    let elements: usize = shape.iter().product();
    let bytes = elements
        .checked_mul(dtype.size_in_bytes())
        .ok_or_else(|| Error::Other("qwen_drive: tensor size overflow".into()))?;
    let buffer = kernels::scratch_buffer(ctx, bytes.max(1))?;
    buffer
        .as_tensor(Shape::new(shape.to_vec()), dtype)
        .map_err(Error::Cuda)
}

fn alloc_zeros(ctx: &Context, bytes: usize) -> Result<DeviceBuffer> {
    kernels::scratch_buffer_zeroed(ctx, bytes.max(1))
}

fn alloc_scan_scratch(ctx: &Context, bytes: usize, padded: bool) -> Result<DeviceBuffer> {
    if padded {
        alloc_zeros(ctx, bytes)
    } else {
        kernels::scratch_buffer(ctx, bytes.max(1))
    }
}

fn linear_checkpoint(
    layout: ProjectionLayout,
    ctx: &Context,
    input: &Tensor,
    weight: &Tensor,
) -> Result<Tensor> {
    // Under the gate the loader has already stored this weight as [in, out],
    // so the raw path's [out, in] check does not apply to it.
    if layout == ProjectionLayout::Tuned {
        return gemm::bf16(ctx, input, weight);
    }
    let x = input.shape().dims();
    let w = weight.shape().dims();
    if x.len() != 2
        || w.len() != 2
        || x[1] != w[1]
        || input.dtype() != DType::BF16
        || weight.dtype() != DType::BF16
    {
        return Err(Error::Other(
            "qwen_drive: checkpoint linear shape/dtype mismatch".into(),
        ));
    }
    let stride =
        i32::try_from(x[1]).map_err(|_| Error::Other("linear input stride overflow".into()))?;
    let columns =
        i32::try_from(w[0]).map_err(|_| Error::Other("linear output stride overflow".into()))?;
    let output = device_tensor(ctx, &[x[0], w[0]], DType::BF16)?;
    gemm::write_ex(
        ctx,
        DType::BF16,
        CublasTranspose::None,
        CublasTranspose::Transpose,
        x[0],
        w[0],
        x[1],
        1.0,
        &DeviceBuffer::from_tensor(input).map_err(Error::Cuda)?,
        stride,
        &DeviceBuffer::from_tensor(weight).map_err(Error::Cuda)?,
        stride,
        0.0,
        &DeviceBuffer::from_tensor(&output).map_err(Error::Cuda)?,
        columns,
    )?;
    Ok(output)
}

fn project_and_pack(
    layout: ProjectionLayout,
    ctx: &Context,
    input: &Tensor,
    weights: &[&Tensor],
) -> Result<Tensor> {
    let rows = input.shape().dims()[0];
    let mut outputs = Vec::with_capacity(weights.len());
    for weight in weights {
        let value = linear_checkpoint(layout, ctx, input, weight)?;
        outputs.push(value.reshape(vec![rows, value.shape().dims()[1], 1, 1])?);
    }
    let packed = elementwise::concat_channels_bf16(ctx, &outputs.iter().collect::<Vec<_>>())?;
    packed.reshape(vec![rows, packed.shape().dims()[1]])
}

pub(crate) fn upload_u32(ctx: &Context, values: &[u32]) -> Result<DeviceBuffer> {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    let buffer = DeviceBuffer::alloc(bytes.len().max(1), ctx.device_id()).map_err(Error::Cuda)?;
    buffer.copy_from_host(&bytes).map_err(Error::Cuda)?;
    Ok(buffer)
}

fn cache_view(cache: &Tensor, kv_len: usize) -> Result<Tensor> {
    let dims = cache.shape().dims().to_vec();
    if dims.len() != 3 || kv_len == 0 || kv_len > dims[0] {
        return Err(Error::Other("qwen_drive: cache view shape mismatch".into()));
    }
    let buffer = DeviceBuffer::from_tensor(cache).map_err(Error::Cuda)?;
    let bytes = kv_len * dims[1] * dims[2] * DType::BF16.size_in_bytes();
    let view = buffer.view(0, bytes).map_err(Error::Cuda)?;
    view.as_tensor(Shape::new(vec![kv_len, dims[1], dims[2]]), DType::BF16)
        .map_err(Error::Cuda)
}
impl BackboneBf16 {
    pub(crate) fn fresh_caches(
        config: &QwenDriveConfig,
        cuda: &RuntimeBackend,
        max_seq_len: usize,
    ) -> Result<Vec<LayerCache>> {
        let text = &config.text;
        let device = cuda.device_id();
        let mut caches = Vec::with_capacity(text.n_layers);
        for index in 0..text.n_layers {
            if text.is_full_attention(index) {
                let bytes =
                    max_seq_len * text.n_kv_heads * text.head_dim * DType::BF16.size_in_bytes();
                let k = DeviceBuffer::alloc_zeros(bytes, device).map_err(Error::Cuda)?;
                let v = DeviceBuffer::alloc_zeros(bytes, device).map_err(Error::Cuda)?;
                let shape = Shape::new(vec![max_seq_len, text.n_kv_heads, text.head_dim]);
                caches.push(LayerCache::FullAttention {
                    k: k.as_tensor(shape.clone(), DType::BF16)
                        .map_err(Error::Cuda)?,
                    v: v.as_tensor(shape, DType::BF16).map_err(Error::Cuda)?,
                });
            } else {
                let conv_dim = 2 * text.linear_num_key_heads * text.linear_key_head_dim
                    + text.linear_num_value_heads * text.linear_value_head_dim;
                let conv_bytes =
                    conv_dim * text.linear_conv_kernel_dim * DType::BF16.size_in_bytes();
                let conv_shape = Shape::new(vec![conv_dim, text.linear_conv_kernel_dim]);
                let conv_a = DeviceBuffer::alloc_zeros(conv_bytes, device).map_err(Error::Cuda)?;
                let conv_b = DeviceBuffer::alloc_zeros(conv_bytes, device).map_err(Error::Cuda)?;
                let rec_bytes = text.linear_num_value_heads
                    * text.linear_key_head_dim
                    * text.linear_value_head_dim
                    * DType::F32.size_in_bytes();
                let recurrent =
                    DeviceBuffer::alloc_zeros(rec_bytes, device).map_err(Error::Cuda)?;
                caches.push(LayerCache::Gdn {
                    conv_state_a: conv_a
                        .as_tensor(conv_shape.clone(), DType::BF16)
                        .map_err(Error::Cuda)?,
                    conv_state_b: conv_b
                        .as_tensor(conv_shape, DType::BF16)
                        .map_err(Error::Cuda)?,
                    flip: false,
                    recurrent: recurrent
                        .as_tensor(
                            Shape::new(vec![
                                text.linear_num_value_heads,
                                text.linear_key_head_dim,
                                text.linear_value_head_dim,
                            ]),
                            DType::F32,
                        )
                        .map_err(Error::Cuda)?,
                });
            }
        }
        Ok(caches)
    }
    pub(crate) fn ctx(&self) -> &Context {
        self.cuda.context()
    }
    pub(crate) fn rope_index(
        &self,
        token_ids: &[u32],
        grid_thw: &[[u32; 3]],
    ) -> Result<Vec<[u32; 3]>> {
        let merge = self.config.vision.spatial_merge_size as u32;
        let image_tok = self.config.image_token_id;
        let mut out: Vec<[u32; 3]> = Vec::with_capacity(token_ids.len());
        let mut grids = grid_thw.iter();
        let mut current_pos: u32 = 0;
        let mut index = 0usize;
        while index < token_ids.len() {
            if token_ids[index] == image_tok {
                let start = index;
                while index < token_ids.len() && token_ids[index] == image_tok {
                    index += 1;
                }
                let run = index - start;
                let grid = grids.next().ok_or_else(|| {
                    Error::Other("qwen_drive: more image-token runs than image grids".into())
                })?;
                let (t, h, w) = (grid[0], grid[1] / merge, grid[2] / merge);
                if run != (t * h * w) as usize {
                    return Err(Error::Other(format!(
                        "qwen_drive: image-token run of {run} != grid tokens {}",
                        t * h * w
                    )));
                }
                for ti in 0..t {
                    for hi in 0..h {
                        for wi in 0..w {
                            out.push([current_pos + ti, current_pos + hi, current_pos + wi]);
                        }
                    }
                }
                current_pos += grid[1].max(grid[2]) / merge;
            } else {
                out.push([current_pos, current_pos, current_pos]);
                current_pos += 1;
                index += 1;
            }
        }
        if grids.next().is_some() {
            return Err(Error::Other(
                "qwen_drive: more image grids than image-token runs".into(),
            ));
        }
        Ok(out)
    }
    pub(crate) fn mrope_tables(&self, positions: &[[u32; 3]]) -> Result<(Tensor, Tensor)> {
        let rotary = self.config.text.rotary_dim();
        let pairs = rotary / 2;
        let theta = self.config.text.rope_theta;
        let section = self.config.text.mrope_section;
        let mut inv_freq = vec![0.0f32; pairs];
        for (i, slot) in inv_freq.iter_mut().enumerate() {
            *slot = 1.0 / theta.powf(2.0 * i as f32 / rotary as f32);
        }
        let mut cos = Vec::with_capacity(positions.len() * rotary);
        let mut sin = Vec::with_capacity(positions.len() * rotary);
        for token in positions {
            let mut merged = vec![0.0f32; pairs];
            for p in 0..pairs {
                let axis = if p % 3 == 1 && p < section[1] * 3 {
                    1
                } else if p % 3 == 2 && p < section[2] * 3 {
                    2
                } else {
                    0
                };
                merged[p] = token[axis] as f32 * inv_freq[p];
            }
            let mut cos_row = vec![0.0f32; rotary];
            let mut sin_row = vec![0.0f32; rotary];
            for p in 0..pairs {
                cos_row[p] = bf16_round(merged[p].cos());
                cos_row[pairs + p] = cos_row[p];
                sin_row[p] = bf16_round(merged[p].sin());
                sin_row[pairs + p] = sin_row[p];
            }
            cos.extend_from_slice(&cos_row);
            sin.extend_from_slice(&sin_row);
        }
        let ctx = self.ctx();
        let cos_t = {
            let rounded: Vec<half::bf16> = cos.iter().map(|&v| half::bf16::from_f32(v)).collect();
            transfers::to_cuda(
                &Tensor::from_bf16(vec![positions.len(), rotary], &rounded)?,
                ctx.device_id(),
            )?
        };
        let sin_t = {
            let rounded: Vec<half::bf16> = sin.iter().map(|&v| half::bf16::from_f32(v)).collect();
            transfers::to_cuda(
                &Tensor::from_bf16(vec![positions.len(), rotary], &rounded)?,
                ctx.device_id(),
            )?
        };
        Ok((cos_t, sin_t))
    }
    pub(crate) fn lm_head(&self, x: &Tensor) -> Result<Tensor> {
        let ctx = self.ctx();
        let (m, k) = {
            let dims = x.shape().dims();
            if dims.len() != 2 {
                return Err(Error::Other("qwen_drive: lm_head input must be 2D".into()));
            }
            (dims[0], dims[1])
        };
        let vocab = self.config.text.vocab_size;
        if m == 1 {
            if let Some(file) = std::env::var_os("APXINF_QWEN_HEAD_INPUT") {
                let values = transfers::to_cpu(x)?
                    .to_f32_vec()
                    .map_err(|error| Error::Other(error.to_string()))?;
                let bytes: Vec<u8> = values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect();
                std::fs::write(file, bytes).map_err(|error| Error::Other(error.to_string()))?;
            }
        }
        let out = device_tensor(ctx, &[m, vocab], DType::BF16)?;
        let a = DeviceBuffer::from_tensor(x).map_err(Error::Cuda)?;
        let b = DeviceBuffer::from_tensor(&self.weights.embed_tokens).map_err(Error::Cuda)?;
        let c = DeviceBuffer::from_tensor(&out).map_err(Error::Cuda)?;
        gemm::write_ex(
            ctx,
            DType::BF16,
            CublasTranspose::None,
            CublasTranspose::Transpose,
            m,
            vocab,
            k,
            1.0,
            &a,
            k as i32,
            &b,
            k as i32,
            0.0,
            &c,
            vocab as i32,
        )?;
        Ok(out)
    }
    pub(crate) fn forward_mlp(
        &self,
        x: &Tensor,
        delta: &Tensor,
        post_norm: &Tensor,
        gate_up_w: &Tensor,
        down_w: &Tensor,
        trace: bool,
    ) -> Result<Tensor> {
        let ctx = self.ctx();
        let eps = self.config.text.rms_norm_eps;
        let (x, normed) = la::add_rms_norm_plus1(ctx, x, delta, post_norm, eps)?;
        if trace {
            trace_rows("text0_residual", &x)?;
        }
        let gu = gemm::bf16(ctx, &normed, gate_up_w)?;
        let act = activation::swiglu_bf16_rounded(ctx, &gu)?;
        let down = linear_checkpoint(self.weights.projection_layout, ctx, &act, down_w)?;
        if trace {
            trace_rows("text0_post_norm", &normed)?;
            trace_rows("text0_gate_up", &gu)?;
            trace_rows("text0_swiglu", &act)?;
            trace_rows("text0_down", &down)?;
        }
        elementwise::add(ctx, &x, &down)
    }
    pub(crate) fn forward_full_attention(
        &self,
        state: &mut BackboneState,
        x: Tensor,
        layer_idx: usize,
        cos: &Tensor,
        sin: &Tensor,
        seq: usize,
    ) -> Result<Tensor> {
        let cuda = Arc::clone(&self.cuda);
        let ctx = cuda.context();
        let text = &self.config.text;
        let w = match &self.weights.layers[layer_idx] {
            MixerWeights::FullAttention(w) => w,
            _ => return Err(Error::Other("qwen_drive: layer kind mismatch".into())),
        };
        let eps = text.rms_norm_eps;
        let heads = text.n_heads;
        let kv_heads = text.n_kv_heads;
        let head_dim = text.head_dim;
        let rotary = text.rotary_dim();
        let normed = la::rms_norm_plus1(ctx, &x, &w.input_norm, eps)?;
        let fused = project_and_pack(
            self.weights.projection_layout,
            ctx,
            &normed,
            &[&w.q_w, &w.k_w, &w.v_w],
        )?;
        if layer_idx == 3 && seq > 1 {
            trace_rows("text3_input_norm", &normed)?;
            trace_rows("text3_fused_qkv", &fused)?;
            trace_rows("text3_cos", cos)?;
            trace_rows("text3_sin", sin)?;
        }
        let q_out = device_tensor(ctx, &[seq, heads, head_dim], DType::BF16)?;
        let (k_cache, v_cache) = match &state.caches[layer_idx] {
            LayerCache::FullAttention { k, v } => (k.clone(), v.clone()),
            _ => return Err(Error::Other("qwen_drive: cache kind mismatch".into())),
        };
        la::full_attn_prepare(
            ctx,
            &fused,
            &w.q_norm,
            &w.k_norm,
            cos,
            sin,
            &q_out,
            &k_cache,
            &v_cache,
            state.cache_len,
            heads,
            kv_heads,
            head_dim,
            rotary,
            eps,
        )?;
        let kv_len = state.cache_len + seq;
        let k_view = cache_view(&k_cache, kv_len)?;
        let v_view = cache_view(&v_cache, kv_len)?;
        let attn = attention::causal_gqa_bf16(ctx, &q_out, &k_view, &v_view, kv_len)?;
        if layer_idx == 3 && seq > 1 {
            trace_rows("text3_q", &q_out.reshape(vec![seq, heads * head_dim])?)?;
            trace_rows(
                "text3_k",
                &k_view.reshape(vec![kv_len, kv_heads * head_dim])?,
            )?;
            trace_rows(
                "text3_v",
                &v_view.reshape(vec![kv_len, kv_heads * head_dim])?,
            )?;
            trace_rows(
                "text3_attention",
                &attn.reshape(vec![seq, heads * head_dim])?,
            )?;
        }
        let attn = attn.reshape(vec![seq, heads * head_dim])?;
        la::sigmoid_gate_mul(ctx, &attn, &fused, heads, head_dim)?;
        let proj = linear_checkpoint(self.weights.projection_layout, ctx, &attn, &w.o_w)?;
        if layer_idx == 3 && seq > 1 {
            trace_rows("text3_gated_attention", &attn)?;
            trace_rows("text3_out_proj", &proj)?;
        }
        self.forward_mlp(&x, &proj, &w.post_norm, &w.gate_up_w, &w.down_w, false)
    }
    pub(crate) fn forward_gdn_eager(
        &self,
        state: &mut BackboneState,
        x: Tensor,
        layer_idx: usize,
        seq: usize,
    ) -> Result<Tensor> {
        let cuda = Arc::clone(&self.cuda);
        let ctx = cuda.context();
        let text = &self.config.text;
        let w = match &self.weights.layers[layer_idx] {
            MixerWeights::Gdn(w) => w,
            _ => return Err(Error::Other("qwen_drive: layer kind mismatch".into())),
        };
        let eps = text.rms_norm_eps;
        let num_k_heads = text.linear_num_key_heads;
        let num_v_heads = text.linear_num_value_heads;
        let head_k = text.linear_key_head_dim;
        let head_v = text.linear_value_head_dim;
        let key_dim = num_k_heads * head_k;
        let value_dim = num_v_heads * head_v;
        let conv_dim = 2 * key_dim + value_dim;
        let z_col = conv_dim;
        let b_col = conv_dim + value_dim;
        let a_col = b_col + num_v_heads;
        let kernel = text.linear_conv_kernel_dim;
        let has_state = state.cache_len > 0;

        // The scan is only part of a GDN layer; attribute what precedes it too.
        let gdn_timed = seq > 1 && diagnostics_enabled();
        let mut gdn_since = std::time::Instant::now();
        let normed = la::rms_norm_plus1(ctx, &x, &w.input_norm, eps)?;
        if gdn_timed {
            gdn_stage_mark(ctx, layer_idx, "pre_norm", &mut gdn_since)?;
        }
        // Preserve all four reference GEMM geometries. Transposing or fusing
        // these weights selects different BF16 reductions in cuBLAS.
        // One GEMM over the packed projection. Four separate ones read the same
        // weight bytes, but two were only [32, hidden] and dragged the group to
        // about 70GB/s where the MLP projections reach 172GB/s; the packed
        // weight also arrives in the layout the on-device pack used to build.
        let zba = linear_checkpoint(self.weights.projection_layout, ctx, &normed, &w.zba_w)?
            .reshape(vec![seq, conv_dim + value_dim + 2 * num_v_heads])?;
        if gdn_timed {
            gdn_stage_mark(ctx, layer_idx, "project", &mut gdn_since)?;
        }
        if layer_idx == 0 && seq > 1 {
            trace_rows("text0_input", &x)?;
            trace_rows("text0_input_norm", &normed)?;
            trace_rows("text0_zba", &zba)?;
        }
        let conv_out = device_tensor(ctx, &[seq, conv_dim], DType::BF16)?;
        let (state_current, state_next) = match &state.caches[layer_idx] {
            LayerCache::Gdn {
                conv_state_a,
                conv_state_b,
                flip,
                ..
            } => {
                if *flip {
                    (conv_state_b.clone(), conv_state_a.clone())
                } else {
                    (conv_state_a.clone(), conv_state_b.clone())
                }
            }
            _ => return Err(Error::Other("qwen_drive: cache kind mismatch".into())),
        };
        la::causal_conv1d_silu_bf16(
            ctx,
            &zba,
            &w.conv_w,
            if has_state {
                Some(&state_current)
            } else {
                None
            },
            &conv_out,
            &state_next,
            kernel,
        )?;
        if let LayerCache::Gdn { flip, .. } = &mut state.caches[layer_idx] {
            *flip = !*flip;
        }
        if layer_idx == 0 && seq > 1 {
            trace_rows("text0_conv_silu", &conv_out)?;
        }
        if gdn_timed {
            gdn_stage_mark(ctx, layer_idx, "conv_silu", &mut gdn_since)?;
        }

        let recurrent_decode = has_state && seq == 1;
        let seq_pad = if recurrent_decode {
            1
        } else {
            seq.div_ceil(GDN_CHUNK) * GDN_CHUNK
        };
        let scan_padded = seq_pad != seq;
        let q_buf = alloc_scan_scratch(
            ctx,
            num_v_heads * seq_pad * head_k * DType::F32.size_in_bytes(),
            scan_padded,
        )?;
        let k_buf = alloc_scan_scratch(
            ctx,
            num_v_heads * seq_pad * head_k * DType::F32.size_in_bytes(),
            scan_padded,
        )?;
        let v_buf = alloc_scan_scratch(
            ctx,
            num_v_heads * seq_pad * head_v * DType::F32.size_in_bytes(),
            scan_padded,
        )?;
        let beta_buf = alloc_scan_scratch(
            ctx,
            num_v_heads * seq_pad * DType::F32.size_in_bytes(),
            scan_padded,
        )?;
        let g_buf = alloc_scan_scratch(
            ctx,
            num_v_heads * seq_pad * DType::F32.size_in_bytes(),
            scan_padded,
        )?;
        if gdn_timed {
            gdn_stage_mark(ctx, layer_idx, "qkvbg_alloc", &mut gdn_since)?;
        }
        la::gdn_qk_prep(
            ctx,
            &conv_out,
            &q_buf,
            &k_buf,
            seq_pad,
            num_k_heads,
            num_v_heads,
            head_k,
            key_dim,
            recurrent_decode,
            1e-6,
        )?;
        la::gdn_vb_prep(
            ctx,
            &conv_out,
            &zba,
            b_col,
            a_col,
            &w.dt_bias,
            &w.a_log,
            &v_buf,
            &beta_buf,
            &g_buf,
            seq_pad,
            num_v_heads,
            head_v,
            2 * key_dim,
        )?;
        let gdn_out = device_tensor(ctx, &[seq, value_dim], DType::BF16)?;
        let rec_state = match &state.caches[layer_idx] {
            LayerCache::Gdn { recurrent, .. } => recurrent.clone(),
            _ => return Err(Error::Other("qwen_drive: cache kind mismatch".into())),
        };
        let rec_buf = DeviceBuffer::from_tensor(&rec_state).map_err(Error::Cuda)?;
        if recurrent_decode {
            la::gdn_recurrent(
                ctx,
                &q_buf,
                &k_buf,
                &v_buf,
                &beta_buf,
                &g_buf,
                &rec_buf,
                &gdn_out,
                num_v_heads,
                head_k,
                head_v,
            )?;
        } else {
            let chunks = seq_pad / GDN_CHUNK;
            let g_cum = alloc_zeros(ctx, num_v_heads * seq_pad * DType::F32.size_in_bytes())?;
            let a_buf = alloc_zeros(
                ctx,
                num_v_heads * chunks * GDN_CHUNK * GDN_CHUNK * DType::F32.size_in_bytes(),
            )?;
            let t_buf = alloc_zeros(
                ctx,
                num_v_heads * chunks * GDN_CHUNK * GDN_CHUNK * DType::F32.size_in_bytes(),
            )?;
            let vt_buf = alloc_zeros(
                ctx,
                num_v_heads * chunks * GDN_CHUNK * head_v * DType::F32.size_in_bytes(),
            )?;
            let kcd_buf = alloc_zeros(
                ctx,
                num_v_heads * chunks * GDN_CHUNK * head_k * DType::F32.size_in_bytes(),
            )?;
            // Covers the q/k/v/beta/g preparation kernels and the scan's own
            // zeroed allocations, which sit between conv_silu and cumsum.
            if gdn_timed {
                gdn_stage_mark(ctx, layer_idx, "prep_alloc", &mut gdn_since)?;
            }
            la::gdn_cumsum(ctx, &g_buf, &g_cum, seq_pad, num_v_heads, GDN_CHUNK)?;
            if gdn_timed {
                gdn_stage_mark(ctx, layer_idx, "cumsum", &mut gdn_since)?;
            }
            la::gdn_attn_raw(
                ctx,
                &q_buf,
                &k_buf,
                &beta_buf,
                &g_cum,
                &a_buf,
                &t_buf,
                seq_pad,
                num_v_heads,
                head_k,
                GDN_CHUNK,
            )?;
            if gdn_timed {
                gdn_stage_mark(ctx, layer_idx, "attn_raw", &mut gdn_since)?;
            }
            la::gdn_tri_solve(ctx, &a_buf, num_v_heads * chunks, GDN_CHUNK)?;
            if gdn_timed {
                gdn_stage_mark(ctx, layer_idx, "tri_solve", &mut gdn_since)?;
            }
            la::gdn_chunk_gemm(
                ctx,
                &a_buf,
                &v_buf,
                &k_buf,
                &beta_buf,
                &g_cum,
                &vt_buf,
                &kcd_buf,
                seq_pad,
                num_v_heads,
                head_k,
                head_v,
                GDN_CHUNK,
            )?;
            if gdn_timed {
                gdn_stage_mark(ctx, layer_idx, "chunk_gemm", &mut gdn_since)?;
            }
            la::gdn_chunk_state(
                ctx,
                &q_buf,
                &k_buf,
                &g_cum,
                &t_buf,
                &vt_buf,
                &kcd_buf,
                &rec_buf,
                &gdn_out,
                seq_pad,
                num_v_heads,
                head_k,
                head_v,
                GDN_CHUNK,
            )?;
            if gdn_timed {
                gdn_stage_mark(ctx, layer_idx, "chunk_state", &mut gdn_since)?;
            }
        }
        if layer_idx == 0 && seq > 1 {
            trace_rows("text0_core", &gdn_out)?;
        }
        // Opt-in fingerprints of the first two recurrent outputs and convolution row.
        if layer_idx == 0 && seq > 1 && diagnostics_enabled() {
            let gbuf = DeviceBuffer::from_tensor(&gdn_out).map_err(Error::Cuda)?;
            let gview = gbuf
                .view(0, 2 * value_dim * DType::BF16.size_in_bytes())
                .map_err(Error::Cuda)?;
            let gt = gview
                .as_tensor(Shape::new(vec![2, value_dim]), DType::BF16)
                .map_err(Error::Cuda)?;
            let gvals = transfers::to_cpu(&gt)?
                .to_f32_vec()
                .map_err(|e| Error::Other(format!("qwen_drive gdn_row: {e}")))?;
            let gl2 = |r: usize| -> f64 {
                gvals[r * value_dim..(r + 1) * value_dim]
                    .iter()
                    .map(|&v| (v as f64) * (v as f64))
                    .sum::<f64>()
                    .sqrt()
            };
            let g_line0 = format!(
                "[qwen_drive] gdn_row0 l2={:.4} first4={:?}",
                gl2(0),
                &gvals[..gvals.len().min(4)]
            );
            let g_line1 = format!(
                "[qwen_drive] gdn_row1 l2={:.4} first4={:?}",
                gl2(1),
                &gvals[value_dim..value_dim + 4]
            );
            let cbuf = DeviceBuffer::from_tensor(&conv_out).map_err(Error::Cuda)?;
            let cview = cbuf
                .view(0, conv_dim * DType::BF16.size_in_bytes())
                .map_err(Error::Cuda)?;
            let ct = cview
                .as_tensor(Shape::new(vec![1, conv_dim]), DType::BF16)
                .map_err(Error::Cuda)?;
            let cvals = transfers::to_cpu(&ct)?
                .to_f32_vec()
                .map_err(|e| Error::Other(format!("qwen_drive conv_row: {e}")))?;
            let cl2: f64 = cvals
                .iter()
                .map(|&v| (v as f64) * (v as f64))
                .sum::<f64>()
                .sqrt();
            let c_line = format!(
                "[qwen_drive] conv_row0 l2={:.4} first4={:?}",
                cl2,
                &cvals[..cvals.len().min(4)]
            );
            qdiag!("{}", g_line0);
            qdiag!("{}", g_line1);
            qdiag!("{}", c_line);
        }
        let gated = la::gated_rms_silu(
            ctx,
            &gdn_out.reshape(vec![seq * num_v_heads, head_v])?,
            &zba,
            z_col,
            num_v_heads,
            &w.gated_norm,
            1e-6,
        )?;
        let gated = gated.reshape(vec![seq, value_dim])?;
        let proj = linear_checkpoint(self.weights.projection_layout, ctx, &gated, &w.out_w)?;
        if layer_idx == 0 && seq > 1 {
            trace_rows("text0_gated_norm", &gated)?;
            trace_rows("text0_out_proj", &proj)?;
        }
        self.forward_mlp(
            &x,
            &proj,
            &w.post_norm,
            &w.gate_up_w,
            &w.down_w,
            layer_idx == 0 && seq > 1,
        )
    }
    pub(crate) fn embed_tokens(&self, token_ids: &[u32]) -> Result<Tensor> {
        let ids = upload_u32(self.ctx(), token_ids)?;
        // Qwen uses the raw embedding table; lookup_bf16 applies Gemma scaling.
        let output = embedding::lookup(
            self.ctx(),
            &self.weights.embed_tokens,
            &ids,
            token_ids.len(),
        )?;
        trace_rows("model_language_model_embed_tokens", &output)?;
        Ok(output)
    }
    pub(crate) fn upload_pixels(&self, pixels: &Tensor) -> Result<Tensor> {
        let ctx = self.ctx();
        let on_device = if pixels.device() != Device::Cuda(ctx.device_id()) {
            transfers::to_cuda(pixels, ctx.device_id())?
        } else {
            pixels.clone()
        };
        match on_device.dtype() {
            DType::BF16 => Ok(on_device),
            DType::F32 => {
                let dims = on_device.shape().dims().to_vec();
                let out = device_tensor(ctx, &dims, DType::BF16)?;
                la::cast_f32_to_bf16(ctx, &on_device, &out)?;
                Ok(out)
            }
            dtype => Err(Error::Other(format!(
                "qwen_drive: pixel_values must be f32 or bf16, got {dtype}"
            ))),
        }
    }
    pub(crate) fn scene_caches(&self, state: &mut BackboneState) -> Result<Vec<(Tensor, Tensor)>> {
        let mut out = Vec::new();
        for layer_idx in self.config.text.full_attention_layers() {
            match &state.caches[layer_idx] {
                LayerCache::FullAttention { k, v } => {
                    out.push((
                        cache_view(k, state.cache_len)?,
                        cache_view(v, state.cache_len)?,
                    ));
                }
                _ => return Err(Error::Other("qwen_drive: cache kind mismatch".into())),
            }
        }
        Ok(out)
    }
    pub(crate) fn new_state(&self, vision: std::rc::Rc<VisionState>) -> Result<BackboneState> {
        let max_seq_len = self.config.text.max_position_embeddings.min(16384);
        Ok(BackboneState {
            vision,
            caches: Self::fresh_caches(&self.config, &self.cuda, max_seq_len)?,
            cache_len: 0,
            rope_delta: 0,
            max_seq_len,
            last_position: 0,
            decode_step: None,
        })
    }
    pub(crate) fn run_text(
        &self,
        state: &mut BackboneState,
        execution: &mut dyn GdnExecution,
        x: Tensor,
        positions: &[[u32; 3]],
    ) -> Result<Tensor> {
        let seq = positions.len();
        if seq == 0 {
            return Err(Error::Other("qwen_drive: empty forward span".into()));
        }
        let last = positions[seq - 1];
        state.last_position = last[0].max(last[1]).max(last[2]) as i64;
        let (cos, sin) = self.mrope_tables(positions)?;
        let mut hidden = x;
        for layer in 0..self.config.text.n_layers {
            hidden = match &self.weights.layers[layer] {
                MixerWeights::FullAttention(_) => {
                    self.forward_full_attention(state, hidden, layer, &cos, &sin, seq)?
                }
                MixerWeights::Gdn(_) => {
                    let parity = match &state.caches[layer] {
                        LayerCache::Gdn { flip, .. } => usize::from(*flip),
                        _ => unreachable!(),
                    };
                    let request = GdnRequest {
                        backend: &self.cuda,
                        layer,
                        layer_count: self.weights.layers.len(),
                        parity,
                        decode: seq == 1 && state.cache_len > 0,
                        decode_step: state.decode_step.unwrap_or(0),
                    };
                    let (output, replayed) = execution.run(&request, hidden, &mut |x| {
                        self.forward_gdn_eager(state, x, layer, seq)
                    })?;
                    if replayed {
                        if let LayerCache::Gdn { flip, .. } = &mut state.caches[layer] {
                            *flip = !*flip;
                        }
                    }
                    output
                }
            };
            trace_rows(&format!("backbone_layer_{layer}"), &hidden)?;
        }
        state.cache_len += seq;
        Ok(hidden)
    }
    pub(crate) fn next_logits(&self, hidden: &Tensor) -> Result<Tensor> {
        let normed = la::rms_norm_plus1(
            self.ctx(),
            hidden,
            &self.weights.final_norm,
            self.config.text.rms_norm_eps,
        )?;
        let dims = normed.shape().dims();
        let row_bytes = dims[1] * DType::BF16.size_in_bytes();
        let buffer = DeviceBuffer::from_tensor(&normed).map_err(Error::Cuda)?;
        let last = buffer
            .view((dims[0] - 1) * row_bytes, row_bytes)
            .map_err(Error::Cuda)?
            .as_tensor(Shape::new(vec![1, dims[1]]), DType::BF16)
            .map_err(Error::Cuda)?;
        self.lm_head(&last)
    }
    pub(crate) fn prefill(
        &self,
        state: &mut BackboneState,
        execution: &mut dyn GdnExecution,
        token_ids: &[u32],
        pixels: &Tensor,
        grids: &[[u32; 3]],
    ) -> Result<Tensor> {
        if token_ids.is_empty() || token_ids.len() > state.max_seq_len || state.cache_len != 0 {
            return Err(Error::Other(
                "qwen_drive: prefill requires a nonempty prompt within capacity and a fresh cache"
                    .into(),
            ));
        }
        let x = self.embed_tokens(token_ids)?;
        let pixels = self.upload_pixels(pixels)?;
        let vis = vision::forward(
            &self.config,
            &self.weights.vision,
            &state.vision,
            self.ctx(),
            &pixels,
            grids,
        )?;
        let image_token = self.config.image_token_id;
        let count = token_ids.iter().filter(|&&t| t == image_token).count();
        if count != vis.shape().dims()[0] {
            return Err(Error::Other(
                "qwen_drive: image token / vision row mismatch".into(),
            ));
        }
        let mut ordinal = 0;
        let rows: Vec<u32> = token_ids
            .iter()
            .map(|&t| {
                if t == image_token {
                    let r = ordinal;
                    ordinal += 1;
                    r
                } else {
                    u32::MAX
                }
            })
            .collect();
        let rows = upload_u32(self.ctx(), &rows)?;
        let x = elementwise::replace_rows_bf16(self.ctx(), &x, &vis, &rows)?;
        let positions = self.rope_index(token_ids, grids)?;
        let max_pos = positions
            .iter()
            .map(|p| p[0].max(p[1]).max(p[2]))
            .max()
            .unwrap_or(0) as i64;
        state.rope_delta = max_pos + 1 - token_ids.len() as i64;
        self.run_text(state, execution, x, &positions)
    }
    pub(crate) fn forward_tokens(
        &self,
        state: &mut BackboneState,
        execution: &mut dyn GdnExecution,
        token_ids: &[u32],
    ) -> Result<Tensor> {
        if token_ids.is_empty() || token_ids.len() > state.max_seq_len - state.cache_len {
            return Err(Error::Other(
                "qwen_drive: continuation is empty or exceeds cache capacity".into(),
            ));
        }
        let positions: Vec<[u32; 3]> = (0..token_ids.len())
            .map(|i| {
                let p = (state.cache_len + i) as i64 + state.rope_delta;
                [p.max(0) as u32; 3]
            })
            .collect();
        let x = self.embed_tokens(token_ids)?;
        self.run_text(state, execution, x, &positions)
    }
}
mod vision {
    use apxinf_core::{Error, Result, Tensor};

    use crate::qwen_drive::backend::{kernels, transfers, Context, DeviceBuffer};
    use kernels::{activation, attention, elementwise, gemm, linear_attention as la, norm};

    use crate::qwen_drive::config::QwenDriveConfig;
    use crate::qwen_drive::weights::bf16::VisionDeviceWeights;

    fn upload_u32(ctx: &Context, values: &[u32]) -> Result<DeviceBuffer> {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect();
        let buffer =
            DeviceBuffer::alloc(bytes.len().max(1), ctx.device_id()).map_err(Error::Cuda)?;
        buffer.copy_from_host(&bytes).map_err(Error::Cuda)?;
        Ok(buffer)
    }

    /// Run the vision tower over the concatenated per-image patch stream.
    /// `pixel_values` is `[patches, 3 * temporal * patch * patch]` BF16 on device;
    /// `grid_thw` holds one `[T, H, W]` entry per image.
    pub fn forward(
        config: &QwenDriveConfig,
        weights: &VisionDeviceWeights,
        state: &super::VisionState,
        ctx: &Context,
        pixel_values: &Tensor,
        grid_thw: &[[u32; 3]],
    ) -> Result<Tensor> {
        let vc = &config.vision;
        let hidden = vc.hidden_size;
        let heads = vc.num_heads;
        let head_dim = vc.head_dim();
        let merge = vc.spatial_merge_size;
        let eps = 1e-6f32;

        let dims = pixel_values.shape().dims();
        let patch_vec = vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size;
        if grid_thw.is_empty() || dims.len() != 2 || dims[1] != patch_vec {
            return Err(Error::Other(format!(
                "qwen_drive vision: pixel_values must be [patches, {patch_vec}], got {dims:?}"
            )));
        }
        let total_patches = dims[0];
        let mut offsets = vec![0u32];
        let mut max_tokens = 0usize;
        for grid in grid_thw {
            let (t, h, w) = (grid[0] as usize, grid[1] as usize, grid[2] as usize);
            if t == 0 || h == 0 || w == 0 || h % merge != 0 || w % merge != 0 {
                return Err(Error::Other(format!(
                    "qwen_drive vision: invalid grid [{t}, {h}, {w}] for merge {merge}"
                )));
            }
            let patches = t * h * w;
            max_tokens = max_tokens.max(patches);
            offsets.push(offsets[offsets.len() - 1] + patches as u32);
        }
        if offsets[offsets.len() - 1] as usize != total_patches {
            return Err(Error::Other(format!(
                "qwen_drive vision: grids cover {} patches but pixel_values has {total_patches}",
                offsets[offsets.len() - 1]
            )));
        }
        // TEMP-DIAG (implement_r3): vision-entry geometry; revert in the acceptance-bound revision.
        qdiag!("[qwen_drive] vision_entry total_patches={} segments={} offsets_len={} offsets_max={} max_tokens={} heads={} head_dim={} n={} grids={:?}", total_patches, grid_thw.len(), offsets.len(), offsets.last().copied().unwrap_or(0), max_tokens, heads, head_dim, weights.blocks.len(), grid_thw);
        // TEMP-DIAG (implement_r4): per-op elapsed-ms reference; revert in the acceptance-bound revision.
        let vis_t0 = std::time::Instant::now();
        let offsets_dev = upload_u32(ctx, &offsets)?;

        // Patch embedding: [N, patch_vec] @ [patch_vec, hidden] + bias.
        let mut x = gemm::bf16(ctx, pixel_values, &weights.patch_w)?;
        x = elementwise::bias_bf16(ctx, &x, Some(&weights.patch_b))?;
        super::trace_rows("model_visual_patch_embed", &x)?;

        // Bilinear-interpolated learned position embeddings (host canonicalization).
        let pos_embeds = compute_pos_embeds(config, weights, state, ctx, grid_thw)?;
        x = elementwise::add(ctx, &x, &pos_embeds)?;

        // 2D rope position ids in the merge-block-major patch order.
        let pos_ids = compute_vision_pos_ids(grid_thw, merge);
        let pos_ids_dev = upload_u32(ctx, &pos_ids)?;

        for (block_idx, block) in weights.blocks.iter().enumerate() {
            // TEMP-DIAG (implement_r2, ungated in implement_r3): vision-block heartbeat every block; revert in the acceptance-bound revision.
            qdiag!(
                "[qwen_drive] vision_block k={} n={} ms={}",
                block_idx,
                weights.blocks.len(),
                vis_t0.elapsed().as_millis()
            );
            let normed = norm::layer_bf16(ctx, &x, &block.norm1_w, &block.norm1_b, eps)?;
            if block_idx == 0 {
                super::trace_rows("model_visual_blocks_0_norm1", &normed)?;
            }
            // nn.Linear adds bias before its final BF16 rounding.
            let qkv = gemm::bf16_bias(ctx, &normed, &block.qkv_w, &block.qkv_b)?;
            if block_idx == 0 {
                super::trace_rows("model_visual_blocks_0_attn_qkv", &qkv)?;
            }
            let qkv = attention::split_vision_qkv_rope_bf16(
                ctx,
                &qkv,
                None,
                &pos_ids_dev,
                heads,
                head_dim,
                10000.0,
            )?;
            let attn = attention::segmented_mha_bf16(
                ctx,
                &qkv.q,
                &qkv.k,
                &qkv.v,
                &offsets_dev,
                &offsets,
                grid_thw.len(),
                max_tokens,
            )?;
            let attn = attn.reshape(vec![total_patches, hidden])?;
            let proj = gemm::bf16_bias(ctx, &attn, &block.proj_w, &block.proj_b)?;
            if block_idx == 0 {
                super::trace_rows("model_visual_blocks_0_attn", &proj)?;
            }
            x = elementwise::add(ctx, &x, &proj)?;

            let normed = norm::layer_bf16(ctx, &x, &block.norm2_w, &block.norm2_b, eps)?;
            if block_idx == 0 {
                super::trace_rows("model_visual_blocks_0_norm2", &normed)?;
            }
            let h = gemm::bf16_bias(ctx, &normed, &block.fc1_w, &block.fc1_b)?;
            if block_idx == 0 {
                super::trace_rows("model_visual_blocks_0_mlp_linear_fc1", &h)?;
            }
            let h = activation::gelu_tanh(ctx, &h)?;
            let h2 = gemm::bf16_bias(ctx, &h, &block.fc2_w, &block.fc2_b)?;
            if block_idx == 0 {
                super::trace_rows("model_visual_blocks_0_mlp", &h2)?;
            }
            x = elementwise::add(ctx, &x, &h2)?;
            super::trace_rows(&format!("model_visual_blocks_{block_idx}"), &x)?;
        }

        // Merger: LayerNorm(1024) -> merge 4 rows -> fc1 -> exact erf GELU -> fc2.
        let normed =
            norm::layer_bf16(ctx, &x, &weights.merger_norm_w, &weights.merger_norm_b, eps)?;
        // Merging rows is a reshape and nothing else: the merger takes `merge*merge`
        // consecutive rows as one row of `cols * merge * merge`, which is the same
        // contiguous elements in the same order. On a GPU tensor `reshape` only
        // rewrites the metadata, so this replaces a kernel launch and a full copy
        // of the tower's output with no work at all.
        let (rows, cols) = {
            let dims = normed.shape().dims();
            (dims[0], dims[1])
        };
        let factor = merge * merge;
        if rows % factor != 0 {
            return Err(Error::Other(format!(
                "qwen_drive vision merger: {rows} rows do not divide by {factor}"
            )));
        }
        let merged = normed.reshape(vec![rows / factor, cols * factor])?;
        let h = gemm::bf16_bias(ctx, &merged, &weights.merger_fc1_w, &weights.merger_fc1_b)?;
        let h = la::gelu_exact(ctx, &h)?;
        let primary = gemm::bf16_bias(ctx, &h, &weights.merger_fc2_w, &weights.merger_fc2_b)?;
        Ok(primary)
    }

    /// Bilinear-interpolate the learned 48x48 position table to each image's
    /// patch grid and permute to the merge-block-major layout. Copied with
    /// provenance from qwen3vl's vision tower (HF `fast_pos_embed_interpolate` +
    /// the spatial-merge shuffle). The runner retains a bounded grid cache.
    fn compute_pos_embeds(
        config: &QwenDriveConfig,
        weights: &VisionDeviceWeights,
        state: &super::VisionState,
        ctx: &Context,
        grid_thw: &[[u32; 3]],
    ) -> Result<Tensor> {
        let timed = {
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ON.get_or_init(|| std::env::var_os("APXINF_QWEN_VISION_TIMING").is_some())
        };
        let cached_enabled = {
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ON.get_or_init(|| {
                !matches!(
                    std::env::var("APXINF_QWEN_POS_CACHE").as_deref(),
                    Ok("0") | Ok("off") | Ok("false")
                )
            })
        };
        let started = std::time::Instant::now();
        if cached_enabled {
            if let Ok(cache) = state.pos_embed_cache.lock() {
                if let Some((_, tensor)) = cache.iter().find(|(key, _)| key.as_slice() == grid_thw)
                {
                    let hit = tensor.clone();
                    if timed {
                        qdiag!(
                            "[qwen_drive] pos_embed cache hit {:.2}ms",
                            started.elapsed().as_secs_f64() * 1e3
                        );
                    }
                    return Ok(hit);
                }
            }
        }
        let vc = &config.vision;
        let hidden = vc.hidden_size;
        let merge = vc.spatial_merge_size;
        let grid_side = (vc.num_position_embeddings as f64).sqrt().round() as usize;
        // The table is a checkpoint constant, so it is read back once per model
        // rather than once per request.
        let table_owned: Vec<f32>;
        let table: &[f32] = if cached_enabled {
            if state.pos_table_host.get().is_none() {
                let values = transfers::to_cpu(&weights.pos_embed)?
                    .to_f32_vec()
                    .map_err(|e| Error::Other(format!("qwen_drive vision pos_embed table: {e}")))?;
                let _ = state.pos_table_host.set(values);
            }
            state
                .pos_table_host
                .get()
                .expect("pos_embed table is installed")
                .as_slice()
        } else {
            table_owned = transfers::to_cpu(&weights.pos_embed)?
                .to_f32_vec()
                .map_err(|e| Error::Other(format!("qwen_drive vision pos_embed table: {e}")))?;
            table_owned.as_slice()
        };
        let after_readback = started.elapsed();
        if table.len() != grid_side * grid_side * hidden {
            return Err(Error::Other(
                "qwen_drive vision: pos_embed table shape mismatch".into(),
            ));
        }
        let total: usize = grid_thw
            .iter()
            .map(|grid| grid[0] as usize * grid[1] as usize * grid[2] as usize)
            .sum();
        let mut out = vec![0.0f32; total * hidden];
        let mut dst_token = 0usize;
        for grid in grid_thw {
            let (t, h, w) = (grid[0] as usize, grid[1] as usize, grid[2] as usize);
            let merged_h = h / merge;
            let merged_w = w / merge;
            for _ti in 0..t {
                for mh in 0..merged_h {
                    for mw in 0..merged_w {
                        for ih in 0..merge {
                            for iw in 0..merge {
                                let hi = mh * merge + ih;
                                let wi = mw * merge + iw;
                                let hf = hi as f32 * (grid_side - 1) as f32 / (h - 1).max(1) as f32;
                                let h0 = hf.floor() as usize;
                                let h1 = (h0 + 1).min(grid_side - 1);
                                let dh = hf - h0 as f32;
                                let wf = wi as f32 * (grid_side - 1) as f32 / (w - 1).max(1) as f32;
                                let w0 = wf.floor() as usize;
                                let w1 = (w0 + 1).min(grid_side - 1);
                                let dw = wf - w0 as f32;
                                let dst = dst_token * hidden;
                                for c in 0..hidden {
                                    let v00 = table[(h0 * grid_side + w0) * hidden + c];
                                    let v01 = table[(h0 * grid_side + w1) * hidden + c];
                                    let v10 = table[(h1 * grid_side + w0) * hidden + c];
                                    let v11 = table[(h1 * grid_side + w1) * hidden + c];
                                    out[dst + c] = (1.0 - dh) * (1.0 - dw) * v00
                                        + (1.0 - dh) * dw * v01
                                        + dh * (1.0 - dw) * v10
                                        + dh * dw * v11;
                                }
                                dst_token += 1;
                            }
                        }
                    }
                }
            }
        }
        let after_interp = started.elapsed();
        let rounded: Vec<half::bf16> = out.iter().map(|&v| half::bf16::from_f32(v)).collect();
        let tensor = Tensor::from_bf16(vec![total, hidden], &rounded)?;
        let uploaded = transfers::to_cuda(&tensor, ctx.device_id());
        if cached_enabled {
            if let (Ok(value), Ok(mut cache)) = (uploaded.as_ref(), state.pos_embed_cache.lock()) {
                // A rig presents a handful of grids; keep the cache small rather
                // than letting an unexpected stream of shapes grow it without end.
                const MAX_GRIDS: usize = 4;
                if cache.len() >= MAX_GRIDS {
                    cache.remove(0);
                }
                cache.push((grid_thw.to_vec(), value.clone()));
            }
        }
        if timed {
            qdiag!(
            "[qwen_drive] pos_embed total={:.1}ms readback={:.1}ms interp={:.1}ms round+upload={:.1}ms tokens={} hidden={}",
            started.elapsed().as_secs_f64() * 1e3,
            after_readback.as_secs_f64() * 1e3,
            (after_interp - after_readback).as_secs_f64() * 1e3,
            (started.elapsed() - after_interp).as_secs_f64() * 1e3,
            total,
            hidden
        );
        }
        uploaded
    }

    /// Vision 2D-RoPE position ids `(h, w)` per patch in the merge-block-major
    /// layout (copied with provenance from qwen3vl's `compute_vision_pos_ids`).
    fn compute_vision_pos_ids(grid_thw: &[[u32; 3]], merge: usize) -> Vec<u32> {
        let mut ids = Vec::new();
        for grid in grid_thw {
            let (t, h, w) = (grid[0] as usize, grid[1] as usize, grid[2] as usize);
            let merged_h = h / merge;
            let merged_w = w / merge;
            for _ti in 0..t {
                for mh in 0..merged_h {
                    for mw in 0..merged_w {
                        for ih in 0..merge {
                            for iw in 0..merge {
                                ids.push((mh * merge + ih) as u32);
                                ids.push((mw * merge + iw) as u32);
                            }
                        }
                    }
                }
            }
        }
        ids
    }
}
pub(crate) mod expert {
    use apxinf_core::{DType, Error, Result, Shape, Tensor};

    use crate::qwen_drive::backend::{kernels, transfers, Context, DeviceBuffer};
    use kernels::{activation, elementwise, embedding, gemm, linear_attention as la, norm};

    use crate::qwen_drive::config::{PlanningExpertConfig, QwenDriveConfig};
    use crate::qwen_drive::inputs::ExpertConditioning;
    use crate::qwen_drive::weights::bf16::{DeviceMlp, ExpertDeviceWeights};

    // The expert has only one token per waypoint, so bounded opt-in diagnostics
    // can retain every row instead of sampling the much larger VLM sequence.
    fn trace_rows(name: &str, tensor: &Tensor) -> Result<()> {
        if std::env::var_os("APXINF_QWEN_TRACE_DIR").is_none() {
            return Ok(());
        }
        super::trace_rows(name, &tensor.reshape(vec![1, tensor.numel()])?)
    }

    /// bf16 rounding of one f32 value, returned as f32 (the value a bf16 tensor
    /// would hold). Mirrors the scaffold's semantic-rounding helper.
    fn bf16_round(value: f32) -> f32 {
        half::bf16::from_f32(value).to_f32()
    }

    fn upload_u32(ctx: &Context, values: &[u32]) -> Result<DeviceBuffer> {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect();
        let buffer =
            DeviceBuffer::alloc(bytes.len().max(1), ctx.device_id()).map_err(Error::Cuda)?;
        buffer.copy_from_host(&bytes).map_err(Error::Cuda)?;
        Ok(buffer)
    }

    fn device_tensor(ctx: &Context, shape: &[usize], dtype: DType) -> Result<Tensor> {
        let elements: usize = shape.iter().product();
        let bytes = elements
            .checked_mul(dtype.size_in_bytes())
            .ok_or_else(|| Error::Other("qwen_drive expert: tensor size overflow".into()))?;
        let buffer = kernels::scratch_buffer(ctx, bytes.max(1))?;
        buffer
            .as_tensor(Shape::new(shape.to_vec()), dtype)
            .map_err(Error::Cuda)
    }

    fn bf16_tensor(ctx: &Context, values: &[f32], shape: Vec<usize>) -> Result<Tensor> {
        let rounded: Vec<half::bf16> = values.iter().map(|&v| half::bf16::from_f32(v)).collect();
        let tensor = Tensor::from_bf16(shape, &rounded)?;
        transfers::to_cuda(&tensor, ctx.device_id())
    }

    /// One `[cols]` column view of a `[1, N * cols]` row tensor.
    fn row_col_slice(tensor: &Tensor, col: usize, cols: usize) -> Result<Tensor> {
        let buffer = DeviceBuffer::from_tensor(tensor).map_err(Error::Cuda)?;
        let view = buffer
            .view(
                col * DType::BF16.size_in_bytes(),
                cols * DType::BF16.size_in_bytes(),
            )
            .map_err(Error::Cuda)?;
        view.as_tensor(Shape::new(vec![cols]), DType::BF16)
            .map_err(Error::Cuda)
    }

    fn mlp(ctx: &Context, mlp: &DeviceMlp, x: &Tensor) -> Result<Tensor> {
        let rows = x.shape().dims()[0];
        let h = if mlp.checkpoint_layout {
            if rows != 1 {
                return Err(Error::Other("checkpoint-layout MLP expects one row".into()));
            }
            gemm::bf16_addmv(
                ctx,
                &mlp.fc1_w,
                &x.reshape(vec![x.shape().dims()[1]])?,
                &mlp.fc1_b,
            )?
            .reshape(vec![1, mlp.fc1_w.shape().dims()[0]])?
        } else {
            gemm::bf16_bias(ctx, x, &mlp.fc1_w, &mlp.fc1_b)?
        };
        let h = activation::silu(ctx, &h)?;
        if mlp.checkpoint_layout {
            gemm::bf16_addmv(
                ctx,
                &mlp.fc2_w,
                &h.reshape(vec![h.shape().dims()[1]])?,
                &mlp.fc2_b,
            )?
            .reshape(vec![1, mlp.fc2_w.shape().dims()[0]])
        } else {
            gemm::bf16_bias(ctx, &h, &mlp.fc2_w, &mlp.fc2_b)
        }
    }

    fn one_hot(classes: usize, index: i64) -> Vec<f32> {
        let mut out = vec![0.0f32; classes];
        if index >= 0 && (index as usize) < classes {
            out[index as usize] = 1.0;
        }
        out
    }

    /// Waypoint rope tables (scaffold semantics): positions `anchor + 1 ..= anchor
    /// + length` cast to bf16, multiplied with the bf16-rounded inverse-frequency
    /// table, merged section-wise, cos/sin rounded to bf16. Returns `(cos, sin)`,
    /// each `[length, rotary_dim]` in f32 (bf16-rounded values).
    fn waypoint_rope_tables(
        config: &PlanningExpertConfig,
        anchor: i64,
        length: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let rotary_dim = config.rotary_dim();
        let pairs = rotary_dim / 2;
        let theta = config.rope_theta;
        let mut inv_freq = Vec::with_capacity(pairs);
        for i in 0..pairs {
            let exponent = (2 * i) as f32 / rotary_dim as f32;
            inv_freq.push(bf16_round(1.0 / theta.powf(exponent)));
        }
        let mut angles = vec![[0.0f32; 3].map(|_| vec![0.0f32; pairs]); length];
        let mut cos = vec![0.0f32; length * rotary_dim];
        let mut sin = vec![0.0f32; length * rotary_dim];
        for l in 0..length {
            let position = bf16_round((anchor + 1 + l as i64) as f32);
            for p in 0..pairs {
                let angle = bf16_round(position * inv_freq[p]);
                for section in &mut angles[l] {
                    section[p] = angle;
                }
            }
            let mut merged = angles[l][0].clone();
            let mut offset = 1usize;
            for (section_index, section_length) in config.mrope_section.iter().enumerate().skip(1) {
                let mut p = offset;
                let mut taken = 0usize;
                while p < pairs && taken < *section_length {
                    merged[p] = angles[l][section_index][p];
                    taken += 1;
                    p += 3;
                }
                offset += 1;
            }
            for p in 0..pairs {
                cos[l * rotary_dim + p] = bf16_round(merged[p].cos());
                cos[l * rotary_dim + pairs + p] = cos[l * rotary_dim + p];
                sin[l * rotary_dim + p] = bf16_round(merged[p].sin());
                sin[l * rotary_dim + pairs + p] = sin[l * rotary_dim + p];
            }
        }
        Ok((cos, sin))
    }

    /// Inputs for one expert planning call. `scene` holds one `(K, V)` pair per
    /// expert KV source (VLM full-attention layers, post-rotary, bf16
    /// `[scene_len, kv_heads, head_dim]` on device).
    pub struct ExpertPlan<'a> {
        pub scene: &'a [(Tensor, Tensor)],
        pub scene_len: usize,
        pub anchor: i64,
        pub cond: &'a ExpertConditioning,
        pub noise: &'a Tensor,
        pub num_steps: usize,
    }

    pub(crate) struct ExpertState {
        pose_q: Tensor,
        velocity_q: Tensor,
        acceleration_q: Tensor,
        nav_out: Tensor,
        ego_out: Tensor,
        wp_idx: Tensor,
        cos_t: Tensor,
        sin_t: Tensor,
        waypoints: Tensor,
        time_buffer: DeviceBuffer,
        time_row_bytes: usize,
        step_f64: f64,
    }
    pub(crate) fn prepare(
        config: &QwenDriveConfig,
        weights: &ExpertDeviceWeights,
        ctx: &Context,
        input: &ExpertPlan,
    ) -> Result<ExpertState> {
        let ec = &config.expert;
        let length = config.num_future_points;
        let point_dim = config.trajectory_point_dim;
        let rotary_dim = ec.rotary_dim();
        let steps = input.num_steps.max(1);
        if input.scene.len() != ec.num_kv_sources() {
            return Err(Error::Other(format!(
                "qwen_drive expert: {} scene caches, expected {}",
                input.scene.len(),
                ec.num_kv_sources()
            )));
        }
        if input.noise.numel() != length * point_dim {
            return Err(Error::Other(format!(
                "qwen_drive expert: noise has {} values, expected {}",
                input.noise.numel(),
                length * point_dim
            )));
        }
        input.cond.validate(config)?;

        // Conditioning (request lifetime).
        let nav = one_hot(ec.nav_command_classes, input.cond.nav_command);
        let mut history_in = input.cond.history.clone();
        history_in.extend_from_slice(&nav);
        let history_t = bf16_tensor(ctx, &history_in, vec![1, history_in.len()])?;
        let velocity_t = bf16_tensor(
            ctx,
            &input.cond.history_velocity,
            vec![1, input.cond.history_velocity.len()],
        )?;
        let acceleration_t = bf16_tensor(
            ctx,
            &input.cond.history_acceleration,
            vec![1, input.cond.history_acceleration.len()],
        )?;
        let ego_t = bf16_tensor(
            ctx,
            &input.cond.ego_status,
            vec![1, input.cond.ego_status.len()],
        )?;
        let nav_t = bf16_tensor(ctx, &nav, vec![1, nav.len()])?;
        let pose_q = mlp(ctx, &weights.history_encoder, &history_t)?;
        let velocity_q = mlp(ctx, &weights.history_velocity_encoder, &velocity_t)?;
        let acceleration_q = mlp(ctx, &weights.history_acceleration_encoder, &acceleration_t)?;
        let nav_out = mlp(ctx, &weights.nav_mlp, &nav_t)?;
        let ego_out = mlp(ctx, &weights.ego_mlp, &ego_t)?;
        for (name, value) in [
            ("history_encoder", &pose_q),
            ("history_velocity_encoder", &velocity_q),
            ("history_acceleration_encoder", &acceleration_q),
            ("nav_mlp", &nav_out),
            ("ego_mlp", &ego_out),
        ] {
            trace_rows(&format!("expert_{name}"), value)?;
        }

        let ids: Vec<u32> = (0..length as u32).collect();
        let ids_dev = upload_u32(ctx, &ids)?;
        let wp_idx = embedding::lookup(ctx, &weights.waypoint_embed, &ids_dev, length)?;

        let (cos, sin) = waypoint_rope_tables(ec, input.anchor, length)?;
        let cos_t = bf16_tensor(ctx, &cos, vec![length, rotary_dim])?;
        let sin_t = bf16_tensor(ctx, &sin, vec![length, rotary_dim])?;

        // fp32 solver state on device.
        let waypoints = device_tensor(ctx, &[length, point_dim], DType::F32)?;
        let src = DeviceBuffer::from_tensor(input.noise).map_err(Error::Cuda)?;
        DeviceBuffer::from_tensor(&waypoints)
            .map_err(Error::Cuda)?
            .copy_from_device_async(&src, length * point_dim * 4, ctx.stream())
            .map_err(Error::Cuda)?;

        let step_f64 = 1.0f64 / steps as f64;
        let times: Vec<f32> = (0..steps).map(|i| (i as f64 * step_f64) as f32).collect();
        let times = transfers::to_cuda(&Tensor::from_f32(vec![steps], &times)?, ctx.device_id())?;
        let decay = ((10000.0f64).ln() / ((ec.time_embed_dim / 2).max(2) - 1) as f64) as f32;
        let time_embeddings =
            embedding::sinusoidal_bf16(ctx, &times, ec.time_embed_dim, ec.time_embed_scale, decay)?;
        let time_buffer = DeviceBuffer::from_tensor(&time_embeddings).map_err(Error::Cuda)?;
        let time_row_bytes = ec.time_embed_dim * DType::BF16.size_in_bytes();
        Ok(ExpertState {
            pose_q,
            velocity_q,
            acceleration_q,
            nav_out,
            ego_out,
            wp_idx,
            cos_t,
            sin_t,
            waypoints,
            time_buffer,
            time_row_bytes,
            step_f64,
        })
    }
    pub(crate) fn step(
        config: &QwenDriveConfig,
        weights: &ExpertDeviceWeights,
        ctx: &Context,
        input: &ExpertPlan,
        state: &ExpertState,
        index: usize,
    ) -> Result<()> {
        let ec = &config.expert;
        let hidden_size = ec.hidden_size;
        let length = config.num_future_points;
        let point_dim = config.trajectory_point_dim;
        let heads = ec.n_heads;
        let kv_heads = ec.n_kv_heads;
        let head_dim = ec.head_dim;
        let rotary_dim = ec.rotary_dim();
        let eps = ec.rms_norm_eps;
        let trace_layer = std::env::var("APXINF_QWEN_EXPERT_TRACE_LAYER")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        let trace_step = std::env::var("APXINF_QWEN_EXPERT_TRACE_STEP")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        let pose_q = &state.pose_q;
        let velocity_q = &state.velocity_q;
        let acceleration_q = &state.acceleration_q;
        let nav_out = &state.nav_out;
        let ego_out = &state.ego_out;
        let wp_idx = &state.wp_idx;
        let cos_t = &state.cos_t;
        let sin_t = &state.sin_t;
        let waypoints = &state.waypoints;
        let time_buffer = &state.time_buffer;
        let time_row_bytes = state.time_row_bytes;
        let step_f64 = state.step_f64;

        let t_f64 = index as f64 * step_f64;
        let t_emb_t = time_buffer
            .view(index * time_row_bytes, time_row_bytes)
            .map_err(Error::Cuda)?
            .as_tensor(Shape::new(vec![1, ec.time_embed_dim]), DType::BF16)
            .map_err(Error::Cuda)?;
        let time_condition = mlp(ctx, &weights.time_mlp, &t_emb_t)?;
        let condition = elementwise::add(ctx, &time_condition, &nav_out)?;
        let condition = elementwise::add(ctx, &condition, &ego_out)?;
        let condition_silu = activation::silu(ctx, &condition)?;
        trace_rows(&format!("expert_time_embed_{index}"), &t_emb_t)?;
        trace_rows(&format!("expert_time_condition_{index}"), &time_condition)?;

        let wp_bf16 = device_tensor(ctx, &[length, point_dim], DType::BF16)?;
        la::cast_f32_to_bf16(ctx, &waypoints, &wp_bf16)?;
        let traj = gemm::bf16_bias(
            ctx,
            &wp_bf16,
            &weights.trajectory_proj_w,
            &weights.trajectory_proj_b,
        )?;
        let fourier_feat = la::fourier_features(
            ctx,
            &wp_bf16,
            &weights.fourier_freqs,
            point_dim,
            ec.fourier_num_features,
        )?;
        let fourier = mlp(ctx, &weights.fourier, &fourier_feat)?;
        if index == trace_step {
            trace_rows("expert_trajectory_proj", &traj)?;
            trace_rows("expert_fourier_encoder", &fourier)?;
        }
        // broadcast bits: time(2), pose(3), velocity(5), acceleration(6).
        let fused = la::concat7_cols(
            ctx,
            [
                &traj,
                &fourier,
                &time_condition,
                &pose_q,
                &wp_idx,
                &velocity_q,
                &acceleration_q,
            ],
            (1 << 2) | (1 << 3) | (1 << 5) | (1 << 6),
        )?;
        let mut hidden = mlp(ctx, &weights.query_fusion, &fused)?;
        if index == trace_step {
            trace_rows("expert_query_fusion", &hidden)?;
        }

        for (layer_index, layer) in weights.layers.iter().enumerate() {
            let (scene_k, scene_v) = &input.scene[layer_index / ec.layers_per_kv];
            let modulation = gemm::bf16_addmv(
                ctx,
                &layer.modulation_w,
                &condition_silu.reshape(vec![hidden_size])?,
                &layer.modulation_b,
            )?
            .reshape(vec![1, 6 * hidden_size])?;
            let shift_attn = row_col_slice(&modulation, 0, hidden_size)?;
            let scale_attn = row_col_slice(&modulation, hidden_size, hidden_size)?;
            let gate_attn = row_col_slice(&modulation, 2 * hidden_size, hidden_size)?;
            let shift_ffn = row_col_slice(&modulation, 3 * hidden_size, hidden_size)?;
            let scale_ffn = row_col_slice(&modulation, 4 * hidden_size, hidden_size)?;
            let gate_ffn = row_col_slice(&modulation, 5 * hidden_size, hidden_size)?;

            let x = la::adaln_rms_norm(
                ctx,
                &hidden,
                &layer.input_norm,
                &scale_attn,
                &shift_attn,
                eps,
            )?;
            let qkv = gemm::bf16(ctx, &x, &layer.qkv_w)?;
            if index == trace_step && layer_index == trace_layer {
                trace_rows("expert_modulation", &modulation)?;
                trace_rows("expert_adaln_input", &x)?;
                trace_rows("expert_qkv", &qkv)?;
            }
            let q_out = device_tensor(ctx, &[length, heads, head_dim], DType::BF16)?;
            let gate_out = device_tensor(ctx, &[length, heads, head_dim], DType::BF16)?;
            let k_out = device_tensor(ctx, &[length, kv_heads, head_dim], DType::BF16)?;
            let v_out = device_tensor(ctx, &[length, kv_heads, head_dim], DType::BF16)?;
            la::expert_qkv_prepare(
                ctx,
                &qkv,
                &layer.q_norm,
                &layer.k_norm,
                &cos_t,
                &sin_t,
                &q_out,
                &gate_out,
                &k_out,
                &v_out,
                heads,
                kv_heads,
                head_dim,
                rotary_dim,
                eps,
            )?;
            let k_cat = elementwise::concat_rows_bf16(
                ctx,
                &scene_k.reshape(vec![input.scene_len, kv_heads * head_dim])?,
                &k_out.reshape(vec![length, kv_heads * head_dim])?,
            )?;
            let v_cat = elementwise::concat_rows_bf16(
                ctx,
                &scene_v.reshape(vec![input.scene_len, kv_heads * head_dim])?,
                &v_out.reshape(vec![length, kv_heads * head_dim])?,
            )?;
            let k_cat = k_cat.reshape(vec![input.scene_len + length, kv_heads, head_dim])?;
            let v_cat = v_cat.reshape(vec![input.scene_len + length, kv_heads, head_dim])?;
            let attn = la::gqa_bf16(ctx, &q_out, &k_cat, &v_cat, input.scene_len + length)?;
            if index == trace_step && layer_index == trace_layer {
                trace_rows("expert_q", &q_out.reshape(vec![length, heads * head_dim])?)?;
                trace_rows(
                    "expert_attention",
                    &attn.reshape(vec![length, heads * head_dim])?,
                )?;
            }
            la::expert_sigmoid_gate_mul(ctx, &attn, &gate_out)?;
            let attn = attn.reshape(vec![length, heads * head_dim])?;
            let proj = gemm::bf16(ctx, &attn, &layer.o_w)?;
            if index == trace_step && layer_index == trace_layer {
                trace_rows("expert_o_proj", &proj)?;
            }
            hidden = la::adaln_gate_residual(ctx, &proj, &hidden, &gate_attn)?;

            let x2 =
                la::adaln_rms_norm(ctx, &hidden, &layer.post_norm, &scale_ffn, &shift_ffn, eps)?;
            let gu = gemm::bf16(ctx, &x2, &layer.gate_up_w)?;
            let act = activation::swiglu_bf16_rounded(ctx, &gu)?;
            let down = gemm::bf16(ctx, &act, &layer.down_w)?;
            if index == trace_step && layer_index == trace_layer {
                trace_rows("expert_adaln_ffn", &x2)?;
                trace_rows("expert_gate_up", &gu)?;
                trace_rows("expert_down", &down)?;
            }
            hidden = la::adaln_gate_residual(ctx, &down, &hidden, &gate_ffn)?;
            if index == trace_step {
                trace_rows(&format!("expert_layer_{layer_index}"), &hidden)?;
            }
        }

        let final_normed = norm::rms_bf16(ctx, &hidden, &weights.final_norm, eps)?;
        if index == trace_step {
            trace_rows("expert_final_norm", &final_normed)?;
        }
        let endpoint =
            gemm::bf16_bias(ctx, &final_normed, &weights.out_proj_w, &weights.out_proj_b)?;
        let endpoint_f32 = device_tensor(ctx, &[length, point_dim], DType::F32)?;
        trace_rows(
            &format!("expert_endpoint_{index}"),
            &endpoint.reshape(vec![1, length * point_dim])?,
        )?;
        la::cast_bf16_to_f32(ctx, &endpoint, &endpoint_f32)?;
        let remaining = (1.0f64 - t_f64).max(config.min_one_minus_t as f64) as f32;
        la::flow_update(ctx, &waypoints, &endpoint_f32, remaining, step_f64 as f32)?;
        trace_rows(
            &format!("expert_waypoints_{index}"),
            &waypoints.reshape(vec![1, length * point_dim])?,
        )?;
        Ok(())
    }
    pub(crate) fn output(state: ExpertState) -> Tensor {
        state.waypoints
    }
}
