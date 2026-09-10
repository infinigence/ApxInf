//! Qwen-Drive native model: Qwen3.5 hybrid VLM (gated-delta linear attention
//! + gated full attention with partial interleaved mRoPE) plus the planning
//! expert driver, on the CUDA kernel path.
//!
//! The hybrid cache is model-owned: full-attention layers append post-rotary
//! K/V into per-layer `[capacity, kv_heads, head_dim]` BF16 buffers (exactly
//! the post-rotary caches the planning expert reads), while GDN layers
//! advance a causal-conv state and an fp32 recurrent state. The type
//! implements `LlmTrait` for the maintained registry/AutoModel surface and
//! exposes the dedicated VQA / direct-planning / reasoning-planning flows
//! used by the PyO3 binding. All layer mathematics run on device.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use apxinf_core::{
    Backend, DType, Device, Error, NextTokenLogits, Result, RngKey, Shape, Tensor,
    TokenSamplingInit, TokenSamplingParams, TokenSamplingSpec,
};
use apxinf_loader::ModelConfig;

use crate::accelerator::create_backend;
use crate::llm_trait::{LlmCapabilities, LlmInput, LlmTrait};

use super::backend::{
    downcast_arc, kernels, transfers, Context, CublasTranspose, DeviceBuffer, RuntimeBackend,
};
use kernels::{activation, attention, elementwise, embedding, gemm, linear_attention as la};

use super::config::QwenDriveConfig;
use super::device_weights::{MixerWeights, QwenDriveDeviceWeights};
use super::expert::{self, ExpertPlan};
use super::planner::ExpertConditioning;
use super::vision;
use super::weights::{QwenDriveExpertWeights, QwenDriveVlmWeights};

const GDN_CHUNK: usize = 64;

fn bf16_round(value: f32) -> f32 {
    half::bf16::from_f32(value).to_f32()
}

fn device_tensor(ctx: &Context, shape: &[usize], dtype: DType) -> Result<Tensor> {
    let elements: usize = shape.iter().product();
    let bytes = elements
        .checked_mul(dtype.size_in_bytes())
        .ok_or_else(|| Error::Other("qwen_drive: tensor size overflow".into()))?;
    let buffer = DeviceBuffer::alloc(bytes.max(1), ctx.device_id()).map_err(Error::Cuda)?;
    buffer
        .as_tensor(Shape::new(shape.to_vec()), dtype)
        .map_err(Error::Cuda)
}

fn alloc_zeros(ctx: &Context, bytes: usize) -> Result<DeviceBuffer> {
    DeviceBuffer::alloc_zeros(bytes.max(1), ctx.device_id()).map_err(Error::Cuda)
}

fn upload_u32(ctx: &Context, values: &[u32]) -> Result<DeviceBuffer> {
    let bytes: Vec<u8> = values.iter().flat_map(|value| value.to_ne_bytes()).collect();
    let buffer = DeviceBuffer::alloc(bytes.len().max(1), ctx.device_id()).map_err(Error::Cuda)?;
    buffer.copy_from_host(&bytes).map_err(Error::Cuda)?;
    Ok(buffer)
}

/// `[kv_len, kv_heads, head_dim]` prefix view of a cache tensor.
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

enum LayerCache {
    FullAttention { k: Tensor, v: Tensor },
    Gdn {
        conv_state_a: Tensor,
        conv_state_b: Tensor,
        /// false -> a is current, b is scratch; true -> b is current.
        flip: bool,
        recurrent: Tensor,
    },
}

/// The native Qwen-Drive model (VLM + optional planning expert).
pub struct QwenDriveModel {
    config: QwenDriveConfig,
    backend: Arc<dyn Backend>,
    cuda: Arc<RuntimeBackend>,
    weights: QwenDriveDeviceWeights,
    caches: Vec<LayerCache>,
    cache_len: usize,
    rope_delta: i64,
    max_seq_len: usize,
    /// mRoPE position of the last processed token (all three axes equal for
    /// the text tokens that close every supported prompt).
    last_position: i64,
}

impl QwenDriveModel {
    /// Load the VLM (and optional planner head) onto a CUDA device.
    pub fn load(model_dir: &Path, planner_dir: Option<&Path>, device: Device) -> Result<Self> {
        let backend = create_backend(device)?;
        Self::load_with_backend(model_dir, planner_dir, backend)
    }

    pub fn load_with_backend(
        model_dir: &Path,
        planner_dir: Option<&Path>,
        backend: Arc<dyn Backend>,
    ) -> Result<Self> {
        let cuda = downcast_arc(backend.clone()).ok_or_else(|| {
            Error::Other(
                "qwen_drive requires the CUDA backend (native device execution); \
                 the CPU backend is not a deployment target"
                    .into(),
            )
        })?;
        let config = QwenDriveConfig::from_json_file(&model_dir.join("config.json"))?;
        eprintln!("[qwen_drive] loading VLM weights from {}", model_dir.display());
        let (tensors, _meta) = apxinf_loader::safetensors::load_native_path(model_dir)
            .map_err(|e| Error::Other(format!("qwen_drive: load VLM weights: {e}")))?;
        eprintln!("[qwen_drive] VLM safetensors loaded: {} tensors", tensors.len());
        let vlm = QwenDriveVlmWeights::from_map(tensors)?;
        eprintln!(
            "[qwen_drive] VLM weights classified: {} language tensors, {} visual tensors",
            vlm.language_tensor_count(),
            vlm.visual_tensor_count()
        );
        let expert = planner_dir
            .map(|dir| -> Result<QwenDriveExpertWeights> {
                let (tensors, _meta) = apxinf_loader::safetensors::load_native_path(dir)
                    .map_err(|e| Error::Other(format!("qwen_drive: load planner weights: {e}")))?;
                QwenDriveExpertWeights::from_map(&config, &tensors)
            })
            .transpose()?;
        let weights = QwenDriveDeviceWeights::from_maps(&config, vlm, expert, &*backend)?;
        eprintln!(
            "[qwen_drive] device weights resident (planner={}); allocating caches",
            weights.expert.is_some()
        );
        let max_seq_len = config.text.max_position_embeddings.min(16384); // FIX (implement_r5): 8192 < measured 10457-token VQA prompt (12x868 image tokens + text); 16384 covers the post-clamp-revert worst case 10457+2048=12505 under the config ceiling 32768 (+256MiB full-attn KV cache).
        let caches = Self::fresh_caches(&config, &cuda, max_seq_len)?;
        Ok(Self {
            config,
            backend,
            cuda,
            weights,
            caches,
            cache_len: 0,
            rope_delta: 0,
            max_seq_len,
            last_position: 0,
        })
    }

    fn fresh_caches(
        config: &QwenDriveConfig,
        cuda: &RuntimeBackend,
        max_seq_len: usize,
    ) -> Result<Vec<LayerCache>> {
        let text = &config.text;
        let device = cuda.device_id();
        let mut caches = Vec::with_capacity(text.n_layers);
        for index in 0..text.n_layers {
            if text.is_full_attention(index) {
                let bytes = max_seq_len * text.n_kv_heads * text.head_dim * DType::BF16.size_in_bytes();
                let k = DeviceBuffer::alloc_zeros(bytes, device).map_err(Error::Cuda)?;
                let v = DeviceBuffer::alloc_zeros(bytes, device).map_err(Error::Cuda)?;
                let shape = Shape::new(vec![max_seq_len, text.n_kv_heads, text.head_dim]);
                caches.push(LayerCache::FullAttention {
                    k: k.as_tensor(shape.clone(), DType::BF16).map_err(Error::Cuda)?,
                    v: v.as_tensor(shape, DType::BF16).map_err(Error::Cuda)?,
                });
            } else {
                let conv_dim = 2 * text.linear_num_key_heads * text.linear_key_head_dim
                    + text.linear_num_value_heads * text.linear_value_head_dim;
                let conv_bytes = conv_dim * text.linear_conv_kernel_dim * DType::BF16.size_in_bytes();
                let conv_shape = Shape::new(vec![conv_dim, text.linear_conv_kernel_dim]);
                let conv_a = DeviceBuffer::alloc_zeros(conv_bytes, device).map_err(Error::Cuda)?;
                let conv_b = DeviceBuffer::alloc_zeros(conv_bytes, device).map_err(Error::Cuda)?;
                let rec_bytes = text.linear_num_value_heads
                    * text.linear_key_head_dim
                    * text.linear_value_head_dim
                    * DType::F32.size_in_bytes();
                let recurrent = DeviceBuffer::alloc_zeros(rec_bytes, device).map_err(Error::Cuda)?;
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

    fn reset_state(&mut self) -> Result<()> {
        self.caches = Self::fresh_caches(&self.config, &self.cuda, self.max_seq_len)?;
        self.cache_len = 0;
        self.rope_delta = 0;
        self.last_position = 0;
        Ok(())
    }

    fn ctx(&self) -> &Context {
        self.cuda.context()
    }

    pub fn device(&self) -> Device {
        Device::Cuda(self.cuda.device_id())
    }

    pub fn has_planner(&self) -> bool {
        self.weights.expert.is_some()
    }

    pub fn config(&self) -> &QwenDriveConfig {
        &self.config
    }

    // ---- positions / rope tables ------------------------------------------

    /// Qwen3.5 `get_rope_index`: text tokens take continuing linear
    /// positions; image-token runs take 3D grid positions (T, H/merge,
    /// W/merge) offset by the running position, which then advances by
    /// `max(H, W) // merge`.
    fn rope_index(&self, token_ids: &[u32], grid_thw: &[[u32; 3]]) -> Result<Vec<[u32; 3]>> {
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

    /// Interleaved mRoPE cos/sin tables for a position list, computed on the
    /// host in fp32 and rounded to bf16 on upload (the reference's rounding).
    fn mrope_tables(&self, positions: &[[u32; 3]]) -> Result<(Tensor, Tensor)> {
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
            transfers::to_cuda(&Tensor::from_bf16(vec![positions.len(), rotary], &rounded)?, ctx.device_id())?
        };
        let sin_t = {
            let rounded: Vec<half::bf16> = sin.iter().map(|&v| half::bf16::from_f32(v)).collect();
            transfers::to_cuda(&Tensor::from_bf16(vec![positions.len(), rotary], &rounded)?, ctx.device_id())?
        };
        Ok((cos_t, sin_t))
    }

    // ---- text stack --------------------------------------------------------

    fn lm_head(&self, x: &Tensor) -> Result<Tensor> {
        let ctx = self.ctx();
        let (m, k) = {
            let dims = x.shape().dims();
            if dims.len() != 2 {
                return Err(Error::Other("qwen_drive: lm_head input must be 2D".into()));
            }
            (dims[0], dims[1])
        };
        let vocab = self.config.text.vocab_size;
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

    fn forward_mlp(&self, x: Tensor, post_norm: &Tensor, gate_up_w: &Tensor, down_w: &Tensor) -> Result<Tensor> {
        let ctx = self.ctx();
        let eps = self.config.text.rms_norm_eps;
        let normed = la::rms_norm_plus1(ctx, &x, post_norm, eps)?;
        let gu = gemm::matmul(ctx, &normed, gate_up_w)?;
        let act = activation::swiglu_bf16(ctx, &gu)?;
        let down = gemm::matmul(ctx, &act, down_w)?;
        elementwise::add(ctx, &x, &down)
    }

    fn forward_full_attention(
        &mut self,
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
        let fused = gemm::matmul(ctx, &normed, &w.qkv_w)?;
        let q_out = device_tensor(ctx, &[seq, heads, head_dim], DType::BF16)?;
        let (k_cache, v_cache) = match &self.caches[layer_idx] {
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
            self.cache_len,
            heads,
            kv_heads,
            head_dim,
            rotary,
            eps,
        )?;
        let kv_len = self.cache_len + seq;
        let k_view = cache_view(&k_cache, kv_len)?;
        let v_view = cache_view(&v_cache, kv_len)?;
        let attn = attention::causal_gqa_bf16(ctx, &q_out, &k_view, &v_view, kv_len)?;
        let attn = attn.reshape(vec![seq, heads * head_dim])?;
        la::sigmoid_gate_mul(ctx, &attn, &fused, heads, head_dim)?;
        let proj = gemm::matmul(ctx, &attn, &w.o_w)?;
        let hidden = elementwise::add(ctx, &x, &proj)?;
        self.forward_mlp(hidden, &w.post_norm, &w.gate_up_w, &w.down_w)
    }

    fn forward_gdn(
        &mut self,
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
        let has_state = self.cache_len > 0;

        let normed = la::rms_norm_plus1(ctx, &x, &w.input_norm, eps)?;
        let zba = gemm::matmul(ctx, &normed, &w.qkvzba_w)?;
        let conv_out = device_tensor(ctx, &[seq, conv_dim], DType::BF16)?;
        let (state_current, state_next) = match &self.caches[layer_idx] {
            LayerCache::Gdn { conv_state_a, conv_state_b, flip, .. } => {
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
            if has_state { Some(&state_current) } else { None },
            &conv_out,
            &state_next,
            kernel,
        )?;
        if let LayerCache::Gdn { flip, .. } = &mut self.caches[layer_idx] {
            *flip = !*flip;
        }

        let recurrent_decode = has_state && seq == 1;
        let seq_pad = if recurrent_decode { 1 } else { seq.div_ceil(GDN_CHUNK) * GDN_CHUNK };
        let q_buf = alloc_zeros(ctx, num_v_heads * seq_pad * head_k * DType::F32.size_in_bytes())?;
        let k_buf = alloc_zeros(ctx, num_v_heads * seq_pad * head_k * DType::F32.size_in_bytes())?;
        let v_buf = alloc_zeros(ctx, num_v_heads * seq_pad * head_v * DType::F32.size_in_bytes())?;
        let beta_buf = alloc_zeros(ctx, num_v_heads * seq_pad * DType::F32.size_in_bytes())?;
        let g_buf = alloc_zeros(ctx, num_v_heads * seq_pad * DType::F32.size_in_bytes())?;
        la::gdn_qk_prep(ctx, &conv_out, &q_buf, &k_buf, seq_pad, num_k_heads, num_v_heads, head_k, key_dim, 1e-6)?;
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
        let rec_state = match &self.caches[layer_idx] {
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
            let a_buf = alloc_zeros(ctx, num_v_heads * chunks * GDN_CHUNK * GDN_CHUNK * DType::F32.size_in_bytes())?;
            let t_buf = alloc_zeros(ctx, num_v_heads * chunks * GDN_CHUNK * GDN_CHUNK * DType::F32.size_in_bytes())?;
            let vt_buf = alloc_zeros(ctx, num_v_heads * chunks * GDN_CHUNK * head_v * DType::F32.size_in_bytes())?;
            let kcd_buf = alloc_zeros(ctx, num_v_heads * chunks * GDN_CHUNK * head_k * DType::F32.size_in_bytes())?;
            la::gdn_cumsum(ctx, &g_buf, &g_cum, seq_pad, num_v_heads, GDN_CHUNK)?;
            la::gdn_attn_raw(ctx, &q_buf, &k_buf, &beta_buf, &g_cum, &a_buf, &t_buf, seq_pad, num_v_heads, head_k, GDN_CHUNK)?;
            la::gdn_tri_solve(ctx, &a_buf, num_v_heads * chunks, GDN_CHUNK)?;
            la::gdn_chunk_gemm(ctx, &a_buf, &v_buf, &k_buf, &beta_buf, &g_cum, &vt_buf, &kcd_buf, seq_pad, num_v_heads, head_k, head_v, GDN_CHUNK)?;
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
        let proj = gemm::matmul(ctx, &gated, &w.out_w)?;
        let hidden = elementwise::add(ctx, &x, &proj)?;
        self.forward_mlp(hidden, &w.post_norm, &w.gate_up_w, &w.down_w)
    }

    /// Run the text transformer over one token span, appending to the hybrid
    /// cache. `positions` is the per-token mRoPE position triple.
    fn run_text(&mut self, x: Tensor, positions: &[[u32; 3]]) -> Result<Tensor> {
        let seq = positions.len();
        if seq == 0 {
            return Err(Error::Other("qwen_drive: empty forward span".into()));
        }
        let last = positions[seq - 1];
        self.last_position = last[0].max(last[1]).max(last[2]) as i64;
        let (cos, sin) = self.mrope_tables(positions)?;
        let mut hidden = x;
        for layer_idx in 0..self.config.text.n_layers {
            let is_full = matches!(
                &self.weights.layers[layer_idx],
                MixerWeights::FullAttention(_)
            );
            // TEMP-DIAG (implement_r2): per-layer prefill heartbeat; the last printed k names the hanging layer; revert in the acceptance-bound revision.
            eprintln!("[qwen_drive] prefill_layer k={} kind={}", layer_idx, if is_full { "full" } else { "gdn" });
            hidden = if is_full {
                self.forward_full_attention(hidden, layer_idx, &cos, &sin, seq)?
            } else {
                self.forward_gdn(hidden, layer_idx, seq)?
            };
        }
        self.cache_len += seq;
        let normed = la::rms_norm_plus1(self.ctx(), &hidden, &self.weights.final_norm, self.config.text.rms_norm_eps)?;
        self.lm_head(&normed)
    }

    fn embed_tokens(&self, token_ids: &[u32]) -> Result<Tensor> {
        let ids = upload_u32(self.ctx(), token_ids)?;
        embedding::lookup_bf16(self.ctx(), &self.weights.embed_tokens, &ids, token_ids.len())
    }

    fn upload_pixels(&self, pixels: &Tensor) -> Result<Tensor> {
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

    /// Multimodal prefill: vision tower, image-embedding scatter, text stack.
    fn prefill_impl(
        &mut self,
        token_ids: &[u32],
        image: Option<(&Tensor, &[[u32; 3]])>,
    ) -> Result<Tensor> {
        if token_ids.is_empty() {
            return Err(Error::Other("qwen_drive: empty prompt".into()));
        }
        if self.cache_len != 0 {
            return Err(Error::Other(
                "qwen_drive: prefill requires a fresh cache (call reset first)".into(),
            ));
        }
        let mut x = self.embed_tokens(token_ids)?;
        let empty_grids: &[[u32; 3]] = &[];
        let grids = if let Some((pixels, grid_thw)) = image {
            let pixels = self.upload_pixels(pixels)?;
            let vis = vision::forward(&self.config, &self.weights.vision, self.ctx(), &pixels, grid_thw)?;
            // TEMP-DIAG (implement_r2): vision-tower completion marker; revert in the acceptance-bound revision.
            eprintln!("[qwen_drive] vision_done vision_rows={}", vis.primary.shape().dims()[0]);
            let image_tok = self.config.image_token_id;
            let image_positions = token_ids
                .iter()
                .filter(|&&token| token == image_tok)
                .count();
            let vision_rows = vis.primary.shape().dims()[0];
            if image_positions != vision_rows {
                return Err(Error::Other(format!(
                    "qwen_drive: {image_positions} image tokens but vision produced {vision_rows} rows"
                )));
            }
            let mut ordinal = 0u32;
            let row_map: Vec<u32> = token_ids
                .iter()
                .map(|&token| {
                    if token == image_tok {
                        let row = ordinal;
                        ordinal += 1;
                        row
                    } else {
                        u32::MAX
                    }
                })
                .collect();
            let row_map_dev = upload_u32(self.ctx(), &row_map)?;
            x = elementwise::replace_rows_bf16(self.ctx(), &x, &vis.primary, &row_map_dev)?;
            // TEMP-DIAG (implement_r2): image-embedding scatter completion marker; revert in the acceptance-bound revision.
            eprintln!("[qwen_drive] embed_scatter_done");
            grid_thw
        } else {
            empty_grids
        };
        let positions = self.rope_index(token_ids, grids)?;
        let max_pos = positions
            .iter()
            .map(|triple| triple[0].max(triple[1]).max(triple[2]))
            .max()
            .unwrap_or(0) as i64;
        self.rope_delta = max_pos + 1 - token_ids.len() as i64;
        if self.cache_len + token_ids.len() > self.max_seq_len {
            return Err(Error::Other("qwen_drive: prompt exceeds the cache capacity".into()));
        }
        self.run_text(x, &positions)
    }

    /// Continuation/decode forward over already-tokenized ids.
    fn forward_tokens(&mut self, token_ids: &[u32]) -> Result<Tensor> {
        if token_ids.is_empty() {
            return Err(Error::Other("qwen_drive: empty continuation".into()));
        }
        if self.cache_len + token_ids.len() > self.max_seq_len {
            return Err(Error::Other("qwen_drive: sequence exceeds the cache capacity".into()));
        }
        let positions: Vec<[u32; 3]> = (0..token_ids.len())
            .map(|i| {
                let p = (self.cache_len + i) as i64 + self.rope_delta;
                [p.max(0) as u32; 3]
            })
            .collect();
        let x = self.embed_tokens(token_ids)?;
        self.run_text(x, &positions)
    }

    fn scene_caches(&self) -> Result<Vec<(Tensor, Tensor)>> {
        let mut out = Vec::new();
        for layer_idx in self.config.text.full_attention_layers() {
            match &self.caches[layer_idx] {
                LayerCache::FullAttention { k, v } => {
                    out.push((cache_view(k, self.cache_len)?, cache_view(v, self.cache_len)?));
                }
                _ => return Err(Error::Other("qwen_drive: cache kind mismatch".into())),
            }
        }
        Ok(out)
    }

    /// Greedy generation with optional min-new-token EOS suppression. Returns
    /// the generated token ids (including any terminator, like HF generate).
    pub fn generate(
        &mut self,
        token_ids: &[u32],
        pixel_values: Option<(&Tensor, &[[u32; 3]])>,
        max_new_tokens: usize,
        min_new_tokens: usize,
        eos_token_ids: &[u32],
    ) -> Result<Vec<u32>> {
        // TEMP-DIAG (implement_r2): generate() entry marker (channel vs pre-entry stall disambiguator); revert in the acceptance-bound revision.
        eprintln!("[qwen_drive] gen_entry prompt_tokens={} has_pixels={} max_new_tokens={}", token_ids.len(), pixel_values.is_some(), max_new_tokens);
        self.reset_state()?;
        // TEMP-DIAG (implement_r1): generation-entry clock for prefill/decode timing; revert in the acceptance-bound revision.
        let diag_start = std::time::Instant::now();
        let mut sampler = self.backend.create_token_sampler(TokenSamplingSpec {
            vocab_size: self.config.text.vocab_size,
            max_sequence_len: token_ids.len() + max_new_tokens + 1,
        })?;
        sampler.begin(TokenSamplingInit {
            prompt_token_ids: token_ids,
            params: &TokenSamplingParams::greedy(),
            rng: RngKey::default(),
        })?;
        let mut logits = self.prefill_impl(token_ids, pixel_values)?;
        // TEMP-DIAG (implement_r1): prefill completion marker + decode heartbeat clock; revert in the acceptance-bound revision.
        eprintln!("[qwen_drive] prefill_done prompt_tokens={} elapsed_ms={:.1}", token_ids.len(), diag_start.elapsed().as_secs_f64() * 1000.0);
        let mut diag_last_step = std::time::Instant::now();
        let eos_dev = upload_u32(self.ctx(), eos_token_ids)?;
        let mut generated = Vec::new();
        let mut diag_eos = false;
        for step in 0..max_new_tokens {
            if step < min_new_tokens {
                let row = logits.shape().dims()[0] - 1;
                la::suppress_logits(self.ctx(), &logits, row, &eos_dev)?;
            }
            let sample = sampler.sample(NextTokenLogits::last(&logits, self.config.text.vocab_size)?)?;
            generated.push(sample.token_id);
            // TEMP-DIAG (implement_r1): heartbeat at step 0 and every 25 steps; revert in the acceptance-bound revision.
            if step % 25 == 0 {
                eprintln!("[qwen_drive] decode_step k={} token_id={} ms_since_last={:.1}", step, sample.token_id, diag_last_step.elapsed().as_secs_f64() * 1000.0);
                diag_last_step = std::time::Instant::now();
            }
            if eos_token_ids.contains(&sample.token_id) || step + 1 == max_new_tokens {
                diag_eos = eos_token_ids.contains(&sample.token_id);
                break;
            }
            logits = self.forward_tokens(&[sample.token_id])?;
        }
        // TEMP-DIAG (implement_r1): exit summary with the first 8 generated ids; revert in the acceptance-bound revision.
        eprintln!("[qwen_drive] decode_exit steps={} eos={} first_ids={:?}", generated.len(), diag_eos, &generated[..generated.len().min(8)]);
        Ok(generated)
    }

    fn require_expert(&self) -> Result<&super::device_weights::ExpertDeviceWeights> {
        self.weights.expert.as_ref().ok_or_else(|| {
            Error::Other(
                "qwen_drive: no planning expert loaded; pass a planner directory at load".into(),
            )
        })
    }

    /// Direct planning: prefill the closed-empty-assistant prompt and run the
    /// flow-matching sampler against its scene caches.
    pub fn plan_direct(
        &mut self,
        token_ids: &[u32],
        pixel_values: &Tensor,
        grid_thw: &[[u32; 3]],
        cond: &ExpertConditioning,
        noise: &[f32],
        num_steps: Option<usize>,
    ) -> Result<Vec<f32>> {
        self.reset_state()?;
        self.prefill_impl(token_ids, Some((pixel_values, grid_thw)))?;
        let anchor = self.last_position;
        let scene = self.scene_caches()?;
        let weights = self.require_expert()?;
        expert::plan(
            &self.config,
            weights,
            self.ctx(),
            &ExpertPlan {
                scene: &scene,
                scene_len: self.cache_len,
                anchor,
                cond,
                noise,
                num_steps: num_steps.unwrap_or(self.config.num_inference_steps),
            },
        )
    }

    /// Reasoning planning: greedy assistant turn with min/max token bounds,
    /// cache completion to the trained turn ending, then the sampler. Returns
    /// `(generated_token_ids, normalized_trajectory)`.
    #[allow(clippy::too_many_arguments)]
    pub fn plan_reasoning(
        &mut self,
        token_ids: &[u32],
        pixel_values: &Tensor,
        grid_thw: &[[u32; 3]],
        max_new_tokens: usize,
        min_new_tokens: usize,
        terminator_ids: &[u32],
        im_end_id: u32,
        newline_ids: &[u32],
        cond: &ExpertConditioning,
        noise: &[f32],
        num_steps: Option<usize>,
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        self.reset_state()?;
        let mut sampler = self.backend.create_token_sampler(TokenSamplingSpec {
            vocab_size: self.config.text.vocab_size,
            max_sequence_len: token_ids.len() + max_new_tokens + 1,
        })?;
        sampler.begin(TokenSamplingInit {
            prompt_token_ids: token_ids,
            params: &TokenSamplingParams::greedy(),
            rng: RngKey::default(),
        })?;
        let mut logits = self.prefill_impl(token_ids, Some((pixel_values, grid_thw)))?;
        let prompt_anchor = self.last_position;
        let eos_dev = upload_u32(self.ctx(), terminator_ids)?;
        let mut generated: Vec<u32> = Vec::new();
        for step in 0..max_new_tokens {
            if step < min_new_tokens {
                let row = logits.shape().dims()[0] - 1;
                la::suppress_logits(self.ctx(), &logits, row, &eos_dev)?;
            }
            let sample = sampler.sample(NextTokenLogits::last(&logits, self.config.text.vocab_size)?)?;
            generated.push(sample.token_id);
            if terminator_ids.contains(&sample.token_id) || step + 1 == max_new_tokens {
                break;
            }
            logits = self.forward_tokens(&[sample.token_id])?;
        }
        // Close the turn in the cache exactly like the reference.
        let mut content = generated.clone();
        for (position, token) in generated.iter().enumerate() {
            if terminator_ids.contains(token) {
                content.truncate(position);
                break;
            }
        }
        let mut closed_turn = content;
        closed_turn.push(im_end_id);
        closed_turn.extend_from_slice(newline_ids);
        let already_cached = generated.len().saturating_sub(1);
        if already_cached < closed_turn.len() {
            let pending = closed_turn[already_cached..].to_vec();
            self.forward_tokens(&pending)?;
        }
        let anchor = prompt_anchor + closed_turn.len() as i64;
        let scene = self.scene_caches()?;
        let weights = self.require_expert()?;
        let trajectory = expert::plan(
            &self.config,
            weights,
            self.ctx(),
            &ExpertPlan {
                scene: &scene,
                scene_len: self.cache_len,
                anchor,
                cond,
                noise,
                num_steps: num_steps.unwrap_or(self.config.num_inference_steps),
            },
        )?;
        Ok((generated, trajectory))
    }
}

impl LlmTrait for QwenDriveModel {
    fn load(
        _config: ModelConfig,
        _weights: HashMap<String, Tensor>,
        _device: Device,
    ) -> Result<Self>
    where
        Self: Sized,
    {
        Err(Error::Other(
            "QwenDriveModel::load(ModelConfig) is not supported; use \
             QwenDriveModel::load(dir, planner, device) or the registry loader"
                .into(),
        ))
    }

    /// Token-level forward over the internal hybrid cache. `start_pos` is
    /// advisory: positions derive from the model-owned cache length plus the
    /// multimodal rope delta, matching the shared generation loop's usage.
    fn forward(&mut self, token_ids: &[u32], _start_pos: u32) -> Result<Tensor> {
        self.forward_tokens(token_ids)
    }

    fn backend(&self) -> &dyn Backend {
        &*self.backend
    }

    fn capabilities(&self) -> LlmCapabilities {
        LlmCapabilities::VISION
    }

    fn prefill(&mut self, input: LlmInput<'_>) -> Result<Tensor> {
        self.reset_state()?;
        match input.image {
            Some(image) => self.prefill_impl(input.token_ids, Some((image.pixel_values, image.grid_thw))),
            None => self.prefill_impl(input.token_ids, None),
        }
    }

    fn reset(&mut self) {
        let _ = self.reset_state();
    }

    fn vocab_size(&self) -> usize {
        self.config.text.vocab_size
    }
}
