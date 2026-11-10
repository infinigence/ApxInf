//! GeneralQwen3VL — device-agnostic Qwen3-VL model driven by `dyn Backend`.
//!
//! Text stack only (Phase 3). Vision + multimodal wiring come in later
//! phases. Uses the primitive ops (matmul, rms_norm, rope_mrope, sdpa) from
//! the Backend trait — no direct CUDA dependency here.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use apxinf_core::{Backend, Device, Error, KvCache, Result, Tensor};
use apxinf_loader::safetensors;

use super::config::Qwen3VLConfig;
use super::vision::{self, VisionOutput};
use super::vision_weights::{transfer_vision_weights, Qwen3VLVisionWeights};
use super::weights::Qwen3VLTextWeights;
use crate::accelerator::create_backend;
#[cfg(feature = "cuda")]
use crate::accelerator::cuda::downcast as cuda_backend;
use crate::llm_trait::{LlmCapabilities, LlmInput, LlmTrait};

pub struct GeneralQwen3VL {
    config: Qwen3VLConfig,
    weights: Qwen3VLTextWeights,
    vision_weights: Qwen3VLVisionWeights,
    /// Pre-transposed `token_embedding` for the tied lm_head matmul. Shape
    /// `[hidden, vocab]`, on the backend's device. Same dtype as
    /// `token_embedding`. Cached once at load so the transpose is not on
    /// the hot path.
    lm_head: Tensor,
    backend: Arc<dyn Backend>,
    kv: Box<dyn KvCache>,
    /// mRoPE position delta set after a multimodal prefill. For decode
    /// tokens, the mRoPE position = linear_position + rope_delta. 0 for
    /// text-only (no image). Matches HF's `rope_deltas`.
    rope_delta: i64,
    /// Allocation-free decode fast path (CUDA + bf16 only). `None` on CPU,
    /// on fp32 CUDA, or if fused weight packing failed. When present,
    /// `forward` uses it for seq_len=1 decode.
    #[cfg(feature = "cuda")]
    decode_graph: Option<super::decode_graph::Qwen3VLDecodeGraph>,
}

impl GeneralQwen3VL {
    /// Load Qwen3-VL from a model directory containing `config.json` and
    /// `model.safetensors`. Weights are always loaded native (no bf16→f32
    /// upcast) because the CUDA text path is bf16-only.
    pub fn from_dir(model_dir: &Path, device: Device) -> Result<Self> {
        let cfg_path = model_dir.join("config.json");
        let cfg = Qwen3VLConfig::from_json_file(&cfg_path)?;
        let (tensors, _meta) = safetensors::load_native_path(model_dir)
            .map_err(|e| Error::Other(format!("load safetensors: {e}")))?;
        Self::from_weights(cfg, tensors, device)
    }

    /// Build from a pre-parsed config + weight map + device. Useful for
    /// tests that construct weights in-memory.
    pub fn from_weights(
        config: Qwen3VLConfig,
        tensors: HashMap<String, Tensor>,
        device: Device,
    ) -> Result<Self> {
        let backend = create_backend(device)?;
        Self::from_weights_with_backend(config, tensors, backend)
    }

    pub(crate) fn from_weights_with_backend(
        config: Qwen3VLConfig,
        tensors: HashMap<String, Tensor>,
        backend: Arc<dyn Backend>,
    ) -> Result<Self> {
        let weights = Qwen3VLTextWeights::from_map(&config, tensors.clone())?;
        let vision_weights = Qwen3VLVisionWeights::from_map(&config, tensors)?;
        // Transfer text weights to backend's device.
        let mut weights = transfer_weights(&weights, &*backend)?;
        let vision_weights = transfer_vision_weights(&vision_weights, &*backend)?;

        // Cache the transposed embedding for the tied lm_head matmul.
        let lm_head = transpose_tensor_bf16_or_f32(&weights.token_embedding, &*backend)?;

        // Build fused weight matrices (qkv_packed, gate_up_packed) for the
        // fused-GEMM decode path. No-ops on backends without concat_2d.
        pack_fused_weights(&mut weights, &*backend);

        let kv = backend.create_kv_cache(
            config.text.n_layers,
            config.text.n_kv_heads,
            config.text.head_dim,
            config.text.max_position_embeddings.min(4096),
        );

        // Create the CUDA decode graph if the backend is CUDA and the
        // weights are bf16 (Qwen3-VL fast path is bf16-only).
        #[cfg(feature = "cuda")]
        let decode_graph = {
            use apxinf_core::DType;
            let dtype = weights.token_embedding.dtype();
            if dtype == DType::BF16 {
                let cfg = Self::decode_graph_cfg(&config, dtype);
                cuda_backend(&*backend)
                    .map(|cb| super::decode_graph::Qwen3VLDecodeGraph::new(cb, cfg))
                    .transpose()?
            } else {
                None
            }
        };

        Ok(Self {
            config,
            weights,
            vision_weights,
            lm_head,
            backend,
            kv,
            rope_delta: 0,
            #[cfg(feature = "cuda")]
            decode_graph,
        })
    }

    #[cfg(feature = "cuda")]
    fn decode_graph_cfg(
        config: &Qwen3VLConfig,
        dtype: apxinf_core::DType,
    ) -> super::decode_graph::Qwen3VLDecodeGraphConfig {
        let tc = &config.text;
        super::decode_graph::Qwen3VLDecodeGraphConfig {
            n_layers: tc.n_layers,
            n_heads: tc.n_heads,
            n_kv_heads: tc.n_kv_heads,
            head_dim: tc.head_dim,
            hidden_size: tc.hidden_size,
            intermediate_size: tc.intermediate_size,
            vocab_size: tc.vocab_size,
            max_seq_len: tc.max_position_embeddings.min(4096),
            rope_theta: tc.rope_theta,
            mrope_section: tc.mrope_section,
            rms_norm_eps: tc.rms_norm_eps,
            dtype,
        }
    }

    /// Access the backend (for tests that need to upload tensors).
    pub fn backend(&self) -> &dyn Backend {
        &*self.backend
    }

    /// Access the config (for debug tools).
    pub fn config_ref(&self) -> &Qwen3VLConfig {
        &self.config
    }

    /// Access the vision weights (for debug tools).
    pub fn vision_weights_ref(&self) -> &Qwen3VLVisionWeights {
        &self.vision_weights
    }

    /// Run the vision tower. `pixel_values` is `[N, 1536]` bf16 on device;
    /// `grid_thw` is `[[T, H, W]]`. Returns primary + 3 deepstack embeddings.
    pub fn forward_vision(
        &self,
        pixel_values: &Tensor,
        grid_thw: &[[u32; 3]],
    ) -> Result<VisionOutput> {
        vision::forward(
            &self.config,
            &self.vision_weights,
            &*self.backend,
            pixel_values,
            grid_thw,
        )
    }

    /// Compute 3D mRoPE position IDs for a text+image prompt.
    /// Returns a flat `[seq_len * 3]` u32 slice (t, h, w per token).
    /// Matches HF's `get_rope_index`. For text tokens (including
    /// vision_start/vision_end), all three axes equal the linear position.
    /// For image_pad tokens, the axes follow the 2D spatial grid.
    pub fn get_rope_index(&self, token_ids: &[u32], grid_thw: &[[u32; 3]]) -> Vec<u32> {
        let merge = self.config.vision.spatial_merge_size as u32;
        let image_tok = self.config.image_token_id;
        let mut pos_ids: Vec<u32> = Vec::with_capacity(token_ids.len() * 3);

        let mut st = 0usize;
        let mut image_index = 0usize;
        let mut next_pos: u32 = 0; // Tracks the next linear position to assign

        loop {
            // Find next image_pad from st.
            let ed = (st..token_ids.len()).find(|&i| token_ids[i] == image_tok);
            let Some(ed) = ed else {
                // Remaining text — all (p, p, p)
                for _ in st..token_ids.len() {
                    pos_ids.extend_from_slice(&[next_pos, next_pos, next_pos]);
                    next_pos += 1;
                }
                break;
            };
            // Text before image_pad: linear positions.
            for _ in st..ed {
                pos_ids.extend_from_slice(&[next_pos, next_pos, next_pos]);
                next_pos += 1;
            }
            // Image tokens: 2D grid positions.
            let (t, h, w) = (
                grid_thw[image_index][0],
                grid_thw[image_index][1] / merge,
                grid_thw[image_index][2] / merge,
            );
            image_index += 1;
            let n_img = (t * h * w) as usize;
            for ti in 0..t {
                for hi in 0..h {
                    for wi in 0..w {
                        pos_ids.extend_from_slice(&[next_pos + ti, next_pos + hi, next_pos + wi]);
                    }
                }
            }
            // The image positions' max is next_pos + max(t-1, h-1, w-1).
            // HF's next text position = max(image_positions) + 1.
            let max_img = next_pos + [t - 1, h - 1, w - 1].iter().max().copied().unwrap();
            next_pos = max_img + 1;
            st = ed + n_img;
        }
        pos_ids
    }

    /// Multimodal prefill: run vision tower, inject embeddings, run text transformer.
    /// `token_ids` is the full chat-templated prompt (including image_pad tokens).
    /// `pixel_values` is `[N, 1536]` bf16 on CPU or the model device;
    /// `grid_thw` is `[[T, H, W]]`.
    fn prefill_with_image(
        &mut self,
        token_ids: &[u32],
        pixel_values: &Tensor,
        grid_thw: &[[u32; 3]],
    ) -> Result<Tensor> {
        let _img_range = crate::profiling::trace::range("multimodal_forward");
        let seq_len = token_ids.len();
        if seq_len == 0 {
            return Err(Error::Other("prefill: empty token_ids".into()));
        }
        if grid_thw.is_empty() {
            return Err(Error::Other(
                "Qwen3-VL image input requires at least one grid_thw entry".into(),
            ));
        }

        let merge = self.config.vision.spatial_merge_size as u32;
        if merge == 0 {
            return Err(Error::Other(
                "Qwen3-VL spatial_merge_size must be greater than zero".into(),
            ));
        }
        let expected_image_tokens = grid_thw.iter().try_fold(0usize, |total, &[t, h, w]| {
            if t == 0 || h == 0 || w == 0 || h % merge != 0 || w % merge != 0 {
                return Err(Error::Other(format!(
                    "invalid Qwen3-VL image grid [{t}, {h}, {w}] for merge size {merge}"
                )));
            }
            let grid_tokens = (t as usize)
                .checked_mul((h / merge) as usize)
                .and_then(|value| value.checked_mul((w / merge) as usize))
                .ok_or_else(|| Error::Other("Qwen3-VL image grid is too large".into()))?;
            total
                .checked_add(grid_tokens)
                .ok_or_else(|| Error::Other("Qwen3-VL image token count overflow".into()))
        })?;
        let actual_image_tokens = token_ids
            .iter()
            .filter(|&&token| token == self.config.image_token_id)
            .count();
        if actual_image_tokens != expected_image_tokens {
            return Err(Error::Other(format!(
                "image_pad count {actual_image_tokens} != grid image tokens {expected_image_tokens}"
            )));
        }

        // The public interface also accepts CPU processor output. Transfer is
        // performed once at prefill and never appears in autoregressive decode.
        let uploaded_pixels = if pixel_values.device() != self.backend.device() {
            Some(self.backend.to_device(pixel_values)?)
        } else {
            None
        };
        let pixel_values = uploaded_pixels.as_ref().unwrap_or(pixel_values);

        // Run vision tower once for this prompt.
        let _vis_range = crate::profiling::trace::range("vision");
        let vis = self.forward_vision(pixel_values, grid_thw)?;
        drop(_vis_range);

        // Embedding lookup.
        let mut x = self
            .backend
            .embedding(&self.weights.token_embedding, token_ids)?;

        // Replace image_pad embeddings with vision primary output.
        let image_tok = self.config.image_token_id;
        let n_img_tokens = vis.primary.shape().dims()[0];
        let mut img_positions: Vec<usize> = Vec::with_capacity(n_img_tokens);
        for (i, &tok) in token_ids.iter().enumerate() {
            if tok == image_tok {
                img_positions.push(i);
            }
        }
        if img_positions.len() != n_img_tokens {
            return Err(Error::Other(format!(
                "image_pad count {} != vision primary tokens {}",
                img_positions.len(),
                n_img_tokens
            )));
        }
        x = scatter_add(&x, &img_positions, &vis.primary, &*self.backend)?;

        // mRoPE position IDs (3D for image tokens).
        let pos_ids = self.get_rope_index(token_ids, grid_thw);
        // Compute rope_delta = max(mRoPE positions) + 1 - seq_len, matching
        // HF's `mrope_position_deltas`. Used by decode to offset the linear
        // position into mRoPE space.
        let max_mrope = pos_ids
            .chunks(3)
            .map(|c| c[0].max(c[1]).max(c[2]))
            .max()
            .unwrap_or(0) as i64;
        self.rope_delta = max_mrope + 1 - seq_len as i64;

        // Run layers 0..n_layers, injecting deepstack at layers 0, 1, 2.
        let _prefill_range = crate::profiling::trace::range("prefill");
        for i in 0..self.config.text.n_layers {
            let _layer_range = crate::profiling::trace::range(&format!("layer_{i}"));
            x = self.forward_layer(&x, i, &pos_ids, 0)?;
            // Deepstack injection at layers 0, 1, 2.
            if i < vis.deepstack.len() {
                x = scatter_add(&x, &img_positions, &vis.deepstack[i], &*self.backend)?;
            }
        }

        self.kv.advance(seq_len);

        // Final norm + tied lm_head.
        let x = self.backend.rms_norm(
            &x,
            &self.weights.output_norm_weight,
            self.config.text.rms_norm_eps,
        )?;
        let logits = self.backend.matmul(&x, &self.lm_head)?;
        self.backend.synchronize()?;
        self.backend.to_cpu(&logits)
    }

    /// Text-only mRoPE position IDs. For a token at index `t` (absolute
    /// position `start_pos + t`), Qwen3-VL uses `(pos, pos, pos)` (all
    /// three axes equal) so mRoPE degenerates to 1-D RoPE. Vision phase
    /// will replace this with `get_rope_index` — see plan.md.
    fn text_only_pos_ids(&self, seq_len: usize, start_pos: u32) -> Vec<u32> {
        // After a multimodal prefill, rope_delta offsets the linear position
        // into mRoPE space (image tokens share 2D positions, so the mRoPE
        // position lags behind the linear position).
        let mrope_start = (start_pos as i64 + self.rope_delta) as u32;
        let mut out = Vec::with_capacity(seq_len * 3);
        for i in 0..seq_len {
            let p = mrope_start + i as u32;
            out.extend_from_slice(&[p, p, p]);
        }
        out
    }

    fn forward_layer(
        &mut self,
        x: &Tensor,
        layer_idx: usize,
        pos_ids: &[u32],
        start_pos: u32,
    ) -> Result<Tensor> {
        let seq_len = x.shape().dims()[0];
        let tc = &self.config.text;
        let n_heads = tc.n_heads;
        let n_kv_heads = tc.n_kv_heads;
        let head_dim = tc.head_dim;
        let b = &*self.backend;
        let layer = &self.weights.layers[layer_idx];

        // Pre-attention RMSNorm.
        let normed = b.rms_norm(x, &layer.attn_norm_weight, tc.rms_norm_eps)?;

        // Q/K/V projections. wq/wk/wv are transposed → [hidden_in, out].
        let q = b.matmul(&normed, &layer.wq)?;
        let k = b.matmul(&normed, &layer.wk)?;
        let v = b.matmul(&normed, &layer.wv)?;

        // Reshape to [seq_len, heads, head_dim] then to
        // [seq_len * heads, head_dim] for per-head QK-norm.
        let q = q.reshape(vec![seq_len * n_heads, head_dim])?;
        let k = k.reshape(vec![seq_len * n_kv_heads, head_dim])?;
        let v = v.reshape(vec![seq_len, n_kv_heads, head_dim])?;

        // QK-norm (per-head RMSNorm, weight shape [head_dim]).
        let q = b.rms_norm(&q, &layer.q_norm_weight, tc.rms_norm_eps)?;
        let k = b.rms_norm(&k, &layer.k_norm_weight, tc.rms_norm_eps)?;

        // Reshape back to [seq_len, heads, head_dim] for mRoPE.
        let q = q.reshape(vec![seq_len, n_heads, head_dim])?;
        let k = k.reshape(vec![seq_len, n_kv_heads, head_dim])?;

        // mRoPE (rotate_half + axis-interleaved). Text-only → all axes = pos.
        let q = b.rope_mrope(
            &q,
            n_heads,
            head_dim,
            tc.rope_theta,
            tc.mrope_section,
            pos_ids,
        )?;
        let k = b.rope_mrope(
            &k,
            n_kv_heads,
            head_dim,
            tc.rope_theta,
            tc.mrope_section,
            pos_ids,
        )?;

        // Append to KV cache.
        b.kv_append(&mut *self.kv, layer_idx, &k, &v, seq_len)?;

        // Attention.
        let kv_len = self.kv.seq_len() + seq_len;
        let attn_out = if seq_len == 1 {
            b.sdpa_decode(
                &q,
                &mut *self.kv,
                layer_idx,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_len,
                tc.max_position_embeddings.min(4096),
            )?
        } else {
            b.sdpa_prefill(
                &q,
                &mut *self.kv,
                layer_idx,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_len,
                tc.max_position_embeddings.min(4096),
            )?
        };

        // Output projection + residual.
        let attn_out = attn_out.reshape(vec![seq_len, n_heads * head_dim])?;
        let attn_out = b.matmul(&attn_out, &layer.wo)?;
        let x = b.add(x, &attn_out)?;

        // Pre-FFN RMSNorm + SwiGLU MLP.
        let normed = b.rms_norm(&x, &layer.ffn_norm_weight, tc.rms_norm_eps)?;
        let gate = b.matmul(&normed, &layer.w_gate)?;
        let gate = b.silu(&gate)?;
        let up = b.matmul(&normed, &layer.w_up)?;
        let hidden = b.mul(&gate, &up)?;
        let mlp_out = b.matmul(&hidden, &layer.w_down)?;

        // Residual.
        let _ = start_pos; // unused for now; kept for parity with Llama
        b.add(&x, &mlp_out)
    }
}

impl LlmTrait for GeneralQwen3VL {
    /// Stub: Qwen3-VL uses its own config schema; call `from_dir` /
    /// `from_weights` directly. `AutoModel` uses its registered loader.
    fn load(
        _config: apxinf_loader::ModelConfig,
        _weights: HashMap<String, Tensor>,
        _device: Device,
    ) -> Result<Self>
    where
        Self: Sized,
    {
        Err(Error::Other(
            "GeneralQwen3VL::load(ModelConfig) not supported; use GeneralQwen3VL::from_dir or from_weights"
                .into()))
    }

    fn forward(&mut self, token_ids: &[u32], start_pos: u32) -> Result<Tensor> {
        let seq_len = token_ids.len();
        if seq_len == 0 {
            return Err(Error::Other("forward: empty token_ids".into()));
        }

        let _decode_range =
            crate::profiling::trace::range(if seq_len == 1 { "decode" } else { "prefill" });

        // Decode fast path: seq_len=1 with the CUDA decode graph.
        #[cfg(feature = "cuda")]
        if seq_len == 1 {
            if let Some(dg) = self.decode_graph.as_mut() {
                use super::decode_graph::{Qwen3VLDecodeGraphWeights, Qwen3VLDecodeLayerWeights};
                let weights = Qwen3VLDecodeGraphWeights {
                    token_embedding: &self.weights.token_embedding,
                    layers: self
                        .weights
                        .layers
                        .iter()
                        .map(|l| Qwen3VLDecodeLayerWeights {
                            attn_norm_weight: &l.attn_norm_weight,
                            wq: &l.wq,
                            wk: &l.wk,
                            wv: &l.wv,
                            wo: &l.wo,
                            ffn_norm_weight: &l.ffn_norm_weight,
                            w_gate: &l.w_gate,
                            w_up: &l.w_up,
                            w_down: &l.w_down,
                            q_norm_weight: &l.q_norm_weight,
                            k_norm_weight: &l.k_norm_weight,
                            qkv_packed: l.qkv_packed.as_ref(),
                            gate_up_packed: l.gate_up_packed.as_ref(),
                        })
                        .collect(),
                    output_norm_weight: &self.weights.output_norm_weight,
                    lm_head: &self.lm_head,
                };
                let cb = cuda_backend(&*self.backend).expect("decode_graph requires CudaBackend");
                // For decode: mRoPE position = linear + rope_delta (text-only
                // after multimodal prefill), all three axes equal.
                // Cache position = linear (unshifted).
                let mrope_pos = (start_pos as i64 + self.rope_delta) as u32;
                let cache_pos = start_pos;
                let logits = dg.decode(
                    cb,
                    &weights,
                    &mut *self.kv,
                    token_ids[0],
                    [mrope_pos, mrope_pos, mrope_pos],
                    cache_pos,
                )?;
                self.kv.advance(1);
                return Ok(logits);
            }
        }

        // Fallback: dyn Backend op-by-op path (prefill or non-CUDA).
        // Embedding lookup.
        let mut x = self
            .backend
            .embedding(&self.weights.token_embedding, token_ids)?;

        // mRoPE position IDs (text-only for now).
        let pos_ids = self.text_only_pos_ids(seq_len, start_pos);

        for i in 0..self.config.text.n_layers {
            let _layer_range = crate::profiling::trace::range(&format!("layer_{i}"));
            x = self.forward_layer(&x, i, &pos_ids, start_pos)?;
        }

        self.kv.advance(seq_len);

        // Final norm + tied lm_head. The output projection is
        // token_embedding^T: cuBLAS row-major GEMM against [vocab, hidden]
        // gives [seq_len, vocab] directly.
        let x = self.backend.rms_norm(
            &x,
            &self.weights.output_norm_weight,
            self.config.text.rms_norm_eps,
        )?;
        let logits = self.backend.matmul(&x, &self.lm_head)?;

        self.backend.synchronize()?;
        self.backend.to_cpu(&logits)
    }

    fn capabilities(&self) -> LlmCapabilities {
        LlmCapabilities::VISION
    }

    fn prefill(&mut self, input: LlmInput<'_>) -> Result<Tensor> {
        match input.image {
            Some(image) => {
                self.prefill_with_image(input.token_ids, image.pixel_values, image.grid_thw)
            }
            None => self.forward(input.token_ids, 0),
        }
    }

    fn reset(&mut self) {
        let _ = self.kv.clear();
        self.rope_delta = 0;
    }

    fn vocab_size(&self) -> usize {
        self.config.text.vocab_size
    }
}

/// Transfer text weights to backend's device.
fn transfer_weights(w: &Qwen3VLTextWeights, backend: &dyn Backend) -> Result<Qwen3VLTextWeights> {
    let layers = w
        .layers
        .iter()
        .map(|l| {
            Ok::<_, Error>(super::weights::Qwen3VLLayer {
                attn_norm_weight: backend.to_device(&l.attn_norm_weight)?,
                wq: backend.to_device(&l.wq)?,
                wk: backend.to_device(&l.wk)?,
                wv: backend.to_device(&l.wv)?,
                wo: backend.to_device(&l.wo)?,
                q_norm_weight: backend.to_device(&l.q_norm_weight)?,
                k_norm_weight: backend.to_device(&l.k_norm_weight)?,
                ffn_norm_weight: backend.to_device(&l.ffn_norm_weight)?,
                w_gate: backend.to_device(&l.w_gate)?,
                w_up: backend.to_device(&l.w_up)?,
                w_down: backend.to_device(&l.w_down)?,
                qkv_packed: None,
                gate_up_packed: None,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Qwen3VLTextWeights {
        token_embedding: backend.to_device(&w.token_embedding)?,
        layers,
        output_norm_weight: backend.to_device(&w.output_norm_weight)?,
    })
}

/// Transpose an on-device 2D tensor by round-tripping through CPU. Slow;
/// only used for the tied-embedding lm_head materialization above until we
/// cache the transposed copy. Handles both F32 and BF16.
fn transpose_tensor_bf16_or_f32(t: &Tensor, backend: &dyn Backend) -> Result<Tensor> {
    let cpu = backend.to_cpu(t)?;
    let dims = cpu.shape().dims();
    if dims.len() != 2 {
        return Err(Error::Other("transpose expected 2D".into()));
    }
    let (rows, cols) = (dims[0], dims[1]);
    let transposed = match cpu.dtype() {
        apxinf_core::DType::F32 => {
            let src = cpu.as_f32()?;
            let mut out = vec![0.0f32; rows * cols];
            for i in 0..rows {
                for j in 0..cols {
                    out[j * rows + i] = src[i * cols + j];
                }
            }
            Tensor::from_f32(vec![cols, rows], &out)?
        }
        apxinf_core::DType::BF16 => {
            let src = cpu.as_bf16()?;
            let mut out = vec![half::bf16::from_f32(0.0); rows * cols];
            for i in 0..rows {
                for j in 0..cols {
                    out[j * rows + i] = src[i * cols + j];
                }
            }
            Tensor::from_bf16(vec![cols, rows], &out)?
        }
        dtype => {
            return Err(Error::Other(format!(
                "Qwen3-VL tied embedding transpose does not support {dtype}"
            )))
        }
    };
    backend.to_device(&transposed)
}

/// Build fused weight matrices (qkv_packed, gate_up_packed) for the fused
/// decode path. No-op if `Backend::concat_2d` returns Unsupported (CPU).
fn pack_fused_weights(weights: &mut Qwen3VLTextWeights, backend: &dyn Backend) {
    for layer in &mut weights.layers {
        if let Ok(packed) = backend.concat_2d(&[&layer.wq, &layer.wk, &layer.wv]) {
            layer.qkv_packed = Some(packed);
        }
        if let Ok(packed) = backend.concat_2d(&[&layer.w_gate, &layer.w_up]) {
            layer.gate_up_packed = Some(packed);
        }
    }
}

/// Scatter-add: for each (i, row) in positions, x[positions[i], :] += src[i, :].
/// Used to inject vision embeddings at image_pad positions. Done on CPU
/// (round-trip through to_cpu/to_device) since the Backend trait has no
/// scatter op. One-time per forward, not on the hot path.
fn scatter_add(
    x: &Tensor,
    positions: &[usize],
    src: &Tensor,
    backend: &dyn Backend,
) -> Result<Tensor> {
    let x_cpu = backend.to_cpu(x)?;
    let src_cpu = backend.to_cpu(src)?;
    let dims = x_cpu.shape().dims().to_vec();
    let hidden = dims[dims.len() - 1];
    match x_cpu.dtype() {
        apxinf_core::DType::F32 => {
            let mut data = x_cpu.as_f32()?.to_vec();
            let src_data = src_cpu.to_f32_vec()?;
            for (i, &pos) in positions.iter().enumerate() {
                for c in 0..hidden {
                    data[pos * hidden + c] += src_data[i * hidden + c];
                }
            }
            let out = Tensor::from_f32(dims, &data)?;
            backend.to_device(&out)
        }
        apxinf_core::DType::BF16 => {
            // Work in f32 for the addition, then cast back to bf16.
            let mut data = x_cpu.to_f32_vec()?;
            let src_data = src_cpu.to_f32_vec()?;
            for (i, &pos) in positions.iter().enumerate() {
                for c in 0..hidden {
                    data[pos * hidden + c] += src_data[i * hidden + c];
                }
            }
            let bf16: Vec<half::bf16> = data.iter().map(|&v| half::bf16::from_f32(v)).collect();
            let out = Tensor::from_bf16(dims, &bf16)?;
            backend.to_device(&out)
        }
        dtype => Err(Error::Other(format!(
            "Qwen3-VL scatter does not support {dtype}"
        ))),
    }
}

// The GenerationProfile trait sits on LlmTrait's default `generate_streaming`
// impl. No re-export needed — callers just `use apxinf_model::LlmTrait`.
