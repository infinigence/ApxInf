//! Llama legacy model (CPU/CUDA full-stack impl).
//!
//! This is the pre-Backend-trait implementation. The modern path uses
//! `GeneralLlama` (in `general.rs`) which goes through `dyn Backend`.

use std::collections::HashMap;

use apxinf_core::{Device, Error, Result, Tensor};
use apxinf_loader::ModelConfig;

use crate::debug::DebugCapture;
use crate::profiling::GenerationProfile;

#[cfg(feature = "cuda")]
use std::sync::Arc;
#[cfg(feature = "cuda")]
use crate::accelerator::cuda::{
    kernels as cuda_kernels, transfers as cuda_ops, Context as CudaContext,
    DeviceBuffer as CudaBuffer, KvCache as CudaKVCache,
};
#[cfg(feature = "cuda")]
use apxinf_core::KvCache;

use super::weights::{LlamaWeights, TransformerLayer};

/// Llama model weights and architecture.
pub struct LlamaModel {
    pub config: ModelConfig,
    pub weights: LlamaWeights,
    #[cfg(feature = "cuda")]
    cuda_ctx: Option<Arc<CudaContext>>,
    #[cfg(feature = "cuda")]
    cuda_device_id: Option<usize>,
    #[cfg(feature = "cuda")]
    cuda_kv_cache: Option<CudaKVCache>,
}

/// All weights for a Llama model.
impl LlamaModel {
    /// Load a Llama model from a weight map (CPU only).
    ///
    /// Use `to_device()` to transfer to a CUDA GPU.
    ///
    /// Expected weight naming convention (HuggingFace style):
    /// - `model.embed_tokens.weight` — token embeddings [vocab_size, hidden_size]
    /// - `model.layers.{i}.input_layernorm.weight` — attention norm
    /// - `model.layers.{i}.self_attn.q_proj.weight` — query projection
    /// - `model.layers.{i}.self_attn.k_proj.weight` — key projection
    /// - `model.layers.{i}.self_attn.v_proj.weight` — value projection
    /// - `model.layers.{i}.self_attn.o_proj.weight` — output projection
    /// - `model.layers.{i}.post_attention_layernorm.weight` — FFN norm
    /// - `model.layers.{i}.mlp.gate_proj.weight` — MLP gate
    /// - `model.layers.{i}.mlp.up_proj.weight` — MLP up
    /// - `model.layers.{i}.mlp.down_proj.weight` — MLP down
    /// - `model.norm.weight` — final norm
    /// - `lm_head.weight` — output projection to vocab
    pub fn from_weights(config: ModelConfig, tensors: HashMap<String, Tensor>) -> Result<Self> {
        let weights = LlamaWeights::from_map(&config, tensors)?;
        Ok(Self {
            config,
            weights,
            #[cfg(feature = "cuda")]
            cuda_ctx: None,
            #[cfg(feature = "cuda")]
            cuda_device_id: None,
            #[cfg(feature = "cuda")]
            cuda_kv_cache: None,
        })
    }

    /// Transfer model weights to the specified device.
    ///
    /// For CUDA, initializes the GPU context and kernels.
    #[cfg(feature = "cuda")]
    pub fn to_device(&mut self, device: Device) -> Result<()> {
        match device {
            Device::Cpu => Ok(()),
            Device::Cuda(device_id) => {
                let ctx = CudaContext::new(device_id)
                    .map_err(|e| Error::Cuda(format!("CUDA init: {e}")))?;
                let ctx = Arc::new(ctx);

                self.weights.token_embedding = cuda_ops::to_cuda(&self.weights.token_embedding, device_id)?;
                self.weights.output_norm_weight = cuda_ops::to_cuda(&self.weights.output_norm_weight, device_id)?;
                self.weights.output_weight = cuda_ops::to_cuda(&self.weights.output_weight, device_id)?;

                for layer in &mut self.weights.layers {
                    layer.attn_norm_weight = cuda_ops::to_cuda(&layer.attn_norm_weight, device_id)?;
                    layer.wq = cuda_ops::to_cuda(&layer.wq, device_id)?;
                    layer.wk = cuda_ops::to_cuda(&layer.wk, device_id)?;
                    layer.wv = cuda_ops::to_cuda(&layer.wv, device_id)?;
                    layer.wo = cuda_ops::to_cuda(&layer.wo, device_id)?;
                    layer.ffn_norm_weight = cuda_ops::to_cuda(&layer.ffn_norm_weight, device_id)?;
                    layer.w_gate = cuda_ops::to_cuda(&layer.w_gate, device_id)?;
                    layer.w_up = cuda_ops::to_cuda(&layer.w_up, device_id)?;
                    layer.w_down = cuda_ops::to_cuda(&layer.w_down, device_id)?;
                }

                self.cuda_ctx = Some(ctx);
                self.cuda_device_id = Some(device_id);
                self.cuda_kv_cache = None;
                Ok(())
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    pub fn to_device(&mut self, device: Device) -> Result<()> {
        match device {
            Device::Cpu => Ok(()),
            Device::Cuda(_) => Err(Error::Other("CUDA not compiled in".into())),
        }
    }

    /// Get the device the model is currently on.
    pub fn device(&self) -> Device {
        self.weights.token_embedding.device()
    }

    /// Generate tokens autoregressively, calling `on_token` for each generated token.
    ///
    /// `prompt_tokens`: initial prompt tokens
    /// `max_new_tokens`: maximum number of new tokens to generate
    /// `on_token`: called with each generated token ID immediately after it is sampled
    /// `debug`: optional debug capture for activations
    /// `eos_token_id`: optional EOS token ID for early stopping (generation stops when EOS is produced)
    ///
    /// Returns generated token IDs (excluding prompt) and a generation profile.
    pub fn generate_streaming(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        mut on_token: impl FnMut(u32),
        mut debug: Option<&mut DebugCapture>,
        eos_token_id: Option<u32>,
    ) -> Result<(Vec<u32>, GenerationProfile)> {
        let mut profile = GenerationProfile::new();
        let mut generated = Vec::with_capacity(max_new_tokens);

        // Set up KV cache based on device
        #[cfg(feature = "cuda")]
        let mut cache: Option<KVCache> = if self.cuda_ctx.is_some() {
            // CUDA path: create CudaKVCache, no CPU KVCache needed
            let device_id = self.cuda_device_id.unwrap();
            self.cuda_kv_cache = Some(CudaKVCache::new(
                device_id,
                self.config.n_layers,
                self.config.n_kv_heads,
                self.config.head_dim(),
                self.config.max_seq_len,
            )?);
            None
        } else {
            Some(KVCache::new(&self.config))
        };
        #[cfg(not(feature = "cuda"))]
        let mut cache: Option<KVCache> = Some(KVCache::new(&self.config));

        // Prefill: process all prompt tokens in one forward pass
        let logits = match cache.as_mut() {
            Some(c) => self.forward(prompt_tokens, 0, Some(c), &mut debug)?,
            None => self.forward(prompt_tokens, 0, None, &mut debug)?,
        };

        // Prefill complete — first token is ready
        profile.record_first_token();

        // Extract last-position logits from prefill output [prompt_len, vocab_size]
        #[cfg(feature = "cuda")]
        let last_logits = match logits.device() {
            Device::Cuda(_) => cuda_ops::to_cpu(&logits)?,
            _ => logits.clone(),
        };
        #[cfg(not(feature = "cuda"))]
        let last_logits = logits.clone();

        let prompt_len = prompt_tokens.len();
        let vocab_size = self.config.vocab_size;
        let logits_data = last_logits.as_f32()?;
        // Last row of [prompt_len, vocab_size]
        let last_row_offset = (prompt_len - 1) * vocab_size;
        let max_idx = logits_data[last_row_offset..last_row_offset + vocab_size]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let next_token = max_idx as u32;
        generated.push(next_token);
        on_token(next_token);

        // Stop on EOS token
        if let Some(eos) = eos_token_id {
            if next_token == eos {
                profile.finalize(prompt_len, generated.len());
                return Ok((generated, profile));
            }
        }

        // Decode: generate remaining tokens one at a time
        let mut current_token = next_token;
        for i in 0..max_new_tokens - 1 {
            let pos = prompt_len + i;
            let decode_logits = match cache.as_mut() {
                Some(c) => self.forward(&[current_token], pos, Some(c), &mut debug)?,
                None => self.forward(&[current_token], pos, None, &mut debug)?,
            };

            #[cfg(feature = "cuda")]
            let logits_cpu = match decode_logits.device() {
                Device::Cuda(_) => cuda_ops::to_cpu(&decode_logits)?,
                _ => decode_logits.clone(),
            };
            #[cfg(not(feature = "cuda"))]
            let logits_cpu = decode_logits.clone();

            let logits_data = logits_cpu.as_f32()?;
            let max_idx = logits_data
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);

            current_token = max_idx as u32;
            generated.push(current_token);
            on_token(current_token);

            // Stop on EOS token
            if let Some(eos) = eos_token_id {
                if current_token == eos {
                    break;
                }
            }
        }

        profile.finalize(prompt_len, generated.len());
        Ok((generated, profile))
    }

    /// Generate tokens autoregressively (non-streaming, no debug).
    ///
    /// `prompt_tokens`: initial prompt tokens
    /// `max_new_tokens`: maximum number of new tokens to generate
    /// `eos_token_id`: optional EOS token ID for early stopping
    ///
    /// Returns generated token IDs (excluding prompt) and a generation profile.
    pub fn generate(&mut self, prompt_tokens: &[u32], max_new_tokens: usize, eos_token_id: Option<u32>) -> Result<(Vec<u32>, GenerationProfile)> {
        self.generate_streaming(prompt_tokens, max_new_tokens, |_| {}, None, eos_token_id)
    }

    // ── CUDA GPU-only forward pass ───────────────────────────────────────

    /// Fully GPU-resident forward pass for CUDA.
    ///
    /// All ops stay on GPU; single stream sync at the end.
    #[cfg(feature = "cuda")]
    fn forward_cuda(&mut self, token_ids: &[u32], start_pos: usize) -> Result<Tensor> {
        let ctx = self.cuda_ctx.as_ref().unwrap();
        let kv_cache = self.cuda_kv_cache.as_ref().unwrap();

        // Embedding lookup on GPU
        let seq_len = token_ids.len();
        let device_id = self.cuda_device_id.unwrap();

        // Upload token IDs to GPU
        let ids_bytes = seq_len * std::mem::size_of::<u32>();
        let ids_buf = CudaBuffer::alloc(ids_bytes, device_id)
            .map_err(Error::Cuda)?;
        let host_bytes: Vec<u8> = token_ids.iter()
            .flat_map(|id| id.to_ne_bytes())
            .collect();
        ids_buf.copy_from_host(&host_bytes)
            .map_err(Error::Cuda)?;

        let mut x = cuda_kernels::embedding::lookup(ctx, &self.weights.token_embedding, &ids_buf, seq_len)?;

        // Transformer layers
        for (layer_idx, layer) in self.weights.layers.iter().enumerate() {
            x = self.transformer_layer_cuda(&x, layer, layer_idx, start_pos, ctx, kv_cache)?;
        }

        // Final RMS norm + output projection
        x = cuda_kernels::norm::rms(ctx, &x, &self.weights.output_norm_weight, self.config.rms_norm_eps)?;
        x = cuda_kernels::gemm::matmul(ctx, &x, &self.weights.output_weight)?;

        // Single sync point — wait for all queued work
        ctx.synchronize().map_err(Error::Cuda)?;

        // Advance KV cache
        self.cuda_kv_cache.as_mut().unwrap().advance(seq_len);

        Ok(x)
    }

    #[cfg(feature = "cuda")]
    fn transformer_layer_cuda(
        &self,
        x: &Tensor,
        layer: &TransformerLayer,
        layer_idx: usize,
        start_pos: usize,
        ctx: &CudaContext,
        kv_cache: &CudaKVCache,
    ) -> Result<Tensor> {
        // Pre-attention norm
        let normed = cuda_kernels::norm::rms(ctx, x, &layer.attn_norm_weight, self.config.rms_norm_eps)?;

        // Attention
        let attn_out = self.attention_gpu(&normed, layer, layer_idx, start_pos, ctx, kv_cache)?;

        // Residual
        let x = cuda_kernels::elementwise::add(ctx, x, &attn_out)?;

        // Pre-FFN norm
        let normed = cuda_kernels::norm::rms(ctx, &x, &layer.ffn_norm_weight, self.config.rms_norm_eps)?;

        // MLP
        let gate = cuda_kernels::gemm::matmul(ctx, &normed, &layer.w_gate)?;
        let gate = cuda_kernels::activation::silu(ctx, &gate)?;
        let up = cuda_kernels::gemm::matmul(ctx, &normed, &layer.w_up)?;
        let hidden = cuda_kernels::elementwise::mul(ctx, &gate, &up)?;
        let mlp_out = cuda_kernels::gemm::matmul(ctx, &hidden, &layer.w_down)?;

        // Residual
        cuda_kernels::elementwise::add(ctx, &x, &mlp_out)
    }

    /// GPU-only attention using cuBLAS GEMM for GQA.
    #[cfg(feature = "cuda")]
    fn attention_gpu(
        &self,
        x: &Tensor,
        layer: &TransformerLayer,
        layer_idx: usize,
        start_pos: usize,
        ctx: &CudaContext,
        kv_cache: &CudaKVCache,
    ) -> Result<Tensor> {
        let n_heads = self.config.n_heads;
        let n_kv_heads = self.config.n_kv_heads;
        let head_dim = self.config.head_dim();
        let seq_len = x.shape().dims()[0];
        // Project to Q, K, V
        let q = cuda_kernels::gemm::matmul(ctx, x, &layer.wq)?;
        let k = cuda_kernels::gemm::matmul(ctx, x, &layer.wk)?;
        let v = cuda_kernels::gemm::matmul(ctx, x, &layer.wv)?;

        // Reshape to [seq_len, n_heads, head_dim] / [seq_len, n_kv_heads, head_dim]
        let q = q.reshape(vec![seq_len, n_heads, head_dim])?;
        let k = k.reshape(vec![seq_len, n_kv_heads, head_dim])?;
        let v = v.reshape(vec![seq_len, n_kv_heads, head_dim])?;

        // Apply batched RoPE (half-split)
        let q_rope = cuda_kernels::rope::apply_batched(
            ctx, &q, n_heads, head_dim, self.config.rope_theta, start_pos as u32)?;
        let k_rope = cuda_kernels::rope::apply_batched(
            ctx, &k, n_kv_heads, head_dim, self.config.rope_theta, start_pos as u32)?;

        // Append K/V to GPU KV cache
        kv_cache.append(ctx, layer_idx, &k_rope, &v, seq_len)?;

        let kv_len = kv_cache.seq_len() + seq_len;
        let attn_out = cuda_kernels::attention::sdpa(
            ctx,
            &q_rope,
            kv_cache,
            layer_idx,
            n_heads,
            n_kv_heads,
            head_dim,
            kv_len,
            self.config.max_seq_len,
            start_pos as u32,
        )?;
        // Output projection
        let proj = cuda_kernels::gemm::matmul(ctx, &attn_out, &layer.wo)?;

        Ok(proj)
    }

    // ── Original forward pass (CPU or CUDA with debug) ──────────────────

    /// Forward pass for a batch of tokens.
    ///
    /// `token_ids`: token IDs to process (all prompt tokens for prefill, or one for decode)
    /// `start_pos`: starting position in the sequence (for RoPE and KV cache)
    /// `kv_cache`: optional KV cache for past tokens
    /// `debug`: optional debug capture for activations
    ///
    /// Returns logits `[seq_len, vocab_size]`.
    pub fn forward(
        &mut self,
        token_ids: &[u32],
        start_pos: usize,
        mut kv_cache: Option<&mut KVCache>,
        debug: &mut Option<&mut DebugCapture>,
    ) -> Result<Tensor> {
        // Branch to CUDA GPU path when CUDA is active and no debug
        #[cfg(feature = "cuda")]
        {
            if self.cuda_ctx.is_some() && self.cuda_kv_cache.is_some() && debug.is_none() {
                return self.forward_cuda(token_ids, start_pos);
            }
        }

        let seq_len = token_ids.len();
        if seq_len == 0 {
            return Err(Error::Other("forward: empty token_ids".into()));
        }

        if let Some(d) = debug {
            d.set_position(start_pos + seq_len - 1);
        }

        let device = self.weights.token_embedding.device();

        // Embedding lookup: [seq_len, hidden_size]
        let mut x = self.embedding_lookup(token_ids, device)?;

        if let Some(d) = debug {
            let x_cpu = self.tensor_to_cpu(&x)?;
            d.capture("embed.token", &x_cpu.as_f32()?, x.shape().dims());
        }

        // Transformer layers
        for (layer_idx, layer) in self.weights.layers.iter().enumerate() {
            x = self.transformer_layer(&x, layer, layer_idx, start_pos, kv_cache.as_deref_mut(), debug)?;
        }

        // Advance KV cache by seq_len (after all layers have appended)
        if let Some(ref mut cache) = kv_cache {
            cache.advance(seq_len);
        }
        #[cfg(feature = "cuda")]
        if let Some(ref mut cuda_cache) = self.cuda_kv_cache {
            cuda_cache.advance(seq_len);
        }

        // Final norm
        if let Some(d) = debug {
            let x_cpu = self.tensor_to_cpu(&x)?;
            d.capture("final.norm.input", &x_cpu.as_f32()?, x.shape().dims());
        }
        x = self.rms_norm(&x, &self.weights.output_norm_weight)?;

        if let Some(d) = debug {
            let x_cpu = self.tensor_to_cpu(&x)?;
            d.capture("final.norm.output", &x_cpu.as_f32()?, x.shape().dims());
        }

        // Output projection
        let logits = self.matmul(&x, &self.weights.output_weight)?;

        if let Some(d) = debug {
            let logits_cpu = self.tensor_to_cpu(&logits)?;
            d.capture("final.logits", &logits_cpu.as_f32()?, logits.shape().dims());
        }

        Ok(logits)
    }

    fn embedding_lookup(&self, token_ids: &[u32], device: Device) -> Result<Tensor> {
        let embed_dim = self.config.hidden_size;
        let seq_len = token_ids.len();

        // Get the embedding table on CPU
        #[cfg(feature = "cuda")]
        let table_cpu = match device {
            Device::Cuda(_) => cuda_ops::to_cpu(&self.weights.token_embedding)?,
            _ => self.weights.token_embedding.clone(),
        };
        #[cfg(not(feature = "cuda"))]
        let table_cpu = self.weights.token_embedding.clone();

        let table = table_cpu.as_f32()?;

        // Gather embedding rows for all tokens
        let mut embed_data = vec![0.0f32; seq_len * embed_dim];
        for (i, &tid) in token_ids.iter().enumerate() {
            let offset = tid as usize * embed_dim;
            embed_data[i * embed_dim..(i + 1) * embed_dim]
                .copy_from_slice(&table[offset..offset + embed_dim]);
        }

        match device {
            Device::Cpu => {
                Tensor::from_f32(vec![seq_len, embed_dim], &embed_data)
            }
            #[cfg(feature = "cuda")]
            Device::Cuda(device_id) => {
                let cpu_tensor = Tensor::from_f32(vec![seq_len, embed_dim], &embed_data)?;
                cuda_ops::to_cuda(&cpu_tensor, device_id)
            }
            #[cfg(not(feature = "cuda"))]
            Device::Cuda(_) => Err(Error::Other("CUDA not compiled in".into())),
        }
    }

    fn transformer_layer(
        &self,
        x: &Tensor,
        layer: &TransformerLayer,
        layer_idx: usize,
        start_pos: usize,
        kv_cache: Option<&mut KVCache>,
        debug: &mut Option<&mut DebugCapture>,
    ) -> Result<Tensor> {
        let prefix = format!("layer.{}", layer_idx);

        // Pre-attention norm
        if let Some(d) = debug {
            let x_cpu = self.tensor_to_cpu(x)?;
            d.capture(&format!("{}.norm_attn.input", prefix), &x_cpu.as_f32()?, x.shape().dims());
        }
        let normed = self.rms_norm(x, &layer.attn_norm_weight)?;

        if let Some(d) = debug {
            let normed_cpu = self.tensor_to_cpu(&normed)?;
            d.capture(&format!("{}.norm_attn.output", prefix), &normed_cpu.as_f32()?, normed.shape().dims());
        }

        // Attention
        let attn_out = self.attention(&normed, layer, layer_idx, start_pos, kv_cache, debug)?;

        if let Some(d) = debug {
            let attn_cpu = self.tensor_to_cpu(&attn_out)?;
            d.capture(&format!("{}.attn.proj_output", prefix), &attn_cpu.as_f32()?, attn_out.shape().dims());
        }

        // Residual
        let x = self.add(x, &attn_out)?;

        if let Some(d) = debug {
            let x_cpu = self.tensor_to_cpu(&x)?;
            d.capture(&format!("{}.residual_attn", prefix), &x_cpu.as_f32()?, x.shape().dims());
        }

        // Pre-FFN norm
        if let Some(d) = debug {
            let x_cpu = self.tensor_to_cpu(&x)?;
            d.capture(&format!("{}.norm_ffn.input", prefix), &x_cpu.as_f32()?, x.shape().dims());
        }
        let normed = self.rms_norm(&x, &layer.ffn_norm_weight)?;

        if let Some(d) = debug {
            let normed_cpu = self.tensor_to_cpu(&normed)?;
            d.capture(&format!("{}.norm_ffn.output", prefix), &normed_cpu.as_f32()?, normed.shape().dims());
        }

        // MLP
        let mlp_out = self.mlp(&normed, layer, debug, &prefix)?;

        // Residual
        let x = self.add(&x, &mlp_out)?;

        if let Some(d) = debug {
            let x_cpu = self.tensor_to_cpu(&x)?;
            d.capture(&format!("{}.residual_ffn", prefix), &x_cpu.as_f32()?, x.shape().dims());
        }

        Ok(x)
    }

    fn attention(
        &self,
        x: &Tensor,
        layer: &TransformerLayer,
        layer_idx: usize,
        start_pos: usize,
        kv_cache: Option<&mut KVCache>,
        debug: &mut Option<&mut DebugCapture>,
    ) -> Result<Tensor> {
        let n_heads = self.config.n_heads;
        let n_kv_heads = self.config.n_kv_heads;
        let head_dim = self.config.head_dim();
        let seq_len = x.shape().dims()[0];
        let prefix = format!("layer.{}", layer_idx);

        // Project to Q, K, V: [seq_len, n_heads*head_dim] or [seq_len, n_kv_heads*head_dim]
        let q = self.matmul(x, &layer.wq)?;
        let k = self.matmul(x, &layer.wk)?;
        let v = self.matmul(x, &layer.wv)?;

        if let Some(d) = debug {
            let q_cpu = self.tensor_to_cpu(&q)?;
            let k_cpu = self.tensor_to_cpu(&k)?;
            let v_cpu = self.tensor_to_cpu(&v)?;
            d.capture(&format!("{}.attn.q", prefix), &q_cpu.as_f32()?, q.shape().dims());
            d.capture(&format!("{}.attn.k", prefix), &k_cpu.as_f32()?, k.shape().dims());
            d.capture(&format!("{}.attn.v", prefix), &v_cpu.as_f32()?, v.shape().dims());
        }

        // Reshape to [seq_len, n_heads, head_dim] / [seq_len, n_kv_heads, head_dim]
        let q = q.reshape(vec![seq_len, n_heads, head_dim])?;
        let k = k.reshape(vec![seq_len, n_kv_heads, head_dim])?;
        let v = v.reshape(vec![seq_len, n_kv_heads, head_dim])?;

// ── CPU fallback path ────────────────────────────────────────

        // Convert to f32 vectors (CPU-based attention)
        #[cfg(feature = "cuda")]
        let (q_data, k_data, v_data) = match x.device() {
            Device::Cuda(_) => {
                let q_cpu = cuda_ops::to_cpu(&q)?;
                let k_cpu = cuda_ops::to_cpu(&k)?;
                let v_cpu = cuda_ops::to_cpu(&v)?;
                (q_cpu.as_f32()?.to_vec(), k_cpu.as_f32()?.to_vec(), v_cpu.as_f32()?.to_vec())
            }
            _ => (q.as_f32()?.to_vec(), k.as_f32()?.to_vec(), v.as_f32()?.to_vec()),
        };
        #[cfg(not(feature = "cuda"))]
        let (q_data, k_data, v_data) = (q.as_f32()?.to_vec(), k.as_f32()?.to_vec(), v.as_f32()?.to_vec());

        // Apply RoPE with position offsets per token
        let q_rope = self.rope_batched(&q_data, seq_len, n_heads, head_dim, start_pos)?;
        let k_rope = self.rope_batched(&k_data, seq_len, n_kv_heads, head_dim, start_pos)?;

        if let Some(d) = debug {
            d.capture(&format!("{}.attn.q_rope", prefix), &q_rope, &[seq_len, n_heads, head_dim]);
            d.capture(&format!("{}.attn.k_rope", prefix), &k_rope, &[seq_len, n_kv_heads, head_dim]);
        }

        // Update KV cache and compute attention
        let attn_out = if let Some(cache) = kv_cache {
            // Batched KV cache append for all positions in this forward pass
            cache.append(layer_idx, &k_rope, &v_data, seq_len, n_kv_heads, head_dim);

            let (k_cached, v_cached) = cache.get_kv(layer_idx);

            // Compute attention for each position and each head
            let mut output = vec![0.0f32; seq_len * n_heads * head_dim];
            let scale = 1.0 / (head_dim as f32).sqrt();

            for s in 0..seq_len {
                let pos_offset = start_pos + s;
                for h in 0..n_heads {
                    let kv_h = h * n_kv_heads / n_heads;

                    let attn_len = pos_offset + 1;
                    let mut scores = vec![0.0f32; attn_len];
                    for t in 0..attn_len {
                        for d in 0..head_dim {
                            scores[t] += q_rope[s * n_heads * head_dim + h * head_dim + d]
                                * k_cached[kv_h][t][d];
                        }
                        scores[t] *= scale;
                    }

                    // Softmax
                    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let exp_sum: f32 = scores.iter().map(|&s| (s - max_score).exp()).sum();
                    for t in 0..attn_len {
                        scores[t] = (scores[t] - max_score).exp() / exp_sum;
                    }

                    // Weighted sum of V
                    for t in 0..attn_len {
                        for d in 0..head_dim {
                            output[s * n_heads * head_dim + h * head_dim + d]
                                += scores[t] * v_cached[kv_h][t][d];
                        }
                    }
                }
            }

            output
        } else {
            // No cache: self-attention with causal mask
            let mut output = vec![0.0f32; seq_len * n_heads * head_dim];
            let scale = 1.0 / (head_dim as f32).sqrt();

            for s in 0..seq_len {
                for h in 0..n_heads {
                    let kv_h = h * n_kv_heads / n_heads;

                    let mut scores = vec![0.0f32; s + 1];
                    for t in 0..=s {
                        for d in 0..head_dim {
                            scores[t] += q_rope[s * n_heads * head_dim + h * head_dim + d]
                                * k_rope[t * n_kv_heads * head_dim + kv_h * head_dim + d];
                        }
                        scores[t] *= scale;
                    }

                    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let exp_sum: f32 = scores.iter().map(|&sc| (sc - max_score).exp()).sum();
                    for t in 0..=s {
                        scores[t] = (scores[t] - max_score).exp() / exp_sum;
                    }

                    for t in 0..=s {
                        for d in 0..head_dim {
                            output[s * n_heads * head_dim + h * head_dim + d]
                                += scores[t] * v_data[t * n_kv_heads * head_dim + kv_h * head_dim + d];
                        }
                    }
                }
            }

            output
        };

        // Reshape output for projection
        if let Some(d) = debug {
            d.capture(&format!("{}.attn.output", prefix), &attn_out, &[seq_len, n_heads * head_dim]);
        }

        // Create output tensor on the correct device: [seq_len, hidden_size]
        let attn_out_tensor = match x.device() {
            Device::Cpu => Tensor::from_f32(vec![seq_len, n_heads * head_dim], &attn_out)?,
            #[cfg(feature = "cuda")]
            Device::Cuda(device_id) => {
                let cpu = Tensor::from_f32(vec![seq_len, n_heads * head_dim], &attn_out)?;
                cuda_ops::to_cuda(&cpu, device_id)?
            }
            #[cfg(not(feature = "cuda"))]
            Device::Cuda(_) => Tensor::from_f32(vec![seq_len, n_heads * head_dim], &attn_out)?,
        };

        // Output projection
        let proj = self.matmul(&attn_out_tensor, &layer.wo)?;

        Ok(proj)
    }

    fn mlp(&self, x: &Tensor, layer: &TransformerLayer, debug: &mut Option<&mut DebugCapture>, prefix: &str) -> Result<Tensor> {
        // gate = silu(x @ w_gate)
        let gate = self.matmul(x, &layer.w_gate)?;

        if let Some(d) = debug {
            let gate_cpu = self.tensor_to_cpu(&gate)?;
            d.capture(&format!("{}.ffn.gate", prefix), &gate_cpu.as_f32()?, gate.shape().dims());
        }
        let gate = self.silu(&gate)?;

        // up = x @ w_up
        let up = self.matmul(x, &layer.w_up)?;

        if let Some(d) = debug {
            let up_cpu = self.tensor_to_cpu(&up)?;
            d.capture(&format!("{}.ffn.up", prefix), &up_cpu.as_f32()?, up.shape().dims());
        }

        // hidden = gate * up
        let hidden = self.mul(&gate, &up)?;

        if let Some(d) = debug {
            let hidden_cpu = self.tensor_to_cpu(&hidden)?;
            d.capture(&format!("{}.ffn.gated", prefix), &hidden_cpu.as_f32()?, hidden.shape().dims());
        }

        // out = hidden @ w_down
        let out = self.matmul(&hidden, &layer.w_down)?;

        if let Some(d) = debug.as_mut() {
            let out_cpu = self.tensor_to_cpu(&out)?;
            d.capture(&format!("{}.ffn.output", prefix), &out_cpu.as_f32()?, out.shape().dims());
        }

        Ok(out)
    }

    // ── Low-level ops (dispatch to CPU or GPU) ──────────────────────

    /// Helper to transfer a tensor to CPU if it's on GPU.
    /// For CPU tensors, returns a clone.
    #[cfg(feature = "cuda")]
    fn tensor_to_cpu(&self, x: &Tensor) -> Result<Tensor> {
        match x.device() {
            Device::Cuda(_) => cuda_ops::to_cpu(x),
            _ => Ok(x.clone()),
        }
    }

    #[cfg(not(feature = "cuda"))]
    fn tensor_to_cpu(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.clone())
    }

    fn rms_norm(&self, x: &Tensor, weight: &Tensor) -> Result<Tensor> {
        match x.device() {
            Device::Cpu => {
                let eps = self.config.rms_norm_eps;
                let data = x.as_f32()?;
                let w = weight.as_f32()?;
                let dims = x.shape().dims();

                // Row-wise RMSNorm: norm each row independently
                // x shape: [seq_len, hidden_size] or [1, hidden_size]
                let seq_len = dims[0];
                let hidden = dims[1];

                let mut out = vec![0.0f32; data.len()];
                for s in 0..seq_len {
                    let row_start = s * hidden;
                    let row = &data[row_start..row_start + hidden];

                    let mean_sq: f32 = row.iter().map(|v| v * v).sum::<f32>() / hidden as f32;
                    let rms = (mean_sq + eps).sqrt();

                    for (i, &val) in row.iter().enumerate() {
                        out[row_start + i] = (val / rms) * w[i];
                    }
                }

                Tensor::from_f32(dims.to_vec(), &out)
            }
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => {
                let ctx = self.cuda_ctx.as_ref()
                    .ok_or_else(|| Error::Other("CUDA context not initialized".into()))?;
                cuda_kernels::norm::rms(ctx, x, weight, self.config.rms_norm_eps)
            }
            #[cfg(not(feature = "cuda"))]
            Device::Cuda(_) => {
                Err(Error::Other("CUDA not compiled in".into()))
            }
        }
    }

    /// Apply RoPE to a batched tensor with position offsets.
    ///
    /// `data`: flat f32 data with layout [seq_len, n_heads, head_dim]
    /// `seq_len`: number of tokens in the batch
    /// `n_heads`: number of heads (n_heads for Q, n_kv_heads for K)
    /// `head_dim`: dimension per head
    /// `start_pos`: starting position offset (first token's position)
    fn rope_batched(&self, data: &[f32], seq_len: usize, n_heads: usize, head_dim: usize, start_pos: usize) -> Result<Vec<f32>> {
        let theta = self.config.rope_theta;
        let half_dim = head_dim / 2;
        let mut out = vec![0.0f32; data.len()];

        // Compute frequency table once (same for all positions)
        let freqs: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32))
            .collect();

        for s in 0..seq_len {
            let pos = start_pos + s;
            for h in 0..n_heads {
                let base = s * n_heads * head_dim + h * head_dim;
                // Half-style RoPE: split head into [0..half_dim] and [half_dim..head_dim]
                // x1 = head[0..half_dim], x2 = head[half_dim..head_dim]
                // rotate_half(x) = (-x2, x1)
                // out = x * cos + rotate_half(x) * sin
                // => out[0..half_dim] = x1*cos - x2*sin
                // => out[half_dim..head_dim] = x1*sin + x2*cos
                for i in 0..half_dim {
                    let angle = pos as f32 * freqs[i];
                    let cos_v = angle.cos();
                    let sin_v = angle.sin();

                    let x1 = data[base + i];            // first half
                    let x2 = data[base + half_dim + i]; // second half

                    out[base + i] = x1 * cos_v - x2 * sin_v;
                    out[base + half_dim + i] = x1 * sin_v + x2 * cos_v;
                }
            }
        }

        Ok(out)
    }

    fn silu(&self, x: &Tensor) -> Result<Tensor> {
        match x.device() {
            Device::Cpu => {
                let data = x.as_f32()?;
                let out: Vec<f32> = data.iter()
                    .map(|&v| v / (1.0 + (-v).exp()))
                    .collect();
                Tensor::from_f32(x.shape().dims().to_vec(), &out)
            }
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => {
                let ctx = self.cuda_ctx.as_ref()
                    .ok_or_else(|| Error::Other("CUDA context not initialized".into()))?;
                cuda_kernels::activation::silu(ctx, x)
            }
            #[cfg(not(feature = "cuda"))]
            Device::Cuda(_) => {
                Err(Error::Other("CUDA not compiled in".into()))
            }
        }
    }

    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        match a.device() {
            Device::Cpu => {
                let a_data = a.as_f32()?;
                let b_data = b.as_f32()?;
                let out: Vec<f32> = a_data.iter().zip(b_data.iter())
                    .map(|(a, b)| a + b)
                    .collect();
                Tensor::from_f32(a.shape().dims().to_vec(), &out)
            }
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => {
                let ctx = self.cuda_ctx.as_ref()
                    .ok_or_else(|| Error::Other("CUDA context not initialized".into()))?;
                cuda_kernels::elementwise::add(ctx, a, b)
            }
            #[cfg(not(feature = "cuda"))]
            Device::Cuda(_) => {
                Err(Error::Other("CUDA not compiled in".into()))
            }
        }
    }

    fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        match a.device() {
            Device::Cpu => {
                let a_data = a.as_f32()?;
                let b_data = b.as_f32()?;
                let out: Vec<f32> = a_data.iter().zip(b_data.iter())
                    .map(|(a, b)| a * b)
                    .collect();
                Tensor::from_f32(a.shape().dims().to_vec(), &out)
            }
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => {
                let ctx = self.cuda_ctx.as_ref()
                    .ok_or_else(|| Error::Other("CUDA context not initialized".into()))?;
                cuda_kernels::elementwise::mul(ctx, a, b)
            }
            #[cfg(not(feature = "cuda"))]
            Device::Cuda(_) => {
                Err(Error::Other("CUDA not compiled in".into()))
            }
        }
    }

    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        match a.device() {
            Device::Cpu => a.matmul_cpu(b),
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => {
                let ctx = self.cuda_ctx.as_ref()
                    .ok_or_else(|| Error::Other("CUDA context not initialized".into()))?;
                cuda_kernels::gemm::matmul(ctx, a, b)
            }
            #[cfg(not(feature = "cuda"))]
            Device::Cuda(_) => {
                Err(Error::Other("CUDA not compiled in".into()))
            }
        }
    }
}

/// KV cache for autoregressive generation.
///
/// Stores K and V vectors for all layers across all past positions.
/// Shape: [n_layers, 2, n_kv_heads, max_seq_len, head_dim]
///   - 2 is for K and V
pub struct KVCache {
    /// K cache: [n_layers, n_kv_heads, max_seq_len, head_dim]
    k_cache: Vec<Vec<Vec<Vec<f32>>>>,
    /// V cache: [n_layers, n_kv_heads, max_seq_len, head_dim]
    v_cache: Vec<Vec<Vec<Vec<f32>>>>,
    /// Current sequence length (number of cached tokens)
    seq_len: usize,
}

impl KVCache {
    /// Create a new KV cache.
    pub fn new(config: &ModelConfig) -> Self {
        let head_dim = config.head_dim();
        let n_kv_heads = config.n_kv_heads;
        let max_seq_len = config.max_seq_len;
        let n_layers = config.n_layers;

        // Initialize with zeros
        let k_cache = (0..n_layers)
            .map(|_| {
                (0..n_kv_heads)
                    .map(|_| {
                        (0..max_seq_len)
                            .map(|_| vec![0.0f32; head_dim])
                            .collect()
                    })
                    .collect()
            })
            .collect();

        let v_cache = (0..n_layers)
            .map(|_| {
                (0..n_kv_heads)
                    .map(|_| {
                        (0..max_seq_len)
                            .map(|_| vec![0.0f32; head_dim])
                            .collect()
                    })
                    .collect()
            })
            .collect();

        Self {
            k_cache,
            v_cache,
            seq_len: 0,
        }
    }

    /// Append K, V for multiple positions.
    ///
    /// `layer_idx`: which layer
    /// `k_rope`: [seq_len, n_kv_heads, head_dim] flat data with RoPE applied
    /// `v`: [seq_len, n_kv_heads, head_dim] flat data
    /// `seq_len`: number of positions to append
    /// `n_kv_heads`: number of KV heads
    /// `head_dim`: dimension per head
    pub fn append(&mut self, layer_idx: usize, k_rope: &[f32], v: &[f32], seq_len: usize, n_kv_heads: usize, head_dim: usize) {
        for s in 0..seq_len {
            let pos = self.seq_len + s;
            for h in 0..n_kv_heads {
                for d in 0..head_dim {
                    self.k_cache[layer_idx][h][pos][d] = k_rope[s * n_kv_heads * head_dim + h * head_dim + d];
                    self.v_cache[layer_idx][h][pos][d] = v[s * n_kv_heads * head_dim + h * head_dim + d];
                }
            }
        }
    }

    /// Get K and V for a layer up to current position.
    /// Returns (K, V) where each is [n_kv_heads, seq_len, head_dim]
    pub fn get_kv(&self, layer_idx: usize) -> (&[Vec<Vec<f32>>], &[Vec<Vec<f32>>]) {
        (
            &self.k_cache[layer_idx],
            &self.v_cache[layer_idx],
        )
    }

    /// Advance sequence length by n positions.
    pub fn advance(&mut self, n: usize) {
        self.seq_len += n;
    }

    /// Current sequence length.
    pub fn len(&self) -> usize {
        self.seq_len
    }

    /// Clear the cache for a new generation.
    pub fn clear(&mut self) {
        self.seq_len = 0;
        // Zero out caches
        for layer in &mut self.k_cache {
            for head in layer {
                for pos in head {
                    for v in pos {
                        *v = 0.0;
                    }
                }
            }
        }
        for layer in &mut self.v_cache {
            for head in layer {
                for pos in head {
                    for v in pos {
                        *v = 0.0;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apxinf_core::Tensor;

    fn tiny_config() -> ModelConfig {
        ModelConfig {
            hidden_size: 64,
            intermediate_size: 128,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 4, // Match n_heads to avoid GQA complexity for now
            vocab_size: 100,
            max_seq_len: 128,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
        }
    }

    fn make_weight(shape: Vec<usize>) -> Tensor {
        let numel: usize = shape.iter().product();
        let data: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.01).collect();
        Tensor::from_f32(shape, &data).unwrap()
    }

    #[test]
    fn test_model_construction() {
        let config = tiny_config();
        let mut tensors = HashMap::new();

        // Token embedding
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            make_weight(vec![config.vocab_size, config.hidden_size]),
        );

        // Layers
        for i in 0..config.n_layers {
            let prefix = format!("model.layers.{i}");
            tensors.insert(
                format!("{prefix}.input_layernorm.weight"),
                make_weight(vec![config.hidden_size]),
            );
            // Attention weights: PyTorch format [out_features, in_features]
            tensors.insert(
                format!("{prefix}.self_attn.q_proj.weight"),
                make_weight(vec![config.hidden_size, config.hidden_size]),
            );
            tensors.insert(
                format!("{prefix}.self_attn.k_proj.weight"),
                make_weight(vec![config.n_kv_heads * config.head_dim(), config.hidden_size]),
            );
            tensors.insert(
                format!("{prefix}.self_attn.v_proj.weight"),
                make_weight(vec![config.n_kv_heads * config.head_dim(), config.hidden_size]),
            );
            tensors.insert(
                format!("{prefix}.self_attn.o_proj.weight"),
                make_weight(vec![config.hidden_size, config.hidden_size]),
            );
            tensors.insert(
                format!("{prefix}.post_attention_layernorm.weight"),
                make_weight(vec![config.hidden_size]),
            );
            // MLP weights: PyTorch format [out_features, in_features]
            tensors.insert(
                format!("{prefix}.mlp.gate_proj.weight"),
                make_weight(vec![config.intermediate_size, config.hidden_size]),
            );
            tensors.insert(
                format!("{prefix}.mlp.up_proj.weight"),
                make_weight(vec![config.intermediate_size, config.hidden_size]),
            );
            tensors.insert(
                format!("{prefix}.mlp.down_proj.weight"),
                make_weight(vec![config.hidden_size, config.intermediate_size]),
            );
        }

        tensors.insert(
            "model.norm.weight".to_string(),
            make_weight(vec![config.hidden_size]),
        );
        // lm_head: PyTorch format [vocab_size, hidden_size]
        tensors.insert(
            "lm_head.weight".to_string(),
            make_weight(vec![config.vocab_size, config.hidden_size]),
        );

        let mut model = LlamaModel::from_weights(config.clone(), tensors).unwrap();

        // Forward pass for token 5 at position 0
        let mut debug: Option<&mut DebugCapture> = None;
        let logits = model.forward(&[5], 0, None, &mut debug).unwrap();
        assert_eq!(logits.shape().dims(), &[1, config.vocab_size]);
        assert_eq!(logits.device(), Device::Cpu);
    }

    #[test]
    fn test_kv_cache() {
        let config = tiny_config();
        let cache = KVCache::new(&config);

        assert_eq!(cache.len(), 0);
        assert_eq!(cache.k_cache.len(), config.n_layers);
        assert_eq!(cache.v_cache.len(), config.n_layers);
    }

    #[test]
    fn test_generation() {
        let config = tiny_config();
        let mut tensors = HashMap::new();

        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            make_weight(vec![config.vocab_size, config.hidden_size]),
        );

        for i in 0..config.n_layers {
            let prefix = format!("model.layers.{i}");
            tensors.insert(
                format!("{prefix}.input_layernorm.weight"),
                make_weight(vec![config.hidden_size]),
            );
            // PyTorch format [out_features, in_features]
            tensors.insert(
                format!("{prefix}.self_attn.q_proj.weight"),
                make_weight(vec![config.hidden_size, config.hidden_size]),
            );
            tensors.insert(
                format!("{prefix}.self_attn.k_proj.weight"),
                make_weight(vec![config.n_kv_heads * config.head_dim(), config.hidden_size]),
            );
            tensors.insert(
                format!("{prefix}.self_attn.v_proj.weight"),
                make_weight(vec![config.n_kv_heads * config.head_dim(), config.hidden_size]),
            );
            tensors.insert(
                format!("{prefix}.self_attn.o_proj.weight"),
                make_weight(vec![config.hidden_size, config.hidden_size]),
            );
            tensors.insert(
                format!("{prefix}.post_attention_layernorm.weight"),
                make_weight(vec![config.hidden_size]),
            );
            // PyTorch format [out_features, in_features]
            tensors.insert(
                format!("{prefix}.mlp.gate_proj.weight"),
                make_weight(vec![config.intermediate_size, config.hidden_size]),
            );
            tensors.insert(
                format!("{prefix}.mlp.up_proj.weight"),
                make_weight(vec![config.intermediate_size, config.hidden_size]),
            );
            tensors.insert(
                format!("{prefix}.mlp.down_proj.weight"),
                make_weight(vec![config.hidden_size, config.intermediate_size]),
            );
        }

        tensors.insert(
            "model.norm.weight".to_string(),
            make_weight(vec![config.hidden_size]),
        );
        tensors.insert(
            "lm_head.weight".to_string(),
            make_weight(vec![config.vocab_size, config.hidden_size]),
        );

        let mut model = LlamaModel::from_weights(config.clone(), tensors).unwrap();

        // Generate 5 tokens from prompt [1, 2, 3]
        let prompt = vec![1u32, 2, 3];
        let generated = model.generate(&prompt, 5, None).unwrap();

        assert_eq!(generated.0.len(), 5);
        // Generated tokens should be within vocab range
        for &tok in &generated.0 {
            assert!((tok as usize) < config.vocab_size);
        }
    }
}
