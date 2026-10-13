//! General-purpose Llama model using the Backend trait.
//!
//! Device-agnostic: works on any backend that implements `apxinf_core::Backend`.
//! All compute is dispatched via `&dyn Backend`; no `match device` or `#[cfg]`.

use std::collections::HashMap;
use std::sync::Arc;

use apxinf_core::{Backend, Device, Error, KvCache, Result, Tensor};
#[cfg(feature = "cuda")]
use apxinf_core::{DType, RopeKind};
use apxinf_loader::ModelConfig;

use super::weights::{LlamaWeights, TransformerLayer};
use crate::llm_trait::LlmTrait;
#[cfg(feature = "cuda")]
use crate::accelerator::cuda::downcast as cuda_backend;
use crate::accelerator::create_backend;
#[cfg(feature = "cuda")]
use super::decode_graph::{DecodeGraphConfig, DecodeGraphWeights, DecodeLayerWeights};

/// Device-agnostic Llama model.
pub struct GeneralLlama {
    config: ModelConfig,
    weights: LlamaWeights,
    backend: Arc<dyn Backend>,
    kv: Box<dyn KvCache>,
    /// Allocation-free decode fast path (CUDA only). `None` on CPU —
    /// `forward` falls back to the `dyn Backend` op-by-op path.
    #[cfg(feature = "cuda")]
    decode_graph: Option<crate::llama::decode_graph::DecodeGraph>,
}

impl GeneralLlama {
    /// Construct from weights and a backend.
    pub fn new(config: ModelConfig, weights: LlamaWeights, backend: Arc<dyn Backend>) -> Result<Self> {
        let kv = backend.create_kv_cache(
            config.n_layers,
            config.n_kv_heads,
            config.head_dim(),
            config.max_seq_len,
        );

        // Transfer weights to the backend's device.
        let mut weights = transfer_weights(&weights, &*backend)?;

        // Build fused weight matrices (qkv_packed, gate_up_packed) for
        // the fused-GEMM decode path. No-ops if the backend doesn't
        // support concat_2d (CPU backend) — the decode falls back to
        // three separate GEMMs.
        pack_fused_weights(&mut weights, &*backend);

        // Create the decode workspace + graph capture state if the backend
        // is CUDA (the fast path intentionally uses the concrete CudaBackend
        // type).
        #[cfg(feature = "cuda")]
        let decode_graph = {
            let cfg = Self::decode_cfg_static(&config, weights.token_embedding.dtype());
            cuda_backend(&*backend)
                .map(|cb| crate::llama::decode_graph::DecodeGraph::new(cb, cfg))
                .transpose()
        };

        Ok(Self {
            config, weights, backend, kv,
            #[cfg(feature = "cuda")]
            decode_graph: decode_graph?,
        })
    }

    /// Build the primitive config for the decode-workspace fast path.
    #[cfg(feature = "cuda")]
    fn decode_cfg_static(config: &ModelConfig, dtype: DType) -> DecodeGraphConfig {
        DecodeGraphConfig {
            n_layers: config.n_layers,
            n_heads: config.n_heads,
            n_kv_heads: config.n_kv_heads,
            head_dim: config.head_dim(),
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            vocab_size: config.vocab_size,
            max_seq_len: config.max_seq_len,
            rope_theta: config.rope_theta,
            rms_norm_eps: config.rms_norm_eps,
            dtype,
            rope_kind: RopeKind::OneD { theta: config.rope_theta },
            qk_norm: false,
            tie_embeddings: false,
        }
    }

    fn forward_layer(&mut self, x: &Tensor, layer_idx: usize, start_pos: u32) -> Result<Tensor> {
        let seq_len = x.shape().dims()[0];
        let n_heads = self.config.n_heads;
        let n_kv_heads = self.config.n_kv_heads;
        let head_dim = self.config.head_dim();
        let b = &*self.backend;
        let layer = &self.weights.layers[layer_idx];
        // Pre-attention norm
        let normed = b.rms_norm(x, &layer.attn_norm_weight, self.config.rms_norm_eps)?;

        // Q/K/V projections
        let q = b.matmul(&normed, &layer.wq)?;
        let k = b.matmul(&normed, &layer.wk)?;
        let v = b.matmul(&normed, &layer.wv)?;

        // Reshape to [seq_len, n_heads, head_dim]
        let q = q.reshape(vec![seq_len, n_heads, head_dim])?;
        let k = k.reshape(vec![seq_len, n_kv_heads, head_dim])?;
        let v = v.reshape(vec![seq_len, n_kv_heads, head_dim])?;

        // RoPE
        let q = b.rope(&q, n_heads, head_dim, self.config.rope_theta, start_pos)?;
        let k = b.rope(&k, n_kv_heads, head_dim, self.config.rope_theta, start_pos)?;

        // Append to KV cache via backend (backend has the stream/context).
        b.kv_append(&mut *self.kv, layer_idx, &k, &v, seq_len)?;

        // Attention
        let kv_len = self.kv.seq_len() + seq_len;
        let attn_out = if seq_len == 1 {
            b.sdpa_decode(&q, &mut *self.kv, layer_idx,
                          n_heads, n_kv_heads, head_dim, kv_len, self.config.max_seq_len)?
        } else {
            b.sdpa_prefill(&q, &mut *self.kv, layer_idx,
                           n_heads, n_kv_heads, head_dim, kv_len, self.config.max_seq_len)?
        };

        // Reshape and project
        let attn_out = attn_out.reshape(vec![seq_len, n_heads * head_dim])?;
        let attn_out = b.matmul(&attn_out, &layer.wo)?;

        // Residual
        let x = b.add(x, &attn_out)?;

        // Pre-FFN norm + MLP
        let normed = b.rms_norm(&x, &layer.ffn_norm_weight, self.config.rms_norm_eps)?;
        let gate = b.matmul(&normed, &layer.w_gate)?;
        let gate = b.silu(&gate)?;
        let up = b.matmul(&normed, &layer.w_up)?;
        let hidden = b.mul(&gate, &up)?;
        let mlp_out = b.matmul(&hidden, &layer.w_down)?;

        // Residual
        b.add(&x, &mlp_out)
    }
}

impl LlmTrait for GeneralLlama {
    fn load(config: ModelConfig, weights: HashMap<String, Tensor>, device: Device) -> Result<Self> {
        let llama_weights = LlamaWeights::from_map(&config, weights)?;
        let backend = create_backend(device)?;
        Self::new(config, llama_weights, backend)
    }

    fn forward(&mut self, token_ids: &[u32], start_pos: u32) -> Result<Tensor> {
        let seq_len = token_ids.len();
        if seq_len == 0 {
            return Err(Error::Other("forward: empty token_ids".into()));
        }

        // Decode fast path: seq_len==1 with an allocation-free decode
        // workspace + CUDA Graph capture (CUDA only). Falls back to the
        // op-by-op `dyn Backend` path on CPU.
        #[cfg(feature = "cuda")]
        if seq_len == 1 {
            if let Some(dg) = self.decode_graph.as_mut() {
                let weights = DecodeGraphWeights {
                    token_embedding: &self.weights.token_embedding,
                    layers: self.weights.layers.iter()
                        .map(|l| DecodeLayerWeights {
                            attn_norm_weight: &l.attn_norm_weight,
                            wq: &l.wq, wk: &l.wk, wv: &l.wv, wo: &l.wo,
                            ffn_norm_weight: &l.ffn_norm_weight,
                            w_gate: &l.w_gate, w_up: &l.w_up, w_down: &l.w_down,
                            q_norm_weight: None,
                            k_norm_weight: None,
                            qkv_packed: l.qkv_packed.as_ref(),
                            gate_up_packed: l.gate_up_packed.as_ref(),
                        })
                        .collect(),
                    output_norm_weight: &self.weights.output_norm_weight,
                    output_weight: &self.weights.output_weight,
                };
                let cb = cuda_backend(&*self.backend)
                    .expect("decode_graph requires CudaBackend");
                let logits = dg.decode(cb, &weights, &mut *self.kv, token_ids[0], start_pos)?;
                self.kv.advance(1);
                return Ok(logits);
            }
        }

        // Embedding lookup
        let mut x = self.backend.embedding(&self.weights.token_embedding, token_ids)?;

        // Transformer layers
        for layer_idx in 0..self.config.n_layers {
            x = self.forward_layer(&x, layer_idx, start_pos)?;
        }

        // Advance KV cache after all layers consumed it
        self.kv.advance(seq_len);

        // Final norm + output
        let x = self.backend.rms_norm(&x, &self.weights.output_norm_weight, self.config.rms_norm_eps)?;
        let logits = self.backend.matmul(&x, &self.weights.output_weight)?;

        Ok(logits)
    }

    fn backend(&self) -> &dyn Backend {
        &*self.backend
    }

    #[cfg(feature = "cuda")]
    fn prewarm_decode(&mut self, prompt_len: usize, max_new_tokens: usize) {
        let Some(cb) = cuda_backend(&*self.backend) else { return; };
        // Called before prefill: the cache is empty, so the decode loop will
        // write positions prompt_len..prompt_len+max_new_tokens.
        let kv_len = prompt_len as u32;
        let max_pos = kv_len.saturating_add(max_new_tokens as u32);
        // Build the weights view inline (field-disjoint borrows so it
        // coexists with the mutable borrows of decode_graph / kv below).
        let weights = DecodeGraphWeights {
            token_embedding: &self.weights.token_embedding,
            layers: self.weights.layers.iter()
                .map(|l| DecodeLayerWeights {
                    attn_norm_weight: &l.attn_norm_weight,
                    wq: &l.wq, wk: &l.wk, wv: &l.wv, wo: &l.wo,
                    ffn_norm_weight: &l.ffn_norm_weight,
                    w_gate: &l.w_gate, w_up: &l.w_up, w_down: &l.w_down,
                    q_norm_weight: None,
                    k_norm_weight: None,
                    qkv_packed: l.qkv_packed.as_ref(),
                    gate_up_packed: l.gate_up_packed.as_ref(),
                })
                .collect(),
            output_norm_weight: &self.weights.output_norm_weight,
            output_weight: &self.weights.output_weight,
        };
        if let Some(dg) = self.decode_graph.as_mut() {
            if let Err(e) = dg.prewarm(cb, &weights, &mut *self.kv, max_pos) {
                // Prewarm is best-effort: on failure the lazy capture in
                // decode() still works, just slower.
                eprintln!("[apxinf] prewarm warning: {e}");
            }
        }
    }

    fn reset(&mut self) {
        let _ = self.kv.clear();
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }
}

/// Build fused weight matrices for the fused-GEMM decode path.
///
/// For each layer, creates:
/// - `qkv_packed` = concat(wq, wk, wv) along the output axis →
///   `[hidden, hidden + 2*kv_proj]`. One GEMM produces all three
///   projections; the output is split by pointer offsets.
/// - `gate_up_packed` = concat(w_gate, w_up) along the output axis →
///   `[hidden, 2*intermediate]`. One GEMM + a fused `silu_mul` kernel
///   replaces two GEMMs + silu + mul.
///
/// Both are D2D `cudaMemcpy2DAsync` on CUDA; no-ops on backends without
/// `concat_2d`. The original wq/wk/wv/w_gate/w_up stay in place for the
/// fallback path and for prefill (which doesn't use the fused path yet).
fn pack_fused_weights(weights: &mut LlamaWeights, backend: &dyn Backend) {
    for layer in &mut weights.layers {
        // qkv_packed: concat [hidden, hidden] + [hidden, kv_proj] + [hidden, kv_proj]
        if let Ok(packed) = backend.concat_2d(&[&layer.wq, &layer.wk, &layer.wv]) {
            layer.qkv_packed = Some(packed);
        }
        // gate_up_packed: concat [hidden, inter] + [hidden, inter]
        if let Ok(packed) = backend.concat_2d(&[&layer.w_gate, &layer.w_up]) {
            layer.gate_up_packed = Some(packed);
        }
    }
}

/// Transfer all weights in `weights` to the backend's device.
fn transfer_weights(weights: &LlamaWeights, backend: &dyn Backend) -> Result<LlamaWeights> {
    let layers = weights.layers.iter()
        .map(|l| Ok::<_, Error>(TransformerLayer {
            attn_norm_weight: backend.to_device(&l.attn_norm_weight)?,
            wq: backend.to_device(&l.wq)?,
            wk: backend.to_device(&l.wk)?,
            wv: backend.to_device(&l.wv)?,
            wo: backend.to_device(&l.wo)?,
            ffn_norm_weight: backend.to_device(&l.ffn_norm_weight)?,
            w_gate: backend.to_device(&l.w_gate)?,
            w_up: backend.to_device(&l.w_up)?,
            w_down: backend.to_device(&l.w_down)?,
            qkv_packed: None,
            gate_up_packed: None,
        }))
        .collect::<Result<Vec<_>>>()?;

    Ok(LlamaWeights {
        token_embedding: backend.to_device(&weights.token_embedding)?,
        layers,
        output_norm_weight: backend.to_device(&weights.output_norm_weight)?,
        output_weight: backend.to_device(&weights.output_weight)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llama::LlamaModel;
    use crate::debug::DebugCapture;

    fn tiny_config() -> ModelConfig {
        ModelConfig {
            hidden_size: 64,
            intermediate_size: 128,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 4,
            vocab_size: 100,
            max_seq_len: 128,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
        }
    }

    fn make_weight(shape: Vec<usize>) -> Tensor {
        let numel: usize = shape.iter().product();
        let data: Vec<f32> = (0..numel).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
        Tensor::from_f32(shape, &data).unwrap()
    }

    fn make_test_weights(config: &ModelConfig) -> HashMap<String, Tensor> {
        let mut tensors = HashMap::new();
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            make_weight(vec![config.vocab_size, config.hidden_size]),
        );
        for i in 0..config.n_layers {
            let prefix = format!("model.layers.{i}");
            tensors.insert(format!("{prefix}.input_layernorm.weight"),
                make_weight(vec![config.hidden_size]));
            tensors.insert(format!("{prefix}.self_attn.q_proj.weight"),
                make_weight(vec![config.hidden_size, config.hidden_size]));
            tensors.insert(format!("{prefix}.self_attn.k_proj.weight"),
                make_weight(vec![config.n_kv_heads * config.head_dim(), config.hidden_size]));
            tensors.insert(format!("{prefix}.self_attn.v_proj.weight"),
                make_weight(vec![config.n_kv_heads * config.head_dim(), config.hidden_size]));
            tensors.insert(format!("{prefix}.self_attn.o_proj.weight"),
                make_weight(vec![config.hidden_size, config.hidden_size]));
            tensors.insert(format!("{prefix}.post_attention_layernorm.weight"),
                make_weight(vec![config.hidden_size]));
            tensors.insert(format!("{prefix}.mlp.gate_proj.weight"),
                make_weight(vec![config.intermediate_size, config.hidden_size]));
            tensors.insert(format!("{prefix}.mlp.up_proj.weight"),
                make_weight(vec![config.intermediate_size, config.hidden_size]));
            tensors.insert(format!("{prefix}.mlp.down_proj.weight"),
                make_weight(vec![config.hidden_size, config.intermediate_size]));
        }
        tensors.insert("model.norm.weight".to_string(),
            make_weight(vec![config.hidden_size]));
        tensors.insert("lm_head.weight".to_string(),
            make_weight(vec![config.vocab_size, config.hidden_size]));
        tensors
    }

    #[test]
    fn test_general_llama_forward_shape() {
        let config = tiny_config();
        let tensors = make_test_weights(&config);
        let mut model = GeneralLlama::load(config.clone(), tensors, Device::Cpu).unwrap();
        let logits = model.forward(&[5], 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, config.vocab_size]);
    }

    /// Verify GeneralLlama produces the same output as legacy LlamaModel
    /// on CPU. This is the cross-check that proves the trait extraction
    /// preserved correctness.
    #[test]
    fn test_general_llama_matches_legacy_cpu() {
        let config = tiny_config();
        let tensors = make_test_weights(&config);

        // Legacy path
        let mut legacy = LlamaModel::from_weights(config.clone(), tensors.clone()).unwrap();
        let mut debug: Option<&mut DebugCapture> = None;
        let legacy_logits = legacy.forward(&[5], 0, None, &mut debug).unwrap();
        let legacy_data = legacy_logits.as_f32().unwrap().to_vec();

        // GeneralLlama path
        let mut general = GeneralLlama::load(config.clone(), tensors, Device::Cpu).unwrap();
        let general_logits = general.forward(&[5], 0).unwrap();
        let general_data = general_logits.as_f32().unwrap().to_vec();

        assert_eq!(legacy_data.len(), general_data.len(),
                   "logit count mismatch");

        let max_err: f32 = legacy_data.iter().zip(general_data.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(max_err < 1e-4,
                "GeneralLlama logits diverge from legacy: max error {max_err}");
    }

    #[test]
    fn test_general_llama_multi_token_prefill() {
        let config = tiny_config();
        let tensors = make_test_weights(&config);

        let mut legacy = LlamaModel::from_weights(config.clone(), tensors.clone()).unwrap();
        let mut debug: Option<&mut DebugCapture> = None;
        let legacy_logits = legacy.forward(&[5, 10, 15], 0, None, &mut debug).unwrap();
        let legacy_data = legacy_logits.as_f32().unwrap().to_vec();

        let mut general = GeneralLlama::load(config.clone(), tensors, Device::Cpu).unwrap();
        let general_logits = general.forward(&[5, 10, 15], 0).unwrap();
        let general_data = general_logits.as_f32().unwrap().to_vec();

        assert_eq!(legacy_data.len(), general_data.len());
        let max_err: f32 = legacy_data.iter().zip(general_data.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(max_err < 1e-3,
                "Multi-token: GeneralLlama logits diverge: max error {max_err}");
    }

    /// Multi-step autoregressive decode: prefill once, then decode several
    /// tokens. This exercises sdpa_decode at non-zero positions and the
    /// KV cache advance() across forward calls — exactly the pattern that
    /// surfaced the CUDA kv_offset bug.
    #[test]
    fn test_general_llama_multi_step_decode() {
        let config = tiny_config();
        let tensors = make_test_weights(&config);

        // Legacy path: prefill [5, 10, 15], then decode 3 tokens
        let mut legacy = LlamaModel::from_weights(config.clone(), tensors.clone()).unwrap();
        let mut legacy_cache = crate::llama::KVCache::new(&config);
        let mut debug: Option<&mut DebugCapture> = None;
        let _ = legacy.forward(&[5, 10, 15], 0, Some(&mut legacy_cache), &mut debug).unwrap();
        legacy_cache.advance(3);
        let mut legacy_tokens: Vec<u32> = Vec::new();
        let mut pos = 3usize;
        let mut current: u32 = 20;
        for _ in 0..3 {
            let logits = legacy.forward(&[current], pos, Some(&mut legacy_cache), &mut debug).unwrap();
            legacy_cache.advance(1);
            let data = logits.as_f32().unwrap();
            let argmax = data.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i).unwrap_or(0);
            legacy_tokens.push(argmax as u32);
            current = argmax as u32;
            pos += 1;
        }

        // General path: same sequence via Backend trait
        let mut general = GeneralLlama::load(config.clone(), tensors, Device::Cpu).unwrap();
        let _ = general.forward(&[5, 10, 15], 0).unwrap();
        let mut general_tokens: Vec<u32> = Vec::new();
        let mut pos = 3u32;
        let mut current: u32 = 20;
        for _ in 0..3 {
            let logits = general.forward(&[current], pos).unwrap();
            let data = logits.as_f32().unwrap();
            let argmax = data.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i).unwrap_or(0);
            general_tokens.push(argmax as u32);
            current = argmax as u32;
            pos += 1;
        }

        assert_eq!(legacy_tokens, general_tokens,
                   "Multi-step decode: token sequences diverge");
    }
}
