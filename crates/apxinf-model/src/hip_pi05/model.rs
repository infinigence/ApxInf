//! π0.5 BF16 dataflow on HIP: vision → prefix KV → fixed-step flow matching.
//!
//! Each layer follows the CUDA BF16 blocks (`pi05/model/blocks/bf16.rs`)
//! operator for operator on `apxinf_hip::native`, so the two implementations
//! round at the same points and their stages can be compared directly — by
//! design, not by sharing code. Every activation comes from the caller's
//! [`HipArena`] in a fixed order, so a run allocates the same addresses each
//! time; nothing synchronizes until the caller reads a result.
//!
//! Two deliberate differences from CUDA, neither of which changes a value:
//! the per-step modulation vectors depend only on the fixed time schedule, so
//! they are computed once at construction rather than every inference; and the
//! language layers write K/V straight into the prefix cache, which already has
//! the action expert's rows reserved, instead of copying them into it.

use std::sync::Arc;

use apxinf_core::{DType, Error, Result, Tensor};
use apxinf_hip::native::{self, Activation, AttentionGeometry, HipArena, HipDeviceBytes, QkvRope};
use apxinf_hip::HipContext;
use half::bf16;

use super::checkpoint::Checkpoint;
use super::config::{sinusoidal_time_embedding, Config};
use super::weights::{ActionLayer, HipPi05Weights, LanguageLayer, Linear, VisionLayer};

/// Per-layer K and V of the prefix, `[prefix + action_horizon, kv_heads, head_dim]`.
/// Rows `[0, tokens)` hold the prefix; each denoise step overwrites the rest
/// with the action tokens' K/V before attending over all of them.
pub struct PrefixKvCache {
    pub keys: Vec<Tensor>,
    pub values: Vec<Tensor>,
    pub tokens: usize,
}

/// Modulation for one flow step: per action layer, the `[3 * width]`
/// scale/shift/gate of the attention input and of the MLP input, plus the
/// final norm's.
struct StepModulation {
    attention: Vec<Tensor>,
    mlp: Vec<Tensor>,
    final_norm: Tensor,
}

pub struct HipPi05Model {
    ctx: Arc<HipContext>,
    config: Arc<Config>,
    weights: HipPi05Weights,
    modulation: Vec<StepModulation>,
    /// Owns the modulation tensors; never reset.
    constants: HipArena,
}

/// Allocation and the two compound steps every layer repeats.
struct Ops<'a> {
    ctx: &'a HipContext,
    arena: &'a HipArena,
}

impl Ops<'_> {
    fn tensor(&self, dims: &[usize]) -> Result<Tensor> {
        self.arena.tensor(dims.to_vec(), DType::BF16)
    }

    /// `x @ weight` alone, for projections whose bias a later fused kernel adds.
    fn matmul(&self, x: &Tensor, weight: &Tensor) -> Result<Tensor> {
        let rows = x.numel() / x.shape().dims().last().copied().unwrap_or(1);
        let out = self.tensor(&[rows, weight.shape().dims()[1]])?;
        native::gemm(self.ctx, x, weight, &out)?;
        Ok(out)
    }

    /// `x @ W`, then the bias in a second rounding — CUDA's GEMM-then-bias.
    fn linear(&self, x: &Tensor, linear: &Linear) -> Result<Tensor> {
        self.linear_activation(x, linear, Activation::None)
    }

    fn linear_activation(
        &self,
        x: &Tensor,
        linear: &Linear,
        activation: Activation,
    ) -> Result<Tensor> {
        let projected = self.matmul(x, &linear.weight)?;
        if linear.bias.is_none() && activation == Activation::None {
            return Ok(projected);
        }
        let out = self.tensor(projected.shape().dims())?;
        native::bias_activation(self.ctx, &projected, linear.bias.as_ref(), activation, &out)?;
        Ok(out)
    }

    /// Packed gate/up GEMM followed by GeGLU.
    fn geglu(&self, x: &Tensor, gate_up: &Linear) -> Result<Tensor> {
        let packed = self.linear(x, gate_up)?;
        let dims = packed.shape().dims();
        let out = self.tensor(&[dims[0], dims[1] / 2])?;
        native::geglu(self.ctx, &packed, &out)?;
        Ok(out)
    }
}

impl HipPi05Model {
    pub fn new(ctx: Arc<HipContext>, config: Arc<Config>, checkpoint: &Checkpoint) -> Result<Self> {
        config.validate()?;
        let weights = HipPi05Weights::from_checkpoint(checkpoint, &ctx)?;
        if weights.vision_layers.len() != config.vision_depth
            || weights.language_layers.len() != config.language.depth
            || weights.action_layers.len() != config.action_expert.depth
        {
            return Err(Error::Other(
                "π0.5 HIP: weight depth does not match the config".into(),
            ));
        }
        let constants = HipArena::new(&ctx, 64 << 20)?;
        let mut model = Self {
            ctx,
            config,
            weights,
            modulation: Vec::new(),
            constants,
        };
        model.modulation = model.prepare_modulation()?;
        model.ctx.synchronize()?;
        Ok(model)
    }

    pub fn config(&self) -> &Arc<Config> {
        &self.config
    }

    pub fn context(&self) -> &Arc<HipContext> {
        &self.ctx
    }

    fn ops<'a>(&'a self, arena: &'a HipArena) -> Ops<'a> {
        Ops {
            ctx: &self.ctx,
            arena,
        }
    }

    /// Bytes of arena one inference over `token_count` prompt tokens uses,
    /// with headroom. Counts the same allocations, in the same order, as the
    /// stage methods below.
    pub fn arena_bytes(config: &Config, token_count: usize) -> usize {
        let mut total = 0usize;
        let mut add = |elements: usize| total += (elements * 2).next_multiple_of(256);
        let (vw, vm) = (config.vision_width, config.vision_mlp_dim);
        let rows = config.num_views * config.patches_per_view();
        let lang = config.language;
        let act = config.action_expert;
        // Vision: patch embedding, layers, post-norm and projector.
        add(rows * vw);
        add(rows * vw);
        for _ in 0..config.vision_depth {
            for width in [vw, 3 * vw, vw, vw, vw, vw, vw, vw, vw, vm, vm, vw, vw] {
                add(rows * width);
            }
        }
        for width in [vw, lang.width, lang.width] {
            add(rows * width);
        }
        // Prefix embedding.
        let prefix = rows + token_count;
        add(token_count * lang.width);
        add(prefix * lang.width);
        // Language layers and the prefix cache.
        let cache_rows = prefix + config.action_horizon;
        for _ in 0..lang.depth {
            add(cache_rows * lang.num_kv_heads * lang.head_dim);
            add(cache_rows * lang.num_kv_heads * lang.head_dim);
            let attention = lang.num_heads * lang.head_dim;
            for width in [
                lang.width,
                lang.qkv_width(),
                attention,
                attention,
                lang.width,
                lang.width,
                lang.width,
            ] {
                add(prefix * width);
            }
            for width in [2 * lang.mlp_dim, lang.mlp_dim, lang.width, lang.width] {
                add(prefix * width);
            }
        }
        // Flow steps.
        let h = config.action_horizon;
        for _ in 0..config.num_flow_steps {
            add(h * act.width);
            add(h * act.width);
            add(h * act.width);
            for _ in 0..act.depth {
                let attention = act.num_heads * act.head_dim;
                for width in [
                    act.qkv_width(),
                    attention,
                    attention,
                    act.width,
                    act.width,
                    act.width,
                ] {
                    add(h * width);
                }
                for width in [
                    2 * act.mlp_dim,
                    act.mlp_dim,
                    act.width,
                    act.width,
                    act.width,
                ] {
                    add(h * width);
                }
            }
            for _ in 0..3 {
                add(h * config.action_dim);
            }
        }
        total + total / 8 + (32 << 20)
    }

    /// Patches `[views * patches_per_view, 3 * patch²]` → vision tokens at the
    /// language width.
    pub fn encode_vision(&self, arena: &HipArena, patches: &Tensor) -> Result<Tensor> {
        let ops = self.ops(arena);
        let c = &self.config;
        let w = &self.weights;
        let projected = ops.matmul(patches, &w.patch_embedding.weight)?;
        let mut hidden = ops.tensor(projected.shape().dims())?;
        native::add_position(
            &self.ctx,
            &projected,
            w.patch_embedding.bias.as_ref(),
            &w.position_embedding,
            c.patches_per_view(),
            &hidden,
        )?;
        for layer in &w.vision_layers {
            hidden = self.vision_layer(&ops, layer, &hidden)?;
        }
        let normalized = ops.tensor(hidden.shape().dims())?;
        native::layer_norm(
            &self.ctx,
            &hidden,
            &w.vision_post_norm.weight,
            &w.vision_post_norm.bias,
            c.layer_norm_eps,
            &normalized,
        )?;
        let projected = ops.tensor(&[hidden.shape().dims()[0], c.language.width])?;
        native::gemm(
            &self.ctx,
            &normalized,
            &w.multimodal_projector.weight,
            &projected,
        )?;
        let out = ops.tensor(projected.shape().dims())?;
        native::bias_activation(
            &self.ctx,
            &projected,
            w.multimodal_projector.bias.as_ref(),
            Activation::None,
            &out,
        )?;
        Ok(out)
    }

    fn vision_layer(&self, ops: &Ops, layer: &VisionLayer, input: &Tensor) -> Result<Tensor> {
        let c = &self.config;
        let rows = input.shape().dims()[0];
        let width = c.vision_width;
        let normalized = ops.tensor(&[rows, width])?;
        native::layer_norm(
            &self.ctx,
            input,
            &layer.norm1.weight,
            &layer.norm1.bias,
            c.layer_norm_eps,
            &normalized,
        )?;
        let qkv = ops.tensor(&[rows, 3 * width])?;
        native::gemm(&self.ctx, &normalized, &layer.qkv.weight, &qkv)?;
        let heads = [rows, c.vision_heads, c.vision_head_dim];
        let (q, k, v) = (
            ops.tensor(&heads)?,
            ops.tensor(&heads)?,
            ops.tensor(&heads)?,
        );
        native::split_qkv_bias(&self.ctx, &qkv, layer.qkv.bias.as_ref(), &q, &k, &v)?;
        let attention = ops.tensor(&[rows, width])?;
        let per_view = c.patches_per_view();
        native::attention(
            &self.ctx,
            &q,
            &k,
            &v,
            &AttentionGeometry {
                batch: rows / per_view,
                q_tokens: per_view,
                q_heads: c.vision_heads,
                kv_heads: c.vision_heads,
                head_dim: c.vision_head_dim,
                kv_tokens: per_view,
            },
            &attention,
        )?;
        let projected = ops.tensor(&[rows, width])?;
        native::gemm(&self.ctx, &attention, &layer.output.weight, &projected)?;
        let (hidden, normalized) = (ops.tensor(&[rows, width])?, ops.tensor(&[rows, width])?);
        native::bias_residual_layer_norm(
            &self.ctx,
            &projected,
            layer.output.bias.as_ref(),
            input,
            &layer.norm2.weight,
            &layer.norm2.bias,
            c.layer_norm_eps,
            &hidden,
            &normalized,
        )?;
        let activated = ops.linear_activation(&normalized, &layer.fc1, Activation::GeluTanh)?;
        let projected = ops.tensor(&[rows, width])?;
        native::gemm(&self.ctx, &activated, &layer.fc2.weight, &projected)?;
        let out = ops.tensor(&[rows, width])?;
        native::bias_residual(
            &self.ctx,
            &projected,
            layer.fc2.bias.as_ref(),
            &hidden,
            &out,
        )?;
        Ok(out)
    }

    /// Vision tokens followed by the scaled embeddings of the first
    /// `token_count` ids: the `[prefix, language_width]` prefix.
    pub fn embed_prefix(
        &self,
        arena: &HipArena,
        vision: &Tensor,
        token_ids: &HipDeviceBytes,
        token_count: usize,
    ) -> Result<Tensor> {
        if token_count == 0 || token_count > self.config.max_token_len {
            return Err(Error::Other(format!(
                "π0.5 token count must be in 1..={}, got {token_count}",
                self.config.max_token_len
            )));
        }
        let ops = self.ops(arena);
        let width = self.config.language.width;
        let language = ops.tensor(&[token_count, width])?;
        native::embedding(
            &self.ctx,
            &self.weights.token_embedding,
            token_ids,
            &language,
        )?;
        let vision_rows = vision.shape().dims()[0];
        let prefix = ops.tensor(&[vision_rows + token_count, width])?;
        native::copy_rows(&self.ctx, vision, &prefix, 0)?;
        native::copy_rows(&self.ctx, &language, &prefix, vision_rows)?;
        Ok(prefix)
    }

    /// The language model over the prefix, keeping only each layer's K/V. The
    /// last layer stops after its QKV projection: nothing reads its output.
    pub fn prefix_forward(&self, arena: &HipArena, prefix: &Tensor) -> Result<PrefixKvCache> {
        let ops = self.ops(arena);
        let c = &self.config;
        let lang = c.language;
        let tokens = prefix.shape().dims()[0];
        let cache_rows = tokens + c.action_horizon;
        let mut hidden = prefix.clone();
        let (mut keys, mut values) = (Vec::new(), Vec::new());
        for (index, layer) in self.weights.language_layers.iter().enumerate() {
            let kv = [cache_rows, lang.num_kv_heads, lang.head_dim];
            let (key, value) = (ops.tensor(&kv)?, ops.tensor(&kv)?);
            let last = index + 1 == lang.depth;
            hidden = self.language_layer(&ops, layer, &hidden, &key, &value, !last)?;
            keys.push(key);
            values.push(value);
        }
        Ok(PrefixKvCache {
            keys,
            values,
            tokens,
        })
    }

    fn language_layer(
        &self,
        ops: &Ops,
        layer: &LanguageLayer,
        input: &Tensor,
        key: &Tensor,
        value: &Tensor,
        compute_tail: bool,
    ) -> Result<Tensor> {
        let c = &self.config;
        let lang = c.language;
        let tokens = input.shape().dims()[0];
        let normalized = ops.tensor(&[tokens, lang.width])?;
        native::rms_norm(
            &self.ctx,
            input,
            &layer.input_norm_scale,
            c.rms_norm_eps,
            &normalized,
        )?;
        let qkv = ops.matmul(&normalized, &layer.qkv.weight)?;
        let q = ops.tensor(&[tokens, lang.num_heads, lang.head_dim])?;
        let rope = QkvRope {
            q_heads: lang.num_heads,
            kv_heads: lang.num_kv_heads,
            head_dim: lang.head_dim,
            theta: c.rope_theta,
            position_offset: 0,
        };
        native::split_qkv_rope(
            &self.ctx,
            &qkv,
            layer.qkv.bias.as_ref(),
            &rope,
            &q,
            key,
            value,
            0,
        )?;
        if !compute_tail {
            return Ok(input.clone());
        }
        let attention = ops.tensor(&[tokens, lang.num_heads * lang.head_dim])?;
        native::attention(
            &self.ctx,
            &q,
            key,
            value,
            &AttentionGeometry {
                batch: 1,
                q_tokens: tokens,
                q_heads: lang.num_heads,
                kv_heads: lang.num_kv_heads,
                head_dim: lang.head_dim,
                kv_tokens: tokens,
            },
            &attention,
        )?;
        let projected = ops.tensor(&[tokens, lang.width])?;
        native::gemm(&self.ctx, &attention, &layer.output.weight, &projected)?;
        let (hidden, normalized) = (
            ops.tensor(&[tokens, lang.width])?,
            ops.tensor(&[tokens, lang.width])?,
        );
        native::bias_residual_rms_norm(
            &self.ctx,
            &projected,
            layer.output.bias.as_ref(),
            input,
            &layer.post_attention_norm_scale,
            c.rms_norm_eps,
            &hidden,
            &normalized,
        )?;
        let activated = ops.geglu(&normalized, &layer.gate_up)?;
        let projected = ops.tensor(&[tokens, lang.width])?;
        native::gemm(&self.ctx, &activated, &layer.down.weight, &projected)?;
        let out = ops.tensor(&[tokens, lang.width])?;
        native::bias_residual(
            &self.ctx,
            &projected,
            layer.down.bias.as_ref(),
            &hidden,
            &out,
        )?;
        Ok(out)
    }

    /// Time conditioning → per-layer modulation, for every flow step.
    fn prepare_modulation(&self) -> Result<Vec<StepModulation>> {
        let ops = self.ops(&self.constants);
        let c = &self.config;
        let w = &self.weights;
        (0..c.num_flow_steps)
            .map(|step| {
                let time = c.flow_start_time * (1.0 - step as f32 / c.num_flow_steps as f32);
                let embedding: Vec<bf16> = sinusoidal_time_embedding(
                    time,
                    c.action_expert.width,
                    c.time_min_period,
                    c.time_max_period,
                )
                .into_iter()
                .map(bf16::from_f32)
                .collect();
                let embedding = self.ctx.to_device(&Tensor::from_bf16(
                    vec![1, c.action_expert.width],
                    &embedding,
                )?)?;
                let hidden = ops.linear_activation(&embedding, &w.time_mlp_in, Activation::Silu)?;
                let conditioning =
                    ops.linear_activation(&hidden, &w.time_mlp_out, Activation::Silu)?;
                let project = |linear: &Linear| -> Result<Tensor> {
                    let m = ops.linear(&conditioning, linear)?;
                    m.reshape(vec![m.numel()])
                };
                Ok(StepModulation {
                    attention: w
                        .action_layers
                        .iter()
                        .map(|l| project(&l.input_modulation))
                        .collect::<Result<_>>()?,
                    mlp: w
                        .action_layers
                        .iter()
                        .map(|l| project(&l.post_attention_modulation))
                        .collect::<Result<_>>()?,
                    final_norm: project(&w.action_final_modulation)?,
                })
            })
            .collect()
    }

    /// One Euler step of flow `step` from `state` `[action_horizon, action_dim]`.
    pub fn denoise_step(
        &self,
        arena: &HipArena,
        state: &Tensor,
        step: usize,
        prefix: &PrefixKvCache,
    ) -> Result<Tensor> {
        let ops = self.ops(arena);
        let c = &self.config;
        let modulation = self
            .modulation
            .get(step)
            .ok_or_else(|| Error::Other(format!("π0.5 flow step {step} out of range")))?;
        let depth = c.action_expert.depth;
        if prefix.keys.len() != depth || prefix.values.len() != depth {
            return Err(Error::Other("π0.5 HIP: prefix cache depth mismatch".into()));
        }
        let mut hidden = ops.linear(state, &self.weights.action_in)?;
        let mut normalized: Option<Tensor> = None;
        for (index, layer) in self.weights.action_layers.iter().enumerate() {
            let next_norm = modulation
                .attention
                .get(index + 1)
                .unwrap_or(&modulation.final_norm);
            let (h, n) = self.action_layer(
                &ops,
                layer,
                &hidden,
                normalized.as_ref(),
                &modulation.attention[index],
                &modulation.mlp[index],
                next_norm,
                &prefix.keys[index],
                &prefix.values[index],
                prefix.tokens,
            )?;
            hidden = h;
            normalized = Some(n);
        }
        let normalized =
            normalized.ok_or_else(|| Error::Other("π0.5 action expert has no layers".into()))?;
        let velocity = ops.linear(&normalized, &self.weights.action_out)?;
        let next = ops.tensor(state.shape().dims())?;
        let dt = -c.flow_start_time / c.num_flow_steps as f32;
        native::euler_update(&self.ctx, state, &velocity, dt, &next)?;
        Ok(next)
    }

    #[allow(clippy::too_many_arguments)]
    fn action_layer(
        &self,
        ops: &Ops,
        layer: &ActionLayer,
        input: &Tensor,
        attention_normalized: Option<&Tensor>,
        attention_modulation: &Tensor,
        mlp_modulation: &Tensor,
        next_norm_modulation: &Tensor,
        key: &Tensor,
        value: &Tensor,
        prefix_tokens: usize,
    ) -> Result<(Tensor, Tensor)> {
        let c = &self.config;
        let act = c.action_expert;
        let tokens = input.shape().dims()[0];
        let normalized = match attention_normalized {
            Some(normalized) => normalized.clone(),
            None => {
                let normalized = ops.tensor(&[tokens, act.width])?;
                native::ada_rms_norm(
                    &self.ctx,
                    input,
                    attention_modulation,
                    c.rms_norm_eps,
                    &normalized,
                )?;
                normalized
            }
        };
        let qkv = ops.matmul(&normalized, &layer.qkv.weight)?;
        let q = ops.tensor(&[tokens, act.num_heads, act.head_dim])?;
        let rope = QkvRope {
            q_heads: act.num_heads,
            kv_heads: act.num_kv_heads,
            head_dim: act.head_dim,
            theta: c.rope_theta,
            position_offset: prefix_tokens,
        };
        native::split_qkv_rope(
            &self.ctx,
            &qkv,
            layer.qkv.bias.as_ref(),
            &rope,
            &q,
            key,
            value,
            prefix_tokens,
        )?;
        let attention = ops.tensor(&[tokens, act.num_heads * act.head_dim])?;
        native::attention(
            &self.ctx,
            &q,
            key,
            value,
            &AttentionGeometry {
                batch: 1,
                q_tokens: tokens,
                q_heads: act.num_heads,
                kv_heads: act.num_kv_heads,
                head_dim: act.head_dim,
                kv_tokens: prefix_tokens + tokens,
            },
            &attention,
        )?;
        let projected = ops.tensor(&[tokens, act.width])?;
        native::gemm(&self.ctx, &attention, &layer.output.weight, &projected)?;
        let (hidden, normalized) = (
            ops.tensor(&[tokens, act.width])?,
            ops.tensor(&[tokens, act.width])?,
        );
        native::ada_gate_residual_rms_norm(
            &self.ctx,
            &projected,
            input,
            attention_modulation,
            mlp_modulation,
            c.rms_norm_eps,
            &hidden,
            &normalized,
        )?;
        let activated = ops.geglu(&normalized, &layer.gate_up)?;
        let projected = ops.tensor(&[tokens, act.width])?;
        native::gemm(&self.ctx, &activated, &layer.down.weight, &projected)?;
        let (out, out_normalized) = (
            ops.tensor(&[tokens, act.width])?,
            ops.tensor(&[tokens, act.width])?,
        );
        native::ada_gate_residual_rms_norm(
            &self.ctx,
            &projected,
            &hidden,
            mlp_modulation,
            next_norm_modulation,
            c.rms_norm_eps,
            &out,
            &out_normalized,
        )?;
        Ok((out, out_normalized))
    }

    /// The whole inference: patches, token ids and initial noise → actions
    /// `[action_horizon, action_dim]`, in the arena.
    pub fn infer(
        &self,
        arena: &HipArena,
        patches: &Tensor,
        token_ids: &HipDeviceBytes,
        token_count: usize,
        noise: &Tensor,
    ) -> Result<Tensor> {
        let vision = self.encode_vision(arena, patches)?;
        let prefix = self.embed_prefix(arena, &vision, token_ids, token_count)?;
        let cache = self.prefix_forward(arena, &prefix)?;
        let mut state = noise.clone();
        for step in 0..self.config.num_flow_steps {
            state = self.denoise_step(arena, &state, step, &cache)?;
        }
        Ok(state)
    }
}
