//! BF16 composition of the MolmoAct2 action expert: the context adapter and the flow velocity network.
//!
//! The expert is its own network (hidden 768, 36 blocks, 8 heads of width 96). Each block runs a
//! self-attention over the action rows, a cross-attention over the VLM context of the same layer, and an
//! MLP, each modulated by the timestep conditioning (`x * (1 + scale) + shift`, residual gate). All
//! rounding follows the official BF16 execution:
//! * RMS norms of the expert are non-affine: `bf16(x * rsqrt(mean(x^2) + eps))`, i.e. the unit-weight L3
//!   norm (the VLM's weighted Q/K norm does not apply here);
//! * `_modulate` is three separately rounded BF16 operations, `x * bf16(1 + scale)` then `+ shift`; the
//!   residual `x + gate * out` is a rounded product then a rounded sum;
//! * RoPE of the self-attention is `x1*cos - x2*sin`, `x1*sin + x2*cos` on a BF16 cache with every product
//!   and the sum rounded to BF16, which is the same arithmetic as the VLM's composed rotation;
//! * cross-attention has no RoPE; the context path is `context_{k,v}_proj`, `context_norm`, head reshape,
//!   and (keys only) the block's cross `k_norm`, in that order;
//! * the cross-attention mask is realised by gathering the visible context rows *before* the projection,
//!   which is the same set of keys as the official additive mask (masked probabilities are exactly zero).
//!
//! Timestep conditioning and per-step modulation depend only on the checkpoint, so they are computed
//! once ([`ExpertCache`]) like the official modulation cache; the VLM context and the per-request
//! projection are rebuilt for every observation.
use super::*;
use crate::cuda_molmoact2::backend::ops::elementwise_scale;
use crate::cuda_molmoact2::backend::{quantization, QuantizationArgs, QuantizationSemantic};
use crate::cuda_molmoact2::geometry::{
    block_modulation_ids, encoder_attention_mask, expert_inv_freq, final_modulation_ids, timestep_embedding,
};

/// Epsilon of the expert's block / context / final RMS norms (the official constructors use 1e-6).
const EXPERT_NORM_EPS: f32 = 1e-6;

/// Rows of the VLM context the action expert attends, derived from the token ids by the official
/// encoder-mask rule (valid tokens, minus every EOS, minus each action span).
pub(in crate::cuda_molmoact2) struct ContextRows {
    pub count: usize,
    ids: Tensor,
}

/// `(shift, 1 + scale, gate)` of one block, each `[steps * 3 * rows, hidden]` BF16. Row
/// `(step * 3 + stage) * rows + r`, stages self / cross / MLP, repeats the modulation vector per row.
struct BlockModulation {
    shift: Tensor,
    one_plus_scale: Tensor,
    gate: Tensor,
}

/// Checkpoint-only expert operands, built once.
pub(in crate::cuda_molmoact2) struct ExpertCache {
    steps: usize,
    rows: usize,
    blocks: Vec<BlockModulation>,
    /// `[steps * rows, hidden]`.
    final_shift: Tensor,
    final_one_plus_scale: Tensor,
    /// `[rows * heads, head_dim]` BF16 `[cos, cos]` / `[-sin, +sin]` rows and the half-swap gather ids.
    rope_cos: Tensor,
    rope_sin: Tensor,
    swap_ids: Tensor,
    /// `[rows, max_action_dim]` BF16: 1 for the valid action dimensions, 0 for the padding.
    action_mask: Tensor,
}

/// Per-request cross-attention context: for each of the 36 blocks the projected, normalised keys and
/// values `[1, rows, heads, head_dim]` over the visible VLM rows.
pub(in crate::cuda_molmoact2) struct ExpertContext {
    pub layers: Vec<(Tensor, Tensor)>,
}

impl Bf16Blocks {
    fn upload_i32(&self, v: &[i32]) -> Result<Tensor> {
        upload(&Tensor::from_i32(vec![v.len()], v)?, self.ctx.device_id())
    }

    fn upload_bf16(&self, shape: &[usize], v: &[half::bf16]) -> Result<Tensor> {
        upload(&Tensor::from_bf16(shape.to_vec(), v)?, self.ctx.device_id())
    }

    fn ones(&self, shape: &[usize]) -> Result<Tensor> {
        self.upload_bf16(shape, &vec![half::bf16::ONE; shape.iter().product()])
    }

    /// Non-affine RMSNorm over the last axis of `x [rows, cols]`: the unit-weight L3 norm.
    fn expert_rms(&self, x: &Tensor, eps: f32) -> Result<Tensor> {
        let cols = *x.shape().dims().last().ok_or_else(|| Error::Other("MolmoAct2 expert: scalar norm input".into()))?;
        let out = self.bf(x.shape().dims())?;
        norm(&self.ctx, NormArgs::rms(x, self.ones_for(cols)?, &out, eps))?;
        Ok(out)
    }

    fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let out = self.bf(a.shape().dims())?;
        elementwise_mul(&self.ctx, a, b, &out)?;
        Ok(out)
    }

    /// Official `_modulate`: `bf16(bf16(x * (1 + scale)) + shift)`.
    fn modulate(&self, x: &Tensor, one_plus_scale: &Tensor, shift: &Tensor) -> Result<Tensor> {
        self.add(&self.mul(x, one_plus_scale)?, shift)
    }

    /// Residual with a gated branch: `bf16(x + bf16(gate * branch))`.
    fn gated_residual(&self, x: &Tensor, branch: &Tensor, gate: &Tensor) -> Result<Tensor> {
        self.add(x, &self.mul(branch, gate)?)
    }

    pub fn cast_f32_to_bf16(&self, x: &Tensor) -> Result<Tensor> {
        let mut out = self.bf(x.shape().dims())?;
        quantization(&self.ctx, QuantizationArgs::new(QuantizationSemantic::CastF32ToBf16, x, &mut out))?;
        Ok(out)
    }

    pub fn cast_bf16_to_f32(&self, x: &Tensor, out: &mut Tensor) -> Result<()> {
        quantization(&self.ctx, QuantizationArgs::new(QuantizationSemantic::CastBf16ToF32, x, out))
    }

    // ---------------------------------------------------------------- request-independent cache

    /// Timestep conditioning and per-step modulation tables for `steps` flow steps over `rows` action
    /// rows, plus the RoPE operands and the padded-dimension mask. `valid_action_dims` is the number of
    /// leading action dimensions that carry an action (the rest of the padded width stays zero).
    pub fn expert_cache(&self, steps: usize, rows: usize, valid_action_dims: usize) -> Result<ExpertCache> {
        let ex = &self.cfg.expert;
        let w = &self.weights.expert;
        if steps == 0 || rows == 0 || valid_action_dims == 0 || valid_action_dims > ex.max_action_dim {
            return Err(Error::Other("MolmoAct2 expert: invalid steps / rows / action width".into()));
        }
        let h = ex.hidden;

        // conditioning = time_embed(t_i): sinusoid (BF16) -> linear -> SiLU -> linear.
        let sinusoid = self.upload_bf16(&[steps, ex.time_embed_dim], &timestep_embedding(steps, ex.time_embed_dim))?;
        let hidden = self.activate(&self.linear(&sinusoid, &w.time_in)?, ElementwiseActivation::Silu)?;
        let conditioning = self.linear(&hidden, &w.time_out)?;
        // Every modulation is `linear(silu(conditioning))`.
        let act = self.activate(&conditioning, ElementwiseActivation::Silu)?;

        let ones_blocks = self.ones(&[steps * 3 * rows, h])?;
        let ids: Vec<Tensor> = (0..3).map(|k| self.upload_i32(&block_modulation_ids(steps, rows, k))).collect::<Result<_>>()?;
        let mut blocks = Vec::with_capacity(w.blocks.len());
        for blk in &w.blocks {
            // [steps, 9 * hidden] -> rows (step * 9 + stage * 3 + kind) of width hidden.
            let table = self.linear(&act, &blk.modulation)?.reshape(dims(&[steps * 9, h]))?;
            let n = steps * 3 * rows;
            let shift = self.gather_rows(&table, &ids[0], n)?;
            let scale = self.gather_rows(&table, &ids[1], n)?;
            let gate = self.gather_rows(&table, &ids[2], n)?;
            blocks.push(BlockModulation { shift, one_plus_scale: self.add(&scale, &ones_blocks)?, gate });
        }
        let final_table = self.linear(&act, &w.final_modulation)?.reshape(dims(&[steps * 2, h]))?;
        let n = steps * rows;
        let final_shift = self.gather_rows(&final_table, &self.upload_i32(&final_modulation_ids(steps, rows, 0))?, n)?;
        let final_scale = self.gather_rows(&final_table, &self.upload_i32(&final_modulation_ids(steps, rows, 1))?, n)?;
        let final_one_plus_scale = self.add(&final_scale, &self.ones(&[n, h])?)?;

        // Self-attention RoPE operands for `rows` positions x `heads` heads.
        let (heads, hd) = (ex.heads, ex.head_dim);
        let table = rope_table(&expert_inv_freq(hd), rows);
        let (cos, sin) = rope_operands_per_head(&table, rows, heads, hd / 2);
        let r = rows * heads;
        let mut mask = vec![half::bf16::ZERO; rows * ex.max_action_dim];
        for row in mask.chunks_mut(ex.max_action_dim) {
            row[..valid_action_dims].fill(half::bf16::ONE);
        }
        Ok(ExpertCache {
            steps,
            rows,
            blocks,
            final_shift,
            final_one_plus_scale,
            rope_cos: self.upload_bf16(&[r, hd], &cos)?,
            rope_sin: self.upload_bf16(&[r, hd], &sin)?,
            swap_ids: self.upload_i32(&half_swap_ids(r))?,
            action_mask: self.upload_bf16(&[rows, ex.max_action_dim], &mask)?,
        })
    }

    // ---------------------------------------------------------------- context adapter

    /// Indices of the VLM rows the expert may attend (official `_get_encoder_attention_mask`, action mode
    /// `both`).
    pub(super) fn visible_rows(&self, token_ids: &[u32], valid: &[bool]) -> Result<Vec<i32>> {
        let mask = encoder_attention_mask(&self.cfg, token_ids, valid);
        let rows: Vec<i32> = mask.iter().enumerate().filter(|&(_, &m)| m).map(|(i, _)| i as i32).collect();
        if rows.is_empty() {
            return Err(Error::Other("MolmoAct2 input: the expert would see no context rows".into()));
        }
        Ok(rows)
    }

    /// Visible VLM rows for one request, uploaded.
    pub fn context_rows_host(&self, host: &HostPlan) -> Result<ContextRows> {
        Ok(ContextRows { count: host.rows.len(), ids: self.upload_i32(&host.rows)? })
    }

    /// Overwrite the visible row indices in place (same count: the shapes of a prepared profile).
    pub(super) fn write_context_rows(&self, rows: &ContextRows, ids: &[i32]) -> Result<()> {
        if ids.len() != rows.count {
            return Err(Error::Other("MolmoAct2: the number of visible context rows changed".into()));
        }
        let bytes: Vec<u8> = ids.iter().flat_map(|x| x.to_ne_bytes()).collect();
        io(io(CudaBuffer::from_tensor(&rows.ids))?.copy_from_host_on(&bytes, self.ctx.stream()))
    }

    /// One layer of the context adapter: gather the visible rows of the VLM cache `[tokens, kv_heads,
    /// head_dim]` -> shared `context_{k,v}_proj` (no bias) -> `context_norm` -> head reshape -> (keys
    /// only) the block's cross `k_norm`. Returns `[1, rows, heads, head_dim]` tensors.
    pub fn expert_context_layer(&self, kv: &LayerKv, rows: &ContextRows) -> Result<(Tensor, Tensor)> {
        let ex = &self.cfg.expert;
        let w = &self.weights.expert;
        let kv_width = self.cfg.text.kv_heads * self.cfg.text.head_dim;
        let tokens = kv.key.shape().dims()[0];
        let project = |t: &Tensor, proj: &Linear, key_norm: bool| -> Result<Tensor> {
            let visible = self.gather_rows(&t.reshape(dims(&[tokens, kv_width]))?, &rows.ids, rows.count)?;
            let normed = self.expert_rms(&self.linear(&visible, proj)?, EXPERT_NORM_EPS)?;
            let heads = normed.reshape(dims(&[rows.count * ex.heads, ex.head_dim]))?;
            let out = if key_norm { self.expert_rms(&heads, ex.qk_norm_eps)? } else { heads };
            out.reshape(dims(&[1, rows.count, ex.heads, ex.head_dim]))
        };
        Ok((project(&kv.key, &w.context_k, true)?, project(&kv.value, &w.context_v, false)?))
    }

    /// The whole adapter: one `(K, V)` pair per VLM layer.
    pub fn expert_context(&self, kv: &[LayerKv], rows: &ContextRows) -> Result<ExpertContext> {
        if kv.len() != self.weights.expert.blocks.len() {
            return Err(Error::Other(format!("MolmoAct2 expert: {} K/V layers for {} blocks", kv.len(), self.weights.expert.blocks.len())));
        }
        let mut layers = Vec::with_capacity(kv.len());
        for layer in kv {
            layers.push(self.expert_context_layer(layer, rows)?);
        }
        Ok(ExpertContext { layers })
    }

    // ---------------------------------------------------------------- velocity network

    /// RoPE of the self-attention on normalised rows `[rows * heads, head_dim]` (see module docs).
    fn expert_rope(&self, n: &Tensor, cache: &ExpertCache) -> Result<Tensor> {
        let ex = &self.cfg.expert;
        let rows = cache.rows * ex.heads;
        let halves = n.reshape(dims(&[rows * 2, ex.head_dim / 2]))?;
        let swapped = self.gather_rows(&halves, &cache.swap_ids, rows * 2)?.reshape(dims(&[rows, ex.head_dim]))?;
        let (p1, p2) = (self.bf(&[rows, ex.head_dim])?, self.bf(&[rows, ex.head_dim])?);
        elementwise_mul(&self.ctx, n, &cache.rope_cos, &p1)?;
        elementwise_mul(&self.ctx, &swapped, &cache.rope_sin, &p2)?;
        self.add(&p1, &p2)
    }

    /// Modulation operands of one `(step, stage)`: `[rows, hidden]` views of a block table.
    fn stage_rows(&self, table: &Tensor, step: usize, stage: usize, cache: &ExpertCache) -> Result<Tensor> {
        self.rows_view(table, (step * 3 + stage) * cache.rows, cache.rows, &[cache.rows, self.cfg.expert.hidden])
    }

    /// One expert block at flow step `step` on action rows `x [rows, hidden]`.
    pub fn expert_block(
        &self,
        index: usize,
        x: &Tensor,
        step: usize,
        cache: &ExpertCache,
        context: &(Tensor, Tensor),
    ) -> Result<Tensor> {
        let ex = &self.cfg.expert;
        let blk = &self.weights.expert.blocks[index];
        let m = &cache.blocks[index];
        let (rows, heads, hd) = (cache.rows, ex.heads, ex.head_dim);
        let attention = |q: &Tensor, k: &Tensor, v: &Tensor| -> Result<Tensor> {
            let mut out = self.bf(&[1, rows, heads, hd])?;
            self.dense_attention(q, k, v, &mut out, false)?;
            out.reshape(dims(&[rows, ex.hidden]))
        };
        let per_head = |t: &Tensor| -> Result<Tensor> { t.reshape(dims(&[rows * heads, hd])) };

        // self-attention: Q/K non-affine norm, RoPE, no mask, no causal.
        let n = self.expert_rms(x, EXPERT_NORM_EPS)?;
        let h = self.modulate(&n, &self.stage_rows(&m.one_plus_scale, step, 0, cache)?, &self.stage_rows(&m.shift, step, 0, cache)?)?;
        let q = self.expert_rms(&per_head(&self.linear(&h, &blk.self_q)?)?, ex.qk_norm_eps)?;
        let k = self.expert_rms(&per_head(&self.linear(&h, &blk.self_k)?)?, ex.qk_norm_eps)?;
        let v = self.linear(&h, &blk.self_v)?;
        let q = self.expert_rope(&q, cache)?.reshape(dims(&[1, rows, heads, hd]))?;
        let k = self.expert_rope(&k, cache)?.reshape(dims(&[1, rows, heads, hd]))?;
        let attn = attention(&q, &k, &v.reshape(dims(&[1, rows, heads, hd]))?)?;
        let x1 = self.gated_residual(x, &self.linear(&attn, &blk.self_out)?, &self.stage_rows(&m.gate, step, 0, cache)?)?;

        // cross-attention over this layer's context; the queries are normalised, the keys already are.
        let n = self.expert_rms(&x1, EXPERT_NORM_EPS)?;
        let h = self.modulate(&n, &self.stage_rows(&m.one_plus_scale, step, 1, cache)?, &self.stage_rows(&m.shift, step, 1, cache)?)?;
        let q = self.expert_rms(&per_head(&self.linear(&h, &blk.cross_q)?)?, ex.qk_norm_eps)?;
        let attn = attention(&q.reshape(dims(&[1, rows, heads, hd]))?, &context.0, &context.1)?;
        let x2 = self.gated_residual(&x1, &self.linear(&attn, &blk.cross_out)?, &self.stage_rows(&m.gate, step, 1, cache)?)?;

        // MLP: down(silu(gate(h)) * up(h)).
        let n = self.expert_rms(&x2, EXPERT_NORM_EPS)?;
        let h = self.modulate(&n, &self.stage_rows(&m.one_plus_scale, step, 2, cache)?, &self.stage_rows(&m.shift, step, 2, cache)?)?;
        let g = self.activate(&self.linear(&h, &blk.gate)?, ElementwiseActivation::Silu)?;
        let u = self.linear(&h, &blk.up)?;
        let mlp = self.linear(&self.mul(&g, &u)?, &blk.down)?;
        self.gated_residual(&x2, &mlp, &self.stage_rows(&m.gate, step, 2, cache)?)
    }

    /// Velocity `[rows, max_action_dim]` of the trajectory `[rows, max_action_dim]` at flow step `step`
    /// (before the padded-dimension mask).
    pub fn expert_velocity(
        &self,
        trajectory: &Tensor,
        step: usize,
        cache: &ExpertCache,
        context: &ExpertContext,
    ) -> Result<Tensor> {
        let w = &self.weights.expert;
        let mut x = self.linear(trajectory, &w.action_embed)?;
        for (i, ctx_layer) in context.layers.iter().enumerate() {
            x = self.expert_block(i, &x, step, cache, ctx_layer)?;
        }
        let n = self.expert_rms(&x, EXPERT_NORM_EPS)?;
        let rows = cache.rows;
        let h = self.modulate(
            &n,
            &self.rows_view(&cache.final_one_plus_scale, step * rows, rows, &[rows, self.cfg.expert.hidden])?,
            &self.rows_view(&cache.final_shift, step * rows, rows, &[rows, self.cfg.expert.hidden])?,
        )?;
        self.linear(&h, &w.final_out)
    }

    /// Forward Euler flow from the BF16 initial noise `[rows, max_action_dim]` (padded dimensions are
    /// zeroed first): `t_i = i / steps`, `x <- bf16(x + bf16(dt * v))`, velocity masked on the padded
    /// dimensions. Returns the normalized padded actions, BF16 `[rows, max_action_dim]`.
    pub fn expert_flow(&self, noise: &Tensor, cache: &ExpertCache, context: &ExpertContext) -> Result<Tensor> {
        let ex = &self.cfg.expert;
        if noise.dtype() != DType::BF16 || noise.shape().dims() != [cache.rows, ex.max_action_dim] {
            return Err(Error::Other(format!("MolmoAct2 flow: noise must be BF16 [{}, {}]", cache.rows, ex.max_action_dim)));
        }
        let dt = (1.0f64 / cache.steps as f64) as f32;
        let mut trajectory = self.mul(noise, &cache.action_mask)?;
        for step in 0..cache.steps {
            let velocity = self.mul(&self.expert_velocity(&trajectory, step, cache, context)?, &cache.action_mask)?;
            let delta = self.bf(velocity.shape().dims())?;
            elementwise_scale(&self.ctx, &velocity, &delta, dt)?;
            trajectory = self.add(&trajectory, &delta)?;
        }
        Ok(trajectory)
    }
}
