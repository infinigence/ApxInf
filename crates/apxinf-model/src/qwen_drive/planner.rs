//! Planning Expert executor: the flow-matching diffusion transformer that
//! turns VLM scene caches into waypoints.
//!
//! The mathematics here are a line-level port of the pinned reference
//! (`src/qwen_drive/planning_expert.py`, revision 28091c1):
//!
//! * joint attention: every layer attends over
//!   `concat(scene_K_from_VLM, waypoint_K)` (same for V), non-causal, GQA
//!   with 4 KV heads and a per-head sigmoid output gate;
//! * adaLN conditioning: `condition = time_mlp(t) + nav_mlp + ego_mlp`,
//!   modulated as `norm(x) * (1 + scale) + shift` and
//!   `residual + (1 + gate) * proj(...)`;
//! * waypoint queries fuse 7 signals (noisy waypoint, Fourier features, flow
//!   time, history poses, waypoint-index embedding, history velocity, history
//!   acceleration);
//! * partial interleaved mRoPE at positions `anchor + 1 ..= anchor + 50`,
//!   with the reference's bfloat16-rounded inverse-frequency table AND
//!   bfloat16-rounded positions (rounding above position 256 is semantic);
//! * the Fourier frequency table is likewise rebuilt in bf16 every call.
//!
//! ## Device status (honest accounting)
//!
//! This executor currently computes `predict_endpoint` on the host in f32
//! with bf16-rounded inputs, as the named correctness scaffold for the CUDA
//! port. It is numerically structured (every op is a pure function of the
//! same tensors the reference multiplies on GPU) but it is not bit-exact
//! bf16 arithmetic: the reference accumulates GEMMs in f32 and rounds on
//! store, which this scaffold does not reproduce. The device replacement
//! plan is the execution ledger L-10..L-17 (safe bf16 GEMM composition, the
//! L-12 QKV split, the L-13 bf16 rope table, the L-14 joint-attention
//! wrapper), landing with the CUDA executor gate. The VLM scene-cache source
//! is likewise pending (see `mod.rs`), so end-to-end planning activates when
//! both halves land. Nothing here fabricates a trajectory.

use apxinf_core::{Error, Result, Tensor};

use super::config::{PlanningExpertConfig, QwenDriveConfig};
use super::weights::QwenDriveExpertWeights;

/// Post-rotary scene K/V exported by the VLM, one entry per expert KV source
/// (8 for the 32-layer expert with `layers_per_kv = 4`). Each entry is the
/// `[seq, kv_heads, head_dim]` cache of one VLM full-attention layer.
pub struct SceneCache {
    pub keys: Vec<Tensor>,
    pub values: Vec<Tensor>,
}

/// Conditioning that the processor produces once per planning request.
pub struct ExpertConditioning {
    /// Normalized history poses re-referenced to the oldest pose, flattened
    /// `[num_history_points - 1, 3]` (the dropped origin row is implied).
    pub history: Vec<f32>,
    /// Raw history velocity `[num_history_points, 2]`, flattened.
    pub history_velocity: Vec<f32>,
    /// Raw history acceleration `[num_history_points, 2]`, flattened.
    pub history_acceleration: Vec<f32>,
    /// Navigation command index; out-of-range maps to an all-zero one-hot,
    /// exactly like the reference `_one_hot`.
    pub nav_command: i64,
    /// `[ego_status_dim]` velocity + acceleration + one-hot driving command.
    pub ego_status: Vec<f32>,
}

impl ExpertConditioning {
    pub fn validate(&self, config: &QwenDriveConfig) -> Result<()> {
        let poses = (config.num_history_points - 1) * config.trajectory_point_dim;
        if self.history.len() != poses {
            return Err(Error::Other(format!(
                "qwen_drive planner: history has {} values, expected {poses}",
                self.history.len()
            )));
        }
        let dynamics = config.num_history_points * config.expert.history_dynamics_dim;
        if self.history_velocity.len() != dynamics {
            return Err(Error::Other(format!(
                "qwen_drive planner: history_velocity has {} values, expected {dynamics}",
                self.history_velocity.len()
            )));
        }
        if self.history_acceleration.len() != dynamics {
            return Err(Error::Other(format!(
                "qwen_drive planner: history_acceleration has {} values, expected {dynamics}",
                self.history_acceleration.len()
            )));
        }
        if self.ego_status.len() != config.expert.ego_status_dim {
            return Err(Error::Other(format!(
                "qwen_drive planner: ego_status has {} values, expected {}",
                self.ego_status.len(),
                config.expert.ego_status_dim
            )));
        }
        Ok(())
    }
}

/// The planning expert with host-mirrored f32 copies of its bf16 weights.
///
/// Weight tensors are cloned to host f32 once at load: the scaffold executor
/// runs on the host, and the future device executor will upload the same
/// transposed tensors. `weights` is retained so both paths share one store.
pub struct PlanningExpertModel {
    config: PlanningExpertConfig,
    num_future_points: usize,
    trajectory_point_dim: usize,
    weights: QwenDriveExpertWeights,
}

/// bf16 rounding of one f32 value, returned as f32 (the value a bf16 tensor
/// would hold). Central so every semantic rounding site is greppable.
fn bf16_round(value: f32) -> f32 {
    half::bf16::from_f32(value).to_f32()
}

/// Sinusoidal time embedding (dim 128, scale 1000) computed in f32 with the
/// output rounded to bf16 (the reference casts the table to the module dtype
/// before the time MLP). decay = ln(10000) / (half - 1).
fn time_embedding(dim: usize, scale: f32, t: f32) -> Vec<f32> {
    let half = dim / 2;
    let decay = (10000.0f64).ln() as f32 / (half.max(2) - 1) as f32;
    let mut out = Vec::with_capacity(dim);
    for i in 0..half {
        let freq = (-(i as f32) * decay).exp();
        out.push(bf16_round((scale * t * freq).sin()));
    }
    for i in 0..half {
        let freq = (-(i as f32) * decay).exp();
        out.push(bf16_round((scale * t * freq).cos()));
    }
    out
}

impl PlanningExpertModel {
    pub fn new(config: &QwenDriveConfig, weights: QwenDriveExpertWeights) -> Result<Self> {
        let expert = config.expert.clone();
        expert.validate()?;
        let model = Self {
            config: expert,
            num_future_points: config.num_future_points,
            trajectory_point_dim: config.trajectory_point_dim,
            weights,
        };
        Ok(model)
    }

    pub fn num_kv_sources(&self) -> usize {
        self.config.num_kv_sources()
    }

    // ---- host scalar helpers (scaffold arithmetic) ---------------------

    fn tensor_f32(tensor: &Tensor, name: &str) -> Result<Vec<f32>> {
        tensor
            .to_f32_vec()
            .map_err(|e| Error::Other(format!("qwen_drive planner: {name} to f32: {e}")))
    }

    /// RMSNorm over the last axis, computed in f32 like the reference's
    /// `RMSNorm.forward` (x.float() ... .to(dtype)); the result is rounded to
    /// bf16 to mirror storing back into the bf16 activation stream.
    fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>> {
        let hidden = weight.len();
        if hidden == 0 || x.len() % hidden != 0 {
            return Err(Error::Other(format!(
                "qwen_drive planner: rms_norm got {} values for hidden {hidden}",
                x.len()
            )));
        }
        let mut out = vec![0.0f32; x.len()];
        for (row, chunk) in x.chunks(hidden).enumerate() {
            let mean_sq = chunk.iter().map(|v| v * v).sum::<f32>() / hidden as f32;
            let inv = 1.0f32 / (mean_sq + eps).sqrt();
            for (c, value) in chunk.iter().enumerate() {
                out[row * hidden + c] = bf16_round(value * inv * weight[c]);
            }
        }
        Ok(out)
    }

    /// y = x @ w + b over `[rows, in] * [in, out]`; f32 accumulation (the
    /// reference's cuBLAS bf16 GEMM also accumulates in f32) with the output
    /// rounded to bf16 on store.
    fn linear(x: &[f32], rows: usize, w: &[f32], b: Option<&[f32]>, in_dim: usize, out_dim: usize) -> Result<Vec<f32>> {
        if x.len() != rows * in_dim || w.len() != in_dim * out_dim {
            return Err(Error::Other(format!(
                "qwen_drive planner: linear shape mismatch (x {} vs {rows}x{in_dim}, w {} vs {in_dim}x{out_dim})",
                x.len(),
                w.len()
            )));
        }
        let mut out = vec![0.0f32; rows * out_dim];
        for r in 0..rows {
            let xr = &x[r * in_dim..(r + 1) * in_dim];
            for o in 0..out_dim {
                let mut acc = 0.0f32;
                for (i, value) in xr.iter().enumerate() {
                    acc += value * w[i * out_dim + o];
                }
                if let Some(bias) = b {
                    acc += bias[o];
                }
                out[r * out_dim + o] = bf16_round(acc);
            }
        }
        Ok(out)
    }

    fn silu(x: &[f32]) -> Vec<f32> {
        x.iter()
            .map(|v| bf16_round(*v / (1.0 + (-*v).exp())))
            .collect()
    }

    fn add(a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        if a.len() != b.len() {
            return Err(Error::Other("qwen_drive planner: add shape mismatch".into()));
        }
        Ok(a.iter().zip(b.iter()).map(|(x, y)| bf16_round(*x + *y)).collect())
    }

    /// `_mlp(in, hidden)`: Linear -> SiLU -> Linear.
    fn mlp(x: &[f32], rows: usize, mlp: &super::weights::ExpertMlp, in_dim: usize, hidden: usize) -> Result<Vec<f32>> {
        let fc1_w = Self::tensor_f32(&mlp.fc1_w, "mlp.fc1_w")?;
        let fc1_b = Self::tensor_f32(&mlp.fc1_b, "mlp.fc1_b")?;
        let fc2_w = Self::tensor_f32(&mlp.fc2_w, "mlp.fc2_w")?;
        let fc2_b = Self::tensor_f32(&mlp.fc2_b, "mlp.fc2_b")?;
        let h = Self::linear(x, rows, &fc1_w, Some(&fc1_b), in_dim, hidden)?;
        let h = Self::silu(&h);
        Self::linear(&h, rows, &fc2_w, Some(&fc2_b), hidden, hidden)
    }

    /// One-hot encode the nav command; out-of-range maps to all zeros,
    /// matching the reference `_one_hot`.
    fn one_hot(classes: usize, index: i64) -> Vec<f32> {
        let mut out = vec![0.0f32; classes];
        if index >= 0 && (index as usize) < classes {
            out[index as usize] = 1.0;
        }
        out
    }

    /// Sinusoidal time embedding, matching `SinusoidalTimeEmbedding.forward`.
    fn time_embedding(&self, t: f32) -> Vec<f32> {
        time_embedding(self.config.time_embed_dim, self.config.time_embed_scale, t)
    }

    /// Fourier features of the noisy waypoints: per-channel logspace
    /// frequencies built in the bf16 compute dtype (rounding is semantic),
    /// angles in f32, sin/cos concatenated, then the encoder MLP.
    fn fourier_features(&self, waypoints: &[f32], length: usize) -> Result<Vec<f32>> {
        let point = self.trajectory_point_dim;
        let features = self.config.fourier_num_features;
        let hidden = self.config.hidden_size;
        // torch.logspace(0, log10(max), steps=features, dtype=bf16).
        let log_max = (self.config.fourier_max_frequency as f64).log10() as f32;
        let mut freq = Vec::with_capacity(features);
        for i in 0..features {
            let exponent = if features > 1 {
                log_max * i as f32 / (features - 1) as f32
            } else {
                0.0
            };
            freq.push(bf16_round(10f32.powf(exponent)));
        }
        let two_pi = 2.0 * std::f32::consts::PI;
        let mut encoded = vec![0.0f32; length * point * features * 2];
        for l in 0..length {
            for c in 0..point {
                let waypoint = waypoints[l * point + c];
                for (f, frequency) in freq.iter().enumerate() {
                    let angle = waypoint * frequency * two_pi;
                    let base = ((l * point + c) * features + f) * 2;
                    encoded[base] = bf16_round(angle.sin());
                    encoded[base + 1] = bf16_round(angle.cos());
                }
            }
        }
        Self::mlp(
            &encoded,
            length,
            &self.weights.fourier_encoder,
            point * features * 2,
            hidden,
        )
    }

    /// Waypoint rope tables: positions `anchor + 1 ..= anchor + length` cast
    /// to bf16, multiplied with the bf16-rounded inverse-frequency table, and
    /// merged section-wise exactly like `WaypointRotaryEmbedding.forward`.
    /// Returns `(cos, sin)`, each `[length, rotary_dim]` in f32 (bf16-rounded).
    fn waypoint_rope_tables(&self, anchor: i64, length: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        let rotary_dim = self.config.rotary_dim();
        let pairs = rotary_dim / 2;
        let theta = self.config.rope_theta;
        // inv_freq = 1 / theta^(arange(0, dim, 2) / dim), rounded to bf16.
        let mut inv_freq = Vec::with_capacity(pairs);
        for i in 0..pairs {
            let exponent = (2 * i) as f32 / rotary_dim as f32;
            inv_freq.push(bf16_round(1.0 / theta.powf(exponent)));
        }
        let mut angles = vec![[0.0f32; 3].map(|_| vec![0.0f32; pairs]); length];
        let mut cos = vec![0.0f32; length * rotary_dim];
        let mut sin = vec![0.0f32; length * rotary_dim];
        for l in 0..length {
            // The position is the same scalar on all three mRoPE sections
            // (the anchor is a text token), rounded to bf16 as training did.
            let position = bf16_round((anchor + 1 + l as i64) as f32);
            for p in 0..pairs {
                let angle = bf16_round(position * inv_freq[p]);
                for section in &mut angles[l] {
                    section[p] = angle;
                }
            }
            // Section merge: pair p takes section s = p % 3 at indices
            // offset..length*3 step 3, exactly the reference slice semantics.
            let mut merged = angles[l][0].clone();
            let mut offset = 1usize;
            for (section_index, section_length) in self.config.mrope_section.iter().enumerate().skip(1) {
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

    /// Apply partial rotary to one `[length, heads, head_dim]` tensor: rotate
    /// the leading `rotary_dim` channels with the bf16-rounded cos/sin table,
    /// leave the remainder untouched (`_apply_rotary` in the reference).
    fn apply_rotary(&self, x: &[f32], length: usize, heads: usize, cos: &[f32], sin: &[f32]) -> Result<Vec<f32>> {
        let head_dim = self.config.head_dim;
        let rotary_dim = self.config.rotary_dim();
        if x.len() != length * heads * head_dim {
            return Err(Error::Other("qwen_drive planner: rotary input shape mismatch".into()));
        }
        let mut out = x.to_vec();
        let half = rotary_dim / 2;
        for l in 0..length {
            for h in 0..heads {
                let base = (l * heads + h) * head_dim;
                for p in 0..half {
                    let first = x[base + p];
                    let second = x[base + half + p];
                    let c = cos[l * rotary_dim + p];
                    let s = sin[l * rotary_dim + p];
                    out[base + p] = bf16_round(first * c - second * s);
                    out[base + half + p] = bf16_round(second * c + first * s);
                }
            }
        }
        Ok(out)
    }

    /// `encode_history`: history poses + nav one-hot, velocity, acceleration.
    fn encode_history(&self, cond: &ExpertConditioning) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let hidden = self.config.hidden_size;
        let nav = Self::one_hot(self.config.nav_command_classes, cond.nav_command);
        let mut history_in = cond.history.clone();
        history_in.extend_from_slice(&nav);
        let pose = Self::mlp(
            &history_in,
            1,
            &self.weights.history_encoder,
            history_in.len(),
            hidden,
        )?;
        let velocity = Self::mlp(
            &cond.history_velocity,
            1,
            &self.weights.history_velocity_encoder,
            cond.history_velocity.len(),
            hidden,
        )?;
        let acceleration = Self::mlp(
            &cond.history_acceleration,
            1,
            &self.weights.history_acceleration_encoder,
            cond.history_acceleration.len(),
            hidden,
        )?;
        Ok((pose, velocity, acceleration))
    }

    /// Split one fused QKV row into (query, gate, key, value), following the
    /// reference `_split_qkv` group-major layout. Input is the transposed
    /// `[length, qkv_out]` projection output.
    fn split_qkv(&self, fused: &[f32], length: usize) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
        let heads = self.config.n_heads;
        let groups = self.config.n_kv_heads;
        let head_dim = self.config.head_dim;
        let per_group = heads / groups;
        let group_width = (per_group * 2 + 2) * head_dim;
        if fused.len() != length * groups * group_width {
            return Err(Error::Other("qwen_drive planner: qkv width mismatch".into()));
        }
        let mut query = vec![0.0f32; length * heads * head_dim];
        let mut gate = vec![0.0f32; length * heads * head_dim];
        let mut key = vec![0.0f32; length * groups * head_dim];
        let mut value = vec![0.0f32; length * groups * head_dim];
        for l in 0..length {
            for g in 0..groups {
                let src = (l * groups + g) * group_width;
                for h in 0..per_group {
                    let head = g * per_group + h;
                    // Within a group: query heads, then their gate heads, then K, V.
                    let q_src = src + h * head_dim;
                    let g_src = src + (per_group + h) * head_dim;
                    query[(l * heads + head) * head_dim..(l * heads + head + 1) * head_dim]
                        .copy_from_slice(&fused[q_src..q_src + head_dim]);
                    gate[(l * heads + head) * head_dim..(l * heads + head + 1) * head_dim]
                        .copy_from_slice(&fused[g_src..g_src + head_dim]);
                }
                let k_src = src + per_group * 2 * head_dim;
                let v_src = k_src + head_dim;
                key[(l * groups + g) * head_dim..(l * groups + g + 1) * head_dim]
                    .copy_from_slice(&fused[k_src..k_src + head_dim]);
                value[(l * groups + g) * head_dim..(l * groups + g + 1) * head_dim]
                    .copy_from_slice(&fused[v_src..v_src + head_dim]);
            }
        }
        Ok((query, gate, key, value))
    }

    /// Non-causal GQA joint attention over `concat(scene_kv, waypoint_kv)`.
    /// fp32 softmax (as SDPA), bf16-rounded output. Shapes: q `[length,
    /// heads, head_dim]`; scene k/v `[scene_len, kv_heads, head_dim]`; self
    /// k/v `[length, kv_heads, head_dim]`. Returns `[length, heads*head_dim]`.
    fn joint_attention(
        &self,
        query: &[f32],
        scene_k: &[f32],
        scene_v: &[f32],
        self_k: &[f32],
        self_v: &[f32],
        length: usize,
        scene_len: usize,
    ) -> Result<Vec<f32>> {
        let heads = self.config.n_heads;
        let groups = self.config.n_kv_heads;
        let head_dim = self.config.head_dim;
        let per_group = heads / groups;
        let kv_len = scene_len + length;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut out = vec![0.0f32; length * heads * head_dim];
        let mut scores = vec![0.0f32; kv_len];
        for l in 0..length {
            for h in 0..heads {
                let kv_head = h / per_group;
                let q_base = (l * heads + h) * head_dim;
                for s in 0..kv_len {
                    let k_base = if s < scene_len {
                        (s * groups + kv_head) * head_dim
                    } else {
                        ((s - scene_len) * groups + kv_head) * head_dim
                    };
                    let k_src = if s < scene_len { scene_k } else { self_k };
                    let mut acc = 0.0f32;
                    for d in 0..head_dim {
                        acc += query[q_base + d] * k_src[k_base + d];
                    }
                    scores[s] = acc * scale;
                }
                // fp32 softmax.
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in scores.iter_mut() {
                    *s = (*s - max).exp();
                    sum += *s;
                }
                if sum > 0.0 {
                    for s in scores.iter_mut() {
                        *s /= sum;
                    }
                }
                let o_base = (l * heads + h) * head_dim;
                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for (s, score) in scores.iter().enumerate() {
                        let v_base = if s < scene_len {
                            (s * groups + kv_head) * head_dim
                        } else {
                            ((s - scene_len) * groups + kv_head) * head_dim
                        };
                        let v_src = if s < scene_len { scene_v } else { self_v };
                        acc += score * v_src[v_base + d];
                    }
                    out[o_base + d] = bf16_round(acc);
                }
            }
        }
        Ok(out)
    }

    /// One diffusion-transformer layer (`PlanningExpertLayer.forward`).
    #[allow(clippy::too_many_arguments)]
    fn layer_forward(
        &self,
        layer: &super::weights::QwenDriveExpertLayer,
        hidden_states: &[f32],
        scene_key: &[f32],
        scene_value: &[f32],
        scene_len: usize,
        cos: &[f32],
        sin: &[f32],
        condition: &[f32],
        length: usize,
    ) -> Result<Vec<f32>> {
        let hidden = self.config.hidden_size;
        let head_dim = self.config.head_dim;
        let heads = self.config.n_heads;
        let eps = self.config.rms_norm_eps;

        // adaLN: SiLU then Linear, split into 6 chunks of `hidden`.
        let mod_w = Self::tensor_f32(&layer.modulation_w, "adaln_modulation.weight")?;
        let mod_b = Self::tensor_f32(&layer.modulation_b, "adaln_modulation.bias")?;
        let modulation = Self::linear(&Self::silu(condition), 1, &mod_w, Some(&mod_b), hidden, 6 * hidden)?;
        let shift_attn = &modulation[0..hidden];
        let scale_attn = &modulation[hidden..2 * hidden];
        let gate_attn = &modulation[2 * hidden..3 * hidden];
        let shift_ffn = &modulation[3 * hidden..4 * hidden];
        let scale_ffn = &modulation[4 * hidden..5 * hidden];
        let gate_ffn = &modulation[5 * hidden..6 * hidden];

        // Attention branch: x = rms(h) * (1 + scale) + shift.
        let norm_w = Self::tensor_f32(&layer.input_layernorm, "input_layernorm")?;
        let normed = Self::rms_norm(hidden_states, &norm_w, eps)?;
        let mut modulated = vec![0.0f32; length * hidden];
        for l in 0..length {
            for c in 0..hidden {
                let idx = l * hidden + c;
                modulated[idx] = bf16_round(normed[idx] * (1.0 + scale_attn[c]) + shift_attn[c]);
            }
        }
        let qkv_w = Self::tensor_f32(&layer.qkv_w, "qkv_proj")?;
        let qkv = Self::linear(&modulated, length, &qkv_w, None, hidden, qkv_w.len() / hidden)?;
        let (query, gate, key, value) = self.split_qkv(&qkv, length)?;

        let q_norm = Self::tensor_f32(&layer.q_norm, "q_norm")?;
        let k_norm = Self::tensor_f32(&layer.k_norm, "k_norm")?;
        // Per-head RMSNorm over head_dim rows.
        let mut query_heads = vec![0.0f32; length * heads * head_dim];
        for h in 0..heads {
            let mut head = vec![0.0f32; length * head_dim];
            for l in 0..length {
                let base = (l * heads + h) * head_dim;
                head[l * head_dim..(l + 1) * head_dim].copy_from_slice(&query[base..base + head_dim]);
            }
            let normed_head = Self::rms_norm(&head, &q_norm, eps)?;
            for l in 0..length {
                let base = (l * heads + h) * head_dim;
                query_heads[base..base + head_dim]
                    .copy_from_slice(&normed_head[l * head_dim..(l + 1) * head_dim]);
            }
        }
        let groups = self.config.n_kv_heads;
        let mut key_heads = vec![0.0f32; length * groups * head_dim];
        for h in 0..groups {
            let mut head = vec![0.0f32; length * head_dim];
            for l in 0..length {
                let base = (l * groups + h) * head_dim;
                head[l * head_dim..(l + 1) * head_dim].copy_from_slice(&key[base..base + head_dim]);
            }
            let normed_head = Self::rms_norm(&head, &k_norm, eps)?;
            for l in 0..length {
                let base = (l * groups + h) * head_dim;
                key_heads[base..base + head_dim]
                    .copy_from_slice(&normed_head[l * head_dim..(l + 1) * head_dim]);
            }
        }

        let query_rot = self.apply_rotary(&query_heads, length, heads, cos, sin)?;
        let key_rot = self.apply_rotary(&key_heads, length, groups, cos, sin)?;

        let attn = self.joint_attention(&query_rot, scene_key, scene_value, &key_rot, &value, length, scene_len)?;
        // Output gate: attn * sigmoid(gate).
        let o_w = Self::tensor_f32(&layer.o_w, "o_proj")?;
        let mut gated = vec![0.0f32; length * heads * head_dim];
        for i in 0..gated.len() {
            let sigmoid = 1.0 / (1.0 + (-gate[i]).exp());
            gated[i] = bf16_round(attn[i] * sigmoid);
        }
        let projected = Self::linear(&gated, length, &o_w, None, heads * head_dim, hidden)?;
        // residual + (1 + gate_attn) * proj
        let mut h = vec![0.0f32; length * hidden];
        for l in 0..length {
            for c in 0..hidden {
                let idx = l * hidden + c;
                h[idx] = bf16_round(hidden_states[idx] + (1.0 + gate_attn[c]) * projected[idx]);
            }
        }

        // FFN branch: x = rms(h) * (1 + scale_ffn) + shift_ffn; SwiGLU.
        let ffn_norm_w = Self::tensor_f32(&layer.post_attention_layernorm, "post_attention_layernorm")?;
        let normed = Self::rms_norm(&h, &ffn_norm_w, eps)?;
        for l in 0..length {
            for c in 0..hidden {
                let idx = l * hidden + c;
                modulated[idx] = bf16_round(normed[idx] * (1.0 + scale_ffn[c]) + shift_ffn[c]);
            }
        }
        let gate_up_w = Self::tensor_f32(&layer.gate_up_w, "gate_up_proj")?;
        let inter = self.config.intermediate_size;
        let gate_up = Self::linear(&modulated, length, &gate_up_w, None, hidden, 2 * inter)?;
        let mut swiglu = vec![0.0f32; length * inter];
        for l in 0..length {
            for c in 0..inter {
                let g = gate_up[l * 2 * inter + c];
                let u = gate_up[l * 2 * inter + inter + c];
                let silu_g = g / (1.0 + (-g).exp());
                swiglu[l * inter + c] = bf16_round(silu_g * u);
            }
        }
        let down_w = Self::tensor_f32(&layer.down_w, "down_proj")?;
        let ffn = Self::linear(&swiglu, length, &down_w, None, inter, hidden)?;
        let mut out = vec![0.0f32; length * hidden];
        for l in 0..length {
            for c in 0..hidden {
                let idx = l * hidden + c;
                out[idx] = bf16_round(h[idx] + (1.0 + gate_ffn[c]) * ffn[idx]);
            }
        }
        Ok(out)
    }

    /// `predict_endpoint`: the clean-trajectory prediction for one flow time.
    /// `waypoints` is `[length, 3]` f32 (unrounded solver state, cast to bf16
    /// on entry like the reference's `.to(dtype)`).
    pub fn predict_endpoint(
        &self,
        waypoints: &[f32],
        flow_time: f32,
        cond: &ExpertConditioning,
        history_queries: &(Vec<f32>, Vec<f32>, Vec<f32>),
        scene: &SceneCache,
        anchor: i64,
    ) -> Result<Vec<f32>> {
        let hidden = self.config.hidden_size;
        let length = self.num_future_points;
        if waypoints.len() != length * self.trajectory_point_dim {
            return Err(Error::Other(format!(
                "qwen_drive planner: waypoints has {} values, expected {}",
                waypoints.len(),
                length * self.trajectory_point_dim
            )));
        }
        if scene.keys.len() != self.num_kv_sources() || scene.values.len() != self.num_kv_sources() {
            return Err(Error::Other(format!(
                "qwen_drive planner: scene cache has {}/{} entries, expected {}",
                scene.keys.len(),
                scene.values.len(),
                self.num_kv_sources()
            )));
        }

        // bf16 cast on entry (reference: `waypoints.to(dtype)`).
        let waypoints_bf16: Vec<f32> = waypoints.iter().map(|v| bf16_round(*v)).collect();

        // The seven fused signals.
        let traj_w = Self::tensor_f32(&self.weights.trajectory_proj_w, "trajectory_proj.weight")?;
        let traj_b = Self::tensor_f32(&self.weights.trajectory_proj_b, "trajectory_proj.bias")?;
        let traj = Self::linear(&waypoints_bf16, length, &traj_w, Some(&traj_b), self.trajectory_point_dim, hidden)?;
        let fourier = self.fourier_features(&waypoints_bf16, length)?;
        let time_condition = Self::mlp(
            &self.time_embedding(flow_time),
            1,
            &self.weights.time_mlp,
            self.config.time_embed_dim,
            hidden,
        )?;
        let (pose_query, velocity_query, acceleration_query) = history_queries;
        let embed_table = Self::tensor_f32(&self.weights.waypoint_embed, "waypoint_embed")?;

        let mut fused = vec![0.0f32; length * 7 * hidden];
        for l in 0..length {
            let row = &mut fused[l * 7 * hidden..(l + 1) * 7 * hidden];
            row[0..hidden].copy_from_slice(&traj[l * hidden..(l + 1) * hidden]);
            row[hidden..2 * hidden].copy_from_slice(&fourier[l * hidden..(l + 1) * hidden]);
            row[2 * hidden..3 * hidden].copy_from_slice(&time_condition);
            row[3 * hidden..4 * hidden].copy_from_slice(pose_query);
            row[4 * hidden..5 * hidden]
                .copy_from_slice(&embed_table[l * hidden..(l + 1) * hidden]);
            row[5 * hidden..6 * hidden].copy_from_slice(velocity_query);
            row[6 * hidden..7 * hidden].copy_from_slice(acceleration_query);
        }
        let mut hidden_states = Self::mlp(&fused, length, &self.weights.query_fusion, 7 * hidden, hidden)?;

        // adaLN condition = time + nav + ego.
        let nav = Self::one_hot(self.config.nav_command_classes, cond.nav_command);
        let nav_out = Self::mlp(&nav, 1, &self.weights.nav_mlp, self.config.nav_command_classes, hidden)?;
        let ego_out = Self::mlp(&cond.ego_status, 1, &self.weights.ego_mlp, self.config.ego_status_dim, hidden)?;
        let mut condition = Self::add(&time_condition, &nav_out)?;
        condition = Self::add(&condition, &ego_out)?;

        let (cos, sin) = self.waypoint_rope_tables(anchor, length)?;
        let layers_per_kv = self.config.layers_per_kv;
        for (index, layer) in self.weights.layers.iter().enumerate() {
            let scene_index = index / layers_per_kv;
            let scene_k = Self::tensor_f32(&scene.keys[scene_index], "scene key")?;
            let scene_v = Self::tensor_f32(&scene.values[scene_index], "scene value")?;
            let scene_len = scene_k.len() / (self.config.n_kv_heads * self.config.head_dim);
            hidden_states = self.layer_forward(
                layer,
                &hidden_states,
                &scene_k,
                &scene_v,
                scene_len,
                &cos,
                &sin,
                &condition,
                length,
            )?;
        }

        let final_norm = Self::tensor_f32(&self.weights.final_layernorm, "final_layernorm")?;
        let normed = Self::rms_norm(&hidden_states, &final_norm, self.config.rms_norm_eps)?;
        let out_w = Self::tensor_f32(&self.weights.out_proj_w, "out_proj.weight")?;
        let out_b = Self::tensor_f32(&self.weights.out_proj_b, "out_proj.bias")?;
        // `.float()` output: the reference returns fp32 endpoint predictions.
        Self::linear(&normed, length, &out_w, Some(&out_b), hidden, self.trajectory_point_dim)
    }

    /// `sample`: flow matching with clean-endpoint parameterization — 10
    /// Euler steps, `x += (x1_hat - x) / max(1 - t, min_one_minus_t) * step`,
    /// solver state in fp32 exactly like the reference.
    pub fn sample(
        &self,
        scene: &SceneCache,
        anchor: i64,
        cond: &ExpertConditioning,
        noise: &[f32],
        num_steps: usize,
        min_one_minus_t: f32,
    ) -> Result<Vec<f32>> {
        let length = self.num_future_points;
        if noise.len() != length * self.trajectory_point_dim {
            return Err(Error::Other(format!(
                "qwen_drive planner: noise has {} values, expected {}",
                noise.len(),
                length * self.trajectory_point_dim
            )));
        }
        if num_steps == 0 {
            return Err(Error::Other("qwen_drive planner: num_steps must be positive".into()));
        }
        let history_queries = self.encode_history(cond)?;
        let mut waypoints: Vec<f32> = noise.to_vec();
        let step = 1.0f32 / num_steps as f32;
        for index in 0..num_steps {
            let t = index as f32 * step;
            let endpoint = self.predict_endpoint(&waypoints, t, cond, &history_queries, scene, anchor)?;
            let remaining = (1.0f32 - t).max(min_one_minus_t);
            for (w, e) in waypoints.iter_mut().zip(endpoint.iter()) {
                *w += (*e - *w) / remaining * step;
            }
        }
        Ok(waypoints)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_rounding_is_semantic() {
        // pi/2 rounds to 1.5703125 in bf16 (the trajectory heading scale).
        assert_eq!(bf16_round(std::f32::consts::FRAC_PI_2), 1.5703125);
        // Integers above 256 round to the bf16 grid (rope position rounding).
        assert_eq!(bf16_round(300.0), 300.0); // 300 is representable
        assert_ne!(bf16_round(257.0), 257.0);
    }

    #[test]
    fn one_hot_maps_out_of_range_to_zero() {
        assert_eq!(PlanningExpertModel::one_hot(3, 1), vec![0.0, 1.0, 0.0]);
        assert_eq!(PlanningExpertModel::one_hot(3, 7), vec![0.0, 0.0, 0.0]);
        assert_eq!(PlanningExpertModel::one_hot(3, -1), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn sinusoidal_time_embedding_matches_reference_formula() {
        let emb = time_embedding(8, 1000.0, 0.5);
        assert_eq!(emb.len(), 8);
        // First channel: sin(1000 * 0.5 * exp(0)) = sin(500) rounded to bf16.
        let expected = bf16_round((500.0f32).sin());
        assert_eq!(emb[0], expected);
        // Decay: freqs[1] = exp(-ln(10000) / 3).
        let freq1 = (-(10000.0f64).ln() as f32 / 3.0).exp();
        let expected1 = bf16_round((1000.0 * 0.5 * freq1).sin());
        assert_eq!(emb[1], expected1);
    }

    /// Zero-initialized expert weights with the shapes the executor expects;
    /// only used where the test never reads them.
    #[allow(dead_code)]
    fn dummy_weights() -> super::super::weights::QwenDriveExpertWeights {
        use super::super::weights::{ExpertMlp, QwenDriveExpertLayer, QwenDriveExpertWeights};
        let z = |rows: usize, cols: usize| Tensor::from_f32(vec![rows, cols], &vec![0.0; rows * cols]).unwrap();
        let mlp = |input: usize, hidden: usize| ExpertMlp {
            fc1_w: z(input, hidden),
            fc1_b: z(hidden, 1),
            fc2_w: z(hidden, hidden),
            fc2_b: z(hidden, 1),
        };
        let hidden = 8;
        QwenDriveExpertWeights {
            trajectory_proj_w: z(3, hidden),
            trajectory_proj_b: z(hidden, 1),
            fourier_encoder: mlp(24, hidden),
            waypoint_embed: z(4, hidden),
            time_mlp: mlp(8, hidden),
            nav_mlp: mlp(3, hidden),
            ego_mlp: mlp(8, hidden),
            history_encoder: mlp(12, hidden),
            history_velocity_encoder: mlp(8, hidden),
            history_acceleration_encoder: mlp(8, hidden),
            query_fusion: mlp(56, hidden),
            layers: (0..4)
                .map(|_| QwenDriveExpertLayer {
                    input_layernorm: z(hidden, 1),
                    qkv_w: z(hidden, hidden),
                    q_norm: z(8, 1),
                    k_norm: z(8, 1),
                    o_w: z(16, hidden),
                    post_attention_layernorm: z(hidden, 1),
                    gate_up_w: z(hidden, 16),
                    down_w: z(8, hidden),
                    modulation_w: z(hidden, 6 * hidden),
                    modulation_b: z(6 * hidden, 1),
                })
                .collect(),
            final_layernorm: z(hidden, 1),
            out_proj_w: z(hidden, 3),
            out_proj_b: z(3, 1),
        }
    }
}
