//! Planning-expert device executor: joint-attention diffusion transformer
//! with flow-matching sampling, on the CUDA kernel path.
//!
//! Ported from the pinned reference (`planning_expert.py`) and the saved K3
//! CPU scaffold (`planner.rs`, retained as the correctness oracle): adaLN
//! conditioning (rms -> *(1+scale) + shift, residual + (1+gate) * proj),
//! group-major fused QKV with plain per-head RMSNorm, partial interleaved
//! mRoPE with the reference's bf16-rounded frequency/position tables,
//! non-causal joint GQA over concat(scene KV, waypoint KV) with a per-head
//! sigmoid output gate, SwiGLU FFN, and clean-endpoint flow matching with an
//! fp32 solver state. All steady-state computation runs on device; host work
//! is limited to per-request table construction and the final output copy.

use apxinf_core::{DType, Error, Result, Shape, Tensor};

use super::backend::{kernels, transfers, Context, DeviceBuffer};
use kernels::{activation, elementwise, embedding, gemm, linear_attention as la, norm};

use super::config::{PlanningExpertConfig, QwenDriveConfig};
use super::device_weights::{DeviceMlp, ExpertDeviceWeights};
use super::planner::ExpertConditioning;

/// bf16 rounding of one f32 value, returned as f32 (the value a bf16 tensor
/// would hold). Mirrors the scaffold's semantic-rounding helper.
fn bf16_round(value: f32) -> f32 {
    half::bf16::from_f32(value).to_f32()
}

fn upload_u32(ctx: &Context, values: &[u32]) -> Result<DeviceBuffer> {
    let bytes: Vec<u8> = values.iter().flat_map(|value| value.to_ne_bytes()).collect();
    let buffer = DeviceBuffer::alloc(bytes.len().max(1), ctx.device_id()).map_err(Error::Cuda)?;
    buffer.copy_from_host(&bytes).map_err(Error::Cuda)?;
    Ok(buffer)
}

fn device_tensor(ctx: &Context, shape: &[usize], dtype: DType) -> Result<Tensor> {
    let elements: usize = shape.iter().product();
    let bytes = elements
        .checked_mul(dtype.size_in_bytes())
        .ok_or_else(|| Error::Other("qwen_drive expert: tensor size overflow".into()))?;
    let buffer = DeviceBuffer::alloc(bytes.max(1), ctx.device_id()).map_err(Error::Cuda)?;
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
        .view(col * DType::BF16.size_in_bytes(), cols * DType::BF16.size_in_bytes())
        .map_err(Error::Cuda)?;
    view.as_tensor(Shape::new(vec![cols]), DType::BF16)
        .map_err(Error::Cuda)
}

fn mlp(ctx: &Context, mlp: &DeviceMlp, x: &Tensor) -> Result<Tensor> {
    let h = gemm::matmul(ctx, x, &mlp.fc1_w)?;
    let h = activation::bias_silu_bf16(ctx, &h, Some(&mlp.fc1_b))?;
    let h = gemm::matmul(ctx, &h, &mlp.fc2_w)?;
    elementwise::bias_bf16(ctx, &h, Some(&mlp.fc2_b))
}

fn one_hot(classes: usize, index: i64) -> Vec<f32> {
    let mut out = vec![0.0f32; classes];
    if index >= 0 && (index as usize) < classes {
        out[index as usize] = 1.0;
    }
    out
}

/// Sinusoidal time embedding (scaffold semantics): decay = ln(10000)/(half-1),
/// angles = (scale * t) * freq in f32, sin/cos rounded to bf16.
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
    pub noise: &'a [f32],
    pub num_steps: usize,
}

/// Run the flow-matching sampler; returns the normalized `[length, 3]`
/// trajectory in fp32 (the caller denormalizes).
pub fn plan(
    config: &QwenDriveConfig,
    weights: &ExpertDeviceWeights,
    ctx: &Context,
    input: &ExpertPlan,
) -> Result<Vec<f32>> {
    let ec = &config.expert;
    let hidden_size = ec.hidden_size;
    let length = config.num_future_points;
    let point_dim = config.trajectory_point_dim;
    let heads = ec.n_heads;
    let kv_heads = ec.n_kv_heads;
    let head_dim = ec.head_dim;
    let rotary_dim = ec.rotary_dim();
    let eps = ec.rms_norm_eps;
    let steps = input.num_steps.max(1);
    if input.scene.len() != ec.num_kv_sources() {
        return Err(Error::Other(format!(
            "qwen_drive expert: {} scene caches, expected {}",
            input.scene.len(),
            ec.num_kv_sources()
        )));
    }
    if input.noise.len() != length * point_dim {
        return Err(Error::Other(format!(
            "qwen_drive expert: noise has {} values, expected {}",
            input.noise.len(),
            length * point_dim
        )));
    }
    input.cond.validate(config)?;

    // Conditioning (request lifetime).
    let nav = one_hot(ec.nav_command_classes, input.cond.nav_command);
    let mut history_in = input.cond.history.clone();
    history_in.extend_from_slice(&nav);
    let history_t = bf16_tensor(ctx, &history_in, vec![1, history_in.len()])?;
    let velocity_t = bf16_tensor(ctx, &input.cond.history_velocity, vec![1, input.cond.history_velocity.len()])?;
    let acceleration_t = bf16_tensor(ctx, &input.cond.history_acceleration, vec![1, input.cond.history_acceleration.len()])?;
    let ego_t = bf16_tensor(ctx, &input.cond.ego_status, vec![1, input.cond.ego_status.len()])?;
    let nav_t = bf16_tensor(ctx, &nav, vec![1, nav.len()])?;
    let pose_q = mlp(ctx, &weights.history_encoder, &history_t)?;
    let velocity_q = mlp(ctx, &weights.history_velocity_encoder, &velocity_t)?;
    let acceleration_q = mlp(ctx, &weights.history_acceleration_encoder, &acceleration_t)?;
    let nav_out = mlp(ctx, &weights.nav_mlp, &nav_t)?;
    let ego_out = mlp(ctx, &weights.ego_mlp, &ego_t)?;

    let ids: Vec<u32> = (0..length as u32).collect();
    let ids_dev = upload_u32(ctx, &ids)?;
    let wp_idx = embedding::lookup_bf16(ctx, &weights.waypoint_embed, &ids_dev, length)?;

    let (cos, sin) = waypoint_rope_tables(ec, input.anchor, length)?;
    let cos_t = bf16_tensor(ctx, &cos, vec![length, rotary_dim])?;
    let sin_t = bf16_tensor(ctx, &sin, vec![length, rotary_dim])?;

    // fp32 solver state on device.
    let noise_tensor = Tensor::from_f32(vec![length, point_dim], input.noise)?;
    let waypoints = transfers::to_cuda(&noise_tensor, ctx.device_id())?;

    let step_f64 = 1.0f64 / steps as f64;
    for index in 0..steps {
        let t_f64 = index as f64 * step_f64;
        let t_emb = time_embedding(ec.time_embed_dim, ec.time_embed_scale, t_f64 as f32);
        let t_emb_t = bf16_tensor(ctx, &t_emb, vec![1, ec.time_embed_dim])?;
        let time_condition = mlp(ctx, &weights.time_mlp, &t_emb_t)?;
        let condition = elementwise::add(ctx, &time_condition, &nav_out)?;
        let condition = elementwise::add(ctx, &condition, &ego_out)?;
        let condition_silu = activation::silu(ctx, &condition)?;

        let wp_bf16 = device_tensor(ctx, &[length, point_dim], DType::BF16)?;
        la::cast_f32_to_bf16(ctx, &waypoints, &wp_bf16)?;
        let traj = gemm::matmul(ctx, &wp_bf16, &weights.trajectory_proj_w)?;
        let traj = elementwise::bias_bf16(ctx, &traj, Some(&weights.trajectory_proj_b))?;
        let fourier_feat = la::fourier_features(
            ctx,
            &wp_bf16,
            &weights.fourier_freqs,
            point_dim,
            ec.fourier_num_features,
        )?;
        let fourier = mlp(ctx, &weights.fourier, &fourier_feat)?;
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

        for (layer_index, layer) in weights.layers.iter().enumerate() {
            let (scene_k, scene_v) = &input.scene[layer_index / ec.layers_per_kv];
            let modulation = gemm::matmul(ctx, &condition_silu, &layer.modulation_w)?;
            let modulation = elementwise::bias_bf16(ctx, &modulation, Some(&layer.modulation_b))?;
            let shift_attn = row_col_slice(&modulation, 0, hidden_size)?;
            let scale_attn = row_col_slice(&modulation, hidden_size, hidden_size)?;
            let gate_attn = row_col_slice(&modulation, 2 * hidden_size, hidden_size)?;
            let shift_ffn = row_col_slice(&modulation, 3 * hidden_size, hidden_size)?;
            let scale_ffn = row_col_slice(&modulation, 4 * hidden_size, hidden_size)?;
            let gate_ffn = row_col_slice(&modulation, 5 * hidden_size, hidden_size)?;

            let x = la::adaln_rms_norm(ctx, &hidden, &layer.input_norm, &scale_attn, &shift_attn, eps)?;
            let qkv = gemm::matmul(ctx, &x, &layer.qkv_w)?;
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
            la::expert_sigmoid_gate_mul(ctx, &attn, &gate_out)?;
            let attn = attn.reshape(vec![length, heads * head_dim])?;
            let proj = gemm::matmul(ctx, &attn, &layer.o_w)?;
            hidden = la::adaln_gate_residual(ctx, &proj, &hidden, &gate_attn)?;

            let x2 = la::adaln_rms_norm(ctx, &hidden, &layer.post_norm, &scale_ffn, &shift_ffn, eps)?;
            let gu = gemm::matmul(ctx, &x2, &layer.gate_up_w)?;
            let act = activation::swiglu_bf16(ctx, &gu)?;
            let down = gemm::matmul(ctx, &act, &layer.down_w)?;
            hidden = la::adaln_gate_residual(ctx, &down, &hidden, &gate_ffn)?;
        }

        let final_normed = norm::rms_bf16(ctx, &hidden, &weights.final_norm, eps)?;
        let endpoint = gemm::matmul(ctx, &final_normed, &weights.out_proj_w)?;
        let endpoint = elementwise::bias_bf16(ctx, &endpoint, Some(&weights.out_proj_b))?;
        let endpoint_f32 = device_tensor(ctx, &[length, point_dim], DType::F32)?;
        la::cast_bf16_to_f32(ctx, &endpoint, &endpoint_f32)?;
        let remaining = (1.0f64 - t_f64).max(config.min_one_minus_t as f64) as f32;
        la::flow_update(ctx, &waypoints, &endpoint_f32, remaining, step_f64 as f32)?;
    }

    let cpu = transfers::to_cpu(&waypoints)?;
    cpu.to_f32_vec()
        .map_err(|e| Error::Other(format!("qwen_drive expert: output download: {e}")))
}
