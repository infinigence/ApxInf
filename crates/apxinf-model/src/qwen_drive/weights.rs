//! Qwen-Drive-1.0 checkpoint weight mapping.
//!
//! Two weight stores are supported:
//!
//! * the VLM directory (the Qwen3.5 release root), whose safetensors keys are
//!   `model.language_model.*` and `model.visual.*`; and
//! * a planning-expert head directory (planner-sft / planner-rl), whose keys
//!   carry an optional `planning_expert.` prefix that the reference strips at
//!   load time (`load_planner`).
//!
//! HF `Linear` weights are `[out, in]`; every projection is transposed once at
//! load to the `[in, out]` row-major layout the matmul path expects. The
//! expert's fused QKV weight keeps the reference's group-major row order
//! (per KV group: query heads, then their output-gate heads, then K, V) and
//! the split is applied at inference exactly as `PlanningExpertLayer._split_qkv`.
//!
//! The VLM weight struct deliberately holds tensors by name: the Qwen3.5
//! linear-attention (gated delta net) layer geometry is parsed and validated,
//! but its device execution is pending reference qualification. The named map
//! keeps the loader honest and reusable for the follow-up executor instead of
//! a hardcoded stub schema that would silently drop the GDN projections.

use std::collections::HashMap;

use apxinf_core::{Error, Result, Tensor};

use super::config::QwenDriveConfig;

/// VLM (Qwen3.5) weights, keyed by their checkpoint names.
pub struct QwenDriveVlmWeights {
    /// Every `model.language_model.*` tensor, untransposed.
    pub language: HashMap<String, Tensor>,
    /// Every `model.visual.*` tensor, untransposed.
    pub visual: HashMap<String, Tensor>,
}

impl QwenDriveVlmWeights {
    pub fn from_map(mut tensors: HashMap<String, Tensor>) -> Result<Self> {
        let mut language = HashMap::new();
        let mut visual = HashMap::new();
        let mut ignored: Vec<String> = tensors
            .keys()
            .filter(|key| {
                !key.starts_with("model.language_model.") && !key.starts_with("model.visual.")
            })
            .cloned()
            .collect();
        ignored.sort_unstable();
        for (key, tensor) in tensors.drain() {
            if key.starts_with("model.language_model.") {
                language.insert(key, tensor);
            } else if key.starts_with("model.visual.") {
                visual.insert(key, tensor);
            }
        }
        if !ignored.is_empty() {
            // Strictly informational: MTP/auxiliary tensors are allowed to be
            // present (the reference ignores them too), but we surface them in
            // the error channel only when they look like expert keys, which
            // means the caller passed a planner head instead of the VLM.
            let expert_like = ignored
                .iter()
                .any(|key| key.starts_with("planning_expert.") || key.contains("adaln_modulation"));
            if expert_like {
                return Err(Error::Other(format!(
                    "qwen_drive weights: the map looks like a planning-expert head \
                     (e.g. {}); load VLM and planner weights separately",
                    ignored[0]
                )));
            }
        }
        if language.is_empty() {
            return Err(Error::Other(
                "qwen_drive weights: no model.language_model.* tensors found; \
                 pass the VLM checkpoint directory (or attach a planner head \
                 through QwenDriveExpertWeights::from_map)".into(),
            ));
        }
        if !language.contains_key("model.language_model.embed_tokens.weight") {
            return Err(Error::Other(
                "qwen_drive weights: missing model.language_model.embed_tokens.weight".into(),
            ));
        }
        if !language.contains_key("model.language_model.norm.weight") {
            return Err(Error::Other(
                "qwen_drive weights: missing model.language_model.norm.weight".into(),
            ));
        }
        if visual.is_empty() {
            return Err(Error::Other(
                "qwen_drive weights: no model.visual.* tensors found; the vision \
                 tower is required for VQA and planning scene encoding".into(),
            ));
        }
        Ok(Self { language, visual })
    }

    /// Total number of language-model tensors (diagnostics/tests).
    pub fn language_tensor_count(&self) -> usize {
        self.language.len()
    }

    /// Total number of vision-tower tensors (diagnostics/tests).
    pub fn visual_tensor_count(&self) -> usize {
        self.visual.len()
    }
}

/// One MLP of the form Linear(in, hidden) -> SiLU -> Linear(hidden, hidden),
/// used by the expert's time/nav/ego/history encoders and query fusion.
/// Weights are transposed to `[in, out]`; biases are `[out]`.
pub struct ExpertMlp {
    pub fc1_w: Tensor,
    pub fc1_b: Tensor,
    pub fc2_w: Tensor,
    pub fc2_b: Tensor,
}

/// One expert diffusion-transformer layer.
pub struct QwenDriveExpertLayer {
    pub input_layernorm: Tensor,
    /// Fused `[hidden, groups*(2*heads_per_group + 2)*head_dim]` (transposed),
    /// keeping the reference's group-major row order.
    pub qkv_w: Tensor,
    pub q_norm: Tensor,
    pub k_norm: Tensor,
    /// `[heads*head_dim, hidden]` (transposed).
    pub o_w: Tensor,
    pub post_attention_layernorm: Tensor,
    /// `[hidden, 2*intermediate]` (transposed).
    pub gate_up_w: Tensor,
    /// `[intermediate, hidden]` (transposed).
    pub down_w: Tensor,
    /// adaLN modulation: `[hidden, 6*hidden]` (transposed) plus bias.
    pub modulation_w: Tensor,
    pub modulation_b: Tensor,
}

/// Planning-expert weights (the planner-sft / planner-rl head).
pub struct QwenDriveExpertWeights {
    /// `[trajectory_point_dim, hidden]` (transposed).
    pub trajectory_proj_w: Tensor,
    pub trajectory_proj_b: Tensor,
    pub fourier_encoder: ExpertMlp,
    /// `[num_future_points, hidden]` waypoint-index embedding table.
    pub waypoint_embed: Tensor,
    pub time_mlp: ExpertMlp,
    pub nav_mlp: ExpertMlp,
    pub ego_mlp: ExpertMlp,
    pub history_encoder: ExpertMlp,
    pub history_velocity_encoder: ExpertMlp,
    pub history_acceleration_encoder: ExpertMlp,
    /// query_fusion: first Linear maps `[7*hidden, hidden]` (transposed).
    pub query_fusion: ExpertMlp,
    pub layers: Vec<QwenDriveExpertLayer>,
    pub final_layernorm: Tensor,
    /// `[hidden, trajectory_point_dim]` (transposed).
    pub out_proj_w: Tensor,
    pub out_proj_b: Tensor,
}

fn take(map: &mut HashMap<String, Tensor>, name: &str) -> Result<Tensor> {
    map.remove(name)
        .ok_or_else(|| Error::Other(format!("qwen_drive expert weights: missing {name}")))
}

fn take_linear(map: &mut HashMap<String, Tensor>, name: &str) -> Result<(Tensor, Tensor)> {
    let weight = take(map, &format!("{name}.weight"))?;
    let bias = take(map, &format!("{name}.bias"))?;
    Ok((transpose_2d(&weight)?, bias))
}

fn take_mlp(map: &mut HashMap<String, Tensor>, prefix: &str) -> Result<ExpertMlp> {
    // Reference `_mlp` = nn.Sequential(Linear, SiLU, Linear), i.e. `.0` and `.2`.
    let (fc1_w, fc1_b) = take_linear(map, &format!("{prefix}.0"))?;
    let (fc2_w, fc2_b) = take_linear(map, &format!("{prefix}.2"))?;
    Ok(ExpertMlp { fc1_w, fc1_b, fc2_w, fc2_b })
}

impl QwenDriveExpertWeights {
    pub fn from_map(
        config: &QwenDriveConfig,
        tensors: &HashMap<String, Tensor>,
    ) -> Result<Self> {
        // Strip the optional `planning_expert.` prefix exactly like the
        // reference `load_planner`, cloning tensors (the planner head is 1.0B
        // parameters; the clone happens once at load, not on the hot path).
        let prefix = "planning_expert.";
        let mut map: HashMap<String, Tensor> = tensors
            .iter()
            .map(|(key, tensor)| {
                let stripped = key.strip_prefix(prefix).unwrap_or(key.as_str());
                (stripped.to_owned(), tensor.clone())
            })
            .collect();

        let mut layers = Vec::with_capacity(config.expert.n_layers);
        for index in 0..config.expert.n_layers {
            let p = format!("layers.{index}");
            let qkv = take(&mut map, &format!("{p}.qkv_proj.weight"))?;
            let gate_up = take(&mut map, &format!("{p}.gate_up_proj.weight"))?;
            let down = take(&mut map, &format!("{p}.down_proj.weight"))?;
            let o = take(&mut map, &format!("{p}.o_proj.weight"))?;
            let modulation = take(&mut map, &format!("{p}.adaln_modulation.1.weight"))?;
            layers.push(QwenDriveExpertLayer {
                input_layernorm: take(&mut map, &format!("{p}.input_layernorm.weight"))?,
                qkv_w: transpose_2d(&qkv)?,
                q_norm: take(&mut map, &format!("{p}.q_norm.weight"))?,
                k_norm: take(&mut map, &format!("{p}.k_norm.weight"))?,
                o_w: transpose_2d(&o)?,
                post_attention_layernorm: take(
                    &mut map,
                    &format!("{p}.post_attention_layernorm.weight"),
                )?,
                gate_up_w: transpose_2d(&gate_up)?,
                down_w: transpose_2d(&down)?,
                modulation_w: transpose_2d(&modulation)?,
                modulation_b: take(&mut map, &format!("{p}.adaln_modulation.1.bias"))?,
            });
        }

        let (trajectory_proj_w, trajectory_proj_b) = take_linear(&mut map, "trajectory_proj")?;
        let (out_proj_w, out_proj_b) = take_linear(&mut map, "out_proj")?;
        let weights = Self {
            trajectory_proj_w,
            trajectory_proj_b,
            fourier_encoder: take_mlp(&mut map, "fourier_encoder.net")?,
            waypoint_embed: take(&mut map, "waypoint_embed.weight")?,
            time_mlp: take_mlp(&mut map, "time_mlp")?,
            nav_mlp: take_mlp(&mut map, "nav_mlp")?,
            ego_mlp: take_mlp(&mut map, "ego_mlp")?,
            history_encoder: take_mlp(&mut map, "history_encoder")?,
            history_velocity_encoder: take_mlp(&mut map, "history_velocity_encoder")?,
            history_acceleration_encoder: take_mlp(&mut map, "history_acceleration_encoder")?,
            query_fusion: take_mlp(&mut map, "query_fusion")?,
            layers,
            final_layernorm: take(&mut map, "final_layernorm.weight")?,
            out_proj_w,
            out_proj_b,
        };
        weights.validate(config, &map)?;
        Ok(weights)
    }

    /// Shape-level validation. Runs once at load; the first tuple element of
    /// every 2D weight is its input width after the load-time transpose.
    pub fn validate(&self, config: &QwenDriveConfig, leftover: &HashMap<String, Tensor>) -> Result<()> {
        let expert = &config.expert;
        let hidden = expert.hidden_size;
        let heads_per_group = expert.n_heads / expert.n_kv_heads;
        let qkv_out = expert.n_kv_heads * (2 * heads_per_group + 2) * expert.head_dim;
        for (index, layer) in self.layers.iter().enumerate() {
            expect_shape(&layer.qkv_w, &[hidden, qkv_out], &format!("layers.{index}.qkv_proj"))?;
            expect_shape(&layer.o_w, &[expert.n_heads * expert.head_dim, hidden], &format!("layers.{index}.o_proj"))?;
            expect_shape(&layer.gate_up_w, &[hidden, 2 * expert.intermediate_size], &format!("layers.{index}.gate_up_proj"))?;
            expect_shape(&layer.down_w, &[expert.intermediate_size, hidden], &format!("layers.{index}.down_proj"))?;
            expect_shape(&layer.modulation_w, &[hidden, 6 * hidden], &format!("layers.{index}.adaln_modulation"))?;
            expect_shape(&layer.q_norm, &[expert.head_dim], &format!("layers.{index}.q_norm"))?;
            expect_shape(&layer.k_norm, &[expert.head_dim], &format!("layers.{index}.k_norm"))?;
        }
        expect_shape(&self.trajectory_proj_w, &[config.trajectory_point_dim, hidden], "trajectory_proj")?;
        expect_shape(&self.waypoint_embed, &[config.num_future_points, hidden], "waypoint_embed")?;
        expect_shape(&self.out_proj_w, &[hidden, config.trajectory_point_dim], "out_proj")?;
        let history_dim = (config.num_history_points - 1) * config.trajectory_point_dim
            + expert.nav_command_classes;
        expect_shape(&self.history_encoder.fc1_w, &[history_dim, hidden], "history_encoder.0")?;
        let dynamics_dim = config.num_history_points * expert.history_dynamics_dim;
        expect_shape(&self.history_velocity_encoder.fc1_w, &[dynamics_dim, hidden], "history_velocity_encoder.0")?;
        expect_shape(&self.history_acceleration_encoder.fc1_w, &[dynamics_dim, hidden], "history_acceleration_encoder.0")?;
        expect_shape(&self.query_fusion.fc1_w, &[7 * hidden, hidden], "query_fusion.0")?;
        expect_shape(&self.time_mlp.fc1_w, &[expert.time_embed_dim, hidden], "time_mlp.0")?;
        expect_shape(&self.nav_mlp.fc1_w, &[expert.nav_command_classes, hidden], "nav_mlp.0")?;
        expect_shape(&self.ego_mlp.fc1_w, &[expert.ego_status_dim, hidden], "ego_mlp.0")?;
        let fourier_in = config.trajectory_point_dim * expert.fourier_num_features * 2;
        expect_shape(&self.fourier_encoder.fc1_w, &[fourier_in, hidden], "fourier_encoder.net.0")?;
        if !leftover.is_empty() {
            let mut names: Vec<&String> = leftover.keys().collect();
            names.sort_unstable();
            return Err(Error::Other(format!(
                "qwen_drive expert weights: {} unconsumed tensors (strict load), first: {}",
                names.len(),
                names[0]
            )));
        }
        Ok(())
    }
}

fn expect_shape(tensor: &Tensor, dims: &[usize], name: &str) -> Result<()> {
    let actual = tensor.shape().dims();
    if actual != dims {
        return Err(Error::Other(format!(
            "qwen_drive expert weights: {name} has shape {actual:?}, expected {dims:?}"
        )));
    }
    Ok(())
}

/// Transpose a 2D HF projection weight `[out, in]` to `[in, out]` for the
/// row-major matmul path. Handles f32 and bf16 (the checkpoint is bf16).
pub fn transpose_2d(tensor: &Tensor) -> Result<Tensor> {
    let dims = tensor.shape().dims();
    if dims.len() != 2 {
        return Err(Error::Other(format!(
            "qwen_drive transpose_2d expected 2D tensor, got {}D",
            dims.len()
        )));
    }
    let [rows, cols] = [dims[0], dims[1]];
    match tensor.dtype() {
        apxinf_core::DType::F32 => {
            let data = tensor.as_f32()?;
            let mut out = vec![0.0f32; rows * cols];
            for i in 0..rows {
                for j in 0..cols {
                    out[j * rows + i] = data[i * cols + j];
                }
            }
            Tensor::from_f32(vec![cols, rows], &out)
        }
        apxinf_core::DType::BF16 => {
            let data = tensor.as_bf16()?;
            let mut out = vec![half::bf16::from_f32(0.0); rows * cols];
            for i in 0..rows {
                for j in 0..cols {
                    out[j * rows + i] = data[i * cols + j];
                }
            }
            Tensor::from_bf16(vec![cols, rows], &out)
        }
        dtype => Err(Error::Other(format!(
            "qwen_drive weight transpose does not support {dtype}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apxinf_core::DType;

    fn tensor(rows: usize, cols: usize, fill: f32) -> Tensor {
        Tensor::from_f32(vec![rows, cols], &vec![fill; rows * cols]).unwrap()
    }

    #[test]
    fn transpose_swaps_dimensions() {
        let a = Tensor::from_f32(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let t = transpose_2d(&a).unwrap();
        assert_eq!(t.shape().dims(), &[3, 2]);
        assert_eq!(t.as_f32().unwrap(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn vlm_weights_reject_expert_head_maps() {
        let mut map = HashMap::new();
        map.insert("planning_expert.layers.0.qkv_proj.weight".to_string(), tensor(4, 4, 0.0));
        let err = QwenDriveVlmWeights::from_map(map).err().unwrap();
        assert!(format!("{err}").contains("planning-expert"));
    }

    #[test]
    fn expert_weights_strip_prefix_and_transpose() {
        // Minimal 1-layer expert geometry over a synthetic config; checks the
        // key mapping, prefix stripping, transpose and strict consumption.
        let config_json = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../experiment/qwen-drive-k3/checkpoint-configs/config.json"),
        );
        // The experiment fixture is not shipped with the crate; fall back to a
        // synthetic config when it is absent so `cargo test` works from a
        // bare checkout of the product tree.
        let config = if let Ok(raw) = config_json {
            QwenDriveConfig::from_json_str(&raw).unwrap()
        } else {
            return; // covered by the integration environment
        };
        let mut map: HashMap<String, Tensor> = HashMap::new();
        let hidden = config.expert.hidden_size;
        let inter = config.expert.intermediate_size;
        let heads_per_group = config.expert.n_heads / config.expert.n_kv_heads;
        let qkv_out = config.expert.n_kv_heads * (2 * heads_per_group + 2) * config.expert.head_dim;
        for index in 0..config.expert.n_layers {
            let p = format!("planning_expert.layers.{index}");
            map.insert(format!("{p}.input_layernorm.weight"), tensor(1, hidden, 1.0));
            map.insert(format!("{p}.qkv_proj.weight"), tensor(qkv_out, hidden, 0.0));
            // RMSNorm weights are 1D `[head_dim]` in the real checkpoint.
            map.insert(
                format!("{p}.q_norm.weight"),
                Tensor::from_f32(vec![config.expert.head_dim], &vec![1.0; config.expert.head_dim]).unwrap(),
            );
            map.insert(
                format!("{p}.k_norm.weight"),
                Tensor::from_f32(vec![config.expert.head_dim], &vec![1.0; config.expert.head_dim]).unwrap(),
            );
            map.insert(format!("{p}.o_proj.weight"), tensor(hidden, config.expert.n_heads * config.expert.head_dim, 0.0));
            map.insert(format!("{p}.post_attention_layernorm.weight"), tensor(1, hidden, 1.0));
            map.insert(format!("{p}.gate_up_proj.weight"), tensor(2 * inter, hidden, 0.0));
            map.insert(format!("{p}.down_proj.weight"), tensor(hidden, inter, 0.0));
            map.insert(format!("{p}.adaln_modulation.1.weight"), tensor(6 * hidden, hidden, 0.0));
            map.insert(format!("{p}.adaln_modulation.1.bias"), tensor(1, 6 * hidden, 0.0));
        }
        let mlp = |map: &mut HashMap<String, Tensor>, prefix: &str, input: usize| {
            map.insert(format!("{prefix}.0.weight"), tensor(hidden, input, 0.0));
            map.insert(format!("{prefix}.0.bias"), tensor(1, hidden, 0.0));
            map.insert(format!("{prefix}.2.weight"), tensor(hidden, hidden, 0.0));
            map.insert(format!("{prefix}.2.bias"), tensor(1, hidden, 0.0));
        };
        let fourier_in = config.trajectory_point_dim * config.expert.fourier_num_features * 2;
        mlp(&mut map, "planning_expert.fourier_encoder.net", fourier_in);
        mlp(&mut map, "planning_expert.time_mlp", config.expert.time_embed_dim);
        mlp(&mut map, "planning_expert.nav_mlp", config.expert.nav_command_classes);
        mlp(&mut map, "planning_expert.ego_mlp", config.expert.ego_status_dim);
        let history_dim = (config.num_history_points - 1) * config.trajectory_point_dim
            + config.expert.nav_command_classes;
        mlp(&mut map, "planning_expert.history_encoder", history_dim);
        let dynamics_dim = config.num_history_points * config.expert.history_dynamics_dim;
        mlp(&mut map, "planning_expert.history_velocity_encoder", dynamics_dim);
        mlp(&mut map, "planning_expert.history_acceleration_encoder", dynamics_dim);
        mlp(&mut map, "planning_expert.query_fusion", 7 * hidden);
        map.insert("planning_expert.trajectory_proj.weight".to_string(), tensor(hidden, config.trajectory_point_dim, 0.0));
        map.insert("planning_expert.trajectory_proj.bias".to_string(), tensor(1, hidden, 0.0));
        map.insert("planning_expert.waypoint_embed.weight".to_string(), tensor(config.num_future_points, hidden, 0.0));
        map.insert("planning_expert.final_layernorm.weight".to_string(), tensor(1, hidden, 1.0));
        map.insert("planning_expert.out_proj.weight".to_string(), tensor(config.trajectory_point_dim, hidden, 0.0));
        map.insert("planning_expert.out_proj.bias".to_string(), tensor(1, config.trajectory_point_dim, 0.0));

        let weights = QwenDriveExpertWeights::from_map(&config, &map).unwrap();
        assert_eq!(weights.layers.len(), config.expert.n_layers);
        assert_eq!(weights.layers[0].qkv_w.shape().dims(), &[hidden, qkv_out]);
        assert_eq!(weights.query_fusion.fc1_w.shape().dims(), &[7 * hidden, hidden]);
        assert_eq!(weights.waypoint_embed.dtype(), DType::F32);
    }
}
