//! MolmoAct2 checkpoint configuration. Only the profile this family implements is accepted;
//! everything else is rejected while the checkpoint config is parsed, not at inference time.
use apxinf_core::{Error, Result};
use serde_json::Value;
use std::path::Path;

/// Vision transformer geometry (SigLIP-style) of the checkpoint.
#[derive(Clone, Debug, PartialEq)]
pub struct VitConfig {
    pub hidden: usize,
    pub intermediate: usize,
    pub heads: usize,
    pub head_dim: usize,
    /// Layers declared by the checkpoint config.
    pub configured_layers: usize,
    /// Layers the official model executes: `max(adapter vit_layers) + 1`.
    pub executed_layers: usize,
    pub patch_size: usize,
    pub image_size: usize,
    pub num_pos: usize,
    pub eps: f32,
}

/// Pooling attention + projector ("adapter") geometry.
#[derive(Clone, Debug, PartialEq)]
pub struct AdapterConfig {
    /// Absolute indices (after resolving negatives against `configured_layers`) of the ViT block
    /// outputs that are concatenated, in concatenation order.
    pub vit_layers: Vec<usize>,
    pub hidden: usize,
    pub intermediate: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub text_hidden: usize,
}

/// Language-model geometry.
#[derive(Clone, Debug, PartialEq)]
pub struct TextConfig {
    pub hidden: usize,
    pub intermediate: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub vocab: usize,
    pub additional_vocab: usize,
    pub eps: f32,
    pub rope_theta: f64,
    pub max_positions: usize,
}

/// Token ids that mark image positions. `is_image_token(id)` is the model's token-type rule.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageTokens {
    pub patch: u32,
    pub col: u32,
    pub start: u32,
    pub low_res_start: u32,
    pub frame_start: u32,
    pub end: u32,
    pub frame_end: u32,
    pub low: u32,
}

impl ImageTokens {
    pub fn is_image_token(&self, id: u32) -> bool {
        [
            self.patch,
            self.col,
            self.start,
            self.low_res_start,
            self.frame_start,
            self.end,
            self.frame_end,
            self.low,
        ]
        .contains(&id)
    }
}

/// Action-expert geometry (`action_expert_config`) and the flow settings of the checkpoint.
#[derive(Clone, Debug, PartialEq)]
pub struct ExpertConfig {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub head_dim: usize,
    /// MLP width: `round_up(int(hidden * mlp_ratio), ffn_multiple_of)`.
    pub mlp_inner: usize,
    pub time_embed_dim: usize,
    pub qk_norm_eps: f32,
    /// Internal (padded) action width and chunk length produced by the flow.
    pub max_action_dim: usize,
    pub max_action_horizon: usize,
    /// Forward Euler steps with `t_i = i / steps`, `dt = 1 / steps`.
    pub flow_steps: usize,
}

/// Width of the LIBERO action (`norm_tag = libero`): the first dimensions of the padded action.
pub const LIBERO_ACTION_DIM: usize = 7;
/// Width of the LIBERO proprioceptive state encoded into the prompt.
pub const LIBERO_STATE_DIM: usize = 8;

#[derive(Clone, Debug, PartialEq)]
pub struct MolmoAct2Config {
    pub vit: VitConfig,
    pub adapter: AdapterConfig,
    pub text: TextConfig,
    pub expert: ExpertConfig,
    pub image_tokens: ImageTokens,
    /// Pooling window (rows, cols) of the connector; fixed to 2x2 in this profile.
    pub pooling: (usize, usize),
    /// `eos_token_id` (equal to the BOS id of this checkpoint); hidden from the expert's context.
    pub eos_token_id: u32,
    pub action_start_token_id: u32,
    pub action_end_token_id: u32,
}

fn bad(what: impl std::fmt::Display) -> Error {
    Error::Other(format!("MolmoAct2 config: {what}"))
}

fn get<'a>(v: &'a Value, path: &[&str]) -> Result<&'a Value> {
    let mut cur = v;
    for key in path {
        cur = cur
            .get(*key)
            .ok_or_else(|| bad(format!("missing `{}`", path.join("."))))?;
    }
    Ok(cur)
}

fn uint(v: &Value, path: &[&str]) -> Result<usize> {
    get(v, path)?
        .as_u64()
        .map(|x| x as usize)
        .ok_or_else(|| bad(format!("`{}` must be an unsigned integer", path.join("."))))
}

fn token(v: &Value, key: &str) -> Result<u32> {
    let id = get(v, &[key])?
        .as_u64()
        .ok_or_else(|| bad(format!("`{key}` must be an unsigned integer")))?;
    u32::try_from(id).map_err(|_| bad(format!("`{key}` is out of range")))
}

fn float(v: &Value, path: &[&str]) -> Result<f64> {
    get(v, path)?
        .as_f64()
        .ok_or_else(|| bad(format!("`{}` must be a number", path.join("."))))
}

fn expect(cond: bool, what: &str) -> Result<()> {
    if cond {
        Ok(())
    } else {
        Err(bad(format!("unsupported profile: {what}")))
    }
}

impl MolmoAct2Config {
    pub fn from_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| bad(format!("read {}: {e}", path.display())))?;
        let value: Value = serde_json::from_str(&text).map_err(|e| bad(format!("parse: {e}")))?;
        Self::from_json(&value)
    }

    pub fn from_json(v: &Value) -> Result<Self> {
        expect(
            get(v, &["model_type"])?.as_str() == Some("molmoact2"),
            "model_type must be molmoact2",
        )?;
        let configured_layers = uint(v, &["vit_config", "num_hidden_layers"])?;
        let layers: Vec<i64> = get(v, &["adapter_config", "vit_layers"])?
            .as_array()
            .ok_or_else(|| bad("`adapter_config.vit_layers` must be an array"))?
            .iter()
            .map(|x| x.as_i64().ok_or_else(|| bad("vit_layers entries must be integers")))
            .collect::<Result<_>>()?;
        let resolved: Vec<usize> = layers
            .iter()
            .map(|&l| {
                let abs = if l >= 0 { l } else { l + configured_layers as i64 };
                usize::try_from(abs).map_err(|_| bad("vit_layers entry out of range"))
            })
            .collect::<Result<_>>()?;
        let executed_layers = resolved.iter().copied().max().ok_or_else(|| bad("empty vit_layers"))? + 1;
        expect(executed_layers <= configured_layers, "vit_layers exceed the ViT depth")?;

        let size = get(v, &["vit_config", "image_default_input_size"])?
            .as_array()
            .ok_or_else(|| bad("image_default_input_size must be an array"))?;
        let (h, w) = (
            size.first().and_then(Value::as_u64).unwrap_or(0) as usize,
            size.get(1).and_then(Value::as_u64).unwrap_or(0) as usize,
        );
        expect(h == w && h > 0, "square input image")?;

        let vit = VitConfig {
            hidden: uint(v, &["vit_config", "hidden_size"])?,
            intermediate: uint(v, &["vit_config", "intermediate_size"])?,
            heads: uint(v, &["vit_config", "num_attention_heads"])?,
            head_dim: uint(v, &["vit_config", "head_dim"])?,
            configured_layers,
            executed_layers,
            patch_size: uint(v, &["vit_config", "image_patch_size"])?,
            image_size: h,
            num_pos: uint(v, &["vit_config", "image_num_pos"])?,
            eps: float(v, &["vit_config", "layer_norm_eps"])? as f32,
        };
        let adapter = AdapterConfig {
            vit_layers: resolved,
            hidden: uint(v, &["adapter_config", "hidden_size"])?,
            intermediate: uint(v, &["adapter_config", "intermediate_size"])?,
            heads: uint(v, &["adapter_config", "num_attention_heads"])?,
            head_dim: uint(v, &["adapter_config", "head_dim"])?,
            text_hidden: uint(v, &["adapter_config", "text_hidden_size"])?,
        };
        let text = TextConfig {
            hidden: uint(v, &["text_config", "hidden_size"])?,
            intermediate: uint(v, &["text_config", "intermediate_size"])?,
            layers: uint(v, &["text_config", "num_hidden_layers"])?,
            heads: uint(v, &["text_config", "num_attention_heads"])?,
            kv_heads: uint(v, &["text_config", "num_key_value_heads"])?,
            head_dim: uint(v, &["text_config", "head_dim"])?,
            vocab: uint(v, &["text_config", "vocab_size"])?,
            additional_vocab: uint(v, &["text_config", "additional_vocab_size"])?,
            eps: float(v, &["text_config", "layer_norm_eps"])? as f32,
            rope_theta: float(v, &["text_config", "rope_theta"])?,
            max_positions: uint(v, &["text_config", "max_position_embeddings"])?,
        };
        let image_tokens = ImageTokens {
            patch: token(v, "image_patch_id")?,
            col: token(v, "image_col_id")?,
            start: token(v, "image_start_token_id")?,
            low_res_start: token(v, "low_res_image_start_token_id")?,
            frame_start: token(v, "frame_start_token_id")?,
            end: token(v, "image_end_token_id")?,
            frame_end: token(v, "frame_end_token_id")?,
            low: token(v, "image_low_res_id")?,
        };

        // Profile this family implements (the LIBERO checkpoint). Anything else would silently
        // compute something other than the checkpoint's model, so fail here.
        let ts = |p: &[&str]| get(v, p).ok().and_then(Value::as_str).map(str::to_owned);
        // Zero sizes would reach a division or modulo below (or in `patch_side`); reject them as an
        // unsupported profile first instead of panicking on a valid-JSON config.
        expect(
            vit.patch_size > 0
                && vit.heads > 0
                && vit.head_dim > 0
                && adapter.heads > 0
                && text.heads > 0
                && text.kv_heads > 0
                && text.layers > 0,
            "patch size, head counts, head dims and the layer count must be positive",
        )?;
        expect(vit.hidden == vit.heads * vit.head_dim, "ViT hidden == heads * head_dim")?;
        expect(
            adapter.hidden == adapter.heads * adapter.head_dim && adapter.hidden == vit.hidden,
            "pooling attention width equals the ViT width",
        )?;
        expect(vit.image_size % vit.patch_size == 0, "image size divisible by patch size")?;
        let side = vit.image_size / vit.patch_size;
        expect(vit.num_pos == side * side, "positional table matches the patch grid (no interpolation)")?;
        expect(adapter.vit_layers.len() == 2, "two concatenated ViT layers")?;
        expect(
            ts(&["vit_config", "hidden_act"]).as_deref() == Some("gelu_pytorch_tanh"),
            "ViT activation gelu_pytorch_tanh",
        )?;
        expect(ts(&["adapter_config", "hidden_act"]).as_deref() == Some("silu"), "adapter activation silu")?;
        expect(ts(&["text_config", "hidden_act"]).as_deref() == Some("silu"), "text activation silu")?;
        expect(
            get(v, &["adapter_config", "pooling_attention_mask"])?.as_bool() == Some(true),
            "pooling_attention_mask",
        )?;
        expect(
            ts(&["text_config", "qk_norm_type"]).as_deref() == Some("qwen3"),
            "per-head (qwen3-style) Q/K norm",
        )?;
        expect(get(v, &["text_config", "use_qk_norm"])?.as_bool() == Some(true), "use_qk_norm")?;
        expect(get(v, &["text_config", "qkv_bias"])?.as_bool() == Some(false), "no QKV bias")?;
        expect(get(v, &["text_config", "norm_after"])?.as_bool() == Some(false), "pre-norm decoder layers")?;
        expect(
            text.heads % text.kv_heads == 0 && text.head_dim == 128,
            "GQA with head_dim 128",
        )?;
        expect(text.hidden == adapter.text_hidden, "connector output width equals the LM width")?;
        expect(
            get(v, &["text_config", "rope_scaling_layers"])?.is_null(),
            "no per-layer RoPE scaling",
        )?;

        // Action expert + flow. A plain VLM / pretrain / depth-reasoning checkpoint is not this model.
        let flag = |p: &[&str]| get(v, p).ok().and_then(Value::as_bool);
        expect(flag(&["add_action_expert"]) == Some(true), "checkpoint has an action expert (add_action_expert)")?;
        expect(ts(&["action_mode"]).as_deref() == Some("both"), "action_mode both (continuous expert + discrete tokens)")?;
        expect(ts(&["state_format"]).as_deref() == Some("discrete"), "discrete state tokens in the prompt")?;
        expect(
            flag(&["enable_depth_reasoning"]) != Some(true) && flag(&["action_expert_depth_gate"]) != Some(true),
            "no depth reasoning / depth gate",
        )?;
        expect(flag(&["mask_action_dim_padding"]) == Some(true), "padded action dimensions are masked")?;
        expect(uint(v, &["n_obs_steps"])? == 1, "a single observation step")?;
        let ec = "action_expert_config";
        let at = |key: &'static str| [ec, key];
        expect(flag(&at("qk_norm")) == Some(true), "expert Q/K norm")?;
        expect(flag(&at("rope")) == Some(true), "expert self-attention RoPE")?;
        expect(flag(&at("context_layer_norm")) == Some(true), "expert context norm")?;
        expect(flag(&at("causal_attn")) == Some(false), "non-causal expert self-attention")?;
        let (hidden, heads) = (uint(v, &at("hidden_size"))?, uint(v, &at("num_heads"))?);
        expect(heads > 0 && hidden % heads == 0, "expert hidden divisible by its head count")?;
        let ratio = float(v, &at("mlp_ratio"))?;
        let multiple = uint(v, &at("ffn_multiple_of"))?;
        expect(multiple > 0 && ratio > 0.0, "positive expert MLP ratio and multiple")?;
        let inner = (hidden as f64 * ratio) as usize;
        let expert = ExpertConfig {
            hidden,
            layers: uint(v, &at("num_layers"))?,
            heads,
            head_dim: hidden / heads,
            mlp_inner: inner.div_ceil(multiple) * multiple,
            time_embed_dim: uint(v, &at("timestep_embed_dim"))?,
            qk_norm_eps: float(v, &at("qk_norm_eps"))? as f32,
            max_action_dim: uint(v, &["max_action_dim"])?,
            max_action_horizon: uint(v, &["max_action_horizon"])?,
            flow_steps: uint(v, &["flow_matching_num_steps"])?,
        };
        expect(expert.layers == text.layers, "one expert block per language-model layer")?;
        expect(expert.time_embed_dim > 0 && expert.time_embed_dim % 2 == 0, "even timestep embedding width")?;
        expect(
            expert.max_action_dim >= LIBERO_ACTION_DIM && expert.max_action_horizon > 0 && expert.flow_steps > 0,
            "action width, horizon and flow steps",
        )?;
        Ok(Self {
            vit,
            adapter,
            text,
            expert,
            image_tokens,
            pooling: (2, 2),
            eos_token_id: token(v, "eos_token_id")?,
            action_start_token_id: token(v, "action_start_token_id")?,
            action_end_token_id: token(v, "action_end_token_id")?,
        })
    }

    /// Patches per image side for the single-crop 378x378 profile.
    pub fn patch_side(&self) -> usize {
        self.vit.image_size / self.vit.patch_size
    }

    /// Checks that `norm_stats.json` describes the LIBERO profile this family produces actions for:
    /// `q01_q99` statistics, a 7-wide action and 8-wide state, and the checkpoint's chunk length. The
    /// runtime only needs the widths (the padded dimensions beyond the 7 LIBERO ones are masked); the
    /// statistics themselves are applied by the policy.
    pub fn check_libero_stats(&self, stats: &Value) -> Result<()> {
        let width = |section: &str| -> Result<usize> {
            Ok(get(stats, &["metadata_by_tag", "libero", section, "q01"])?
                .as_array()
                .ok_or_else(|| bad(format!("norm_stats libero {section}.q01 must be an array")))?
                .len())
        };
        expect(get(stats, &["norm_mode"])?.as_str() == Some("q01_q99"), "norm_stats q01_q99 normalization")?;
        expect(width("action_stats")? == LIBERO_ACTION_DIM, "7-wide LIBERO action statistics")?;
        expect(width("state_stats")? == LIBERO_STATE_DIM, "8-wide LIBERO state statistics")?;
        expect(
            uint(stats, &["metadata_by_tag", "libero", "action_horizon"])? == self.expert.max_action_horizon,
            "norm_stats chunk length equals max_action_horizon",
        )?;
        Ok(())
    }

    pub fn check_libero_stats_file(&self, path: &Path) -> Result<()> {
        let text = std::fs::read_to_string(path).map_err(|e| bad(format!("read {}: {e}", path.display())))?;
        let value: Value = serde_json::from_str(&text).map_err(|e| bad(format!("parse {}: {e}", path.display())))?;
        self.check_libero_stats(&value)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn libero_json() -> Value {
        let mut config = serde_json::json!({
            "model_type": "molmoact2", "eos_token_id": 151645,
            "action_start_token_id": 151932, "action_end_token_id": 151933,
            "image_patch_id": 154626, "image_col_id": 154627, "image_start_token_id": 154624,
            "low_res_image_start_token_id": 154628, "frame_start_token_id": 154631,
            "image_end_token_id": 154625, "frame_end_token_id": 154632, "image_low_res_id": 154630,
            "adapter_config": {"vit_layers": [-3, -9], "hidden_size": 1152, "intermediate_size": 9728,
                "num_attention_heads": 16, "head_dim": 72, "text_hidden_size": 2560,
                "hidden_act": "silu", "pooling_attention_mask": true},
            "vit_config": {"hidden_size": 1152, "intermediate_size": 4304, "num_attention_heads": 16,
                "head_dim": 72, "num_hidden_layers": 27, "image_patch_size": 14, "image_num_pos": 729,
                "image_default_input_size": [378, 378], "layer_norm_eps": 1e-6,
                "hidden_act": "gelu_pytorch_tanh"},
            "text_config": {"hidden_size": 2560, "intermediate_size": 9728, "num_hidden_layers": 36,
                "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128, "vocab_size": 154624,
                "additional_vocab_size": 128, "layer_norm_eps": 1e-6, "rope_theta": 5000000.0,
                "max_position_embeddings": 16384, "hidden_act": "silu", "qk_norm_type": "qwen3",
                "use_qk_norm": true, "qkv_bias": false, "norm_after": false, "rope_scaling_layers": null}
        });
        // Action expert and flow settings (split from the literal above to stay under the macro recursion limit).
        let expert = serde_json::json!({
            "add_action_expert": true, "action_mode": "both", "state_format": "discrete",
            "enable_depth_reasoning": false, "action_expert_depth_gate": false,
            "mask_action_dim_padding": true, "n_obs_steps": 1,
            "max_action_dim": 32, "max_action_horizon": 10, "flow_matching_num_steps": 10,
            "action_expert_config": {"hidden_size": 768, "num_layers": 36, "num_heads": 8, "mlp_ratio": 4.0,
                "ffn_multiple_of": 256, "timestep_embed_dim": 256, "qk_norm": true, "qk_norm_eps": 1e-6,
                "rope": true, "context_layer_norm": true, "causal_attn": false}
        });
        for (key, value) in expert.as_object().unwrap() {
            config[key] = value.clone();
        }
        config
    }

    #[test]
    fn libero_profile_resolves_executed_layers_and_selected_outputs() {
        let c = MolmoAct2Config::from_json(&libero_json()).unwrap();
        assert_eq!(c.vit.configured_layers, 27);
        assert_eq!(c.vit.executed_layers, 25);
        // [-3, -9] on 27 layers: outputs of blocks 24 then 18, in that order.
        assert_eq!(c.adapter.vit_layers, vec![24, 18]);
        assert_eq!(c.patch_side(), 27);
        assert!(c.image_tokens.is_image_token(154626) && !c.image_tokens.is_image_token(151645));
    }

    #[test]
    fn expert_geometry_is_derived_from_the_checkpoint_config() {
        let e = MolmoAct2Config::from_json(&libero_json()).unwrap().expert;
        assert_eq!((e.hidden, e.layers, e.heads, e.head_dim), (768, 36, 8, 96));
        // round_up(int(768 * 4.0), 256) = 3072
        assert_eq!(e.mlp_inner, 3072);
        assert_eq!((e.max_action_dim, e.max_action_horizon, e.flow_steps, e.time_embed_dim), (32, 10, 10, 256));
    }

    /// A checkpoint that is not the plain continuous-action MolmoAct2 must not load as this model,
    /// even though its `model_type` is `molmoact2`.
    #[test]
    fn same_model_type_without_the_continuous_action_profile_is_rejected() {
        for (path, value) in [
            ("add_action_expert", serde_json::json!(false)),
            ("action_mode", serde_json::json!("discrete")),
            ("state_format", serde_json::json!("continuous")),
            ("enable_depth_reasoning", serde_json::json!(true)),
            ("action_expert_depth_gate", serde_json::json!(true)),
            ("mask_action_dim_padding", serde_json::json!(false)),
            ("n_obs_steps", serde_json::json!(2)),
        ] {
            let mut j = libero_json();
            j[path] = value;
            assert!(MolmoAct2Config::from_json(&j).is_err(), "{path}");
        }
        let mut j = libero_json();
        j.as_object_mut().unwrap().remove("add_action_expert");
        assert!(MolmoAct2Config::from_json(&j).is_err(), "pure VLM config without the key");
        for (key, value) in [("causal_attn", serde_json::json!(true)), ("qk_norm", serde_json::json!(false)), ("num_layers", serde_json::json!(35))] {
            let mut j = libero_json();
            j["action_expert_config"][key] = value;
            assert!(MolmoAct2Config::from_json(&j).is_err(), "action_expert_config.{key}");
        }
    }

    #[test]
    fn libero_statistics_must_match_the_seven_wide_action_and_eight_wide_state() {
        let c = MolmoAct2Config::from_json(&libero_json()).unwrap();
        let stats = |action: usize, state: usize, horizon: u64, mode: &str| {
            serde_json::json!({"norm_mode": mode, "metadata_by_tag": {"libero": {
                "action_horizon": horizon,
                "action_stats": {"q01": vec![0.0; action]}, "state_stats": {"q01": vec![0.0; state]}}}})
        };
        assert!(c.check_libero_stats(&stats(7, 8, 10, "q01_q99")).is_ok());
        assert!(c.check_libero_stats(&stats(6, 8, 10, "q01_q99")).is_err());
        assert!(c.check_libero_stats(&stats(7, 7, 10, "q01_q99")).is_err());
        assert!(c.check_libero_stats(&stats(7, 8, 16, "q01_q99")).is_err());
        assert!(c.check_libero_stats(&stats(7, 8, 10, "min_max")).is_err());
        assert!(c.check_libero_stats(&serde_json::json!({"norm_mode": "q01_q99", "metadata_by_tag": {}})).is_err());
    }

    #[test]
    fn other_profiles_are_rejected() {
        let mut j = libero_json();
        j["text_config"]["qk_norm_type"] = "olmo".into();
        assert!(MolmoAct2Config::from_json(&j).is_err());
        let mut j = libero_json();
        j["vit_config"]["image_num_pos"] = 784.into();
        assert!(MolmoAct2Config::from_json(&j).is_err());
    }

    /// Valid JSON with a zero divisor must be an `Err`, never a panic (`%` / `/` on zero).
    #[test]
    fn zero_sizes_are_rejected_without_panicking() {
        for (section, key) in [
            ("vit_config", "image_patch_size"),
            ("vit_config", "num_attention_heads"),
            ("vit_config", "head_dim"),
            ("adapter_config", "num_attention_heads"),
            ("text_config", "num_key_value_heads"),
            ("text_config", "num_attention_heads"),
            ("text_config", "num_hidden_layers"),
        ] {
            let mut j = libero_json();
            j[section][key] = 0.into();
            let r = std::panic::catch_unwind(|| MolmoAct2Config::from_json(&j));
            assert!(matches!(r, Ok(Err(_))), "{section}.{key} = 0 must be rejected with an error, got {r:?}");
        }
    }
}
