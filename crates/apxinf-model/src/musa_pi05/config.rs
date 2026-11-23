use std::path::Path;

use apxinf_core::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GemmaVariantConfig {
    pub width: usize,
    pub depth: usize,
    pub mlp_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl GemmaVariantConfig {
    pub const GEMMA_2B: Self = Self {
        width: 2048,
        depth: 18,
        mlp_dim: 16_384,
        num_heads: 8,
        num_kv_heads: 1,
        head_dim: 256,
    };

    pub const GEMMA_300M: Self = Self {
        width: 1024,
        depth: 18,
        mlp_dim: 4096,
        num_heads: 8,
        num_kv_heads: 1,
        head_dim: 256,
    };
}

#[derive(Clone, Debug, PartialEq)]
pub struct Pi05Config {
    pub action_dim: usize,
    pub action_horizon: usize,
    pub max_token_len: usize,
    pub num_flow_steps: usize,
    pub flow_start_time: f32,
    pub num_views: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub vision_width: usize,
    pub vision_depth: usize,
    pub vision_mlp_dim: usize,
    pub vision_heads: usize,
    pub vision_head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub layer_norm_eps: f32,
    pub rope_theta: f32,
    pub time_min_period: f32,
    pub time_max_period: f32,
    pub discrete_state_input: bool,
    pub language: GemmaVariantConfig,
    pub action_expert: GemmaVariantConfig,
}

impl Default for Pi05Config {
    fn default() -> Self {
        Self {
            action_dim: 32,
            action_horizon: 50,
            max_token_len: 200,
            num_flow_steps: 10,
            flow_start_time: 1.0,
            num_views: 3,
            image_size: 224,
            patch_size: 14,
            vision_width: 1152,
            vision_depth: 27,
            vision_mlp_dim: 4304,
            vision_heads: 16,
            vision_head_dim: 72,
            vocab_size: 257_152,
            rms_norm_eps: 1e-6,
            layer_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            time_min_period: 4e-3,
            time_max_period: 4.0,
            discrete_state_input: true,
            language: GemmaVariantConfig::GEMMA_2B,
            action_expert: GemmaVariantConfig::GEMMA_300M,
        }
    }
}

impl Pi05Config {
    /// The two-view capture shape the device tests drive.
    #[cfg(test)]
    pub(crate) fn two_view_config() -> Self {
        Self {
            num_views: 2,
            action_horizon: 10,
            num_flow_steps: 10,
            ..Self::default()
        }
    }

    /// The three-view capture shape the device tests drive.
    #[cfg(test)]
    pub(crate) fn three_view_config() -> Self {
        Self {
            num_views: 3,
            action_horizon: 10,
            num_flow_steps: 10,
            ..Self::default()
        }
    }

    pub fn from_json_file(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| Error::Other(format!("read {}: {e}", path.display())))?;
        Self::from_json_str(&raw)
    }

    /// Parse LeRobot's policy config while accepting compatible OpenPI aliases.
    /// Architecture constants deliberately retain the validated PI0.5 values
    /// when older checkpoints omit them.
    pub fn from_json_str(raw: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(raw)
            .map_err(|e| Error::Other(format!("pi05 config json: {e}")))?;
        let mut cfg = Self::default();

        cfg.action_dim = usize_field(&v, &["max_action_dim", "action_dim"], cfg.action_dim);
        cfg.action_horizon = usize_field(&v, &["chunk_size", "action_horizon"], cfg.action_horizon);
        cfg.max_token_len = usize_field(
            &v,
            &["tokenizer_max_length", "max_token_len"],
            cfg.max_token_len,
        );
        cfg.num_flow_steps = usize_field(
            &v,
            &["num_inference_steps", "num_flow_steps"],
            cfg.num_flow_steps,
        );
        cfg.flow_start_time = f32_field(&v, "flow_start_time", cfg.flow_start_time);
        cfg.num_views = resolve_num_views(&v, cfg.num_views);
        cfg.discrete_state_input = v
            .get("discrete_state_input")
            .and_then(|x| x.as_bool())
            .unwrap_or(cfg.discrete_state_input);
        cfg.time_min_period = f32_field(&v, "min_period", cfg.time_min_period);
        cfg.time_max_period = f32_field(&v, "max_period", cfg.time_max_period);

        if let Some(resolution) = v.get("image_resolution").and_then(|x| x.as_array()) {
            if let Some(size) = resolution.first().and_then(|x| x.as_u64()) {
                cfg.image_size = size as usize;
            }
        }

        cfg.validate()?;
        Ok(cfg)
    }

    pub fn patches_per_view(&self) -> usize {
        let side = self.image_size / self.patch_size;
        side * side
    }

    pub fn max_prefix_len(&self) -> usize {
        self.num_views * self.patches_per_view() + self.max_token_len
    }

    /// Arena bytes for one allocation-free full inference capture, before the
    /// BF16 factor.
    ///
    /// The current graph intentionally gives every intermediate a stable
    /// address. A later liveness planner can reuse slots and reduce this
    /// reservation without changing the captured execution contract.
    fn graph_workspace_bytes(&self, token_count: usize) -> Result<usize> {
        if token_count == 0 || token_count > self.max_token_len {
            return Err(Error::Other(format!(
                "pi05 token count must be in 1..={}, got {token_count}",
                self.max_token_len
            )));
        }
        const ALIGNMENT: u128 = 256;
        const SAFETY_MARGIN: u128 = 1024 * 1024;
        let mut total = 0u128;
        let mut allocate = |bytes: usize| {
            total = (total + ALIGNMENT - 1) & !(ALIGNMENT - 1);
            total += bytes as u128;
        };

        let patches = self.num_views * self.patches_per_view();
        let patch_width = 3 * self.patch_size * self.patch_size;
        let vision = self.vision_width;
        allocate(patches * patch_width * 2);
        allocate(patches * vision * 2);
        allocate(patches * vision * 2);
        for _ in 0..self.vision_depth {
            allocate(patches * vision);
            allocate(patches * 3 * vision * 2);
            for _ in 0..3 {
                allocate(patches * vision * 2);
            }
            allocate(patches * vision * 2);
            allocate(patches * vision);
            allocate(patches * vision * 2);
            allocate(patches * vision * 2);
            allocate(patches * vision * 2);
            allocate(patches * vision * 2);
            allocate(patches * vision);
            allocate(patches * self.vision_mlp_dim * 2);
            allocate(patches * self.vision_mlp_dim * 2);
            allocate(patches * self.vision_mlp_dim);
            allocate(patches * vision * 2);
            allocate(patches * vision * 2);
        }
        allocate(patches * vision);
        allocate(patches * vision * 2);
        allocate(patches * self.language.width * 2);
        allocate(patches * self.language.width * 2);

        allocate(token_count * self.language.width * 2);
        let prefix = patches + token_count;
        allocate(prefix * self.language.width * 2);
        let language_q = self.language.num_heads * self.language.head_dim;
        let language_kv = self.language.num_kv_heads * self.language.head_dim;
        let language_qkv = language_q + 2 * language_kv;
        for layer_index in 0..self.language.depth {
            allocate(prefix * self.language.width);
            allocate(prefix * language_qkv * 2);
            allocate(prefix * language_q * 2);
            allocate(prefix * language_kv * 2);
            allocate(prefix * language_kv * 2);
            allocate(prefix * self.language.width * 2);
            if layer_index + 1 < self.language.depth {
                allocate(prefix * language_q * 2);
                allocate(prefix * self.language.num_heads * std::mem::size_of::<f32>());
                allocate(prefix * language_q);
                allocate(prefix * self.language.width * 2);
                allocate(prefix * self.language.width * 2);
                allocate(prefix * self.language.width * 2);
                allocate(prefix * self.language.width);
                allocate(prefix * 2 * self.language.mlp_dim * 2);
                allocate(prefix * self.language.mlp_dim);
                allocate(prefix * self.language.width * 2);
                allocate(prefix * self.language.width * 2);
            }
            allocate((prefix + self.action_horizon) * language_kv * 2);
            allocate((prefix + self.action_horizon) * language_kv * 2);
        }

        let horizon = self.action_horizon;
        let action = self.action_expert.width;
        let action_q = self.action_expert.num_heads * self.action_expert.head_dim;
        let action_kv = self.action_expert.num_kv_heads * self.action_expert.head_dim;
        let action_qkv = action_q + 2 * action_kv;
        for _ in 0..self.num_flow_steps {
            allocate(horizon * self.action_dim);
            allocate(horizon * action * 2);
            allocate(horizon * action * 2);
            allocate(horizon * action * 2);
            for _ in 0..self.action_expert.depth {
                allocate(horizon * action);
                allocate(horizon * action_qkv * 2);
                allocate(horizon * action_q * 2);
                allocate(horizon * action_q * 2);
                allocate(horizon * action_q);
                allocate(horizon * action * 2);
                allocate(horizon * action * 2);
                allocate(horizon * action * 2);
                allocate(horizon * action * 2);
                allocate(horizon * action);
                allocate(horizon * 2 * self.action_expert.mlp_dim * 2);
                allocate(horizon * self.action_expert.mlp_dim);
                allocate(horizon * action * 2);
                allocate(horizon * action * 2);
            }
            allocate(horizon * action);
            allocate(horizon * self.action_dim * 2);
            allocate(horizon * self.action_dim * 2);
            allocate(horizon * self.action_dim * 2);
        }

        total += SAFETY_MARGIN;
        usize::try_from(total)
            .map_err(|_| Error::Other("pi05 graph workspace exceeds address space".into()))
    }

    /// Conservative arena reservation for the native-BF16 graph.
    ///
    /// Every intermediate is two bytes wide; the schedule's bound is doubled
    /// once here, so the shape calculation stays in one place.
    pub fn graph_workspace_bytes_bf16(&self, token_count: usize) -> Result<usize> {
        self.graph_workspace_bytes(token_count)?
            .checked_mul(2)
            .ok_or_else(|| Error::Other("pi05 BF16 workspace exceeds address space".into()))
    }

    /// Reservation for the INT8 (W8A8) graph: the BF16 arena, unchanged.
    ///
    /// The `s8mma` tiles write BF16 straight out of the epilogue and the
    /// quantized activation is one byte with FP16 row scales, so the traversal
    /// does not need the extra factor CUDA applies for its INT32 accumulators
    /// and dual-backend verifier. Measured on M1000 (`mp_22`, an INT8 capture
    /// of the default schedule): 3.84 GB touched against the 13.28 GB
    /// reservation — and on a 31 GB device the reservation, not the high-water
    /// mark, is what decides whether a capture runs at all.
    pub fn graph_workspace_bytes_int8_dynamic(&self, token_count: usize) -> Result<usize> {
        self.graph_workspace_bytes_bf16(token_count)
    }

    pub fn validate(&self) -> Result<()> {
        if self.image_size % self.patch_size != 0 {
            return Err(Error::Other(format!(
                "pi05 image size {} is not divisible by patch size {}",
                self.image_size, self.patch_size
            )));
        }
        if self.action_horizon == 0 || self.action_horizon > 50 {
            return Err(Error::Other(format!(
                "pi05 action_horizon must be in 1..=50, got {}",
                self.action_horizon
            )));
        }
        if self.num_flow_steps == 0 {
            return Err(Error::Other("pi05 num_flow_steps must be non-zero".into()));
        }
        if !(self.flow_start_time.is_finite()
            && self.flow_start_time > 0.0
            && self.flow_start_time <= 1.0)
        {
            return Err(Error::Other(format!(
                "pi05 flow_start_time must be in (0, 1], got {}",
                self.flow_start_time
            )));
        }
        if self.language.depth != self.action_expert.depth
            || self.language.num_heads != self.action_expert.num_heads
            || self.language.num_kv_heads != self.action_expert.num_kv_heads
            || self.language.head_dim != self.action_expert.head_dim
        {
            return Err(Error::Other(
                "pi05 language and action experts must share depth/head geometry".into(),
            ));
        }
        Ok(())
    }
}

fn usize_field(v: &serde_json::Value, names: &[&str], default: usize) -> usize {
    names
        .iter()
        .find_map(|name| v.get(*name).and_then(|x| x.as_u64()))
        .map(|x| x as usize)
        .unwrap_or(default)
}

/// Resolve the number of real camera views the checkpoint expects.
///
/// Precedence: an explicit `num_views` override wins (back-compat and bench
/// overrides); otherwise the count of `VISUAL` entries in LeRobot
/// `input_features` is used. `empty_cameras` is deliberately ignored — absent
/// cameras are never sent to the runtime, so it only ever sees real views. That
/// is numerically equivalent to openpi padding + masking a trailing empty
/// camera (a masked trailing view contributes nothing and occupies no RoPE
/// position), while avoiding the wasted compute and the unmasked-padding bug.
fn resolve_num_views(v: &serde_json::Value, default: usize) -> usize {
    if let Some(explicit) = v.get("num_views").and_then(|x| x.as_u64()) {
        return explicit as usize;
    }
    if let Some(features) = v.get("input_features").and_then(|x| x.as_object()) {
        let visual = features
            .values()
            .filter(|f| f.get("type").and_then(|t| t.as_str()) == Some("VISUAL"))
            .count();
        if visual > 0 {
            return visual;
        }
    }
    default
}

fn f32_field(v: &serde_json::Value, name: &str, default: f32) -> f32 {
    v.get(name)
        .and_then(|x| x.as_f64())
        .map(|x| x as f32)
        .unwrap_or(default)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_lerobot_aliases() {
        let cfg = Pi05Config::from_json_str(
            r#"{
            "max_action_dim": 32,
            "chunk_size": 10,
            "tokenizer_max_length": 200,
            "num_inference_steps": 10,
            "flow_start_time": 0.5,
            "num_views": 2,
            "image_resolution": [224, 224],
            "discrete_state_input": false
        }"#,
        )
        .unwrap();
        assert_eq!(
            cfg,
            Pi05Config {
                discrete_state_input: false,
                flow_start_time: 0.5,
                ..Pi05Config::two_view_config()
            }
        );
        assert_eq!(cfg.patches_per_view(), 256);
        assert_eq!(cfg.max_prefix_len(), 712);
    }

    #[test]
    fn rejects_invalid_horizon() {
        let err = Pi05Config::from_json_str(r#"{"chunk_size": 0}"#).unwrap_err();
        assert!(err.to_string().contains("action_horizon"));
    }

    #[test]
    fn derives_real_view_count_from_input_features() {
        // The real pi05_libero_base config has no `num_views`; it declares two
        // VISUAL cameras plus `empty_cameras: 1`. The loader must resolve the
        // real camera count (2) and ignore the empty pad slot, not fall back to
        // the 3-view default.
        let cfg = Pi05Config::from_json_str(
            r#"{
            "input_features": {
                "observation.images.image": {"type": "VISUAL", "shape": [3, 256, 256]},
                "observation.images.image2": {"type": "VISUAL", "shape": [3, 256, 256]},
                "observation.state": {"type": "STATE", "shape": [8]}
            },
            "empty_cameras": 1,
            "chunk_size": 50,
            "num_inference_steps": 10,
            "image_resolution": [224, 224],
            "discrete_state_input": false
        }"#,
        )
        .unwrap();
        assert_eq!(cfg.num_views, 2);
    }

    #[test]
    fn explicit_num_views_overrides_input_features() {
        // A bench/override checkpoint may pin `num_views` directly; it wins over
        // the input_features count.
        let cfg = Pi05Config::from_json_str(
            r#"{
            "num_views": 3,
            "input_features": {
                "observation.images.image": {"type": "VISUAL", "shape": [3, 256, 256]},
                "observation.images.image2": {"type": "VISUAL", "shape": [3, 256, 256]}
            }
        }"#,
        )
        .unwrap();
        assert_eq!(cfg.num_views, 3);
    }

    #[test]
    fn parses_the_config_json_synthesised_from_an_openpi_metadata_pt() {
        // Shape emitted by `apxinf.checkpoints.train_config_facts`.
        let cfg = Pi05Config::from_json_str(
            r#"{"action_dim": 32, "action_horizon": 50, "discrete_state_input": true,
                "max_token_len": 200, "num_views": 3}"#,
        )
        .unwrap();
        assert_eq!(cfg.action_dim, 32);
        assert_eq!(cfg.action_horizon, 50);
        assert_eq!(cfg.max_token_len, 200);
        assert_eq!(cfg.num_views, 3);
        assert!(cfg.discrete_state_input);
    }

    #[test]
    fn bf16_workspace_is_the_doubled_schedule_bound() {
        let config = Pi05Config::three_view_config();
        assert_eq!(config.graph_workspace_bytes_bf16(10).unwrap(), 5_909_589_760);
        assert_eq!(config.graph_workspace_bytes_bf16(21).unwrap(), 5_958_033_664);
        // The INT8 route reserves the BF16 arena unchanged.
        assert_eq!(
            config.graph_workspace_bytes_int8_dynamic(21).unwrap(),
            config.graph_workspace_bytes_bf16(21).unwrap()
        );
    }
}

/// Choice of PI0.5 model implementation, including weight and activation formats.
/// Values are model-local; the loading field `model_variant` is shared.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ModelVariantChoice {
    #[default]
    Auto,
    Bf16,
    Fp8Static,
    Int8Dynamic,
    /// INT8 everywhere a tile serves *except* the language (encode) layers,
    /// whose four projections keep BF16 arithmetic.
    ///
    /// The choose-a-scope sibling of [`Int8Dynamic`](Self::Int8Dynamic): the
    /// encode layers' per-row activation scale collides with the residual
    /// stream's outlier channels and costs accuracy the ViT and the action
    /// expert do not. Everything downstream (dispatch, kernels, captures) is
    /// shared with `int8_dynamic`; only weight preparation differs.
    Int8DynamicNoEncode,
}
impl ModelVariantChoice {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Bf16 => "bf16",
            Self::Fp8Static => "fp8_static",
            Self::Int8Dynamic => "int8_dynamic",
            Self::Int8DynamicNoEncode => "int8_dynamic_no_encode",
        }
    }
    pub fn resolve(self, sm: u32, has_fp8_scales: bool) -> Self {
        match self {
            Self::Auto if sm >= 100 && has_fp8_scales => Self::Fp8Static,
            Self::Auto if (80..100).contains(&sm) => Self::Int8Dynamic,
            Self::Auto => Self::Bf16,
            explicit => explicit,
        }
    }

    /// Reject an explicitly selected implementation before loading or
    /// quantizing weights when the device cannot execute its full model path.
    pub fn ensure_supported(self, sm: u32) -> Result<Self> {
        if self == Self::Fp8Static && sm < 100 {
            return Err(Error::Other(format!(
                "PI0.5 model_variant=fp8_static is unsupported on sm_{sm}; it requires sm_100 or newer and will not fall back to bf16"
            )));
        }
        Ok(self)
    }
}
impl std::str::FromStr for ModelVariantChoice {
    type Err = apxinf_core::Error;
    fn from_str(value: &str) -> apxinf_core::Result<Self> {
        match value {
            "auto" => Ok(Self::Auto), "bf16" => Ok(Self::Bf16),
            "fp8_static" => Ok(Self::Fp8Static), "int8_dynamic" => Ok(Self::Int8Dynamic),
            "int8_dynamic_no_encode" => Ok(Self::Int8DynamicNoEncode),
            _ => Err(apxinf_core::Error::Other(format!("unknown PI0.5 model_variant {value:?}; expected auto, bf16, fp8_static, int8_dynamic or int8_dynamic_no_encode"))),
        }
    }
}
#[cfg(test)]
mod model_variant_tests {
    use super::ModelVariantChoice;
    #[test]
    fn selection_and_identifiers_are_unambiguous() {
        for v in [
            ModelVariantChoice::Auto,
            ModelVariantChoice::Bf16,
            ModelVariantChoice::Fp8Static,
            ModelVariantChoice::Int8Dynamic,
            ModelVariantChoice::Int8DynamicNoEncode,
        ] {
            assert_eq!(v.as_str().parse::<ModelVariantChoice>().unwrap(), v);
        }
        for ambiguous in ["fp8", "int8", "w8a8", "unknown"] {
            assert!(ambiguous.parse::<ModelVariantChoice>().is_err());
        }
        assert_eq!(
            ModelVariantChoice::Auto.resolve(110, true),
            ModelVariantChoice::Fp8Static
        );
        assert_eq!(
            ModelVariantChoice::Auto.resolve(110, false),
            ModelVariantChoice::Bf16
        );
        assert_eq!(
            ModelVariantChoice::Auto.resolve(87, false),
            ModelVariantChoice::Int8Dynamic
        );
        assert_eq!(
            ModelVariantChoice::Bf16.resolve(110, true),
            ModelVariantChoice::Bf16
        );
        let error = ModelVariantChoice::Fp8Static
            .ensure_supported(89)
            .unwrap_err()
            .to_string();
        assert!(error.contains("unsupported on sm_89"));
        assert!(error.contains("will not fall back to bf16"));
        assert_eq!(
            ModelVariantChoice::Fp8Static.ensure_supported(110).unwrap(),
            ModelVariantChoice::Fp8Static
        );
    }
}
