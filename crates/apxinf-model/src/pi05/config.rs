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

/// Shape-stable profile used for CUDA graph capture and tactic selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pi05PerformanceProfile {
    pub num_views: usize,
    pub action_horizon: usize,
    pub num_flow_steps: usize,
}

impl Pi05PerformanceProfile {
    /// Validated Thor profile: 49.83 ms P50 for this exact shape.
    pub const THOR_TWO_VIEW: Self = Self {
        num_views: 2,
        action_horizon: 10,
        num_flow_steps: 10,
    };

    pub const THOR_THREE_VIEW: Self = Self {
        num_views: 3,
        action_horizon: 10,
        num_flow_steps: 10,
    };
}

#[derive(Clone, Debug, PartialEq)]
pub struct Pi05Config {
    pub action_dim: usize,
    pub action_horizon: usize,
    pub max_token_len: usize,
    pub num_flow_steps: usize,
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
    pub fn thor_two_view() -> Self {
        let profile = Pi05PerformanceProfile::THOR_TWO_VIEW;
        Self {
            num_views: profile.num_views,
            action_horizon: profile.action_horizon,
            num_flow_steps: profile.num_flow_steps,
            ..Self::default()
        }
    }

    pub fn thor_three_view() -> Self {
        let profile = Pi05PerformanceProfile::THOR_THREE_VIEW;
        Self {
            num_views: profile.num_views,
            action_horizon: profile.action_horizon,
            num_flow_steps: profile.num_flow_steps,
            ..Self::default()
        }
    }

    pub fn from_json_file(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| Error::Other(format!("read {}: {e}", path.display())))?;
        Self::from_json_str(&raw)
    }

    /// Parse LeRobot's policy config while accepting compatible OpenPI aliases.
    /// Architecture constants deliberately retain the validated π0.5 values
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

    /// Device arena size for one allocation-free full inference capture.
    ///
    /// The current graph intentionally gives every intermediate a stable
    /// address. A later liveness planner can reuse slots and reduce this
    /// reservation without changing the captured execution contract.
    pub fn cuda_graph_workspace_bytes(&self, token_count: usize) -> Result<usize> {
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
        allocate(patches * patch_width);
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
            allocate(patches * vision);
            allocate(patches * self.vision_mlp_dim * 2);
            allocate(patches * self.vision_mlp_dim);
            allocate(patches * vision * 2);
            allocate(patches * vision * 2);
        }
        allocate(patches * vision);
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
            if layer_index + 1 < self.language.depth {
                allocate(prefix * language_q * 2);
                allocate(prefix * self.language.num_heads * std::mem::size_of::<f32>());
                allocate(prefix * language_q);
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
            for _ in 0..self.action_expert.depth {
                allocate(horizon * action);
                allocate(horizon * action_qkv * 2);
                allocate(horizon * action_q * 2);
                allocate(horizon * action_q * 2);
                allocate(horizon * action_q);
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
            .map_err(|_| Error::Other("pi05 CUDA graph workspace exceeds address space".into()))
    }

    /// Maximum FP16 scratch operands needed to execute this FP8 graph on a
    /// CUDA device without native E4M3 GEMM (for example Jetson AGX Orin
    /// SM87). The runtime reuses one activation buffer and one weight buffer
    /// across the serialized graph, so this is bounded by the largest exact
    /// GEMM rather than the sum of all model weights.
    pub fn fp8_emulation_scratch_elements(&self, token_count: usize) -> Result<(usize, usize)> {
        if token_count == 0 || token_count > self.max_token_len {
            return Err(Error::Other(format!(
                "pi05 token count must be in 1..={}, got {token_count}",
                self.max_token_len
            )));
        }
        let patches = self
            .num_views
            .checked_mul(self.patches_per_view())
            .ok_or_else(|| Error::Other("pi05 FP8 patch count overflow".into()))?;
        let prefix = patches
            .checked_add(token_count)
            .ok_or_else(|| Error::Other("pi05 FP8 prefix length overflow".into()))?;
        let patch_width = 3usize
            .checked_mul(self.patch_size)
            .and_then(|value| value.checked_mul(self.patch_size))
            .ok_or_else(|| Error::Other("pi05 FP8 patch width overflow".into()))?;
        let language_query = self
            .language
            .num_heads
            .checked_mul(self.language.head_dim)
            .ok_or_else(|| Error::Other("pi05 FP8 language query width overflow".into()))?;
        let language_kv_twice = self
            .language
            .num_kv_heads
            .checked_mul(self.language.head_dim)
            .and_then(|kv| kv.checked_mul(2))
            .ok_or_else(|| Error::Other("pi05 FP8 language KV width overflow".into()))?;
        let language_qkv = language_query
            .checked_add(language_kv_twice)
            .ok_or_else(|| Error::Other("pi05 FP8 language QKV width overflow".into()))?;
        let action_query = self
            .action_expert
            .num_heads
            .checked_mul(self.action_expert.head_dim)
            .ok_or_else(|| Error::Other("pi05 FP8 action query width overflow".into()))?;
        let action_kv_twice = self
            .action_expert
            .num_kv_heads
            .checked_mul(self.action_expert.head_dim)
            .and_then(|kv| kv.checked_mul(2))
            .ok_or_else(|| Error::Other("pi05 FP8 action KV width overflow".into()))?;
        let action_qkv = action_query
            .checked_add(action_kv_twice)
            .ok_or_else(|| Error::Other("pi05 FP8 action QKV width overflow".into()))?;
        let vision_qkv = self
            .vision_width
            .checked_mul(3)
            .ok_or_else(|| Error::Other("pi05 FP8 vision QKV width overflow".into()))?;
        let language_gate_up = self
            .language
            .mlp_dim
            .checked_mul(2)
            .ok_or_else(|| Error::Other("pi05 FP8 language gate/up width overflow".into()))?;
        let action_gate_up = self
            .action_expert
            .mlp_dim
            .checked_mul(2)
            .ok_or_else(|| Error::Other("pi05 FP8 action gate/up width overflow".into()))?;
        let product = |left: usize, right: usize| {
            left.checked_mul(right)
                .ok_or_else(|| Error::Other("pi05 FP8 scratch shape overflow".into()))
        };

        let activation_elements = [
            product(patches, patch_width)?,
            product(patches, self.vision_width)?,
            product(patches, self.vision_mlp_dim)?,
            product(prefix, self.language.width)?,
            product(prefix, self.language.mlp_dim)?,
            product(self.action_horizon, self.action_dim)?,
            product(self.action_horizon, self.action_expert.width)?,
            product(self.action_horizon, self.action_expert.mlp_dim)?,
        ]
        .into_iter()
        .max()
        .unwrap();
        let weight_elements = [
            product(patch_width, self.vision_width)?,
            product(self.vision_width, vision_qkv)?,
            product(self.vision_width, self.vision_width)?,
            product(self.vision_width, self.vision_mlp_dim)?,
            product(self.vision_mlp_dim, self.vision_width)?,
            product(self.vision_width, self.language.width)?,
            product(self.language.width, language_qkv)?,
            product(self.language.width, self.language.width)?,
            product(self.language.width, language_gate_up)?,
            product(self.language.mlp_dim, self.language.width)?,
            product(self.action_dim, self.action_expert.width)?,
            product(self.action_expert.width, action_qkv)?,
            product(self.action_expert.width, self.action_expert.width)?,
            product(self.action_expert.width, action_gate_up)?,
            product(self.action_expert.mlp_dim, self.action_expert.width)?,
        ]
        .into_iter()
        .max()
        .unwrap();
        Ok((activation_elements, weight_elements))
    }

    /// Conservative arena reservation for the native-BF16 graph.
    ///
    /// The FP8 schedule mixes one-byte quantized intermediates with two-byte
    /// residuals. The BF16 schedule stores every intermediate in two bytes;
    /// doubling the validated FP8 bound is therefore safe while keeping the
    /// shape calculation in one place.
    pub fn cuda_graph_workspace_bytes_bf16(&self, token_count: usize) -> Result<usize> {
        self.cuda_graph_workspace_bytes(token_count)?
            .checked_mul(2)
            .ok_or_else(|| Error::Other("pi05 BF16 CUDA workspace exceeds address space".into()))
    }

    /// Conservative reservation for the first W8A8 implementation.
    ///
    /// Each linear stores a one-byte quantized activation and a two-byte BF16
    /// output. Aligned SM80-family GEMMs scale directly into BF16; the
    /// unaligned patch projection additionally stores one INT32 accumulator.
    /// Twice the BF16 arena remains a simple conservative upper bound and also
    /// leaves room for the optional dual-backend correctness verifier.
    pub fn cuda_graph_workspace_bytes_int8(&self, token_count: usize) -> Result<usize> {
        self.cuda_graph_workspace_bytes_bf16(token_count)?
            .checked_mul(2)
            .ok_or_else(|| Error::Other("pi05 INT8 CUDA workspace exceeds address space".into()))
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
                ..Pi05Config::thor_two_view()
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
    fn thor_graph_workspace_is_bounded() {
        let bytes = Pi05Config::thor_two_view()
            .cuda_graph_workspace_bytes(200)
            .unwrap();
        assert!(bytes > 1_800_000_000);
        assert!(bytes < 2_500_000_000);
    }

    #[test]
    fn orin_fp8_emulation_scratch_is_largest_exact_gemm() {
        let config = Pi05Config::thor_two_view();
        let (activation, weight) = config.fp8_emulation_scratch_elements(21).unwrap();
        assert_eq!(activation, 533 * 16_384);
        assert_eq!(weight, 2_048 * 32_768);

        let (max_activation, max_weight) = config.fp8_emulation_scratch_elements(200).unwrap();
        assert_eq!(max_activation, 712 * 16_384);
        assert_eq!(max_weight, weight);
    }
}
