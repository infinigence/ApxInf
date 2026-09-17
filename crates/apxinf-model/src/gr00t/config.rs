use std::path::Path;

use apxinf_core::{Error, Result};
use serde_json::{Map, Value};

const MODEL_TYPE: &str = "Gr00tN1d7";
const BF16: &str = "bfloat16";

/// Transformer configuration for the GR00T N1.7 diffusion action head.
#[derive(Clone, Debug, PartialEq)]
pub struct Gr00tDiffusionConfig {
    pub positional_embeddings: Option<String>,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub attention_head_dim: usize,
    pub norm_type: String,
    pub dropout: f32,
    pub final_dropout: bool,
    pub output_dim: usize,
    pub interleave_self_attention: bool,
    pub activation_fn: String,
    pub attention_bias: bool,
    pub upcast_attention: bool,
    pub norm_elementwise_affine: bool,
    pub norm_eps: f32,
    pub max_num_positional_embeddings: usize,
}

impl Default for Gr00tDiffusionConfig {
    fn default() -> Self {
        Self {
            positional_embeddings: None,
            num_layers: 16,
            num_attention_heads: 32,
            attention_head_dim: 48,
            norm_type: "ada_norm".into(),
            dropout: 0.2,
            final_dropout: true,
            output_dim: 1024,
            interleave_self_attention: true,
            // This default comes from DiT.__init__, not BasicTransformerBlock.
            activation_fn: "gelu-approximate".into(),
            attention_bias: true,
            upcast_attention: false,
            norm_elementwise_affine: false,
            norm_eps: 1e-5,
            max_num_positional_embeddings: 512,
        }
    }
}

impl Gr00tDiffusionConfig {
    pub fn inner_dim(&self) -> Result<usize> {
        self.num_attention_heads
            .checked_mul(self.attention_head_dim)
            .ok_or_else(|| Error::Other("GR00T diffusion inner dimension overflow".into()))
    }

    fn update_from_json(&mut self, value: &Value) -> Result<()> {
        let object = object(value, "diffusion_model_cfg")?;
        self.positional_embeddings = optional_string(
            object,
            "positional_embeddings",
            self.positional_embeddings.clone(),
        )?;
        self.num_layers = usize_value(object, "num_layers", self.num_layers)?;
        self.num_attention_heads =
            usize_value(object, "num_attention_heads", self.num_attention_heads)?;
        self.attention_head_dim =
            usize_value(object, "attention_head_dim", self.attention_head_dim)?;
        self.norm_type = string_value(object, "norm_type", &self.norm_type)?;
        self.dropout = f32_value(object, "dropout", self.dropout)?;
        self.final_dropout = bool_value(object, "final_dropout", self.final_dropout)?;
        self.output_dim = usize_value(object, "output_dim", self.output_dim)?;
        self.interleave_self_attention = bool_value(
            object,
            "interleave_self_attention",
            self.interleave_self_attention,
        )?;
        self.activation_fn = string_value(object, "activation_fn", &self.activation_fn)?;
        self.attention_bias = bool_value(object, "attention_bias", self.attention_bias)?;
        self.upcast_attention = bool_value(object, "upcast_attention", self.upcast_attention)?;
        self.norm_elementwise_affine = bool_value(
            object,
            "norm_elementwise_affine",
            self.norm_elementwise_affine,
        )?;
        self.norm_eps = f32_value(object, "norm_eps", self.norm_eps)?;
        self.max_num_positional_embeddings = usize_value(
            object,
            "max_num_positional_embeddings",
            self.max_num_positional_embeddings,
        )?;
        Ok(())
    }
}

/// Optional self-attention stack applied to backbone features before DiT.
#[derive(Clone, Debug, PartialEq)]
pub struct Gr00tVlSelfAttentionConfig {
    pub positional_embeddings: Option<String>,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub attention_head_dim: usize,
    pub dropout: f32,
    pub final_dropout: bool,
    pub activation_fn: String,
    pub attention_bias: bool,
    pub upcast_attention: bool,
}

impl Default for Gr00tVlSelfAttentionConfig {
    fn default() -> Self {
        Self {
            positional_embeddings: None,
            num_layers: 0,
            num_attention_heads: 32,
            attention_head_dim: 64,
            dropout: 0.2,
            final_dropout: true,
            activation_fn: "gelu-approximate".into(),
            attention_bias: true,
            upcast_attention: false,
        }
    }
}

impl Gr00tVlSelfAttentionConfig {
    pub fn hidden_size(&self) -> Result<usize> {
        self.num_attention_heads
            .checked_mul(self.attention_head_dim)
            .ok_or_else(|| Error::Other("GR00T VL self-attention width overflow".into()))
    }

    fn from_json(value: &Value) -> Result<Self> {
        let object = object(value, "vl_self_attention_cfg")?;
        let mut config = Self::default();
        config.positional_embeddings = optional_string(
            object,
            "positional_embeddings",
            config.positional_embeddings.clone(),
        )?;
        config.num_layers = usize_value(object, "num_layers", config.num_layers)?;
        config.num_attention_heads =
            usize_value(object, "num_attention_heads", config.num_attention_heads)?;
        config.attention_head_dim =
            usize_value(object, "attention_head_dim", config.attention_head_dim)?;
        config.dropout = f32_value(object, "dropout", config.dropout)?;
        config.final_dropout = bool_value(object, "final_dropout", config.final_dropout)?;
        config.activation_fn = string_value(object, "activation_fn", &config.activation_fn)?;
        config.attention_bias = bool_value(object, "attention_bias", config.attention_bias)?;
        config.upcast_attention = bool_value(object, "upcast_attention", config.upcast_attention)?;
        Ok(config)
    }
}

/// Inference-relevant GR00T N1.7 model configuration.
///
/// NVIDIA checkpoints omit fields whose values come from Python constructor
/// defaults. Parsing therefore starts from the N1.7 defaults and applies every
/// serialized override. Runtime code must use this parsed value rather than
/// assuming the defaults: the released 3B checkpoint uses a deeper backbone
/// and action head than the source defaults.
#[derive(Clone, Debug, PartialEq)]
pub struct Gr00tConfig {
    pub model_type: String,
    pub model_dtype: String,
    pub model_name: String,
    pub backbone_model_type: String,
    pub backbone_embedding_dim: usize,
    pub select_layer: usize,
    pub reproject_vision: bool,
    pub use_flash_attention: bool,
    pub image_crop_size: Option<[usize; 2]>,
    pub image_target_size: Option<[usize; 2]>,
    pub max_state_dim: usize,
    pub max_action_dim: usize,
    pub action_horizon: usize,
    pub hidden_size: usize,
    pub input_embedding_dim: usize,
    pub state_history_length: usize,
    pub add_pos_embed: bool,
    pub use_vlln: bool,
    pub max_seq_len: usize,
    pub use_alternate_vl_dit: bool,
    pub attend_text_every_n_blocks: usize,
    pub diffusion: Gr00tDiffusionConfig,
    pub use_vl_self_attention: bool,
    pub vl_self_attention: Option<Gr00tVlSelfAttentionConfig>,
    pub num_inference_timesteps: usize,
    pub num_timestep_buckets: usize,
    pub max_num_embodiments: usize,
}

impl Default for Gr00tConfig {
    fn default() -> Self {
        Self {
            model_type: MODEL_TYPE.into(),
            model_dtype: BF16.into(),
            model_name: "nvidia/Cosmos-Reason2-2B".into(),
            backbone_model_type: "qwen".into(),
            backbone_embedding_dim: 2048,
            select_layer: 12,
            reproject_vision: false,
            use_flash_attention: true,
            image_crop_size: Some([230, 230]),
            image_target_size: Some([256, 256]),
            max_state_dim: 132,
            max_action_dim: 132,
            action_horizon: 40,
            hidden_size: 1024,
            input_embedding_dim: 1536,
            state_history_length: 1,
            add_pos_embed: true,
            use_vlln: true,
            max_seq_len: 1024,
            use_alternate_vl_dit: true,
            attend_text_every_n_blocks: 2,
            diffusion: Gr00tDiffusionConfig::default(),
            use_vl_self_attention: false,
            vl_self_attention: None,
            num_inference_timesteps: 4,
            num_timestep_buckets: 1000,
            max_num_embodiments: 32,
        }
    }
}

impl Gr00tConfig {
    pub fn from_json_file(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|error| Error::Other(format!("read {}: {error}", path.display())))?;
        Self::from_json_str(&raw)
    }

    pub fn from_json_str(raw: &str) -> Result<Self> {
        let value: Value = serde_json::from_str(raw)
            .map_err(|error| Error::Other(format!("GR00T config JSON: {error}")))?;
        let object = object(&value, "GR00T config")?;
        let mut config = Self::default();

        config.model_type = string_value(object, "model_type", &config.model_type)?;
        config.model_dtype =
            string_alias_value(object, &["model_dtype", "dtype"], &config.model_dtype)?;
        config.model_name = string_value(object, "model_name", &config.model_name)?;
        config.backbone_model_type =
            string_value(object, "backbone_model_type", &config.backbone_model_type)?;
        config.backbone_embedding_dim = usize_value(
            object,
            "backbone_embedding_dim",
            config.backbone_embedding_dim,
        )?;
        config.select_layer = usize_value(object, "select_layer", config.select_layer)?;
        config.reproject_vision = bool_value(object, "reproject_vision", config.reproject_vision)?;
        config.use_flash_attention =
            bool_value(object, "use_flash_attention", config.use_flash_attention)?;
        config.image_crop_size = optional_pair(object, "image_crop_size", config.image_crop_size)?;
        config.image_target_size =
            optional_pair(object, "image_target_size", config.image_target_size)?;
        config.max_state_dim = usize_value(object, "max_state_dim", config.max_state_dim)?;
        config.max_action_dim = usize_value(object, "max_action_dim", config.max_action_dim)?;
        config.action_horizon = usize_value(object, "action_horizon", config.action_horizon)?;
        config.hidden_size = usize_value(object, "hidden_size", config.hidden_size)?;
        config.input_embedding_dim =
            usize_value(object, "input_embedding_dim", config.input_embedding_dim)?;
        config.state_history_length =
            usize_value(object, "state_history_length", config.state_history_length)?;
        config.add_pos_embed = bool_value(object, "add_pos_embed", config.add_pos_embed)?;
        config.use_vlln = bool_value(object, "use_vlln", config.use_vlln)?;
        config.max_seq_len = usize_value(object, "max_seq_len", config.max_seq_len)?;
        config.use_alternate_vl_dit =
            bool_value(object, "use_alternate_vl_dit", config.use_alternate_vl_dit)?;
        config.attend_text_every_n_blocks = usize_value(
            object,
            "attend_text_every_n_blocks",
            config.attend_text_every_n_blocks,
        )?;
        if let Some(value) = object.get("diffusion_model_cfg") {
            config.diffusion.update_from_json(value)?;
        }
        config.use_vl_self_attention = bool_value(
            object,
            "use_vl_self_attention",
            config.use_vl_self_attention,
        )?;
        if let Some(value) = object.get("vl_self_attention_cfg") {
            config.vl_self_attention = Some(Gr00tVlSelfAttentionConfig::from_json(value)?);
        }
        config.num_inference_timesteps = usize_value(
            object,
            "num_inference_timesteps",
            config.num_inference_timesteps,
        )?;
        config.num_timestep_buckets =
            usize_value(object, "num_timestep_buckets", config.num_timestep_buckets)?;
        config.max_num_embodiments =
            usize_value(object, "max_num_embodiments", config.max_num_embodiments)?;

        config.validate()?;
        Ok(config)
    }

    pub fn state_input_dim(&self) -> Result<usize> {
        self.max_state_dim
            .checked_mul(self.state_history_length)
            .ok_or_else(|| Error::Other("GR00T state input dimension overflow".into()))
    }

    pub fn state_action_sequence_len(&self) -> Result<usize> {
        self.action_horizon
            .checked_add(1)
            .ok_or_else(|| Error::Other("GR00T state/action sequence length overflow".into()))
    }

    pub fn validate(&self) -> Result<()> {
        if self.model_type != MODEL_TYPE {
            return Err(Error::Other(format!(
                "expected GR00T model_type {MODEL_TYPE}, got {}",
                self.model_type
            )));
        }
        if self.model_dtype != BF16 {
            return Err(Error::Other(format!(
                "GR00T N1.7 currently supports bfloat16, got {}",
                self.model_dtype
            )));
        }
        if self.backbone_model_type != "qwen" {
            return Err(Error::Other(format!(
                "GR00T N1.7 expects qwen backbone, got {}",
                self.backbone_model_type
            )));
        }
        let positive = [
            ("backbone_embedding_dim", self.backbone_embedding_dim),
            ("select_layer", self.select_layer),
            ("max_state_dim", self.max_state_dim),
            ("max_action_dim", self.max_action_dim),
            ("action_horizon", self.action_horizon),
            ("hidden_size", self.hidden_size),
            ("input_embedding_dim", self.input_embedding_dim),
            ("state_history_length", self.state_history_length),
            ("max_seq_len", self.max_seq_len),
            ("diffusion.num_layers", self.diffusion.num_layers),
            (
                "diffusion.num_attention_heads",
                self.diffusion.num_attention_heads,
            ),
            (
                "diffusion.attention_head_dim",
                self.diffusion.attention_head_dim,
            ),
            ("num_inference_timesteps", self.num_inference_timesteps),
            ("num_timestep_buckets", self.num_timestep_buckets),
            ("max_num_embodiments", self.max_num_embodiments),
        ];
        if let Some((name, _)) = positive.into_iter().find(|(_, value)| *value == 0) {
            return Err(Error::Other(format!("GR00T {name} must be non-zero")));
        }
        if self.diffusion.inner_dim()? != self.input_embedding_dim {
            return Err(Error::Other(format!(
                "GR00T diffusion width {}x{} does not match input_embedding_dim {}",
                self.diffusion.num_attention_heads,
                self.diffusion.attention_head_dim,
                self.input_embedding_dim
            )));
        }
        if self.diffusion.output_dim != self.hidden_size {
            return Err(Error::Other(format!(
                "GR00T diffusion output_dim {} does not match hidden_size {}",
                self.diffusion.output_dim, self.hidden_size
            )));
        }
        if self.diffusion.norm_type != "ada_norm" {
            return Err(Error::Other(format!(
                "GR00T N1.7 requires ada_norm, got {}",
                self.diffusion.norm_type
            )));
        }
        if self.diffusion.activation_fn != "gelu-approximate" {
            return Err(Error::Other(format!(
                "GR00T N1.7 BF16 runtime requires gelu-approximate, got {}",
                self.diffusion.activation_fn
            )));
        }
        if self.state_action_sequence_len()? > self.max_seq_len {
            return Err(Error::Other(format!(
                "GR00T state/action sequence length {} exceeds max_seq_len {}",
                self.state_action_sequence_len()?,
                self.max_seq_len
            )));
        }
        if self.use_alternate_vl_dit {
            if !self.diffusion.interleave_self_attention {
                return Err(Error::Other(
                    "GR00T AlternateVLDiT requires interleaved self-attention".into(),
                ));
            }
            if self.attend_text_every_n_blocks == 0 {
                return Err(Error::Other(
                    "GR00T attend_text_every_n_blocks must be non-zero".into(),
                ));
            }
        }
        if self.use_vl_self_attention {
            let vl = self.vl_self_attention.as_ref().ok_or_else(|| {
                Error::Other("GR00T enables VL self-attention without vl_self_attention_cfg".into())
            })?;
            if vl.num_layers == 0 {
                return Err(Error::Other(
                    "GR00T VL self-attention must contain at least one layer".into(),
                ));
            }
            if vl.hidden_size()? != self.backbone_embedding_dim {
                return Err(Error::Other(format!(
                    "GR00T VL self-attention width {} does not match backbone width {}",
                    vl.hidden_size()?,
                    self.backbone_embedding_dim
                )));
            }
        }
        Ok(())
    }
}

fn object<'a>(value: &'a Value, name: &str) -> Result<&'a Map<String, Value>> {
    value
        .as_object()
        .ok_or_else(|| Error::Other(format!("{name} must be a JSON object")))
}

fn usize_value(object: &Map<String, Value>, key: &str, default: usize) -> Result<usize> {
    match object.get(key) {
        None => Ok(default),
        Some(value) => value
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| Error::Other(format!("GR00T {key} must be a non-negative integer"))),
    }
}

fn f32_value(object: &Map<String, Value>, key: &str, default: f32) -> Result<f32> {
    match object.get(key) {
        None => Ok(default),
        Some(value) => value
            .as_f64()
            .map(|value| value as f32)
            .filter(|value| value.is_finite())
            .ok_or_else(|| Error::Other(format!("GR00T {key} must be finite"))),
    }
}

fn bool_value(object: &Map<String, Value>, key: &str, default: bool) -> Result<bool> {
    match object.get(key) {
        None => Ok(default),
        Some(value) => value
            .as_bool()
            .ok_or_else(|| Error::Other(format!("GR00T {key} must be a boolean"))),
    }
}

fn string_value(object: &Map<String, Value>, key: &str, default: &str) -> Result<String> {
    match object.get(key) {
        None => Ok(default.into()),
        Some(value) => value
            .as_str()
            .map(str::to_owned)
            .ok_or_else(|| Error::Other(format!("GR00T {key} must be a string"))),
    }
}

fn string_alias_value(object: &Map<String, Value>, keys: &[&str], default: &str) -> Result<String> {
    for key in keys {
        if object.contains_key(*key) {
            return string_value(object, key, default);
        }
    }
    Ok(default.into())
}

fn optional_string(
    object: &Map<String, Value>,
    key: &str,
    default: Option<String>,
) -> Result<Option<String>> {
    match object.get(key) {
        None => Ok(default),
        Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_str()
            .map(|value| Some(value.to_owned()))
            .ok_or_else(|| Error::Other(format!("GR00T {key} must be a string or null"))),
    }
}

fn optional_pair(
    object: &Map<String, Value>,
    key: &str,
    default: Option<[usize; 2]>,
) -> Result<Option<[usize; 2]>> {
    let Some(value) = object.get(key) else {
        return Ok(default);
    };
    if value.is_null() {
        return Ok(None);
    }
    let values = value
        .as_array()
        .filter(|values| values.len() == 2)
        .ok_or_else(|| Error::Other(format!("GR00T {key} must contain exactly two integers")))?;
    let first = values[0]
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| Error::Other(format!("GR00T {key}[0] must be an integer")))?;
    let second = values[1]
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| Error::Other(format!("GR00T {key}[1] must be an integer")))?;
    Ok(Some([first, second]))
}

#[cfg(test)]
mod tests {
    use super::*;

    const RELEASED_3B_CONFIG: &str = r#"
    {
      "model_type": "Gr00tN1d7",
      "dtype": "bfloat16",
      "model_dtype": "bfloat16",
      "model_name": "nvidia/Cosmos-Reason2-2B",
      "backbone_embedding_dim": 2048,
      "select_layer": 16,
      "max_state_dim": 132,
      "max_action_dim": 132,
      "action_horizon": 40,
      "hidden_size": 1024,
      "max_seq_len": 1024,
      "use_alternate_vl_dit": true,
      "use_vl_self_attention": true,
      "vl_self_attention_cfg": {
        "num_layers": 4,
        "num_attention_heads": 32,
        "attention_head_dim": 64,
        "dropout": 0.2,
        "final_dropout": true,
        "positional_embeddings": null
      },
      "diffusion_model_cfg": {
        "num_layers": 32,
        "num_attention_heads": 32,
        "attention_head_dim": 48,
        "norm_type": "ada_norm",
        "dropout": 0.2,
        "final_dropout": true,
        "output_dim": 1024,
        "interleave_self_attention": true,
        "positional_embeddings": null
      },
      "num_inference_timesteps": 4,
      "num_timestep_buckets": 1000,
      "max_num_embodiments": 32
    }
    "#;

    #[test]
    fn parses_released_3b_overrides_and_constructor_defaults() {
        let config = Gr00tConfig::from_json_str(RELEASED_3B_CONFIG).unwrap();
        assert_eq!(config.select_layer, 16);
        assert_eq!(config.diffusion.num_layers, 32);
        assert_eq!(config.diffusion.activation_fn, "gelu-approximate");
        assert_eq!(config.input_embedding_dim, 1536);
        assert_eq!(config.state_history_length, 1);
        assert_eq!(config.attend_text_every_n_blocks, 2);
        assert_eq!(config.diffusion.inner_dim().unwrap(), 1536);
        assert_eq!(config.vl_self_attention.unwrap().num_layers, 4);
    }

    #[test]
    fn rejects_diffusion_width_mismatch() {
        let raw =
            RELEASED_3B_CONFIG.replace("\"attention_head_dim\": 48", "\"attention_head_dim\": 32");
        let error = Gr00tConfig::from_json_str(&raw).unwrap_err();
        assert!(error
            .to_string()
            .contains("does not match input_embedding_dim"));
    }

    #[test]
    fn rejects_wrong_model_and_dtype() {
        let wrong_model = RELEASED_3B_CONFIG.replace("Gr00tN1d7", "Gr00tN1d6");
        assert!(Gr00tConfig::from_json_str(&wrong_model).is_err());

        let wrong_dtype = RELEASED_3B_CONFIG.replace("bfloat16", "float16");
        assert!(Gr00tConfig::from_json_str(&wrong_dtype).is_err());
    }
}
