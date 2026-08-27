use std::path::Path;

use apxinf_core::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen35LayerType {
    LinearAttention,
    FullAttention,
}

impl Qwen35LayerType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LinearAttention => "linear_attention",
            Self::FullAttention => "full_attention",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Qwen35TextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub full_attention_interval: usize,
    pub layer_types: Vec<Qwen35LayerType>,
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_value_head_dim: usize,
    pub partial_rotary_factor: f32,
    pub rope_theta: f32,
    pub mrope_section: [usize; 3],
    pub mrope_interleaved: bool,
    pub attn_output_gate: bool,
    pub output_gate_type: String,
    pub mtp_num_hidden_layers: usize,
}

#[derive(Clone, Debug)]
pub struct Qwen35VisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub in_channels: usize,
    pub num_position_embeddings: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub spatial_merge_size: usize,
    pub out_hidden_size: usize,
    pub deepstack_visual_indexes: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct Qwen35QuantizationConfig {
    pub method: String,
    pub format: String,
    pub weight_type: String,
    pub num_bits: usize,
    pub group_size: usize,
    pub strategy: String,
    pub symmetric: bool,
    pub dynamic: bool,
    pub target_linear: bool,
    pub ignored_modules: Vec<String>,
}

impl Qwen35QuantizationConfig {
    pub fn pack_factor(&self) -> Result<usize> {
        if self.num_bits == 0 || 32 % self.num_bits != 0 {
            return Err(Error::Other(format!(
                "qwen3.5 quantization: {} bits cannot be packed into i32",
                self.num_bits
            )));
        }
        Ok(32 / self.num_bits)
    }
}

#[derive(Clone, Debug)]
pub struct Qwen35Config {
    pub architecture: String,
    pub model_type: String,
    pub language_model_only: bool,
    pub text: Qwen35TextConfig,
    pub vision: Qwen35VisionConfig,
    pub quantization: Qwen35QuantizationConfig,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
}

impl Qwen35Config {
    pub fn from_json_file(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|error| Error::Other(format!("read {}: {error}", path.display())))?;
        Self::from_json_str(&raw)
    }

    pub fn from_json_str(raw: &str) -> Result<Self> {
        let root_value: serde_json::Value = serde_json::from_str(raw)
            .map_err(|error| Error::Other(format!("qwen3.5 config json: {error}")))?;
        let root = root_value
            .as_object()
            .ok_or_else(|| Error::Other("qwen3.5 config root must be an object".into()))?;
        let architecture = required_array(root, "architectures")?
            .first()
            .and_then(|value| value.as_str())
            .ok_or_else(|| {
                Error::Other("qwen3.5 config: architectures[0] must be a string".into())
            })?
            .to_owned();
        let text_value = required_object(&root, "text_config")?;
        let vision_value = required_object(&root, "vision_config")?;
        let quant_value = required_object(&root, "quantization_config")?;
        let rope_value = required_object(text_value, "rope_parameters")?;

        let layer_types = required_array(text_value, "layer_types")?
            .iter()
            .enumerate()
            .map(|(index, value)| match value.as_str() {
                Some("linear_attention") => Ok(Qwen35LayerType::LinearAttention),
                Some("full_attention") => Ok(Qwen35LayerType::FullAttention),
                Some(other) => Err(Error::Other(format!(
                    "qwen3.5 config: unsupported layer_types[{index}] `{other}`"
                ))),
                None => Err(Error::Other(format!(
                    "qwen3.5 config: layer_types[{index}] must be a string"
                ))),
            })
            .collect::<Result<Vec<_>>>()?;
        let mrope = required_array(rope_value, "mrope_section")?;
        if mrope.len() != 3 {
            return Err(Error::Other(format!(
                "qwen3.5 config: rope_parameters.mrope_section must have 3 entries, got {}",
                mrope.len()
            )));
        }

        let text = Qwen35TextConfig {
            hidden_size: required_usize(text_value, "hidden_size")?,
            intermediate_size: required_usize(text_value, "intermediate_size")?,
            n_layers: required_usize(text_value, "num_hidden_layers")?,
            n_heads: required_usize(text_value, "num_attention_heads")?,
            n_kv_heads: required_usize(text_value, "num_key_value_heads")?,
            head_dim: required_usize(text_value, "head_dim")?,
            vocab_size: required_usize(text_value, "vocab_size")?,
            max_position_embeddings: required_usize(text_value, "max_position_embeddings")?,
            rms_norm_eps: required_f32(text_value, "rms_norm_eps")?,
            full_attention_interval: required_usize(text_value, "full_attention_interval")?,
            layer_types,
            linear_conv_kernel_dim: required_usize(text_value, "linear_conv_kernel_dim")?,
            linear_key_head_dim: required_usize(text_value, "linear_key_head_dim")?,
            linear_num_key_heads: required_usize(text_value, "linear_num_key_heads")?,
            linear_num_value_heads: required_usize(text_value, "linear_num_value_heads")?,
            linear_value_head_dim: required_usize(text_value, "linear_value_head_dim")?,
            partial_rotary_factor: required_f32(text_value, "partial_rotary_factor")?,
            rope_theta: required_f32(rope_value, "rope_theta")?,
            mrope_section: [
                value_as_usize(&mrope[0], "mrope_section[0]")?,
                value_as_usize(&mrope[1], "mrope_section[1]")?,
                value_as_usize(&mrope[2], "mrope_section[2]")?,
            ],
            mrope_interleaved: required_bool(rope_value, "mrope_interleaved")?,
            attn_output_gate: required_bool(text_value, "attn_output_gate")?,
            output_gate_type: required_string(text_value, "output_gate_type")?,
            mtp_num_hidden_layers: required_usize(text_value, "mtp_num_hidden_layers")?,
        };

        let groups = required_object(quant_value, "config_groups")?;
        if groups.len() != 1 {
            return Err(Error::Other(format!(
                "qwen3.5 quantization: expected one config group, got {}",
                groups.len()
            )));
        }
        let group = groups
            .values()
            .next()
            .and_then(|value| value.as_object())
            .ok_or_else(|| {
                Error::Other("qwen3.5 quantization: config group must be an object".into())
            })?;
        if !group
            .get("input_activations")
            .is_some_and(serde_json::Value::is_null)
            || !group
                .get("output_activations")
                .is_some_and(serde_json::Value::is_null)
        {
            return Err(Error::Other(
                "qwen3.5 quantization: activation quantization is unsupported".into(),
            ));
        }
        let weights = required_object(group, "weights")?;
        let targets = required_array(group, "targets")?;
        let ignored_modules = required_array(quant_value, "ignore")?
            .iter()
            .map(|value| {
                value.as_str().map(str::to_owned).ok_or_else(|| {
                    Error::Other("qwen3.5 quantization: ignore entries must be strings".into())
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let quantization = Qwen35QuantizationConfig {
            method: required_string(quant_value, "quant_method")?,
            format: required_string(quant_value, "format")?,
            weight_type: required_string(weights, "type")?,
            num_bits: required_usize(weights, "num_bits")?,
            group_size: required_usize(weights, "group_size")?,
            strategy: required_string(weights, "strategy")?,
            symmetric: required_bool(weights, "symmetric")?,
            dynamic: required_bool(weights, "dynamic")?,
            target_linear: targets.iter().any(|value| value.as_str() == Some("Linear")),
            ignored_modules,
        };

        let config = Self {
            architecture,
            model_type: required_string(root, "model_type")?,
            language_model_only: required_bool(root, "language_model_only")?,
            text,
            vision: Qwen35VisionConfig {
                depth: required_usize(vision_value, "depth")?,
                hidden_size: required_usize(vision_value, "hidden_size")?,
                intermediate_size: required_usize(vision_value, "intermediate_size")?,
                num_heads: required_usize(vision_value, "num_heads")?,
                in_channels: required_usize(vision_value, "in_channels")?,
                num_position_embeddings: required_usize(vision_value, "num_position_embeddings")?,
                patch_size: required_usize(vision_value, "patch_size")?,
                temporal_patch_size: required_usize(vision_value, "temporal_patch_size")?,
                spatial_merge_size: required_usize(vision_value, "spatial_merge_size")?,
                out_hidden_size: required_usize(vision_value, "out_hidden_size")?,
                deepstack_visual_indexes: required_array(vision_value, "deepstack_visual_indexes")?
                    .iter()
                    .enumerate()
                    .map(|(index, value)| {
                        value_as_usize(value, &format!("deepstack_visual_indexes[{index}]"))
                    })
                    .collect::<Result<Vec<_>>>()?,
            },
            quantization,
            image_token_id: required_u32(root, "image_token_id")?,
            video_token_id: required_u32(root, "video_token_id")?,
            vision_start_token_id: required_u32(root, "vision_start_token_id")?,
            vision_end_token_id: required_u32(root, "vision_end_token_id")?,
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        if self.model_type != "qwen3_5" || self.architecture != "Qwen3_5ForConditionalGeneration" {
            return Err(Error::Other(format!(
                "qwen3.5 config identity mismatch: model_type=`{}`, architecture=`{}`",
                self.model_type, self.architecture
            )));
        }
        if self.language_model_only {
            return Err(Error::Other(
                "qwen3.5 config declares language_model_only=true; multimodal contract required"
                    .into(),
            ));
        }
        if self.text.layer_types.len() != self.text.n_layers {
            return Err(Error::Other(format!(
                "qwen3.5 config: {} layer types for {} hidden layers",
                self.text.layer_types.len(),
                self.text.n_layers
            )));
        }
        if self.text.full_attention_interval == 0 {
            return Err(Error::Other(
                "qwen3.5 config: full_attention_interval must be nonzero".into(),
            ));
        }
        for (index, layer_type) in self.text.layer_types.iter().enumerate() {
            let expected = if (index + 1) % self.text.full_attention_interval == 0 {
                Qwen35LayerType::FullAttention
            } else {
                Qwen35LayerType::LinearAttention
            };
            if *layer_type != expected {
                return Err(Error::Other(format!(
                    "qwen3.5 config: layer {index} is {}, expected {} from full_attention_interval={}",
                    layer_type.as_str(), expected.as_str(), self.text.full_attention_interval
                )));
            }
        }
        let rotary_pairs =
            (self.text.head_dim as f32 * self.text.partial_rotary_factor / 2.0) as usize;
        if self.text.mrope_section.iter().sum::<usize>() != rotary_pairs {
            return Err(Error::Other(format!(
                "qwen3.5 config: mrope sections {:?} do not cover {rotary_pairs} rotary pairs",
                self.text.mrope_section
            )));
        }
        let quant = &self.quantization;
        if quant.method != "compressed-tensors"
            || quant.format != "pack-quantized"
            || quant.weight_type != "int"
            || quant.num_bits != 4
            || quant.group_size != 32
            || quant.strategy != "group"
            || quant.symmetric
            || quant.dynamic
            || !quant.target_linear
        {
            return Err(Error::Other(format!(
                "qwen3.5 quantization must be compressed-tensors pack-quantized W4A16 group32 asymmetric; got method={}, format={}, type={}, bits={}, group={}, strategy={}, symmetric={}, dynamic={}, target_linear={}",
                quant.method,
                quant.format,
                quant.weight_type,
                quant.num_bits,
                quant.group_size,
                quant.strategy,
                quant.symmetric,
                quant.dynamic,
                quant.target_linear
            )));
        }
        quant.pack_factor()?;
        Ok(())
    }
}

type JsonObject = serde_json::Map<String, serde_json::Value>;

fn required_object<'a>(value: &'a JsonObject, key: &str) -> Result<&'a JsonObject> {
    value
        .get(key)
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| Error::Other(format!("qwen3.5 config: missing object `{key}`")))
}

fn required_array<'a>(value: &'a JsonObject, key: &str) -> Result<&'a Vec<serde_json::Value>> {
    value
        .get(key)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| Error::Other(format!("qwen3.5 config: missing array `{key}`")))
}

fn required_string(value: &JsonObject, key: &str) -> Result<String> {
    value
        .get(key)
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| Error::Other(format!("qwen3.5 config: missing string `{key}`")))
}

fn required_bool(value: &JsonObject, key: &str) -> Result<bool> {
    value
        .get(key)
        .and_then(serde_json::Value::as_bool)
        .ok_or_else(|| Error::Other(format!("qwen3.5 config: missing bool `{key}`")))
}

fn required_usize(value: &JsonObject, key: &str) -> Result<usize> {
    value
        .get(key)
        .ok_or_else(|| Error::Other(format!("qwen3.5 config: missing integer `{key}`")))
        .and_then(|value| value_as_usize(value, key))
}

fn required_u32(value: &JsonObject, key: &str) -> Result<u32> {
    let number = required_usize(value, key)?;
    u32::try_from(number).map_err(|_| Error::Other(format!("qwen3.5 config: `{key}` exceeds u32")))
}

fn value_as_usize(value: &serde_json::Value, label: &str) -> Result<usize> {
    let number = value.as_u64().ok_or_else(|| {
        Error::Other(format!(
            "qwen3.5 config: `{label}` must be a nonnegative integer"
        ))
    })?;
    usize::try_from(number)
        .map_err(|_| Error::Other(format!("qwen3.5 config: `{label}` exceeds usize")))
}

fn required_f32(value: &JsonObject, key: &str) -> Result<f32> {
    value
        .get(key)
        .and_then(serde_json::Value::as_f64)
        .map(|number| number as f32)
        .ok_or_else(|| Error::Other(format!("qwen3.5 config: missing number `{key}`")))
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIN_CONFIG: &str = r#"{
      "architectures":["Qwen3_5ForConditionalGeneration"],
      "model_type":"qwen3_5","language_model_only":false,
      "image_token_id":248056,"video_token_id":248057,
      "vision_start_token_id":248053,"vision_end_token_id":248054,
      "text_config":{
        "hidden_size":5120,"intermediate_size":17408,"num_hidden_layers":4,
        "num_attention_heads":24,"num_key_value_heads":4,"head_dim":256,
        "vocab_size":248320,"max_position_embeddings":262144,"rms_norm_eps":0.000001,
        "full_attention_interval":4,
        "layer_types":["linear_attention","linear_attention","linear_attention","full_attention"],
        "linear_conv_kernel_dim":4,"linear_key_head_dim":128,"linear_num_key_heads":16,
        "linear_num_value_heads":48,"linear_value_head_dim":128,
        "partial_rotary_factor":0.25,"attn_output_gate":true,"output_gate_type":"swish",
        "mtp_num_hidden_layers":1,
        "rope_parameters":{"rope_theta":10000000,"mrope_section":[11,11,10],"mrope_interleaved":true}
      },
      "vision_config":{"depth":27,"hidden_size":1152,"intermediate_size":4304,"num_heads":16,
        "in_channels":3,"num_position_embeddings":2304,"deepstack_visual_indexes":[],
        "patch_size":16,"temporal_patch_size":2,"spatial_merge_size":2,"out_hidden_size":5120},
      "quantization_config":{"quant_method":"compressed-tensors","format":"pack-quantized",
        "ignore":["lm_head"],"config_groups":{"group_0":{"format":"pack-quantized",
          "targets":["Linear"],"input_activations":null,"output_activations":null,
          "weights":{"type":"int","num_bits":4,"group_size":32,"strategy":"group",
            "symmetric":false,"dynamic":false}}}}
    }"#;

    #[test]
    fn parses_strict_qwen38_awq_contract() {
        let config = Qwen35Config::from_json_str(MIN_CONFIG).unwrap();
        assert_eq!(config.text.hidden_size, 5120);
        assert_eq!(config.text.layer_types[3], Qwen35LayerType::FullAttention);
        assert_eq!(config.quantization.pack_factor().unwrap(), 8);
        assert_eq!(config.vision.depth, 27);
    }

    #[test]
    fn rejects_layer_schedule_drift() {
        let invalid = MIN_CONFIG.replace(
            "\"linear_attention\",\"linear_attention\",\"linear_attention\",\"full_attention\"",
            "\"full_attention\",\"linear_attention\",\"linear_attention\",\"full_attention\"",
        );
        let error = Qwen35Config::from_json_str(&invalid).unwrap_err();
        assert!(error.to_string().contains("layer 0"), "{error}");
    }
}
