//! Strict Qwen2.5-Omni Thinker configuration.
//!
//! The deployment owns one pinned architecture. Identity-critical fields are
//! required and validated instead of receiving permissive defaults; accepting
//! a nearby Qwen model would make checkpoint shape validation meaningless.

use std::path::Path;

use apxinf_core::{Error, Result};
use serde_json::{Map, Value};

pub const MODEL_TYPE: &str = "qwen2_5_omni";
pub const ARCHITECTURE: &str = "Qwen2_5OmniModel";

#[derive(Clone, Debug, PartialEq)]
pub struct Qwen25OmniTextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub mrope_section: [usize; 3],
}

#[derive(Clone, Debug, PartialEq)]
pub struct Qwen25OmniVisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub spatial_merge_size: usize,
    pub in_channels: usize,
    pub out_hidden_size: usize,
    pub window_size: usize,
    pub full_attention_blocks: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Qwen25OmniAudioConfig {
    pub num_mel_bins: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub output_dim: usize,
    pub max_source_positions: usize,
    pub n_window: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Qwen25OmniProcessorConfig {
    pub sampling_rate: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub feature_size: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Qwen25OmniConfig {
    pub model_type: String,
    pub architecture: String,
    pub torch_dtype: String,
    pub text: Qwen25OmniTextConfig,
    pub vision: Qwen25OmniVisionConfig,
    pub audio: Qwen25OmniAudioConfig,
    pub processor: Qwen25OmniProcessorConfig,
    pub pad_token_id: u32,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub audio_token_id: u32,
    pub audio_start_token_id: u32,
    pub audio_end_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
    pub vision_token_id: u32,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub position_id_per_seconds: usize,
    pub seconds_per_chunk: usize,
}

impl Qwen25OmniConfig {
    pub fn from_json_file(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|error| Error::Other(format!("read {}: {error}", path.display())))?;
        Self::from_json_str(&raw)
    }

    pub fn from_model_dir(model_dir: &Path) -> Result<Self> {
        let mut config = Self::from_json_file(&model_dir.join("config.json"))?;
        config.processor = parse_processor_file(&model_dir.join("preprocessor_config.json"))?;
        config.validate_pinned_contract()?;
        Ok(config)
    }

    pub fn from_json_str(raw: &str) -> Result<Self> {
        let value: Value = serde_json::from_str(raw)
            .map_err(|error| Error::Other(format!("qwen2.5-omni config json: {error}")))?;
        let root = object(&value, "config root")?;
        let architectures = array(root, "architectures")?;
        if architectures.len() != 1 {
            return Err(Error::Other(format!(
                "qwen2.5-omni config: architectures must contain exactly one entry, got {}",
                architectures.len()
            )));
        }
        let architecture = architectures[0]
            .as_str()
            .ok_or_else(|| {
                Error::Other("qwen2.5-omni config: architectures[0] must be a string".into())
            })?
            .to_owned();
        let thinker = object_field(root, "thinker_config")?;
        let text = object_field(thinker, "text_config")?;
        let vision = object_field(thinker, "vision_config")?;
        let audio = object_field(thinker, "audio_config")?;
        validate_runtime_semantics(thinker, text, vision, audio)?;
        let rope = object_field(text, "rope_scaling")?;
        let mrope = array(rope, "mrope_section")?;
        if mrope.len() != 3 {
            return Err(Error::Other(format!(
                "qwen2.5-omni config: text rope mrope_section must have 3 entries, got {}",
                mrope.len()
            )));
        }
        let full_attention_blocks = array(vision, "fullatt_block_indexes")?
            .iter()
            .enumerate()
            .map(|(index, value)| value_usize(value, &format!("fullatt_block_indexes[{index}]")))
            .collect::<Result<Vec<_>>>()?;
        let hidden_size = usize_field(text, "hidden_size")?;
        let n_heads = usize_field(text, "num_attention_heads")?;
        if hidden_size % n_heads != 0 {
            return Err(Error::Other(format!(
                "qwen2.5-omni config: text hidden_size {hidden_size} is not divisible by heads {n_heads}"
            )));
        }
        let vision_hidden = usize_field(vision, "hidden_size")?;
        let vision_heads = usize_field(vision, "num_heads")?;
        if vision_hidden % vision_heads != 0 {
            return Err(Error::Other(format!(
                "qwen2.5-omni config: vision hidden_size {vision_hidden} is not divisible by heads {vision_heads}"
            )));
        }
        let audio_hidden = usize_field(audio, "d_model")?;
        let audio_heads = usize_field(audio, "encoder_attention_heads")?;
        if audio_hidden % audio_heads != 0 {
            return Err(Error::Other(format!(
                "qwen2.5-omni config: audio d_model {audio_hidden} is not divisible by heads {audio_heads}"
            )));
        }

        let config = Self {
            model_type: string_field(root, "model_type")?,
            architecture,
            torch_dtype: string_field(root, "torch_dtype")?,
            text: Qwen25OmniTextConfig {
                hidden_size,
                intermediate_size: usize_field(text, "intermediate_size")?,
                n_layers: usize_field(text, "num_hidden_layers")?,
                n_heads,
                n_kv_heads: usize_field(text, "num_key_value_heads")?,
                head_dim: hidden_size / n_heads,
                vocab_size: usize_field(text, "vocab_size")?,
                max_position_embeddings: usize_field(text, "max_position_embeddings")?,
                rms_norm_eps: f32_field(text, "rms_norm_eps")?,
                rope_theta: f32_field(text, "rope_theta")?,
                mrope_section: [
                    value_usize(&mrope[0], "mrope_section[0]")?,
                    value_usize(&mrope[1], "mrope_section[1]")?,
                    value_usize(&mrope[2], "mrope_section[2]")?,
                ],
            },
            vision: Qwen25OmniVisionConfig {
                depth: usize_field(vision, "depth")?,
                hidden_size: vision_hidden,
                intermediate_size: usize_field(vision, "intermediate_size")?,
                n_heads: vision_heads,
                head_dim: vision_hidden / vision_heads,
                patch_size: usize_field(vision, "patch_size")?,
                temporal_patch_size: usize_field(vision, "temporal_patch_size")?,
                spatial_merge_size: usize_field(vision, "spatial_merge_size")?,
                in_channels: usize_field_any(vision, &["in_channels", "in_chans"])?,
                out_hidden_size: usize_field(vision, "out_hidden_size")?,
                window_size: usize_field(vision, "window_size")?,
                full_attention_blocks,
            },
            audio: Qwen25OmniAudioConfig {
                num_mel_bins: usize_field(audio, "num_mel_bins")?,
                hidden_size: audio_hidden,
                intermediate_size: usize_field(audio, "encoder_ffn_dim")?,
                n_layers: usize_field(audio, "encoder_layers")?,
                n_heads: audio_heads,
                head_dim: audio_hidden / audio_heads,
                output_dim: usize_field(audio, "output_dim")?,
                max_source_positions: usize_field(audio, "max_source_positions")?,
                n_window: usize_field(audio, "n_window")?,
            },
            processor: Qwen25OmniProcessorConfig {
                sampling_rate: 0,
                n_fft: 0,
                hop_length: 0,
                feature_size: 0,
            },
            pad_token_id: u32_field(thinker, "pad_token_id")?,
            bos_token_id: u32_field(thinker, "bos_token_id")?,
            eos_token_id: u32_field(thinker, "eos_token_id")?,
            audio_token_id: u32_field(thinker, "audio_token_index")?,
            audio_start_token_id: u32_field(thinker, "audio_start_token_id")?,
            audio_end_token_id: u32_field(thinker, "audio_end_token_id")?,
            vision_start_token_id: u32_field(thinker, "vision_start_token_id")?,
            vision_end_token_id: u32_field(thinker, "vision_end_token_id")?,
            vision_token_id: u32_field(thinker, "vision_token_id")?,
            image_token_id: u32_field(thinker, "image_token_index")?,
            video_token_id: u32_field(thinker, "video_token_index")?,
            position_id_per_seconds: usize_field(thinker, "position_id_per_seconds")?,
            seconds_per_chunk: usize_field(thinker, "seconds_per_chunk")?,
        };
        config.validate_architecture()?;
        Ok(config)
    }

    fn validate_architecture(&self) -> Result<()> {
        require_eq("model_type", &self.model_type, MODEL_TYPE)?;
        require_eq("architecture", &self.architecture, ARCHITECTURE)?;
        require_eq("torch_dtype", &self.torch_dtype, "bfloat16")?;
        if self.text.mrope_section.iter().sum::<usize>() != self.text.head_dim / 2 {
            return Err(Error::Other(format!(
                "qwen2.5-omni config: mrope_section {:?} does not cover head_dim/2={}",
                self.text.mrope_section,
                self.text.head_dim / 2
            )));
        }
        Ok(())
    }

    pub fn validate_pinned_contract(&self) -> Result<()> {
        self.validate_architecture()?;
        macro_rules! exact {
            ($field:expr, $actual:expr, $expected:expr) => {
                if $actual != $expected {
                    return Err(Error::Other(format!(
                        "qwen2.5-omni config: {}={:?}, expected {:?}",
                        $field, $actual, $expected
                    )));
                }
            };
        }
        exact!("text.hidden_size", self.text.hidden_size, 2048);
        exact!("text.intermediate_size", self.text.intermediate_size, 11008);
        exact!("text.layers", self.text.n_layers, 36);
        exact!("text.heads", self.text.n_heads, 16);
        exact!("text.kv_heads", self.text.n_kv_heads, 2);
        exact!("text.head_dim", self.text.head_dim, 128);
        exact!("text.vocab_size", self.text.vocab_size, 151936);
        exact!(
            "text.max_positions",
            self.text.max_position_embeddings,
            32768
        );
        exact!("text.rope_theta", self.text.rope_theta, 1_000_000.0);
        exact!("text.mrope_section", self.text.mrope_section, [16, 24, 24]);
        exact!("vision.depth", self.vision.depth, 32);
        exact!("vision.hidden_size", self.vision.hidden_size, 1280);
        exact!(
            "vision.intermediate_size",
            self.vision.intermediate_size,
            3420
        );
        exact!("vision.heads", self.vision.n_heads, 16);
        exact!("vision.head_dim", self.vision.head_dim, 80);
        exact!("vision.in_channels", self.vision.in_channels, 3);
        exact!("vision.patch_size", self.vision.patch_size, 14);
        exact!(
            "vision.temporal_patch_size",
            self.vision.temporal_patch_size,
            2
        );
        exact!("vision.merge_size", self.vision.spatial_merge_size, 2);
        exact!("vision.output", self.vision.out_hidden_size, 2048);
        exact!("vision.window_size", self.vision.window_size, 112);
        exact!(
            "vision.full_attention_blocks",
            self.vision.full_attention_blocks.as_slice(),
            &[7, 15, 23, 31]
        );
        exact!("audio.mel_bins", self.audio.num_mel_bins, 128);
        exact!("audio.hidden_size", self.audio.hidden_size, 1280);
        exact!("audio.layers", self.audio.n_layers, 32);
        exact!("audio.heads", self.audio.n_heads, 20);
        exact!("audio.head_dim", self.audio.head_dim, 64);
        exact!(
            "audio.intermediate_size",
            self.audio.intermediate_size,
            5120
        );
        exact!("audio.output_dim", self.audio.output_dim, 2048);
        exact!(
            "audio.max_source_positions",
            self.audio.max_source_positions,
            1500
        );
        exact!("audio.n_window", self.audio.n_window, 100);
        exact!(
            "processor.sampling_rate",
            self.processor.sampling_rate,
            16000
        );
        exact!("processor.n_fft", self.processor.n_fft, 400);
        exact!("processor.hop_length", self.processor.hop_length, 160);
        exact!("processor.feature_size", self.processor.feature_size, 128);
        exact!("pad_token_id", self.pad_token_id, 151643);
        exact!("bos_token_id", self.bos_token_id, 151644);
        exact!("eos_token_id", self.eos_token_id, 151645);
        exact!("audio_token_id", self.audio_token_id, 151646);
        exact!("audio_start_token_id", self.audio_start_token_id, 151647);
        exact!("audio_end_token_id", self.audio_end_token_id, 151648);
        exact!("vision_start_token_id", self.vision_start_token_id, 151652);
        exact!("vision_end_token_id", self.vision_end_token_id, 151653);
        exact!("vision_token_id", self.vision_token_id, 151654);
        exact!("image_token_id", self.image_token_id, 151655);
        exact!("video_token_id", self.video_token_id, 151656);
        exact!("position_id_per_seconds", self.position_id_per_seconds, 25);
        exact!("seconds_per_chunk", self.seconds_per_chunk, 2);
        if (self.text.rms_norm_eps - 1e-6).abs() > f32::EPSILON {
            return Err(Error::Other(format!(
                "qwen2.5-omni config: text.rms_norm_eps={}, expected 1e-6",
                self.text.rms_norm_eps
            )));
        }
        Ok(())
    }
}

fn validate_runtime_semantics(
    thinker: &Map<String, Value>,
    text: &Map<String, Value>,
    vision: &Map<String, Value>,
    audio: &Map<String, Value>,
) -> Result<()> {
    require_eq(
        "thinker.model_type",
        &string_field(thinker, "model_type")?,
        "qwen2_5_omni_thinker",
    )?;
    require_eq(
        "thinker.torch_dtype",
        &string_field(thinker, "torch_dtype")?,
        "bfloat16",
    )?;
    require_eq(
        "text.model_type",
        &string_field(text, "model_type")?,
        "qwen2_5_omni_text",
    )?;
    require_eq(
        "text.hidden_act",
        &string_field(text, "hidden_act")?,
        "silu",
    )?;
    require_bool(text, "use_cache", true)?;
    require_bool(text, "use_sliding_window", false)?;
    require_bool(text, "tie_word_embeddings", false)?;
    require_zero(text, "attention_dropout")?;
    require_eq(
        "vision.model_type",
        &string_field(vision, "model_type")?,
        "qwen2_5_omni_vision_encoder",
    )?;
    require_eq(
        "vision.hidden_act",
        &string_field(vision, "hidden_act")?,
        "silu",
    )?;
    require_eq(
        "audio.model_type",
        &string_field(audio, "model_type")?,
        "qwen2_5_omni_audio_encoder",
    )?;
    require_eq(
        "audio.activation_function",
        &string_field(audio, "activation_function")?,
        "gelu",
    )?;
    for field in [
        "activation_dropout",
        "attention_dropout",
        "dropout",
        "encoder_layerdrop",
    ] {
        require_zero(audio, field)?;
    }
    Ok(())
}

fn parse_processor_file(path: &Path) -> Result<Qwen25OmniProcessorConfig> {
    let raw = std::fs::read_to_string(path)
        .map_err(|error| Error::Other(format!("read {}: {error}", path.display())))?;
    let value: Value = serde_json::from_str(&raw)
        .map_err(|error| Error::Other(format!("parse {}: {error}", path.display())))?;
    let root = object(&value, "preprocessor config root")?;
    let processor_class = string_field(root, "processor_class")?;
    require_eq("processor_class", &processor_class, "Qwen2_5OmniProcessor")?;
    Ok(Qwen25OmniProcessorConfig {
        sampling_rate: usize_field(root, "sampling_rate")?,
        n_fft: usize_field(root, "n_fft")?,
        hop_length: usize_field(root, "hop_length")?,
        feature_size: usize_field(root, "feature_size")?,
    })
}

fn object<'a>(value: &'a Value, name: &str) -> Result<&'a Map<String, Value>> {
    value
        .as_object()
        .ok_or_else(|| Error::Other(format!("qwen2.5-omni config: {name} must be an object")))
}
fn object_field<'a>(map: &'a Map<String, Value>, key: &str) -> Result<&'a Map<String, Value>> {
    object(
        map.get(key)
            .ok_or_else(|| Error::Other(format!("qwen2.5-omni config: missing {key}")))?,
        key,
    )
}
fn array<'a>(map: &'a Map<String, Value>, key: &str) -> Result<&'a Vec<Value>> {
    map.get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| Error::Other(format!("qwen2.5-omni config: {key} must be an array")))
}
fn string_field(map: &Map<String, Value>, key: &str) -> Result<String> {
    map.get(key)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| Error::Other(format!("qwen2.5-omni config: {key} must be a string")))
}
fn usize_field(map: &Map<String, Value>, key: &str) -> Result<usize> {
    value_usize(
        map.get(key)
            .ok_or_else(|| Error::Other(format!("qwen2.5-omni config: missing {key}")))?,
        key,
    )
}
fn usize_field_any(map: &Map<String, Value>, keys: &[&str]) -> Result<usize> {
    for key in keys {
        if let Some(value) = map.get(*key) {
            return value_usize(value, key);
        }
    }
    Err(Error::Other(format!(
        "qwen2.5-omni config: missing one of {}",
        keys.join(", ")
    )))
}
fn value_usize(value: &Value, name: &str) -> Result<usize> {
    let value = value.as_u64().ok_or_else(|| {
        Error::Other(format!(
            "qwen2.5-omni config: {name} must be a non-negative integer"
        ))
    })?;
    usize::try_from(value)
        .map_err(|_| Error::Other(format!("qwen2.5-omni config: {name} exceeds usize")))
}
fn u32_field(map: &Map<String, Value>, key: &str) -> Result<u32> {
    let value = usize_field(map, key)?;
    u32::try_from(value)
        .map_err(|_| Error::Other(format!("qwen2.5-omni config: {key} exceeds u32")))
}
fn f32_field(map: &Map<String, Value>, key: &str) -> Result<f32> {
    let value = map
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| Error::Other(format!("qwen2.5-omni config: {key} must be numeric")))?;
    if !value.is_finite() || value < f32::MIN as f64 || value > f32::MAX as f64 {
        return Err(Error::Other(format!(
            "qwen2.5-omni config: {key} is not finite f32"
        )));
    }
    Ok(value as f32)
}
fn require_bool(map: &Map<String, Value>, key: &str, expected: bool) -> Result<()> {
    let actual = map
        .get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| Error::Other(format!("qwen2.5-omni config: {key} must be boolean")))?;
    if actual != expected {
        return Err(Error::Other(format!(
            "qwen2.5-omni config: {key}={actual}, expected {expected}"
        )));
    }
    Ok(())
}
fn require_zero(map: &Map<String, Value>, key: &str) -> Result<()> {
    let actual = map
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| Error::Other(format!("qwen2.5-omni config: {key} must be numeric")))?;
    if actual != 0.0 {
        return Err(Error::Other(format!(
            "qwen2.5-omni config: {key}={actual}, expected 0"
        )));
    }
    Ok(())
}
fn require_eq(name: &str, actual: &str, expected: &str) -> Result<()> {
    if actual != expected {
        return Err(Error::Other(format!(
            "qwen2.5-omni config: {name}={actual}, expected {expected}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: &str = r#"{
      "architectures":["Qwen2_5OmniModel"],"model_type":"qwen2_5_omni","torch_dtype":"bfloat16",
      "thinker_config":{"model_type":"qwen2_5_omni_thinker","torch_dtype":"bfloat16",
        "pad_token_id":151643,"bos_token_id":151644,"eos_token_id":151645,
        "audio_token_index":151646,"audio_start_token_id":151647,"audio_end_token_id":151648,
        "vision_start_token_id":151652,"vision_end_token_id":151653,"vision_token_id":151654,
        "image_token_index":151655,"video_token_index":151656,"position_id_per_seconds":25,"seconds_per_chunk":2,
        "text_config":{"model_type":"qwen2_5_omni_text","hidden_act":"silu","attention_dropout":0,
          "use_cache":true,"use_sliding_window":false,"tie_word_embeddings":false,
          "hidden_size":2048,"intermediate_size":11008,"num_hidden_layers":36,
          "num_attention_heads":16,"num_key_value_heads":2,"vocab_size":151936,"max_position_embeddings":32768,
          "rms_norm_eps":0.000001,"rope_theta":1000000,"rope_scaling":{"mrope_section":[16,24,24]}},
        "vision_config":{"model_type":"qwen2_5_omni_vision_encoder","hidden_act":"silu",
          "depth":32,"hidden_size":1280,"intermediate_size":3420,"num_heads":16,
          "patch_size":14,"temporal_patch_size":2,"spatial_merge_size":2,"in_channels":3,
          "out_hidden_size":2048,"window_size":112,"fullatt_block_indexes":[7,15,23,31]},
        "audio_config":{"model_type":"qwen2_5_omni_audio_encoder","activation_function":"gelu",
          "activation_dropout":0,"attention_dropout":0,"dropout":0,"encoder_layerdrop":0,
          "num_mel_bins":128,"d_model":1280,"encoder_ffn_dim":5120,"encoder_layers":32,
          "encoder_attention_heads":20,"output_dim":2048,"max_source_positions":1500,"n_window":100}}
    }"#;

    #[test]
    fn parses_exact_nested_architecture_without_defaults() {
        let config = Qwen25OmniConfig::from_json_str(CONFIG).unwrap();
        assert_eq!(config.text.head_dim, 128);
        assert_eq!(config.vision.head_dim, 80);
        assert_eq!(config.audio.head_dim, 64);
        assert_eq!(config.text.mrope_section, [16, 24, 24]);
        assert_eq!(config.vision.full_attention_blocks, [7, 15, 23, 31]);
    }

    #[test]
    fn rejects_identity_and_required_field_drift() {
        let wrong = CONFIG.replace("qwen2_5_omni", "qwen2_5_vl");
        assert!(Qwen25OmniConfig::from_json_str(&wrong)
            .unwrap_err()
            .to_string()
            .contains("model_type"));
        let missing = CONFIG.replace("\"num_hidden_layers\":36,", "");
        assert!(Qwen25OmniConfig::from_json_str(&missing)
            .unwrap_err()
            .to_string()
            .contains("num_hidden_layers"));
        let unsupported_activation = CONFIG.replace(
            "\"activation_function\":\"gelu\"",
            "\"activation_function\":\"relu\"",
        );
        assert!(Qwen25OmniConfig::from_json_str(&unsupported_activation)
            .unwrap_err()
            .to_string()
            .contains("activation_function"));
    }
}
