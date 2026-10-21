//! Qwen3-TTS-12Hz-1.7B-CustomVoice configuration: the fixed geometry this
//! family admits, its voices and languages, and the validated sampling policy.
use apxinf_core::{Error, Result};
use serde_json::Value;
use std::collections::BTreeMap;

pub const HIDDEN: usize = 2048;
pub const HEADS: usize = 16;
pub const KV_HEADS: usize = 8;
pub const HEAD_DIM: usize = 128;
pub const LAYERS: usize = 28;
pub const MLP: usize = 6144;
pub const VOCAB: usize = 3072;
pub const TEXT_VOCAB: usize = 151_936;
pub const CODE_GROUPS: usize = 16;
pub const PREDICTOR_HIDDEN: usize = 1024;
pub const PREDICTOR_LAYERS: usize = 5;
pub const PREDICTOR_MLP: usize = 3072;
pub const PREDICTOR_VOCAB: usize = 2048;
pub const ROPE_THETA: f32 = 1_000_000.0;
pub const EPS: f32 = 1e-6;

/// Waveform decoder geometry (`speech_tokenizer/config.json`, decoder part).
pub mod codec {
    pub const LATENT: usize = 1024;
    pub const CODEBOOK_DIM: usize = 512;
    pub const CODEBOOK_SIZE: usize = 2048;
    pub const DECODER_DIM: usize = 1536;
    pub const HIDDEN: usize = 512;
    pub const MLP: usize = 1024;
    pub const HEADS: usize = 16;
    pub const HEAD_DIM: usize = 64;
    pub const LAYERS: usize = 8;
    pub const EPS: f32 = 1e-5;
    pub const ROPE_THETA: f32 = 10000.0;
    pub const UPSAMPLE_RATES: [usize; 4] = [8, 5, 4, 3];
    pub const UPSAMPLING_RATIOS: [usize; 2] = [2, 2];
    /// Samples per codec frame: the product of all upsampling factors.
    pub const SAMPLES_PER_FRAME: usize = 1920;
    pub const SAMPLE_RATE: u32 = 24000;
}

/// Special text and codec token ids.
#[derive(Debug, Clone, Copy)]
pub struct Tokens {
    pub tts_bos: u32,
    pub tts_eos: u32,
    pub tts_pad: u32,
    pub codec_bos: u32,
    pub codec_eos: u32,
    pub codec_pad: u32,
    pub think: u32,
    pub nothink: u32,
    pub think_bos: u32,
    pub think_eos: u32,
}

/// The only sampling policy this family has validated, from
/// `generation_config.json`; `max_frames` is the configured length limit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sampling {
    pub max_frames: usize,
}

pub struct Config {
    pub tokens: Tokens,
    /// Voice name (lowercase) -> codec embedding row.
    pub voices: BTreeMap<String, u32>,
    /// Language or dialect name (lowercase) -> codec embedding row.
    pub languages: BTreeMap<String, u32>,
    /// Voice name -> dialect that replaces Chinese or automatic language.
    pub dialects: BTreeMap<String, String>,
    pub sampling: Sampling,
}

fn field<'a>(value: &'a Value, path: &str) -> Result<&'a Value> {
    path.split('.')
        .try_fold(value, |v, key| v.get(key))
        .ok_or_else(|| Error::Other(format!("Qwen3-TTS config lacks `{path}`")))
}

fn uint(value: &Value, path: &str) -> Result<u32> {
    field(value, path)?
        .as_u64()
        .and_then(|v| u32::try_from(v).ok())
        .ok_or_else(|| Error::Other(format!("Qwen3-TTS config `{path}` is not an id")))
}

fn expect(value: &Value, path: &str, want: Value) -> Result<()> {
    let got = field(value, path)?;
    // JSON integers and floats with the same value compare equal here.
    let same = match (got.as_f64(), want.as_f64()) {
        (Some(a), Some(b)) => a == b,
        _ => *got == want,
    };
    if !same {
        return Err(Error::Other(format!(
            "Qwen3-TTS `{path}` is {got}, but this family supports only {want}"
        )));
    }
    Ok(())
}

fn ids(value: &Value, path: &str) -> Result<BTreeMap<String, u32>> {
    field(value, path)?
        .as_object()
        .ok_or_else(|| Error::Other(format!("Qwen3-TTS config `{path}` is not a map")))?
        .iter()
        .map(|(k, v)| {
            let id = v
                .as_u64()
                .and_then(|v| u32::try_from(v).ok())
                .filter(|&v| (v as usize) < VOCAB)
                .ok_or_else(|| Error::Other(format!("Qwen3-TTS `{path}.{k}` is not a codec id")))?;
            Ok((k.to_lowercase(), id))
        })
        .collect()
}

impl Config {
    pub fn from_json(main: &str, codec: &str, generation: &str) -> Result<Self> {
        let parse = |raw: &str, what: &str| -> Result<Value> {
            serde_json::from_str(raw).map_err(|e| Error::Other(format!("parse {what}: {e}")))
        };
        let main = parse(main, "config.json")?;
        let codec_config = parse(codec, "speech_tokenizer/config.json")?;
        let generation = parse(generation, "generation_config.json")?;
        expect(&main, "tts_model_type", "custom_voice".into())?;
        expect(&main, "tts_model_size", "1b7".into())?;
        let t = "talker_config";
        let p = "talker_config.code_predictor_config";
        for (path, want) in [
            (format!("{t}.hidden_size"), HIDDEN),
            (format!("{t}.num_attention_heads"), HEADS),
            (format!("{t}.num_key_value_heads"), KV_HEADS),
            (format!("{t}.head_dim"), HEAD_DIM),
            (format!("{t}.num_hidden_layers"), LAYERS),
            (format!("{t}.intermediate_size"), MLP),
            (format!("{t}.vocab_size"), VOCAB),
            (format!("{t}.text_vocab_size"), TEXT_VOCAB),
            (format!("{t}.text_hidden_size"), HIDDEN),
            (format!("{t}.num_code_groups"), CODE_GROUPS),
            (format!("{p}.hidden_size"), PREDICTOR_HIDDEN),
            (format!("{p}.num_attention_heads"), HEADS),
            (format!("{p}.num_key_value_heads"), KV_HEADS),
            (format!("{p}.head_dim"), HEAD_DIM),
            (format!("{p}.num_hidden_layers"), PREDICTOR_LAYERS),
            (format!("{p}.intermediate_size"), PREDICTOR_MLP),
            (format!("{p}.vocab_size"), PREDICTOR_VOCAB),
            (format!("{p}.num_code_groups"), CODE_GROUPS),
        ] {
            expect(&main, &path, want.into())?;
        }
        for prefix in [t, p] {
            expect(&main, &format!("{prefix}.rope_theta"), 1_000_000.into())?;
            expect(&main, &format!("{prefix}.rms_norm_eps"), 1e-6.into())?;
            expect(&main, &format!("{prefix}.hidden_act"), "silu".into())?;
            expect(&main, &format!("{prefix}.attention_bias"), false.into())?;
        }
        expect(
            &main,
            &format!("{t}.rope_scaling.mrope_section"),
            serde_json::json!([24, 20, 20]),
        )?;
        expect(&main, &format!("{t}.rope_scaling.interleaved"), true.into())?;

        let d = "decoder_config";
        for (path, want) in [
            (format!("{d}.latent_dim"), codec::LATENT),
            (format!("{d}.codebook_dim"), codec::CODEBOOK_DIM),
            (format!("{d}.codebook_size"), codec::CODEBOOK_SIZE),
            (format!("{d}.decoder_dim"), codec::DECODER_DIM),
            (format!("{d}.hidden_size"), codec::HIDDEN),
            (format!("{d}.intermediate_size"), codec::MLP),
            (format!("{d}.num_attention_heads"), codec::HEADS),
            (format!("{d}.num_key_value_heads"), codec::HEADS),
            (format!("{d}.head_dim"), codec::HEAD_DIM),
            (format!("{d}.num_hidden_layers"), codec::LAYERS),
            (format!("{d}.num_quantizers"), CODE_GROUPS),
            (format!("{d}.num_semantic_quantizers"), 1),
            (format!("{d}.rope_theta"), 10000),
        ] {
            expect(&codec_config, &path, want.into())?;
        }
        expect(&codec_config, &format!("{d}.rms_norm_eps"), 1e-5.into())?;
        expect(&codec_config, &format!("{d}.attention_bias"), false.into())?;
        expect(
            &codec_config,
            &format!("{d}.upsample_rates"),
            serde_json::json!(codec::UPSAMPLE_RATES),
        )?;
        expect(
            &codec_config,
            &format!("{d}.upsampling_ratios"),
            serde_json::json!(codec::UPSAMPLING_RATIOS),
        )?;
        expect(
            &codec_config,
            "output_sample_rate",
            codec::SAMPLE_RATE.into(),
        )?;
        expect(
            &codec_config,
            "decode_upsample_rate",
            codec::SAMPLES_PER_FRAME.into(),
        )?;

        for (path, want) in [
            ("do_sample", Value::from(true)),
            ("temperature", 0.9.into()),
            ("top_k", 50.into()),
            ("top_p", 1.0.into()),
            ("repetition_penalty", 1.05.into()),
            ("subtalker_dosample", true.into()),
            ("subtalker_temperature", 0.9.into()),
            ("subtalker_top_k", 50.into()),
            ("subtalker_top_p", 1.0.into()),
        ] {
            expect(&generation, path, want)?;
        }
        let max_frames = field(&generation, "max_new_tokens")?
            .as_u64()
            .and_then(|v| usize::try_from(v).ok())
            .filter(|&v| v > 0)
            .ok_or_else(|| Error::Other("Qwen3-TTS max_new_tokens must be positive".into()))?;

        let tokens = Tokens {
            tts_bos: uint(&main, "tts_bos_token_id")?,
            tts_eos: uint(&main, "tts_eos_token_id")?,
            tts_pad: uint(&main, "tts_pad_token_id")?,
            codec_bos: uint(&main, &format!("{t}.codec_bos_id"))?,
            codec_eos: uint(&main, &format!("{t}.codec_eos_token_id"))?,
            codec_pad: uint(&main, &format!("{t}.codec_pad_id"))?,
            think: uint(&main, &format!("{t}.codec_think_id"))?,
            nothink: uint(&main, &format!("{t}.codec_nothink_id"))?,
            think_bos: uint(&main, &format!("{t}.codec_think_bos_id"))?,
            think_eos: uint(&main, &format!("{t}.codec_think_eos_id"))?,
        };
        let codec_ids = [
            tokens.codec_bos,
            tokens.codec_eos,
            tokens.codec_pad,
            tokens.think,
            tokens.nothink,
            tokens.think_bos,
            tokens.think_eos,
        ];
        if codec_ids.iter().any(|&id| id as usize >= VOCAB)
            || [tokens.tts_bos, tokens.tts_eos, tokens.tts_pad]
                .iter()
                .any(|&id| id as usize >= TEXT_VOCAB)
        {
            return Err(Error::Other(
                "Qwen3-TTS special token outside its vocabulary".into(),
            ));
        }
        let voices = ids(&main, &format!("{t}.spk_id"))?;
        let languages = ids(&main, &format!("{t}.codec_language_id"))?;
        let mut dialects = BTreeMap::new();
        if let Some(map) = main
            .get(t)
            .and_then(|c| c.get("spk_is_dialect"))
            .and_then(Value::as_object)
        {
            for (voice, dialect) in map {
                if let Some(name) = dialect.as_str() {
                    if !languages.contains_key(name) {
                        return Err(Error::Other(format!(
                            "Qwen3-TTS voice `{voice}` names unknown dialect `{name}`"
                        )));
                    }
                    dialects.insert(voice.to_lowercase(), name.to_owned());
                }
            }
        }
        Ok(Self {
            tokens,
            voices,
            languages,
            dialects,
            sampling: Sampling { max_frames },
        })
    }

    /// Requestable languages: `auto` and every non-dialect language id.
    pub fn supports_language(&self, language: &str) -> bool {
        let language = language.to_lowercase();
        language == "auto"
            || (self.languages.contains_key(&language) && !language.contains("dialect"))
    }

    /// The codec language row for a request, after the voice's dialect
    /// override, as mlx-audio `_prepare_generation_inputs` selects it.
    pub fn language_id(&self, language: &str, voice: &str) -> Option<u32> {
        let language = language.to_lowercase();
        let mut id = (language != "auto")
            .then(|| self.languages.get(&language).copied())
            .flatten();
        if matches!(language.as_str(), "chinese" | "auto") {
            if let Some(dialect) = self.dialects.get(&voice.to_lowercase()) {
                id = self.languages.get(dialect).copied();
            }
        }
        id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn configs() -> (Value, Value, Value) {
        let shared = json!({"rope_theta": 1000000, "rms_norm_eps": 1e-6, "hidden_act": "silu",
                            "attention_bias": false, "head_dim": 128, "num_attention_heads": 16,
                            "num_key_value_heads": 8, "num_code_groups": 16});
        let mut predictor = shared.clone();
        for (k, v) in [
            ("hidden_size", 1024),
            ("num_hidden_layers", 5),
            ("intermediate_size", 3072),
            ("vocab_size", 2048),
        ] {
            predictor[k] = v.into();
        }
        let mut talker = shared;
        for (k, v) in [
            ("hidden_size", 2048),
            ("num_hidden_layers", 28),
            ("intermediate_size", 6144),
            ("vocab_size", 3072),
            ("text_vocab_size", 151936),
            ("text_hidden_size", 2048),
            ("codec_bos_id", 2149),
            ("codec_eos_token_id", 2150),
            ("codec_pad_id", 2148),
            ("codec_think_id", 2154),
            ("codec_nothink_id", 2155),
            ("codec_think_bos_id", 2156),
            ("codec_think_eos_id", 2157),
        ] {
            talker[k] = v.into();
        }
        talker["code_predictor_config"] = predictor;
        talker["rope_scaling"] = json!({"interleaved": true, "mrope_section": [24, 20, 20]});
        talker["codec_language_id"] =
            json!({"chinese": 2055, "english": 2050, "sichuan_dialect": 2062});
        talker["spk_id"] = json!({"Vivian": 3065, "eric": 2875});
        talker["spk_is_dialect"] = json!({"vivian": false, "eric": "sichuan_dialect"});
        let main = json!({"tts_model_type": "custom_voice", "tts_model_size": "1b7", "tts_bos_token_id": 151672,
                          "tts_eos_token_id": 151673, "tts_pad_token_id": 151671, "talker_config": talker});
        let codec = json!({"output_sample_rate": 24000, "decode_upsample_rate": 1920, "decoder_config": {
            "latent_dim": 1024, "codebook_dim": 512, "codebook_size": 2048, "decoder_dim": 1536, "hidden_size": 512,
            "intermediate_size": 1024, "num_attention_heads": 16, "num_key_value_heads": 16, "head_dim": 64,
            "num_hidden_layers": 8, "num_quantizers": 16, "num_semantic_quantizers": 1, "rope_theta": 10000,
            "rms_norm_eps": 1e-5, "attention_bias": false, "upsample_rates": [8, 5, 4, 3],
            "upsampling_ratios": [2, 2], "sliding_window": 72}});
        let generation = json!({"do_sample": true, "temperature": 0.9, "top_k": 50, "top_p": 1.0,
                                "repetition_penalty": 1.05, "subtalker_dosample": true,
                                "subtalker_temperature": 0.9, "subtalker_top_k": 50, "subtalker_top_p": 1.0,
                                "max_new_tokens": 8192});
        (main, codec, generation)
    }

    fn parse(main: &Value, codec: &Value, generation: &Value) -> Result<Config> {
        Config::from_json(
            &main.to_string(),
            &codec.to_string(),
            &generation.to_string(),
        )
    }

    #[test]
    fn voices_languages_and_dialects_follow_mlx_audio() {
        let (main, codec, generation) = configs();
        let config = parse(&main, &codec, &generation).unwrap();
        assert_eq!(config.sampling.max_frames, 8192);
        assert_eq!(config.voices["vivian"], 3065);
        assert!(config.supports_language("English") && config.supports_language("auto"));
        assert!(
            !config.supports_language("sichuan_dialect") && !config.supports_language("klingon")
        );
        assert_eq!(config.language_id("English", "Vivian"), Some(2050));
        assert_eq!(config.language_id("auto", "Vivian"), None);
        // A dialect voice replaces Chinese or automatic language, not English.
        assert_eq!(config.language_id("Chinese", "Eric"), Some(2062));
        assert_eq!(config.language_id("auto", "eric"), Some(2062));
        assert_eq!(config.language_id("english", "eric"), Some(2050));
    }

    #[test]
    fn other_geometry_or_sampling_is_rejected() {
        let (main, codec, generation) = configs();
        let mut wrong = main.clone();
        wrong["talker_config"]["num_hidden_layers"] = 20.into();
        assert!(parse(&wrong, &codec, &generation).is_err());
        let mut base = main.clone();
        base["tts_model_type"] = "base".into();
        assert!(parse(&base, &codec, &generation).is_err());
        let mut greedy = generation.clone();
        greedy["do_sample"] = false.into();
        assert!(parse(&main, &codec, &greedy).is_err());
        let mut codec_wrong = codec.clone();
        codec_wrong["decoder_config"]["upsample_rates"] = json!([8, 5, 4, 4]);
        assert!(parse(&main, &codec_wrong, &generation).is_err());
    }
}
