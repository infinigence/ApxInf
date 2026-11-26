//! Native Qwen3-TTS-12Hz-1.7B-CustomVoice on Apple Metal: the Talker, the
//! code predictor and the 12 Hz waveform decoder, following the EngineTailor
//! MLX parity path (mlx-audio 70f4add with reference cast seams). Batch one,
//! non-streaming. No Python model execution.
mod codec;
mod config;
mod ops;
mod sample;
mod talker;
mod weights;

use crate::tts_trait::{GeneratedAudio, TtsContract, TtsRequest, TtsTrait};
use crate::{LoadOptions, LoadedModel, ModelPrecision};
use apxinf_core::{Backend, Device, Error, Result};
use apxinf_mlx::{Array, MlxBackend};
use apxinf_tokenizer::Tokenizer;
use config::{codec as codec_config, Config, CODE_GROUPS, TEXT_VOCAB, VOCAB};
use ops::Ops;
use std::path::Path;
use std::sync::Arc;
use talker::{embed, new_caches, Model};

pub struct QwenTts {
    config: Config,
    tokenizer: Tokenizer,
    model: Model,
    decoder: codec::Decoder,
    sampler: sample::Sampler,
    ops: Ops,
}

/// The predictor MLP selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Variant {
    /// Use the predictor MLP fusion only on measured architectures.
    Auto,
    /// The reference SwiGLU composition of the parity path.
    Reference,
}

/// Device-independent option admission, checked before the checkpoint is read.
fn admit(options: &LoadOptions) -> Result<Variant> {
    let variant = match options.model_variant.as_deref() {
        None | Some("auto") => Variant::Auto,
        Some("reference") => Variant::Reference,
        Some(_) => {
            return Err(Error::Contract(
                "Qwen3-TTS MLX variants are `auto` (architecture-selected predictor MLP) and `reference`",
            ))
        }
    };
    if !matches!(
        options.precision,
        ModelPrecision::Auto | ModelPrecision::Bf16
    ) || options.text_weight_dtype.is_some()
        || options.config.is_some()
        || options.synthetic.is_some()
        || options.calibration_path.is_some()
        || options.tuning_path.is_some()
        || options.autotune
        || options.uniform_fp8_scale.is_some()
        || !options.assets.is_empty()
    {
        return Err(Error::Contract(
            "Qwen3-TTS MLX loads the BF16 checkpoint as stored; CUDA tuning, \
             calibration, synthetic weights and auxiliary assets are unsupported",
        ));
    }
    Ok(variant)
}

fn use_predictor_fusion(variant: Variant, architecture: &str) -> bool {
    variant == Variant::Auto && architecture == "applegpu_g16g"
}

pub fn load(
    path: &Path,
    device: Device,
    backend: Arc<dyn Backend>,
    options: &LoadOptions,
) -> Result<LoadedModel> {
    if !matches!(device, Device::Metal(_)) {
        return Err(Error::UnsupportedDevice(device));
    }
    let variant = admit(options)?;
    let native = backend
        .as_any()
        .downcast_ref::<MlxBackend>()
        .ok_or(Error::Contract("Qwen3-TTS requires the native MLX backend"))?
        .clone();
    if native.device() != device {
        return Err(Error::DeviceMismatch {
            expected: device,
            got: native.device(),
        });
    }
    if !path.is_dir() {
        return Err(Error::Contract("Qwen3-TTS loads a checkpoint directory"));
    }
    let read = |name: &str| {
        std::fs::read_to_string(path.join(name))
            .map_err(|e| Error::Other(format!("cannot read Qwen3-TTS {name}: {e}")))
    };
    let config = Config::from_json(
        &read("config.json")?,
        &read("speech_tokenizer/config.json")?,
        &read("generation_config.json")?,
    )?;
    let tokenizer = Tokenizer::from_qwen2_vocab(path)?;
    let stream = native.stream().clone();
    let architecture = stream
        .metal_caps()
        .map_or("", |caps| caps.architecture.as_str());
    let fused = use_predictor_fusion(variant, architecture);
    if variant == Variant::Auto && !fused {
        eprintln!(
            "Qwen3-TTS: predictor fusion not used (not measured on {architecture:?}); using reference SwiGLU"
        );
    }
    let ops = Ops::new(&stream)?;
    let (talker, predictor) = weights::Talker::load(&stream, path)?;
    let model = Model::new(talker, predictor, &ops, fused)?;
    let decoder = codec::Decoder::new(weights::Codec::load(&stream, path)?, &ops)?;
    Ok(LoadedModel::Tts(Box::new(QwenTts {
        config,
        tokenizer,
        model,
        decoder,
        sampler: sample::Sampler::new(&stream)?,
        ops,
    })))
}

/// Request admission, before any device work.
fn admit_request(config: &Config, request: &TtsRequest<'_>) -> Result<()> {
    if request.text.trim().is_empty() {
        return Err(Error::Contract("Qwen3-TTS needs nonempty text"));
    }
    let voice = request.sampling.voice.unwrap_or("");
    if !config.voices.contains_key(&voice.to_lowercase()) {
        return Err(Error::Other(format!(
            "Qwen3-TTS has no voice `{}`; supported: {}",
            voice,
            config.voices.keys().cloned().collect::<Vec<_>>().join(", ")
        )));
    }
    if !config.supports_language(request.sampling.language.unwrap_or("")) {
        return Err(Error::Other(format!(
            "Qwen3-TTS does not support language `{}`",
            request.sampling.language.unwrap_or("")
        )));
    }
    Ok(())
}

fn text_ids(ids: Vec<u32>) -> Result<Vec<i32>> {
    ids.into_iter()
        .enumerate()
        .map(|(position, id)| {
            if id as usize >= TEXT_VOCAB {
                return Err(Error::Other(format!(
                    "Qwen3-TTS text token {id} at position {position} outside vocabulary {TEXT_VOCAB}"
                )));
            }
            Ok(id as i32)
        })
        .collect()
}

fn collect_frames<T>(
    limit: usize,
    mut step: impl FnMut(usize) -> Result<Option<T>>,
) -> Result<(Vec<T>, bool)> {
    let mut frames = Vec::new();
    for index in 0..limit {
        match step(index)? {
            Some(frame) => frames.push(frame),
            None => return Ok((frames, true)),
        }
    }
    Ok((frames, false))
}

impl QwenTts {
    fn ids(&self, text: &str) -> Result<Vec<i32>> {
        text_ids(self.tokenizer.encode_with_special_tokens(text, true)?)
    }

    fn rows(&self, ids: &[i32]) -> Result<Array> {
        Array::from_i32(self.ops.stream(), &[1, ids.len()], ids)
    }

    fn text(&self, ids: &[i32]) -> Result<Array> {
        self.model
            .text_projection(&embed(&self.model.talker.text_embedding, &self.rows(ids)?)?)
    }

    fn codec(&self, ids: &[u32]) -> Result<Array> {
        let ids: Vec<i32> = ids.iter().map(|&id| id as i32).collect();
        embed(&self.model.talker.codec_embedding, &self.rows(&ids)?)
    }

    /// The non-streaming prefill `[1, L, 2048]` and the TTS pad embedding: the
    /// mlx-audio speaker/language/instruction prefix with its streaming text
    /// suffix replaced by the official non-streaming layout.
    fn prefill(&self, request: &TtsRequest<'_>) -> Result<(Array, Array)> {
        let t = self.config.tokens;
        let chat = format!(
            "<|im_start|>assistant\n{}<|im_end|>\n<|im_start|>assistant\n",
            request.text
        );
        let ids = self.ids(&chat)?;
        if ids.len() < 9 {
            return Err(Error::Other("Qwen3-TTS text produced no tokens".into()));
        }
        let text_embed = self.text(&ids)?;
        // [TTS BOS, EOS, pad] projected together; the EOS row is unused here.
        let tts = self.text(&[t.tts_bos as i32, t.tts_eos as i32, t.tts_pad as i32])?;
        let (bos, pad) = (tts.slice_axis(1, 0, 1)?, tts.slice_axis(1, 2, 3)?);
        let voice = self.config.voices[&request.sampling.voice.unwrap_or("").to_lowercase()];
        let prefix = match self.config.language_id(
            request.sampling.language.unwrap_or(""),
            request.sampling.voice.unwrap_or(""),
        ) {
            None => vec![t.nothink, t.think_bos, t.think_eos],
            Some(language) => vec![t.think, t.think_bos, language, t.think_eos],
        };
        let codec = Array::concat(
            &[
                &self.codec(&prefix)?,
                &self.codec(&[voice])?,
                &self.codec(&[t.codec_pad, t.codec_bos])?,
            ],
            1,
        )?;
        let n = codec.shape()[1];
        let pads = pad.broadcast_to(&[1, n - 2, pad.shape()[2]])?;
        let combined = Array::concat(&[&pads, &bos], 1)?.add(&codec.slice_axis(1, 0, n - 1)?)?;
        let role = text_embed.slice_axis(1, 0, 3)?;
        let mut parts = Vec::new();
        if let Some(instruction) = request.sampling.instruction.filter(|v| !v.is_empty()) {
            let instruct = format!("<|im_start|>user\n{}<|im_end|>\n", instruction);
            parts.push(self.text(&self.ids(&instruct)?)?);
        }
        parts.extend([role, combined]);
        // Non-streaming text: all text tokens and the TTS EOS over codec pad.
        let body = self.text(&ids[3..ids.len() - 5])?;
        let eos = self.text(&[t.tts_eos as i32])?;
        let codec_pad = self.codec(&[t.codec_pad])?;
        let codec_bos = self.codec(&[t.codec_bos])?;
        parts.push(Array::concat(&[&body, &eos], 1)?.add(&codec_pad)?);
        parts.push(pad.add(&codec_bos)?);
        let embeds = Array::concat(&parts.iter().collect::<Vec<_>>(), 1)?;
        Ok((embeds, pad))
    }

    /// All 16 codes of one frame from the Talker's hidden state and first
    /// code; returns the codes `[1, 16]` and the next Talker input.
    fn frame(&mut self, hidden: &Array, first: &Array, pad: &Array) -> Result<(Array, Array)> {
        let l = hidden.shape()[1];
        let last = hidden.slice_axis(1, l - 1, l)?;
        let mut cache = new_caches(self.model.predictor.layers.len());
        let mut codes = vec![first.clone()];
        for group in 0..CODE_GROUPS - 1 {
            let inputs = if group == 0 {
                Array::concat(
                    &[&last, &embed(&self.model.talker.codec_embedding, first)?],
                    1,
                )?
            } else {
                embed(
                    &self.model.predictor.codec_embedding[group - 1],
                    &codes[group],
                )?
            };
            let logits = self.model.predictor(&inputs, group, &mut cache)?;
            codes.push(self.sampler.sample(&logits, None, &[])?);
        }
        let all = Array::concat(&codes.iter().collect::<Vec<_>>(), 1)?;
        let mut embeds = vec![embed(&self.model.talker.codec_embedding, &codes[0])?];
        for (table, code) in self.model.predictor.codec_embedding.iter().zip(&codes[1..]) {
            embeds.push(embed(table, code)?);
        }
        let sum = Array::concat(&embeds.iter().collect::<Vec<_>>(), 1)?.sum(&[1], true)?;
        Ok((all, pad.add(&sum)?))
    }
}

impl TtsTrait for QwenTts {
    fn contract(&self) -> Result<TtsContract> {
        Ok(TtsContract {
            sample_rate: codec_config::SAMPLE_RATE,
            mel_frames_per_second: 0,
            semantic_frames_per_second: 0,
        })
    }

    fn synthesize(&mut self, request: &TtsRequest<'_>) -> Result<GeneratedAudio> {
        admit_request(&self.config, request)?;
        let stream = self.ops.stream().clone();
        self.sampler.seed(request.seed)?;
        let (mut embeds, pad) = self.prefill(request)?;
        stream.eval(&[embeds.clone(), pad.clone()])?;
        self.ops.prepare_rope(embeds.shape()[1])?;
        let eos = self.config.tokens.codec_eos;
        let eos = Array::from_i32(&stream, &[], &[eos as i32])?.cast(apxinf_mlx::MlxDType::U32)?;
        // Codec ids 2048..3072 except EOS are control tokens; EOS is also
        // suppressed for the first two frames.
        let control: Vec<u32> = (VOCAB as u32 - 1024..VOCAB as u32)
            .filter(|&i| i != self.config.tokens.codec_eos)
            .collect();
        let early: Vec<u32> = control
            .iter()
            .copied()
            .chain([self.config.tokens.codec_eos])
            .collect();
        let limit = self.config.sampling.max_frames;
        let mut cache = new_caches(self.model.talker.layers.len());
        let mut history = sample::History::new(&stream, VOCAB)?;
        let (frames, complete) = collect_frames(limit, |step| {
            let (logits, hidden) = self.model.talker(&embeds, &mut cache)?;
            let suppress = if step < 2 { &early } else { &control };
            let token = self.sampler.sample(&logits, Some(&history), suppress)?;
            token.eval()?;
            history.push(&token)?;
            let stop = token.equal(&eos)?.to_bytes()?;
            if stop[0] != 0 {
                return Ok(None);
            }
            let (codes, next) = self.frame(&hidden, &token, &pad)?;
            stream.eval(&[next.clone(), codes.clone()])?;
            embeds = next;
            Ok(Some(codes))
        })?;
        if frames.is_empty() {
            return Err(Error::Other("Qwen3-TTS produced no codec frames".into()));
        }
        // [F] x [1, 16] -> [1, 16, F].
        let codes = Array::concat(&frames.iter().collect::<Vec<_>>(), 0)?
            .reshape(&[1, frames.len(), CODE_GROUPS])?
            .transpose(&[0, 2, 1])?;
        drop(cache);
        Ok(GeneratedAudio {
            sample_rate: codec_config::SAMPLE_RATE,
            channels: 1,
            samples: self.decoder.decode(&codes)?,
            complete,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_caps_keep_the_final_complete_frame() {
        for limit in [1, 2, 4] {
            let (frames, complete) =
                collect_frames(limit, |step| Ok(Some([step; CODE_GROUPS]))).unwrap();
            assert_eq!(frames.len(), limit);
            assert_eq!(frames.last(), Some(&[limit - 1; CODE_GROUPS]));
            assert!(!complete);
        }
    }

    #[test]
    fn eos_finishes_without_appending_a_frame() {
        let mut steps = Vec::new();
        let (frames, complete) = collect_frames(5, |step| {
            steps.push(step);
            Ok((step < 2).then_some([step; CODE_GROUPS]))
        })
        .unwrap();
        assert_eq!(steps, [0, 1, 2]);
        assert_eq!(frames, [[0; CODE_GROUPS], [1; CODE_GROUPS]]);
        assert!(complete);
    }

    #[test]
    fn eos_on_the_final_allowed_step_is_complete() {
        let (frames, complete) =
            collect_frames(3, |step| Ok((step < 2).then_some([step; CODE_GROUPS]))).unwrap();
        assert_eq!(frames, [[0; CODE_GROUPS], [1; CODE_GROUPS]]);
        assert!(complete);
    }

    #[test]
    fn text_ids_preserve_valid_tokens_and_reject_out_of_vocabulary_ids() {
        assert_eq!(
            text_ids(vec![0, 151_672, 151_673, 151_671, 151_935]).unwrap(),
            [0, 151_672, 151_673, 151_671, 151_935]
        );
        for invalid in [TEXT_VOCAB as u32, i32::MAX as u32 + 1, u32::MAX] {
            let error = text_ids(vec![1, invalid]).unwrap_err().to_string();
            assert!(error.contains(&format!("token {invalid} at position 1")));
        }
    }

    #[test]
    fn automatic_fusion_requires_a_measured_architecture() {
        let requested = admit(&LoadOptions::default()).unwrap();
        assert!(use_predictor_fusion(requested, "applegpu_g16g"));
        assert!(!use_predictor_fusion(Variant::Reference, "applegpu_g16g"));
        for architecture in [
            "applegpu_g13s",
            "applegpu_g14g",
            "applegpu_g16s",
            "",
            "unknown",
        ] {
            assert!(!use_predictor_fusion(requested, architecture));
            assert!(!use_predictor_fusion(Variant::Reference, architecture));
        }
    }

    #[test]
    fn admission_needs_no_device_or_checkpoint() {
        assert_eq!(admit(&LoadOptions::default()).unwrap(), Variant::Auto);
        let reference = LoadOptions {
            model_variant: Some("reference".into()),
            ..LoadOptions::default()
        };
        assert_eq!(admit(&reference).unwrap(), Variant::Reference);
        for options in [
            LoadOptions {
                model_variant: Some("int8".into()),
                ..LoadOptions::default()
            },
            LoadOptions {
                autotune: true,
                ..LoadOptions::default()
            },
        ] {
            assert!(admit(&options).is_err());
        }
    }
}
