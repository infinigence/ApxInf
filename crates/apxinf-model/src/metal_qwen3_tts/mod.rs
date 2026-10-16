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

use crate::speech::{GeneratedAudio, SpeechRequest, SpeechRuntime};
use crate::{LoadOptions, LoadedModel, ModelPrecision};
use apxinf_core::{Backend, Device, Error, Result};
use apxinf_mlx::{Array, MlxBackend};
use apxinf_tokenizer::Tokenizer;
use config::{codec as codec_config, Config, CODE_GROUPS, VOCAB};
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

/// Device-independent option admission, checked before the checkpoint is read.
fn admit(options: &LoadOptions) -> Result<()> {
    if !matches!(options.model_variant.as_deref(), None | Some("auto"))
        || !matches!(
            options.precision,
            ModelPrecision::Auto | ModelPrecision::Bf16
        )
        || options.text_weight_dtype.is_some()
        || options.config.is_some()
        || options.synthetic.is_some()
        || options.calibration_path.is_some()
        || options.tuning_path.is_some()
        || options.autotune
        || options.uniform_fp8_scale.is_some()
        || !options.assets.is_empty()
    {
        return Err(Error::Contract(
            "Qwen3-TTS MLX loads the BF16 checkpoint as stored; variants, CUDA tuning, \
             calibration, synthetic weights and auxiliary assets are unsupported",
        ));
    }
    Ok(())
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
    admit(options)?;
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
    let ops = Ops::new(&stream)?;
    let (talker, predictor) = weights::Talker::load(&stream, path)?;
    let model = Model::new(talker, predictor, &ops)?;
    let decoder = codec::Decoder::new(weights::Codec::load(&stream, path)?, &ops)?;
    Ok(LoadedModel::Speech(Box::new(QwenTts {
        config,
        tokenizer,
        model,
        decoder,
        sampler: sample::Sampler::new(&stream)?,
        ops,
    })))
}

/// Request admission, before any device work.
fn admit_request(config: &Config, request: &SpeechRequest<'_>) -> Result<()> {
    if request.text.trim().is_empty() {
        return Err(Error::Contract("Qwen3-TTS needs nonempty text"));
    }
    if !config.voices.contains_key(&request.voice.to_lowercase()) {
        return Err(Error::Other(format!(
            "Qwen3-TTS has no voice `{}`; supported: {}",
            request.voice,
            config.voices.keys().cloned().collect::<Vec<_>>().join(", ")
        )));
    }
    if !config.supports_language(request.language) {
        return Err(Error::Other(format!(
            "Qwen3-TTS does not support language `{}`",
            request.language
        )));
    }
    Ok(())
}

impl QwenTts {
    fn ids(&self, text: &str) -> Result<Vec<i32>> {
        Ok(self
            .tokenizer
            .encode_with_special_tokens(text, true)?
            .into_iter()
            .map(|id| id as i32)
            .collect())
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
    fn prefill(&self, request: &SpeechRequest<'_>) -> Result<(Array, Array)> {
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
        let voice = self.config.voices[&request.voice.to_lowercase()];
        let prefix = match self.config.language_id(request.language, request.voice) {
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
        if !request.instruction.is_empty() {
            let instruct = format!("<|im_start|>user\n{}<|im_end|>\n", request.instruction);
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
            codes.push(self.sampler.sample(&logits, &[], &[])?);
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

impl SpeechRuntime for QwenTts {
    fn synthesize(&mut self, request: &SpeechRequest<'_>) -> Result<GeneratedAudio> {
        admit_request(&self.config, request)?;
        let stream = self.ops.stream().clone();
        self.sampler.seed(request.seed)?;
        let (mut embeds, pad) = self.prefill(request)?;
        stream.eval(&[embeds.clone(), pad.clone()])?;
        let eos = self.config.tokens.codec_eos;
        // Codec ids 2048..3072 except EOS are control tokens; EOS is also
        // suppressed for the first two frames.
        let control: Vec<u32> = (VOCAB as u32 - 1024..VOCAB as u32)
            .filter(|&i| i != eos)
            .collect();
        let early: Vec<u32> = control.iter().copied().chain([eos]).collect();
        let limit = self.config.sampling.max_frames;
        let mut cache = new_caches(self.model.talker.layers.len());
        let (mut history, mut frames) = (Vec::new(), Vec::new());
        let mut complete = false;
        for step in 0..limit {
            let (logits, hidden) = self.model.talker(&embeds, &mut cache)?;
            let suppress = if step < 2 { &early } else { &control };
            let token = self.sampler.sample(&logits, &history, suppress)?;
            token.eval()?;
            let first = token.to_u32_scalar()?;
            history.push(first);
            if first == eos {
                complete = true;
                break;
            }
            if step == limit - 1 {
                // The capped final first code has no residual codes.
                break;
            }
            let (codes, next) = self.frame(&hidden, &token, &pad)?;
            stream.eval(&[next.clone(), codes.clone()])?;
            frames.push(codes);
            embeds = next;
        }
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
    fn admission_needs_no_device_or_checkpoint() {
        assert!(admit(&LoadOptions::default()).is_ok());
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
