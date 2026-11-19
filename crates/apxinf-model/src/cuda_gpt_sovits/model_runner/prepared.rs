//! The prepared plan: fixed-shape graph capture, replay, and validation.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use apxinf_core::{Error, Result, Tensor};
use apxinf_cuda_new::{transfers, CudaContext as Context};

use super::cache::{ConditioningCache, ReferenceFeatures, ReferenceKey, TextKey};
use super::graph::{BertGraph, CfmChunkGraph, HubertGraph, SovitsGraph, StyleGraph, VocoderGraph};
use crate::cuda_gpt_sovits::model::{
    denormalize_mel, max_semantic_tokens, mono_audio, owned_tensor, reference_feature_window,
    run_chunked, sampling_of, validate_sample_steps, Bert, Gpt, GptSovitsModel, Hubert, Sovits,
    Vocoder, MAX_TARGET_CHUNK, REFERENCE_FRAMES, SPEC_MAX, SPEC_MIN, TOTAL_CHUNK,
};
use crate::cuda_gpt_sovits::ops;
use crate::cuda_gpt_sovits::weights::fp16_tensor;
use crate::tts_trait::{
    gpt_sovits_inputs, GeneratedAudio, GptSovitsInputs, PreparedTts, TtsRequest, TtsSpec,
};

pub struct PreparedGptSovitsTts {
    _bert: Arc<Bert>,
    _hubert: Arc<Hubert>,
    _sovits: Arc<Sovits>,
    // Captured vocoder nodes reference the deferred conv biases owned by the
    // `Vocoder`, and pointwise launches do not retain their operands, so the
    // plan must keep the vocoder alive for as long as it can be replayed.
    _vocoder: Arc<Vocoder>,
    spec: TtsSpec,
    cache: Rc<RefCell<ConditioningCache>>,
    ctx: Arc<Context>,
    reference_bert: BertGraph,
    target_bert: BertGraph,
    hubert_graph: HubertGraph,
    style_graph: StyleGraph,
    reference_sovits: SovitsGraph,
    target_sovits: SovitsGraph,
    gpt: Arc<Gpt>,
    cfm_graphs: HashMap<usize, CfmChunkGraph>,
    vocoder_graph: VocoderGraph,
    mel_scale: Tensor,
    mel_shift: Tensor,
    mel_one: Tensor,
    mel_zero: Tensor,
}

impl GptSovitsModel {
    /// Build the fixed-shape plan for `spec`.
    /// [`crate::tts_trait::TtsTrait::prepare`] wraps the result in the
    /// model-neutral [`PreparedTts`] object; the vocoder lifecycle test calls
    /// this directly for the concrete type.
    pub(super) fn prepare_plan(&self, spec: &TtsSpec) -> Result<PreparedGptSovitsTts> {
        if spec.phone_count == 0
            || spec.bert_token_count < 3
            || spec.reference_phone_count == 0
            || spec.reference_bert_token_count < 3
            || spec.reference_audio_samples < 400
            || spec.reference_frames == 0
            || spec.reference_spec_frames == 0
            || spec.target_frames == 0
            || validate_sample_steps(spec.sample_steps).is_err()
            || spec.target_frames % 4 != 0
        {
            return Err(Error::Other(format!(
                "invalid GPT-SoVITS prepared profile: {spec:?}"
            )));
        }
        let prompt_len = spec.reference_frames.min(REFERENCE_FRAMES);
        let chunk = (TOTAL_CHUNK - prompt_len).min(MAX_TARGET_CHUNK);
        let mut lengths = Vec::new();
        let mut start = 0usize;
        while start < spec.target_frames {
            let length = chunk.min(spec.target_frames - start);
            if !lengths.contains(&length) {
                lengths.push(length);
            }
            start += length;
        }
        let mut cfm_graphs = HashMap::new();
        for length in lengths {
            cfm_graphs.insert(
                length,
                CfmChunkGraph::prepare(&self.ctx, &self.cfm, prompt_len, length, spec.sample_steps)
                    .map_err(|error| {
                        Error::Other(format!("prepare CFM chunk {prompt_len}+{length}: {error}"))
                    })?,
            );
        }
        let vocoder_graph = VocoderGraph::prepare(&self.ctx, &self.vocoder, spec.target_frames)
            .map_err(|error| Error::Other(format!("prepare vocoder: {error}")))?;
        let mel_scale = fp16_tensor(&self.ctx, &[100], &vec![(SPEC_MAX - SPEC_MIN) / 2.0; 100])?;
        let mel_shift = fp16_tensor(&self.ctx, &[100], &vec![SPEC_MIN; 100])?;
        let mel_one = fp16_tensor(&self.ctx, &[100], &[1.; 100])?;
        let mel_zero = fp16_tensor(&self.ctx, &[100], &[0.; 100])?;
        Ok(PreparedGptSovitsTts {
            _bert: Arc::clone(&self.bert),
            _hubert: Arc::clone(&self.hubert),
            _sovits: Arc::clone(&self.sovits),
            _vocoder: Arc::clone(&self.vocoder),
            spec: *spec,
            cache: Rc::clone(&self.cache),
            ctx: Arc::clone(&self.ctx),
            reference_bert: BertGraph::prepare(
                &self.ctx,
                &self.bert,
                spec.reference_bert_token_count,
            )
            .map_err(|e| Error::Other(format!("prepare reference BERT: {e}")))?,
            target_bert: BertGraph::prepare(&self.ctx, &self.bert, spec.bert_token_count)
                .map_err(|e| Error::Other(format!("prepare target BERT: {e}")))?,
            hubert_graph: HubertGraph::prepare(
                &self.ctx,
                &self.hubert,
                spec.reference_audio_samples,
            )
            .map_err(|e| Error::Other(format!("prepare HuBERT: {e}")))?,
            style_graph: StyleGraph::prepare(&self.ctx, &self.sovits, spec.reference_spec_frames)
                .map_err(|e| Error::Other(format!("prepare style: {e}")))?,
            reference_sovits: SovitsGraph::prepare(
                &self.ctx,
                &self.sovits,
                self.hubert
                    .semantic_frames(spec.reference_audio_samples)
                    .map_err(|e| Error::Other(format!("prepare reference SoVITS: {e}")))?,
                spec.reference_phone_count,
            )?,
            target_sovits: SovitsGraph::prepare(
                &self.ctx,
                &self.sovits,
                spec.target_frames / 4,
                spec.phone_count,
            )
            .map_err(|e| Error::Other(format!("prepare target SoVITS: {e}")))?,
            gpt: Arc::clone(&self.gpt),
            cfm_graphs,
            vocoder_graph,
            mel_scale,
            mel_shift,
            mel_one,
            mel_zero,
        })
    }
}

impl PreparedGptSovitsTts {
    fn validate_request(&self, inputs: &GptSovitsInputs<'_>) -> Result<()> {
        let actual = TtsSpec {
            phone_count: inputs.phone_ids.len(),
            bert_token_count: inputs.bert_token_ids.len(),
            reference_phone_count: inputs.reference_phone_ids.len(),
            reference_bert_token_count: inputs.reference_bert_token_ids.len(),
            reference_audio_samples: inputs.reference_audio_16k.len(),
            reference_frames: inputs.reference_mel.len() / 100,
            reference_spec_frames: inputs.reference_spec.len() / 704,
            target_frames: self.spec.target_frames,
            sample_steps: inputs.sample_steps,
        };
        if inputs.reference_mel.len() != self.spec.reference_frames * 100
            || inputs.reference_spec.len() != self.spec.reference_spec_frames * 704
            || actual.phone_count != self.spec.phone_count
            || actual.bert_token_count != self.spec.bert_token_count
            || actual.reference_phone_count != self.spec.reference_phone_count
            || actual.reference_bert_token_count != self.spec.reference_bert_token_count
            || actual.reference_audio_samples != self.spec.reference_audio_samples
            || actual.reference_frames != self.spec.reference_frames
            || actual.reference_spec_frames != self.spec.reference_spec_frames
            || actual.sample_steps != self.spec.sample_steps
        {
            return Err(Error::Other(format!(
                "tts request does not match prepared profile: expected {:?}, got {:?}",
                self.spec, actual
            )));
        }
        if inputs.cfg_rate > 1e-5 {
            return Err(Error::Other(
                "gpt_sovits v5turbo does not support classifier-free guidance".into(),
            ));
        }
        if (inputs.speed_factor - 1.0).abs() > 1e-6 {
            return Err(Error::Other(
                "prepared gpt_sovits profiles require speed_factor=1.0; use eager synthesis for other speeds"
                    .into(),
            ));
        }
        Ok(())
    }

    fn cached_bert(
        &self,
        graph: &BertGraph,
        ids: &[u32],
        word2ph: &[u32],
        phones: usize,
        cache_text: bool,
    ) -> Result<Tensor> {
        let key = TextKey::new(ids, word2ph, phones, true);
        if cache_text {
            if let Some(value) = self.cache.borrow_mut().text(&key) {
                return Ok(value);
            }
        }
        let generated = graph.replay(&self.ctx, ids, word2ph, phones)?;
        let value = owned_tensor(&self.ctx, &generated)?;
        if cache_text {
            self.cache.borrow_mut().insert_text(key, value.clone());
        }
        Ok(value)
    }

    fn reference_features(&self, inputs: &GptSovitsInputs<'_>) -> Result<Rc<ReferenceFeatures>> {
        let key = ReferenceKey::new(inputs, true);
        if let Some(value) = self.cache.borrow_mut().reference(&key) {
            return Ok(value);
        }
        let mel = fp16_tensor(
            &self.ctx,
            &[self.spec.reference_frames, 100],
            inputs.reference_mel,
        )?;
        let spec = fp16_tensor(
            &self.ctx,
            &[self.spec.reference_spec_frames, 704],
            inputs.reference_spec,
        )?;
        let bert = self.cached_bert(
            &self.reference_bert,
            inputs.reference_bert_token_ids,
            inputs.reference_word2ph,
            inputs.reference_phone_ids.len(),
            false,
        )?;
        let codes = self
            .hubert_graph
            .replay(&self.ctx, inputs.reference_audio_16k)?;
        let host = transfers::to_cpu(&codes)?;
        let prompt = host.as_i32()?.iter().map(|x| *x as u32).collect();
        let generated_style = self.style_graph.replay(&self.ctx, &spec)?;
        let style = owned_tensor(&self.ctx, &generated_style)?;
        let generated_features =
            self.reference_sovits
                .replay(&self.ctx, &codes, inputs.reference_phone_ids, &style)?;
        let features = owned_tensor(&self.ctx, &generated_features)?;
        let value = Rc::new(ReferenceFeatures {
            bert,
            prompt,
            style,
            features,
            mel,
        });
        self.cache
            .borrow_mut()
            .insert_reference(key, Rc::clone(&value));
        Ok(value)
    }

    fn run_prepared(&self, request: &TtsRequest<'_>) -> Result<GeneratedAudio> {
        let inputs = gpt_sovits_inputs(request)?;
        let (top_k, top_p, temperature, repetition_penalty, max_seconds) = sampling_of(request);
        self.validate_request(inputs)?;
        // NVTX segment timeline for nsys. This layer is a black box of graph
        // replays, so the four stages are only separable with host-side marks;
        // without the `nvtx` feature each range is a zero-cost no-op.
        let _tts_reference_range = crate::profiling::trace::range("tts.reference");
        let reference = self.reference_features(inputs)?;
        drop(_tts_reference_range);
        let reference_mel = reference.mel.clone();
        let reference_bert = reference.bert.clone();
        let _tts_bert_range = crate::profiling::trace::range("tts.bert");
        let target_bert = self.cached_bert(
            &self.target_bert,
            inputs.bert_token_ids,
            inputs.word2ph,
            inputs.phone_ids.len(),
            true,
        )?;
        drop(_tts_bert_range);
        let bert = ops::concat_rows(&self.ctx, &reference_bert, &target_bert)?;
        let mut phones = inputs.reference_phone_ids.to_vec();
        phones.extend_from_slice(inputs.phone_ids);
        let prompt = &reference.prompt;
        let _tts_gpt_range = crate::profiling::trace::range("tts.gpt");
        let max_new_tokens = max_semantic_tokens(max_seconds);
        let generated = self
            .gpt
            .generate(
                &self.ctx,
                &phones,
                &bert,
                &prompt,
                request.seed,
                top_k,
                top_p,
                temperature,
                repetition_penalty,
                max_new_tokens,
            )
            .map_err(|error| Error::Other(format!("prepared gpt: {error}")))?;
        drop(_tts_gpt_range);
        if generated.is_empty() {
            return Err(Error::Other("semantic decoder produced no tokens".into()));
        }
        let generated_frames = generated
            .len()
            .checked_mul(4)
            .ok_or_else(|| Error::Other("prepared target length overflow".into()))?;
        if generated_frames != self.spec.target_frames {
            return Err(Error::Other(format!(
                "prepared target frame mismatch: graph expects {}, GPT produced {generated_frames} frames from {} semantic tokens",
                self.spec.target_frames, generated.len()
            )));
        }
        let target_ids = generated
            .iter()
            .map(|value| *value as i32)
            .collect::<Vec<_>>();
        let target_codes = transfers::to_cuda(
            &Tensor::from_i32(vec![target_ids.len()], &target_ids)?,
            self.ctx.device_id(),
        )?;
        let ge = reference.style.clone();
        let reference_features = reference.features.clone();
        let _tts_sovits_range = crate::profiling::trace::range("tts.sovits");
        let target_features =
            self.target_sovits
                .replay(&self.ctx, &target_codes, inputs.phone_ids, &ge)?;
        drop(_tts_sovits_range);
        if target_features.shape().dims() != [self.spec.target_frames, 512] {
            return Err(Error::Other(format!(
                "prepared target frame mismatch: graph expects {}, GPT produced {} frames from {} semantic tokens",
                self.spec.target_frames,
                target_features.shape().dims()[0],
                generated.len()
            )));
        }
        let decoded_frames = reference_features.shape().dims()[0];
        let reference_features = match reference_feature_window(
            decoded_frames,
            self.spec.reference_spec_frames,
            self.spec.reference_frames,
        ) {
            Some(offset) => ops::slice_rows(
                &self.ctx,
                &reference_features,
                offset,
                self.spec.reference_frames,
            )?,
            None => reference_features,
        };
        let reference_frames = self
            .spec
            .reference_frames
            .min(reference_features.shape().dims()[0])
            .min(REFERENCE_FRAMES);
        let original_mel = ops::slice_rows(
            &self.ctx,
            &reference_mel,
            self.spec.reference_frames - reference_frames,
            reference_frames,
        )?;
        let original_features = ops::slice_rows(
            &self.ctx,
            &reference_features,
            reference_features.shape().dims()[0] - reference_frames,
            reference_frames,
        )?;
        // The `[target_frames, 512]` check above already pinned the row count
        // to `self.spec.target_frames`, so chunking on `target_features` rows
        // reproduces the previous `self.spec.target_frames` loop exactly.
        let mel = run_chunked(
            &self.ctx,
            &original_mel,
            &original_features,
            &target_features,
            reference_frames,
            |args| {
                let graph = self.cfm_graphs.get(&args.length).ok_or_else(|| {
                    Error::Other(format!(
                        "missing prepared CFM chunk profile {}",
                        args.length
                    ))
                })?;
                let _tts_cfm_range = crate::profiling::trace::range("tts.cfm.chunk");
                let generated = graph.replay(
                    &self.ctx,
                    args.prompt_features,
                    args.target,
                    args.prompt_mel,
                    request.seed.wrapping_add(args.chunk_index as u64),
                )?;
                drop(_tts_cfm_range);
                Ok(generated)
            },
        )?;
        let mel = denormalize_mel(
            &self.ctx,
            &mel,
            &self.mel_one,
            &self.mel_scale,
            &self.mel_zero,
            &self.mel_shift,
        )?;
        let _tts_vocoder_range = crate::profiling::trace::range("tts.vocoder");
        let samples = self.vocoder_graph.replay(&self.ctx, &mel)?;
        drop(_tts_vocoder_range);
        // The public pipeline adds `fragment_interval` silence; the acoustic
        // graph itself must not append an implicit pause.
        Ok(mono_audio(samples, generated.len(), max_new_tokens))
    }
}

impl PreparedTts for PreparedGptSovitsTts {
    fn spec(&self) -> &TtsSpec {
        &self.spec
    }

    fn run(&self, request: &TtsRequest<'_>) -> Result<GeneratedAudio> {
        self.run_prepared(request)
    }
}

#[cfg(all(test, feature = "cuda"))]
mod vocoder_lifecycle_tests {
    use super::*;

    /// The captured vocoder graph reads the deferred conv biases owned by the
    /// `Vocoder` weights; replaying it after the model has been dropped must
    /// still be well-defined, which only holds while the plan retains the
    /// vocoder. Requires a real checkpoint because the capture runs the full
    /// forward once.
    #[test]
    #[ignore = "requires CUDA and APXINF_GSV_V5_CHECKPOINT with a real GPT-SoVITS V5 checkpoint"]
    fn vocoder_replay_survives_dropping_the_model() {
        let directory = std::path::PathBuf::from(
            std::env::var_os("APXINF_GSV_V5_CHECKPOINT")
                .expect("set APXINF_GSV_V5_CHECKPOINT to the GPT-SoVITS V5 checkpoint"),
        );
        let frames = 128; // prepare() admits multiples of four
        let model =
            GptSovitsModel::load_with_autotune(&directory, 0, false).expect("load checkpoint");
        let spec = TtsSpec {
            phone_count: 8,
            bert_token_count: 8,
            reference_phone_count: 8,
            reference_bert_token_count: 8,
            reference_audio_samples: 16_000,
            reference_frames: 500,
            reference_spec_frames: 500,
            target_frames: frames,
            sample_steps: 4,
        };
        let prepared = model.prepare_plan(&spec).expect("prepare");
        drop(model);
        let mel = ops::alloc(&prepared.ctx, &[frames, 100]).expect("device mel");
        ops::fill_zero(&prepared.ctx, &mel).expect("zero mel");
        let samples = prepared
            .vocoder_graph
            .replay(&prepared.ctx, &mel)
            .expect("replay after the model was dropped");
        assert_eq!(samples.len(), frames * 480);
    }
}
