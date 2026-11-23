//! The prepared plan: fixed-shape graph capture, replay, and validation.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use apxinf_core::{Error, Result, Tensor};
use apxinf_cuda_new::transfers;

use super::cache::ConditioningCache;
use super::gpt_decode::{generate_semantic, GptExecutionState};
use super::graph::{BertGraph, CfmChunkGraph, HubertGraph, SovitsGraph, StyleGraph, VocoderGraph};
use super::GptSovitsTts;
use crate::cuda_gpt_sovits::model::conditioning::{ReferenceFeatures, ReferenceKey, TextKey};
use crate::cuda_gpt_sovits::model::{
    owned_tensor, validate_sample_steps, ChunkArgs, GptSovitsModel, SynthesisExecutor,
    MAX_TARGET_CHUNK, REFERENCE_FRAMES, TOTAL_CHUNK,
};
use crate::cuda_gpt_sovits::weights::fp16_tensor;
use crate::tts_trait::{
    gpt_sovits_inputs, GeneratedAudio, GptSovitsInputs, PreparedTts, TtsRequest, TtsSpec,
};

pub struct PreparedGptSovitsTts {
    // Captured nodes reference storage owned by the subnet weights (the
    // vocoder's deferred conv biases in particular), and pointwise launches
    // do not retain their operands, so the plan must keep the whole model
    // alive for as long as it can be replayed.
    model: Arc<GptSovitsModel>,
    spec: TtsSpec,
    cache: Rc<RefCell<ConditioningCache>>,
    gpt_state: RefCell<GptExecutionState>,
    reference_bert: BertGraph,
    target_bert: BertGraph,
    hubert_graph: HubertGraph,
    style_graph: StyleGraph,
    reference_sovits: SovitsGraph,
    target_sovits: SovitsGraph,
    cfm_graphs: HashMap<usize, CfmChunkGraph>,
    vocoder_graph: VocoderGraph,
}

impl GptSovitsTts {
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
        let ctx = &self.model.ctx;
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
                CfmChunkGraph::prepare(ctx, &self.model.cfm, prompt_len, length, spec.sample_steps)
                    .map_err(|error| {
                        Error::Other(format!("prepare CFM chunk {prompt_len}+{length}: {error}"))
                    })?,
            );
        }
        let vocoder_graph = VocoderGraph::prepare(ctx, &self.model.vocoder, spec.target_frames)
            .map_err(|error| Error::Other(format!("prepare vocoder: {error}")))?;
        Ok(PreparedGptSovitsTts {
            model: Arc::clone(&self.model),
            spec: *spec,
            cache: Rc::clone(&self.cache),
            gpt_state: RefCell::new(GptExecutionState::new()),
            reference_bert: BertGraph::prepare(
                ctx,
                &self.model.bert,
                spec.reference_bert_token_count,
            )
            .map_err(|e| Error::Other(format!("prepare reference BERT: {e}")))?,
            target_bert: BertGraph::prepare(ctx, &self.model.bert, spec.bert_token_count)
                .map_err(|e| Error::Other(format!("prepare target BERT: {e}")))?,
            hubert_graph: HubertGraph::prepare(
                ctx,
                &self.model.hubert,
                spec.reference_audio_samples,
            )
            .map_err(|e| Error::Other(format!("prepare HuBERT: {e}")))?,
            style_graph: StyleGraph::prepare(ctx, &self.model.sovits, spec.reference_spec_frames)
                .map_err(|e| Error::Other(format!("prepare style: {e}")))?,
            reference_sovits: SovitsGraph::prepare(
                ctx,
                &self.model.sovits,
                self.model
                    .hubert
                    .semantic_frames(spec.reference_audio_samples)
                    .map_err(|e| Error::Other(format!("prepare reference SoVITS: {e}")))?,
                spec.reference_phone_count,
            )?,
            target_sovits: SovitsGraph::prepare(
                ctx,
                &self.model.sovits,
                spec.target_frames / 4,
                spec.phone_count,
            )
            .map_err(|e| Error::Other(format!("prepare target SoVITS: {e}")))?,
            cfm_graphs,
            vocoder_graph,
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
        let generated = graph.replay(&self.model.ctx, ids, word2ph, phones)?;
        let value = owned_tensor(&self.model.ctx, &generated)?;
        if cache_text {
            self.cache.borrow_mut().insert_text(key, value.clone());
        }
        Ok(value)
    }
}

impl SynthesisExecutor for PreparedGptSovitsTts {
    fn reference_features(&self, inputs: &GptSovitsInputs<'_>) -> Result<Rc<ReferenceFeatures>> {
        let key = ReferenceKey::new(inputs, true);
        if let Some(value) = self.cache.borrow_mut().reference(&key) {
            return Ok(value);
        }
        let ctx = &self.model.ctx;
        let mel = fp16_tensor(
            ctx,
            &[self.spec.reference_frames, 100],
            inputs.reference_mel,
        )?;
        let spec = fp16_tensor(
            ctx,
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
        let codes = self.hubert_graph.replay(ctx, inputs.reference_audio_16k)?;
        let host = transfers::to_cpu(&codes)?;
        let prompt = host.as_i32()?.iter().map(|x| *x as u32).collect();
        let generated_style = self.style_graph.replay(ctx, &spec)?;
        let style = owned_tensor(ctx, &generated_style)?;
        let generated_features =
            self.reference_sovits
                .replay(ctx, &codes, inputs.reference_phone_ids, &style)?;
        let features = owned_tensor(ctx, &generated_features)?;
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

    fn target_bert(&self, inputs: &GptSovitsInputs<'_>) -> Result<Tensor> {
        self.cached_bert(
            &self.target_bert,
            inputs.bert_token_ids,
            inputs.word2ph,
            inputs.phone_ids.len(),
            true,
        )
    }

    fn expected_target_frames(&self) -> Option<usize> {
        Some(self.spec.target_frames)
    }

    fn generate_semantic(
        &self,
        phones: &[u32],
        bert: &Tensor,
        prompt: &[u32],
        seed: u64,
        top_k: usize,
        top_p: f32,
        temperature: f32,
        repetition_penalty: f32,
        max_new_tokens: usize,
    ) -> Result<Vec<u32>> {
        generate_semantic(
            self.model.gpt.as_ref(),
            &self.model.ctx,
            &mut self.gpt_state.borrow_mut(),
            phones,
            bert,
            prompt,
            seed,
            top_k,
            top_p,
            temperature,
            repetition_penalty,
            max_new_tokens,
        )
    }

    fn target_features(
        &self,
        codes: &Tensor,
        phones: &[u32],
        style: &Tensor,
        _speed_factor: f32,
    ) -> Result<Tensor> {
        // `validate_request` rejects profiles with speed_factor != 1.0
        // before the orchestration runs.
        self.target_sovits
            .replay(&self.model.ctx, codes, phones, style)
    }

    fn cfm_chunk(&self, args: ChunkArgs<'_>, _steps: usize, seed: u64) -> Result<Tensor> {
        let graph = self.cfm_graphs.get(&args.length).ok_or_else(|| {
            Error::Other(format!(
                "missing prepared CFM chunk profile {}",
                args.length
            ))
        })?;
        graph.replay(
            &self.model.ctx,
            args.prompt_features,
            args.target,
            args.prompt_mel,
            seed,
        )
    }

    fn vocode(&self, mel: &Tensor) -> Result<Vec<f32>> {
        self.vocoder_graph.replay(&self.model.ctx, mel)
    }
}

impl PreparedTts for PreparedGptSovitsTts {
    fn spec(&self) -> &TtsSpec {
        &self.spec
    }

    fn run(&self, request: &TtsRequest<'_>) -> Result<GeneratedAudio> {
        let inputs = gpt_sovits_inputs(request)?;
        self.validate_request(inputs)?;
        self.model.synthesize_with(request, self)
    }
}

#[cfg(all(test, feature = "cuda"))]
mod vocoder_lifecycle_tests {
    use super::*;
    use crate::cuda_gpt_sovits::ops;

    /// The captured vocoder graph reads the deferred conv biases owned by the
    /// `Vocoder` weights; replaying it after the model has been dropped must
    /// still be well-defined, which only holds while the plan retains the
    /// model. Requires a real checkpoint because the capture runs the full
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
            GptSovitsTts::load_with_autotune(&directory, 0, false).expect("load checkpoint");
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
        let ctx = &prepared.model.ctx;
        let mel = ops::alloc(ctx, &[frames, 100]).expect("device mel");
        ops::fill_zero(ctx, &mel).expect("zero mel");
        let samples = prepared
            .vocoder_graph
            .replay(ctx, &mel)
            .expect("replay after the model was dropped");
        assert_eq!(samples.len(), frames * 480);
    }
}
