//! GPT-SoVITS V5 runtime orchestration.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;

use apxinf_core::{Error, Result, Tensor};
use apxinf_cuda_new::{ops as cuda_ops, transfers, CudaContext as Context, PreparedPhase};

use super::bert::Bert;
use super::cache::{ConditioningCache, ReferenceFeatures, ReferenceKey, TextKey};
use super::cfm::{Cfm, StepModulation};
use super::config::GptSovitsConfig;
use super::gpt::Gpt;
use super::hubert::Hubert;
use super::ops;
use super::sovits::Sovits;
use super::vocoder::Vocoder;
use super::weights::Weights;
use crate::tts_trait::{
    gpt_sovits_inputs, GeneratedAudio, GptSovitsInputs, PreparedTts, TtsCacheConfig, TtsCacheInfo,
    TtsContract, TtsRequest, TtsTrait, TtsSampling, TtsSpec,
};

/// Sampling defaults used when a [`TtsRequest`] leaves its knobs unset. They
/// mirror the official GPT-SoVITS V5 inference settings.
const DEFAULT_TOP_K: usize = 5;
const DEFAULT_TOP_P: f32 = 1.0;
const DEFAULT_TEMPERATURE: f32 = 1.0;
const DEFAULT_REPETITION_PENALTY: f32 = 1.35;
const DEFAULT_MAX_SECONDS: f32 = 30.0;

/// Resolve a request's sampling knobs: explicit values win, otherwise the
/// family defaults above.
fn sampling_of(request: &TtsRequest<'_>) -> (usize, f32, f32, f32, f32) {
    let s: &TtsSampling<'_> = &request.sampling;
    (
        s.top_k.unwrap_or(DEFAULT_TOP_K),
        s.top_p.unwrap_or(DEFAULT_TOP_P),
        s.temperature.unwrap_or(DEFAULT_TEMPERATURE),
        s.repetition_penalty.unwrap_or(DEFAULT_REPETITION_PENALTY),
        s.max_seconds.unwrap_or(DEFAULT_MAX_SECONDS),
    )
}

const SPEC_MIN: f32 = -12.0;
const SPEC_MAX: f32 = 2.0;
const OUTPUT_SAMPLE_RATE: u32 = 48000;
const CFM_REFERENCE_FRAMES: usize = 500;
/// Semantic tokens per second and the hard cap on one request's generation.
const SEMANTIC_TOKENS_PER_SECOND: f32 = 25.0;
const MAX_SEMANTIC_TOKENS: usize = 1500;

/// Semantic-token budget for `max_seconds` (at least one second's worth).
fn max_semantic_tokens(max_seconds: f32) -> usize {
    ((max_seconds.max(1.0) * SEMANTIC_TOKENS_PER_SECOND) as usize).min(MAX_SEMANTIC_TOKENS)
}

/// Wrap vocoder output. `Gpt::generate` stops before pushing EOS, so it
/// returns fewer than `max_new_tokens` tokens exactly when EOS was sampled;
/// a full budget means the length limit cut generation short.
fn mono_audio(samples: Vec<f32>, generated_tokens: usize, max_new_tokens: usize) -> GeneratedAudio {
    GeneratedAudio {
        sample_rate: OUTPUT_SAMPLE_RATE,
        channels: 1,
        samples,
        complete: generated_tokens < max_new_tokens,
    }
}

/// Start offset of the reference-feature window paired with the prompt mel.
///
/// Official `synthesize_v5_mel` first truncates both streams to
/// `min(mel_frames, feature_frames)`, then keeps their last 500 frames. The
/// public request only carries the already-capped prompt mel, so the original
/// mel length is reconstructed from the spectrogram (`2 * spec_frames`, whose
/// hop is twice the mel hop). A zero offset still needs an explicit slice:
/// short prompts must use the head of the feature stream, not its tail.
fn reference_feature_window(
    decoded_frames: usize,
    reference_spec_frames: usize,
    reference_frames: usize,
) -> Option<usize> {
    let limit = reference_spec_frames.saturating_mul(2).min(decoded_frames);
    let offset = limit.saturating_sub(reference_frames);
    (offset + reference_frames <= decoded_frames).then_some(offset)
}
const CFM_MAX_TARGET_CHUNK: usize = 640;
const CFM_TOTAL_CHUNK: usize = 1000;
const CFM_ROLLING_TAIL: usize = 32;
const MIB: usize = 1024 * 1024;

// Upstream evaluates (x + 1) / 2 * 14 - 12 in FP16. The addition before
// scaling and the subtraction after scaling each round independently.
// For FP16 x, dividing round(x + 1) by two is exact, so the two scales can
// combine to seven while preserving those rounding boundaries.
fn denormalize_mel(
    ctx: &Context,
    input: &Tensor,
    one: &Tensor,
    scale: &Tensor,
    zero: &Tensor,
    minimum: &Tensor,
) -> Result<Tensor> {
    let shifted = ops::add_bias(ctx, input, one)?;
    let scaled = ops::affine(ctx, &shifted, scale, zero)?;
    ops::add_bias(ctx, &scaled, minimum)
}

fn owned_tensor(ctx: &Context, source: &Tensor) -> Result<Tensor> {
    let value = ctx.allocate_output(source.shape().clone(), source.dtype())?;
    transfers::copy_cuda_to_cuda(ctx, source, &value)?;
    Ok(value)
}

fn cfm_workspace_bytes(prompt_len: usize, target_len: usize) -> Result<usize> {
    let total = prompt_len
        .checked_add(target_len)
        .ok_or_else(|| Error::Other("prepared CFM length overflow".into()))?;
    // One reusable solver traversal retains one step's intermediates and native
    // executions, independently of sample_steps. Initialization has its own arena.
    (2usize * 1024 * MIB)
        .checked_mul(total)
        .and_then(|bytes| bytes.checked_add(999))
        .map(|bytes| (bytes / 1000).max(512 * MIB))
        .ok_or_else(|| Error::Other("prepared CFM workspace size overflow".into()))
}

fn vocoder_workspace_bytes(frames: usize) -> Result<usize> {
    // A 500-frame vocoder profile measures below 2 GiB on Orin. Its retained
    // activations grow linearly with mel frames.
    (2usize * 1024 * MIB)
        .checked_mul(frames)
        .and_then(|bytes| bytes.checked_add(499))
        .map(|bytes| (bytes / 500).max(512 * MIB))
        .ok_or_else(|| Error::Other("prepared vocoder workspace size overflow".into()))
}

pub struct GptSovitsModel {
    ctx: Arc<Context>,
    bert: Arc<Bert>,
    hubert: Arc<Hubert>,
    gpt: Arc<Gpt>,
    sovits: Arc<Sovits>,
    cfm: Arc<Cfm>,
    vocoder: Arc<Vocoder>,
    config: GptSovitsConfig,
    cache: Rc<RefCell<ConditioningCache>>,
}

struct CfmChunkGraph {
    conditioning: PreparedPhase,
    solver: PreparedPhase,
    prompt_features: Tensor,
    target_features: Tensor,
    prompt_mel: Tensor,
    output: Tensor,
    // Both allocations are created outside the solver arena. Captured nodes
    // retain their addresses, so the plan must retain the owning tensors.
    _state: Tensor,
    _static_condition: Tensor,
    modulation: StepModulation,
    steps: usize,
    cfm: Arc<Cfm>,
    noise_generator: RefCell<Box<dyn apxinf_core::NormalGenerator>>,
}

impl CfmChunkGraph {
    fn prepare(
        ctx: &Context,
        cfm: &Arc<Cfm>,
        prompt_len: usize,
        target_len: usize,
        steps: usize,
    ) -> Result<Self> {
        let total = prompt_len
            .checked_add(target_len)
            .ok_or_else(|| Error::Other("prepared CFM length overflow".into()))?;
        let prompt_features = ops::alloc(ctx, &[prompt_len, 512])?;
        let target_features = ops::alloc(ctx, &[target_len, 512])?;
        let prompt_mel = ops::alloc(ctx, &[prompt_len, 100])?;
        let noise = ops::alloc(ctx, &[total, 100])?;
        let x = ops::alloc(ctx, &[total, 100])?;
        let static_condition = ops::alloc(ctx, &[total, super::config::CFM_DIM])?;
        let output = ops::alloc(ctx, &[target_len, 100])?;
        let noise_generator = RefCell::new(apxinf_cuda_new::sampling::create_normal_generator(
            ctx,
            noise.clone(),
        )?);
        let conditioning = PreparedPhase::prepare_and_capture(
            ctx,
            cuda_ops::ExecutionSession::with_capacity_pinning_prepared_buffers(128 * MIB, ctx.device_id())?,
            || {
                let (condition, initial) = cfm.prepared_chunk_inputs(
                    ctx,
                    &prompt_features,
                    &target_features,
                    &prompt_mel,
                    &noise,
                )?;
                transfers::copy_cuda_to_cuda(ctx, &condition, &static_condition)?;
                transfers::copy_cuda_to_cuda(ctx, &initial, &x)
            },
        )?;
        let modulation = cfm.allocate_step_modulation(ctx)?;
        cfm.bind_step_modulation(ctx, 0, &modulation)?;
        let solver = PreparedPhase::prepare_and_capture(
            ctx,
            cuda_ops::ExecutionSession::with_capacity_pinning_prepared_buffers(
                cfm_workspace_bytes(prompt_len, target_len)?,
                ctx.device_id(),
            )?,
            || {
                let next = cfm.prepared_solver_step(
                    ctx,
                    &x,
                    &static_condition,
                    &modulation,
                    steps,
                    prompt_len,
                    target_len,
                )?;
                transfers::copy_cuda_to_cuda(ctx, &next, &x)?;
                let target = ops::slice_rows(ctx, &next, prompt_len, target_len)?;
                transfers::copy_cuda_to_cuda(ctx, &target, &output)
            },
        )?;
        Ok(Self {
            conditioning,
            solver,
            prompt_features,
            target_features,
            prompt_mel,
            output,
            _state: x,
            _static_condition: static_condition,
            modulation,
            steps,
            cfm: Arc::clone(cfm),
            noise_generator,
        })
    }

    fn replay(
        &self,
        ctx: &Context,
        prompt_features: &Tensor,
        target_features: &Tensor,
        prompt_mel: &Tensor,
        seed: u64,
    ) -> Result<Tensor> {
        transfers::copy_cuda_to_cuda(ctx, prompt_features, &self.prompt_features)?;
        transfers::copy_cuda_to_cuda(ctx, target_features, &self.target_features)?;
        transfers::copy_cuda_to_cuda(ctx, prompt_mel, &self.prompt_mel)?;
        self.noise_generator
            .borrow_mut()
            .generate(apxinf_core::RngKey::new(seed, 0, 1))?;
        self.replay_bound(ctx)?;
        let copy = ops::alloc(ctx, self.output.shape().dims())?;
        transfers::copy_cuda_to_cuda(ctx, &self.output, &copy)?;
        Ok(copy)
    }
    fn replay_bound(&self, ctx: &Context) -> Result<()> {
        self.conditioning.replay()?;
        for index in 0..self.steps {
            self.cfm
                .bind_step_modulation(ctx, index * 32 / self.steps, &self.modulation)?;
            self.solver.replay()?;
        }
        Ok(())
    }
}

struct VocoderGraph {
    phase: PreparedPhase,
    input: Tensor,
    output: Tensor,
}

impl VocoderGraph {
    fn prepare(ctx: &Context, vocoder: &Vocoder, frames: usize) -> Result<Self> {
        let input = ops::alloc(ctx, &[frames, 100])?;
        let output = ops::alloc(ctx, &[1, frames * 480, 1])?;
        let session = cuda_ops::ExecutionSession::with_capacity_pinning_prepared_buffers(
            vocoder_workspace_bytes(frames)?,
            ctx.device_id(),
        )?;
        let phase = PreparedPhase::prepare_and_capture(ctx, session, || {
            let generated = vocoder.forward(ctx, &input)?;
            transfers::copy_cuda_to_cuda(ctx, &generated, &output)
        })?;
        Ok(Self {
            phase,
            input,
            output,
        })
    }

    fn replay(&self, ctx: &Context, mel: &Tensor) -> Result<Vec<f32>> {
        transfers::copy_cuda_to_cuda(ctx, mel, &self.input)?;
        self.phase.replay()?;
        Vocoder::pcm_from_output(ctx, &self.output)
    }
}

// Frontend and acoustic encoder graphs own stable mutable inputs. Public
// requests rebind values; no reference content is baked into their profiles.
struct BertGraph {
    phase: PreparedPhase,
    _positions: Tensor,
    _types: Tensor,
    ids: Tensor,
    output: Tensor,
}
impl BertGraph {
    fn prepare(ctx: &Context, bert: &Bert, tokens: usize) -> Result<Self> {
        let ids = super::weights::ids_tensor(ctx, &vec![0; tokens])?;
        let positions = super::weights::ids_tensor(ctx, &(0..tokens as u32).collect::<Vec<_>>())?;
        let types = super::weights::ids_tensor(ctx, &vec![0; tokens])?;
        let output = ops::alloc(ctx, &[tokens - 2, super::config::BERT_HIDDEN])?;
        let session = cuda_ops::ExecutionSession::with_capacity_pinning_prepared_buffers(
            (tokens * MIB).max(64 * MIB),
            ctx.device_id(),
        )?;
        let phase = PreparedPhase::prepare_and_capture(ctx, session, || {
            let hidden = bert.forward_device(ctx, &ids, &positions, &types)?;
            transfers::copy_cuda_to_cuda(ctx, &hidden, &output)
        })?;
        Ok(Self {
            phase,
            _positions: positions,
            _types: types,
            ids,
            output,
        })
    }
    fn replay(&self, ctx: &Context, ids: &[u32], word2ph: &[u32], phones: usize) -> Result<Tensor> {
        let values = ids.iter().map(|v| *v as i32).collect::<Vec<_>>();
        transfers::copy_cpu_to_cuda(&Tensor::from_i32(vec![values.len()], &values)?, &self.ids)?;
        if word2ph.len() != self.output.shape().dims()[0] {
            return Err(Error::Other("BERT token/word2ph mismatch".into()));
        }
        let mut map = Vec::with_capacity(phones);
        for (index, count) in word2ph.iter().enumerate() {
            map.extend(std::iter::repeat(index as i32).take(*count as usize));
        }
        if map.len() != phones {
            return Err(Error::Other("word2ph phone count mismatch".into()));
        }
        self.phase.replay()?;
        let map = transfers::to_cuda(&Tensor::from_i32(vec![phones], &map)?, ctx.device_id())?;
        ops::gather_rows(ctx, &self.output, &map)
    }
}

struct HubertGraph {
    phase: PreparedPhase,
    input: Tensor,
    codes: Tensor,
}
impl HubertGraph {
    fn prepare(ctx: &Context, hubert: &Hubert, samples: usize) -> Result<Self> {
        let input = ops::alloc(ctx, &[1, samples, 1])?;
        ops::fill_zero(ctx, &input)?;
        let frames = hubert.semantic_frames(samples)?;
        let codes = ctx.allocate_output(
            apxinf_core::Shape::new(vec![frames]),
            apxinf_core::DType::I32,
        )?;
        // The first 512-channel stage dominates the retained activation arena.
        let workspace = samples
            .checked_mul(1500)
            .and_then(|v| v.checked_add(128 * MIB))
            .ok_or_else(|| Error::Other("HuBERT workspace overflow".into()))?;
        let phase = PreparedPhase::prepare_and_capture(
            ctx,
            cuda_ops::ExecutionSession::with_capacity_pinning_prepared_buffers(workspace, ctx.device_id())?,
            || {
                let features = hubert.encode_features_device(ctx, &input)?;
                let generated = hubert.quantize(ctx, &features)?;
                transfers::copy_cuda_to_cuda(ctx, &generated, &codes)
            },
        )?;
        Ok(Self {
            phase,
            input,
            codes,
        })
    }
    fn replay(&self, ctx: &Context, audio: &[f32]) -> Result<Tensor> {
        let input = super::weights::fp16_tensor(ctx, self.input.shape().dims(), audio)?;
        transfers::copy_cuda_to_cuda(ctx, &input, &self.input)?;
        self.phase.replay()?;
        Ok(self.codes.clone())
    }
}

struct StyleGraph {
    phase: PreparedPhase,
    input: Tensor,
    output: Tensor,
}
impl StyleGraph {
    fn prepare(ctx: &Context, sovits: &Sovits, frames: usize) -> Result<Self> {
        let input = ops::alloc(ctx, &[frames, 704])?;
        let output = ops::alloc(ctx, &[512])?;
        ops::fill_zero(ctx, &input)?;
        let phase = PreparedPhase::prepare_and_capture(
            ctx,
            cuda_ops::ExecutionSession::with_capacity_pinning_prepared_buffers(64 * MIB, ctx.device_id())?,
            || {
                let style = sovits.style(ctx, &input)?;
                transfers::copy_cuda_to_cuda(ctx, &style, &output)
            },
        )?;
        Ok(Self {
            phase,
            input,
            output,
        })
    }
    fn replay(&self, ctx: &Context, spec: &Tensor) -> Result<Tensor> {
        transfers::copy_cuda_to_cuda(ctx, spec, &self.input)?;
        self.phase.replay()?;
        Ok(self.output.clone())
    }
}

struct SovitsGraph {
    phase: PreparedPhase,
    codes: Tensor,
    phones: Tensor,
    _upsample_codes: Tensor,
    _upsample_bridge: Tensor,
    style: Tensor,
    output: Tensor,
}
impl SovitsGraph {
    fn prepare(ctx: &Context, sovits: &Sovits, codes: usize, phones: usize) -> Result<Self> {
        let codes_tensor = super::weights::ids_tensor(ctx, &vec![0; codes])?;
        let phone_tensor = super::weights::ids_tensor(ctx, &vec![0; phones])?;
        let upsample_codes = super::sovits::repeat_row_ids(ctx, codes)?;
        let upsample_bridge = super::sovits::repeat_row_ids(ctx, codes * 2)?;
        let style = ops::alloc(ctx, &[512])?;
        ops::fill_zero(ctx, &style)?;
        let output = ops::alloc(ctx, &[codes * 4, 512])?;
        let workspace = codes
            .checked_mul(4 * MIB)
            .and_then(|v| v.checked_add(128 * MIB))
            .ok_or_else(|| Error::Other("SoVITS workspace overflow".into()))?;
        let phase = PreparedPhase::prepare_and_capture(
            ctx,
            cuda_ops::ExecutionSession::with_capacity_pinning_prepared_buffers(workspace, ctx.device_id())?,
            || {
                let generated = sovits.decode_features_device(
                    ctx,
                    &codes_tensor,
                    &phone_tensor,
                    &style,
                    &upsample_codes,
                    &upsample_bridge,
                    1.0,
                )?;
                transfers::copy_cuda_to_cuda(ctx, &generated, &output)
            },
        )?;
        Ok(Self {
            phase,
            codes: codes_tensor,
            phones: phone_tensor,
            _upsample_codes: upsample_codes,
            _upsample_bridge: upsample_bridge,
            style,
            output,
        })
    }
    fn replay(
        &self,
        ctx: &Context,
        codes: &Tensor,
        phones: &[u32],
        style: &Tensor,
    ) -> Result<Tensor> {
        transfers::copy_cuda_to_cuda(ctx, codes, &self.codes)?;
        let values = phones.iter().map(|v| *v as i32).collect::<Vec<_>>();
        transfers::copy_cpu_to_cuda(
            &Tensor::from_i32(vec![values.len()], &values)?,
            &self.phones,
        )?;
        transfers::copy_cuda_to_cuda(ctx, style, &self.style)?;
        self.phase.replay()?;
        Ok(self.output.clone())
    }
}

pub struct PreparedGptSovitsTts {
    _bert: Arc<Bert>,
    _hubert: Arc<Hubert>,
    _sovits: Arc<Sovits>,
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
    pub fn load_with_autotune(directory: &Path, device_id: usize, autotune: bool) -> Result<Self> {
        let config = GptSovitsConfig::from_dir(directory)?;
        let ctx = Context::new_with_autotune(device_id, autotune).map_err(Error::Cuda)?;
        let mut weights = Weights::load(directory)?;
        let bert = Bert::load(&mut weights, &ctx)?;
        let hubert = Hubert::load(&mut weights, &ctx)?;
        let gpt = Gpt::load(&mut weights, &ctx)?;
        let sovits = Sovits::load(&mut weights, &ctx)?;
        let cfm = Cfm::load(&mut weights, &ctx)?;
        let vocoder = Vocoder::load(&mut weights, &ctx)?;
        Ok(Self {
            ctx: Arc::new(ctx),
            bert: Arc::new(bert),
            hubert: Arc::new(hubert),
            gpt: Arc::new(gpt),
            sovits: Arc::new(sovits),
            cfm: Arc::new(cfm),
            vocoder: Arc::new(vocoder),
            config,
            cache: Rc::new(RefCell::new(ConditioningCache::default())),
        })
    }

    fn bert_phone_features(
        &self,
        token_ids: &[u32],
        word2ph: &[u32],
        phone_count: usize,
    ) -> Result<Tensor> {
        self.bert_phone_features_inner(token_ids, word2ph, phone_count, true)
    }

    fn bert_phone_features_inner(
        &self,
        token_ids: &[u32],
        word2ph: &[u32],
        phone_count: usize,
        cache_text: bool,
    ) -> Result<Tensor> {
        let key = TextKey::new(token_ids, word2ph, phone_count, false);
        if cache_text {
            if let Some(value) = self.cache.borrow_mut().text(&key) {
                return Ok(value);
            }
        }
        let hidden = self.bert.forward(&self.ctx, token_ids)?;
        if hidden.shape().dims()[0] != word2ph.len() {
            return Err(Error::Other(format!(
                "BERT token/word2ph mismatch: {} tokens vs {} entries",
                hidden.shape().dims()[0],
                word2ph.len()
            )));
        }
        let mut map: Vec<i32> = Vec::with_capacity(phone_count);
        for (index, count) in word2ph.iter().enumerate() {
            for _ in 0..*count {
                map.push(index as i32);
            }
        }
        if map.len() != phone_count {
            return Err(Error::Other(format!(
                "word2ph expands to {} phones, expected {phone_count}",
                map.len()
            )));
        }
        let ids = Tensor::from_i32(vec![map.len()], &map)?;
        let ids = transfers::to_cuda(&ids, self.ctx.device_id())?;
        let value = ops::gather_rows(&self.ctx, &hidden, &ids)?;
        if cache_text {
            self.cache.borrow_mut().insert_text(key, value.clone());
        }
        Ok(value)
    }

    fn codes_to_vec(&self, codes: &Tensor) -> Result<Vec<u32>> {
        let host = transfers::to_cpu(codes)?;
        let values = host.as_i32()?;
        Ok(values.iter().map(|value| *value as u32).collect())
    }

    fn codes_from_vec(&self, codes: &[u32]) -> Result<Tensor> {
        let values: Vec<i32> = codes.iter().map(|value| *value as i32).collect();
        let host = Tensor::from_i32(vec![values.len()], &values)?;
        transfers::to_cuda(&host, self.ctx.device_id())
    }

    fn reference_features(&self, inputs: &GptSovitsInputs<'_>) -> Result<Rc<ReferenceFeatures>> {
        let key = ReferenceKey::new(inputs, false);
        if let Some(value) = self.cache.borrow_mut().reference(&key) {
            return Ok(value);
        }
        let frames = inputs.reference_mel.len() / 100;
        let spec_frames = inputs.reference_spec.len() / 704;
        let mel = super::weights::fp16_tensor(&self.ctx, &[frames, 100], inputs.reference_mel)?;
        let spec =
            super::weights::fp16_tensor(&self.ctx, &[spec_frames, 704], inputs.reference_spec)?;
        let bert = self.bert_phone_features_inner(
            inputs.reference_bert_token_ids,
            inputs.reference_word2ph,
            inputs.reference_phone_ids.len(),
            false,
        )?;
        let codes = self.hubert.encode(&self.ctx, inputs.reference_audio_16k)?;
        let prompt = self.codes_to_vec(&codes)?;
        let style = self.sovits.style(&self.ctx, &spec)?;
        let features = self.sovits.decode_features(
            &self.ctx,
            &codes,
            inputs.reference_phone_ids,
            &style,
            1.0,
        )?;
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

    pub fn synthesize(&self, request: &TtsRequest<'_>) -> Result<GeneratedAudio> {
        let inputs = gpt_sovits_inputs(request)?;
        let (top_k, top_p, temperature, repetition_penalty, max_seconds) = sampling_of(request);
        if !inputs.speed_factor.is_finite() || inputs.speed_factor <= 0.0 {
            return Err(Error::Other(format!(
                "speed_factor must be finite and positive, got {}",
                inputs.speed_factor
            )));
        }
        if inputs.phone_ids.is_empty() || inputs.reference_phone_ids.is_empty() {
            return Err(Error::Other(
                "tts request requires target and reference text".into(),
            ));
        }
        if inputs.reference_audio_16k.len() < 400 {
            return Err(Error::Other("reference audio is too short".into()));
        }
        let reference_frames = inputs.reference_mel.len() / 100;
        let reference_spec_frames = inputs.reference_spec.len() / 704;
        if reference_frames == 0 || reference_spec_frames == 0 {
            return Err(Error::Other("reference mel/spec is empty".into()));
        }
        let reference = self.reference_features(inputs)?;
        let reference_mel = reference.mel.clone();
        let reference_bert = reference.bert.clone();
        let target_bert = self
            .bert_phone_features(
                inputs.bert_token_ids,
                inputs.word2ph,
                inputs.phone_ids.len(),
            )
            .map_err(|error| Error::Other(format!("target bert: {error}")))?;
        let bert = ops::concat_rows(&self.ctx, &reference_bert, &target_bert)
            .map_err(|error| Error::Other(format!("bert concat: {error}")))?;
        let mut phones = inputs.reference_phone_ids.to_vec();
        phones.extend_from_slice(inputs.phone_ids);

        let prompt = &reference.prompt;
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
            .map_err(|error| Error::Other(format!("gpt: {error}")))?;
        if generated.is_empty() {
            return Err(Error::Other("semantic decoder produced no tokens".into()));
        }
        let target_codes = self.codes_from_vec(&generated)?;

        let ge = reference.style.clone();
        let reference_features = reference.features.clone();
        let target_features = self
            .sovits
            .decode_features(
                &self.ctx,
                &target_codes,
                inputs.phone_ids,
                &ge,
                inputs.speed_factor,
            )
            .map_err(|error| Error::Other(format!("target decode: {error}")))?;
        // The tts request carries the final prompt mel (typically 500
        // frames), while reference features are decoded at the pre-truncation
        // rate derived from the full reference spectrogram. PyTorch first
        // truncates to min(2 * spec_frames, feature_frames), then keeps the
        // final prompt-mel window. Reproduce that two-stage alignment here.
        let decoded_frames = reference_features.shape().dims()[0];
        let reference_features =
            match reference_feature_window(decoded_frames, reference_spec_frames, reference_frames)
            {
                Some(offset) => {
                    ops::slice_rows(&self.ctx, &reference_features, offset, reference_frames)?
                }
                None => reference_features,
            };
        let mel = self
            .cfm
            .synthesize(
                &self.ctx,
                &reference_features,
                &target_features,
                &reference_mel,
                inputs.sample_steps,
                inputs.cfg_rate,
                request.seed,
            )
            .map_err(|error| Error::Other(format!("cfm: {error}")))?;
        let scale = super::weights::fp16_tensor(
            &self.ctx,
            &[100],
            &vec![(SPEC_MAX - SPEC_MIN) / 2.0; 100],
        )?;
        let shift = super::weights::fp16_tensor(&self.ctx, &[100], &vec![SPEC_MIN; 100])?;
        let one = super::weights::fp16_tensor(&self.ctx, &[100], &[1.; 100])?;
        let zero = super::weights::fp16_tensor(&self.ctx, &[100], &[0.; 100])?;
        let mel = denormalize_mel(&self.ctx, &mel, &one, &scale, &zero, &shift)?;
        let samples = self
            .vocoder
            .synth(&self.ctx, &mel)
            .map_err(|error| Error::Other(format!("vocoder: {error}")))?;
        // No implicit trailing pause. The public pipeline appends
        // `fragment_interval` silence after every text fragment, matching the
        // official `audio_postprocess`.
        Ok(mono_audio(samples, generated.len(), max_new_tokens))
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
        let mel = super::weights::fp16_tensor(
            &self.ctx,
            &[self.spec.reference_frames, 100],
            inputs.reference_mel,
        )?;
        let spec = super::weights::fp16_tensor(
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
            .min(CFM_REFERENCE_FRAMES);
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
        let chunk = (CFM_TOTAL_CHUNK - reference_frames).min(CFM_MAX_TARGET_CHUNK);
        let mut rolling_mel = original_mel.clone();
        let mut rolling_features = original_features.clone();
        let mut results = Vec::new();
        let mut start = 0usize;
        let mut chunk_index = 0usize;
        while start < self.spec.target_frames {
            let length = chunk.min(self.spec.target_frames - start);
            let target = ops::slice_rows(&self.ctx, &target_features, start, length)?;
            let (prompt_mel, prompt_features) = if chunk_index == 0 {
                (original_mel.clone(), original_features.clone())
            } else {
                let tail = CFM_ROLLING_TAIL
                    .min(reference_frames)
                    .min(rolling_mel.shape().dims()[0])
                    .min(rolling_features.shape().dims()[0]);
                let prefix = reference_frames - tail;
                (
                    ops::concat_rows(
                        &self.ctx,
                        &ops::slice_rows(&self.ctx, &original_mel, 0, prefix)?,
                        &ops::slice_rows(
                            &self.ctx,
                            &rolling_mel,
                            rolling_mel.shape().dims()[0] - tail,
                            tail,
                        )?,
                    )?,
                    ops::concat_rows(
                        &self.ctx,
                        &ops::slice_rows(&self.ctx, &original_features, 0, prefix)?,
                        &ops::slice_rows(
                            &self.ctx,
                            &rolling_features,
                            rolling_features.shape().dims()[0] - tail,
                            tail,
                        )?,
                    )?,
                )
            };
            let graph = self.cfm_graphs.get(&length).ok_or_else(|| {
                Error::Other(format!("missing prepared CFM chunk profile {length}"))
            })?;
            let _tts_cfm_range = crate::profiling::trace::range("tts.cfm.chunk");
            let generated = graph.replay(
                &self.ctx,
                &prompt_features,
                &target,
                &prompt_mel,
                request.seed.wrapping_add(chunk_index as u64),
            )?;
            drop(_tts_cfm_range);
            rolling_mel = ops::slice_rows(
                &self.ctx,
                &generated,
                generated.shape().dims()[0].saturating_sub(reference_frames),
                reference_frames.min(generated.shape().dims()[0]),
            )?;
            rolling_features = ops::slice_rows(
                &self.ctx,
                &target,
                length.saturating_sub(reference_frames),
                reference_frames.min(length),
            )?;
            results.push(generated);
            start += length;
            chunk_index += 1;
        }
        let mut mel = results.remove(0);
        for result in results {
            mel = ops::concat_rows(&self.ctx, &mel, &result)?;
        }
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

impl TtsTrait for GptSovitsModel {
    fn contract(&self) -> Result<TtsContract> {
        Ok(TtsContract {
            sample_rate: OUTPUT_SAMPLE_RATE,
            mel_frames_per_second: self.config.mel_frames_per_second,
            semantic_frames_per_second: self.config.semantic_frames_per_second,
        })
    }

    fn synthesize(&mut self, request: &TtsRequest<'_>) -> Result<GeneratedAudio> {
        GptSovitsModel::synthesize(self, request)
    }

    fn configure_cache(&self, config: TtsCacheConfig) -> Result<()> {
        self.cache.borrow_mut().configure(config);
        Ok(())
    }
    fn clear_cache(&self, reset_stats: bool) -> Result<()> {
        self.cache.borrow_mut().clear(reset_stats);
        Ok(())
    }
    fn cache_info(&self) -> TtsCacheInfo {
        self.cache.borrow().info()
    }

    fn prepare(&self, spec: &TtsSpec) -> Result<Box<dyn PreparedTts>> {
        if spec.phone_count == 0
            || spec.bert_token_count < 3
            || spec.reference_phone_count == 0
            || spec.reference_bert_token_count < 3
            || spec.reference_audio_samples < 400
            || spec.reference_frames == 0
            || spec.reference_spec_frames == 0
            || spec.target_frames == 0
            || super::cfm::validate_sample_steps(spec.sample_steps).is_err()
            || spec.target_frames % 4 != 0
        {
            return Err(Error::Other(format!(
                "invalid GPT-SoVITS prepared profile: {spec:?}"
            )));
        }
        let prompt_len = spec.reference_frames.min(CFM_REFERENCE_FRAMES);
        let chunk = (CFM_TOTAL_CHUNK - prompt_len).min(CFM_MAX_TARGET_CHUNK);
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
        let mel_scale = super::weights::fp16_tensor(
            &self.ctx,
            &[100],
            &vec![(SPEC_MAX - SPEC_MIN) / 2.0; 100],
        )?;
        let mel_shift = super::weights::fp16_tensor(&self.ctx, &[100], &vec![SPEC_MIN; 100])?;
        let mel_one = super::weights::fp16_tensor(&self.ctx, &[100], &[1.; 100])?;
        let mel_zero = super::weights::fp16_tensor(&self.ctx, &[100], &[0.; 100])?;
        Ok(Box::new(PreparedGptSovitsTts {
            _bert: Arc::clone(&self.bert),
            _hubert: Arc::clone(&self.hubert),
            _sovits: Arc::clone(&self.sovits),
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
        }))
    }
}

#[cfg(test)]
mod reference_window_tests {
    use super::reference_feature_window;

    #[test]
    fn short_prompt_uses_feature_head_not_tail() {
        // A 4.95 s reference: 552 decoded feature rows, 245 spectrogram
        // frames, 492 prompt mel frames. The window must start at zero.
        assert_eq!(reference_feature_window(552, 245, 492), Some(0));
    }

    #[test]
    fn long_prompt_uses_last_500_frames_of_estimated_mel() {
        // A 6.09 s reference with a 500-frame capped prompt mel.
        assert_eq!(reference_feature_window(668, 302, 500), Some(104));
    }

    #[test]
    fn window_is_clamped_to_decoded_features() {
        assert_eq!(reference_feature_window(300, 302, 300), Some(0));
        assert_eq!(reference_feature_window(100, 10, 200), None);
    }
}

#[cfg(test)]
mod output_tests {
    use super::{max_semantic_tokens, mono_audio, MAX_SEMANTIC_TOKENS, OUTPUT_SAMPLE_RATE};

    #[test]
    fn semantic_budget_matches_seconds_and_cap() {
        assert_eq!(max_semantic_tokens(0.2), 25);
        assert_eq!(max_semantic_tokens(30.0), 750);
        assert_eq!(max_semantic_tokens(600.0), MAX_SEMANTIC_TOKENS);
    }

    #[test]
    fn audio_is_mono_and_reports_length_limit() {
        let audio = mono_audio(vec![0.0; 8], 12, 750);
        assert_eq!(audio.sample_rate, OUTPUT_SAMPLE_RATE);
        assert_eq!(audio.channels, 1);
        assert_eq!(audio.samples.len(), 8);
        assert!(audio.complete, "stopping before the budget means EOS");
        assert!(!mono_audio(vec![0.0; 8], 750, 750).complete);
    }
}
