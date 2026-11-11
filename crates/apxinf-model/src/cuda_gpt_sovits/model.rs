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
    PcmAudio, PreparedSpeech, SpeechContract, SpeechRequest, SpeechSpec, TtsCacheConfig,
    TtsCacheInfo, TtsTrait,
};

const SPEC_MIN: f32 = -12.0;
const SPEC_MAX: f32 = 2.0;
const OUTPUT_SAMPLE_RATE: u32 = 48000;
const CFM_REFERENCE_FRAMES: usize = 500;

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
    noise: Tensor,
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
            noise,
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

pub struct PreparedGptSovitsSpeech {
    _bert: Arc<Bert>,
    _hubert: Arc<Hubert>,
    _sovits: Arc<Sovits>,
    spec: SpeechSpec,
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
    pub fn load(directory: &Path, device_id: usize) -> Result<Self> {
        Self::load_with_autotune(directory, device_id, false)
    }

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

    pub fn context(&self) -> &Context {
        self.ctx.as_ref()
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

    fn reference_features(&self, request: &SpeechRequest<'_>) -> Result<Rc<ReferenceFeatures>> {
        let key = ReferenceKey::new(request, false);
        if let Some(value) = self.cache.borrow_mut().reference(&key) {
            return Ok(value);
        }
        let frames = request.reference_mel.len() / 100;
        let spec_frames = request.reference_spec.len() / 704;
        let mel = super::weights::fp16_tensor(&self.ctx, &[frames, 100], request.reference_mel)?;
        let spec =
            super::weights::fp16_tensor(&self.ctx, &[spec_frames, 704], request.reference_spec)?;
        let bert = self.bert_phone_features_inner(
            request.reference_bert_token_ids,
            request.reference_word2ph,
            request.reference_phone_ids.len(),
            false,
        )?;
        let codes = self.hubert.encode(&self.ctx, request.reference_audio_16k)?;
        let prompt = self.codes_to_vec(&codes)?;
        let style = self.sovits.style(&self.ctx, &spec)?;
        let features = self.sovits.decode_features(
            &self.ctx,
            &codes,
            request.reference_phone_ids,
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

    pub fn synthesize(&self, request: &SpeechRequest<'_>) -> Result<PcmAudio> {
        if !request.speed_factor.is_finite() || request.speed_factor <= 0.0 {
            return Err(Error::Other(format!(
                "speed_factor must be finite and positive, got {}",
                request.speed_factor
            )));
        }
        if request.phone_ids.is_empty() || request.reference_phone_ids.is_empty() {
            return Err(Error::Other(
                "speech request requires target and reference text".into(),
            ));
        }
        if request.reference_audio_16k.len() < 400 {
            return Err(Error::Other("reference audio is too short".into()));
        }
        let reference_frames = request.reference_mel.len() / 100;
        let reference_spec_frames = request.reference_spec.len() / 704;
        if reference_frames == 0 || reference_spec_frames == 0 {
            return Err(Error::Other("reference mel/spec is empty".into()));
        }
        let reference = self.reference_features(request)?;
        let reference_mel = reference.mel.clone();
        let reference_bert = reference.bert.clone();
        let target_bert = self
            .bert_phone_features(
                request.bert_token_ids,
                request.word2ph,
                request.phone_ids.len(),
            )
            .map_err(|error| Error::Other(format!("target bert: {error}")))?;
        let bert = ops::concat_rows(&self.ctx, &reference_bert, &target_bert)
            .map_err(|error| Error::Other(format!("bert concat: {error}")))?;
        let mut phones = request.reference_phone_ids.to_vec();
        phones.extend_from_slice(request.phone_ids);

        let prompt = &reference.prompt;
        let max_new_tokens = ((request.max_seconds.max(1.0) * 25.0) as usize).min(1500);
        let generated = self
            .gpt
            .generate(
                &self.ctx,
                &phones,
                &bert,
                &prompt,
                request.seed,
                request.top_k,
                request.top_p,
                request.temperature,
                request.repetition_penalty,
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
                request.phone_ids,
                &ge,
                request.speed_factor,
            )
            .map_err(|error| Error::Other(format!("target decode: {error}")))?;
        // The speech request carries the final prompt mel (typically 500
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
                request.sample_steps.max(1),
                request.cfg_rate,
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
        let mut samples = self
            .vocoder
            .synth(&self.ctx, &mel)
            .map_err(|error| Error::Other(format!("vocoder: {error}")))?;
        // No implicit trailing pause. The public pipeline appends
        // `fragment_interval` silence after every text fragment, matching the
        // official `audio_postprocess`.
        let valid_length = samples.len();
        Ok(PcmAudio {
            samples,
            sample_rate: OUTPUT_SAMPLE_RATE,
            valid_length,
        })
    }
}

impl PreparedGptSovitsSpeech {
    fn validate_request(&self, request: &SpeechRequest<'_>) -> Result<()> {
        let actual = SpeechSpec {
            phone_count: request.phone_ids.len(),
            bert_token_count: request.bert_token_ids.len(),
            reference_phone_count: request.reference_phone_ids.len(),
            reference_bert_token_count: request.reference_bert_token_ids.len(),
            reference_audio_samples: request.reference_audio_16k.len(),
            reference_frames: request.reference_mel.len() / 100,
            reference_spec_frames: request.reference_spec.len() / 704,
            target_frames: self.spec.target_frames,
            sample_steps: request.sample_steps,
        };
        if request.reference_mel.len() != self.spec.reference_frames * 100
            || request.reference_spec.len() != self.spec.reference_spec_frames * 704
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
                "speech request does not match prepared profile: expected {:?}, got {:?}",
                self.spec, actual
            )));
        }
        if request.cfg_rate > 1e-5 {
            return Err(Error::Other(
                "gpt_sovits v5turbo does not support classifier-free guidance".into(),
            ));
        }
        if (request.speed_factor - 1.0).abs() > 1e-6 {
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

    fn reference_features(&self, request: &SpeechRequest<'_>) -> Result<Rc<ReferenceFeatures>> {
        let key = ReferenceKey::new(request, true);
        if let Some(value) = self.cache.borrow_mut().reference(&key) {
            return Ok(value);
        }
        let mel = super::weights::fp16_tensor(
            &self.ctx,
            &[self.spec.reference_frames, 100],
            request.reference_mel,
        )?;
        let spec = super::weights::fp16_tensor(
            &self.ctx,
            &[self.spec.reference_spec_frames, 704],
            request.reference_spec,
        )?;
        let bert = self.cached_bert(
            &self.reference_bert,
            request.reference_bert_token_ids,
            request.reference_word2ph,
            request.reference_phone_ids.len(),
            false,
        )?;
        let codes = self
            .hubert_graph
            .replay(&self.ctx, request.reference_audio_16k)?;
        let host = transfers::to_cpu(&codes)?;
        let prompt = host.as_i32()?.iter().map(|x| *x as u32).collect();
        let generated_style = self.style_graph.replay(&self.ctx, &spec)?;
        let style = owned_tensor(&self.ctx, &generated_style)?;
        let generated_features =
            self.reference_sovits
                .replay(&self.ctx, &codes, request.reference_phone_ids, &style)?;
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

    fn run_prepared(&self, request: &SpeechRequest<'_>) -> Result<PcmAudio> {
        self.validate_request(request)?;
        let reference = self.reference_features(request)?;
        let reference_mel = reference.mel.clone();
        let reference_bert = reference.bert.clone();
        let target_bert = self.cached_bert(
            &self.target_bert,
            request.bert_token_ids,
            request.word2ph,
            request.phone_ids.len(),
            true,
        )?;
        let bert = ops::concat_rows(&self.ctx, &reference_bert, &target_bert)?;
        let mut phones = request.reference_phone_ids.to_vec();
        phones.extend_from_slice(request.phone_ids);
        let prompt = &reference.prompt;
        let generated = self
            .gpt
            .generate(
                &self.ctx,
                &phones,
                &bert,
                &prompt,
                request.seed,
                request.top_k,
                request.top_p,
                request.temperature,
                request.repetition_penalty,
                ((request.max_seconds.max(1.0) * 25.0) as usize).min(1500),
            )
            .map_err(|error| Error::Other(format!("prepared gpt: {error}")))?;
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
        let target_features =
            self.target_sovits
                .replay(&self.ctx, &target_codes, request.phone_ids, &ge)?;
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
            let generated = graph.replay(
                &self.ctx,
                &prompt_features,
                &target,
                &prompt_mel,
                request.seed + chunk_index as u64,
            )?;
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
        let samples = self.vocoder_graph.replay(&self.ctx, &mel)?;
        // The public pipeline adds `fragment_interval` silence; the acoustic
        // graph itself must not append an implicit pause.
        let valid_length = samples.len();
        Ok(PcmAudio {
            samples,
            sample_rate: OUTPUT_SAMPLE_RATE,
            valid_length,
        })
    }
}

impl PreparedSpeech for PreparedGptSovitsSpeech {
    fn spec(&self) -> &SpeechSpec {
        &self.spec
    }

    fn run(&self, request: &SpeechRequest<'_>) -> Result<PcmAudio> {
        self.run_prepared(request)
    }
}

impl TtsTrait for GptSovitsModel {
    fn contract(&self) -> SpeechContract {
        SpeechContract {
            sample_rate: OUTPUT_SAMPLE_RATE,
            mel_frames_per_second: self.config.mel_frames_per_second,
            semantic_frames_per_second: self.config.semantic_frames_per_second,
        }
    }

    fn synthesize(&self, request: &SpeechRequest<'_>) -> Result<PcmAudio> {
        GptSovitsModel::synthesize(self, request)
    }

    fn configure_cache(&self, config: TtsCacheConfig) -> Result<()> {
        self.cache.borrow_mut().configure(config);
        Ok(())
    }
    fn clear_cache(&self, reset_stats: bool) {
        self.cache.borrow_mut().clear(reset_stats);
    }
    fn cache_info(&self) -> TtsCacheInfo {
        self.cache.borrow().info()
    }

    fn prepare(&self, spec: &SpeechSpec) -> Result<Box<dyn PreparedSpeech>> {
        if spec.phone_count == 0
            || spec.bert_token_count < 2
            || spec.reference_phone_count == 0
            || spec.reference_bert_token_count < 2
            || spec.reference_audio_samples < 400
            || spec.reference_frames == 0
            || spec.reference_spec_frames == 0
            || spec.target_frames == 0
            || spec.sample_steps == 0
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
        Ok(Box::new(PreparedGptSovitsSpeech {
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
mod capture_tests {
    use super::*;
    use std::{fs, path::PathBuf};

    #[test]
    #[ignore = "requires APXINF_GPT_CAPTURE_ROOT and APXINF_GPT_CHECKPOINT"]
    fn frontend_and_acoustic_graphs_match_eager_with_changed_inputs() {
        let root = PathBuf::from(std::env::var("APXINF_GPT_CAPTURE_ROOT").unwrap());
        let checkpoint = PathBuf::from(std::env::var("APXINF_GPT_CHECKPOINT").unwrap());
        let model = GptSovitsModel::load(&checkpoint, 0).unwrap();
        let ctx = model.context();
        let read_u32 = |p: &Path| {
            fs::read(p)
                .unwrap()
                .chunks_exact(4)
                .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let read_f32 = |p: &Path| {
            fs::read(p)
                .unwrap()
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let read_f16 = |p: &Path| {
            fs::read(p)
                .unwrap()
                .chunks_exact(2)
                .map(|v| half::f16::from_bits(u16::from_le_bytes(v.try_into().unwrap())).to_f32())
                .collect::<Vec<_>>()
        };
        let exact = |label: &str, actual: &Tensor, expected: &Tensor| {
            ctx.synchronize().unwrap();
            let actual = transfers::to_cpu(actual).unwrap();
            let expected = transfers::to_cpu(expected).unwrap();
            assert_eq!(actual.shape(), expected.shape(), "{label} shape");
            assert_eq!(
                actual.storage().as_cpu().unwrap(),
                expected.storage().as_cpu().unwrap(),
                "{label} values"
            );
            println!("{label}: eager/graph bytes exact");
        };
        for pair in ["pair1", "pair2"] {
            let req = root.join(pair).join("request");
            let mut ids = read_u32(&req.join("bert_ids.bin"));
            let word2ph = read_u32(&req.join("word2ph.bin"));
            let phones = read_u32(&req.join("phones.bin"));
            let graph = BertGraph::prepare(ctx, &model.bert, ids.len()).unwrap();
            for changed in [false, true] {
                if changed {
                    ids[1] = (ids[1] + 1) % 20000;
                }
                let eager = model
                    .bert_phone_features(&ids, &word2ph, phones.len())
                    .unwrap();
                let actual = graph.replay(ctx, &ids, &word2ph, phones.len()).unwrap();
                exact(&format!("{pair}/bert changed={changed}"), &actual, &eager);
            }
            drop(graph);
            let mut audio = read_f32(&req.join("ref_audio.f32"));
            let graph = HubertGraph::prepare(ctx, &model.hubert, audio.len()).unwrap();
            for changed in [false, true] {
                if changed {
                    for x in &mut audio {
                        *x *= 0.75;
                    }
                }
                let eager = model.hubert.encode(ctx, &audio).unwrap();
                let actual = graph.replay(ctx, &audio).unwrap();
                exact(&format!("{pair}/hubert changed={changed}"), &actual, &eager);
            }
            drop(graph);
            let mut spec = read_f16(&req.join("ref_spec.f16"));
            let graph = StyleGraph::prepare(ctx, &model.sovits, spec.len() / 704).unwrap();
            for changed in [false, true] {
                if changed {
                    for x in &mut spec {
                        *x *= 0.9;
                    }
                }
                let input =
                    super::super::weights::fp16_tensor(ctx, &[spec.len() / 704, 704], &spec)
                        .unwrap();
                let eager = model.sovits.style(ctx, &input).unwrap();
                let actual = graph.replay(ctx, &input).unwrap();
                exact(&format!("{pair}/style changed={changed}"), &actual, &eager);
            }
            let spec =
                super::super::weights::fp16_tensor(ctx, &[spec.len() / 704, 704], &spec).unwrap();
            let style = model.sovits.style(ctx, &spec).unwrap();
            drop(graph);
            let mut codes = read_u32(
                &root
                    .parent()
                    .unwrap()
                    .join("results/gpt-teacher-forcing-20261029")
                    .join(format!("{pair}-target-tokens.i32")),
            );
            let graph =
                SovitsGraph::prepare(ctx, &model.sovits, codes.len(), phones.len()).unwrap();
            for changed in [false, true] {
                if changed {
                    codes[0] = (codes[0] + 1) % 1024;
                }
                let input = model.codes_from_vec(&codes).unwrap();
                let eager = model
                    .sovits
                    .decode_features(ctx, &input, &phones, &style, 1.0)
                    .unwrap();
                let actual = graph.replay(ctx, &input, &phones, &style).unwrap();
                exact(&format!("{pair}/sovits changed={changed}"), &actual, &eager);
            }
        }
    }

    #[test]
    #[ignore = "requires APXINF_GPT_CAPTURE_ROOT and APXINF_GPT_CHECKPOINT"]
    fn public_conditioning_matches_reference_token_stream() {
        let root = PathBuf::from(std::env::var("APXINF_GPT_CAPTURE_ROOT").unwrap());
        let checkpoint = PathBuf::from(std::env::var("APXINF_GPT_CHECKPOINT").unwrap());
        let model = GptSovitsModel::load(&checkpoint, 0).unwrap();
        let read_ids = |p: &Path| {
            fs::read(p)
                .unwrap()
                .chunks_exact(4)
                .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let mut failures = Vec::new();
        for pair in ["pair1", "pair2"] {
            let request = root.join(pair).join("request");
            let mut phones = read_ids(&request.join("ref_phones.bin"));
            let target_phones = read_ids(&request.join("phones.bin"));
            let reference_bert = model
                .bert_phone_features(
                    &read_ids(&request.join("ref_bert_ids.bin")),
                    &read_ids(&request.join("ref_word2ph.bin")),
                    phones.len(),
                )
                .unwrap();
            let target_bert = model
                .bert_phone_features(
                    &read_ids(&request.join("bert_ids.bin")),
                    &read_ids(&request.join("word2ph.bin")),
                    target_phones.len(),
                )
                .unwrap();
            let bert = ops::concat_rows(&model.ctx, &reference_bert, &target_bert).unwrap();
            phones.extend_from_slice(&target_phones);
            let audio = fs::read(request.join("ref_audio.f32"))
                .unwrap()
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                .collect::<Vec<_>>();
            let codes = model.hubert.encode(&model.ctx, &audio).unwrap();
            let prompt = model.codes_to_vec(&codes).unwrap();
            let reference = root
                .parent()
                .unwrap()
                .join("results/gpt-teacher-forcing-20261029");
            let expected_prompt = read_ids(&reference.join(format!("{pair}-prompt.i32")));
            let prompt_mismatches = prompt
                .iter()
                .zip(&expected_prompt)
                .enumerate()
                .filter(|(_, (a, b))| a != b)
                .map(|(i, (a, b))| (i, *a, *b))
                .collect::<Vec<_>>();
            // Keep a generous stop bound and allow the model to reach EOS.
            // Equal-length truncation cannot prove generation parity.
            let actual = model
                .gpt
                .generate(
                    &model.ctx, &phones, &bert, &prompt, 1234, 5, 1.0, 1.0, 1.35, 1425,
                )
                .unwrap();
            let expected = read_ids(&reference.join(format!("{pair}-target-tokens.i32")));
            let first_difference = actual.iter().zip(&expected).position(|(a, b)| a != b);
            println!("{pair}: prompt_mismatches={prompt_mismatches:?} natural_tokens={}/{} first_difference={first_difference:?}",
                actual.len(), expected.len());
            if let Some(directory) = std::env::var_os("APXINF_GPT_PUBLIC_TOKENS_CAPTURE") {
                let directory = PathBuf::from(directory);
                fs::create_dir_all(&directory).unwrap();
                let bytes = actual
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>();
                fs::write(directory.join(format!("{pair}-tokens.u32")), bytes).unwrap();
            }
            if prompt != expected_prompt || actual != expected {
                failures.push((
                    pair,
                    prompt_mismatches,
                    actual.len(),
                    expected.len(),
                    first_difference,
                ));
            }
        }
        assert!(
            failures.is_empty(),
            "public generation acceptance failed: {failures:?}"
        );
    }

    #[test]
    #[ignore = "requires captured official fixtures and CUDA checkpoint"]
    fn bert_conditioning_crossover() {
        let root = PathBuf::from(std::env::var("APXINF_GPT_CAPTURE_ROOT").unwrap());
        let checkpoint = PathBuf::from(std::env::var("APXINF_GPT_CHECKPOINT").unwrap());
        let model = GptSovitsModel::load_with_autotune(
            &checkpoint,
            0,
            std::env::var_os("APXINF_ALIGNMENT_AUTOTUNE").is_some(),
        )
        .unwrap();
        let fixture = root
            .parent()
            .unwrap()
            .join("results/gpt-teacher-forcing-20261029");
        let read_ids = |p: &Path| {
            fs::read(p)
                .unwrap()
                .chunks_exact(4)
                .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        for pair in ["pair1", "pair2"] {
            let request = root.join(pair).join("request");
            let mut phones = read_ids(&request.join("ref_phones.bin"));
            let reference_count = phones.len();
            let target_phones = read_ids(&request.join("phones.bin"));
            let native_ref = model
                .bert_phone_features(
                    &read_ids(&request.join("ref_bert_ids.bin")),
                    &read_ids(&request.join("ref_word2ph.bin")),
                    reference_count,
                )
                .unwrap();
            let native_target = model
                .bert_phone_features(
                    &read_ids(&request.join("bert_ids.bin")),
                    &read_ids(&request.join("word2ph.bin")),
                    target_phones.len(),
                )
                .unwrap();
            phones.extend_from_slice(&target_phones);
            let official_values = fs::read(fixture.join(format!("{pair}-bert-rows.f32")))
                .unwrap()
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                .collect::<Vec<_>>();
            let official = super::super::weights::fp16_tensor(
                &model.ctx,
                &[phones.len(), 1024],
                &official_values,
            )
            .unwrap();
            let official_ref = ops::slice_rows(&model.ctx, &official, 0, reference_count).unwrap();
            let official_target =
                ops::slice_rows(&model.ctx, &official, reference_count, target_phones.len())
                    .unwrap();
            for (label, a, b) in [
                ("reference", &native_ref, &official_ref),
                ("target", &native_target, &official_target),
            ] {
                metrics(
                    &format!("{pair}/bert/{label}"),
                    &transfers::to_cpu(a).unwrap().to_f32_vec().unwrap(),
                    &transfers::to_cpu(b).unwrap().to_f32_vec().unwrap(),
                );
            }
            let prompt = read_ids(&fixture.join(format!("{pair}-prompt.i32")));
            let expected = read_ids(&fixture.join(format!("{pair}-target-tokens.i32")));
            for (label, reference, target) in [
                ("official", &official_ref, &official_target),
                ("native_reference", &native_ref, &official_target),
                ("native_target", &official_ref, &native_target),
                ("native_both", &native_ref, &native_target),
            ] {
                let bert = ops::concat_rows(&model.ctx, reference, target).unwrap();
                let actual = model
                    .gpt
                    .generate(
                        &model.ctx, &phones, &bert, &prompt, 1234, 5, 1., 1., 1.35, 1425,
                    )
                    .unwrap();
                let first = actual.iter().zip(&expected).position(|(a, b)| a != b);
                println!(
                    "{pair}/crossover/{label}: actual={} expected={} first_difference={first:?}",
                    actual.len(),
                    expected.len()
                );
                if label == "official" {
                    assert_eq!(
                        actual, expected,
                        "official conditioning baseline must match"
                    );
                }
            }
        }
    }

    struct Capture {
        root: PathBuf,
        entries: Vec<serde_json::Value>,
    }
    impl Capture {
        fn load(root: PathBuf) -> Self {
            let manifest: serde_json::Value =
                serde_json::from_slice(&fs::read(root.join("manifest.json")).unwrap()).unwrap();
            Self {
                root,
                entries: manifest["entries"].as_array().unwrap().clone(),
            }
        }
        fn values(&self, name: &str) -> (Vec<usize>, Vec<f32>) {
            let entry = self.entries.iter().find(|e| e["name"] == name).unwrap();
            let dims = entry["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect::<Vec<_>>();
            let bytes = fs::read(self.root.join(entry["path"].as_str().unwrap())).unwrap();
            assert_eq!(entry["dtype"], "f32");
            let values = bytes
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                .collect();
            (dims, values)
        }
        fn nlc(&self, ctx: &Context, name: &str) -> Tensor {
            let (dims, values) = self.values(name);
            let (channels, frames) = (dims[1], dims[2]);
            let mut reordered = vec![0.; values.len()];
            for t in 0..frames {
                for c in 0..channels {
                    reordered[t * channels + c] = values[c * frames + t];
                }
            }
            super::super::weights::fp16_tensor(ctx, &[frames, channels], &reordered).unwrap()
        }
    }
    fn metrics(label: &str, actual: &[f32], expected: &[f32]) -> (f64, f32, usize) {
        assert_eq!(actual.len(), expected.len(), "{label}");
        let mut signal = 0f64;
        let mut error = 0f64;
        let mut max_abs = 0f32;
        let mut violations = 0usize;
        for (&a, &b) in actual.iter().zip(expected) {
            assert!(a.is_finite(), "{label} has non-finite output");
            let diff = (a - b).abs();
            max_abs = max_abs.max(diff);
            violations += usize::from(diff > 0.02 + 0.02 * b.abs());
            signal += f64::from(b).powi(2);
            error += f64::from(a - b).powi(2);
        }
        println!("{label}: elements={} snr_db={:.6} max_abs={max_abs:.6} component_tolerance_violations={violations}",
            actual.len(), 10. * (signal / error).log10());
        (10. * (signal / error).log10(), max_abs, violations)
    }
    fn graph_cfm(
        ctx: &Context,
        cfm: &Arc<Cfm>,
        reference: &Tensor,
        target: &Tensor,
        mel: &Tensor,
        noise: &[Vec<f32>],
    ) -> Tensor {
        let prompt_len = mel.shape().dims()[0];
        let target_len = target.shape().dims()[0];
        let chunk = (CFM_TOTAL_CHUNK - prompt_len).min(CFM_MAX_TARGET_CHUNK);
        let mut rolling_mel = mel.clone();
        let mut rolling_features = reference.clone();
        let mut outputs = Vec::new();
        for (i, start) in (0..target_len).step_by(chunk).enumerate() {
            let len = chunk.min(target_len - start);
            let part = ops::slice_rows(ctx, target, start, len).unwrap();
            let (prompt, features) = if i == 0 {
                (mel.clone(), reference.clone())
            } else {
                let tail = CFM_ROLLING_TAIL
                    .min(prompt_len)
                    .min(rolling_mel.shape().dims()[0])
                    .min(rolling_features.shape().dims()[0]);
                (
                    ops::concat_rows(
                        ctx,
                        &ops::slice_rows(ctx, mel, 0, prompt_len - tail).unwrap(),
                        &ops::slice_rows(
                            ctx,
                            &rolling_mel,
                            rolling_mel.shape().dims()[0] - tail,
                            tail,
                        )
                        .unwrap(),
                    )
                    .unwrap(),
                    ops::concat_rows(
                        ctx,
                        &ops::slice_rows(ctx, reference, 0, prompt_len - tail).unwrap(),
                        &ops::slice_rows(
                            ctx,
                            &rolling_features,
                            rolling_features.shape().dims()[0] - tail,
                            tail,
                        )
                        .unwrap(),
                    )
                    .unwrap(),
                )
            };
            let graph = CfmChunkGraph::prepare(ctx, cfm, prompt_len, len, 4).unwrap();
            transfers::copy_cuda_to_cuda(ctx, &features, &graph.prompt_features).unwrap();
            transfers::copy_cuda_to_cuda(ctx, &part, &graph.target_features).unwrap();
            transfers::copy_cuda_to_cuda(ctx, &prompt, &graph.prompt_mel).unwrap();
            let injected =
                super::super::weights::fp16_tensor(ctx, &[prompt_len + len, 100], &noise[i])
                    .unwrap();
            transfers::copy_cuda_to_cuda(ctx, &injected, &graph.noise).unwrap();
            graph.replay_bound(ctx).unwrap();
            let out = ops::alloc(ctx, &[len, 100]).unwrap();
            transfers::copy_cuda_to_cuda(ctx, &graph.output, &out).unwrap();
            rolling_mel = ops::slice_rows(
                ctx,
                &out,
                len.saturating_sub(prompt_len),
                len.min(prompt_len),
            )
            .unwrap();
            rolling_features = ops::slice_rows(
                ctx,
                &part,
                len.saturating_sub(prompt_len),
                len.min(prompt_len),
            )
            .unwrap();
            outputs.push(out);
        }
        let mut result = outputs.remove(0);
        for out in outputs {
            result = ops::concat_rows(ctx, &result, &out).unwrap();
        }
        result
    }
    #[test]
    #[ignore = "requires APXINF_GPT_CAPTURE_ROOT and CUDA"]
    fn mel_denormalization_matches_reference_and_graph() {
        let root = PathBuf::from(std::env::var("APXINF_GPT_CAPTURE_ROOT").unwrap());
        let ctx = Context::new(0).unwrap();
        let one = super::super::weights::fp16_tensor(&ctx, &[100], &[1.; 100]).unwrap();
        let scale = super::super::weights::fp16_tensor(&ctx, &[100], &[7.; 100]).unwrap();
        let zero = super::super::weights::fp16_tensor(&ctx, &[100], &[0.; 100]).unwrap();
        let minimum = super::super::weights::fp16_tensor(&ctx, &[100], &[-12.; 100]).unwrap();
        for pair in ["pair1", "pair2"] {
            let capture = Capture::load(root.join(pair).join("raw"));
            let input = capture.nlc(&ctx, "synth_v5.0.out_mel");
            let expected = capture.nlc(&ctx, "vocoder.0.mel_in");
            let expected = transfers::to_cpu(&expected).unwrap().to_f32_vec().unwrap();
            let output = ops::alloc(&ctx, input.shape().dims()).unwrap();
            let forward = || {
                let generated = denormalize_mel(&ctx, &input, &one, &scale, &zero, &minimum)?;
                transfers::copy_cuda_to_cuda(&ctx, &generated, &output)
            };
            let session =
                apxinf_cuda_new::ops::ExecutionSession::with_capacity(32 * MIB, 0).unwrap();
            apxinf_cuda_new::ops::prepare_with_session(&session, forward).unwrap();
            let graph = apxinf_cuda_new::capture(&ctx, || {
                apxinf_cuda_new::ops::with_session(&session, forward)
            })
            .unwrap();
            for _ in 0..2 {
                graph.replay().unwrap();
                ctx.synchronize().unwrap();
                assert_eq!(
                    transfers::to_cpu(&output).unwrap().to_f32_vec().unwrap(),
                    expected
                );
            }
            let changed = super::super::weights::fp16_tensor(
                &ctx,
                input.shape().dims(),
                &vec![0.; input.numel()],
            )
            .unwrap();
            transfers::copy_cuda_to_cuda(&ctx, &changed, &input).unwrap();
            graph.replay().unwrap();
            ctx.synchronize().unwrap();
            assert!(transfers::to_cpu(&output)
                .unwrap()
                .to_f32_vec()
                .unwrap()
                .iter()
                .all(|&value| value == -5.0));
            let original = capture.nlc(&ctx, "synth_v5.0.out_mel");
            transfers::copy_cuda_to_cuda(&ctx, &original, &input).unwrap();
            graph.replay().unwrap();
            ctx.synchronize().unwrap();
            assert_eq!(
                transfers::to_cpu(&output).unwrap().to_f32_vec().unwrap(),
                expected
            );
            println!("{pair}: captured denormalization and changed-input graph replay exact");
        }
    }

    #[test]
    #[ignore = "requires captured official fixtures and CUDA checkpoint"]
    fn captured_noise_cfm_step_metrics() {
        let root = PathBuf::from(std::env::var("APXINF_GPT_CAPTURE_ROOT").unwrap());
        let checkpoint = PathBuf::from(std::env::var("APXINF_GPT_CHECKPOINT").unwrap());
        let model = GptSovitsModel::load(&checkpoint, 0).unwrap();
        let nct_to_nlc = |dims: &[usize], values: &[f32]| {
            assert_eq!(dims.len(), 3);
            let (channels, frames) = (dims[1], dims[2]);
            let mut reordered = vec![0.; values.len()];
            for t in 0..frames {
                for c in 0..channels {
                    reordered[t * channels + c] = values[c * frames + t];
                }
            }
            reordered
        };
        for pair in ["pair1", "pair2"] {
            let capture = Capture::load(root.join(pair).join("raw"));
            let reference = capture.nlc(&model.ctx, "synth_v5.0.reference_features");
            let target = capture.nlc(&model.ctx, "synth_v5.0.target_features");
            let mel = capture.nlc(&model.ctx, "synth_v5.0.reference_mel");
            let count = (target.shape().dims()[0] + 499) / 500;
            let noise = (0..count)
                .map(|i| capture.values(&format!("cfm_noise.{i}")).1)
                .collect::<Vec<_>>();
            let noise_refs = noise.iter().map(Vec::as_slice).collect::<Vec<_>>();
            let (_, trace, _) = model
                .cfm
                .synthesize_with_captured_noise(
                    &model.ctx,
                    &reference,
                    &target,
                    &mel,
                    4,
                    0.0,
                    &noise_refs,
                )
                .unwrap();
            assert_eq!(trace.len(), count * 4);
            for (step_index, step) in trace.iter().enumerate() {
                let actual = transfers::to_cpu(&step.x0).unwrap().to_f32_vec().unwrap();
                let (dims, expected) = capture.values(&format!("cfm_steps.{step_index}.x0"));
                metrics(
                    &format!("{pair}/step{step_index}/x0"),
                    &actual,
                    &nct_to_nlc(&dims, &expected),
                );

                let actual = transfers::to_cpu(&step.text0)
                    .unwrap()
                    .to_f32_vec()
                    .unwrap();
                let (dims, expected) = capture.values(&format!("cfm_steps.{step_index}.text0"));
                metrics(
                    &format!("{pair}/step{step_index}/text0"),
                    &actual,
                    &nct_to_nlc(&dims, &expected),
                );

                let actual = transfers::to_cpu(&step.static_condition)
                    .unwrap()
                    .to_f32_vec()
                    .unwrap();
                let (_, expected) =
                    capture.values(&format!("cfm_steps.{step_index}.static_cache.condition"));
                metrics(
                    &format!("{pair}/step{step_index}/static_condition"),
                    &actual,
                    &expected,
                );

                let actual = transfers::to_cpu(&step.velocity)
                    .unwrap()
                    .to_f32_vec()
                    .unwrap();
                let (_, expected) = capture.values(&format!("cfm_steps.{step_index}.velocity"));
                metrics(
                    &format!("{pair}/step{step_index}/velocity"),
                    &actual,
                    &expected,
                );
            }
        }
    }

    #[test]
    #[ignore = "requires captured official fixtures and CUDA checkpoint"]
    fn captured_noise_cfm_and_pcm_comparison() {
        let root = PathBuf::from(std::env::var("APXINF_GPT_CAPTURE_ROOT").unwrap());
        let checkpoint = PathBuf::from(std::env::var("APXINF_GPT_CHECKPOINT").unwrap());
        let model = GptSovitsModel::load(&checkpoint, 0).unwrap();
        for pair in ["pair1", "pair2"] {
            let capture = Capture::load(root.join(pair).join("raw"));
            let reference = capture.nlc(&model.ctx, "synth_v5.0.reference_features");
            let target = capture.nlc(&model.ctx, "synth_v5.0.target_features");
            let mel = capture.nlc(&model.ctx, "synth_v5.0.reference_mel");
            let count = (target.shape().dims()[0] + 499) / 500;
            let noise = (0..count)
                .map(|i| {
                    let (dims, values) = capture.values(&format!("cfm_noise.{i}"));
                    let frames = dims[2];
                    let mut nlc = vec![0.; values.len()];
                    for t in 0..frames {
                        for c in 0..100 {
                            nlc[t * 100 + c] = values[c * frames + t];
                        }
                    }
                    nlc
                })
                .collect::<Vec<_>>();
            let generated = graph_cfm(&model.ctx, &model.cfm, &reference, &target, &mel, &noise);
            let expected = capture.nlc(&model.ctx, "synth_v5.0.out_mel");
            let (_, _, violations) = metrics(
                &format!("{pair}/cfm_captured_conditioning"),
                &transfers::to_cpu(&generated).unwrap().to_f32_vec().unwrap(),
                &transfers::to_cpu(&expected).unwrap().to_f32_vec().unwrap(),
            );
            assert_eq!(
                violations, 0,
                "{pair}: captured-conditioning CFM component tolerance"
            );
            let scale = super::super::weights::fp16_tensor(&model.ctx, &[100], &[7.; 100]).unwrap();
            // Probe the upstream FP16 sequence (x + 1) / 2 * 14 - 12.
            // Dividing by two is exact for these captured normalized mel values.
            let one = super::super::weights::fp16_tensor(&model.ctx, &[100], &[1.; 100]).unwrap();
            let zero = super::super::weights::fp16_tensor(&model.ctx, &[100], &[0.; 100]).unwrap();
            let minimum =
                super::super::weights::fp16_tensor(&model.ctx, &[100], &[-12.; 100]).unwrap();
            let sequential = |input: &Tensor| {
                denormalize_mel(&model.ctx, input, &one, &scale, &zero, &minimum).unwrap()
            };
            let oracle = sequential(&expected);
            let denorm = sequential(&generated);
            let expected_denorm = capture.nlc(&model.ctx, "vocoder.0.mel_in");
            let (_, oracle_max, oracle_violations) = metrics(
                &format!("{pair}/denorm_official_input"),
                &transfers::to_cpu(&oracle).unwrap().to_f32_vec().unwrap(),
                &transfers::to_cpu(&expected_denorm)
                    .unwrap()
                    .to_f32_vec()
                    .unwrap(),
            );
            assert_eq!(oracle_violations, 0);
            assert_eq!(
                oracle_max, 0.0,
                "denormalization must match the captured FP16 sequence"
            );
            metrics(
                &format!("{pair}/denorm"),
                &transfers::to_cpu(&denorm).unwrap().to_f32_vec().unwrap(),
                &transfers::to_cpu(&expected_denorm)
                    .unwrap()
                    .to_f32_vec()
                    .unwrap(),
            );
            let graph = VocoderGraph::prepare(&model.ctx, &model.vocoder, target.shape().dims()[0])
                .unwrap();
            let pcm = graph.replay(&model.ctx, &denorm).unwrap();
            let (_, expected_pcm) = capture.values("vocoder.0.wav_out");
            if let Some(directory) = std::env::var_os("APXINF_CFM_CROSSOVER_CAPTURE") {
                let directory = PathBuf::from(directory);
                fs::create_dir_all(&directory).unwrap();
                for (name, values) in [
                    (
                        "native-normalized-mel",
                        transfers::to_cpu(&generated).unwrap().to_f32_vec().unwrap(),
                    ),
                    (
                        "native-denormalized-mel",
                        transfers::to_cpu(&denorm).unwrap().to_f32_vec().unwrap(),
                    ),
                    ("native-pcm", pcm.clone()),
                ] {
                    let bytes = values
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>();
                    fs::write(directory.join(format!("{pair}-{name}.f32")), bytes).unwrap();
                }
                let official_input_pcm = graph.replay(&model.ctx, &expected_denorm).unwrap();
                let bytes = official_input_pcm
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>();
                fs::write(
                    directory.join(format!("{pair}-official-mel-native-pcm.f32")),
                    bytes,
                )
                .unwrap();
            }
            // Diagnostic only: waveform SNR is not an acceptance gate (see
            // doc/gpt-sovits-v5.md). The FP16 vocoder amplifies sub-ULP mel
            // differences into phase differences; the official implementation
            // reproduces its own captured PCM at only 14-19 dB under the same
            // fixed conditioning (cfm-fixed-input-same-library-v26 report),
            // so a 25 dB gate exceeded the official-vs-official spread. The
            // acceptance gates are the component tolerances asserted above
            // (CFM mel, denormalization) plus the isolated vocoder parity
            // (53-54 dB on identical mel, vocoder-crossover-report.json).
            let (_, _, _) = metrics(
                &format!("{pair}/pcm_captured_conditioning"),
                &pcm,
                &expected_pcm,
            );
        }
    }
}

#[cfg(test)]
mod reference_window_tests {
    use super::reference_feature_window;

    #[test]
    fn short_prompt_uses_feature_head_not_tail() {
        // F09: 4.95 s reference. 552 decoded feature rows, 245 spectrogram
        // frames, 492 prompt mel frames. The window must start at zero.
        assert_eq!(reference_feature_window(552, 245, 492), Some(0));
    }

    #[test]
    fn long_prompt_uses_last_500_frames_of_estimated_mel() {
        // F02: 6.09 s reference, 500-frame capped prompt mel.
        assert_eq!(reference_feature_window(668, 302, 500), Some(104));
    }

    #[test]
    fn window_is_clamped_to_decoded_features() {
        assert_eq!(reference_feature_window(300, 302, 300), Some(0));
        assert_eq!(reference_feature_window(100, 10, 200), None);
    }
}
