//! qwen3-asr model runtime: waveform -> mel -> audio tower -> Qwen3 decode.

use std::cell::RefCell;
use std::path::Path;
use std::sync::Arc;

use apxinf_core::{Device, Error, Result, Tensor};

use super::config::*;
use super::decode_graph::DecodeState;
use super::decoder::{ensure_prefill_workspace, Decoder};
use super::encoder;
use super::frontend::Frontend;
use super::weights::Weights;
use crate::asr_trait::{
    AsrContract, AsrRequest, AsrStopReason, AsrTrait, AsrTranscription,
};
use apxinf_cuda_new::{CudaContext as Context, ExecutionSession};

/// Arena reservation for the frontend + audio tower + prefill stage.
///
/// Most stage outputs are caller-owned tensors (the encoder and the prefill
/// path write into persistent workspaces), but the log-mel frontend's
/// intermediates (frames [FRAMES, N_FFT] f32, power bins, mel bins) are
/// allocated through the session, so the arena must hold the largest of them:
/// the frames buffer alone is 4.8 MB at the 30 s window. 16 MB leaves headroom
/// for the remaining stage scratch.
const STAGE_WORKSPACE_BYTES: usize = 16 * 1024 * 1024;

/// Execution session for one stage shape, kept across transcriptions.
///
/// A session can only `run` the operator sequence it prepared, so the session
/// is keyed by the valid frame count: every stage buffer shape derives from it
/// (the audio token count, and through it the prompt length, is a pure function
/// of it). A repeated key replays prepared executions instead of paying for
/// native preparation on every transcription.
struct StageSession {
    valid_frames: usize,
    session: ExecutionSession,
}

pub struct Qwen3AsrModel {
    ctx: Arc<Context>,
    frontend: Frontend,
    /// Lazy-upload weight store: uploads are mutating, `AsrTrait::transcribe`
    /// takes `&self` (the same interior-mutability shape as the other ASR
    /// models).
    weights: RefCell<Weights>,
    /// Persistent decode state (KV caches + fixed-address step workspace +
    /// the captured decode graph, built lazily). Owned by the model so the
    /// graph's captured addresses stay valid across transcriptions; see
    /// `decode_graph::DecodeState`.
    decode_state: RefCell<DecodeState>,
    /// Persistent audio-tower workspace, rebuilt when the valid frame count
    /// changes (see `encoder::EncodeWorkspace`). Fixed intermediate addresses
    /// keep the stage session's prepared executions valid across
    /// transcriptions and remove thousands of per-transcription
    /// `cudaMalloc`/`cudaFree` calls.
    encode_ws: RefCell<Option<encoder::EncodeWorkspace>>,
    /// Frontend + audio tower + prefill session, see [`StageSession`].
    stage: RefCell<Option<StageSession>>,
}

impl Qwen3AsrModel {
    /// Load a converted checkpoint directory (single `weights.safetensors`).
    pub fn load(directory: &Path, device_id: usize, autotune: bool) -> Result<Self> {
        let ctx = Context::new_with_autotune(device_id, autotune).map_err(Error::Cuda)?;
        let mut weights = Weights::load(directory)?;
        let frontend = Frontend::load(&mut weights, &ctx)?;
        let decode_state = RefCell::new(DecodeState::alloc(&ctx)?);
        Ok(Self {
            ctx: Arc::new(ctx),
            frontend,
            weights: RefCell::new(weights),
            decode_state,
            encode_ws: RefCell::new(None),
            stage: RefCell::new(None),
        })
    }

    /// Full greedy transcription used by both `AsrTrait::transcribe` and the
    /// parity tests. The prompt is built internally from the fixed chat
    /// template; `prompt_token_ids` must be empty (v1 does not splice user
    /// text prefixes).
    fn transcribe_internal(&self, waveform: &[f32], language: &str) -> Result<AsrTranscription> {
        let suffix = prompt_suffix(language).ok_or_else(|| {
            Error::Other(format!(
                "qwen3-asr supports the languages {SUPPORTED_LANGUAGES:?}, got {language:?}"
            ))
        })?;
        let samples = waveform.len();
        if samples > SAMPLES {
            return Err(Error::Other(format!(
                "waveform has {samples} samples; qwen3-asr accepts at most {SAMPLES} (30 s at 16 kHz)"
            )));
        }

        // Canonical window: zero-pad to 30 s on the host, upload once.
        let mut wave = vec![0.0f32; SAMPLES];
        wave[..samples].copy_from_slice(&waveform[..samples]);
        let host = Tensor::from_f32(vec![SAMPLES], &wave)?;
        let wave_dev =
            apxinf_cuda_new::transfers::to_cuda(&host, self.ctx.device_id())?;
        if wave_dev.device() != Device::Cuda(self.ctx.device_id()) {
            return Err(Error::Other("qwen3-asr waveform upload failed".into()));
        }
        let valid = valid_frames(samples);
        let timing = std::env::var("APXINF_QWEN3ASR_TIMING").is_ok();
        let t0 = std::time::Instant::now();

        // CUDA Graph decode is the default; APXINF_QWEN3ASR_GRAPH=0 selects
        // the eager loop. Decode always runs after the stage session: graph
        // capture lives in its own prepared phase, and nested sessions are
        // rejected.
        let graph_decode = std::env::var("APXINF_QWEN3ASR_GRAPH")
            .map(|value| value != "0")
            .unwrap_or(true);

        // Persistent workspaces are caller-owned and live across
        // transcriptions, so they are built here, outside the stage session:
        // `allocate_output` inside a session sub-allocates from its
        // deterministic arena, which is reserved for operator-internal scratch.
        // Both workspace sizes are pure functions of the valid-frame count,
        // so the prefill row count below matches the prompt `run_stage` builds.
        encoder::ensure_workspace(&self.ctx, &mut self.encode_ws.borrow_mut(), valid)?;
        let prompt_len = prompt_prefix().len()
            + encoder::audio_token_count(valid)
            + suffix.len();
        {
            let mut state = self.decode_state.borrow_mut();
            ensure_prefill_workspace(&self.ctx, &mut state.prefill_ws, prompt_len)?;
        }

        // Stage: frontend + audio tower + prefill. The first transcription for
        // a valid-frame count prepares its native executions; repeats of the
        // same shape replay the prepared sequence. `prepare` executes the
        // traversal for real, so only the miss case runs the stage once.
        let (first_token, prompt_len) = {
            let mut slot = self.stage.borrow_mut();
            let reuse = matches!(slot.as_ref(), Some(existing) if existing.valid_frames == valid);
            if reuse {
                let session = slot
                    .as_ref()
                    .map(|existing| existing.session.clone())
                    .ok_or_else(|| Error::Other("qwen3-asr stage session disappeared".into()))?;
                session.run(|| self.run_stage(&wave_dev, valid, &suffix))?
            } else {
                *slot = None;
                let session = ExecutionSession::with_capacity(
                    STAGE_WORKSPACE_BYTES,
                    self.ctx.device_id(),
                )?;
                let prepared = session.prepare(|| self.run_stage(&wave_dev, valid, &suffix))?;
                *slot = Some(StageSession {
                    valid_frames: valid,
                    session,
                });
                prepared
            }
        };
        let t_stage = t0.elapsed();

        let t2 = std::time::Instant::now();
        let decoder = Decoder::new();
        let generated = if graph_decode {
            let mut state = self.decode_state.borrow_mut();
            let mut weights = self.weights.borrow_mut();
            decoder.generate_graph(
                &self.ctx,
                &mut weights,
                &mut state,
                first_token,
                prompt_len,
                MAX_NEW_TOKENS,
                &EOS_TOKENS,
            )?
        } else {
            let mut state = self.decode_state.borrow_mut();
            let mut weights = self.weights.borrow_mut();
            let session =
                ExecutionSession::with_capacity(STAGE_WORKSPACE_BYTES, self.ctx.device_id())?;
            session.prepare(|| {
                decoder.decode_eager(
                    &self.ctx,
                    &mut weights,
                    &mut state,
                    first_token,
                    prompt_len,
                    MAX_NEW_TOKENS,
                    &EOS_TOKENS,
                )
            })?
        };
        if timing {
            eprintln!(
                "[qwen3-asr] samples={samples} stage={t_stage:?} decode={:?} total={:?}",
                t2.elapsed(),
                t0.elapsed()
            );
        }

        let stopped_reason = match generated.last() {
            Some(&id) if EOS_TOKENS.contains(&id) => AsrStopReason::Eos,
            _ => AsrStopReason::MaxTokens,
        };
        let valid_tokens = generated.len();
        Ok(AsrTranscription {
            token_ids: generated,
            stopped_reason,
            valid_tokens,
        })
    }

    /// Log-mel frontend + audio tower + prefill: everything that runs before
    /// the decode loop, and the traversal a [`StageSession`] prepares.
    fn run_stage(&self, wave_dev: &Tensor, valid: usize, suffix: &[u32]) -> Result<(i32, usize)> {
        let t0 = std::time::Instant::now();
        let mel = self.frontend.run(&self.ctx, wave_dev, valid)?;
        let t_frontend = t0.elapsed();

        let mut weights = self.weights.borrow_mut();
        let mut encode_ws = self.encode_ws.borrow_mut();
        let t1 = std::time::Instant::now();
        let encoded = encoder::encode(
            &self.ctx,
            &mut weights,
            &mut encode_ws,
            &mel,
            valid,
            None,
        )?;
        let t_encode = t1.elapsed();
        let n_audio = encoded.n;

        // Prompt: prefix + audio pads + suffix; audio pads are placeholders
        // overwritten by the projected embeddings during prefill.
        let mut prompt: Vec<i32> = prompt_prefix().iter().map(|&id| id as i32).collect();
        let audio_offset = prompt.len();
        prompt.extend((0..n_audio).map(|_| AUDIO_TOKEN as i32));
        prompt.extend(suffix.iter().map(|&id| id as i32));

        let decoder = Decoder::new();
        // RefMut derefs mutably only once: take a raw `&mut` so the two
        // disjoint fields (caches / prefill_ws) can be borrowed together.
        let mut state = self.decode_state.borrow_mut();
        let state_ref = &mut *state;
        let next = decoder.prefill(
            &self.ctx,
            &mut weights,
            &mut state_ref.caches,
            &mut state_ref.prefill_ws,
            &prompt,
            &encoded.projected,
            audio_offset,
            n_audio,
        )?;
        if std::env::var("APXINF_QWEN3ASR_TIMING").is_ok() {
            eprintln!(
                "[qwen3_asr] frontend={t_frontend:?} encode={t_encode:?} stage={:?}",
                t0.elapsed()
            );
        }
        Ok((next, prompt.len()))
    }
}

impl AsrTrait for Qwen3AsrModel {
    fn contract(&self) -> AsrContract {
        AsrContract {
            sample_rate: SAMPLE_RATE,
            num_mel_bins: MEL_BINS,
            max_audio_samples: SAMPLES,
            max_decoder_positions: KV_CAPACITY,
            multilingual: true,
            languages: SUPPORTED_LANGUAGES.iter().map(|s| s.to_string()).collect(),
        }
    }

    fn transcribe(&self, request: &AsrRequest<'_>) -> Result<AsrTranscription> {
        if !request.prompt_token_ids.is_empty() {
            return Err(Error::Other(
                "qwen3-asr builds its prompt internally; pass an empty prompt_token_ids".into(),
            ));
        }
        let language = request.language.ok_or_else(|| {
            Error::Other("qwen3-asr needs an explicit language (no auto-detection)".into())
        })?;
        if let Some(task) = request.task {
            if task != "transcribe" {
                return Err(Error::Other(format!(
                    "qwen3-asr supports task \"transcribe\", got {task:?}"
                )));
            }
        }
        self.transcribe_internal(request.waveform, language)
    }
}

/// Languages with a hard-coded prompt suffix in v1.
pub const SUPPORTED_LANGUAGES: [&str; 1] = ["Chinese"];
