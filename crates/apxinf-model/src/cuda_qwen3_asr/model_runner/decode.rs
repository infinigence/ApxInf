//! CUDA Graph decode for qwen3-asr: capture one decode step, replay it for
//! every generated token.
//!
//! The decode loop's ~170 launches/step are pure launch-gap overhead on
//! Orin (48 steps x ~170 launches ≈ 1.3 s of the 3.21 s total, quantified by
//! the torch.compile + CUDA Graph baseline at 1.89 s). A captured graph
//! collapses each step to one `cudaGraphLaunch`.
//!
//! Graph-safety requirements, and where each is met:
//!
//! * **Fixed addresses for every intermediate** — the persistent
//!   [`StepWorkspace`] (introduced for the relaxed-session cache in v1) is
//!   exactly the precondition capture needs; the per-head q/k norm outputs
//!   move from fresh allocations into fixed `q_normed`/`k_normed` buffers.
//! * **Token id and position as device values** — persistent `[1]` I32
//!   tensors (`ws.token`, `ws.positions`) rewritten by 4-byte host copies
//!   between replays; the embedding gather and both RoPE kernels read them
//!   at kernel runtime.
//! * **KV-cache row write** — `ops::cache_append_row` scatters one row at
//!   the index read from device memory, replacing the host-offset
//!   `copy_cuda_to_cuda` views that would need recapture per position.
//! * **Attention length/position** — `ops::kv_cache_attention` in dynamic
//!   decode mode over the *full-capacity* cache, with the runtime
//!   `[valid_key_tokens, query_start]` read from fixed-address
//!   [`KvCacheDecodeMeta`].
//! * **Greedy token** — `ops::argmax` writes into the persistent `ws.index`
//!   inside the graph; the host reads it back after each replay (the single
//!   per-step sync, which also orders the next step's metadata update
//!   against attention kernels still in flight).
//!
//! Capture runs in `DecodeGraph::capture` via
//! [`PreparedPhase::prepare_and_capture`]: the prepare traversal executes the
//! first decode step for real (preparing every native execution at the
//! workspace's addresses) and leaves the step's greedy argmax in `ws.index`;
//! the capture traversal then records the same launches without executing
//! them. `PreparedPhase` must not be built inside another execution session
//! (nested sessions are rejected), so the model runs frontend + encoder +
//! prefill under its relaxed session and only then captures.

use apxinf_core::{Error, Result};
use apxinf_cuda_new::{CudaContext as Context, ExecutionSession, PreparedPhase};

use crate::cuda_qwen3_asr::config::*;
use crate::cuda_qwen3_asr::model::decode_step::step_body;
use crate::cuda_qwen3_asr::model::decoder::{
    id_to_host, Decoder, KvCaches, PrefillWorkspace, StepWorkspace,
};
use crate::cuda_qwen3_asr::weights::Weights;

/// Arena reservation for the graph's own execution session. Every step
/// output is a caller-owned workspace tensor, so the arena is only used by
/// operator-internal scratch during the prepare traversal; 4 MiB covers it
/// with the same margin qwen38's phases use.
const PHASE_WORKSPACE_BYTES: usize = 4 * 1024 * 1024;

/// Persistent decode state owned by the model and shared across
/// transcriptions: the full-capacity KV caches, the fixed-address step
/// workspace, and (captured once, lazily) the decode graph.
///
/// Persisting them is what makes the whole scheme sound:
///
/// * the captured graph addresses caches/workspace/weight tensors directly,
///   so they must live as long as the model — re-allocating per
///   transcription would invalidate every captured pointer;
/// * the relaxed session's execution cache is keyed on binding addresses, so
///   fixed workspace/cache addresses make the keys recur between
///   transcriptions (fresh per-transcription allocations changed the
///   addresses, prepared a whole new set of native executions each
///   transcription, and leaked their cuBLASLt workspaces — ~5 GiB per call
///   on orin, exhausting the heap after ~5 transcriptions);
/// * capture runs once, not per transcription, so its prepare cost is paid
///   on the first decode only.
///
/// The graph itself is prompt-length-agnostic: the cache row to write and
/// the attention length come from device-side values, so one capture serves
/// every transcription (bounded by `KV_CAPACITY`).
pub(crate) struct DecodeState {
    /// Full-capacity KV caches; prefill rewrites rows `[0, T)` each
    /// transcription (rows above `T` are stale but never read).
    pub(crate) caches: KvCaches,
    /// Fixed-address decode intermediates, also used by the eager path.
    pub(crate) ws: StepWorkspace,
    /// Persistent prefill (m=T) workspace, for the same reasons as `ws`: fixed
    /// intermediate addresses drop the `cudaMalloc` calls and make the
    /// execution cache recur between transcriptions. Built lazily for the
    /// prompt length T and rebuilt when it changes (decided at the prefill
    /// entry point), then reused across transcriptions for the model's
    /// lifetime.
    pub(crate) prefill_ws: Option<PrefillWorkspace>,
    graph: Option<DecodeGraph>,
}

impl DecodeState {
    pub(crate) fn alloc(ctx: &Context) -> Result<Self> {
        Ok(Self {
            caches: KvCaches::alloc(ctx)?,
            ws: StepWorkspace::alloc(ctx)?,
            prefill_ws: None,
            graph: None,
        })
    }
}

/// The captured decode step. `ws`, the KV caches and the weight tensors are
/// owned by the caller and must outlive this object (the graph's captured
/// nodes address them directly); `generate_graph` guarantees that by owning
/// all three for the duration of the loop.
pub(crate) struct DecodeGraph {
    phase: PreparedPhase,
}

impl DecodeGraph {
    /// Capture the decode step for `first_token` writing cache row
    /// `first_position`.
    ///
    /// The prepare traversal runs that step for real — preparing every
    /// native execution at the workspace's fixed addresses and leaving the
    /// step's greedy argmax in `ws.index` — and the capture traversal
    /// records the identical launches.
    pub(crate) fn capture(
        ctx: &Context,
        weights: &mut Weights,
        caches: &KvCaches,
        ws: &mut StepWorkspace,
        first_token: i32,
        first_position: usize,
    ) -> Result<Self> {
        if first_position + 1 > KV_CAPACITY {
            return Err(Error::Other(format!(
                "kv cache overflow: {} rows exceed capacity {KV_CAPACITY}",
                first_position + 1
            )));
        }
        ws.set_token(first_token)?;
        ws.set_position(first_position)?;
        let session = ExecutionSession::with_capacity(PHASE_WORKSPACE_BYTES, ctx.device_id())?;
        let phase = PreparedPhase::prepare_and_capture(ctx, session, || {
            step_body(ctx, weights, caches, ws, first_position)
        })?;
        if std::env::var("APXINF_QWEN3ASR_TIMING").is_ok() {
            eprintln!(
                "[qwen3-asr] decode graph captured: workspace {}/{} bytes",
                phase.workspace_used(),
                phase.workspace_capacity()
            );
        }
        Ok(Self { phase })
    }

    /// Replay one decode step: `token` at cache row `position`, greedy next
    /// token back on the host.
    ///
    /// The metadata update is safe against the previous replay: `step` is
    /// only ever called after the previous step's `id_to_host` synchronized
    /// the stream.
    pub(crate) fn step(
        &self,
        ctx: &Context,
        ws: &mut StepWorkspace,
        token: i32,
        position: usize,
    ) -> Result<i32> {
        ws.decode_meta
            .update(position + 1, position)
            .map_err(|e| Error::Other(format!("decode meta update: {e}")))?;
        ws.set_token(token)?;
        ws.set_position(position)?;
        self.phase.replay()?;
        let next = id_to_host(ctx, &ws.index)?;
        crate::cuda_qwen3_asr::model::decoder::dump_logits_step(ctx, ws)?;
        Ok(next)
    }
}

impl Decoder {
    /// Greedy decode loop with the step captured as one CUDA Graph
    /// (APXINF_QWEN3ASR_GRAPH=0 selects the eager [`Decoder::decode_eager`]
    /// path instead). Prefill has already run into `state.caches`:
    /// `first_token` is the token its argmax selected and `prompt_len` the
    /// row the first step writes. Returns the generated ids (EOS included).
    ///
    /// The graph is captured once, on the first transcription that reaches
    /// decode; every later transcription (any prompt length) replays it.
    pub(crate) fn generate_graph(
        &self,
        ctx: &Context,
        weights: &mut Weights,
        state: &mut DecodeState,
        first_token: i32,
        prompt_len: usize,
        max_new_tokens: usize,
        eos_tokens: &[u32],
    ) -> Result<Vec<u32>> {
        if prompt_len + 1 >= KV_CAPACITY {
            return Err(Error::Other(format!(
                "kv cache overflow: prompt {prompt_len} rows leave no decode room in capacity {KV_CAPACITY}"
            )));
        }
        let mut generated: Vec<u32> = Vec::with_capacity(max_new_tokens);
        // The prefill token itself: same loop-entry bookkeeping as the eager
        // path, so both modes push the identical sequence.
        if eos_tokens.contains(&(first_token as u32)) {
            generated.push(first_token as u32);
            return Ok(generated);
        }
        generated.push(first_token as u32);
        if generated.len() >= max_new_tokens {
            return Ok(generated);
        }

        // Split borrows: capture needs caches+ws+weights, replays need
        // graph+ws.
        let DecodeState { caches, ws, graph, .. } = state;
        if graph.is_none() {
            let captured =
                DecodeGraph::capture(ctx, weights, caches, ws, first_token, prompt_len)?;
            *graph = Some(captured);
        }
        let graph = graph.as_ref().expect("captured above");

        // First decode step. For a freshly captured graph this replays the
        // step the prepare traversal already ran — idempotent (same token,
        // same cache row, same logits) — which keeps one code path for fresh
        // captures and reused graphs.
        let mut next = graph.step(ctx, ws, first_token, prompt_len)?;
        let mut position = prompt_len + 1;
        let timing = std::env::var("APXINF_QWEN3ASR_TIMING").is_ok();
        let mut steps: usize = 0;
        let mut step_us: u128 = 0;
        loop {
            if eos_tokens.contains(&(next as u32)) {
                generated.push(next as u32);
                return Ok(generated);
            }
            generated.push(next as u32);
            if generated.len() >= max_new_tokens {
                return Ok(generated);
            }
            if position + 1 >= KV_CAPACITY {
                // Position cap: stop without consuming another row.
                return Ok(generated);
            }
            let t_step = std::time::Instant::now();
            next = graph.step(ctx, ws, next, position)?;
            step_us += t_step.elapsed().as_micros();
            steps += 1;
            position += 1;
            if timing && steps % 16 == 0 {
                eprintln!(
                    "[qwen3-asr] decode replay {steps} cumulative={:?}",
                    std::time::Duration::from_micros(step_us as u64)
                );
            }
        }
    }
}

impl Decoder {
    /// Greedy decode loop, eager mode (no CUDA Graph): every step runs
    /// op-by-op through [`crate::cuda_qwen3_asr::model::decode_step::step_body`].
    /// Prefill has already run; `first_token` is the token its argmax
    /// selected and `prompt_len` the row the first step writes. Returns the
    /// generated ids (EOS included).
    pub(crate) fn decode_eager(
        &self,
        ctx: &Context,
        weights: &mut Weights,
        state: &mut DecodeState,
        first_token: i32,
        prompt_len: usize,
        max_new_tokens: usize,
        eos_tokens: &[u32],
    ) -> Result<Vec<u32>> {
        let mut next = first_token;
        let mut generated: Vec<u32> = Vec::with_capacity(max_new_tokens);
        let mut position = prompt_len;
        let timing = std::env::var("APXINF_QWEN3ASR_TIMING").is_ok();
        let mut steps: usize = 0;
        let mut step_us: u128 = 0;
        // Persistent decode workspace (model-level DecodeState): every step
        // reuses the same intermediate addresses, which is also what makes
        // the relaxed session's execution-cache keys recur between
        // transcriptions.
        let DecodeState { caches, ws, .. } = state;
        loop {
            if eos_tokens.contains(&(next as u32)) {
                generated.push(next as u32);
                return Ok(generated);
            }
            generated.push(next as u32);
            if generated.len() >= max_new_tokens {
                return Ok(generated);
            }
            if position + 1 >= KV_CAPACITY {
                // Position cap: stop without consuming another row.
                return Ok(generated);
            }
            let t_step = std::time::Instant::now();
            next = self.step_fast(ctx, weights, caches, ws, next, position)?;
            step_us += t_step.elapsed().as_micros();
            steps += 1;
            position += 1;
            if timing && steps % 16 == 0 {
                eprintln!(
                    "[qwen3_asr] decode step {steps} cumulative={:?}",
                    std::time::Duration::from_micros(step_us as u64)
                );
            }
        }
    }

    /// One eager decode step: publish the token/position, run
    /// [`crate::cuda_qwen3_asr::model::decode_step::step_body`], and read the
    /// greedy next token back (the `id_to_host` sync is also what orders the
    /// next step's `KvCacheDecodeMeta` update against kernels still in
    /// flight).
    fn step_fast(
        &self,
        ctx: &Context,
        weights: &mut Weights,
        caches: &mut KvCaches,
        ws: &mut StepWorkspace,
        token: i32,
        position: usize,
    ) -> Result<i32> {
        let kv_len = position + 1;
        if kv_len > KV_CAPACITY {
            return Err(Error::Other(format!(
                "kv cache overflow: {kv_len} rows exceed capacity {KV_CAPACITY}"
            )));
        }
        ws.set_token(token)?;
        ws.set_position(position)?;
        step_body(ctx, weights, caches, ws, position)?;
        let next = id_to_host(ctx, &ws.index)?;
        crate::cuda_qwen3_asr::model::decoder::dump_logits_step(ctx, ws)?;
        Ok(next)
    }
}
