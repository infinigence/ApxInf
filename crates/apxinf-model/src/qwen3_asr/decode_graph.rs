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

use apxinf_core::{Error, Result, Shape};
use apxinf_cuda_new::ops::{self, ArgmaxArgs, KvCacheAttentionArgs};
use apxinf_cuda_new::{CudaContext as Context, ExecutionSession, PreparedPhase};

use super::config::*;
use super::decoder::{
    add_rms_norm_into, gemm_weight_into, id_to_host, rms_norm_into, view_rows, Decoder, KvCaches, PrefillWorkspace,
    StepWorkspace,
};
use super::weights::Weights;

/// Arena reservation for the graph's own execution session. Every step
/// output is a caller-owned workspace tensor, so the arena is only used by
/// operator-internal scratch during the prepare traversal; 4 MiB covers it
/// with the same margin qwen38's phases use.
const PHASE_WORKSPACE_BYTES: usize = 4 * 1024 * 1024;

/// One CUDA-graph-capturable decode step, from the persistent token tensor
/// to the argmax in `ws.index`. No host synchronization, no allocation, no
/// H2D copy: everything position-dependent is read on the device.
///
/// `position` (the cache row this step writes) is a host-side parameter used
/// only to seed the attention spec and its prepare-time metadata write; the
/// captured kernels never see it directly.
pub(crate) fn step_body(
    ctx: &Context,
    weights: &mut Weights,
    caches: &KvCaches,
    ws: &mut StepWorkspace,
    position: usize,
) -> Result<()> {
    let kv_len = position + 1;
    // Embedding gather from the persistent token tensor (fixed address).
    let table = weights.get(ctx, "text.embed.w")?;
    ops::embedding_gather(ctx, &table, &ws.token, &mut ws.hidden[0])
        .map_err(|e| Error::Other("embedding_gather: ".to_owned() + &e.to_string()))?;
    let mut cur = 0usize;

    for layer in 0..TXT_LAYERS {
        let p = format!("text.layers.{layer}");
        // Shallow clone sharing the current slot's storage; the add below
        // targets the other slot, so it stays a safe snapshot.
        let residual = ws.hidden[cur].clone();
        if layer == 0 {
            // Layer 0's entry norm has no preceding add to fuse with (the
            // gather writes hidden[0] directly); from layer 1 on, the entry
            // norm is produced by the previous layer's tail fusion (see the
            // end of the loop).
            let iln_w = weights.get(ctx, &format!("{p}.input_layernorm.w"))?;
            rms_norm_into(ctx, &ws.hidden[cur], &iln_w, &mut ws.normed)?;
        }

        // Q/K/V projections (no bias) into the stable buffers.
        gemm_weight_into(
            ctx,
            weights,
            &ws.normed,
            &format!("{p}.self_attn.q.w"),
            &mut ws.q,
        )?; // [1, 2048]
        gemm_weight_into(
            ctx,
            weights,
            &ws.normed,
            &format!("{p}.self_attn.k.w"),
            &mut ws.k,
        )?; // [1, 1024]
        gemm_weight_into(
            ctx,
            weights,
            &ws.normed,
            &format!("{p}.self_attn.v.w"),
            &mut ws.v,
        )?; // [1, 1024]

        // Per-head q_norm / k_norm into fixed buffers. Qwen3's RMSNorm
        // multiplies the weight directly (plain-weight semantics, same as
        // the prefill path's rank-2 norm); the fused kernel below is
        // bit-identical to that. Fixed addresses so the captured graph's
        // fused launch stays valid.
        let q_flat = ws.q.reshape(Shape::new(vec![TXT_HEADS, TXT_HEAD_DIM]))?;
        let k_flat = ws.k.reshape(Shape::new(vec![TXT_KV_HEADS, TXT_HEAD_DIM]))?;
        let q_norm_w = weights.get(ctx, &format!("{p}.self_attn.q_norm.w"))?;
        let k_norm_w = weights.get(ctx, &format!("{p}.self_attn.k_norm.w"))?;
        // Fused per-head RMSNorm + full-width rope: bit-identical to the
        // rms_norm_into + partial_rope pair it replaces (norm.cuh plain-weight
        // semantics, 256-thread reduction, bf16 round-trip preserved).
        ops::qk_norm_rope(
            ctx,
            &q_flat,
            &q_norm_w,
            &mut ws.q_normed,
            &ws.positions,
            TXT_HEADS,
            TXT_HEAD_DIM,
            TXT_HEAD_DIM,
            ROPE_THETA,
            RMS_EPS,
        )
        .map_err(|e| Error::Other(format!("layer {layer} q norm+rope: {e}")))?;
        ops::qk_norm_rope(
            ctx,
            &k_flat,
            &k_norm_w,
            &mut ws.k_normed,
            &ws.positions,
            TXT_KV_HEADS,
            TXT_HEAD_DIM,
            TXT_HEAD_DIM,
            ROPE_THETA,
            RMS_EPS,
        )
        .map_err(|e| Error::Other(format!("layer {layer} k norm+rope: {e}")))?;

        // Heads views are taken AFTER the fused kernel has written
        // q_normed/k_normed: a view built before the write could hand the
        // attention/cache stage stale pre-norm data.
        let q_heads = ws
            .q_normed
            .reshape(Shape::new(vec![1, TXT_HEADS, TXT_HEAD_DIM]))?;
        let k_heads = ws
            .k_normed
            .reshape(Shape::new(vec![1, TXT_KV_HEADS, TXT_HEAD_DIM]))?;

        // K/V into the cache row `position`: the scatter kernel reads the
        // row index from device memory, so the captured launch is valid for
        // every later position too.
        let k_row = k_heads.reshape(Shape::new(vec![1, TXT_KV_HEADS * TXT_HEAD_DIM]))?;
        ops::cache_append_row(ctx, &k_row, &caches.keys[layer], &ws.positions)
            .map_err(|e| Error::Other(format!("layer {layer} k cache append: {e}")))?;
        ops::cache_append_row(ctx, &ws.v, &caches.values[layer], &ws.positions)
            .map_err(|e| Error::Other(format!("layer {layer} v cache append: {e}")))?;

        // Attention over the full-capacity cache; the runtime length and
        // query position come from the fixed-address decode metadata, read
        // by the kernel during replay.
        let k_cache = caches.key_view(layer, 0, KV_CAPACITY)?;
        let v_cache = caches.value_view(layer, 0, KV_CAPACITY)?;
        let q_attn = q_heads.reshape(Shape::new(vec![1, 1, TXT_HEADS, TXT_HEAD_DIM]))?;
        {
            let mut out_heads = ws
                .attn_out
                .reshape(Shape::new(vec![1, 1, TXT_HEADS, TXT_HEAD_DIM]))?;
            let mut args =
                KvCacheAttentionArgs::new(&q_attn, &k_cache, &v_cache, &mut out_heads)
                    .with_decode_meta(&ws.decode_meta);
            // Seeds the prepare traversal's metadata write; during capture
            // the spec records capacity/0 and the kernel reads the live
            // values instead.
            args.valid_key_tokens = kv_len;
            args.query_start = position;
            ops::kv_cache_attention(ctx, args).map_err(|e| {
                Error::Other(format!("layer {layer} kv attention (kv_len {kv_len}): {e}"))
            })?;
        }
        gemm_weight_into(
            ctx,
            weights,
            &ws.attn_out,
            &format!("{p}.self_attn.o.w"),
            &mut ws.attn_proj,
        )?;
        // Residual add + next-entry norm fused (C3): one kernel in place of
        // add_into + rms_norm_into. weights.get has no graph node, so
        // hoisting it above the fused launch keeps the captured node order
        // identical.
        let pln_w = weights.get(ctx, &format!("{p}.post_attention_layernorm.w"))?;
        add_rms_norm_into(
            ctx,
            &residual,
            &ws.attn_proj,
            &pln_w,
            &mut ws.hidden[(cur + 1) % 2],
            &mut ws.normed2,
        )?;
        cur = (cur + 1) % 2;

        // MLP: fused gate+up -> SwiGLU -> down.
        let residual2 = ws.hidden[cur].clone();
        gemm_weight_into(
            ctx,
            weights,
            &ws.normed2,
            &format!("{p}.mlp.gate_up.w"),
            &mut ws.fused,
        )?; // [1, 12288]
        ops::swiglu_into(ctx, &ws.fused, &ws.inner)
            .map_err(|e| Error::Other(format!("layer {layer} swiglu: {e}")))?;
        gemm_weight_into(ctx, weights, &ws.inner, &format!("{p}.mlp.down.w"), &mut ws.outer)?;
        // Tail fusion (C3): the residual add pairs with the next layer's
        // entry norm (or the final norm on the last layer) into one kernel;
        // the two original nodes are adjacent in the captured stream (only
        // alias clones and weights.get sit between them).
        if layer + 1 < TXT_LAYERS {
            let next_p = format!("text.layers.{}", layer + 1);
            let next_iln_w = weights.get(ctx, &format!("{next_p}.input_layernorm.w"))?;
            add_rms_norm_into(
                ctx,
                &residual2,
                &ws.outer,
                &next_iln_w,
                &mut ws.hidden[(cur + 1) % 2],
                &mut ws.normed,
            )?;
        } else {
            let norm_w = weights.get(ctx, "text.norm.w")?;
            add_rms_norm_into(
                ctx,
                &residual2,
                &ws.outer,
                &norm_w,
                &mut ws.hidden[(cur + 1) % 2],
                &mut ws.final_norm,
            )?;
        }
        cur = (cur + 1) % 2;
    }
    // lm_head logits + argmax into the workspace buffers (view first, then
    // weights.get — same borrow order as logits_argmax_into).
    let last = view_rows(&ws.final_norm, 0, 1, TXT_HIDDEN)?;
    gemm_weight_into(ctx, weights, &last, "text.lm_head.w", &mut ws.logits)?;
    ops::argmax(ctx, ArgmaxArgs::new(&ws.logits, &ws.index))
        .map_err(|e| Error::Other(format!("argmax: {e}")))?;
    Ok(())
}

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
        super::decoder::dump_logits_step(ctx, ws)?;
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
