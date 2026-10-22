//! Execution ownership: CUDA-graph capture, replay-driven decode and
//! session state reset.
//!
//! Calls the model computation in `super::model`; does not re-implement it.


use apxinf_cuda_new::{ops, CudaBuffer, CudaContext, PreparedPhase};

use super::backend::{graph_tensor_bytes, zero_tensor};
use super::config::*;
use super::model::{
    decode_step_inner, gdn_decode_layer, nvfp4_mlp, GdnState, KvCache, PrefillScratch, Scratch,
};
use super::weights::{Layer, Model};


/// Install the validated kernel-path selectors, unless the caller already set
/// one. Called by `Qwen38::new`; capture-level tests must call it too, or they
/// would build graphs under a configuration the acceptance run never used.
pub(crate) fn apply_validated_kernel_defaults() {
    for (key, value) in [
        // Tiled GDN prefill conv path, tile 64 (report 05/23).
        ("APXINF_GDN_PREFILL_TILED", "1"),
        ("APXINF_GDN_PREFILL_TILE", "64"),
        // cuBLASLt native-NVFP4 candidates for the block-scaled GEMMs
        // (report 06).
        ("APXINF_GEMM_CUBLASLT_NVFP4", "1"),
        // 16-byte vectorized elementwise kernels (report 08).
        ("APXINF_QWEN38_VECTOR_ELEMENTWISE", "1"),
        // Paired-channel conv decode kernel (report 23).
        ("APXINF_GDN_CONV_PAIR", "1"),
        // The shared-activation FP8 GEMV variant lost its A/B (report 25).
        ("APXINF_FP8_GEMV_SHARED", "0"),
        // Native fp8x2 pair quantization, part of the D3 contract
        // (reports 30/62).
        ("APXINF_FP8_NATIVE_PAIR", "1"),
        // Five-warp-group FlashInfer prepare reduction (report 11).
        ("APXINF_GDN_PREPARE_PARALLEL", "5"),
        // Single split for the split-KV FA2 decode kernel (report 47).
        ("APXINF_FA2_DECODE_SPLITS", "1"),
    ] {
        if std::env::var_os(key).is_none() {
            std::env::set_var(key, value);
        }
    }
}

/// Snapshot the persistent GDN tensors so a graph capture can be undone.
///
/// `PreparedPhase::prepare_and_capture` runs the phase body for real once
/// before capturing it, and `gdn_decode_layer` is read-modify-write on both of
/// these tensors (`gdn_causal_conv_step` shifts the window in place,
/// `gdn_recurrent_step` decays and updates the state in place). `Qwen38::forward`
/// calls `ensure_graphs()` immediately before `prefill_step` with no reset in
/// between, and GDN prefill reads `state.recurrent` in place as its initial
/// carry — so a leftover write from the prepare traversal would land in the
/// first prefill chunk.
///
/// Today that is invisible, but only by accident: capture happens with the
/// state and `scratch.hidden` at bitwise zero, where `s*exp(decay) + ((v -
/// s·k)*beta)*k` is multiplicatively zero and the prepare pass writes zeros
/// over zeros. Nothing enforces that precondition, and the kernel harness
/// hides a regression of it with a `reset_state` right before its timed
/// prefill that this module does not do.
///
/// Snapshotted per layer rather than for all 48 at once: 3 MiB of recurrent
/// state plus 160 KiB of conv window keeps ~3.2 MiB of host staging live at a
/// time instead of ~151 MiB.
fn snapshot_gdn_state(ctx: &CudaContext, state: &GdnState) -> (Vec<u8>, Vec<u8>) {
    // The context stream is non-blocking, so a blocking cudaMemcpy on the null
    // stream is not ordered against work already queued on it. Drain first.
    ctx.synchronize().unwrap();
    (
        graph_tensor_bytes(&state.recurrent),
        graph_tensor_bytes(&state.conv_window),
    )
}

/// Put back what [`snapshot_gdn_state`] saved. Safe to call after capture: the
/// graph records device addresses, not values.
fn restore_gdn_state(ctx: &CudaContext, state: &GdnState, saved: &(Vec<u8>, Vec<u8>)) {
    ctx.synchronize().unwrap();
    CudaBuffer::from_tensor(&state.recurrent)
        .unwrap()
        .copy_from_host(&saved.0)
        .unwrap();
    CudaBuffer::from_tensor(&state.conv_window)
        .unwrap()
        .copy_from_host(&saved.1)
        .unwrap();
    ctx.synchronize().unwrap();
}

/// `APXINF_QWEN38_GRAPH_REPORT=1` prints how much of each phase arena the
/// capture actually used; `=2` adds one line per phase.
///
/// This is the instrument that showed [`PHASE_WORKSPACE`] was two orders of
/// magnitude too large. Keep it: a captured graph cannot be re-pointed at a
/// different arena, so any future resizing is a re-capture driven by these
/// numbers, and a phase that starts using the arena shows up here first.
fn graph_report_level() -> u8 {
    match std::env::var("APXINF_QWEN38_GRAPH_REPORT").as_deref() {
        Ok("2") => 2,
        Ok("0") | Err(_) => 0,
        Ok(_) => 1,
    }
}

const MIB: f64 = (1024 * 1024) as f64;

fn report_phases(label: &str, phases: &[&PreparedPhase], memory_delta: Option<(usize, usize)>) {
    let level = graph_report_level();
    if level == 0 || phases.is_empty() {
        return;
    }
    let reserved: usize = phases.iter().map(|p| p.workspace_capacity()).sum();
    let used: usize = phases.iter().map(|p| p.workspace_used()).sum();
    let peak = phases.iter().map(|p| p.workspace_used()).max().unwrap_or(0);
    if level >= 2 {
        for (index, phase) in phases.iter().enumerate() {
            println!(
                "  {label} phase {index:>3}: used {:8.2} MiB / cap {:8.2} MiB",
                phase.workspace_used() as f64 / MIB,
                phase.workspace_capacity() as f64 / MIB,
            );
        }
    }
    println!(
        "  {label}: {} phases, reserved {:9.2} MiB, used {:8.2} MiB ({:.1}%), peak phase {:8.2} MiB",
        phases.len(),
        reserved as f64 / MIB,
        used as f64 / MIB,
        100.0 * used as f64 / reserved as f64,
        peak as f64 / MIB,
    );
    if let Some((before_free, after_free)) = memory_delta {
        // Unified memory on Thor: this is system memory, not discrete VRAM.
        // The delta is the signal; the absolute figure is not a VRAM budget.
        println!(
            "  {label}: device memory free {:9.2} -> {:9.2} MiB (capture consumed {:9.2} MiB)",
            before_free as f64 / MIB,
            after_free as f64 / MIB,
            before_free.saturating_sub(after_free) as f64 / MIB,
        );
    }
}

/// Free device memory, or `None` when the report is off (the query is cheap
/// but it is a device round trip, so it stays behind the same switch).
fn free_memory_if_reporting(ctx: &CudaContext) -> Option<usize> {
    (graph_report_level() > 0).then(|| ctx.memory_info().unwrap().0)
}

/// Same report for a bare session, which has no graph: the prefill arena is a
/// reservation of the same kind and belongs in the same accounting.
pub(crate) fn report_session(label: &str, session: &ops::ExecutionSession) {
    if graph_report_level() == 0 {
        return;
    }
    let workspace = session.workspace();
    println!(
        "  {label}: 1 session, reserved {:9.2} MiB, used {:8.2} MiB ({:.1}%)",
        workspace.capacity() as f64 / MIB,
        workspace.used() as f64 / MIB,
        100.0 * workspace.used() as f64 / workspace.capacity() as f64,
    );
}


/// Arena capacity per decode phase.
///
/// Measured, not guessed: `APXINF_QWEN38_GRAPH_REPORT=1` reports 0 bytes used
/// for every phase, and that is the true figure, not an artefact of when the
/// high-water mark is read. The arena's only consumer is
/// `workspace::output_buffer`, reached through `CudaContext::allocate_output`,
/// and nothing in this model calls it: every operator writes into a `Scratch`
/// tensor the caller already owns, and the native GEMM/Attention executions
/// bind views of those same tensors while allocating their own scratch inside
/// the native library, outside this arena.
///
/// So the earlier 64/128 MiB per phase was 7 GiB of `cudaMalloc` that never
/// held a byte. 4 MiB keeps a margin for a future operator that does take the
/// `allocate_output` path; exceeding it is a loud `static inference workspace
/// exhausted` error at capture, not silent corruption. Sized the same for both
/// layer kinds because neither uses any.
const PHASE_WORKSPACE: usize = 4 * 1024 * 1024;

/// One CUDA-graph phase per decoder layer, dense and in layer order.
///
/// A full-attention layer captures only its MLP: the mixer's KV view and FA2
/// split are sized by the live position, which a graph would freeze. A GDN
/// layer is position-independent -- history lives in the recurrent state -- so
/// its mixer and MLP go into a single graph.
///
/// This replaces two earlier sets, a 64-entry MLP set plus a 48-entry GDN set.
/// The GDN phase has always contained its own MLP, so the 48 MLP entries built
/// for GDN layers were never replayed; they cost 3 GiB of arena and 48
/// pointless prepare/capture passes.
pub(crate) fn capture_decode_phases(
    ctx: &CudaContext,
    model: &Model,
    scratch: &mut Scratch,
    states: &mut [GdnState],
) -> Vec<PreparedPhase> {
    let before_free = free_memory_if_reporting(ctx);
    let mut phases = Vec::with_capacity(model.layers.len());
    let mut gdn_index = 0usize;
    for layer in &model.layers {
        let phase = match layer {
            Layer::Attention(attention) => PreparedPhase::prepare_and_capture(
                ctx,
                ops::ExecutionSession::with_capacity(PHASE_WORKSPACE, ctx.device_id())
                    .unwrap(),
                || {
                    nvfp4_mlp(
                        ctx,
                        &attention.gate_up,
                        &attention.down,
                        &attention.post_norm,
                        scratch,
                    );
                    Ok(())
                },
            )
            .unwrap(),
            Layer::Gdn(gdn) => {
                let state_index = gdn_index;
                gdn_index += 1;
                // The prepare traversal runs this body for real against the
                // live state; see snapshot_gdn_state for why that has to be
                // undone.
                let saved = snapshot_gdn_state(ctx, &states[state_index]);
                let state = &mut states[state_index];
                let phase = PreparedPhase::prepare_and_capture(
                    ctx,
                    ops::ExecutionSession::with_capacity(PHASE_WORKSPACE, ctx.device_id())
                        .unwrap(),
                    || {
                        gdn_decode_layer(ctx, gdn, scratch, state);
                        nvfp4_mlp(ctx, &gdn.gate_up, &gdn.down, &gdn.post_norm, scratch);
                        Ok(())
                    },
                )
                .unwrap();
                restore_gdn_state(ctx, &states[state_index], &saved);
                phase
            }
        };
        phases.push(phase);
    }
    ctx.synchronize().unwrap();
    // Structural, not diagnostic: decode_step_inner indexes this by
    // layer_index, so a sparse or short vector is a dispatch bug.
    assert_eq!(phases.len(), model.layers.len());
    report_phases(
        "decode phases",
        &phases.iter().collect::<Vec<_>>(),
        before_free.map(|before| (before, ctx.memory_info().unwrap().0)),
    );
    phases
}

// Diagnostic: bit-compare a captured GDN layer graph against its eager
// replay. Unreferenced in the product path; kept for kernel debugging.
//
// Zeroes GDN state as it goes, so it may only run before any real work --
// the same hazard snapshot_gdn_state exists to contain.
#[allow(dead_code)]
fn verify_gdn_graphs(
    ctx: &CudaContext,
    model: &Model,
    scratch: &mut Scratch,
    states: &mut [GdnState],
    phases: &[PreparedPhase],
) {
    let mut state_index = 0usize;
    let mut checked = 0usize;
    for (layer_index, layer) in model.layers.iter().enumerate() {
        let Layer::Gdn(gdn) = layer else { continue };
        let state = &mut states[state_index];
        state_index += 1;
        let graph = &phases[layer_index];
        zero_tensor(&state.recurrent);
        zero_tensor(&state.conv_window);
        for step in 0..3 {
            let hidden = (0..HIDDEN)
                .flat_map(|index| {
                    half::bf16::from_f32(((index * 17 + step * 23) % 101) as f32 / 50.0 - 1.0)
                        .to_bits().to_le_bytes()
                })
                .collect::<Vec<_>>();
            CudaBuffer::from_tensor(&scratch.hidden).unwrap().copy_from_host(&hidden).unwrap();
            ctx.synchronize().unwrap();
            let initial_state = graph_tensor_bytes(&state.recurrent);
            let initial_conv = graph_tensor_bytes(&state.conv_window);
            gdn_decode_layer(ctx, gdn, scratch, state);
            nvfp4_mlp(ctx, &gdn.gate_up, &gdn.down, &gdn.post_norm, scratch);
            ctx.synchronize().unwrap();
            let expected_hidden = graph_tensor_bytes(&scratch.hidden);
            let expected_state = graph_tensor_bytes(&state.recurrent);
            let expected_conv = graph_tensor_bytes(&state.conv_window);
            let expected_readout = graph_tensor_bytes(&scratch.gdn_readout);
            assert!(expected_hidden.chunks_exact(2).all(|bytes| {
                half::bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).is_finite()
            }), "nonfinite hidden at layer {layer_index}, step {step}");
            assert!(expected_state.chunks_exact(4).all(|bytes| {
                f32::from_le_bytes(bytes.try_into().unwrap()).is_finite()
            }), "nonfinite state at layer {layer_index}, step {step}");
            CudaBuffer::from_tensor(&scratch.hidden).unwrap().copy_from_host(&hidden).unwrap();
            CudaBuffer::from_tensor(&state.recurrent).unwrap().copy_from_host(&initial_state).unwrap();
            CudaBuffer::from_tensor(&state.conv_window).unwrap().copy_from_host(&initial_conv).unwrap();
            graph.replay().unwrap();
            ctx.synchronize().unwrap();
            for (name, expected, tensor) in [
                ("hidden", &expected_hidden, &scratch.hidden),
                ("state", &expected_state, &state.recurrent),
                ("conv", &expected_conv, &state.conv_window),
                ("readout", &expected_readout, &scratch.gdn_readout),
            ] {
                assert!(expected == &graph_tensor_bytes(tensor),
                    "GDN graph mismatch: {name}, layer {layer_index}, step {step}");
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 48 * 3 * 4);
    println!("GDN graph verification: {checked} bit-exact tensor comparisons passed");
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_step_with_phases(
    ctx: &CudaContext,
    model: &Model,
    scratch: &mut Scratch,
    gdn_states: &mut [GdnState],
    kv_caches: &mut [KvCache],
    position: usize,
    phases: &[PreparedPhase],
) {
    decode_step_inner(
        ctx,
        model,
        scratch,
        gdn_states,
        kv_caches,
        position,
        Some(phases),
    );
}

/// Upload the prompt into the prefill token buffer.
pub(crate) fn write_tokens(ctx: &CudaContext, prefill: &PrefillScratch, ids: &[u32]) {
    let _ = ctx;
    let mut raw = Vec::with_capacity(ids.len() * 4);
    for &id in ids {
        raw.extend_from_slice(&(id as i32).to_le_bytes());
    }
    CudaBuffer::from_tensor(&prefill.tokens)
        .unwrap()
        .copy_from_host(&raw)
        .unwrap();
    // Positions are simply 0..tokens for a fresh prompt.
    let mut positions = Vec::with_capacity(ids.len() * 4);
    for index in 0..ids.len() {
        positions.extend_from_slice(&(index as i32).to_le_bytes());
    }
    CudaBuffer::from_tensor(&prefill.positions)
        .unwrap()
        .copy_from_host(&positions)
        .unwrap();
}

// --- host-side token/position IO and state reset (from the harness's test
// section; the model wrapper drives these) -------------------------------

pub(crate) fn set_token(scratch: &Scratch, id: i32) {
    CudaBuffer::from_tensor(&scratch.token)
        .unwrap()
        .copy_from_host(&id.to_le_bytes())
        .unwrap();
}

pub(crate) fn set_position(scratch: &Scratch, pos: usize) {
    CudaBuffer::from_tensor(&scratch.positions)
        .unwrap()
        .copy_from_host(&(pos as i32).to_le_bytes())
        .unwrap();
}

pub(crate) fn reset_state(ctx: &CudaContext, gdn_states: &mut [GdnState], kv_caches: &mut [KvCache]) {
    for s in gdn_states.iter() {
        zero_tensor(&s.recurrent);
        zero_tensor(&s.conv_window);
    }
    for c in kv_caches.iter() {
        zero_tensor(&c.keys);
        zero_tensor(&c.values);
    }
    ctx.synchronize().unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::weights;

    /// Fill a device tensor with a finite, layer-dependent pattern.
    fn fill_f32(tensor: &apxinf_core::Tensor, seed: usize) {
        let count = tensor.shape().numel();
        let mut raw = Vec::with_capacity(count * 4);
        for index in 0..count {
            // Bounded and non-zero: a GDN state of this magnitude is what a
            // few hundred decode steps produce.
            let value = (((index * 31 + seed * 17) % 2003) as f32 / 2003.0) - 0.5;
            raw.extend_from_slice(&value.to_le_bytes());
        }
        CudaBuffer::from_tensor(tensor).unwrap().copy_from_host(&raw).unwrap();
    }

    fn fill_bf16(tensor: &apxinf_core::Tensor, seed: usize) {
        let count = tensor.shape().numel();
        let mut raw = Vec::with_capacity(count * 2);
        for index in 0..count {
            let value = (((index * 17 + seed * 23) % 101) as f32 / 50.0) - 1.0;
            raw.extend_from_slice(&half::bf16::from_f32(value).to_bits().to_le_bytes());
        }
        CudaBuffer::from_tensor(tensor).unwrap().copy_from_host(&raw).unwrap();
    }

    /// Graph capture must leave GDN state exactly as it found it.
    ///
    /// `Qwen38::forward` calls `ensure_graphs()` immediately before
    /// `prefill_step` with no reset in between, and GDN prefill reads
    /// `state.recurrent` in place as its initial carry — so a write escaping
    /// the prepare traversal lands in the first prefill chunk. See
    /// [`snapshot_gdn_state`] for the full argument.
    ///
    /// The state here is deliberately NON-ZERO. With it at zero the recurrence
    /// is multiplicatively zero, the prepare traversal writes zeros over
    /// zeros, and this test passes whether or not the guard exists — which is
    /// exactly how the invariant stayed unasserted. Single-threaded because
    /// `apply_validated_kernel_defaults` writes the environment.
    #[test]
    #[ignore = "needs a CUDA device and APXINF_QWEN38_CHECKPOINT"]
    fn capture_preserves_gdn_state() {
        use std::path::PathBuf;

        let path: PathBuf = std::env::var_os("APXINF_QWEN38_CHECKPOINT")
            .expect("set APXINF_QWEN38_CHECKPOINT")
            .into();
        apply_validated_kernel_defaults();

        let ctx = CudaContext::new(0).unwrap();
        let tensors = weights::checkpoint(&path);
        let model = weights::load_model(&ctx, &tensors, true);
        ctx.synchronize().unwrap();
        drop(tensors);

        let mut scratch = Scratch::new(&ctx);
        let gdn_layers = LAYERS - LAYERS / FULL_ATTENTION_INTERVAL;
        let mut states: Vec<GdnState> = (0..gdn_layers).map(|_| GdnState::new(&ctx)).collect();

        for (index, state) in states.iter().enumerate() {
            fill_f32(&state.recurrent, index + 1);
            fill_f32(&state.conv_window, index + 7);
        }
        fill_bf16(&scratch.hidden, 3);
        ctx.synchronize().unwrap();

        let expected: Vec<(Vec<u8>, Vec<u8>)> = states
            .iter()
            .map(|s| {
                (
                    graph_tensor_bytes(&s.recurrent),
                    graph_tensor_bytes(&s.conv_window),
                )
            })
            .collect();

        let phases = capture_decode_phases(&ctx, &model, &mut scratch, &mut states);
        // One phase per layer, including the full-attention ones.
        assert_eq!(phases.len(), LAYERS);
        ctx.synchronize().unwrap();

        for (index, (state, (recurrent, conv_window))) in
            states.iter().zip(expected.iter()).enumerate()
        {
            assert!(
                &graph_tensor_bytes(&state.recurrent) == recurrent,
                "capture mutated GDN{index} recurrent state"
            );
            assert!(
                &graph_tensor_bytes(&state.conv_window) == conv_window,
                "capture mutated GDN{index} conv window"
            );
        }
    }
}
