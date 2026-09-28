//! Execution ownership: CUDA-graph capture, replay-driven decode and
//! session state reset.
//!
//! Calls the model computation in `super::model`; does not re-implement it.


use apxinf_cuda_new::{capture, ops, CapturedGraph, CudaBuffer, CudaContext};

use super::backend::{graph_tensor_bytes, zero_tensor};
use super::config::*;
use super::model::{
    decode_step_inner, gdn_decode_layer, nvfp4_mlp, GdnState, KvCache, PrefillScratch, Scratch,
};
use super::weights::{Layer, Model};


pub(crate) fn capture_gdn_graphs(
    ctx: &CudaContext,
    model: &Model,
    scratch: &mut Scratch,
    states: &mut [GdnState],
) -> Vec<Option<CapturedGraph>> {
    let mut graphs = (0..model.layers.len()).map(|_| None).collect::<Vec<_>>();
    let mut gdn_index = 0usize;
    for (layer_index, layer) in model.layers.iter().enumerate() {
        let Layer::Gdn(gdn) = layer else { continue };
        let state = &mut states[gdn_index];
        gdn_index += 1;
        let session = ops::ExecutionSession::with_capacity(128 * 1024 * 1024, ctx.device_id()).unwrap();
        ops::prepare_with_session(&session, || {
            gdn_decode_layer(ctx, gdn, scratch, state);
            nvfp4_mlp(ctx, &gdn.gate_up, &gdn.down, &gdn.post_norm, scratch);
            Ok(())
        }).unwrap();
        ctx.synchronize().unwrap();
        let graph = ops::with_session(&session, || {
            capture(ctx, || {
                gdn_decode_layer(ctx, gdn, scratch, state);
                nvfp4_mlp(ctx, &gdn.gate_up, &gdn.down, &gdn.post_norm, scratch);
                Ok(())
            })
        }).unwrap();
        graphs[layer_index] = Some(graph);
    }
    ctx.synchronize().unwrap();
    graphs
}

// Diagnostic: bit-compare a captured GDN layer graph against its eager
// replay. Unreferenced in the product path; kept for kernel debugging.
#[allow(dead_code)]
fn verify_gdn_graphs(
    ctx: &CudaContext,
    model: &Model,
    scratch: &mut Scratch,
    states: &mut [GdnState],
    graphs: &[Option<CapturedGraph>],
) {
    let mut state_index = 0usize;
    let mut checked = 0usize;
    for (layer_index, layer) in model.layers.iter().enumerate() {
        let Layer::Gdn(gdn) = layer else { continue };
        let state = &mut states[state_index];
        state_index += 1;
        let graph = graphs[layer_index].as_ref().expect("missing GDN graph");
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
pub(crate) fn decode_step_with_mlp_graphs(
    ctx: &CudaContext,
    model: &Model,
    scratch: &mut Scratch,
    gdn_states: &mut [GdnState],
    kv_caches: &mut [KvCache],
    position: usize,
    mlp_graphs: &[CapturedGraph],
    gdn_graphs: Option<&[Option<CapturedGraph>]>,
) {
    decode_step_inner(
        ctx,
        model,
        scratch,
        gdn_states,
        kv_caches,
        position,
        Some(mlp_graphs),
        gdn_graphs,
    );
}

pub(crate) fn capture_mlp_graphs(
    ctx: &CudaContext,
    model: &Model,
    scratch: &mut Scratch,
) -> Vec<CapturedGraph> {
    let mut graphs = Vec::with_capacity(model.layers.len());
    for layer in &model.layers {
        let (gate_up, down, norm_weight) = match layer {
            Layer::Attention(layer) => (&layer.gate_up, &layer.down, &layer.post_norm),
            Layer::Gdn(layer) => (&layer.gate_up, &layer.down, &layer.post_norm),
        };
        let session = ops::ExecutionSession::with_capacity(64 * 1024 * 1024, ctx.device_id())
            .unwrap();
        ops::prepare_with_session(&session, || {
            nvfp4_mlp(ctx, gate_up, down, norm_weight, scratch);
            Ok(())
        })
        .unwrap();
        ctx.synchronize().unwrap();
        let graph = ops::with_session(&session, || {
            capture(ctx, || {
                nvfp4_mlp(ctx, gate_up, down, norm_weight, scratch);
                Ok(())
            })
        })
        .unwrap();
        graphs.push(graph);
    }
    ctx.synchronize().unwrap();
    graphs
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
