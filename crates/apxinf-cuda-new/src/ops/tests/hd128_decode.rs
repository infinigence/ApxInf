//! hd128 single-token decode fast-path precision and dispatch regressions.
//!
//! The custom KV-cache provider dispatches `single_token_attention_hd128` for
//! batch-1 single-token Flexible decode on 128-channel heads (capacity
//! <= 1024, 16-byte aligned bindings). That kernel reduces the dot product,
//! the softmax sums and the value accumulation as trees rather than the
//! sequential reference scan, so this module is its precision gate: it
//! compares the candidate against `kv_cache_attention_reference` at the family
//! tolerance (`0.03 + 0.02 * |expected|`), sweeps the key-length and warp-slice
//! boundaries of the dispatch window, and covers the shapes and bindings that
//! must stay on the reference path.

use super::framework::{f16_tensor, tensor, zeros_tensor};
use super::l3_behavior::kv_cache_attention_reference;
use super::*;
use crate::CudaContext;
use apxinf_core::{DType, Shape};
use half::f16;

const HEAD_DIM: usize = 128;
const QUERY_HEADS: usize = 16;
const TOLERANCE_ABSOLUTE: f32 = 0.03;
const TOLERANCE_RELATIVE: f32 = 0.02;

fn scale() -> f32 {
    1.0 / (HEAD_DIM as f32).sqrt()
}

fn generated(count: usize, multiplier: usize) -> Vec<f32> {
    (0..count)
        .map(|index| ((index * multiplier % 17) as f32 - 8.0) / 16.0)
        .collect()
}

fn lengths() -> Vec<usize> {
    let mut lengths: Vec<usize> = (1..=40).collect();
    lengths.extend([
        47, 48, 49, 63, 64, 65, 95, 96, 97, 127, 128, 129, 191, 255, 256, 257, 383, 511, 512, 513,
        767, 1023, 1024,
    ]);
    lengths
}

fn f16_output_values(output: &apxinf_core::Tensor) -> Vec<f32> {
    assert_eq!(output.dtype(), DType::F16);
    let buffer = crate::CudaBuffer::from_tensor(output).unwrap();
    let mut bytes = vec![0; buffer.len()];
    buffer.copy_to_host(&mut bytes).unwrap();
    bytes
        .chunks_exact(2)
        .map(|chunk| f16::from_bits(u16::from_ne_bytes([chunk[0], chunk[1]])).to_f32())
        .collect()
}

fn assert_matches_reference(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    let (mut worst_delta, mut worst_index, mut worst_limit) = (0.0f32, 0usize, 0.0f32);
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let delta = (actual - expected).abs();
        let limit = TOLERANCE_ABSOLUTE + TOLERANCE_RELATIVE * expected.abs();
        if delta > worst_delta {
            (worst_delta, worst_index, worst_limit) = (delta, index, limit);
        }
        assert!(
            delta < limit,
            "{label} output[{index}] is {actual}, expected {expected} (delta {delta} > {limit})"
        );
    }
    eprintln!(
        "{label}: max |delta| {worst_delta:.3e} at [{worst_index}] (limit {worst_limit:.3e})"
    );
}

/// One batch-1 single-token decode shape with deterministic inputs.
struct DecodeCase {
    query: apxinf_core::Tensor,
    key: apxinf_core::Tensor,
    value: apxinf_core::Tensor,
    query_values: Vec<f32>,
    key_values: Vec<f32>,
    value_values: Vec<f32>,
    kv_heads: usize,
    capacity: usize,
}

fn decode_case(kv_heads: usize, capacity: usize, bf16: bool) -> DecodeCase {
    let query_values = generated(QUERY_HEADS * HEAD_DIM, 3);
    let key_values = generated(capacity * kv_heads * HEAD_DIM, 5);
    let value_values = generated(capacity * kv_heads * HEAD_DIM, 7);
    let query_shape = vec![1usize, 1, QUERY_HEADS, HEAD_DIM];
    let cache_shape = vec![1usize, capacity, kv_heads, HEAD_DIM];
    let (query, key, value) = if bf16 {
        (
            tensor(0, query_shape, &query_values),
            tensor(0, cache_shape.clone(), &key_values),
            tensor(0, cache_shape, &value_values),
        )
    } else {
        (
            f16_tensor(0, query_shape, &query_values),
            f16_tensor(0, cache_shape.clone(), &key_values),
            f16_tensor(0, cache_shape, &value_values),
        )
    };
    DecodeCase {
        query,
        key,
        value,
        query_values,
        key_values,
        value_values,
        kv_heads,
        capacity,
    }
}

impl DecodeCase {
    fn expected(&self, valid: usize, query_start: usize) -> Vec<f32> {
        kv_cache_attention_reference(
            &self.query_values,
            &self.key_values,
            &self.value_values,
            1,
            1,
            valid,
            self.capacity,
            QUERY_HEADS,
            self.kv_heads,
            HEAD_DIM,
            query_start,
            scale(),
            true,
        )
    }
}

fn decode_args<'a>(
    case: &'a DecodeCase,
    decode_meta: &'a KvCacheDecodeMeta,
    out: &'a mut apxinf_core::Tensor,
    valid: usize,
    allow_fallback: bool,
) -> KvCacheAttentionArgs<'a> {
    let mut args = KvCacheAttentionArgs::new(&case.query, &case.key, &case.value, out)
        .with_decode_meta(decode_meta);
    args.valid_key_tokens = valid;
    args.query_start = valid - 1;
    args.policy.online_tune = false;
    args.policy.allow_fallback = allow_fallback;
    args.policy.cache_dir = Some(String::new());
    args
}

#[test]
fn hd128_decode_custom_candidate_matches_reference_across_lengths_and_gqa() {
    let ctx = CudaContext::new(0).unwrap();
    const CAPACITY: usize = 1024;
    for kv_heads in [2usize, 4, 8] {
        let case = decode_case(kv_heads, CAPACITY, false);
        let decode_meta = KvCacheDecodeMeta::new(0, CAPACITY).unwrap();
        for valid in lengths() {
            let mut out = zeros_tensor(0, vec![1, 1, QUERY_HEADS, HEAD_DIM], DType::F16);
            let args = decode_args(&case, &decode_meta, &mut out, valid, false);
            let normalized = super::normalize_kv_cache_attention(&ctx, args).unwrap();
            super::attention_execution::validate_candidates(
                &ctx,
                &normalized,
                &case.expected(valid, valid - 1),
            )
            .unwrap();
        }
    }
}

#[test]
fn hd128_decode_matches_reference_for_f16_and_bf16() {
    let ctx = CudaContext::new(0).unwrap();
    for (kv_heads, bf16) in [(8usize, false), (8usize, true)] {
        let case = decode_case(kv_heads, 512, bf16);
        let decode_meta = KvCacheDecodeMeta::new(0, 512).unwrap();
        let valid = 257;
        let dtype = if bf16 { DType::BF16 } else { DType::F16 };
        let mut out = zeros_tensor(0, vec![1, 1, QUERY_HEADS, HEAD_DIM], dtype);
        let args = decode_args(&case, &decode_meta, &mut out, valid, false);
        let normalized = super::normalize_kv_cache_attention(&ctx, args).unwrap();
        super::attention_execution::validate_candidates(
            &ctx,
            &normalized,
            &case.expected(valid, valid - 1),
        )
        .unwrap();
    }
}

#[test]
fn hd128_decode_beyond_the_dispatch_limit_keeps_the_reference_path() {
    let ctx = CudaContext::new(0).unwrap();
    // The kernel keeps its scores in a fixed 1024-entry shared array, so a
    // dispatch that ignored the capacity window would write past it. The
    // reference path must serve the longer cache instead.
    const CAPACITY: usize = 4096;
    const KV_HEADS: usize = 8;
    let case = decode_case(KV_HEADS, CAPACITY, false);
    let decode_meta = KvCacheDecodeMeta::new(0, CAPACITY).unwrap();
    let mut out = zeros_tensor(0, vec![1, 1, QUERY_HEADS, HEAD_DIM], DType::F16);
    let args = decode_args(&case, &decode_meta, &mut out, CAPACITY, true);
    let normalized = super::normalize_kv_cache_attention(&ctx, args).unwrap();
    let execution = super::attention_execution::prepare(&ctx, normalized).unwrap();
    assert!(
        execution.summary().starts_with("custom-kv-cache-attention"),
        "unexpected hd128 over-limit provider: {}",
        execution.summary()
    );
    execution.enqueue_for_test().unwrap();
    ctx.synchronize().unwrap();
    assert_matches_reference(
        &f16_output_values(&out),
        &case.expected(CAPACITY, CAPACITY - 1),
        "hd128 over-limit decode",
    );
}

#[test]
fn hd128_decode_misaligned_bindings_keep_the_reference_contract() {
    let ctx = CudaContext::new(0).unwrap();
    const CAPACITY: usize = 1024;
    const KV_HEADS: usize = 8;
    const VALID: usize = 96;

    // Two-byte-offset views: valid F16 bindings, but not 8-byte aligned. The
    // fast path reads q/k/v rows as int2 vectors, so dispatching it on these
    // bindings would fault; the launch guard must keep the reference path.
    let misaligned = |values: &[f32], shape: Vec<usize>| -> apxinf_core::Tensor {
        let mut bytes = vec![0u8, 0u8];
        for value in values {
            bytes.extend_from_slice(&f16::from_f32(*value).to_bits().to_ne_bytes());
        }
        let payload = bytes.len() - 2;
        let backing = crate::CudaBuffer::alloc(bytes.len(), 0).unwrap();
        backing.copy_from_host(&bytes).unwrap();
        backing
            .view(2, payload)
            .unwrap()
            .as_tensor(Shape::new(shape), DType::F16)
            .unwrap()
    };
    let case = decode_case(KV_HEADS, CAPACITY, false);
    let query = misaligned(&case.query_values, vec![1, 1, QUERY_HEADS, HEAD_DIM]);
    let key = misaligned(&case.key_values, vec![1, CAPACITY, KV_HEADS, HEAD_DIM]);
    let value = misaligned(&case.value_values, vec![1, CAPACITY, KV_HEADS, HEAD_DIM]);
    let mut out = zeros_tensor(0, vec![1, 1, QUERY_HEADS, HEAD_DIM], DType::F16);
    let decode_meta = KvCacheDecodeMeta::new(0, CAPACITY).unwrap();
    let mut args =
        KvCacheAttentionArgs::new(&query, &key, &value, &mut out).with_decode_meta(&decode_meta);
    args.valid_key_tokens = VALID;
    args.query_start = VALID - 1;
    args.policy.online_tune = false;
    args.policy.allow_fallback = true;
    args.policy.cache_dir = Some(String::new());
    let normalized = super::normalize_kv_cache_attention(&ctx, args).unwrap();
    assert_eq!(normalized.spec.q_alignment, 2);
    assert_eq!(normalized.spec.k_alignment, 2);
    assert_eq!(normalized.spec.v_alignment, 2);
    let execution = super::attention_execution::prepare(&ctx, normalized).unwrap();
    assert!(
        execution.summary().starts_with("custom-kv-cache-attention"),
        "misaligned hd128 decode lost its only dynamic-decode candidate: {}",
        execution.summary()
    );
    execution.enqueue_for_test().unwrap();
    ctx.synchronize().unwrap();
    assert_matches_reference(
        &f16_output_values(&out),
        &case.expected(VALID, VALID - 1),
        "hd128 misaligned decode",
    );
}

#[test]
fn hd128_decode_graph_replay_sweeps_valid_lengths() {
    let ctx = CudaContext::new(0).unwrap();
    const CAPACITY: usize = 256;
    const KV_HEADS: usize = 8;
    let case = decode_case(KV_HEADS, CAPACITY, false);
    let decode_meta = KvCacheDecodeMeta::new(0, CAPACITY).unwrap();

    // The dynamic-decode candidate for this spec is the custom provider, so
    // the captured graph below replays the hd128 fast path.
    {
        let mut out = zeros_tensor(0, vec![1, 1, QUERY_HEADS, HEAD_DIM], DType::F16);
        let args = decode_args(&case, &decode_meta, &mut out, 1, true);
        let normalized = super::normalize_kv_cache_attention(&ctx, args).unwrap();
        let execution = super::attention_execution::prepare(&ctx, normalized).unwrap();
        assert!(
            execution.summary().starts_with("custom-kv-cache-attention"),
            "unexpected hd128 decode provider: {}",
            execution.summary()
        );
    }

    let mut out = zeros_tensor(0, vec![1, 1, QUERY_HEADS, HEAD_DIM], DType::F16);
    let session = ExecutionSession::with_capacity(4096, 0).unwrap();
    prepare_with_session(&session, || {
        let args = decode_args(&case, &decode_meta, &mut out, 1, true);
        kv_cache_attention(&ctx, args)
    })
    .unwrap();
    let graph = crate::capture(&ctx, || {
        with_session(&session, || {
            let args = decode_args(&case, &decode_meta, &mut out, 1, true);
            kv_cache_attention(&ctx, args)
        })
    })
    .unwrap();

    for valid in 1..=CAPACITY {
        // Updating after synchronization is required because replay is async.
        decode_meta.update(valid, valid - 1).unwrap();
        graph.replay().unwrap();
        ctx.synchronize().unwrap();
        assert_matches_reference(
            &f16_output_values(&out),
            &case.expected(valid, valid - 1),
            &format!("hd128 graph step-{valid}"),
        );
    }
}
