//! Per-operator L3 semantic and conditional Graph behavior tests.
//!
//! Every new L3 operator must add a public semantic-contract test here. Add a
//! Graph replay test only when the operator has an independent execution path,
//! resource lifetime, binding rule, or capture behavior not already covered by
//! the shared framework tests.

use super::framework::{tensor, values};
use super::*;
use crate::CudaContext;
use half::bf16;

fn u32_buffer(device: usize, values: &[u32]) -> crate::CudaBuffer {
    let bytes: Vec<_> = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    let buffer = crate::CudaBuffer::alloc(bytes.len(), device).unwrap();
    buffer.copy_from_host(&bytes).unwrap();
    buffer
}

#[allow(clippy::too_many_arguments)]
pub(super) fn attention_reference(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    batch: usize,
    query_tokens: usize,
    key_tokens: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    scale: f32,
    causal: bool,
) -> Vec<f32> {
    let mut output = vec![0.0; batch * query_tokens * query_heads * head_dim];
    for batch_index in 0..batch {
        for query_token in 0..query_tokens {
            for query_head in 0..query_heads {
                let kv_head = query_head / (query_heads / kv_heads);
                let valid_keys = if causal {
                    (query_token + key_tokens - query_tokens + 1).min(key_tokens)
                } else {
                    key_tokens
                };
                let mut scores = Vec::with_capacity(valid_keys);
                for key_token in 0..valid_keys {
                    let mut score = 0.0;
                    for dimension in 0..head_dim {
                        let q = (((batch_index * query_tokens + query_token) * query_heads
                            + query_head)
                            * head_dim)
                            + dimension;
                        let k = (((batch_index * key_tokens + key_token) * kv_heads + kv_head)
                            * head_dim)
                            + dimension;
                        score += query[q] * key[k];
                    }
                    scores.push(score * scale);
                }
                let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let denominator: f32 = scores.iter().map(|score| (score - maximum).exp()).sum();
                for dimension in 0..head_dim {
                    let result = scores
                        .iter()
                        .enumerate()
                        .map(|(key_token, score)| {
                            let v = (((batch_index * key_tokens + key_token) * kv_heads + kv_head)
                                * head_dim)
                                + dimension;
                            ((*score - maximum).exp() / denominator) * value[v]
                        })
                        .sum();
                    let output_index = (((batch_index * query_tokens + query_token) * query_heads
                        + query_head)
                        * head_dim)
                        + dimension;
                    output[output_index] = result;
                }
            }
        }
    }
    output
}

#[allow(clippy::too_many_arguments)]
pub(super) fn kv_cache_attention_reference(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    batch: usize,
    query_tokens: usize,
    valid_key_tokens: usize,
    key_capacity: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    query_start: usize,
    scale: f32,
    causal: bool,
) -> Vec<f32> {
    let mut output = vec![0.0; batch * query_tokens * query_heads * head_dim];
    for batch_index in 0..batch {
        for query_token in 0..query_tokens {
            for query_head in 0..query_heads {
                let kv_head = query_head / (query_heads / kv_heads);
                let keys = if causal {
                    (query_start + query_token + 1).min(valid_key_tokens)
                } else {
                    valid_key_tokens
                };
                let mut scores = Vec::with_capacity(keys);
                for key_token in 0..keys {
                    let mut score = 0.0;
                    for dimension in 0..head_dim {
                        let q = (((batch_index * query_tokens + query_token) * query_heads
                            + query_head)
                            * head_dim)
                            + dimension;
                        let k = (((batch_index * key_capacity + key_token) * kv_heads + kv_head)
                            * head_dim)
                            + dimension;
                        score += query[q] * key[k];
                    }
                    scores.push(score * scale);
                }
                let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let denominator: f32 = scores.iter().map(|score| (score - maximum).exp()).sum();
                for dimension in 0..head_dim {
                    let result = scores
                        .iter()
                        .enumerate()
                        .map(|(key_token, score)| {
                            let v = (((batch_index * key_capacity + key_token) * kv_heads
                                + kv_head)
                                * head_dim)
                                + dimension;
                            ((*score - maximum).exp() / denominator) * value[v]
                        })
                        .sum();
                    let output_index = (((batch_index * query_tokens + query_token) * query_heads
                        + query_head)
                        * head_dim)
                        + dimension;
                    output[output_index] = result;
                }
            }
        }
    }
    output
}

#[test]
fn gemm_has_its_own_semantic_api() {
    let ctx = CudaContext::new(0).unwrap();
    let a = tensor(0, vec![2, 3], &[1.0; 6]);
    let b = tensor(0, vec![3, 4], &[1.0; 12]);
    let mut out = tensor(0, vec![2, 4], &[0.0; 8]);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    args.policy.online_tune = false;
    gemm(&ctx, args).unwrap();
    assert!(values(&out).iter().all(|&value| value == 3.0));
}

#[test]
fn gemm_geglu_is_a_separate_semantic_domain() {
    let ctx = CudaContext::new(0).unwrap();
    let (m, k, n) = (3, 5, 1024);
    let a_values = vec![0.25; m * k];
    let b_values: Vec<_> = (0..k * n).map(|i| ((i % 11) as f32 - 5.0) / 16.0).collect();
    let a = tensor(0, vec![m, k], &a_values);
    let b = tensor(0, vec![k, n], &b_values);
    let mut out = tensor(0, vec![m, n / 2], &vec![0.0; m * n / 2]);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    args.policy.online_tune = false;
    gemm_geglu(&ctx, GemmGegluArgs { gemm: args }).unwrap();
    let actual = values(&out);
    assert_eq!(actual.len(), m * n / 2);
    for row in 0..m {
        for column in 0..n / 2 {
            let dot = |target| {
                bf16::from_f32(
                    (0..k)
                        .map(|inner| a_values[row * k + inner] * b_values[inner * n + target])
                        .sum(),
                )
                .to_f32()
            };
            let gate = dot(column);
            let expected = 0.5
                * gate
                * (1.0 + (0.79788456 * (gate + 0.044715 * gate.powi(3))).tanh())
                * dot(column + n / 2);
            assert!((actual[row * n / 2 + column] - expected).abs() < 0.01);
        }
    }
}

#[test]
fn gemm_bias_gelu_has_its_own_semantic_api() {
    let ctx = CudaContext::new(0).unwrap();
    let a = tensor(0, vec![2, 3], &[1.0; 6]);
    let b = tensor(0, vec![3, 4], &[1.0; 12]);
    let bias = tensor(0, vec![4], &[0.5; 4]);
    let mut out = tensor(0, vec![2, 4], &[0.0; 8]);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    args.policy.online_tune = false;
    gemm_bias_gelu(
        &ctx,
        GemmBiasGeluArgs {
            gemm: args,
            bias: &bias,
        },
    )
    .unwrap();
    let output = values(&out);
    let x: f32 = 3.5;
    let expected = 0.5 * x * (1.0 + (0.79788456 * (x + 0.044715 * x.powi(3))).tanh());
    assert!(output.iter().all(|value| (*value - expected).abs() < 0.02));
}

#[test]
fn gemm_bias_has_its_own_semantic_api() {
    let ctx = CudaContext::new(0).unwrap();
    let a = tensor(0, vec![2, 3], &[1.0; 6]);
    let b = tensor(0, vec![3, 4], &[1.0; 12]);
    let bias = tensor(0, vec![4], &[0.5, -0.5, 1.0, -1.0]);
    let mut out = tensor(0, vec![2, 4], &[0.0; 8]);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    args.policy.online_tune = false;
    gemm_bias(
        &ctx,
        GemmBiasArgs {
            gemm: args,
            bias: &bias,
        },
    )
    .unwrap();
    let expected = [3.5, 2.5, 4.0, 2.0, 3.5, 2.5, 4.0, 2.0];
    for (actual, expected) in values(&out).iter().zip(expected) {
        assert!((*actual - expected).abs() < 0.01);
    }
}

#[test]
fn attention_supports_mha_gqa_mqa_and_causal_masking() {
    let ctx = CudaContext::new(0).unwrap();
    let (batch, query_tokens, key_tokens, query_heads, head_dim) = (1, 2, 3, 4, 2);
    let query_values: Vec<_> = (0..batch * query_tokens * query_heads * head_dim)
        .map(|index| (index as f32 - 7.0) / 8.0)
        .collect();
    let scale = 0.5;

    for (kv_heads, causal) in [(4, false), (2, false), (1, false), (2, true)] {
        let key_values: Vec<_> = (0..batch * key_tokens * kv_heads * head_dim)
            .map(|index| ((index * 3 % 11) as f32 - 5.0) / 8.0)
            .collect();
        let value_values: Vec<_> = (0..batch * key_tokens * kv_heads * head_dim)
            .map(|index| ((index * 5 % 13) as f32 - 6.0) / 8.0)
            .collect();
        let query = tensor(
            0,
            vec![batch, query_tokens, query_heads, head_dim],
            &query_values,
        );
        let key = tensor(0, vec![batch, key_tokens, kv_heads, head_dim], &key_values);
        let value = tensor(
            0,
            vec![batch, key_tokens, kv_heads, head_dim],
            &value_values,
        );
        let mut out = tensor(
            0,
            vec![batch, query_tokens, query_heads, head_dim],
            &vec![0.0; query_values.len()],
        );
        let mut args = AttentionArgs::new(&query, &key, &value, &mut out);
        args.scale = scale;
        args.mask = if causal {
            AttentionMask::Causal
        } else {
            AttentionMask::None
        };
        args.policy.online_tune = false;
        attention(&ctx, args).unwrap();

        let expected = attention_reference(
            &query_values,
            &key_values,
            &value_values,
            batch,
            query_tokens,
            key_tokens,
            query_heads,
            kv_heads,
            head_dim,
            scale,
            causal,
        );
        for (index, (actual, expected)) in values(&out).iter().zip(expected).enumerate() {
            assert!(
                (*actual - expected).abs() < 0.02,
                "attention element {index} for kv_heads={kv_heads}, causal={causal} is {actual}, expected {expected}"
            );
        }
    }
}

#[test]
fn attention_prepares_before_capture_and_replays_from_cache() {
    let ctx = CudaContext::new(0).unwrap();
    let query = tensor(0, vec![1, 2, 2, 2], &[0.25; 8]);
    let key = tensor(0, vec![1, 3, 1, 2], &[0.5; 6]);
    let value = tensor(0, vec![1, 3, 1, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let mut out = tensor(0, vec![1, 2, 2, 2], &[0.0; 8]);
    let session = ExecutionSession::with_capacity(4096, 0).unwrap();

    prepare_with_session(&session, || {
        let mut prepare = AttentionArgs::new(&query, &key, &value, &mut out);
        prepare.policy.online_tune = false;
        attention(&ctx, prepare)
    })
    .unwrap();
    let expected = values(&out);

    let graph = crate::capture(&ctx, || {
        with_session(&session, || {
            let mut captured = AttentionArgs::new(&query, &key, &value, &mut out);
            captured.policy.online_tune = false;
            attention(&ctx, captured)
        })
    })
    .unwrap();
    let output = crate::CudaBuffer::from_tensor(&out).unwrap();
    let sentinel: Vec<_> = (0..8)
        .flat_map(|_| bf16::from_f32(-123.0).to_bits().to_ne_bytes())
        .collect();
    output.copy_from_host(&sentinel).unwrap();
    graph.replay().unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(values(&out), expected);
}

#[test]
fn kv_cache_attention_uses_trailing_causal_positions() {
    let ctx = CudaContext::new(0).unwrap();
    let query_values = vec![0.5, -0.25, 0.25, 0.75];
    let key_values = vec![0.25, 0.5, -0.5, 0.25, 0.75, -0.25];
    let value_values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let query = tensor(0, vec![1, 2, 1, 2], &query_values);
    let key = tensor(0, vec![1, 3, 1, 2], &key_values);
    let value = tensor(0, vec![1, 3, 1, 2], &value_values);
    let mut out = tensor(0, vec![1, 2, 1, 2], &[0.0; 4]);
    let mut args = KvCacheAttentionArgs::new(&query, &key, &value, &mut out);
    args.policy.online_tune = false;
    kv_cache_attention(&ctx, args).unwrap();

    let expected = attention_reference(
        &query_values,
        &key_values,
        &value_values,
        1,
        2,
        3,
        1,
        1,
        2,
        1.0 / 2.0_f32.sqrt(),
        true,
    );
    for (actual, expected) in values(&out).iter().zip(expected) {
        assert!((*actual - expected).abs() < 0.02);
    }
}

#[test]
fn kv_cache_attention_honors_capacity_valid_length_and_query_position() {
    let ctx = CudaContext::new(0).unwrap();
    let query_values = vec![0.5, -0.25, 0.25, 0.75, -0.5, 0.5, 0.75, 0.25];
    let key_values = vec![
        0.25, 0.5, -0.5, 0.25, 0.75, -0.25, 0.5, 0.5, 99.0, 99.0, -0.25, 0.75, 0.5, -0.5, 0.25,
        0.25, -0.75, 0.5, 88.0, 88.0,
    ];
    let value_values = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 999.0, 999.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7,
        1.8, 888.0, 888.0,
    ];
    let query = tensor(0, vec![2, 2, 1, 2], &query_values);
    let key = tensor(0, vec![2, 5, 1, 2], &key_values);
    let value = tensor(0, vec![2, 5, 1, 2], &value_values);
    let mut out = tensor(0, vec![2, 2, 1, 2], &[0.0; 8]);
    let mut args = KvCacheAttentionArgs::new(&query, &key, &value, &mut out);
    args.valid_key_tokens = 4;
    args.query_start = 1;
    args.policy.online_tune = false;
    kv_cache_attention(&ctx, args).unwrap();

    let expected = kv_cache_attention_reference(
        &query_values,
        &key_values,
        &value_values,
        2,
        2,
        4,
        5,
        1,
        1,
        2,
        1,
        1.0 / 2.0_f32.sqrt(),
        true,
    );
    for (index, (actual, expected)) in values(&out).iter().zip(expected).enumerate() {
        assert!(
            (*actual - expected).abs() < 0.02,
            "KV-cache capacity result {index} is {actual}, expected {expected}"
        );
    }
}

#[test]
fn kv_cache_graph_replays_all_positions_with_one_capacity_spec() {
    let ctx = CudaContext::new(0).unwrap();
    let head_dim = 64;
    let key_capacity = 64;
    let query_values: Vec<_> = (0..head_dim)
        .map(|index| ((index * 3 % 17) as f32 - 8.0) / 16.0)
        .collect();
    let key_values: Vec<_> = (0..key_capacity * head_dim)
        .map(|index| ((index * 5 % 19) as f32 - 9.0) / 16.0)
        .collect();
    let value_values: Vec<_> = (0..key_capacity * head_dim)
        .map(|index| ((index * 7 % 23) as f32 - 11.0) / 16.0)
        .collect();
    let query = tensor(0, vec![1, 1, 1, head_dim], &query_values);
    let key = tensor(0, vec![1, key_capacity, 1, head_dim], &key_values);
    let value = tensor(0, vec![1, key_capacity, 1, head_dim], &value_values);
    let mut out = tensor(0, vec![1, 1, 1, head_dim], &vec![0.0; head_dim]);
    let decode_meta = KvCacheDecodeMeta::new(0, key_capacity).unwrap();
    let session = ExecutionSession::with_capacity(4096, 0).unwrap();

    prepare_with_session(&session, || {
        let mut args = KvCacheAttentionArgs::new(&query, &key, &value, &mut out)
            .with_decode_meta(&decode_meta);
        args.valid_key_tokens = 1;
        args.query_start = 0;
        args.policy.online_tune = false;
        kv_cache_attention(&ctx, args)
    })
    .unwrap();

    let graph = crate::capture(&ctx, || {
        with_session(&session, || {
            let mut args = KvCacheAttentionArgs::new(&query, &key, &value, &mut out)
                .with_decode_meta(&decode_meta);
            args.valid_key_tokens = 1;
            args.query_start = 0;
            args.policy.online_tune = false;
            kv_cache_attention(&ctx, args)
        })
    })
    .unwrap();

    // This crosses every old power-of-two bucket boundary, including 7/8/9,
    // and reaches valid_len == key_capacity without preparing or capturing
    // another execution.
    for valid_key_tokens in 1..=key_capacity {
        // Updating after synchronization is required because replay is async.
        decode_meta
            .update(valid_key_tokens, valid_key_tokens - 1)
            .unwrap();
        graph.replay().unwrap();
        ctx.synchronize().unwrap();
        let expected = kv_cache_attention_reference(
            &query_values,
            &key_values,
            &value_values,
            1,
            1,
            valid_key_tokens,
            key_capacity,
            1,
            1,
            head_dim,
            valid_key_tokens - 1,
            1.0 / (head_dim as f32).sqrt(),
            true,
        );
        let actual = values(&out);
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (*actual - expected).abs() < 0.03 + 0.02 * expected.abs(),
                "step-{valid_key_tokens} graph output {index} is {actual}, expected {expected}"
            );
        }
    }
}

#[test]
fn kv_cache_decode_meta_enforces_capacity() {
    let decode_meta = KvCacheDecodeMeta::new(0, 9).unwrap();
    decode_meta.update(9, 8).unwrap();
    assert!(decode_meta.update(10, 9).is_err());
}

#[test]
fn kv_cache_decode_meta_rejects_non_decode_queries() {
    let ctx = CudaContext::new(0).unwrap();
    let key = tensor(0, vec![1, 4, 1, 2], &[0.25; 8]);
    let value = tensor(0, vec![1, 4, 1, 2], &[0.5; 8]);
    let decode_meta = KvCacheDecodeMeta::new(0, 4).unwrap();

    let query = tensor(0, vec![1, 2, 1, 2], &[0.75; 4]);
    let mut out = tensor(0, vec![1, 2, 1, 2], &[0.0; 4]);
    let args =
        KvCacheAttentionArgs::new(&query, &key, &value, &mut out).with_decode_meta(&decode_meta);
    let error = kv_cache_attention(&ctx, args).unwrap_err();
    assert!(error.to_string().contains("causal single-token"), "{error}");

    let query = tensor(0, vec![1, 1, 1, 2], &[0.75; 2]);
    let mut out = tensor(0, vec![1, 1, 1, 2], &[0.0; 2]);
    let args = KvCacheAttentionArgs::new(&query, &key, &value, &mut out)
        .with_decode_meta(&decode_meta)
        .non_causal();
    let error = kv_cache_attention(&ctx, args).unwrap_err();
    assert!(error.to_string().contains("causal single-token"), "{error}");

    let query = tensor(0, vec![2, 1, 1, 2], &[0.75; 4]);
    let key = tensor(0, vec![2, 4, 1, 2], &[0.25; 16]);
    let value = tensor(0, vec![2, 4, 1, 2], &[0.5; 16]);
    let mut out = tensor(0, vec![2, 1, 1, 2], &[0.0; 4]);
    let args =
        KvCacheAttentionArgs::new(&query, &key, &value, &mut out).with_decode_meta(&decode_meta);
    let error = kv_cache_attention(&ctx, args).unwrap_err();
    assert!(error.to_string().contains("batch-1"), "{error}");
}

#[test]
fn kv_cache_attention_non_causal_uses_only_valid_cache_rows() {
    let ctx = CudaContext::new(0).unwrap();
    let query_values = vec![0.5, -0.25];
    let key_values = vec![0.25, 0.5, -0.5, 0.25, 0.75, -0.25, 99.0, 99.0];
    let value_values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 999.0, 999.0];
    let query = tensor(0, vec![1, 1, 1, 2], &query_values);
    let key = tensor(0, vec![1, 4, 1, 2], &key_values);
    let value = tensor(0, vec![1, 4, 1, 2], &value_values);
    let mut out = tensor(0, vec![1, 1, 1, 2], &[0.0; 2]);
    let mut args = KvCacheAttentionArgs::new(&query, &key, &value, &mut out).non_causal();
    args.valid_key_tokens = 3;
    args.policy.online_tune = false;
    kv_cache_attention(&ctx, args).unwrap();

    let expected = kv_cache_attention_reference(
        &query_values,
        &key_values,
        &value_values,
        1,
        1,
        3,
        4,
        1,
        1,
        2,
        0,
        1.0 / 2.0_f32.sqrt(),
        false,
    );
    for (actual, expected) in values(&out).iter().zip(expected) {
        assert!((*actual - expected).abs() < 0.02);
    }
}

#[test]
fn segmented_attention_keeps_packed_segments_isolated() {
    let ctx = CudaContext::new(0).unwrap();
    let query_values = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let key_values = vec![1.0, 0.0, 0.0, 1.0, 1.0, -1.0];
    let value_values = vec![2.0, 4.0, 6.0, 8.0, 100.0, 200.0];
    let query = tensor(0, vec![3, 1, 2], &query_values);
    let key = tensor(0, vec![3, 1, 2], &key_values);
    let value = tensor(0, vec![3, 1, 2], &value_values);
    let mut out = tensor(0, vec![3, 1, 2], &[0.0; 6]);
    let host_offsets = [0, 2, 3];
    let offsets = u32_buffer(0, &host_offsets);
    let mut args =
        SegmentedAttentionArgs::new(&query, &key, &value, &mut out, &offsets, &host_offsets);
    args.policy.online_tune = false;
    segmented_attention(&ctx, args).unwrap();

    let mut expected = attention_reference(
        &query_values[..4],
        &key_values[..4],
        &value_values[..4],
        1,
        2,
        2,
        1,
        1,
        2,
        1.0 / 2.0_f32.sqrt(),
        false,
    );
    expected.extend_from_slice(&value_values[4..]);
    for (actual, expected) in values(&out).iter().zip(expected) {
        assert!((*actual - expected).abs() < 0.02);
    }
}

#[test]
fn segmented_attention_prepares_before_capture_and_replays_from_cache() {
    let ctx = CudaContext::new(0).unwrap();
    let query = tensor(0, vec![3, 1, 2], &[0.25; 6]);
    let key = tensor(0, vec![3, 1, 2], &[0.5; 6]);
    let value = tensor(0, vec![3, 1, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let mut out = tensor(0, vec![3, 1, 2], &[0.0; 6]);
    let host_offsets = [0, 2, 3];
    let offsets = u32_buffer(0, &host_offsets);
    let session = ExecutionSession::with_capacity(4096, 0).unwrap();

    prepare_with_session(&session, || {
        let mut args =
            SegmentedAttentionArgs::new(&query, &key, &value, &mut out, &offsets, &host_offsets);
        args.policy.online_tune = false;
        segmented_attention(&ctx, args)
    })
    .unwrap();
    let expected = values(&out);

    let graph = crate::capture(&ctx, || {
        with_session(&session, || {
            let mut args = SegmentedAttentionArgs::new(
                &query,
                &key,
                &value,
                &mut out,
                &offsets,
                &host_offsets,
            );
            args.policy.online_tune = false;
            segmented_attention(&ctx, args)
        })
    })
    .unwrap();
    let output = crate::CudaBuffer::from_tensor(&out).unwrap();
    let sentinel: Vec<_> = (0..6)
        .flat_map(|_| bf16::from_f32(-123.0).to_bits().to_ne_bytes())
        .collect();
    output.copy_from_host(&sentinel).unwrap();
    graph.replay().unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(values(&out), expected);
}

#[test]
fn attention_semantics_use_distinct_native_registries() {
    let ctx = CudaContext::new(0).unwrap();
    let query = tensor(0, vec![1, 1, 1, 2], &[0.5, -0.25]);
    let key = tensor(0, vec![1, 2, 1, 2], &[0.25, 0.5, -0.5, 0.25]);
    let value = tensor(0, vec![1, 2, 1, 2], &[1.0, 2.0, 3.0, 4.0]);

    let mut dense_out = tensor(0, vec![1, 1, 1, 2], &[0.0; 2]);
    let mut dense_args = AttentionArgs::new(&query, &key, &value, &mut dense_out);
    dense_args.policy.online_tune = false;
    let dense = super::attention_execution::prepare(
        &ctx,
        super::attention_contracts::normalize(&ctx, dense_args).unwrap(),
    )
    .unwrap();

    let mut cache_out = tensor(0, vec![1, 1, 1, 2], &[0.0; 2]);
    let mut cache_args = KvCacheAttentionArgs::new(&query, &key, &value, &mut cache_out);
    cache_args.valid_key_tokens = 1;
    cache_args.query_start = 0;
    cache_args.policy.online_tune = false;
    let cache = super::attention_execution::prepare(
        &ctx,
        super::normalize_kv_cache_attention(&ctx, cache_args).unwrap(),
    )
    .unwrap();

    let segmented_query = tensor(0, vec![2, 1, 2], &[0.5, -0.25, 0.25, 0.75]);
    let segmented_key = tensor(0, vec![2, 1, 2], &[0.25, 0.5, -0.5, 0.25]);
    let segmented_value = tensor(0, vec![2, 1, 2], &[1.0, 2.0, 3.0, 4.0]);
    let mut segmented_out = tensor(0, vec![2, 1, 2], &[0.0; 4]);
    let host_offsets = [0, 2];
    let offsets = u32_buffer(0, &host_offsets);
    let mut segmented_args = SegmentedAttentionArgs::new(
        &segmented_query,
        &segmented_key,
        &segmented_value,
        &mut segmented_out,
        &offsets,
        &host_offsets,
    );
    segmented_args.policy.online_tune = false;
    let segmented = super::attention_execution::prepare(
        &ctx,
        super::normalize_segmented_attention(&ctx, segmented_args).unwrap(),
    )
    .unwrap();

    assert!(dense.summary().starts_with("custom-attention-fallback "));
    assert!(cache.summary().starts_with("custom-kv-cache-attention "));
    assert!(segmented
        .summary()
        .starts_with("custom-segmented-attention "));
}
