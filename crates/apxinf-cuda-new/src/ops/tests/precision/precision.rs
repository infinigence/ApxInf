//! Torch-golden precision tests for every public L3 operator.
//!
//! Every new L3 operator must add `<op>_all_candidates_match_torch` here,
//! extend `generate_torch_l3_fixtures.py`, and regenerate
//! `torch_l3_fixtures.rs`. Tests compare final L3 outputs only; they must not
//! depend on or prescribe a candidate's internal implementation.

use super::framework::{
    bf16_bits_tensor, bytes_tensor, f16_tensor, f32_tensor, scales, tensor, zeros_tensor,
};
use super::l3_behavior::{attention_reference, kv_cache_attention_reference};
use super::*;
use crate::CudaContext;
use apxinf_core::{DType, Tensor};
use std::time::Instant;

#[path = "torch_l3_fixtures.rs"]
mod torch_fixture;

#[path = "torch_split_geglu_fixtures.rs"]
mod split_fixture;

fn u32_buffer(device: usize, values: &[u32]) -> crate::CudaBuffer {
    let bytes: Vec<_> = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    let buffer = crate::CudaBuffer::alloc(bytes.len(), device).unwrap();
    buffer.copy_from_host(&bytes).unwrap();
    buffer
}

fn validate_all_candidates<'a>(
    ctx: &CudaContext,
    args: GemmArgs<'a>,
    semantic: super::contracts::Semantic,
    bias: Option<&'a Tensor>,
    expected: &'a [f32],
) -> apxinf_core::Result<()> {
    let normalized = super::contracts::normalize(ctx, args, semantic, bias)?;
    super::execution::validate_candidates(ctx, &normalized, expected)
}

fn configure_torch_case(args: &mut GemmArgs<'_>, alpha: f32, output_scale: f32) {
    args.alpha = alpha;
    args.output_scale = output_scale;
    args.policy.allow_fallback = false;
    args.policy.graph_safe = true;
}

#[test]
fn attention_all_candidates_match_reference() {
    let ctx = CudaContext::new(0).unwrap();
    let (batch, query_tokens, key_tokens, query_heads, kv_heads, head_dim) = (1, 3, 4, 4, 2, 8);
    let query_values: Vec<_> = (0..batch * query_tokens * query_heads * head_dim)
        .map(|index| ((index * 3 % 17) as f32 - 8.0) / 16.0)
        .collect();
    let key_values: Vec<_> = (0..batch * key_tokens * kv_heads * head_dim)
        .map(|index| ((index * 5 % 19) as f32 - 9.0) / 16.0)
        .collect();
    let value_values: Vec<_> = (0..batch * key_tokens * kv_heads * head_dim)
        .map(|index| ((index * 7 % 23) as f32 - 11.0) / 16.0)
        .collect();
    let scale = 1.0 / (head_dim as f32).sqrt();
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
    let mut out = zeros_tensor(
        0,
        vec![batch, query_tokens, query_heads, head_dim],
        DType::BF16,
    );
    let mut args = AttentionArgs::new(&query, &key, &value, &mut out).causal();
    args.policy.allow_fallback = false;
    args.policy.graph_safe = true;
    let normalized = super::attention_contracts::normalize(&ctx, args).unwrap();
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
        true,
    );
    super::attention_execution::validate_candidates(&ctx, &normalized, &expected).unwrap();
}

#[test]
fn kv_cache_attention_all_candidates_match_reference() {
    let ctx = CudaContext::new(0).unwrap();
    let (batch, query_tokens, valid_key_tokens, key_capacity, heads, head_dim) = (2, 2, 4, 6, 2, 8);
    let query_values: Vec<_> = (0..batch * query_tokens * heads * head_dim)
        .map(|index| ((index * 3 % 17) as f32 - 8.0) / 16.0)
        .collect();
    let key_values: Vec<_> = (0..batch * key_capacity * heads * head_dim)
        .map(|index| ((index * 5 % 19) as f32 - 9.0) / 16.0)
        .collect();
    let value_values: Vec<_> = (0..batch * key_capacity * heads * head_dim)
        .map(|index| ((index * 7 % 23) as f32 - 11.0) / 16.0)
        .collect();
    let query = tensor(0, vec![batch, query_tokens, heads, head_dim], &query_values);
    let key = tensor(0, vec![batch, key_capacity, heads, head_dim], &key_values);
    let value = tensor(0, vec![batch, key_capacity, heads, head_dim], &value_values);
    let mut out = zeros_tensor(0, vec![batch, query_tokens, heads, head_dim], DType::BF16);
    let mut args = KvCacheAttentionArgs::new(&query, &key, &value, &mut out);
    args.valid_key_tokens = valid_key_tokens;
    args.query_start = 1;
    args.policy.allow_fallback = false;
    args.policy.graph_safe = true;
    let normalized = super::normalize_kv_cache_attention(&ctx, args).unwrap();
    let expected = kv_cache_attention_reference(
        &query_values,
        &key_values,
        &value_values,
        batch,
        query_tokens,
        valid_key_tokens,
        key_capacity,
        heads,
        heads,
        head_dim,
        1,
        1.0 / (head_dim as f32).sqrt(),
        true,
    );
    super::attention_execution::validate_candidates(&ctx, &normalized, &expected).unwrap();
}

#[test]
fn segmented_attention_all_candidates_match_reference() {
    let ctx = CudaContext::new(0).unwrap();
    let (tokens, heads, head_dim) = (4, 2, 8);
    let query_values: Vec<_> = (0..tokens * heads * head_dim)
        .map(|index| ((index * 3 % 17) as f32 - 8.0) / 16.0)
        .collect();
    let key_values: Vec<_> = (0..tokens * heads * head_dim)
        .map(|index| ((index * 5 % 19) as f32 - 9.0) / 16.0)
        .collect();
    let value_values: Vec<_> = (0..tokens * heads * head_dim)
        .map(|index| ((index * 7 % 23) as f32 - 11.0) / 16.0)
        .collect();
    let query = tensor(0, vec![tokens, heads, head_dim], &query_values);
    let key = tensor(0, vec![tokens, heads, head_dim], &key_values);
    let value = tensor(0, vec![tokens, heads, head_dim], &value_values);
    let mut out = zeros_tensor(0, vec![tokens, heads, head_dim], DType::BF16);
    let host_offsets = [0, 2, 2, 4];
    let offsets = u32_buffer(0, &host_offsets);
    let mut args =
        SegmentedAttentionArgs::new(&query, &key, &value, &mut out, &offsets, &host_offsets);
    args.policy.allow_fallback = false;
    args.policy.graph_safe = true;
    let normalized = super::normalize_segmented_attention(&ctx, args).unwrap();
    let elements_per_segment = 2 * heads * head_dim;
    let mut expected = attention_reference(
        &query_values[..elements_per_segment],
        &key_values[..elements_per_segment],
        &value_values[..elements_per_segment],
        1,
        2,
        2,
        heads,
        heads,
        head_dim,
        1.0 / (head_dim as f32).sqrt(),
        false,
    );
    expected.extend(attention_reference(
        &query_values[elements_per_segment..],
        &key_values[elements_per_segment..],
        &value_values[elements_per_segment..],
        1,
        2,
        2,
        heads,
        heads,
        head_dim,
        1.0 / (head_dim as f32).sqrt(),
        false,
    ));
    super::attention_execution::validate_candidates(&ctx, &normalized, &expected).unwrap();
}

#[test]
fn attention_sm100_fmha_candidate_matches_reference() {
    let ctx = CudaContext::new(0).unwrap();
    let (batch, tokens, heads, head_dim) = (1, 256, 16, 72);
    let elements = batch * tokens * heads * head_dim;
    let query = tensor(
        0,
        vec![batch, tokens, heads, head_dim],
        &vec![0.0; elements],
    );
    let key = tensor(
        0,
        vec![batch, tokens, heads, head_dim],
        &vec![0.0; elements],
    );
    let value = tensor(
        0,
        vec![batch, tokens, heads, head_dim],
        &vec![1.0; elements],
    );
    let mut out = zeros_tensor(0, vec![batch, tokens, heads, head_dim], DType::BF16);
    let mut args = AttentionArgs::new(&query, &key, &value, &mut out);
    args.policy.allow_fallback = false;
    args.policy.graph_safe = true;
    let normalized = super::attention_contracts::normalize(&ctx, args).unwrap();
    super::attention_execution::validate_candidates(&ctx, &normalized, &vec![1.0; elements])
        .unwrap();
}

#[test]
fn attention_f16_vision_recipe_miss_uses_fa2_and_matches_reference() {
    let ctx = CudaContext::new(0).unwrap();
    let (batch, tokens, heads, head_dim) = (1, 256, 16, 72);
    let elements = batch * tokens * heads * head_dim;
    let shape = vec![batch, tokens, heads, head_dim];
    let query = f16_tensor(0, shape.clone(), &vec![0.0; elements]);
    let key = f16_tensor(0, shape.clone(), &vec![0.0; elements]);
    let value = f16_tensor(0, shape.clone(), &vec![1.0; elements]);
    let mut out = zeros_tensor(0, shape, DType::F16);
    let mut args = AttentionArgs::new(&query, &key, &value, &mut out);
    args.policy.online_tune = false;
    args.policy.cache_dir = Some(String::new());
    args.policy.graph_safe = true;
    let normalized = super::attention_contracts::normalize(&ctx, args).unwrap();
    super::attention_execution::validate_candidates(&ctx, &normalized, &vec![1.0; elements])
        .unwrap();
    let execution = super::attention_execution::prepare(&ctx, normalized).unwrap();
    assert!(
        execution.summary().starts_with("flash-attention-2 "),
        "unexpected F16 vision provider: {}",
        execution.summary()
    );
}

#[test]
fn attention_f16_language_mqa_uses_fa2() {
    let ctx = CudaContext::new(0).unwrap();
    let (batch, tokens, query_heads, kv_heads, head_dim) = (1, 522, 8, 1, 256);
    let query_shape = vec![batch, tokens, query_heads, head_dim];
    let kv_shape = vec![batch, tokens, kv_heads, head_dim];
    let query = zeros_tensor(0, query_shape.clone(), DType::F16);
    let key = zeros_tensor(0, kv_shape.clone(), DType::F16);
    let value = zeros_tensor(0, kv_shape, DType::F16);
    let mut out = zeros_tensor(0, query_shape, DType::F16);
    let mut args = AttentionArgs::new(&query, &key, &value, &mut out);
    args.policy.allow_fallback = false;
    let normalized = super::attention_contracts::normalize(&ctx, args).unwrap();
    let execution = super::attention_execution::prepare(&ctx, normalized).unwrap();
    assert!(
        execution.summary().starts_with("flash-attention-2 "),
        "unexpected F16 language MQA provider: {}",
        execution.summary()
    );
}

#[test]
fn attention_f16_action_cache_uses_fa2() {
    let ctx = CudaContext::new(0).unwrap();
    let (batch, query_tokens, key_tokens, key_capacity, query_heads, kv_heads, head_dim) =
        (1, 10, 532, 640, 8, 1, 256);
    let query_shape = vec![batch, query_tokens, query_heads, head_dim];
    let cache_shape = vec![batch, key_capacity, kv_heads, head_dim];
    let query = zeros_tensor(0, query_shape.clone(), DType::F16);
    let key = zeros_tensor(0, cache_shape.clone(), DType::F16);
    let value = f16_tensor(
        0,
        cache_shape.clone(),
        &vec![1.0; batch * key_capacity * kv_heads * head_dim],
    );
    let mut out = zeros_tensor(0, query_shape.clone(), DType::F16);
    let mut args = KvCacheAttentionArgs::new(&query, &key, &value, &mut out).non_causal();
    args.valid_key_tokens = key_tokens;
    args.policy.allow_fallback = false;
    let normalized = super::normalize_kv_cache_attention(&ctx, args).unwrap();
    super::attention_execution::validate_candidates(
        &ctx,
        &normalized,
        &vec![1.0; batch * query_tokens * query_heads * head_dim],
    )
    .unwrap();
    let execution = super::attention_execution::prepare(&ctx, normalized).unwrap();
    assert!(
        execution
            .summary()
            .starts_with("flash-attention-2-split-kv ")
            || execution
                .summary()
                .starts_with("flash-attention-2-kv-cache "),
        "unexpected F16 action KV-cache provider: {}",
        execution.summary()
    );
}

#[test]
#[ignore = "Thor-only performance characterization"]
fn profile_pi05_f16_attention_shapes() {
    fn measure(
        ctx: &CudaContext,
        execution: &super::attention_execution::Execution,
        iterations: usize,
    ) -> f64 {
        for _ in 0..20 {
            execution.enqueue_for_test().unwrap();
        }
        ctx.synchronize().unwrap();
        let start = Instant::now();
        for _ in 0..iterations {
            execution.enqueue_for_test().unwrap();
        }
        ctx.synchronize().unwrap();
        start.elapsed().as_secs_f64() * 1_000.0 / iterations as f64
    }

    let ctx = CudaContext::new(0).unwrap();

    let vision_shape = vec![2, 256, 16, 72];
    let vision_q = zeros_tensor(0, vision_shape.clone(), DType::F16);
    let vision_k = zeros_tensor(0, vision_shape.clone(), DType::F16);
    let vision_v = zeros_tensor(0, vision_shape.clone(), DType::F16);
    let mut vision_out = zeros_tensor(0, vision_shape, DType::F16);
    let vision = super::attention_execution::prepare(
        &ctx,
        super::attention_contracts::normalize(
            &ctx,
            AttentionArgs::new(&vision_q, &vision_k, &vision_v, &mut vision_out),
        )
        .unwrap(),
    )
    .unwrap();

    let language_q_shape = vec![1, 522, 8, 256];
    let language_kv_shape = vec![1, 522, 1, 256];
    let language_q = zeros_tensor(0, language_q_shape.clone(), DType::F16);
    let language_k = zeros_tensor(0, language_kv_shape.clone(), DType::F16);
    let language_v = zeros_tensor(0, language_kv_shape, DType::F16);
    let mut language_out = zeros_tensor(0, language_q_shape, DType::F8E4M3);
    let mut language_args =
        AttentionArgs::new(&language_q, &language_k, &language_v, &mut language_out);
    language_args.output_scale = 1.0 / 16.0;
    let language = super::attention_execution::prepare(
        &ctx,
        super::attention_contracts::normalize(&ctx, language_args).unwrap(),
    )
    .unwrap();

    let action_q_shape = vec![1, 10, 8, 256];
    let action_kv_shape = vec![1, 532, 1, 256];
    let action_q = zeros_tensor(0, action_q_shape.clone(), DType::F16);
    let action_k = zeros_tensor(0, action_kv_shape.clone(), DType::F16);
    let action_v = zeros_tensor(0, action_kv_shape, DType::F16);
    let mut action_out = zeros_tensor(0, action_q_shape, DType::F16);
    let action_args =
        KvCacheAttentionArgs::new(&action_q, &action_k, &action_v, &mut action_out).non_causal();
    let action = super::attention_execution::prepare(
        &ctx,
        super::normalize_kv_cache_attention(&ctx, action_args).unwrap(),
    )
    .unwrap();

    let vision_ms = measure(&ctx, &vision, 200);
    let language_ms = measure(&ctx, &language, 200);
    let action_ms = measure(&ctx, &action, 500);
    let weighted_ms = 27.0 * vision_ms + 17.0 * language_ms + 180.0 * action_ms;
    eprintln!(
        "PI0.5 Attention profile: vision={vision_ms:.6}ms ({}) language={language_ms:.6}ms ({}) action={action_ms:.6}ms ({}) weighted={weighted_ms:.3}ms",
        vision.summary(),
        language.summary(),
        action.summary()
    );
}

#[test]
fn attention_f16_mqa_522_uses_direct_e4m3_fa2() {
    let ctx = CudaContext::new(0).unwrap();
    if ctx.caps().sm < 100 {
        return;
    }
    let (batch, tokens, query_heads, kv_heads, head_dim) = (1, 522, 8, 1, 256);
    let query_shape = vec![batch, tokens, query_heads, head_dim];
    let kv_shape = vec![batch, tokens, kv_heads, head_dim];
    let query = f16_tensor(
        0,
        query_shape.clone(),
        &vec![0.0; batch * tokens * query_heads * head_dim],
    );
    let key = f16_tensor(
        0,
        kv_shape.clone(),
        &vec![0.0; batch * tokens * kv_heads * head_dim],
    );
    let value = f16_tensor(
        0,
        kv_shape,
        &vec![1.0; batch * tokens * kv_heads * head_dim],
    );
    let mut out = zeros_tensor(0, query_shape, DType::F8E4M3);
    let output_scale = 1.0 / 16.0;
    let mut args = AttentionArgs::new(&query, &key, &value, &mut out);
    args.output_scale = output_scale;
    args.policy.allow_fallback = false;
    args.policy.graph_safe = true;
    let normalized = super::attention_contracts::normalize(&ctx, args).unwrap();
    let expected = vec![1.0; batch * tokens * query_heads * head_dim];
    super::attention_execution::validate_candidates(&ctx, &normalized, &expected).unwrap();
    let execution = super::attention_execution::prepare(&ctx, normalized).unwrap();
    assert!(
        execution
            .summary()
            .starts_with("flash-attention-2-f16-e4m3-522 "),
        "unexpected direct-E4M3 provider: {}",
        execution.summary()
    );
    drop(execution);
    let mut args = AttentionArgs::new(&query, &key, &value, &mut out);
    args.output_scale = output_scale;
    args.policy.allow_fallback = false;
    super::attention(&ctx, args).unwrap();

    let output = crate::CudaBuffer::from_tensor(&out).unwrap();
    let mut bytes = vec![0; output.len()];
    output.copy_to_host(&mut bytes).unwrap();
    // E4M3 encodes 1.0 / output_scale = 16.0 as 0x58.
    assert!(bytes.iter().all(|&value| value == 0x58));
}

#[test]
fn torch_validation_requires_the_l3_output_shape() {
    let ctx = CudaContext::new(0).unwrap();
    let a = tensor(0, vec![2, 3], &[1.0; 6]);
    let b = tensor(0, vec![3, 4], &[1.0; 12]);
    let mut out = tensor(0, vec![2, 4], &[0.0; 8]);
    let args = GemmArgs::new(&a, &b, &mut out);
    let error = validate_all_candidates(
        &ctx,
        args,
        super::contracts::Semantic::Gemm,
        None,
        &[1.0; 7],
    )
    .unwrap_err();
    assert!(error
        .to_string()
        .contains("Torch validation output does not match"));
}

#[test]
fn gemm_all_candidates_match_torch() {
    use torch_fixture as f;

    let ctx = CudaContext::new(0).unwrap();
    let a = bf16_bits_tensor(0, vec![f::M, f::K], f::BF16_A);
    let b = bf16_bits_tensor(0, vec![f::K, f::N], f::BF16_B);
    let mut out = zeros_tensor(0, vec![f::M, f::N], DType::F32);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    configure_torch_case(&mut args, f::BF16_ALPHA, f::BF16_OUTPUT_SCALE);
    validate_all_candidates(
        &ctx,
        args,
        super::contracts::Semantic::Gemm,
        None,
        f::BF16_GEMM,
    )
    .unwrap();

    let ctx = CudaContext::new(0).unwrap();
    let a = bytes_tensor(0, vec![f::M, f::K], DType::F8E4M3, f::FP8_A);
    let b = bytes_tensor(0, vec![f::K, f::N], DType::F8E4M3, f::FP8_B);
    let mut out = zeros_tensor(0, vec![f::M, f::N], DType::F16);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    args.quantization = GemmQuantization::Fp8UnitScale;
    configure_torch_case(&mut args, f::FP8_UNIT_ALPHA, f::FP8_UNIT_OUTPUT_SCALE);
    validate_all_candidates(
        &ctx,
        args,
        super::contracts::Semantic::Gemm,
        None,
        f::FP8_UNIT_GEMM,
    )
    .unwrap();

    let ctx = CudaContext::new(0).unwrap();
    let a = bytes_tensor(0, vec![f::M, f::K], DType::F8E4M3, f::FP8_A);
    let b = bytes_tensor(0, vec![f::K, f::N], DType::F8E4M3, f::FP8_B);
    let row_scales = scales(0, f::ROW_SCALES);
    let channel_scales = scales(0, f::CHANNEL_SCALES);
    let mut out = zeros_tensor(0, vec![f::M, f::N], DType::F32);
    let mut args = GemmArgs::row_channel_quantized(&a, &row_scales, &b, &channel_scales, &mut out);
    configure_torch_case(&mut args, f::FP8_SCALED_ALPHA, f::FP8_SCALED_OUTPUT_SCALE);
    validate_all_candidates(
        &ctx,
        args,
        super::contracts::Semantic::Gemm,
        None,
        f::FP8_SCALED_GEMM,
    )
    .unwrap();

    let ctx = CudaContext::new(0).unwrap();
    let a = bytes_tensor(0, vec![f::M, f::K], DType::I8, f::W8A8_A);
    let b = bytes_tensor(0, vec![f::K, f::N], DType::I8, f::W8A8_B);
    let row_scales = scales(0, f::ROW_SCALES);
    let channel_scales = scales(0, f::CHANNEL_SCALES);
    let mut out = zeros_tensor(0, vec![f::M, f::N], DType::BF16);
    let mut args = GemmArgs::row_channel_quantized(&a, &row_scales, &b, &channel_scales, &mut out);
    configure_torch_case(&mut args, f::W8A8_ALPHA, f::W8A8_OUTPUT_SCALE);
    validate_all_candidates(
        &ctx,
        args,
        super::contracts::Semantic::Gemm,
        None,
        f::W8A8_GEMM,
    )
    .unwrap();
}

#[test]
fn gemm_bias_all_candidates_match_torch() {
    use torch_fixture as f;

    let ctx = CudaContext::new(0).unwrap();
    let a = bf16_bits_tensor(0, vec![f::M, f::K], f::BF16_A);
    let b = bf16_bits_tensor(0, vec![f::K, f::N], f::BF16_B);
    let bias = bf16_bits_tensor(0, vec![f::N], f::BF16_BIAS);
    let mut out = zeros_tensor(0, vec![f::M, f::N], DType::F32);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    configure_torch_case(&mut args, f::BF16_ALPHA, f::BF16_OUTPUT_SCALE);
    validate_all_candidates(
        &ctx,
        args,
        super::contracts::Semantic::GemmBias,
        Some(&bias),
        f::BF16_GEMM_BIAS,
    )
    .unwrap();

    let ctx = CudaContext::new(0).unwrap();
    let a = bytes_tensor(0, vec![f::M, f::K], DType::F8E4M3, f::FP8_A);
    let b = bytes_tensor(0, vec![f::K, f::N], DType::F8E4M3, f::FP8_B);
    let bias = f32_tensor(0, vec![f::N], f::FP8_BIAS);
    let row_scales = scales(0, f::ROW_SCALES);
    let channel_scales = scales(0, f::CHANNEL_SCALES);
    let mut out = zeros_tensor(0, vec![f::M, f::N], DType::F32);
    let mut args = GemmArgs::row_channel_quantized(&a, &row_scales, &b, &channel_scales, &mut out);
    configure_torch_case(&mut args, f::FP8_SCALED_ALPHA, f::FP8_SCALED_OUTPUT_SCALE);
    validate_all_candidates(
        &ctx,
        args,
        super::contracts::Semantic::GemmBias,
        Some(&bias),
        f::FP8_SCALED_GEMM_BIAS,
    )
    .unwrap();

    let ctx = CudaContext::new(0).unwrap();
    let a = bytes_tensor(0, vec![f::M, f::K], DType::I8, f::W8A8_A);
    let b = bytes_tensor(0, vec![f::K, f::N], DType::I8, f::W8A8_B);
    let bias = bf16_bits_tensor(0, vec![f::N], f::BF16_BIAS);
    let row_scales = scales(0, f::ROW_SCALES);
    let channel_scales = scales(0, f::CHANNEL_SCALES);
    let mut out = zeros_tensor(0, vec![f::M, f::N], DType::BF16);
    let mut args = GemmArgs::row_channel_quantized(&a, &row_scales, &b, &channel_scales, &mut out);
    configure_torch_case(&mut args, f::W8A8_ALPHA, f::W8A8_OUTPUT_SCALE);
    validate_all_candidates(
        &ctx,
        args,
        super::contracts::Semantic::GemmBias,
        Some(&bias),
        f::W8A8_GEMM_BIAS,
    )
    .unwrap();
}

#[test]
fn gemm_bias_f16_all_candidates_match_reference() {
    // The native epilogue is the only non-fallback candidate this bias
    // projection has, so this test is its single precision gate. The reference
    // accumulates in f64 and rounds to f32, which is the same contract as an
    // F32 projection with the bias added before the cast; compare_reference
    // supplies the tolerance.
    let ctx = CudaContext::new(0).unwrap();
    let (m, k, n) = (64usize, 256usize, 256usize);
    // Eighths, sixteenths and quarters are exact in F16, so the reference can
    // read the same f32 inputs without carrying an extra input-quantization
    // error.
    let a_values: Vec<f32> = (0..m * k)
        .map(|index| ((index * 7 % 33) as f32 - 16.0) / 8.0)
        .collect();
    let b_values: Vec<f32> = (0..k * n)
        .map(|index| ((index * 5 % 31) as f32 - 15.0) / 16.0)
        .collect();
    let bias_values: Vec<f32> = (0..n)
        .map(|index| ((index * 3 % 17) as f32 - 8.0) / 4.0)
        .collect();

    let a = f16_tensor(0, vec![m, k], &a_values);
    let b = f16_tensor(0, vec![k, n], &b_values);
    // contracts::normalize ties the bias dtype to the projection: F16 A means an
    // F16 bias.
    let bias = f16_tensor(0, vec![n], &bias_values);
    let mut out = zeros_tensor(0, vec![m, n], DType::F16);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    args.alpha = 1.0;
    args.output_scale = 1.0;
    args.policy.allow_fallback = false;
    args.policy.graph_safe = true;

    let mut expected = vec![0f32; m * n];
    for row in 0..m {
        for column in 0..n {
            let mut accumulator = 0f64;
            for inner in 0..k {
                accumulator +=
                    a_values[row * k + inner] as f64 * b_values[inner * n + column] as f64;
            }
            expected[row * n + column] = (accumulator + bias_values[column] as f64) as f32;
        }
    }

    let normalized = super::contracts::normalize(
        &ctx,
        args,
        super::contracts::Semantic::GemmBias,
        Some(&bias),
    )
    .unwrap();
    super::execution::validate_candidates(&ctx, &normalized, &expected).unwrap();
}

#[test]
fn gemm_bias_gelu_all_candidates_match_torch() {
    use torch_fixture as f;

    let ctx = CudaContext::new(0).unwrap();
    let a = bf16_bits_tensor(0, vec![f::M, f::K], f::BF16_A);
    let b = bf16_bits_tensor(0, vec![f::K, f::N], f::BF16_B);
    let bias = bf16_bits_tensor(0, vec![f::N], f::BF16_BIAS);
    let mut out = zeros_tensor(0, vec![f::M, f::N], DType::F32);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    configure_torch_case(&mut args, f::BF16_ALPHA, f::BF16_OUTPUT_SCALE);
    validate_all_candidates(
        &ctx,
        args,
        super::contracts::Semantic::GemmBiasGelu,
        Some(&bias),
        f::BF16_GEMM_BIAS_GELU,
    )
    .unwrap();

    let ctx = CudaContext::new(0).unwrap();
    let a = bytes_tensor(0, vec![f::M, f::K], DType::F8E4M3, f::FP8_A);
    let b = bytes_tensor(0, vec![f::K, f::N], DType::F8E4M3, f::FP8_B);
    let bias = f32_tensor(0, vec![f::N], f::FP8_BIAS);
    let row_scales = scales(0, f::ROW_SCALES);
    let channel_scales = scales(0, f::CHANNEL_SCALES);
    let mut out = zeros_tensor(0, vec![f::M, f::N], DType::F32);
    let mut args = GemmArgs::row_channel_quantized(&a, &row_scales, &b, &channel_scales, &mut out);
    configure_torch_case(&mut args, f::FP8_SCALED_ALPHA, f::FP8_SCALED_OUTPUT_SCALE);
    validate_all_candidates(
        &ctx,
        args,
        super::contracts::Semantic::GemmBiasGelu,
        Some(&bias),
        f::FP8_SCALED_GEMM_BIAS_GELU,
    )
    .unwrap();
}

#[test]
fn gemm_geglu_all_candidates_match_torch() {
    use torch_fixture as f;

    let ctx = CudaContext::new(0).unwrap();
    let a = bf16_bits_tensor(0, vec![f::M, f::K], f::BF16_A);
    let b = bf16_bits_tensor(0, vec![f::K, 2 * f::N], f::BF16_GEGLU_B);
    let mut out = zeros_tensor(0, vec![f::M, f::N], DType::F32);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    configure_torch_case(&mut args, f::BF16_ALPHA, f::BF16_OUTPUT_SCALE);
    validate_all_candidates(
        &ctx,
        args,
        super::contracts::Semantic::GemmGeglu,
        None,
        f::BF16_GEMM_GEGLU,
    )
    .unwrap();

    let ctx = CudaContext::new(0).unwrap();
    let a = bytes_tensor(0, vec![f::M, f::K], DType::F8E4M3, f::FP8_A);
    let b = bytes_tensor(0, vec![f::K, 2 * f::N], DType::F8E4M3, f::FP8_GEGLU_B);
    let mut out = zeros_tensor(0, vec![f::M, f::N], DType::F32);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    args.quantization = GemmQuantization::Fp8UnitScale;
    configure_torch_case(&mut args, f::FP8_UNIT_ALPHA, f::FP8_UNIT_OUTPUT_SCALE);
    validate_all_candidates(
        &ctx,
        args,
        super::contracts::Semantic::GemmGeglu,
        None,
        f::FP8_UNIT_GEMM_GEGLU,
    )
    .unwrap();
}

#[test]
fn gemm_geglu_three_view_all_candidates_match_torch() {
    use split_fixture as fixture;

    let ctx = CudaContext::new(0).unwrap();
    // T10 (768 patches + 10) and T21 (768 patches + 21) three-view rows.
    for rows in [778usize, 789] {
        let (inner, width) = (2048, 16384);
        let mut activation = vec![0u8; rows * inner];
        let mut weight = vec![0u8; inner * width * 2];
        for row in 0..rows {
            activation[row * inner + row % 8] = 0x38;
        }
        for channel in 0..8 {
            for column in 0..width {
                weight[channel * width * 2 + column] = fixture::SPLIT_GEGLU_VALUES[channel];
                weight[channel * width * 2 + width + column] =
                    fixture::SPLIT_GEGLU_VALUES[column % 8];
            }
        }
        let activation = bytes_tensor(0, vec![rows, inner], DType::F8E4M3, &activation);
        let weight = bytes_tensor(0, vec![inner, width * 2], DType::F8E4M3, &weight);
        let mut output = zeros_tensor(0, vec![rows, width], DType::F8E4M3);
        for (alpha, scale, golden) in [
            (1.0, 1.0, fixture::SPLIT_GEGLU_UNIT),
            (0.75, 1.25, fixture::SPLIT_GEGLU_SCALED),
        ] {
            let expected: Vec<_> = (0..rows * width)
                .map(|index| golden[(index / width % 8) * 8 + index % 8])
                .collect();
            let mut args = GemmArgs::new(&activation, &weight, &mut output)
                .with_immutable_weight(WeightVersion::new(1));
            args.quantization = GemmQuantization::Fp8UnitScale;
            configure_torch_case(&mut args, alpha, scale);
            args.policy.workspace_limit = 1024 * 1024 * 1024;
            validate_all_candidates(
                &ctx,
                args,
                super::contracts::Semantic::GemmGeglu,
                None,
                &expected,
            )
            .unwrap();
        }
    }
}

/// Host reference for `conv1d`: PyTorch's temporal cross-correlation,
/// accumulated in F32 from the channel-last operands.
#[allow(clippy::too_many_arguments)]
fn conv1d_reference(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    batch: usize,
    in_length: usize,
    in_channels: usize,
    out_channels: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Vec<f32> {
    let out_length = (in_length + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;
    let in_per_group = in_channels / groups;
    let out_per_group = out_channels / groups;
    let mut output = vec![0.0f32; batch * out_length * out_channels];
    for item in 0..batch {
        for out_channel in 0..out_channels {
            let group = out_channel / out_per_group;
            for position in 0..out_length {
                let mut sum = 0.0f32;
                for tap in 0..kernel {
                    let offset = position * stride + tap * dilation;
                    if offset < padding {
                        continue;
                    }
                    let source = offset - padding;
                    if source >= in_length {
                        continue;
                    }
                    for channel in 0..in_per_group {
                        let input_index = (item * in_length + source) * in_channels
                            + group * in_per_group
                            + channel;
                        let weight_index = (out_channel * in_per_group + channel) * kernel + tap;
                        sum += input[input_index] * weight[weight_index];
                    }
                }
                output[(item * out_length + position) * out_channels + out_channel] =
                    sum + bias[out_channel];
            }
        }
    }
    output
}

/// Host reference for `conv-transpose1d`: the adjoint scatter of `conv1d`.
#[allow(clippy::too_many_arguments)]
fn conv_transpose1d_reference(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    batch: usize,
    in_length: usize,
    in_channels: usize,
    out_channels: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    output_padding: usize,
    groups: usize,
) -> Vec<f32> {
    let out_length = (in_length - 1) * stride - 2 * padding + (kernel - 1) + output_padding + 1;
    let in_per_group = in_channels / groups;
    let out_per_group = out_channels / groups;
    let mut output = vec![0.0f32; batch * out_length * out_channels];
    for item in 0..batch {
        for position in 0..in_length {
            for in_channel in 0..in_channels {
                let group = in_channel / in_per_group;
                for tap in 0..kernel {
                    let scatter = position * stride + tap;
                    if scatter < padding {
                        continue;
                    }
                    let target = scatter - padding;
                    if target >= out_length {
                        continue;
                    }
                    for channel in 0..out_per_group {
                        let out_channel = group * out_per_group + channel;
                        output[(item * out_length + target) * out_channels + out_channel] += input
                            [(item * in_length + position) * in_channels + in_channel]
                            * weight[(in_channel * out_per_group + channel) * kernel + tap];
                    }
                }
            }
        }
    }
    for position in 0..batch * out_length {
        for out_channel in 0..out_channels {
            output[position * out_channels + out_channel] += bias[out_channel];
        }
    }
    output
}

fn f32_tensor_values(tensor: &Tensor) -> Vec<f32> {
    let buffer = crate::CudaBuffer::from_tensor(tensor).unwrap();
    let mut bytes = vec![0; buffer.len()];
    buffer.copy_to_host(&mut bytes).unwrap();
    bytes
        .chunks_exact(4)
        .map(|value| f32::from_ne_bytes(value.try_into().unwrap()))
        .collect()
}

fn assert_matches_reference(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    let error: f32 = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected) * (actual - expected))
        .sum();
    let signal: f32 = expected.iter().map(|value| value * value).sum();
    let relative_l2 = (error / signal.max(f32::MIN_POSITIVE)).sqrt();
    assert!(
        relative_l2 < 1e-5,
        "{label} drifted from the reference by relative L2 {relative_l2}"
    );
}

#[test]
fn conv1d_matches_reference() {
    let ctx = CudaContext::new(0).unwrap();
    let (batch, in_length, in_channels, out_channels, kernel) = (2, 5, 4, 6, 3);
    let (stride, padding, dilation, groups) = (2, 1, 2, 2);
    let out_length = (in_length + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;
    let input_values: Vec<_> = (0..batch * in_length * in_channels)
        .map(|index| ((index * 7 % 23) as f32 - 11.0) / 32.0)
        .collect();
    let weight_values: Vec<_> = (0..out_channels * (in_channels / groups) * kernel)
        .map(|index| ((index * 5 % 19) as f32 - 9.0) / 16.0)
        .collect();
    let bias_values: Vec<_> = (0..out_channels)
        .map(|index| (index as f32 - 2.0) / 8.0)
        .collect();
    let expected = conv1d_reference(
        &input_values,
        &weight_values,
        &bias_values,
        batch,
        in_length,
        in_channels,
        out_channels,
        kernel,
        stride,
        padding,
        dilation,
        groups,
    );

    let input = f32_tensor(0, vec![batch, in_length, in_channels], &input_values);
    let weight = f32_tensor(
        0,
        vec![out_channels, in_channels / groups, kernel],
        &weight_values,
    );
    let bias = f32_tensor(0, vec![out_channels], &bias_values);
    let mut output = zeros_tensor(0, vec![batch, out_length, out_channels], DType::F32);
    let mut args =
        Conv1dArgs::new(Conv1dSemantic::Conv1d, &input, &weight, &mut output).with_bias(&bias);
    args.stride = stride;
    args.padding = padding;
    args.dilation = dilation;
    args.groups = groups;
    conv1d(&ctx, args).unwrap();
    ctx.synchronize().unwrap();
    assert_matches_reference(&f32_tensor_values(&output), &expected, "conv1d");
}

#[test]
fn conv_transpose1d_matches_reference() {
    let ctx = CudaContext::new(0).unwrap();
    let (batch, in_length, in_channels, out_channels, kernel) = (2, 4, 4, 6, 3);
    let (stride, padding, output_padding, groups) = (2, 1, 1, 2);
    let out_length = (in_length - 1) * stride - 2 * padding + (kernel - 1) + output_padding + 1;
    let input_values: Vec<_> = (0..batch * in_length * in_channels)
        .map(|index| ((index * 11 % 17) as f32 - 8.0) / 16.0)
        .collect();
    let weight_values: Vec<_> = (0..in_channels * (out_channels / groups) * kernel)
        .map(|index| ((index * 3 % 13) as f32 - 6.0) / 16.0)
        .collect();
    let bias_values: Vec<_> = (0..out_channels)
        .map(|index| (index as f32 - 2.0) / 16.0)
        .collect();
    let expected = conv_transpose1d_reference(
        &input_values,
        &weight_values,
        &bias_values,
        batch,
        in_length,
        in_channels,
        out_channels,
        kernel,
        stride,
        padding,
        output_padding,
        groups,
    );

    let input = f32_tensor(0, vec![batch, in_length, in_channels], &input_values);
    let weight = f32_tensor(
        0,
        vec![in_channels, out_channels / groups, kernel],
        &weight_values,
    );
    let bias = f32_tensor(0, vec![out_channels], &bias_values);
    let mut output = zeros_tensor(0, vec![batch, out_length, out_channels], DType::F32);
    let mut args = Conv1dArgs::new(
        Conv1dSemantic::ConvTranspose1d,
        &input,
        &weight,
        &mut output,
    )
    .with_bias(&bias);
    args.stride = stride;
    args.padding = padding;
    args.output_padding = output_padding;
    args.groups = groups;
    conv1d(&ctx, args).unwrap();
    ctx.synchronize().unwrap();
    assert_matches_reference(&f32_tensor_values(&output), &expected, "conv-transpose1d");
}
