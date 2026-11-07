//! The typed-operator surface must produce exactly what the kernel facade
//! produces: it is an unpacking layer, not a second implementation.
//!
//! Every case here runs the operator twice — once through `ops`, once through
//! `kernels` — and compares the bytes. That is the strongest statement the
//! adapter can make, and it is the one that matters while both entry points
//! exist.
//!
//! ```text
//! cargo test -p apxinf-musa --test ops_parity -- --ignored --nocapture
//! ```

use apxinf_core::{DType, Shape, Tensor};
use apxinf_musa::{kernels, ops, transfers, MusaContext};
use half::bf16;

fn upload(ctx: &MusaContext, dims: &[usize], seed: u32) -> Tensor {
    let len = dims.iter().product::<usize>();
    let mut value = seed as f32;
    let values: Vec<bf16> = (0..len)
        .map(|index| {
            value = value * 1.000_001 + 0.001;
            bf16::from_f32((value.sin() + (index % 7) as f32 * 0.01) * 0.1)
        })
        .collect();
    let host = Tensor::from_bf16(dims.to_vec(), &values).unwrap();
    transfers::to_device(&host, ctx.device_id()).unwrap()
}

fn output(ctx: &MusaContext, dims: &[usize]) -> Tensor {
    ctx.allocate_output(Shape::new(dims.to_vec()), DType::BF16)
        .unwrap()
}

fn read(ctx: &MusaContext, tensor: &Tensor) -> Vec<u8> {
    ctx.synchronize().unwrap();
    let host = transfers::to_cpu(tensor).unwrap();
    let values = host.as_bf16().unwrap();
    values.iter().flat_map(|value| value.to_bits().to_le_bytes()).collect()
}

fn assert_same(ctx: &MusaContext, ours: &Tensor, theirs: &Tensor, what: &str) {
    assert_eq!(ours.shape().dims(), theirs.shape().dims(), "{what} shape");
    assert_eq!(read(ctx, ours), read(ctx, theirs), "{what} bits");
}

#[test]
#[ignore = "needs a MUSA device"]
fn normalization_matches_the_kernel_facade() {
    let ctx = MusaContext::new(0).unwrap();
    let (rows, cols) = (8, 16);
    let input = upload(&ctx, &[rows, cols], 1);
    let weight = upload(&ctx, &[cols], 2);
    let bias = upload(&ctx, &[cols], 3);
    let residual = upload(&ctx, &[rows, cols], 4);
    let style = upload(&ctx, &[3 * cols], 5);
    let gate_style = upload(&ctx, &[3 * cols], 6);
    let eps = 1e-6;

    let mut out = output(&ctx, &[rows, cols]);
    ops::rms_norm(&ctx, ops::RmsNormArgs::new(&input, &weight, &mut out, eps)).unwrap();
    assert_same(&ctx, &out, &kernels::norm::rms_bf16(&ctx, &input, &weight, eps).unwrap(), "rms_norm");

    let mut out = output(&ctx, &[rows, cols]);
    ops::layer_norm(&ctx, ops::LayerNormArgs::new(&input, &weight, &bias, &mut out, eps)).unwrap();
    assert_same(
        &ctx,
        &out,
        &kernels::norm::layer_bf16(&ctx, &input, &weight, &bias, eps).unwrap(),
        "layer_norm",
    );

    let mut out = output(&ctx, &[rows, cols]);
    ops::adaptive_rms_norm(
        &ctx,
        ops::AdaptiveRmsNormArgs::new(&input, &style, &mut out, eps),
    )
    .unwrap();
    assert_same(
        &ctx,
        &out,
        &kernels::norm::adaptive_rms_bf16(&ctx, &input, &style, eps).unwrap(),
        "adaptive_rms_norm",
    );

    let mut hidden = output(&ctx, &[rows, cols]);
    ops::bias_residual(
        &ctx,
        ops::BiasResidualArgs::new(&input, Some(&bias), &residual, &mut hidden),
    )
    .unwrap();
    assert_same(
        &ctx,
        &hidden,
        &kernels::fused::bias_residual_bf16(&ctx, &input, Some(&bias), &residual).unwrap(),
        "bias_residual",
    );

    let mut hidden = output(&ctx, &[rows, cols]);
    let mut normalized = output(&ctx, &[rows, cols]);
    ops::ada_gate_residual_rms_norm(
        &ctx,
        ops::AdaGateResidualRmsNormArgs::new(
            &input,
            &residual,
            &style,
            &gate_style,
            &mut hidden,
            &mut normalized,
            eps,
        ),
    )
    .unwrap();
    let reference =
        kernels::fused::adaptive_gate_residual_rms_bf16(&ctx, &input, &residual, &gate_style, &style, eps)
            .unwrap();
    assert_same(&ctx, &hidden, &reference.hidden, "ada_gate hidden");
    assert_same(&ctx, &normalized, &reference.normalized, "ada_gate normalized");
}

#[test]
#[ignore = "needs a MUSA device"]
fn gemm_and_activation_match_the_kernel_facade() {
    let ctx = MusaContext::new(0).unwrap();
    let (rows, inner, cols) = (8, 16, 32);
    let a = upload(&ctx, &[rows, inner], 7);
    let b = upload(&ctx, &[inner, cols], 8);

    let mut out = output(&ctx, &[rows, cols]);
    ops::gemm(&ctx, ops::GemmArgs::new(&a, &b, &mut out)).unwrap();
    assert_same(&ctx, &out, &kernels::gemm::bf16(&ctx, &a, &b).unwrap(), "gemm");

    let mut out = output(&ctx, &[rows, cols / 2]);
    ops::gemm_geglu(&ctx, ops::GemmGegluArgs { gemm: ops::GemmArgs::new(&a, &b, &mut out) })
        .unwrap();
    assert_same(
        &ctx,
        &out,
        &kernels::gemm::bf16_geglu_fused(&ctx, &a, &b, false, None, None).unwrap(),
        "gemm_geglu",
    );

    let state = upload(&ctx, &[rows, cols], 9);
    let velocity = upload(&ctx, &[rows, cols], 10);
    let mut out = output(&ctx, &[rows, cols]);
    let mut args = ops::PointwiseArgs::new(ops::PointwiseSemantic::EulerUpdate, &state, &mut out);
    args.secondary = Some(&velocity);
    args.dt = 0.25;
    ops::pointwise(&ctx, args).unwrap();
    assert_same(
        &ctx,
        &out,
        &kernels::elementwise::euler_update_bf16(&ctx, &state, &velocity, 0.25).unwrap(),
        "euler_update",
    );
}

#[test]
#[ignore = "needs a MUSA device"]
fn rope_and_attention_match_the_kernel_facade() {
    let ctx = MusaContext::new(0).unwrap();
    let (tokens, q_heads, kv_heads, head_dim) = (16, 4, 1, 8);
    let theta = 10_000.0;
    let packed = upload(&ctx, &[tokens, (q_heads + 2 * kv_heads) * head_dim], 11);

    let mut q = output(&ctx, &[tokens, q_heads, head_dim]);
    let mut k = output(&ctx, &[tokens, kv_heads, head_dim]);
    let mut v = output(&ctx, &[tokens, kv_heads, head_dim]);
    ops::rope(
        &ctx,
        ops::RopeArgs {
            semantic: ops::RopeSemantic::SplitQkvRope,
            qkv: &packed,
            bias: None,
            q: &mut q,
            k: &mut k,
            v: &mut v,
            q_heads,
            kv_heads,
            head_dim,
            theta,
            position_offset: 3,
            kv_output_offset: 0,
        },
    )
    .unwrap();
    let reference = kernels::rope::split_qkv_apply_bf16(
        &ctx, &packed, None, q_heads, kv_heads, head_dim, theta, 3,
    )
    .unwrap();
    assert_same(&ctx, &q, &reference.q, "rope q");
    assert_same(&ctx, &k, &reference.k, "rope k");
    assert_same(&ctx, &v, &reference.v, "rope v");

    // Vision geometry: three separate [tokens, heads, head_dim] views.
    let (vision_tokens, heads, vision_dim) = (64, 16, 72);
    let q = upload(&ctx, &[vision_tokens, heads, vision_dim], 12);
    let k = upload(&ctx, &[vision_tokens, heads, vision_dim], 13);
    let v = upload(&ctx, &[vision_tokens, heads, vision_dim], 14);
    let out = output(&ctx, &[vision_tokens, heads, vision_dim]);
    let mut out4 = out
        .reshape(vec![1, vision_tokens, heads, vision_dim])
        .unwrap();
    ops::attention(
        &ctx,
        ops::AttentionArgs::new(
            &q.reshape(vec![1, vision_tokens, heads, vision_dim]).unwrap(),
            &k.reshape(vec![1, vision_tokens, heads, vision_dim]).unwrap(),
            &v.reshape(vec![1, vision_tokens, heads, vision_dim]).unwrap(),
            &mut out4,
        ),
    )
    .unwrap();
    assert_same(
        &ctx,
        &out,
        &kernels::attention::mha_bf16(&ctx, &q, &k, &v, vision_tokens).unwrap(),
        "attention",
    );

    // Prefix geometry: the layer hands the MQA op a `[1, keys, 1, head_dim]`
    // cache and a `[1, tokens, heads, head_dim]` query, while the kernel reads
    // the cache as rows of `head_dim`.
    let (keys, heads, mqa_dim) = (24, 8, 256);
    let q = upload(&ctx, &[1, 6, heads, mqa_dim], 18);
    let k = upload(&ctx, &[1, keys, 1, mqa_dim], 19);
    let v = upload(&ctx, &[1, keys, 1, mqa_dim], 20);
    let mut out = output(&ctx, &[1, 6, heads, mqa_dim]);
    ops::attention(
        &ctx,
        ops::AttentionArgs::new(&q, &k, &v, &mut out),
    )
    .unwrap();
    let reference = kernels::attention::mqa_bf16(
        &ctx,
        &q.reshape(vec![6, heads, mqa_dim]).unwrap(),
        &k.reshape(vec![keys, mqa_dim]).unwrap(),
        &v.reshape(vec![keys, mqa_dim]).unwrap(),
        keys,
    )
    .unwrap()
    .reshape(vec![1, 6, heads, mqa_dim])
    .unwrap();
    assert_same(&ctx, &out, &reference, "mqa attention");
}

#[test]
#[ignore = "needs a MUSA device"]
fn gather_and_cache_match_the_kernel_facade() {
    let ctx = MusaContext::new(0).unwrap();
    let (vocab, width, tokens) = (32, 8, 5);
    let table = upload(&ctx, &[vocab, width], 15);
    let ids = apxinf_musa::MusaBuffer::alloc_on(&ctx, tokens * 4).unwrap();
    ids.copy_from_host(&[1u32, 3, 5, 7, 9].iter().flat_map(|id| id.to_le_bytes()).collect::<Vec<u8>>())
        .unwrap();

    let mut language = output(&ctx, &[tokens, width]);
    let mut args = ops::GatherArgs::new(ops::GatherSemantic::EmbeddingLookup, &table, &mut language);
    args.ids = Some(&ids);
    ops::gather(&ctx, args).unwrap();
    assert_same(
        &ctx,
        &language,
        &kernels::embedding::lookup_bf16(&ctx, &table, &ids, tokens).unwrap(),
        "embedding lookup",
    );

    let first = upload(&ctx, &[3, width], 16);
    let second = upload(&ctx, &[4, width], 17);
    assert_same(
        &ctx,
        &ops::concat_rows(&ctx, &first, &second).unwrap(),
        &kernels::elementwise::concat_rows_bf16(&ctx, &first, &second).unwrap(),
        "concat_rows",
    );
}

/// An operator must not care where its operands live.
///
/// A fixed-shape run takes its intermediates from the session arena and its
/// weights from driver memory, so the same operator sees both. Nothing in the
/// contract distinguishes them, and a captured graph only reproduces the eager
/// answer if the operator's result is a function of its values alone — a
/// kernel that dispatched on pointer alignment or read past an input would
/// break exactly that, silently, and only for some layouts.
#[test]
#[ignore = "needs a MUSA device"]
fn operands_from_the_arena_and_from_the_driver_agree() {
    let ctx = MusaContext::new(0).unwrap();
    let device = ctx.device_id();
    let (rows, cols) = (10, 1024);
    let eps = 1e-6;

    let input = upload(&ctx, &[rows, cols], 21);
    let style = upload(&ctx, &[3 * cols], 22);
    let gate = upload(&ctx, &[3 * cols], 23);
    let residual = upload(&ctx, &[rows, cols], 24);
    let bytes = |tensor: &Tensor| read(&ctx, tensor);

    /// A tensor of `host`'s shape whose storage comes from the active session.
    fn resident(ctx: &MusaContext, host: &Tensor, bytes: &[u8]) -> Tensor {
        let tensor = ctx
            .allocate_output(host.shape().clone(), host.dtype())
            .unwrap();
        apxinf_musa::MusaBuffer::from_tensor(&tensor)
            .unwrap()
            .copy_from_host(bytes)
            .unwrap();
        tensor
    }

    let mut driver_out = output(&ctx, &[rows, cols]);
    ops::adaptive_rms_norm(
        &ctx,
        ops::AdaptiveRmsNormArgs::new(&input, &style, &mut driver_out, eps),
    )
    .unwrap();
    let expected = read(&ctx, &driver_out);

    let session = apxinf_musa::ExecutionSession::with_capacity(1 << 26, device).unwrap();
    for (label, arena_input, arena_style) in [
        ("both in the arena", true, true),
        ("input in the arena", true, false),
        ("style in the arena", false, true),
    ] {
        let actual = session
            .prepare(|| {
                let host_input = transfers::to_cpu(&input).unwrap();
                let host_style = transfers::to_cpu(&style).unwrap();
                let input = if arena_input {
                    resident(&ctx, &host_input, &bytes(&input))
                } else {
                    input.clone()
                };
                let style = if arena_style {
                    resident(&ctx, &host_style, &bytes(&style))
                } else {
                    style.clone()
                };
                let mut out = output(&ctx, &[rows, cols]);
                ops::adaptive_rms_norm(
                    &ctx,
                    ops::AdaptiveRmsNormArgs::new(&input, &style, &mut out, eps),
                )
                .unwrap();
                Ok(read(&ctx, &out))
            })
            .unwrap();
        assert_eq!(actual, expected, "adaptive_rms_norm, {label}");
    }

    let mut driver_hidden = output(&ctx, &[rows, cols]);
    let mut driver_normalized = output(&ctx, &[rows, cols]);
    ops::ada_gate_residual_rms_norm(
        &ctx,
        ops::AdaGateResidualRmsNormArgs::new(
            &input,
            &residual,
            &style,
            &gate,
            &mut driver_hidden,
            &mut driver_normalized,
            eps,
        ),
    )
    .unwrap();
    let expected_hidden = read(&ctx, &driver_hidden);
    let expected_normalized = read(&ctx, &driver_normalized);

    let (hidden, normalized) = session
        .prepare(|| {
            let host_input = transfers::to_cpu(&input).unwrap();
            let host_style = transfers::to_cpu(&style).unwrap();
            let host_gate = transfers::to_cpu(&gate).unwrap();
            let host_residual = transfers::to_cpu(&residual).unwrap();
            let input = resident(&ctx, &host_input, &bytes(&input));
            let style = resident(&ctx, &host_style, &bytes(&style));
            let gate = resident(&ctx, &host_gate, &bytes(&gate));
            let residual = resident(&ctx, &host_residual, &bytes(&residual));
            let mut hidden = output(&ctx, &[rows, cols]);
            let mut normalized = output(&ctx, &[rows, cols]);
            ops::ada_gate_residual_rms_norm(
                &ctx,
                ops::AdaGateResidualRmsNormArgs::new(
                    &input,
                    &residual,
                    &style,
                    &gate,
                    &mut hidden,
                    &mut normalized,
                    eps,
                ),
            )
            .unwrap();
            Ok((read(&ctx, &hidden), read(&ctx, &normalized)))
        })
        .unwrap();
    assert_eq!(hidden, expected_hidden, "ada_gate_residual_rms_norm hidden");
    assert_eq!(
        normalized, expected_normalized,
        "ada_gate_residual_rms_norm normalized"
    );
}

/// The decode attention reads a KV cache the traversal only partly wrote: the
/// prefix filled rows 0..n, and each step fills the rows after them. A kernel
/// that reads past `key_tokens` — a padded tile, a rounded capacity — would
/// read whatever the arena holds there, which is nothing the traversal wrote.
#[test]
#[ignore = "needs a MUSA device"]
fn decode_attention_reads_only_the_rows_the_traversal_wrote() {
    let ctx = MusaContext::new(0).unwrap();
    let device = ctx.device_id();
    let (queries, keys, heads, dim) = (10, 788, 8, 256);

    let q_host = Tensor::from_bf16(
        vec![1, queries, heads, dim],
        &vec![half::bf16::from_f32(0.0); queries * heads * dim],
    )
    .unwrap();
    let k_host = upload(&ctx, &[keys, dim], 31);
    let v_host = upload(&ctx, &[keys, dim], 32);
    let q = transfers::to_device(&q_host, device).unwrap();

    let mut driver_out = output(&ctx, &[1, queries, heads, dim]);
    ops::attention(
        &ctx,
        ops::AttentionArgs::new(&q, &k_host, &v_host, &mut driver_out),
    )
    .unwrap();
    let expected = read(&ctx, &driver_out);

    // The same operands from a session arena, with the buffer beyond the cache
    // holding something other than what a capture would leave there.
    let session = apxinf_musa::ExecutionSession::with_capacity(1 << 28, device).unwrap();
    let actual = session
        .prepare(|| {
            let k_bytes = read(&ctx, &k_host);
            let v_bytes = read(&ctx, &v_host);
            let k = ctx
                .allocate_output(Shape::new(vec![keys, dim]), DType::BF16)
                .unwrap();
            apxinf_musa::MusaBuffer::from_tensor(&k)
                .unwrap()
                .copy_from_host(&k_bytes)
                .unwrap();
            let v = ctx
                .allocate_output(Shape::new(vec![keys, dim]), DType::BF16)
                .unwrap();
            apxinf_musa::MusaBuffer::from_tensor(&v)
                .unwrap()
                .copy_from_host(&v_bytes)
                .unwrap();
            let mut out = output(&ctx, &[1, queries, heads, dim]);
            ops::attention(&ctx, ops::AttentionArgs::new(&q, &k, &v, &mut out)).unwrap();
            Ok(read(&ctx, &out))
        })
        .unwrap();
    assert_eq!(actual, expected, "decode attention over an arena-resident cache");
}

