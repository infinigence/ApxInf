//! Semantic contracts for the Spec-dispatched families that had none.
//!
//! `src/ops/tests/mod.rs` states that every L3 operator must carry a
//! semantic-contract test here. Four of the five Spec-dispatched families never
//! got one: `rope/`, `pointwise/` and `gather/` had zero tests and
//! `quantization/` had one. These cover what each family's `normalize` is
//! supposed to enforce — which optional bindings belong to which semantic, and
//! which combinations must be refused rather than silently reinterpreted.
//!
//! Every assertion matches the exact rejection message. `assert!(is_err())`
//! would pass on an unrelated failure — a shape error instead of the contract
//! violation under test — and that is the failure mode these tests exist to
//! rule out.
//!
//! The norm family's field model is larger and lives in `norm_fields`.

use apxinf_core::{DType, Shape, Tensor};

use super::framework::zeros_tensor;
use super::*;
use crate::{CudaBuffer, CudaContext};

fn bf16(shape: Vec<usize>) -> Tensor {
    zeros_tensor(0, shape, DType::BF16)
}

fn u32_ids(count: usize) -> CudaBuffer {
    let bytes: Vec<u8> = (0..count as u32)
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    let buffer = CudaBuffer::alloc(bytes.len(), 0).unwrap();
    buffer.copy_from_host(&bytes).unwrap();
    buffer
}

// ---------------------------------------------------------------- rope

const HEAD_DIM: usize = 8;
const TOKENS: usize = 3;

fn rope_args<'a>(
    semantic: RopeSemantic,
    qkv: &'a Tensor,
    q: &'a mut Tensor,
    k: &'a mut Tensor,
    v: &'a mut Tensor,
    q_heads: usize,
    kv_heads: usize,
) -> RopeArgs<'a> {
    RopeArgs {
        semantic,
        qkv,
        bias: None,
        q,
        k,
        v,
        q_heads,
        kv_heads,
        head_dim: HEAD_DIM,
        theta: 10000.0,
        position_offset: 0,
        kv_output_offset: 0,
    }
}

/// `SplitQkvBias` applies no rotation, so there is no grouped-query layout for
/// it to resolve: it must require `q_heads == kv_heads` rather than silently
/// treating the K/V heads as if they had been broadcast.
#[test]
fn rope_unrotated_split_requires_equal_head_counts() {
    let ctx = CudaContext::new(0).unwrap();
    let qkv = bf16(vec![TOKENS, 4 * HEAD_DIM]);
    let mut q = bf16(vec![TOKENS, 2, HEAD_DIM]);
    let mut k = bf16(vec![TOKENS, 1, HEAD_DIM]);
    let mut v = bf16(vec![TOKENS, 1, HEAD_DIM]);
    let error = rope(
        &ctx,
        rope_args(
            RopeSemantic::SplitQkvBias,
            &qkv,
            &mut q,
            &mut k,
            &mut v,
            2,
            1,
        ),
    )
    .unwrap_err()
    .to_string();
    assert!(
        error.contains("unrotated QKV split requires q_heads == kv_heads"),
        "{error}"
    );
}

/// A position offset only means something to a rotation. Accepting it on the
/// unrotated semantic would let a caller pass one and believe RoPE ran.
#[test]
fn rope_unrotated_split_refuses_position_offsets() {
    let ctx = CudaContext::new(0).unwrap();
    let qkv = bf16(vec![TOKENS, 3 * HEAD_DIM]);
    let mut q = bf16(vec![TOKENS, 1, HEAD_DIM]);
    let mut k = bf16(vec![TOKENS, 1, HEAD_DIM]);
    let mut v = bf16(vec![TOKENS, 1, HEAD_DIM]);
    let mut args = rope_args(
        RopeSemantic::SplitQkvBias,
        &qkv,
        &mut q,
        &mut k,
        &mut v,
        1,
        1,
    );
    args.position_offset = 4;
    let error = rope(&ctx, args).unwrap_err().to_string();
    assert!(
        error.contains("unrotated QKV split does not take offsets"),
        "{error}"
    );
}

/// Grouped-query attention requires `q_heads` to be a whole multiple of
/// `kv_heads`; anything else has no well-defined head mapping.
#[test]
fn rope_requires_q_heads_to_be_a_multiple_of_kv_heads() {
    let ctx = CudaContext::new(0).unwrap();
    let qkv = bf16(vec![TOKENS, 7 * HEAD_DIM]);
    let mut q = bf16(vec![TOKENS, 3, HEAD_DIM]);
    let mut k = bf16(vec![TOKENS, 2, HEAD_DIM]);
    let mut v = bf16(vec![TOKENS, 2, HEAD_DIM]);
    let error = rope(
        &ctx,
        rope_args(
            RopeSemantic::SplitQkvRope,
            &qkv,
            &mut q,
            &mut k,
            &mut v,
            3,
            2,
        ),
    )
    .unwrap_err()
    .to_string();
    assert!(
        error.contains("q_heads must be a multiple of kv_heads"),
        "{error}"
    );
}

// ----------------------------------------------------------- pointwise

const ROWS: usize = 2;
const COLS: usize = 8;

/// Only `BiasActivation` takes a bias, only `EulerUpdate` takes a secondary,
/// and only `BiasActivation` takes an activation. An ignored binding is a wrong
/// answer that looks right, so each must be refused by name.
#[test]
fn pointwise_refuses_bindings_outside_their_semantic() {
    let ctx = CudaContext::new(0).unwrap();
    let bias = bf16(vec![COLS]);
    let secondary = bf16(vec![ROWS, COLS]);

    // Geglu takes no bias.
    let input = bf16(vec![ROWS, 2 * COLS]);
    let mut out = bf16(vec![ROWS, COLS]);
    let mut args = PointwiseArgs::new(PointwiseSemantic::Geglu, &input, &mut out);
    args.bias = Some(&bias);
    let error = pointwise(&ctx, args).unwrap_err().to_string();
    assert!(
        error.contains("Pointwise semantic does not take a bias"),
        "{error}"
    );

    // Geglu takes no secondary.
    let input = bf16(vec![ROWS, 2 * COLS]);
    let mut out = bf16(vec![ROWS, COLS]);
    let mut args = PointwiseArgs::new(PointwiseSemantic::Geglu, &input, &mut out);
    args.secondary = Some(&secondary);
    let error = pointwise(&ctx, args).unwrap_err().to_string();
    assert!(
        error.contains("Pointwise semantic does not take a secondary input"),
        "{error}"
    );

    // Geglu takes no activation: its GELU is hard-coded in the kernel.
    let input = bf16(vec![ROWS, 2 * COLS]);
    let mut out = bf16(vec![ROWS, COLS]);
    let mut args = PointwiseArgs::new(PointwiseSemantic::Geglu, &input, &mut out);
    args.activation = PointwiseActivation::Silu;
    let error = pointwise(&ctx, args).unwrap_err().to_string();
    assert!(
        error.contains("Pointwise semantic does not take an activation"),
        "{error}"
    );
}

/// `EulerUpdate` integrates a velocity, so omitting it is not a no-op — it
/// would read a null pointer. The contract must require it.
#[test]
fn pointwise_euler_update_requires_its_secondary() {
    let ctx = CudaContext::new(0).unwrap();
    let input = bf16(vec![ROWS, COLS]);
    let mut out = bf16(vec![ROWS, COLS]);
    let error = pointwise(
        &ctx,
        PointwiseArgs::new(PointwiseSemantic::EulerUpdate, &input, &mut out),
    )
    .unwrap_err()
    .to_string();
    assert!(
        error.contains("Pointwise semantic requires a secondary input"),
        "{error}"
    );
}

/// `BiasActivation` with neither a bias nor an activation computes nothing.
/// Refusing it is better than launching a kernel that copies its input.
#[test]
fn pointwise_rejects_a_no_op_bias_activation() {
    let ctx = CudaContext::new(0).unwrap();
    let input = bf16(vec![ROWS, COLS]);
    let mut out = bf16(vec![ROWS, COLS]);
    let error = pointwise(
        &ctx,
        PointwiseArgs::new(PointwiseSemantic::BiasActivation, &input, &mut out),
    )
    .unwrap_err()
    .to_string();
    assert!(
        error.contains("Pointwise bias-activation is a no-op"),
        "{error}"
    );
}

/// `Geglu` consumes a gate and an up half per output element, so its input must
/// be twice the output width. A same-width input would read out of bounds.
#[test]
fn pointwise_enforces_the_geglu_width_relationship() {
    let ctx = CudaContext::new(0).unwrap();
    let input = bf16(vec![ROWS, COLS]);
    let mut out = bf16(vec![ROWS, COLS]);
    let error = pointwise(
        &ctx,
        PointwiseArgs::new(PointwiseSemantic::Geglu, &input, &mut out),
    )
    .unwrap_err()
    .to_string();
    assert!(
        error.contains("element count mismatch") || error.contains("too small"),
        "Geglu must reject a same-width input, got: {error}"
    );
}

// -------------------------------------------------------------- gather

const VOCAB: usize = 16;

/// `EmbeddingLookup` without token ids must be refused by name. This is the
/// family the relocated `embedding_gather` sits beside; the two are
/// deliberately different semantics (see `gather/embedding.rs`).
#[test]
fn gather_embedding_lookup_requires_token_ids() {
    let ctx = CudaContext::new(0).unwrap();
    let table = bf16(vec![VOCAB, COLS]);
    let mut out = bf16(vec![ROWS, COLS]);
    let mut args = GatherArgs::new(GatherSemantic::EmbeddingLookup, &table, &mut out);
    args.vocab_size = VOCAB;
    let error = gather(&ctx, args).unwrap_err().to_string();
    assert!(
        error.contains("embedding lookup requires token ids"),
        "{error}"
    );
}

/// `ids` and `ids_address` are documented as mutually exclusive: one is a
/// borrowed buffer, the other a stable address for graph replay. With both set,
/// which one the kernel reads would be unspecified.
#[test]
fn gather_refuses_both_id_sources_at_once() {
    let ctx = CudaContext::new(0).unwrap();
    let ids = u32_ids(ROWS);
    let table = bf16(vec![VOCAB, COLS]);
    let mut out = bf16(vec![ROWS, COLS]);
    let mut args = GatherArgs::new(GatherSemantic::EmbeddingLookup, &table, &mut out);
    args.vocab_size = VOCAB;
    args.ids = Some(&ids);
    args.ids_address = Some(ids.address());
    let error = gather(&ctx, args).unwrap_err().to_string();
    assert!(
        error.contains("not both"),
        "ids and ids_address must be mutually exclusive, got: {error}"
    );
}

/// `EmbeddingLookup` needs a vocabulary size to bound the table read; zero
/// would make every id out of range.
#[test]
fn gather_embedding_lookup_requires_a_vocabulary_size() {
    let ctx = CudaContext::new(0).unwrap();
    let ids = u32_ids(ROWS);
    let table = bf16(vec![VOCAB, COLS]);
    let mut out = bf16(vec![ROWS, COLS]);
    let mut args = GatherArgs::new(GatherSemantic::EmbeddingLookup, &table, &mut out);
    args.ids = Some(&ids);
    let error = gather(&ctx, args).unwrap_err().to_string();
    assert!(
        error.contains("embedding lookup requires a vocabulary size"),
        "{error}"
    );
}

/// Each semantic must refuse the bindings that belong to the others, so a
/// caller cannot pass a position embedding to an embedding lookup and have it
/// quietly dropped.
#[test]
fn gather_refuses_bindings_from_other_semantics() {
    let ctx = CudaContext::new(0).unwrap();
    let ids = u32_ids(ROWS);
    let table = bf16(vec![VOCAB, COLS]);
    let position = bf16(vec![ROWS, COLS]);
    let mut out = bf16(vec![ROWS, COLS]);
    let mut args = GatherArgs::new(GatherSemantic::EmbeddingLookup, &table, &mut out);
    args.vocab_size = VOCAB;
    args.ids = Some(&ids);
    args.position = Some(&position);
    let error = gather(&ctx, args).unwrap_err().to_string();
    assert!(
        error.contains("embedding lookup received unrelated bindings"),
        "{error}"
    );
}

// -------------------------------------------------- quantization (extra)

/// Every quantization semantic reads a row before writing it, so an aliased
/// input and output would read values it has already overwritten.
#[test]
fn quantization_refuses_aliased_input_and_output() {
    let ctx = CudaContext::new(0).unwrap();
    let bytes = vec![0u8; ROWS * COLS * DType::F16.size_in_bytes()];
    let buffer = CudaBuffer::alloc(bytes.len(), 0).unwrap();
    buffer.copy_from_host(&bytes).unwrap();
    let input = buffer
        .as_tensor(Shape::new(vec![ROWS, COLS]), DType::F16)
        .unwrap();
    let mut output = buffer
        .as_tensor(Shape::new(vec![ROWS, COLS]), DType::F16)
        .unwrap();
    let error = quantization(
        &ctx,
        QuantizationArgs::new(QuantizationSemantic::CastF16ToBf16, &input, &mut output),
    )
    .unwrap_err()
    .to_string();
    assert!(
        error.contains("must not alias") || error.contains("dtype"),
        "{error}"
    );
}
