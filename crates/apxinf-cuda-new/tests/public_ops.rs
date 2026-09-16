//! Public-API integration tests from an external executor's point of view.
//!
//! Adding an L3 operator normally does not require editing this file. Extend it
//! only when the operator changes the public preparation, capture, replay,
//! stream, or lifetime contract exposed to downstream crates.

use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda::{
    capture,
    ops::{
        gemm, gemm_bias_gelu, gemm_geglu, prepare_with_session, with_session, ExecutionSession,
        GemmArgs, GemmBiasGeluArgs, GemmGegluArgs,
    },
    CudaBuffer, CudaContext,
};
use half::bf16;

fn bf16_tensor(device: usize, shape: Vec<usize>, values: &[f32]) -> Tensor {
    let words: Vec<u16> = values
        .iter()
        .map(|&value| bf16::from_f32(value).to_bits())
        .collect();
    let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_ne_bytes()).collect();
    let buffer = CudaBuffer::alloc(bytes.len(), device).unwrap();
    buffer.copy_from_host(&bytes).unwrap();
    buffer.as_tensor(Shape::new(shape), DType::BF16).unwrap()
}

fn raw_tensor(device: usize, shape: Vec<usize>, dtype: DType, bytes: &[u8]) -> Tensor {
    let buffer = CudaBuffer::alloc(bytes.len(), device).unwrap();
    buffer.copy_from_host(bytes).unwrap();
    buffer.as_tensor(Shape::new(shape), dtype).unwrap()
}

fn f16_values(tensor: &Tensor) -> Vec<f32> {
    let buffer = CudaBuffer::from_tensor(tensor).unwrap();
    let mut bytes = vec![0; buffer.len()];
    buffer.copy_to_host(&mut bytes).unwrap();
    bytes
        .chunks_exact(2)
        .map(|value| half::f16::from_bits(u16::from_ne_bytes([value[0], value[1]])).to_f32())
        .collect()
}

fn values(tensor: &Tensor) -> Vec<f32> {
    let buffer = CudaBuffer::from_tensor(tensor).unwrap();
    let mut bytes = vec![0; buffer.len()];
    buffer.copy_to_host(&mut bytes).unwrap();
    bytes
        .chunks_exact(2)
        .map(|bytes| bf16::from_bits(u16::from_ne_bytes([bytes[0], bytes[1]])).to_f32())
        .collect()
}

fn run_gemm(
    ctx: &CudaContext,
    a: &Tensor,
    b: &Tensor,
    out: &mut Tensor,
) -> apxinf_core::Result<()> {
    let mut args = GemmArgs::new(a, b, out);
    args.policy.online_tune = false;
    gemm(ctx, args)
}

#[test]
fn public_nvfp4_gemm_contract_is_callable() {
    let ctx = CudaContext::new(0).unwrap();
    let (m, n, k) = (16, 256, 128);
    let a = raw_tensor(0, vec![m, k], DType::F4E2M1, &vec![0x22; m * k / 2]);
    let b = raw_tensor(0, vec![k, n], DType::F4E2M1, &vec![0x22; n * k / 2]);
    let scale_bytes = |rows: usize| rows * (k / 16);
    let a_scales = raw_tensor(
        0,
        vec![m, k / 16],
        DType::F8UE4M3,
        &vec![0x38; scale_bytes(m)],
    );
    let b_scales = raw_tensor(
        0,
        vec![n, k / 16],
        DType::F8UE4M3,
        &vec![0x38; scale_bytes(n)],
    );
    let mut out = raw_tensor(0, vec![m, n], DType::F16, &vec![0; m * n * 2]);

    let mut args = GemmArgs::nvfp4(&a, &a_scales, &b, &b_scales, &mut out);
    args.policy.online_tune = false;
    gemm(&ctx, args).unwrap();
    assert!(f16_values(&out).iter().all(|&value| value == k as f32));
}

#[test]
fn public_nvfp4_fused_contracts_are_callable() {
    let ctx = CudaContext::new(0).unwrap();
    let (m, n, k) = (16, 256, 128);
    let a = raw_tensor(0, vec![m, k], DType::F4E2M1, &vec![0x22; m * k / 2]);
    let b = raw_tensor(0, vec![k, n], DType::F4E2M1, &vec![0x22; k * n / 2]);
    let b_geglu = raw_tensor(0, vec![k, 2 * n], DType::F4E2M1, &vec![0x22; k * n]);
    let a_scales = raw_tensor(
        0,
        vec![m, k / 16],
        DType::F8UE4M3,
        &vec![0x38; m * (k / 16)],
    );
    let b_scales = raw_tensor(
        0,
        vec![n, k / 16],
        DType::F8UE4M3,
        &vec![0x38; n * (k / 16)],
    );
    let b_geglu_scales = raw_tensor(
        0,
        vec![2 * n, k / 16],
        DType::F8UE4M3,
        &vec![0x38; 2 * n * (k / 16)],
    );
    let bias = raw_tensor(0, vec![n], DType::F16, &vec![0; n * 2]);
    let mut gelu_out = raw_tensor(0, vec![m, n], DType::F16, &vec![0; m * n * 2]);
    let mut geglu_out = raw_tensor(0, vec![m, n], DType::F16, &vec![0; m * n * 2]);

    let mut gelu_args = GemmArgs::nvfp4(&a, &a_scales, &b, &b_scales, &mut gelu_out);
    gelu_args.policy.online_tune = false;
    gemm_bias_gelu(
        &ctx,
        GemmBiasGeluArgs {
            gemm: gelu_args,
            bias: &bias,
        },
    )
    .unwrap();
    let mut geglu_args = GemmArgs::nvfp4(&a, &a_scales, &b_geglu, &b_geglu_scales, &mut geglu_out);
    geglu_args.policy.online_tune = false;
    gemm_geglu(&ctx, GemmGegluArgs { gemm: geglu_args }).unwrap();

    assert!(f16_values(&gelu_out).iter().all(|&value| value == 128.0));
    assert!(f16_values(&geglu_out)
        .iter()
        .all(|&value| value == 16_384.0));
}

#[test]
fn same_public_l3_forward_prepares_captures_and_replays() {
    let ctx = CudaContext::new(0).unwrap();
    let a = bf16_tensor(0, vec![2, 3], &[1.0; 6]);
    let b = bf16_tensor(0, vec![3, 4], &[1.0; 12]);
    let mut out = bf16_tensor(0, vec![2, 4], &[0.0; 8]);
    let observed = CudaBuffer::from_tensor(&out).unwrap();
    let session = ExecutionSession::with_capacity(4096, 0).unwrap();

    prepare_with_session(&session, || run_gemm(&ctx, &a, &b, &mut out)).unwrap();
    let graph = capture(&ctx, || {
        with_session(&session, || run_gemm(&ctx, &a, &b, &mut out))
    })
    .unwrap();

    drop(session);
    drop(a);
    drop(b);

    let sentinel: Vec<u8> = (0..8)
        .flat_map(|_| bf16::from_f32(-123.0).to_bits().to_ne_bytes())
        .collect();
    observed.copy_from_host(&sentinel).unwrap();
    graph.replay().unwrap();
    ctx.synchronize().unwrap();

    assert!(values(&out).iter().all(|&value| value == 3.0));
    drop(graph);
}

#[test]
fn public_l3_capture_rejects_cached_instance_from_another_stream() {
    let capture_ctx = CudaContext::new(0).unwrap();
    let execution_ctx = CudaContext::new(0).unwrap();
    let a = bf16_tensor(0, vec![2, 3], &[1.0; 6]);
    let b = bf16_tensor(0, vec![3, 4], &[1.0; 12]);
    let mut out = bf16_tensor(0, vec![2, 4], &[0.0; 8]);
    let session = ExecutionSession::with_capacity(4096, 0).unwrap();

    prepare_with_session(&session, || run_gemm(&execution_ctx, &a, &b, &mut out)).unwrap();
    let error = match capture(&capture_ctx, || {
        with_session(&session, || run_gemm(&execution_ctx, &a, &b, &mut out))
    }) {
        Ok(_) => panic!("capture accepted an execution bound to another stream"),
        Err(error) => error,
    };
    assert!(
        error
            .to_string()
            .contains("multi-stream capture is not supported"),
        "unexpected error: {error}"
    );

    // A failed mismatched capture must not poison later capture on the bound
    // stream.
    let graph = capture(&execution_ctx, || {
        with_session(&session, || run_gemm(&execution_ctx, &a, &b, &mut out))
    })
    .unwrap();
    graph.replay().unwrap();
    execution_ctx.synchronize().unwrap();
    assert!(values(&out).iter().all(|&value| value == 3.0));
}

#[test]
fn public_l3_capture_rejects_an_unprepared_cache_miss() {
    let ctx = CudaContext::new(0).unwrap();
    let a = bf16_tensor(0, vec![2, 3], &[1.0; 6]);
    let b = bf16_tensor(0, vec![3, 4], &[1.0; 12]);
    let mut out = bf16_tensor(0, vec![2, 4], &[0.0; 8]);
    let session = ExecutionSession::with_capacity(4096, 0).unwrap();

    let error = match capture(&ctx, || {
        with_session(&session, || run_gemm(&ctx, &a, &b, &mut out))
    }) {
        Ok(_) => panic!("capture unexpectedly prepared a GEMM instance"),
        Err(error) => error,
    };
    assert!(
        error.to_string().contains("cache miss during capture"),
        "unexpected error: {error}"
    );
}

#[test]
fn public_l3_capture_rejects_a_different_operator_order() {
    let ctx = CudaContext::new(0).unwrap();
    let a = bf16_tensor(0, vec![2, 3], &[1.0; 6]);
    let b = bf16_tensor(0, vec![3, 4], &[1.0; 12]);
    let mut first_out = bf16_tensor(0, vec![2, 4], &[0.0; 8]);
    let mut second_out = bf16_tensor(0, vec![2, 4], &[0.0; 8]);
    let session = ExecutionSession::with_capacity(4096, 0).unwrap();

    prepare_with_session(&session, || {
        run_gemm(&ctx, &a, &b, &mut first_out)?;
        run_gemm(&ctx, &a, &b, &mut second_out)
    })
    .unwrap();

    let error = match capture(&ctx, || {
        with_session(&session, || {
            run_gemm(&ctx, &a, &b, &mut second_out)?;
            run_gemm(&ctx, &a, &b, &mut first_out)
        })
    }) {
        Ok(_) => panic!("capture accepted a different operator order"),
        Err(error) => error,
    };
    assert!(
        error.to_string().contains("operator sequence mismatch"),
        "unexpected error: {error}"
    );
}
