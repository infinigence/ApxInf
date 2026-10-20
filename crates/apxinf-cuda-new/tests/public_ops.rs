//! Public-API integration tests from an external executor's point of view.
//!
//! Adding an L3 operator normally does not require editing this file. Extend it
//! only when the operator changes the public preparation, capture, replay,
//! stream, or lifetime contract exposed to downstream crates.

use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda_new::{
    capture,
    ops::{gemm, prepare_with_session, with_session, ExecutionSession, GemmArgs},
    CudaBuffer, CudaContext, PreparedPhase,
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
fn prepared_phase_runs_the_same_forward_for_prepare_and_capture() {
    let ctx = CudaContext::new(0).unwrap();
    let a = bf16_tensor(0, vec![2, 3], &[1.0; 6]);
    let b = bf16_tensor(0, vec![3, 4], &[1.0; 12]);
    let mut out = bf16_tensor(0, vec![2, 4], &[0.0; 8]);
    let observed = CudaBuffer::from_tensor(&out).unwrap();
    let mut traversals = 0;

    let phase = PreparedPhase::prepare_and_capture(
        &ctx,
        ExecutionSession::with_capacity(4096, 0).unwrap(),
        || {
            traversals += 1;
            run_gemm(&ctx, &a, &b, &mut out)
        },
    )
    .unwrap();

    assert_eq!(traversals, 2);
    let sentinel: Vec<u8> = (0..8)
        .flat_map(|_| bf16::from_f32(-123.0).to_bits().to_ne_bytes())
        .collect();
    observed.copy_from_host(&sentinel).unwrap();
    phase.replay().unwrap();
    ctx.synchronize().unwrap();

    assert!(values(&out).iter().all(|&value| value == 3.0));
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

#[test]
fn additive_attention_graph_retains_bias_and_tensor_storage() {
    use apxinf_cuda_new::{
        additive_attention, AttentionArgs, AttentionBiasArgs, AttentionPrecision,
    };
    for precision in [AttentionPrecision::F32, AttentionPrecision::Input] {
        let ctx = CudaContext::new(0).unwrap();
        let query = bf16_tensor(0, vec![1, 1, 1, 2], &[0.0; 2]);
        let key = bf16_tensor(0, vec![1, 2, 1, 2], &[0.0; 4]);
        let value = bf16_tensor(0, vec![1, 2, 1, 2], &[1.0, 2.0, 3.0, 4.0]);
        let bytes: Vec<_> = [0.0f32, f32::NEG_INFINITY]
            .iter()
            .flat_map(|x| x.to_ne_bytes())
            .collect();
        let bias_storage = CudaBuffer::alloc(bytes.len(), 0).unwrap();
        bias_storage.copy_from_host(&bytes).unwrap();
        let bias = bias_storage
            .as_tensor(Shape::new(vec![1, 1, 1, 2]), DType::F32)
            .unwrap();
        let mut out = bf16_tensor(0, vec![1, 1, 1, 2], &[0.0; 2]);
        let observed = out.clone();
        let session = ExecutionSession::with_capacity(4096, 0).unwrap();
        prepare_with_session(&session, || {
            let mut args =
                AttentionBiasArgs::new(AttentionArgs::new(&query, &key, &value, &mut out), &bias);
            args.precision = precision;
            args.attention.policy.online_tune = false;
            additive_attention(&ctx, args)
        })
        .unwrap();
        let graph = capture(&ctx, || {
            with_session(&session, || {
                let mut args = AttentionBiasArgs::new(
                    AttentionArgs::new(&query, &key, &value, &mut out),
                    &bias,
                );
                args.precision = precision;
                args.attention.policy.online_tune = false;
                additive_attention(&ctx, args)
            })
        })
        .unwrap();
        drop(session);
        drop(query);
        drop(key);
        drop(value);
        drop(bias);
        drop(bias_storage);
        drop(out);
        graph.replay().unwrap();
        ctx.synchronize().unwrap();
        assert_eq!(values(&observed), [1.0, 2.0]);
    }
}

#[test]
fn synchronous_host_uploads_are_visible_to_nonblocking_graph_replay() {
    use apxinf_cuda_new::transfers;

    let ctx = CudaContext::new(0).unwrap();
    // Use pageable host allocations large enough to exercise staged DMA.
    let count = 1 << 20;
    let source =
        transfers::to_cuda(&Tensor::from_i32(vec![count], &vec![0; count]).unwrap(), 0).unwrap();
    let source_buffer = CudaBuffer::from_tensor(&source).unwrap();
    let output = ctx
        .allocate_output(Shape::new(vec![count]), DType::I32)
        .unwrap();
    let graph = capture(&ctx, || {
        transfers::copy_cuda_to_cuda(&ctx, &source, &output)
    })
    .unwrap();
    let output_buffer = CudaBuffer::from_tensor(&output).unwrap();
    let mut actual = vec![0; count * 4];

    for iteration in 0..4 {
        let values = (0..count)
            .map(|i| (i as i32) ^ ((iteration + 1) << 20))
            .collect::<Vec<_>>();
        let host = Tensor::from_i32(vec![count], &values).unwrap();
        let expected = host.storage().as_cpu().unwrap();
        if iteration % 2 == 0 {
            source_buffer.copy_from_host(expected).unwrap();
        } else {
            transfers::copy_cpu_to_cuda(&host, &source).unwrap();
        }
        // No caller-side synchronization between upload and launch: both
        // synchronous upload APIs promise completion before they return.
        graph.replay().unwrap();
        ctx.synchronize().unwrap();
        output_buffer.copy_to_host(&mut actual).unwrap();
        assert_eq!(actual, expected, "upload/replay iteration {iteration}");
    }
}

#[test]
fn cublas_shared_stream_resources_survive_graph_owners_and_isolate_replay() {
    use apxinf_cuda_new::{transfers, CapturedGraph};

    fn make_graph(scale: f32) -> (CapturedGraph, Vec<Tensor>) {
        let ctx = CudaContext::new_with_autotune(0, false).unwrap();
        let inputs = (1..=32)
            .map(|i| bf16_tensor(0, vec![4, 8], &[scale * i as f32 / 8.0; 32]))
            .collect::<Vec<_>>();
        let weight = bf16_tensor(0, vec![8, 6], &[0.25; 48]);
        let mut outputs = (0..32)
            .map(|_| bf16_tensor(0, vec![4, 6], &[0.; 24]))
            .collect::<Vec<_>>();
        let session = ExecutionSession::with_capacity(4096, 0).unwrap();
        let mut operation = || {
            for (input, output) in inputs.iter().zip(&mut outputs) {
                let mut args = GemmArgs::new(input, &weight, output);
                args.policy.online_tune = false;
                args.policy.cache_dir = Some(String::new());
                gemm(&ctx, args)?;
            }
            Ok(())
        };
        prepare_with_session(&session, &mut operation).unwrap();
        let graph = capture(&ctx, || with_session(&session, &mut operation)).unwrap();
        // The returned graph must retain the original execution, weights,
        // inputs, shared cuBLAS state and stream after these owners disappear.
        (graph, outputs)
    }

    let (first, first_outputs) = make_graph(1.0);
    let (second, second_outputs) = make_graph(-2.0);
    for _ in 0..8 {
        first.replay().unwrap();
        second.replay().unwrap();
    }
    for (scale, outputs) in [(1.0, &first_outputs), (-2.0, &second_outputs)] {
        for (i, output) in outputs.iter().enumerate() {
            let actual = transfers::to_cpu(output).unwrap().to_f32_vec().unwrap();
            assert!(actual.iter().all(|&x| x == scale * (i + 1) as f32 / 4.0));
        }
    }
    drop(first);
    second.replay().unwrap();
    assert!(transfers::to_cpu(&second_outputs[31])
        .unwrap()
        .to_f32_vec()
        .unwrap()
        .iter()
        .all(|&x| x == -16.0));
}
