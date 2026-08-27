use half::bf16;

use apxinf_core::{Shape, Tensor};

use crate::kernels::qwen35_gdn::{
    drain_deferred_status, gated_rms_norm_bf16, require_finite_bf16, Qwen35GdnLayout,
    Qwen35GdnState,
};
use crate::test_util::{download_bf16_as_fp32, upload_fp32_as_bf16};
use crate::{CudaBuffer, CudaContext};

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let delta = (actual - expected).abs();
        assert!(
            delta <= tolerance,
            "index {index}: actual={actual} expected={expected} delta={delta} tolerance={tolerance}"
        );
    }
}

fn tiny_layout() -> Qwen35GdnLayout {
    Qwen35GdnLayout::new(4, 1, 1, 2, 2).unwrap()
}

fn upload_weight(ctx: &CudaContext, values: &[f32], shape: Vec<usize>) -> Tensor {
    let rounded = values
        .iter()
        .copied()
        .map(bf16::from_f32)
        .collect::<Vec<_>>();
    let cpu = Tensor::from_bf16(Shape::new(shape), &rounded).unwrap();
    crate::transfers::to_cuda(&cpu, ctx.device_id()).unwrap()
}

#[test]
fn qwen35_cuda_finite_check_rejects_non_finite_bf16() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let finite = upload_fp32_as_bf16(&ctx, &[1.0, -2.0], vec![1, 2]).unwrap();
    require_finite_bf16(&ctx, &finite, "full-attention prefill output").unwrap();

    let non_finite = upload_fp32_as_bf16(&ctx, &[1.0, f32::NAN], vec![1, 2]).unwrap();
    let error =
        require_finite_bf16(&ctx, &non_finite, "full-attention prefill output").unwrap_err();
    assert!(error
        .to_string()
        .contains("full-attention prefill output contains a non-finite value"));
}

/// Restores the deferred-status env var on drop so a panicking assertion
/// cannot leak deferred mode into other tests. The GDN GPU suite is run with
/// `--test-threads=1`, which this test relies on like the other
/// state-mutating tests here.
struct DeferredStatusGuard;

impl DeferredStatusGuard {
    fn enable() -> Self {
        std::env::set_var("APXINF_Q35_DEFERRED_STATUS", "1");
        Self
    }
}

impl Drop for DeferredStatusGuard {
    fn drop(&mut self) {
        std::env::remove_var("APXINF_Q35_DEFERRED_STATUS");
    }
}

#[test]
fn qwen35_gdn_cuda_deferred_status_matches_eager_and_latches_non_finite() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let layout = tiny_layout();
    let weights = upload_weight(
        &ctx,
        &vec![1.0; layout.conv_channels() * layout.conv_kernel],
        vec![layout.conv_channels(), 1, layout.conv_kernel],
    );
    let run_conv_sequence = |state: &mut Qwen35GdnState| -> Vec<Vec<f32>> {
        (1..=4)
            .map(|token| {
                let input = upload_fp32_as_bf16(
                    &ctx,
                    &vec![token as f32; layout.conv_channels()],
                    vec![1, layout.conv_channels()],
                )
                .unwrap();
                download_bf16_as_fp32(&state.causal_conv_silu(&ctx, &input, &weights).unwrap())
                    .unwrap()
            })
            .collect()
    };

    let mut eager_state = Qwen35GdnState::new(&ctx, layout).unwrap();
    let eager_outputs = run_conv_sequence(&mut eager_state);

    let guard = DeferredStatusGuard::enable();
    let mut deferred_state = Qwen35GdnState::new(&ctx, layout).unwrap();
    let deferred_outputs = run_conv_sequence(&mut deferred_state);
    // Same kernels in the same order: outputs must be bit-identical, and a
    // clean run must drain without error.
    assert_eq!(eager_outputs, deferred_outputs);
    drain_deferred_status(&ctx, "clean deferred conv run").unwrap();

    // A non-finite tensor is latched silently in deferred mode...
    let non_finite = upload_fp32_as_bf16(&ctx, &[1.0, f32::NAN], vec![1, 2]).unwrap();
    require_finite_bf16(&ctx, &non_finite, "deferred finite check").unwrap();
    // ...surfaces at the next drain...
    let error = drain_deferred_status(&ctx, "deferred finite check").unwrap_err();
    assert!(
        error.to_string().contains("non-finite"),
        "unexpected drain error: {error}"
    );
    // ...and the latch is cleared for the next request.
    drain_deferred_status(&ctx, "post-drain state").unwrap();
    drop(guard);

    // Back in eager mode the same tensor fails immediately again.
    let error = require_finite_bf16(&ctx, &non_finite, "eager finite check").unwrap_err();
    assert!(error
        .to_string()
        .contains("eager finite check contains a non-finite value"));
}

#[test]
fn qwen35_gdn_cuda_causal_depthwise_convolution_tracks_four_token_ring() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let layout = tiny_layout();
    let mut state = Qwen35GdnState::new(&ctx, layout).unwrap();
    let weights = upload_weight(
        &ctx,
        &vec![1.0; layout.conv_channels() * layout.conv_kernel],
        vec![layout.conv_channels(), 1, layout.conv_kernel],
    );

    let mut output = Vec::new();
    for token in 1..=5 {
        let input = upload_fp32_as_bf16(
            &ctx,
            &vec![token as f32; layout.conv_channels()],
            vec![1, layout.conv_channels()],
        )
        .unwrap();
        output = download_bf16_as_fp32(&state.causal_conv_silu(&ctx, &input, &weights).unwrap())
            .unwrap();
    }

    assert_eq!(state.conv_cursor(), 1);
    assert_eq!(state.position(), 5);
    assert_close(
        &state.conv_ring_channel_host(&ctx, 0).unwrap(),
        &[2.0, 3.0, 4.0, 5.0],
        0.0,
    );
    let expected = 14.0 / (1.0 + (-14.0f32).exp());
    assert_close(&output, &vec![expected; layout.conv_channels()], 0.03125);
}

#[test]
fn qwen35_gdn_cuda_convolution_commit_can_be_rolled_back() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let layout = tiny_layout();
    let mut state = Qwen35GdnState::new(&ctx, layout).unwrap();
    let weights = upload_weight(
        &ctx,
        &vec![1.0; layout.conv_channels() * layout.conv_kernel],
        vec![layout.conv_channels(), 1, layout.conv_kernel],
    );
    let input = upload_fp32_as_bf16(
        &ctx,
        &vec![1.0; layout.conv_channels()],
        vec![1, layout.conv_channels()],
    )
    .unwrap();

    state.causal_conv_silu(&ctx, &input, &weights).unwrap();
    state.rollback_last_convolution().unwrap();

    assert_eq!(state.position(), 0);
    assert_eq!(state.conv_cursor(), 0);
    assert_eq!(
        state.conv_ring_channel_host(&ctx, 0).unwrap(),
        vec![0.0; layout.conv_kernel]
    );
}

#[test]
fn qwen35_gdn_cuda_prefill_convolution_matches_transformers_zero_left_padding() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let layout = Qwen35GdnLayout::new(4, 1, 1, 2, 2).unwrap();
    let mut state = Qwen35GdnState::new(&ctx, layout).unwrap();
    let weights = upload_weight(
        &ctx,
        &vec![1.0; layout.conv_channels() * layout.conv_kernel],
        vec![layout.conv_channels(), 1, layout.conv_kernel],
    );
    let input = upload_fp32_as_bf16(
        &ctx,
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // token 0
            2.0, 4.0, 6.0, 8.0, 10.0, 12.0, // token 1
            3.0, 6.0, 9.0, 12.0, 15.0, 18.0, // token 2
        ],
        vec![3, layout.conv_channels()],
    )
    .unwrap();

    let output = state
        .causal_conv_silu_prefill(&ctx, &input, &weights)
        .unwrap();
    let output = download_bf16_as_fp32(&output).unwrap();
    let expected = [
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 6.0, 12.0, 18.0, 24.0,
        30.0, 36.0,
    ]
    .iter()
    .map(|value| value * (1.0 + (-value).exp()).recip())
    .collect::<Vec<_>>();
    assert_close(&output, &expected, 0.03125);
    assert_eq!(state.position(), 3);
    assert_eq!(state.conv_cursor(), 3);
    assert_close(
        &state.conv_ring_channel_host(&ctx, 0).unwrap(),
        &[0.0, 1.0, 2.0, 3.0],
        0.0,
    );
}

#[test]
fn qwen35_gdn_cuda_fp32_recurrent_update_matches_reference_equation() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let layout = tiny_layout();
    let mut state = Qwen35GdnState::new(&ctx, layout).unwrap();
    let query = upload_fp32_as_bf16(&ctx, &[1.0, 0.0], vec![1, 2]).unwrap();
    let key = upload_fp32_as_bf16(&ctx, &[1.0, 0.0], vec![1, 2]).unwrap();
    let value = upload_fp32_as_bf16(&ctx, &[2.0, 4.0], vec![1, 2]).unwrap();
    let a = upload_fp32_as_bf16(&ctx, &[0.0], vec![1, 1]).unwrap();
    let b = upload_fp32_as_bf16(&ctx, &[0.0], vec![1, 1]).unwrap();
    let a_log = upload_fp32_as_bf16(&ctx, &[0.0], vec![1]).unwrap();
    let dt_bias = upload_fp32_as_bf16(&ctx, &[0.0], vec![1]).unwrap();

    let output = state
        .gated_delta_step(&ctx, &query, &key, &value, &a, &b, &a_log, &dt_bias)
        .unwrap();
    let output = download_bf16_as_fp32(&output).unwrap();
    let normalized = (1.0 + 1e-6f32).sqrt().recip();
    let expected_scale = normalized * normalized / 2.0f32.sqrt();
    assert_close(&output, &[1.0 * expected_scale, 2.0 * expected_scale], 0.01);
    assert_eq!(state.recurrent_dtype(), "f32");

    let recurrent = state.recurrent_host(&ctx).unwrap();
    assert_close(&recurrent, &[normalized, 2.0 * normalized, 0.0, 0.0], 1e-5);
}

#[test]
fn qwen35_gdn_cuda_recurrent_matches_transformers_bf16_qk_and_beta_boundaries() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let layout = Qwen35GdnLayout::new(4, 1, 1, 2, 2).unwrap();
    let mut state = Qwen35GdnState::new(&ctx, layout).unwrap();
    let query = upload_fp32_as_bf16(&ctx, &[1.2969, 0.6992], vec![1, 2]).unwrap();
    let key = upload_fp32_as_bf16(&ctx, &[0.8984, -0.4004], vec![1, 2]).unwrap();
    let value = upload_fp32_as_bf16(&ctx, &[2.0938, 0.0], vec![1, 2]).unwrap();
    let a = upload_fp32_as_bf16(&ctx, &[0.1], vec![1]).unwrap();
    let b = upload_fp32_as_bf16(&ctx, &[0.3], vec![1]).unwrap();
    let a_log = upload_fp32_as_bf16(&ctx, &[1.2], vec![1]).unwrap();
    let dt_bias = upload_fp32_as_bf16(&ctx, &[1.0], vec![1]).unwrap();

    let output = state
        .gated_delta_step(&ctx, &query, &key, &value, &a, &b, &a_log, &dt_bias)
        .unwrap();

    // Transformers 5.15.1's torch_recurrent_gated_delta_rule normalizes q/k
    // in BF16 and materializes sigmoid(b) in BF16 before its FP32 recurrence.
    assert_eq!(
        download_bf16_as_fp32(&output).unwrap(),
        vec![0.5234375, 0.0]
    );
}

#[test]
fn qwen35_gdn_cuda_sequence_recurrent_matches_transformers_chunk_fixture() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let layout = tiny_layout();
    let mut state = Qwen35GdnState::new(&ctx, layout).unwrap();
    let query = upload_fp32_as_bf16(&ctx, &[1.2969, 0.6992, -0.75, 1.125], vec![2, 2]).unwrap();
    let key = upload_fp32_as_bf16(&ctx, &[0.8984, -0.4004, 1.5, -0.25], vec![2, 2]).unwrap();
    let value = upload_fp32_as_bf16(&ctx, &[2.0938, 0.0, -1.25, 0.75], vec![2, 2]).unwrap();
    let a = upload_fp32_as_bf16(&ctx, &[0.1, -0.2], vec![2, 1]).unwrap();
    let b = upload_fp32_as_bf16(&ctx, &[0.3, -0.7], vec![2, 1]).unwrap();
    let a_log = upload_fp32_as_bf16(&ctx, &[1.2], vec![1]).unwrap();
    let dt_bias = upload_fp32_as_bf16(&ctx, &[1.0], vec![1]).unwrap();

    let output = state
        .gated_delta_prefill(&ctx, &query, &key, &value, &a, &b, &a_log, &dt_bias)
        .unwrap();
    let output = download_bf16_as_fp32(&output).unwrap();

    // Values are generated from the Transformers torch_chunk_gated_delta_rule
    // reference for this two-row, one-head fixture. The tolerance includes
    // the specified BF16 q/k and beta boundaries, but not a semantic shortcut.
    assert_close(&output, &[0.515625, 0.0, 0.189453125, -0.12011719], 0.01);
    assert_eq!(state.position(), 0);
    assert_eq!(state.recurrent_dtype(), "f32");
    assert_close(
        &state.recurrent_host(&ctx).unwrap(),
        &[-0.39399505, 0.24513245, 0.05948618, -0.04085541],
        0.01,
    );
}

#[test]
fn qwen35_gdn_cuda_sequence_materializes_beta_scaled_operands() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let layout = Qwen35GdnLayout::new(4, 1, 1, 2, 1).unwrap();
    let mut state = Qwen35GdnState::new(&ctx, layout).unwrap();
    let query = upload_fp32_as_bf16(
        &ctx,
        &[-0.96484375, 1.3203125, 1.3359375, 3.046875],
        vec![2, 2],
    )
    .unwrap();
    let key =
        upload_fp32_as_bf16(&ctx, &[1.484375, 4.59375, -3.203125, 5.15625], vec![2, 2]).unwrap();
    let value = upload_fp32_as_bf16(&ctx, &[-34.75, 4.375], vec![2, 1]).unwrap();
    let a = upload_fp32_as_bf16(&ctx, &[-0.65625, 0.796875], vec![2, 1]).unwrap();
    let b = upload_fp32_as_bf16(&ctx, &[-1.84375, 4.0625], vec![2, 1]).unwrap();
    let a_log = upload_fp32_as_bf16(&ctx, &[-1.421875], vec![1]).unwrap();
    let dt_bias = upload_fp32_as_bf16(&ctx, &[-0.419921875], vec![1]).unwrap();

    let output = state
        .gated_delta_prefill(&ctx, &query, &key, &value, &a, &b, &a_log, &dt_bias)
        .unwrap();

    // The Transformers chunk reference first materializes v_beta = value *
    // beta in FP32, then multiplies attn @ v_beta. The resulting recurrent
    // state distinguishes this grouping even when a later BF16 output happens
    // to land on the same reduction-order rounding bin.
    assert_close(
        &state.recurrent_host(&ctx).unwrap(),
        &[-4.7201009, 2.0719726],
        1e-7,
    );
    assert_eq!(download_bf16_as_fp32(&output).unwrap()[0], -1.96875);
}

#[test]
fn qwen35_gdn_cuda_sequence_matches_transformers_at_real_key_head_width() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let rows = 8usize;
    let key_dim = 128usize;
    let value_dim = 8usize;
    let layout = Qwen35GdnLayout::new(4, 1, 1, key_dim, value_dim).unwrap();
    let mut state = Qwen35GdnState::new(&ctx, layout).unwrap();

    let query = (0..rows * key_dim)
        .map(|index| {
            let index = index as f32;
            (index * 0.071).sin() + (index * 0.013).cos() * 0.4
        })
        .collect::<Vec<_>>();
    let key = (0..rows * key_dim)
        .map(|index| {
            let index = index as f32;
            (index * 0.053).cos() - (index * 0.017).sin() * 0.3
        })
        .collect::<Vec<_>>();
    let value = (0..rows * value_dim)
        .map(|index| {
            let index = index as f32;
            (index * 0.11).sin() * 1.3 + (index * 0.023).cos() * 0.2
        })
        .collect::<Vec<_>>();
    let a = (0..rows)
        .map(|index| -1.2 + 2.0 * index as f32 / (rows - 1) as f32)
        .collect::<Vec<_>>();
    let b = (0..rows)
        .map(|index| 0.9 - 1.6 * index as f32 / (rows - 1) as f32)
        .collect::<Vec<_>>();

    let query = upload_fp32_as_bf16(&ctx, &query, vec![rows, key_dim]).unwrap();
    let key = upload_fp32_as_bf16(&ctx, &key, vec![rows, key_dim]).unwrap();
    let value = upload_fp32_as_bf16(&ctx, &value, vec![rows, value_dim]).unwrap();
    let a = upload_fp32_as_bf16(&ctx, &a, vec![rows, 1]).unwrap();
    let b = upload_fp32_as_bf16(&ctx, &b, vec![rows, 1]).unwrap();
    let a_log = upload_fp32_as_bf16(&ctx, &[0.35], vec![1]).unwrap();
    let dt_bias = upload_fp32_as_bf16(&ctx, &[-0.15], vec![1]).unwrap();

    let output = state
        .gated_delta_prefill(&ctx, &query, &key, &value, &a, &b, &a_log, &dt_bias)
        .unwrap();
    let output = download_bf16_as_fp32(&output).unwrap();
    let expected = [
        0.0067749023,
        0.0115356445,
        0.016357422,
        0.020996094,
        0.025512695,
        0.029785156,
        0.033691406,
        0.037353516,
        -0.027709961,
        -0.03100586,
        -0.033935547,
        -0.036376953,
        -0.03881836,
        -0.040527344,
        -0.041992188,
        -0.04296875,
        -0.0035552979,
        -0.00091552734,
        0.0018539429,
        0.004486084,
        0.007293701,
        0.009765625,
        0.012390137,
        0.014709473,
        0.035888672,
        0.030883789,
        0.025634766,
        0.020141602,
        0.014343262,
        0.008605957,
        0.0027008057,
        -0.0030670166,
        -0.008544922,
        -0.0033416748,
        0.0017852783,
        0.0068359375,
        0.01171875,
        0.016357422,
        0.020751953,
        0.024902344,
        0.008728027,
        0.00793457,
        0.0070495605,
        0.0061035156,
        0.005065918,
        0.004058838,
        0.0029449463,
        0.0018157959,
        -0.022460938,
        -0.020751953,
        -0.018798828,
        -0.016601563,
        -0.014221191,
        -0.011657715,
        -0.008911133,
        -0.0060424805,
        0.0067443848,
        0.0039367676,
        0.0010681152,
        -0.0018005371,
        -0.004699707,
        -0.007507324,
        -0.010314941,
        -0.012939453,
    ];

    assert_close(&output, &expected, 0.0);
    let recurrent = state.recurrent_host(&ctx).unwrap();
    let expected_state_samples = [
        (0, 0.013340455),
        (1, 0.006526541),
        (2, -0.0003805412),
        (7, -0.033634707),
        (8, 0.013411971),
        (15, -0.032674395),
        (16, 0.013545751),
        (31, -0.030349972),
        (32, 0.013696043),
        (63, -0.025091635),
        (64, 0.01355628),
        (127, -0.011606095),
        (128, 0.011511902),
        (255, 0.017674187),
        (511, 0.030540597),
        (767, -0.014084241),
        (1023, -0.008191348),
    ];
    for (index, expected) in expected_state_samples {
        assert_close(&recurrent[index..index + 1], &[expected], 1e-10);
    }
}

#[test]
fn qwen35_gdn_cuda_sequence_qk_l2norm_matches_transformers_bf16_boundary() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let layout = tiny_layout();
    let mut state = Qwen35GdnState::new(&ctx, layout).unwrap();
    let query = upload_fp32_as_bf16(&ctx, &[4.125, 7.59375], vec![1, 2]).unwrap();
    let key = upload_fp32_as_bf16(&ctx, &[4.125, 7.59375], vec![1, 2]).unwrap();
    let value = upload_fp32_as_bf16(&ctx, &[1.0, 0.0], vec![1, 2]).unwrap();
    let a = upload_fp32_as_bf16(&ctx, &[0.0], vec![1, 1]).unwrap();
    let b = upload_fp32_as_bf16(&ctx, &[8.0], vec![1, 1]).unwrap();
    let a_log = upload_fp32_as_bf16(&ctx, &[-20.0], vec![1]).unwrap();
    let dt_bias = upload_fp32_as_bf16(&ctx, &[0.0], vec![1]).unwrap();

    let output = state
        .gated_delta_prefill(&ctx, &query, &key, &value, &a, &b, &a_log, &dt_bias)
        .unwrap();

    // Transformers performs the multiply and reduction for l2norm while q/k
    // are BF16. These inputs normalize to [0.474609375, 0.875], not the
    // values obtained by accumulating their squares in FP32.
    assert_close(
        &download_bf16_as_fp32(&output).unwrap(),
        &[0.69921875, 0.0],
        0.001,
    );
    assert_close(
        &state.recurrent_host(&ctx).unwrap(),
        &[0.474609375, 0.0, 0.875, 0.0],
        0.000001,
    );
}

#[test]
fn qwen35_gdn_cuda_sequence_normalization_matches_pure_reference_for_all_rows() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let layout = tiny_layout();

    let cases = [
        (32usize, 31usize, 1.421875f32),
        (33, 31, 1.421875),
        (33, 32, 1.421875),
        (96, 95, 1.421875),
        (127, 95, 1.421875),
        (127, 96, 1.421875),
        (128, 127, 1.421875),
    ];

    for (rows, active_row, expected) in cases {
        let mut state = Qwen35GdnState::new(&ctx, layout).unwrap();
        let query = [1.2969f32, 0.6992]
            .into_iter()
            .cycle()
            .take(rows * 2)
            .collect::<Vec<_>>();
        let mut value = vec![0.0f32; rows * 2];
        value[active_row * 2] = 2.0;
        let a = vec![-100.0f32; rows];
        let mut b = vec![-100.0f32; rows];
        b[active_row] = 100.0;

        let query = upload_fp32_as_bf16(&ctx, &query, vec![rows, 2]).unwrap();
        let key = upload_fp32_as_bf16(
            &ctx,
            &[1.2969f32, 0.6992]
                .into_iter()
                .cycle()
                .take(rows * 2)
                .collect::<Vec<_>>(),
            vec![rows, 2],
        )
        .unwrap();
        let value = upload_fp32_as_bf16(&ctx, &value, vec![rows, 2]).unwrap();
        let a = upload_fp32_as_bf16(&ctx, &a, vec![rows, 1]).unwrap();
        let b = upload_fp32_as_bf16(&ctx, &b, vec![rows, 1]).unwrap();
        let a_log = upload_fp32_as_bf16(&ctx, &[0.0], vec![1]).unwrap();
        let dt_bias = upload_fp32_as_bf16(&ctx, &[0.0], vec![1]).unwrap();

        let output = state
            .gated_delta_prefill(&ctx, &query, &key, &value, &a, &b, &a_log, &dt_bias)
            .unwrap();
        let output = download_bf16_as_fp32(&output).unwrap();

        assert_close(
            &output[active_row * 2..active_row * 2 + 2],
            &[expected, 0.0],
            0.001,
        );
    }
}

#[test]
fn qwen35_gdn_cuda_sequence_normalization_is_invariant_to_total_row_count() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let layout = tiny_layout();

    fn run_case(ctx: &CudaContext, layout: Qwen35GdnLayout, rows: usize) -> Vec<f32> {
        let active_row = 32usize;
        let mut query = vec![0.0f32; rows * 2];
        let mut key = vec![0.0f32; rows * 2];
        let mut value = vec![0.0f32; rows * 2];
        let a = vec![-100.0f32; rows];
        let mut b = vec![-100.0f32; rows];
        query[active_row * 2..active_row * 2 + 2].copy_from_slice(&[1.2969, 0.6992]);
        key[active_row * 2..active_row * 2 + 2].copy_from_slice(&[1.2969, 0.6992]);
        value[active_row * 2] = 2.0;
        b[active_row] = 100.0;

        let query = upload_fp32_as_bf16(ctx, &query, vec![rows, 2]).unwrap();
        let key = upload_fp32_as_bf16(ctx, &key, vec![rows, 2]).unwrap();
        let value = upload_fp32_as_bf16(ctx, &value, vec![rows, 2]).unwrap();
        let a = upload_fp32_as_bf16(ctx, &a, vec![rows, 1]).unwrap();
        let b = upload_fp32_as_bf16(ctx, &b, vec![rows, 1]).unwrap();
        let a_log = upload_fp32_as_bf16(ctx, &[0.0], vec![1]).unwrap();
        let dt_bias = upload_fp32_as_bf16(ctx, &[0.0], vec![1]).unwrap();
        let mut state = Qwen35GdnState::new(ctx, layout).unwrap();
        let output = state
            .gated_delta_prefill(ctx, &query, &key, &value, &a, &b, &a_log, &dt_bias)
            .unwrap();
        let output = download_bf16_as_fp32(&output).unwrap();
        output[active_row * 2..active_row * 2 + 2].to_vec()
    }

    let tail = run_case(&ctx, layout, 33);
    let full_block = run_case(&ctx, layout, 64);
    assert_close(&tail, &full_block, 0.001);
}

#[test]
fn qwen35_gdn_cuda_sequence_crosses_transformers_64_token_chunk_boundary() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let layout = tiny_layout();
    let mut state = Qwen35GdnState::new(&ctx, layout).unwrap();
    let rows = 65usize;
    let mut query = Vec::with_capacity(rows * 2);
    let mut key = Vec::with_capacity(rows * 2);
    let mut value = Vec::with_capacity(rows * 2);
    let mut a = Vec::with_capacity(rows);
    let mut b = Vec::with_capacity(rows);
    for token in 0..rows {
        query.extend_from_slice(&[
            ((token % 17) as f32) - 8.0,
            (((3 * token) % 19) as f32) - 9.0,
        ]);
        key.extend_from_slice(&[
            (((5 * token) % 23) as f32) - 11.0,
            (((7 * token) % 29) as f32) - 14.0,
        ]);
        value.extend_from_slice(&[
            (((11 * token) % 31) as f32) - 15.0,
            (((13 * token) % 37) as f32) - 18.0,
        ]);
        a.push(if token % 2 == 0 { -2.0 } else { 2.0 });
        b.push(if token % 3 == 0 { -4.0 } else { 4.0 });
    }
    let query = upload_fp32_as_bf16(&ctx, &query, vec![rows, 2]).unwrap();
    let key = upload_fp32_as_bf16(&ctx, &key, vec![rows, 2]).unwrap();
    let value = upload_fp32_as_bf16(&ctx, &value, vec![rows, 2]).unwrap();
    let a = upload_fp32_as_bf16(&ctx, &a, vec![rows, 1]).unwrap();
    let b = upload_fp32_as_bf16(&ctx, &b, vec![rows, 1]).unwrap();
    let a_log = upload_fp32_as_bf16(&ctx, &[0.0], vec![1]).unwrap();
    let dt_bias = upload_fp32_as_bf16(&ctx, &[0.0], vec![1]).unwrap();

    let output = state
        .gated_delta_prefill(&ctx, &query, &key, &value, &a, &b, &a_log, &dt_bias)
        .unwrap();
    let output = download_bf16_as_fp32(&output).unwrap();

    let expected_rows = [
        (0usize, [-0.189453125, -0.2275390625]),
        (1, [-2.71875, -3.40625]),
        (2, [3.5, 3.90625]),
        (62, [-10.0, 7.21875]),
        (63, [-1.171875, 0.95703125]),
        (64, [4.0, -0.68359375]),
    ];
    for (row, expected) in expected_rows {
        assert_close(&output[row * 2..row * 2 + 2], &expected, 0.01);
    }
    // A direct token recurrence produces about [4.0, -0.68359375] at row
    // 64. This oracle requires the chunk triangular solve and padded second
    // chunk used by torch_chunk_gated_delta_rule.
    assert_close(
        &state.recurrent_host(&ctx).unwrap(),
        &[6.6858883, 0.126413, -2.1652663, 1.2823228],
        0.01,
    );
    assert_eq!(state.position(), 0);
}

#[test]
fn qwen35_gdn_cuda_sequence_non_finite_failure_preserves_recurrent_state() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let layout = tiny_layout();
    let mut state = Qwen35GdnState::new(&ctx, layout).unwrap();
    let initial_state = state.recurrent_host(&ctx).unwrap();
    let query = upload_fp32_as_bf16(&ctx, &[1.0, 0.0], vec![1, 2]).unwrap();
    let key = upload_fp32_as_bf16(&ctx, &[1.0, 0.0], vec![1, 2]).unwrap();
    let value = upload_fp32_as_bf16(&ctx, &[2.0, 4.0], vec![1, 2]).unwrap();
    let a = upload_fp32_as_bf16(&ctx, &[0.0], vec![1, 1]).unwrap();
    let b = upload_fp32_as_bf16(&ctx, &[0.0], vec![1, 1]).unwrap();
    let a_log = upload_fp32_as_bf16(&ctx, &[0.0], vec![1]).unwrap();
    let dt_bias = upload_fp32_as_bf16(&ctx, &[0.0], vec![1]).unwrap();
    state
        .gated_delta_prefill(&ctx, &query, &key, &value, &a, &b, &a_log, &dt_bias)
        .unwrap();
    let committed_state = state.recurrent_host(&ctx).unwrap();
    let committed_position = state.position();

    let bad_query = upload_fp32_as_bf16(&ctx, &[f32::NAN, 0.0], vec![1, 2]).unwrap();
    let error = state
        .gated_delta_prefill(&ctx, &bad_query, &key, &value, &a, &b, &a_log, &dt_bias)
        .unwrap_err();
    assert!(error.to_string().contains("non-finite"));
    assert_eq!(state.position(), committed_position);
    assert_eq!(state.recurrent_host(&ctx).unwrap(), committed_state);

    // A failed speculative prefill must not discard the rollback handle for
    // the last successful recurrent commit.
    state.rollback_last_recurrent().unwrap();
    assert_eq!(state.recurrent_host(&ctx).unwrap(), initial_state);
}

#[test]
fn qwen35_gdn_cuda_recurrent_commit_can_be_rolled_back() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let layout = tiny_layout();
    let mut state = Qwen35GdnState::new(&ctx, layout).unwrap();
    let query = upload_fp32_as_bf16(&ctx, &[1.0, 0.0], vec![1, 2]).unwrap();
    let key = upload_fp32_as_bf16(&ctx, &[1.0, 0.0], vec![1, 2]).unwrap();
    let a = upload_fp32_as_bf16(&ctx, &[0.0], vec![1, 1]).unwrap();
    let b = upload_fp32_as_bf16(&ctx, &[0.0], vec![1, 1]).unwrap();
    let a_log = upload_fp32_as_bf16(&ctx, &[0.0], vec![1]).unwrap();
    let dt_bias = upload_fp32_as_bf16(&ctx, &[0.0], vec![1]).unwrap();

    let first_value = upload_fp32_as_bf16(&ctx, &[2.0, 4.0], vec![1, 2]).unwrap();
    state
        .gated_delta_step(&ctx, &query, &key, &first_value, &a, &b, &a_log, &dt_bias)
        .unwrap();
    let first_state = state.recurrent_host(&ctx).unwrap();

    let second_value = upload_fp32_as_bf16(&ctx, &[6.0, 8.0], vec![1, 2]).unwrap();
    state
        .gated_delta_step(&ctx, &query, &key, &second_value, &a, &b, &a_log, &dt_bias)
        .unwrap();
    assert_ne!(state.recurrent_host(&ctx).unwrap(), first_state);

    state.rollback_last_recurrent().unwrap();
    assert_eq!(state.recurrent_host(&ctx).unwrap(), first_state);
}

#[test]
fn qwen35_gdn_cuda_recurrent_repeats_one_key_head_across_three_value_heads() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let layout = Qwen35GdnLayout::new(4, 1, 3, 2, 2).unwrap();
    assert_eq!(layout.query_width(), 2);
    let mut state = Qwen35GdnState::new(&ctx, layout).unwrap();
    let query = upload_fp32_as_bf16(&ctx, &[1.0, 0.0], vec![1, 2]).unwrap();
    let key = upload_fp32_as_bf16(&ctx, &[1.0, 0.0], vec![1, 2]).unwrap();
    let value = upload_fp32_as_bf16(&ctx, &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0], vec![1, 6]).unwrap();
    let a = upload_fp32_as_bf16(&ctx, &[0.0, 0.0, 0.0], vec![3]).unwrap();
    let b = upload_fp32_as_bf16(&ctx, &[0.0, 0.0, 0.0], vec![3]).unwrap();
    let a_log = upload_fp32_as_bf16(&ctx, &[0.0, 0.0, 0.0], vec![3]).unwrap();
    let dt_bias = upload_fp32_as_bf16(&ctx, &[0.0, 0.0, 0.0], vec![3]).unwrap();

    let output = state
        .gated_delta_step(&ctx, &query, &key, &value, &a, &b, &a_log, &dt_bias)
        .unwrap();
    let output = download_bf16_as_fp32(&output).unwrap();
    let scale = (1.0 + 1e-6f32).sqrt().recip().powi(2) / 2.0f32.sqrt();
    assert_close(
        &output,
        &[
            1.0 * scale,
            2.0 * scale,
            3.0 * scale,
            4.0 * scale,
            5.0 * scale,
            6.0 * scale,
        ],
        0.02,
    );
    let recurrent = state.recurrent_host(&ctx).unwrap();
    assert_close(
        &recurrent,
        &[
            (1.0 + 1e-6f32).sqrt().recip(),
            2.0 * (1.0 + 1e-6f32).sqrt().recip(),
            0.0,
            0.0,
            3.0 * (1.0 + 1e-6f32).sqrt().recip(),
            4.0 * (1.0 + 1e-6f32).sqrt().recip(),
            0.0,
            0.0,
            5.0 * (1.0 + 1e-6f32).sqrt().recip(),
            6.0 * (1.0 + 1e-6f32).sqrt().recip(),
            0.0,
            0.0,
        ],
        1e-5,
    );
}

#[test]
fn qwen35_gdn_cuda_gated_rms_norm_uses_direct_weight_and_silu_gate() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let input = upload_fp32_as_bf16(&ctx, &[5.25, -4.78125], vec![1, 2]).unwrap();
    let gate = upload_fp32_as_bf16(&ctx, &[6.875, 8.75], vec![1, 2]).unwrap();
    let weight = upload_fp32_as_bf16(&ctx, &[2.21875, 1.4375], vec![2]).unwrap();

    let output = gated_rms_norm_bf16(&ctx, &input, &gate, &weight, 1, 2, 1e-6).unwrap();
    let output = download_bf16_as_fp32(&output).unwrap();

    let rms = ((5.25f32.powi(2) + (-4.78125f32).powi(2)) / 2.0 + 1e-6)
        .sqrt()
        .recip();
    let silu = |value: f32| value / (1.0 + (-value).exp());
    // Transformers' Qwen3.5 RMSNormGated rounds the normalized hidden to
    // BF16 before multiplying the BF16 norm weight, then applies the FP32
    // SiLU gate and rounds the final result to BF16.
    let bf16 = |value: f32| half::bf16::from_f32(value).to_f32();
    let expected = [
        bf16(bf16(5.25 * rms) * bf16(2.21875) * silu(6.875)),
        bf16(bf16(-4.78125 * rms) * bf16(1.4375) * silu(8.75)),
    ];
    assert_close(&output, &expected, 0.0001);
}

#[test]
fn qwen35_gdn_cuda_failed_step_and_reset_do_not_leak_request_state() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let layout = tiny_layout();
    let mut state = Qwen35GdnState::new(&ctx, layout).unwrap();
    let query = upload_fp32_as_bf16(&ctx, &[1.0, 0.0], vec![1, 2]).unwrap();
    let key = upload_fp32_as_bf16(&ctx, &[1.0, 0.0], vec![1, 2]).unwrap();
    let value = upload_fp32_as_bf16(&ctx, &[2.0, 4.0], vec![1, 2]).unwrap();
    let a = upload_fp32_as_bf16(&ctx, &[0.0], vec![1, 1]).unwrap();
    let b = upload_fp32_as_bf16(&ctx, &[0.0], vec![1, 1]).unwrap();
    let a_log = upload_fp32_as_bf16(&ctx, &[0.0], vec![1]).unwrap();
    let dt_bias = upload_fp32_as_bf16(&ctx, &[0.0], vec![1]).unwrap();
    state
        .gated_delta_step(&ctx, &query, &key, &value, &a, &b, &a_log, &dt_bias)
        .unwrap();
    let before = state.recurrent_host(&ctx).unwrap();
    let before_position = state.position();

    let bad_query = upload_fp32_as_bf16(&ctx, &[f32::NAN, 0.0], vec![1, 2]).unwrap();
    let error = state
        .gated_delta_step(&ctx, &bad_query, &key, &value, &a, &b, &a_log, &dt_bias)
        .unwrap_err();
    assert!(error.to_string().contains("non-finite"));
    assert_eq!(state.position(), before_position);
    assert_eq!(state.recurrent_host(&ctx).unwrap(), before);

    state.reset(&ctx).unwrap();
    assert_eq!(state.position(), 0);
    assert_eq!(state.conv_cursor(), 0);
    assert!(state
        .recurrent_host(&ctx)
        .unwrap()
        .iter()
        .all(|value| *value == 0.0));
    assert!(state
        .conv_ring_channel_host(&ctx, 0)
        .unwrap()
        .iter()
        .all(|value| *value == 0.0));
}

#[test]
fn qwen35_gdn_cuda_rejects_short_state_storage_before_launch() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let layout = tiny_layout();
    let short = CudaBuffer::alloc(4, ctx.device_id()).unwrap();
    assert!(Qwen35GdnState::from_buffers_for_test(&ctx, layout, short).is_err());
}

/// Wall-clock probe for the sequence prefill at the pinned checkpoint's real
/// GDN layout. It is a measurement, not an assertion of any latency target;
/// run it explicitly to size the prefill budget for one of the 48 GDN layers.
#[test]
#[ignore = "timing probe: run explicitly with --ignored"]
fn qwen35_gdn_cuda_sequence_prefill_timing_probe_at_checkpoint_layout() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let key_heads = 16usize;
    let value_heads = 48usize;
    let key_dim = 128usize;
    let value_dim = 128usize;
    let layout = Qwen35GdnLayout::new(4, key_heads, value_heads, key_dim, value_dim).unwrap();

    for rows in [64usize, 256, 1024] {
        let mut state = Qwen35GdnState::new(&ctx, layout).unwrap();
        let query = (0..rows * layout.query_width())
            .map(|index| (index as f32 * 0.017).sin() * 0.5)
            .collect::<Vec<_>>();
        let key = (0..rows * layout.query_width())
            .map(|index| (index as f32 * 0.023).cos() * 0.5)
            .collect::<Vec<_>>();
        let value = (0..rows * layout.value_width())
            .map(|index| (index as f32 * 0.031).sin() * 0.5)
            .collect::<Vec<_>>();
        let a = (0..rows * value_heads)
            .map(|index| -0.5 + (index % 7) as f32 * 0.1)
            .collect::<Vec<_>>();
        let b = (0..rows * value_heads)
            .map(|index| 0.3 - (index % 5) as f32 * 0.05)
            .collect::<Vec<_>>();

        let query = upload_fp32_as_bf16(&ctx, &query, vec![rows, layout.query_width()]).unwrap();
        let key = upload_fp32_as_bf16(&ctx, &key, vec![rows, layout.query_width()]).unwrap();
        let value = upload_fp32_as_bf16(&ctx, &value, vec![rows, layout.value_width()]).unwrap();
        let a = upload_fp32_as_bf16(&ctx, &a, vec![rows, value_heads]).unwrap();
        let b = upload_fp32_as_bf16(&ctx, &b, vec![rows, value_heads]).unwrap();
        let a_log = upload_fp32_as_bf16(&ctx, &vec![0.35; value_heads], vec![value_heads]).unwrap();
        let dt_bias =
            upload_fp32_as_bf16(&ctx, &vec![-0.15; value_heads], vec![value_heads]).unwrap();

        let started = std::time::Instant::now();
        let output = state
            .gated_delta_prefill(&ctx, &query, &key, &value, &a, &b, &a_log, &dt_bias)
            .unwrap();
        let _ = download_bf16_as_fp32(&output).unwrap();
        let elapsed = started.elapsed();
        println!(
            "GDN sequence prefill rows={rows}: {:.3} s per layer, {:.3} s for 48 layers",
            elapsed.as_secs_f64(),
            elapsed.as_secs_f64() * 48.0
        );
    }
}

#[test]
fn qwen35_gdn_cuda_sequence_chunked_matches_eager_step_at_checkpoint_layout() {
    // Compare the staged 64-token chunked sequence path against the eager
    // per-token path at the real checkpoint layout (16 key heads, 48 value
    // heads, key_dim=value_dim=128, repeat factor 3). The eager path is the
    // previously validated decode/reference-compatible recurrence.
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let layout = Qwen35GdnLayout::new(4, 16, 48, 128, 128).unwrap();
    let qw = layout.query_width();
    let vw = layout.value_width();
    let rows = 8usize;

    let mut query = Vec::with_capacity(rows * qw);
    let mut key = Vec::with_capacity(rows * qw);
    let mut value = Vec::with_capacity(rows * vw);
    let mut a = Vec::with_capacity(rows * 48);
    let mut b = Vec::with_capacity(rows * 48);
    for row in 0..rows {
        for d in 0..qw {
            let idx = (row * qw + d) as f32;
            query.push((idx * 0.031).sin() + (idx * 0.007).cos() * 0.35);
            key.push((idx * 0.043).cos() - (idx * 0.011).sin() * 0.28);
        }
        for d in 0..vw {
            let idx = (row * vw + d) as f32;
            value.push((idx * 0.083).sin() * 1.15 + (idx * 0.019).cos() * 0.18);
        }
        for h in 0..48 {
            a.push(-1.4 + 2.6 * ((row * 48 + h) % 7) as f32 / 6.0);
            b.push(0.8 - 1.4 * ((row * 48 + h * 3) % 5) as f32 / 4.0);
        }
    }
    let a_log = (0..48)
        .map(|h| 0.15 + 0.05 * (h % 3) as f32)
        .collect::<Vec<_>>();
    let dt_bias = (0..48)
        .map(|h| -0.12 + 0.04 * (h % 4) as f32)
        .collect::<Vec<_>>();

    let query_t = upload_fp32_as_bf16(&ctx, &query, vec![rows, qw]).unwrap();
    let key_t = upload_fp32_as_bf16(&ctx, &key, vec![rows, qw]).unwrap();
    let value_t = upload_fp32_as_bf16(&ctx, &value, vec![rows, vw]).unwrap();
    let a_t = upload_fp32_as_bf16(&ctx, &a, vec![rows, 48]).unwrap();
    let b_t = upload_fp32_as_bf16(&ctx, &b, vec![rows, 48]).unwrap();
    let a_log_t = upload_fp32_as_bf16(&ctx, &a_log, vec![48]).unwrap();
    let dt_bias_t = upload_fp32_as_bf16(&ctx, &dt_bias, vec![48]).unwrap();

    // Chunked path: one 64-token padded chunk over 8 valid rows.
    let mut chunked = Qwen35GdnState::new(&ctx, layout).unwrap();
    let output_chunked = chunked
        .gated_delta_prefill(
            &ctx, &query_t, &key_t, &value_t, &a_t, &b_t, &a_log_t, &dt_bias_t,
        )
        .unwrap();
    let output_chunked = download_bf16_as_fp32(&output_chunked).unwrap();
    let state_chunked = chunked.recurrent_host(&ctx).unwrap();

    // Eager path: one token at a time with row-sliced inputs.
    let mut eager = Qwen35GdnState::new(&ctx, layout).unwrap();
    let mut output_eager = Vec::with_capacity(rows * vw);
    for row in 0..rows {
        let query_row =
            upload_fp32_as_bf16(&ctx, &query[row * qw..(row + 1) * qw], vec![1, qw]).unwrap();
        let key_row =
            upload_fp32_as_bf16(&ctx, &key[row * qw..(row + 1) * qw], vec![1, qw]).unwrap();
        let value_row =
            upload_fp32_as_bf16(&ctx, &value[row * vw..(row + 1) * vw], vec![1, vw]).unwrap();
        let a_row = upload_fp32_as_bf16(&ctx, &a[row * 48..(row + 1) * 48], vec![48]).unwrap();
        let b_row = upload_fp32_as_bf16(&ctx, &b[row * 48..(row + 1) * 48], vec![48]).unwrap();
        let step = eager
            .gated_delta_step(
                &ctx, &query_row, &key_row, &value_row, &a_row, &b_row, &a_log_t, &dt_bias_t,
            )
            .unwrap();
        output_eager.extend(download_bf16_as_fp32(&step).unwrap());
    }
    let state_eager = eager.recurrent_host(&ctx).unwrap();

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut worst = 0usize;
    for i in 0..output_chunked.len() {
        let delta = (output_chunked[i] - output_eager[i]).abs();
        if delta > max_abs {
            max_abs = delta;
            worst = i;
        }
        let base = output_eager[i].abs().max(1e-3);
        let rel = delta / base;
        if rel > max_rel {
            max_rel = rel;
        }
    }
    let mut state_max_abs = 0.0f32;
    let mut state_worst = 0usize;
    for i in 0..state_chunked.len() {
        let delta = (state_chunked[i] - state_eager[i]).abs();
        if delta > state_max_abs {
            state_max_abs = delta;
            state_worst = i;
        }
    }
    eprintln!("chunked-vs-eager rows=8: output max_abs={max_abs} max_rel={max_rel} worst={worst} ",);
    eprintln!("state max_abs={state_max_abs} worst={state_worst}",);
    // BF16 outputs must agree to a small fraction of an ulp; state is FP32
    // and must track the eager recurrence closely.
    assert!(
        max_abs < 0.02,
        "chunked output diverged from eager: {max_abs}"
    );
    assert!(
        max_rel < 0.05,
        "chunked output relative divergence: {max_rel}"
    );
    assert!(
        state_max_abs < 0.01,
        "chunked state diverged from eager: {state_max_abs}"
    );
}
