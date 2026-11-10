use half::bf16;

use apxinf_core::storage::GpuStorageHandle;
use apxinf_core::{DType, Device, Error, Shape, Storage, Tensor};

use crate::kernels::qwen35_w4::{
    project_bf16, Qwen35W4Buffers, Qwen35W4DeviceProjection, Qwen35W4Layout,
};
use crate::test_util::{assert_bf16_close_reduction, download_bf16_as_fp32, upload_fp32_as_bf16};
use crate::{CudaBuffer, CudaContext};

fn upload_u32(ctx: &CudaContext, values: &[u32]) -> CudaBuffer {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let buffer = CudaBuffer::alloc(bytes.len(), ctx.device_id()).unwrap();
    buffer.copy_from_host(&bytes).unwrap();
    buffer
}

fn cpu_reference(
    layout: Qwen35W4Layout,
    activation: &[f32],
    weights: &[u32],
    scales: &[f32],
    zero_points: &[u32],
) -> Vec<f32> {
    let activation: Vec<f32> = activation
        .iter()
        .map(|value| bf16::from_f32(*value).to_f32())
        .collect();
    let scales: Vec<f32> = scales
        .iter()
        .map(|value| bf16::from_f32(*value).to_f32())
        .collect();
    let mut output = vec![0.0; layout.out_features];
    for out in 0..layout.out_features {
        for k in 0..layout.in_features {
            let group = k / layout.group_size;
            let packed = weights[out * layout.packed_k_columns() + k / 8];
            let quantized = ((packed >> (4 * (k % 8))) & 0x0f) as f32;
            let packed_zero = zero_points[(out / 8) * layout.groups() + group];
            let zero_point = ((packed_zero >> (4 * (out % 8))) & 0x0f) as f32;
            output[out] +=
                activation[k] * (quantized - zero_point) * scales[out * layout.groups() + group];
        }
        output[out] = bf16::from_f32(output[out]).to_f32();
    }
    output
}

fn cpu_reference_with_bf16_dequantized_weights(
    layout: Qwen35W4Layout,
    activation: &[f32],
    weights: &[u32],
    scales: &[f32],
    zero_points: &[u32],
) -> Vec<f32> {
    let activation = activation
        .iter()
        .map(|value| bf16::from_f32(*value).to_f32())
        .collect::<Vec<_>>();
    let scales = scales
        .iter()
        .map(|value| bf16::from_f32(*value).to_f32())
        .collect::<Vec<_>>();
    let mut output = vec![0.0; layout.out_features];
    for out in 0..layout.out_features {
        for k in 0..layout.in_features {
            let group = k / layout.group_size;
            let packed = weights[out * layout.packed_k_columns() + k / 8];
            let quantized = ((packed >> (4 * (k % 8))) & 0x0f) as f32;
            let packed_zero = zero_points[(out / 8) * layout.groups() + group];
            let zero_point = ((packed_zero >> (4 * (out % 8))) & 0x0f) as f32;
            let weight =
                bf16::from_f32((quantized - zero_point) * scales[out * layout.groups() + group])
                    .to_f32();
            output[out] += activation[k] * weight;
        }
        output[out] = bf16::from_f32(output[out]).to_f32();
    }
    output
}

/// Restores the prefill-GEMM threshold env var on drop so a failing
/// assertion cannot leak the setting into other tests.
struct PrefillGemmGuard;

impl PrefillGemmGuard {
    fn set(value: &str) -> Self {
        std::env::set_var("APXINF_Q35_W4_PREFILL_GEMM", value);
        Self
    }
}

impl Drop for PrefillGemmGuard {
    fn drop(&mut self) {
        std::env::remove_var("APXINF_Q35_W4_PREFILL_GEMM");
    }
}

/// Restores the packed-GEMV selector on drop.
struct PackedGemvGuard;

impl PackedGemvGuard {
    fn set(value: &str) -> Self {
        std::env::set_var("APXINF_Q35_W4_PACKED_GEMV", value);
        Self
    }
}

impl Drop for PackedGemvGuard {
    fn drop(&mut self) {
        std::env::remove_var("APXINF_Q35_W4_PACKED_GEMV");
    }
}

#[test]
fn qwen35_w4_cuda_packed_gemv_matches_baseline_kernel_and_cpu_reference() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    // in_features deliberately not a multiple of 8 so the packed tail
    // (fewer than eight valid nibbles in the last uint32) is exercised, and
    // out_features not a multiple of 8 for the zero-point N-packing tail.
    let layout = Qwen35W4Layout::new(11, 100, 32).unwrap();
    let activation: Vec<f32> = (0..layout.in_features)
        .map(|index| ((index % 23) as f32) * 0.03125 - 0.375)
        .collect();
    let mut weights = vec![0u32; layout.out_features * layout.packed_k_columns()];
    for out in 0..layout.out_features {
        for k in 0..layout.in_features {
            let value = ((out * 13 + k * 7) & 0x0f) as u32;
            weights[out * layout.packed_k_columns() + k / 8] |= value << (4 * (k % 8));
        }
    }
    let scales: Vec<f32> = (0..layout.out_features * layout.groups())
        .map(|index| 0.0625 * ((index % 5) + 1) as f32)
        .collect();
    let mut zero_points = vec![0u32; layout.packed_n_rows() * layout.groups()];
    for out in 0..layout.out_features {
        for group in 0..layout.groups() {
            let value = ((out + 3 * group) & 0x0f) as u32;
            zero_points[(out / 8) * layout.groups() + group] |= value << (4 * (out % 8));
        }
    }
    let expected = cpu_reference_with_bf16_dequantized_weights(
        layout,
        &activation,
        &weights,
        &scales,
        &zero_points,
    );

    let activation_gpu =
        upload_fp32_as_bf16(&ctx, &activation, vec![1, layout.in_features]).unwrap();
    let scales_gpu =
        upload_fp32_as_bf16(&ctx, &scales, vec![layout.out_features, layout.groups()]).unwrap();
    let weights_gpu = upload_u32(&ctx, &weights);
    let zero_points_gpu = upload_u32(&ctx, &zero_points);
    let buffers = || Qwen35W4Buffers {
        weight_packed: &weights_gpu,
        scales: &scales_gpu,
        zero_points: &zero_points_gpu,
    };

    // Pin the warp kernel off so this A/B isolates the packed-vs-baseline
    // variable only.
    let _warp_off = WarpGemvGuard::set("0");
    let guard = PackedGemvGuard::set("0");
    let baseline = project_bf16(&ctx, &activation_gpu, buffers(), layout).unwrap();
    let baseline_values = download_bf16_as_fp32(&baseline).unwrap();
    drop(guard);

    let guard = PackedGemvGuard::set("1");
    let packed = project_bf16(&ctx, &activation_gpu, buffers(), layout).unwrap();
    let packed_values = download_bf16_as_fp32(&packed).unwrap();
    drop(guard);

    assert_eq!(packed.shape().dims(), &[1, layout.out_features]);
    // Both kernels decompress each weight identically; only the accumulation
    // order and the reduction shape differ.
    assert_bf16_close_reduction(&baseline_values, &expected);
    assert_bf16_close_reduction(&packed_values, &expected);
    assert_bf16_close_reduction(&packed_values, &baseline_values);
}

/// Restores the warp-GEMV selector on drop.
struct WarpGemvGuard;

impl WarpGemvGuard {
    fn set(value: &str) -> Self {
        std::env::set_var("APXINF_Q35_W4_WARP_GEMV", value);
        Self
    }
}

impl Drop for WarpGemvGuard {
    fn drop(&mut self) {
        std::env::remove_var("APXINF_Q35_W4_WARP_GEMV");
    }
}

#[test]
fn qwen35_w4_cuda_warp_gemv_matches_packed_kernel_and_cpu_reference() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    // out_features deliberately not a multiple of the 8 warps per block, and
    // in_features not a multiple of 8, so both the block tail (idle warps)
    // and the packed-nibble tail are exercised.
    let layout = Qwen35W4Layout::new(19, 100, 32).unwrap();
    let activation: Vec<f32> = (0..layout.in_features)
        .map(|index| ((index % 29) as f32) * 0.03125 - 0.4375)
        .collect();
    let mut weights = vec![0u32; layout.out_features * layout.packed_k_columns()];
    for out in 0..layout.out_features {
        for k in 0..layout.in_features {
            let value = ((out * 9 + k * 5) & 0x0f) as u32;
            weights[out * layout.packed_k_columns() + k / 8] |= value << (4 * (k % 8));
        }
    }
    let scales: Vec<f32> = (0..layout.out_features * layout.groups())
        .map(|index| 0.0625 * ((index % 7) + 1) as f32)
        .collect();
    let mut zero_points = vec![0u32; layout.packed_n_rows() * layout.groups()];
    for out in 0..layout.out_features {
        for group in 0..layout.groups() {
            let value = ((out * 5 + group * 2) & 0x0f) as u32;
            zero_points[(out / 8) * layout.groups() + group] |= value << (4 * (out % 8));
        }
    }
    let expected = cpu_reference_with_bf16_dequantized_weights(
        layout,
        &activation,
        &weights,
        &scales,
        &zero_points,
    );

    let activation_gpu =
        upload_fp32_as_bf16(&ctx, &activation, vec![1, layout.in_features]).unwrap();
    let scales_gpu =
        upload_fp32_as_bf16(&ctx, &scales, vec![layout.out_features, layout.groups()]).unwrap();
    let weights_gpu = upload_u32(&ctx, &weights);
    let zero_points_gpu = upload_u32(&ctx, &zero_points);
    let buffers = || Qwen35W4Buffers {
        weight_packed: &weights_gpu,
        scales: &scales_gpu,
        zero_points: &zero_points_gpu,
    };

    let guard = WarpGemvGuard::set("0");
    let packed = project_bf16(&ctx, &activation_gpu, buffers(), layout).unwrap();
    let packed_values = download_bf16_as_fp32(&packed).unwrap();
    drop(guard);

    let guard = WarpGemvGuard::set("1");
    let warp = project_bf16(&ctx, &activation_gpu, buffers(), layout).unwrap();
    let warp_values = download_bf16_as_fp32(&warp).unwrap();
    drop(guard);

    assert_eq!(warp.shape().dims(), &[1, layout.out_features]);
    assert_bf16_close_reduction(&packed_values, &expected);
    assert_bf16_close_reduction(&warp_values, &expected);
    assert_bf16_close_reduction(&warp_values, &packed_values);
}

#[test]
fn qwen35_w4_cuda_warp_gemv_rejects_non_finite_scale() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let layout = Qwen35W4Layout::new(1, 1, 32).unwrap();
    let activation = upload_fp32_as_bf16(&ctx, &[1.0], vec![1, 1]).unwrap();
    let scales = upload_fp32_as_bf16(&ctx, &[f32::NAN], vec![1, 1]).unwrap();
    let weights = upload_u32(&ctx, &[0x0000_0001]);
    let zero_points = upload_u32(&ctx, &[0]);
    let _guard = WarpGemvGuard::set("1");
    let error = project_bf16(
        &ctx,
        &activation,
        Qwen35W4Buffers {
            weight_packed: &weights,
            scales: &scales,
            zero_points: &zero_points,
        },
        layout,
    )
    .unwrap_err();
    assert!(
        error.to_string().contains("non-finite W4 scale"),
        "unexpected error: {error}"
    );
}

#[test]
fn qwen35_w4_cuda_packed_gemv_rejects_non_finite_scale() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let layout = Qwen35W4Layout::new(1, 1, 32).unwrap();
    let activation = upload_fp32_as_bf16(&ctx, &[1.0], vec![1, 1]).unwrap();
    let scales = upload_fp32_as_bf16(&ctx, &[f32::NAN], vec![1, 1]).unwrap();
    let weights = upload_u32(&ctx, &[0x0000_0001]);
    let zero_points = upload_u32(&ctx, &[0]);
    let _warp_off = WarpGemvGuard::set("0");
    let _guard = PackedGemvGuard::set("1");
    let error = project_bf16(
        &ctx,
        &activation,
        Qwen35W4Buffers {
            weight_packed: &weights,
            scales: &scales,
            zero_points: &zero_points,
        },
        layout,
    )
    .unwrap_err();
    assert!(
        error.to_string().contains("non-finite W4 scale"),
        "unexpected error: {error}"
    );
}

#[test]
fn qwen35_w4_cuda_prefill_gemm_matches_gemv_path_and_cpu_reference() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    // Multi-row, multi-group shape with N not a multiple of 8 so the
    // zero-point N-packing tail is exercised on both paths.
    let layout = Qwen35W4Layout::new(11, 96, 32).unwrap();
    let rows = 24usize;
    let activation: Vec<f32> = (0..rows * layout.in_features)
        .map(|index| ((index % 37) as f32) * 0.015625 - 0.25)
        .collect();
    let mut weights = vec![0u32; layout.out_features * layout.packed_k_columns()];
    for out in 0..layout.out_features {
        for k in 0..layout.in_features {
            let value = ((out * 5 + k * 11) & 0x0f) as u32;
            weights[out * layout.packed_k_columns() + k / 8] |= value << (4 * (k % 8));
        }
    }
    let scales: Vec<f32> = (0..layout.out_features * layout.groups())
        .map(|index| 0.03125 * ((index % 9) + 1) as f32)
        .collect();
    let mut zero_points = vec![0u32; layout.packed_n_rows() * layout.groups()];
    for out in 0..layout.out_features {
        for group in 0..layout.groups() {
            let value = ((out * 3 + group) & 0x0f) as u32;
            zero_points[(out / 8) * layout.groups() + group] |= value << (4 * (out % 8));
        }
    }
    // Row-wise CPU reference using the same single BF16 weight rounding.
    let mut expected = Vec::with_capacity(rows * layout.out_features);
    for row in 0..rows {
        let slice = &activation[row * layout.in_features..(row + 1) * layout.in_features];
        expected.extend(cpu_reference_with_bf16_dequantized_weights(
            layout,
            slice,
            &weights,
            &scales,
            &zero_points,
        ));
    }

    let activation_gpu =
        upload_fp32_as_bf16(&ctx, &activation, vec![rows, layout.in_features]).unwrap();
    let scales_gpu =
        upload_fp32_as_bf16(&ctx, &scales, vec![layout.out_features, layout.groups()]).unwrap();
    let weights_gpu = upload_u32(&ctx, &weights);
    let zero_points_gpu = upload_u32(&ctx, &zero_points);
    let buffers = || Qwen35W4Buffers {
        weight_packed: &weights_gpu,
        scales: &scales_gpu,
        zero_points: &zero_points_gpu,
    };

    // Control: force the per-output GEMV kernel for every row count.
    let guard = PrefillGemmGuard::set("0");
    let gemv = project_bf16(&ctx, &activation_gpu, buffers(), layout).unwrap();
    let gemv_values = download_bf16_as_fp32(&gemv).unwrap();
    drop(guard);

    // Candidate: dequantize once, then tensor-core BF16 GEMM.
    let guard = PrefillGemmGuard::set("8");
    let gemm = project_bf16(&ctx, &activation_gpu, buffers(), layout).unwrap();
    let gemm_values = download_bf16_as_fp32(&gemm).unwrap();
    drop(guard);

    assert_eq!(gemv.shape().dims(), &[rows, layout.out_features]);
    assert_eq!(gemm.shape().dims(), &[rows, layout.out_features]);
    // The decisive check: the dequantized matrix must be bit-identical to the
    // weights the GEMV kernel reconstructs, so any output difference is
    // attributable to accumulation order alone rather than decompression.
    let dequantized = crate::kernels::qwen35_w4::dequantize_bf16(&ctx, buffers(), layout).unwrap();
    assert_eq!(
        dequantized.shape().dims(),
        &[layout.out_features, layout.in_features]
    );
    let dequantized_values = download_bf16_as_fp32(&dequantized).unwrap();
    let mut expected_weights = Vec::with_capacity(layout.out_features * layout.in_features);
    for out in 0..layout.out_features {
        for k in 0..layout.in_features {
            let group = k / layout.group_size;
            let packed = weights[out * layout.packed_k_columns() + k / 8];
            let quantized = ((packed >> (4 * (k % 8))) & 0x0f) as f32;
            let packed_zero = zero_points[(out / 8) * layout.groups() + group];
            let zero_point = ((packed_zero >> (4 * (out % 8))) & 0x0f) as f32;
            let scale = bf16::from_f32(scales[out * layout.groups() + group]).to_f32();
            expected_weights.push(bf16::from_f32((quantized - zero_point) * scale).to_f32());
        }
    }
    assert_eq!(
        dequantized_values, expected_weights,
        "dequantized W4 weights must be bit-identical to the reference decompression"
    );

    // Both paths round each weight to BF16 once; only the dot-product
    // accumulation order differs. The GEMV kernel reduces in a fixed
    // 256-thread tree while cuBLAS tiles the K loop, so compare the GEMM at
    // reduction tolerance and additionally bound the observed drift.
    assert_bf16_close_reduction(&gemv_values, &expected);
    let worst = gemm_values
        .iter()
        .zip(&gemv_values)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        worst <= 0.0625,
        "prefill GEMM drifted {worst} from the GEMV path, beyond BF16 reassociation"
    );

    // Below the threshold the candidate must fall back to the GEMV path and
    // produce exactly the control's bytes for the same single row.
    let single = upload_fp32_as_bf16(
        &ctx,
        &activation[..layout.in_features],
        vec![1, layout.in_features],
    )
    .unwrap();
    let guard = PrefillGemmGuard::set("8");
    let decode_like = project_bf16(&ctx, &single, buffers(), layout).unwrap();
    drop(guard);
    assert_eq!(
        download_bf16_as_fp32(&decode_like).unwrap(),
        gemv_values[..layout.out_features].to_vec()
    );
}

#[test]
fn qwen35_w4_cuda_projection_matches_k_and_n_packed_cpu_reference() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let layout = Qwen35W4Layout::new(9, 35, 32).unwrap();
    let activation: Vec<f32> = (0..layout.in_features)
        .map(|index| index as f32 * 0.03125 - 0.5)
        .collect();
    let mut weights = vec![0u32; layout.out_features * layout.packed_k_columns()];
    for out in 0..layout.out_features {
        for k in 0..layout.in_features {
            let value = ((out * 7 + k * 3) & 0x0f) as u32;
            weights[out * layout.packed_k_columns() + k / 8] |= value << (4 * (k % 8));
        }
    }
    let scales: Vec<f32> = (0..layout.out_features * layout.groups())
        .map(|index| 0.0625 * (1 + index) as f32)
        .collect();
    let mut zero_points = vec![0u32; layout.packed_n_rows() * layout.groups()];
    for out in 0..layout.out_features {
        for group in 0..layout.groups() {
            let value = ((out + 2 * group) & 0x0f) as u32;
            zero_points[(out / 8) * layout.groups() + group] |= value << (4 * (out % 8));
        }
    }
    let expected = cpu_reference(layout, &activation, &weights, &scales, &zero_points);
    let activation = upload_fp32_as_bf16(&ctx, &activation, vec![1, layout.in_features]).unwrap();
    let scales =
        upload_fp32_as_bf16(&ctx, &scales, vec![layout.out_features, layout.groups()]).unwrap();
    let weights = upload_u32(&ctx, &weights);
    let zero_points = upload_u32(&ctx, &zero_points);

    let output = project_bf16(
        &ctx,
        &activation,
        Qwen35W4Buffers {
            weight_packed: &weights,
            scales: &scales,
            zero_points: &zero_points,
        },
        layout,
    )
    .unwrap();
    assert_bf16_close_reduction(&download_bf16_as_fp32(&output).unwrap(), &expected);
}

#[test]
fn qwen35_w4_cuda_projection_matches_bf16_checkpoint_decompression_rounding() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let layout = Qwen35W4Layout::new(1, 2, 32).unwrap();
    let activation_values = [1.296875, 0.010009765625];
    let weights = [0x0000_0047u32];
    let scale_values = [0.010009765625f32];
    let zero_points = [0u32];
    let expected = cpu_reference_with_bf16_dequantized_weights(
        layout,
        &activation_values,
        &weights,
        &scale_values,
        &zero_points,
    );
    let activation = upload_fp32_as_bf16(&ctx, &activation_values, vec![1, 2]).unwrap();
    let scales = upload_fp32_as_bf16(&ctx, &scale_values, vec![1, 1]).unwrap();
    let weights = upload_u32(&ctx, &weights);
    let zero_points = upload_u32(&ctx, &zero_points);

    let output = project_bf16(
        &ctx,
        &activation,
        Qwen35W4Buffers {
            weight_packed: &weights,
            scales: &scales,
            zero_points: &zero_points,
        },
        layout,
    )
    .unwrap();
    assert_eq!(download_bf16_as_fp32(&output).unwrap(), expected);
}

#[test]
fn qwen35_w4_device_projection_owns_uploaded_w4_payloads() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let layout = Qwen35W4Layout::new(2, 3, 32).unwrap();
    let activation_values = [1.0, -0.5, 2.0];
    let weights = [0x0000_0321u32, 0x0000_0765u32];
    let scale_values = [0.25f32, 0.5f32];
    let zero_points = [0x0000_0021u32];
    let expected = cpu_reference(
        layout,
        &activation_values,
        &weights,
        &scale_values,
        &zero_points,
    );
    let scale_values: Vec<bf16> = scale_values.iter().copied().map(bf16::from_f32).collect();
    let scales = Tensor::from_bf16(vec![2, 1], &scale_values).unwrap();
    let projection = Qwen35W4DeviceProjection::upload(
        &ctx,
        layout,
        &weights
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>(),
        &scales,
        &zero_points
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let activation = upload_fp32_as_bf16(&ctx, &activation_values, vec![1, 3]).unwrap();
    let output = projection.project(&ctx, &activation).unwrap();
    assert_bf16_close_reduction(&download_bf16_as_fp32(&output).unwrap(), &expected);
}

#[test]
fn qwen35_w4_cuda_projection_rejects_non_finite_scale() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let layout = Qwen35W4Layout::new(1, 1, 32).unwrap();
    let activation = upload_fp32_as_bf16(&ctx, &[1.0], vec![1, 1]).unwrap();
    let scales = upload_fp32_as_bf16(&ctx, &[f32::INFINITY], vec![1, 1]).unwrap();
    let weights = upload_u32(&ctx, &[1]);
    let zero_points = upload_u32(&ctx, &[0]);

    let error = project_bf16(
        &ctx,
        &activation,
        Qwen35W4Buffers {
            weight_packed: &weights,
            scales: &scales,
            zero_points: &zero_points,
        },
        layout,
    )
    .unwrap_err();
    assert!(matches!(error, Error::Other(message) if message.contains("non-finite W4 scale")));
}

#[test]
fn qwen35_w4_cuda_projection_rejects_non_finite_output() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let layout = Qwen35W4Layout::new(1, 2, 32).unwrap();
    let activation = upload_fp32_as_bf16(&ctx, &[f32::MAX, f32::MAX], vec![1, 2]).unwrap();
    let scales = upload_fp32_as_bf16(&ctx, &[1.0], vec![1, 1]).unwrap();
    let weights = upload_u32(&ctx, &[0x0000_00ff]);
    let zero_points = upload_u32(&ctx, &[0]);

    let error = project_bf16(
        &ctx,
        &activation,
        Qwen35W4Buffers {
            weight_packed: &weights,
            scales: &scales,
            zero_points: &zero_points,
        },
        layout,
    )
    .unwrap_err();
    assert!(matches!(error, Error::Other(message) if message.contains("non-finite W4 output")));
}

#[test]
fn qwen35_w4_cuda_projection_rejects_bf16_conversion_overflow() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let layout = Qwen35W4Layout::new(1, 2, 32).unwrap();
    let activation = upload_fp32_as_bf16(&ctx, &[3.3895e38, 7.0e35], vec![1, 2]).unwrap();
    let scales = upload_fp32_as_bf16(&ctx, &[1.0], vec![1, 1]).unwrap();
    let weights = upload_u32(&ctx, &[0x0000_0011]);
    let zero_points = upload_u32(&ctx, &[0]);

    let error = project_bf16(
        &ctx,
        &activation,
        Qwen35W4Buffers {
            weight_packed: &weights,
            scales: &scales,
            zero_points: &zero_points,
        },
        layout,
    )
    .unwrap_err();
    assert!(matches!(error, Error::Other(message) if message.contains("non-finite W4 output")));
}

#[test]
fn qwen35_w4_cuda_projection_rejects_short_activation_storage() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let layout = Qwen35W4Layout::new(1, 2, 32).unwrap();
    let short = CudaBuffer::alloc(2, ctx.device_id()).unwrap();
    let handle = GpuStorageHandle {
        ptr: short.ptr() as usize,
        len: short.len(),
        _prevent_leak: Some(std::sync::Arc::new(short)),
    };
    let device = Device::Cuda(ctx.device_id());
    let activation = Tensor::from_raw_parts(
        Shape::new(vec![1, 2]),
        DType::BF16,
        device,
        Storage::Gpu { device, handle },
    );
    let scales = upload_fp32_as_bf16(&ctx, &[1.0], vec![1, 1]).unwrap();
    let weights = upload_u32(&ctx, &[0]);
    let zero_points = upload_u32(&ctx, &[0]);

    let error = project_bf16(
        &ctx,
        &activation,
        Qwen35W4Buffers {
            weight_packed: &weights,
            scales: &scales,
            zero_points: &zero_points,
        },
        layout,
    )
    .unwrap_err();
    assert!(
        matches!(error, Error::Other(message) if message.contains("activation") && message.contains("bytes"))
    );
}

#[test]
fn qwen35_w4_cuda_projection_rejects_short_scale_storage() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let layout = Qwen35W4Layout::new(2, 1, 32).unwrap();
    let activation = upload_fp32_as_bf16(&ctx, &[1.0], vec![1, 1]).unwrap();
    let short = CudaBuffer::alloc(2, ctx.device_id()).unwrap();
    let handle = GpuStorageHandle {
        ptr: short.ptr() as usize,
        len: short.len(),
        _prevent_leak: Some(std::sync::Arc::new(short)),
    };
    let device = Device::Cuda(ctx.device_id());
    let scales = Tensor::from_raw_parts(
        Shape::new(vec![2, 1]),
        DType::BF16,
        device,
        Storage::Gpu { device, handle },
    );
    let weights = upload_u32(&ctx, &[0, 0]);
    let zero_points = upload_u32(&ctx, &[0]);

    let error = project_bf16(
        &ctx,
        &activation,
        Qwen35W4Buffers {
            weight_packed: &weights,
            scales: &scales,
            zero_points: &zero_points,
        },
        layout,
    )
    .unwrap_err();
    assert!(
        matches!(error, Error::Other(message) if message.contains("scale") && message.contains("bytes"))
    );
}

#[test]
fn qwen35_w4_cuda_projection_validates_shape_dtype_and_storage() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let layout = Qwen35W4Layout::new(2, 9, 32).unwrap();
    let activation = upload_fp32_as_bf16(&ctx, &[1.0; 8], vec![1, 8]).unwrap();
    let scales = upload_fp32_as_bf16(&ctx, &[1.0; 2], vec![2, 1]).unwrap();
    let weights = upload_u32(&ctx, &[0; 4]);
    let zero_points = upload_u32(&ctx, &[0; 1]);

    let error = project_bf16(
        &ctx,
        &activation,
        Qwen35W4Buffers {
            weight_packed: &weights,
            scales: &scales,
            zero_points: &zero_points,
        },
        layout,
    )
    .unwrap_err();
    assert!(matches!(error, Error::Other(message) if message.contains("activation shape")));
}
