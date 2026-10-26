//! Small CUDA contracts used by the Qwen3.5 vision tower.

use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda_new::{ops, CudaBuffer, CudaContext};

fn upload(ctx: &CudaContext, bytes: &[u8], dims: Vec<usize>, dtype: DType) -> Tensor {
    let buffer = CudaBuffer::alloc(bytes.len(), ctx.device_id()).unwrap();
    buffer.copy_from_host(bytes).unwrap();
    buffer.as_tensor(Shape::new(dims), dtype).unwrap()
}

fn upload_f32(ctx: &CudaContext, values: &[f32], dims: Vec<usize>) -> Tensor {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    upload(ctx, &bytes, dims, DType::F32)
}

fn upload_bf16(ctx: &CudaContext, values: &[f32], dims: Vec<usize>) -> Tensor {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| half::bf16::from_f32(*value).to_bits().to_le_bytes())
        .collect();
    upload(ctx, &bytes, dims, DType::BF16)
}

fn upload_u32(ctx: &CudaContext, values: &[u32]) -> CudaBuffer {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let buffer = CudaBuffer::alloc(bytes.len(), ctx.device_id()).unwrap();
    buffer.copy_from_host(&bytes).unwrap();
    buffer
}

fn read_bf16(tensor: &Tensor) -> Vec<f32> {
    let buffer = CudaBuffer::from_tensor(tensor).unwrap();
    let mut bytes = vec![0u8; buffer.len()];
    buffer.copy_to_host(&mut bytes).unwrap();
    bytes
        .chunks_exact(2)
        .map(|value| half::bf16::from_bits(u16::from_le_bytes([value[0], value[1]])).to_f32())
        .collect()
}

#[test]
fn f32_cast_and_exact_gelu_match_host_reference() {
    let ctx = CudaContext::new(0).unwrap();
    let values = [-4.0, -1.25, -0.0, 0.5, 1.0, 3.25];
    let input = upload_f32(&ctx, &values, vec![1, values.len()]);
    let mut rounded = upload_bf16(&ctx, &vec![0.0; values.len()], vec![1, values.len()]);
    let mut activated = upload_bf16(&ctx, &vec![0.0; values.len()], vec![1, values.len()]);

    ops::quantization(
        &ctx,
        ops::QuantizationArgs::new(
            ops::QuantizationSemantic::CastF32ToBf16,
            &input,
            &mut rounded,
        ),
    )
    .unwrap();
    let mut activation = ops::PointwiseArgs::new(
        ops::PointwiseSemantic::BiasActivation,
        &rounded,
        &mut activated,
    );
    activation.activation = ops::PointwiseActivation::GeluExact;
    ops::pointwise(&ctx, activation).unwrap();
    ctx.synchronize().unwrap();

    assert_eq!(
        read_bf16(&rounded),
        values
            .iter()
            .map(|value| half::bf16::from_f32(*value).to_f32())
            .collect::<Vec<_>>()
    );
    let expected = [-0.0001267, -0.132061, -0.0, 0.345731, 0.841345, 3.248125];
    for (got, expected) in read_bf16(&activated).into_iter().zip(expected) {
        assert!((got - expected).abs() < 2e-2);
    }
}

#[test]
fn row_replacement_preserves_unselected_rows() {
    let ctx = CudaContext::new(0).unwrap();
    let base_values: Vec<f32> = (0..12).map(|value| value as f32).collect();
    let replacement_values = [101.0, 102.0, 103.0, 104.0];
    let base = upload_bf16(&ctx, &base_values, vec![3, 4]);
    let replacement = upload_bf16(&ctx, &replacement_values, vec![1, 4]);
    let row_map = upload_u32(&ctx, &[u32::MAX, 0, u32::MAX]);
    let mut output = upload_bf16(&ctx, &vec![0.0; 12], vec![3, 4]);

    let mut args = ops::GatherArgs::new(ops::GatherSemantic::ReplaceRows, &base, &mut output);
    args.replacement = Some(&replacement);
    args.row_map = Some(&row_map);
    ops::gather(&ctx, args).unwrap();
    ctx.synchronize().unwrap();

    assert_eq!(
        read_bf16(&output),
        vec![0.0, 1.0, 2.0, 3.0, 101.0, 102.0, 103.0, 104.0, 8.0, 9.0, 10.0, 11.0,]
    );
}
