use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda_new::{ops, CudaBuffer, CudaContext};

fn bf16_tensor(ctx: &CudaContext, shape: Vec<usize>, values: &[f32]) -> Tensor {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| half::bf16::from_f32(*value).to_bits().to_le_bytes())
        .collect();
    let buffer = CudaBuffer::alloc(bytes.len(), ctx.device_id()).unwrap();
    buffer.copy_from_host(&bytes).unwrap();
    buffer.as_tensor(Shape::new(shape), DType::BF16).unwrap()
}

fn bits(tensor: &Tensor) -> Vec<u16> {
    let mut bytes = vec![0; tensor.shape().numel() * 2];
    CudaBuffer::from_tensor(tensor)
        .unwrap()
        .copy_to_host(&mut bytes)
        .unwrap();
    bytes
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect()
}

#[test]
fn head_rms_reduction_contract_is_measured_before_family_integration() {
    let ctx = CudaContext::new(0).unwrap();
    for (rows, cols) in [(8, 128), (16, 128), (8, 256)] {
        let values: Vec<f32> = (0..rows * cols)
            .map(|index| ((index * 37 % 509) as f32 - 254.0) / 127.0)
            .collect();
        let weights: Vec<f32> = (0..cols)
            .map(|index| 0.75 + (index * 11 % 67) as f32 / 128.0)
            .collect();
        let head = bf16_tensor(&ctx, vec![rows, cols], &values);
        let generic_input = bf16_tensor(&ctx, vec![rows, cols], &values);
        let weight = bf16_tensor(&ctx, vec![cols], &weights);
        let generic = bf16_tensor(&ctx, vec![rows, cols], &vec![0.0; rows * cols]);

        ops::norm::norm(
            &ctx,
            ops::NormArgs::rms(&head, &weight, &head, 1e-6).with_weight_offset(),
        )
        .unwrap();
        ops::norm::norm(
            &ctx,
            ops::NormArgs::rms(&generic_input, &weight, &generic, 1e-6).with_weight_offset(),
        )
        .unwrap();
        ctx.synchronize().unwrap();

        let head_bits = bits(&head);
        let generic_bits = bits(&generic);
        let differences = head_bits
            .iter()
            .zip(&generic_bits)
            .filter(|(head, generic)| head != generic)
            .count();
        println!("rows={rows} cols={cols} bitwise differences={differences}");
        assert_eq!(head_bits, generic_bits);
    }
}
