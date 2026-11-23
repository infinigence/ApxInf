use apxinf_core::DType;
use apxinf_cuda_new::{
    capture,
    ops::{rope_apply_fractional_mrope_into, rope_apply_mrope},
    CudaBuffer, CudaContext,
};
use half::bf16;

#[test]
fn fractional_positions_keep_integer_parity_and_rebind_in_graph() {
    let ctx = CudaContext::new(0).unwrap();
    let values: Vec<u8> = (0..256)
        .flat_map(|i| {
            bf16::from_f32((i % 7) as f32 * 0.25)
                .to_bits()
                .to_ne_bytes()
        })
        .collect();
    let input_buffer = CudaBuffer::alloc(values.len(), 0).unwrap();
    input_buffer.copy_from_host(&values).unwrap();
    let input = input_buffer
        .as_tensor(vec![2, 1, 128].into(), DType::BF16)
        .unwrap();
    let output_buffer = CudaBuffer::alloc(values.len(), 0).unwrap();
    let output = output_buffer
        .as_tensor(vec![2, 1, 128].into(), DType::BF16)
        .unwrap();
    let ids = [3u32, 1, 2, 4, 2, 3];
    let reference = rope_apply_mrope(&ctx, &input, 1, 128, 1e8, [24, 20, 20], &ids).unwrap();
    let positions = CudaBuffer::alloc(24, 0).unwrap();
    let upload = |ids: &[f32]| {
        let bytes: Vec<u8> = ids.iter().flat_map(|v| v.to_ne_bytes()).collect();
        positions.copy_from_host(&bytes).unwrap();
    };
    let read = |tensor: &apxinf_core::Tensor| {
        ctx.synchronize().unwrap();
        let mut bytes = vec![0; tensor.size_in_bytes()];
        CudaBuffer::from_tensor(tensor)
            .unwrap()
            .copy_to_host(&mut bytes)
            .unwrap();
        bytes
    };
    upload(&ids.map(|v| v as f32));
    let launch = || {
        rope_apply_fractional_mrope_into(
            &ctx,
            &input,
            &output,
            1,
            128,
            1e8,
            [24, 20, 20],
            &positions,
        )
    };
    launch().unwrap();
    let expected = read(&reference);
    assert_eq!(read(&output), expected);
    let graph = capture(&ctx, launch).unwrap();
    upload(&[0.5, 1.0, 2.0, 1.5, 2.0, 3.0]);
    graph.replay().unwrap();
    let fractional = read(&output);
    assert_ne!(fractional, expected);
    // The first frequency pair uses the unscaled temporal position. The
    // independent host expression agrees within one BF16 rounding interval.
    let got = bf16::from_bits(u16::from_ne_bytes(fractional[..2].try_into().unwrap())).to_f32();
    let x0 = 0.0f32;
    let x1 = (64 % 7) as f32 * 0.25;
    let want = bf16::from_f32(x0 * 0.5f32.cos() - x1 * 0.5f32.sin()).to_f32();
    assert!((got - want).abs() <= 0.002);
    upload(&ids.map(|v| v as f32));
    graph.replay().unwrap();
    assert_eq!(read(&output), expected);
}
