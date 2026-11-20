use apxinf_core::DType;
use apxinf_cuda_new::{capture, ops::rgb_patch2x2, CudaBuffer, CudaContext, ExecutionSession};
use half::bf16;

#[test]
fn rectangular_rgb_patches_preserve_rounding_and_mutable_graph_inputs() {
    let ctx = CudaContext::new(0).unwrap();
    // Two rectangular frames include every u8 value and cross frame boundaries.
    let (frames, height, width) = (2, 8, 12);
    let count = frames * height * width * 3;
    let input = CudaBuffer::alloc(count, 0).unwrap();
    let storage = CudaBuffer::alloc(count * 2, 0).unwrap();
    let output = storage
        .as_tensor(vec![frames, height / 2, width / 2, 12].into(), DType::BF16)
        .unwrap();
    let bytes: Vec<u8> = (0..count).map(|i| (i % 256) as u8).collect();
    let expected = |bytes: &[u8]| {
        let mut out = Vec::with_capacity(count * 2);
        for f in 0..frames {
            for y in (0..height).step_by(2) {
                for x in (0..width).step_by(2) {
                    for c in 0..3 {
                        for dx in 0..2 {
                            for dy in 0..2 {
                                let value = bytes[((f * height + y + dy) * width + x + dx) * 3 + c];
                                out.extend_from_slice(
                                    &bf16::from_f32(f32::from(value) / 127.5 - 1.0)
                                        .to_bits()
                                        .to_ne_bytes(),
                                );
                            }
                        }
                    }
                }
            }
        }
        out
    };
    let read = || {
        ctx.synchronize().unwrap();
        let mut bytes = vec![0; count * 2];
        storage.copy_to_host(&mut bytes).unwrap();
        bytes
    };
    input.copy_from_host(&bytes).unwrap();
    rgb_patch2x2(&ctx, &input, &output).unwrap();
    assert_eq!(read(), expected(&bytes));
    let session = ExecutionSession::with_capacity(4096, 0).unwrap();
    session
        .prepare(|| rgb_patch2x2(&ctx, &input, &output))
        .unwrap();
    session.run(|| rgb_patch2x2(&ctx, &input, &output)).unwrap();
    let graph = capture(&ctx, || session.run(|| rgb_patch2x2(&ctx, &input, &output))).unwrap();
    let changed: Vec<_> = bytes.iter().map(|v| 255 - v).collect();
    for values in [&changed, &bytes, &changed] {
        input.copy_from_host(values).unwrap();
        graph.replay().unwrap();
        assert_eq!(read(), expected(values));
    }
    let wrong_storage = CudaBuffer::alloc(count, 0).unwrap();
    let wrong_shape = wrong_storage
        .as_tensor(vec![frames, height / 2, width / 2, 6].into(), DType::BF16)
        .unwrap();
    assert!(rgb_patch2x2(&ctx, &input, &wrong_shape).is_err());
}
