use apxinf_core::{DType, Shape};
use apxinf_cuda_new::{capture, ops::embedding_gather, CudaBuffer, CudaContext};
#[test]
fn gather_vector_and_scalar_paths_preserve_bits_clamping_and_graph() {
    let ctx = CudaContext::new(0).unwrap();
    for width in [1, 7, 8, 96, 384, 2048] {
        for (offset, output_offset) in [(0, 0), (0, 2), (2, 0), (2, 2)] {
            let rows = 17;
            let vocab = 27;
            let values: Vec<u16> = (0..vocab * width).map(|i| (i * 37) as u16).collect();
            let table_buf = CudaBuffer::alloc(values.len() * 2 + offset, 0).unwrap();
            let view = table_buf.view(offset, values.len() * 2).unwrap();
            view.copy_from_host(
                &values
                    .iter()
                    .flat_map(|v| v.to_ne_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            let table = view
                .as_tensor(Shape::new(vec![vocab, width]), DType::BF16)
                .unwrap();
            let ids: Vec<i32> = (0..rows)
                .map(|i| {
                    if i == 0 {
                        -1
                    } else if i == 1 {
                        vocab as i32
                    } else {
                        (i * 7 % vocab) as i32
                    }
                })
                .collect();
            let idbuf = CudaBuffer::alloc(rows * 4, 0).unwrap();
            idbuf
                .copy_from_host(&ids.iter().flat_map(|i| i.to_ne_bytes()).collect::<Vec<_>>())
                .unwrap();
            let indices = idbuf.as_tensor(Shape::new(vec![rows]), DType::I32).unwrap();
            let outbuf = CudaBuffer::alloc(rows * width * 2 + output_offset, 0).unwrap();
            let outview = outbuf.view(output_offset, rows * width * 2).unwrap();
            let output = outview
                .as_tensor(Shape::new(vec![rows, width]), DType::BF16)
                .unwrap();
            let graph =
                capture(&ctx, || embedding_gather(&ctx, &table, &indices, &output)).unwrap();
            for _ in 0..2 {
                graph.replay().unwrap();
                ctx.synchronize().unwrap();
                let mut bytes = vec![0; rows * width * 2];
                outview.copy_to_host(&mut bytes).unwrap();
                let got: Vec<u16> = bytes
                    .chunks_exact(2)
                    .map(|b| u16::from_ne_bytes(b.try_into().unwrap()))
                    .collect();
                let expected: Vec<u16> = ids
                    .iter()
                    .flat_map(|&id| {
                        let id = if id < 0 || id >= vocab as i32 {
                            0
                        } else {
                            id as usize
                        };
                        values[id * width..(id + 1) * width].iter().copied()
                    })
                    .collect();
                assert_eq!(
                    got, expected,
                    "width={width},offset={offset},output_offset={output_offset}"
                );
            }
        }
    }
}
