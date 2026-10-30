//! Public volume lowering, including causal history and captured ownership.
use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda_new::{
    capture, CudaBuffer, CudaContext, PreparedUnfold3d, Unfold3dArgs, Unfold3dGeometry,
};
use half::bf16;

fn upload(shape: Vec<usize>, words: &[u16]) -> Tensor {
    let bytes: Vec<u8> = words.iter().flat_map(|v| v.to_le_bytes()).collect();
    let buffer = CudaBuffer::alloc(bytes.len(), 0).unwrap();
    buffer.copy_from_host(&bytes).unwrap();
    buffer.as_tensor(Shape::new(shape), DType::BF16).unwrap()
}

fn download(tensor: &Tensor) -> Vec<u16> {
    let mut bytes = vec![0; tensor.size_in_bytes()];
    CudaBuffer::from_tensor(tensor)
        .unwrap()
        .copy_to_host(&mut bytes)
        .unwrap();
    bytes
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect()
}

fn values(n: usize, phase: usize) -> Vec<u16> {
    (0..n)
        .map(|i| bf16::from_f32(((i + phase) % 113) as f32 - 56.).to_bits())
        .collect()
}

// Independent dense padded-volume construction, followed by neighborhood
// extraction. Deliberately does not share the kernel's flat index formulas.
fn expected(
    input: &[u16],
    history: &[u16],
    shape: [usize; 4],
    history_t: usize,
    g: Unfold3dGeometry,
) -> (Vec<u16>, usize) {
    let [t, h, w, c] = shape;
    let [pt, ph, pw] = std::array::from_fn(|i| {
        shape[i] + g.padding_before[i] + g.padding_after[i] + if i == 0 { history_t } else { 0 }
    });
    let mut volume = vec![vec![vec![vec![0u16; c]; pw]; ph]; pt];
    for (frame, source) in history
        .chunks_exact(h * w * c)
        .chain(input.chunks_exact(h * w * c))
        .enumerate()
    {
        for (row, line) in source.chunks_exact(w * c).enumerate() {
            for (col, pixel) in line.chunks_exact(c).enumerate() {
                volume[frame + g.padding_before[0]][row + g.padding_before[1]]
                    [col + g.padding_before[2]]
                    .copy_from_slice(pixel);
            }
        }
    }
    assert_eq!(
        t + history_t,
        history.len() / (h * w * c) + input.len() / (h * w * c)
    );
    let mut out = Vec::new();
    for z in (0..=pt - g.kernel[0]).step_by(g.stride[0]) {
        for y in (0..=ph - g.kernel[1]).step_by(g.stride[1]) {
            for x in (0..=pw - g.kernel[2]).step_by(g.stride[2]) {
                for slab in &volume[z..z + g.kernel[0]] {
                    for line in &slab[y..y + g.kernel[1]] {
                        for pixel in &line[x..x + g.kernel[2]] {
                            out.extend_from_slice(pixel);
                        }
                    }
                }
            }
        }
    }
    (out, g.kernel.iter().product::<usize>() * c)
}

#[test]
fn tiled_unfold_history_padding_stride_and_graph_rebinding() {
    let ctx = CudaContext::new(0).unwrap();
    let cases = [
        ([1, 7, 9, 3], 0, [3, 3, 3], [1, 1, 1], [2, 1, 1], [0, 1, 1]),
        ([4, 7, 9, 96], 2, [3, 3, 3], [1, 1, 1], [0, 1, 1], [0, 1, 1]),
        (
            [4, 5, 7, 192],
            1,
            [3, 1, 1],
            [2, 1, 1],
            [0, 0, 0],
            [0, 0, 0],
        ),
        (
            [2, 8, 8, 384],
            0,
            [1, 3, 3],
            [1, 2, 2],
            [0, 0, 0],
            [0, 1, 1],
        ),
        ([3, 5, 7, 32], 0, [1, 1, 1], [1; 3], [0; 3], [0; 3]),
        ([3, 5, 7, 96], 0, [2, 2, 1], [1; 3], [0; 3], [0; 3]),
        ([3, 5, 7, 128], 0, [3; 3], [1; 3], [1; 3], [1; 3]),
    ];
    for (shape, ht, kernel, stride, padding_before, padding_after) in cases {
        let geometry = Unfold3dGeometry {
            kernel,
            stride,
            padding_before,
            padding_after,
        };
        let source = values(shape.iter().product(), 0);
        let cache = values(ht * shape[1] * shape[2] * shape[3], 19);
        let (gold, columns) = expected(&source, &cache, shape, ht, geometry);
        let input = upload(shape.to_vec(), &source);
        let history = (ht > 0).then(|| upload(vec![ht, shape[1], shape[2], shape[3]], &cache));
        let rows = gold.len() / columns;
        // Nonzero offset and a partial last tile exercise bounds independently
        // from the convolution's total number of output rows.
        let start = rows / 3;
        let count = (rows - start).min(37);
        let mut output = upload(vec![count, columns], &vec![0; count * columns]);
        let operation = PreparedUnfold3d::new(
            &ctx,
            Unfold3dArgs {
                input: &input,
                history: history.as_ref(),
                geometry,
                row_start: start,
                out: &mut output,
            },
        )
        .unwrap();
        operation.enqueue().unwrap();
        ctx.synchronize().unwrap();
        assert_eq!(
            download(&output),
            gold[start * columns..(start + count) * columns]
        );
        let graph = capture(&ctx, || operation.enqueue()).unwrap();
        let changed = values(source.len(), 37);
        let bytes: Vec<u8> = changed.iter().flat_map(|v| v.to_le_bytes()).collect();
        CudaBuffer::from_tensor(&input)
            .unwrap()
            .copy_from_host(&bytes)
            .unwrap();
        let (changed_gold, _) = expected(&changed, &cache, shape, ht, geometry);
        drop(operation);
        drop(input);
        drop(history);
        for _ in 0..3 {
            graph.replay().unwrap();
            ctx.synchronize().unwrap();
            assert_eq!(
                download(&output),
                changed_gold[start * columns..(start + count) * columns]
            );
        }
    }
}

#[test]
fn unfold_accepts_bf16_aligned_views_without_vector_alignment() {
    let ctx = CudaContext::new(0).unwrap();
    let shape = [3, 5, 7, 96];
    let source = values(shape.iter().product(), 0);
    let geometry = Unfold3dGeometry {
        kernel: [3; 3],
        stride: [1; 3],
        padding_before: [1; 3],
        padding_after: [1; 3],
    };
    let (gold, columns) = expected(&source, &[], shape, 0, geometry);
    let input = upload(shape.to_vec(), &source);
    // A two-byte output offset is valid BF16 storage. Fast vector copies must
    // not impose a stronger public alignment contract or overwrite the guard.
    let storage = upload(vec![gold.len() + 1], &vec![0x3f80; gold.len() + 1]);
    let mut output = CudaBuffer::from_tensor(&storage)
        .unwrap()
        .view(2, gold.len() * 2)
        .unwrap()
        .as_tensor(Shape::new(vec![gold.len() / columns, columns]), DType::BF16)
        .unwrap();
    let operation = PreparedUnfold3d::new(
        &ctx,
        Unfold3dArgs {
            input: &input,
            history: None,
            geometry,
            row_start: 0,
            out: &mut output,
        },
    )
    .unwrap();
    operation.enqueue().unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(download(&output), gold);
    assert_eq!(download(&storage)[0], 0x3f80);
}

#[test]
fn unfold_rejects_invalid_shapes_alias_and_capture_preparation() {
    let ctx = CudaContext::new(0).unwrap();
    let input = upload(vec![1, 2, 2, 1], &[0; 4]);
    let mut output = upload(vec![4, 1], &[0; 4]);
    let geometry = Unfold3dGeometry {
        kernel: [1; 3],
        stride: [1; 3],
        padding_before: [0; 3],
        padding_after: [0; 3],
    };
    assert!(PreparedUnfold3d::new(
        &ctx,
        Unfold3dArgs {
            input: &input,
            history: None,
            geometry,
            row_start: 1,
            out: &mut output,
        }
    )
    .is_err());
    let mut alias = input.reshape(vec![4, 1]).unwrap();
    assert!(PreparedUnfold3d::new(
        &ctx,
        Unfold3dArgs {
            input: &input,
            history: None,
            geometry,
            row_start: 0,
            out: &mut alias,
        }
    )
    .is_err());
    let error = capture(&ctx, || {
        PreparedUnfold3d::new(
            &ctx,
            Unfold3dArgs {
                input: &input,
                history: None,
                geometry,
                row_start: 0,
                out: &mut output,
            },
        )
        .map(|_| ())
    })
    .err()
    .expect("preparation must fail during capture");
    assert!(error.to_string().contains("before capture"));
    let operation = PreparedUnfold3d::new(
        &ctx,
        Unfold3dArgs {
            input: &input,
            history: None,
            geometry,
            row_start: 0,
            out: &mut output,
        },
    )
    .unwrap();
    let other = CudaContext::new(0).unwrap();
    assert!(capture(&other, || operation.enqueue()).is_err());
    capture(&ctx, || operation.enqueue())
        .unwrap()
        .replay()
        .unwrap();
    ctx.synchronize().unwrap();
}
