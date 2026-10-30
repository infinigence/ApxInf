use super::*;
use crate::{transfers, CudaContext};
use apxinf_core::{DType, Tensor};

fn upload(ctx: &CudaContext, shape: Vec<usize>, values: &[f32], dtype: DType) -> Tensor {
    let host = match dtype {
        DType::F32 => Tensor::from_f32(shape, values),
        DType::F16 => Tensor::from_f16(
            shape,
            &values
                .iter()
                .copied()
                .map(half::f16::from_f32)
                .collect::<Vec<_>>(),
        ),
        DType::BF16 => Tensor::from_bf16(
            shape,
            &values
                .iter()
                .copied()
                .map(half::bf16::from_f32)
                .collect::<Vec<_>>(),
        ),
        _ => unreachable!(),
    }
    .unwrap();
    transfers::to_cuda(&host, ctx.device_id()).unwrap()
}

#[test]
fn unfold3d_causal_padding_dilation_and_tiles_match_scalar_indexing() {
    let ctx = CudaContext::new(0).unwrap();
    let geometry = Unfold3dGeometry {
        kernel: [3, 2, 3],
        stride: [2, 1, 2],
        dilation: [1, 2, 1],
        padding_before: [2, 1, 0],
        padding_after: [0, 1, 1],
    };
    for channels in [3usize, 8, 12] {
        let input_shape = [2, 5, 4, 5, channels];
        assert_eq!(geometry.output_spatial([5, 4, 5]).unwrap(), [3, 4, 2]);
        let source: Vec<_> = (0..input_shape.iter().product())
            .map(|i| ((i % 101) as f32 - 50.) / 8.)
            .collect();
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let input = upload(&ctx, input_shape.to_vec(), &source, dtype);
            for padded_channels in [channels, channels + 2, channels.next_power_of_two() * 2] {
                let cols = 3 * 2 * 3 * padded_channels;
                for (start, rows) in [(0, 48), (7, 13), (47, 1)] {
                    let mut out = upload(&ctx, vec![rows, cols], &vec![0.; rows * cols], dtype);
                    unfold3d(&ctx, &input, geometry, start, &mut out).unwrap();
                    ctx.synchronize().unwrap();
                    let actual = transfers::to_cpu(&out).unwrap().to_f32_vec().unwrap();
                    let mut expected = Vec::new();
                    for row in start..start + rows {
                        let b = row / (3 * 4 * 2);
                        let t = (row / (4 * 2)) % 3;
                        let h = (row / 2) % 4;
                        let w = row % 2;
                        for kt in 0..3 {
                            for kh in 0..2 {
                                for kw in 0..3 {
                                    for c in 0..padded_channels {
                                        let it = (t * 2 + kt) as isize - 2;
                                        let ih = (h + kh * 2) as isize - 1;
                                        let iw = (w * 2 + kw) as isize;
                                        expected.push(
                                            if c < channels
                                                && (0..5).contains(&it)
                                                && (0..4).contains(&ih)
                                                && (0..5).contains(&iw)
                                            {
                                                source[(((b * 5 + it as usize) * 4 + ih as usize)
                                                    * 5
                                                    + iw as usize)
                                                    * channels
                                                    + c]
                                            } else {
                                                0.
                                            },
                                        );
                                    }
                                }
                            }
                        }
                    }
                    assert_eq!(
                        actual,
                        expected,
                        "{dtype:?}, rows {start}..{}",
                        start + rows
                    );
                }
            }
            let cols = 3 * 2 * 3 * channels;
            for invalid_cols in [cols - 1, 3 * 2 * 3 * (channels - 1)] {
                let mut invalid =
                    upload(&ctx, vec![1, invalid_cols], &vec![0.; invalid_cols], dtype);
                assert!(unfold3d(&ctx, &input, geometry, 0, &mut invalid).is_err());
            }
            let mut invalid = upload(&ctx, vec![2, cols], &vec![0.; 2 * cols], dtype);
            assert!(unfold3d(&ctx, &input, geometry, 47, &mut invalid).is_err());
            let one = Unfold3dGeometry {
                kernel: [1, 1, 1],
                stride: [1, 1, 1],
                dilation: [1, 1, 1],
                padding_before: [0, 0, 0],
                padding_after: [0, 0, 0],
            };
            // Correct geometry but overlapping output view must be rejected.
            let mut alias = input.reshape(vec![200, channels]).unwrap();
            assert!(unfold3d(&ctx, &input, one, 0, &mut alias)
                .unwrap_err()
                .to_string()
                .contains("overlap"));
        }
    }
}

#[test]
fn permutation_and_unfold_graph_replay_changed_input_after_owner_drop() {
    let ctx = CudaContext::new(0).unwrap();
    let input = upload(
        &ctx,
        vec![1, 2, 2, 2, 3],
        &(0..24).map(|x| x as f32).collect::<Vec<_>>(),
        DType::BF16,
    );
    let mut output = upload(&ctx, vec![1, 2, 2, 3, 2], &vec![0.; 24], DType::BF16);
    let axes = [0, 2, 3, 4, 1];
    permute(&ctx, &input, &axes, &mut output).unwrap();
    let actual = transfers::to_cpu(&output).unwrap().to_f32_vec().unwrap();
    let expected: Vec<_> = (0..12).flat_map(|i| [i as f32, (12 + i) as f32]).collect();
    assert_eq!(actual, expected);
    assert!(permute(&ctx, &input, &[0, 0, 2, 3, 4], &mut output).is_err());
    let geometry = Unfold3dGeometry {
        kernel: [2, 1, 1],
        stride: [1, 1, 1],
        dilation: [1, 1, 1],
        padding_before: [1, 0, 0],
        padding_after: [0, 0, 0],
    };
    let mut windows = upload(&ctx, vec![12, 8], &vec![23.; 96], DType::BF16);
    let graph = crate::capture(&ctx, || {
        permute(&ctx, &input, &axes, &mut output)?;
        unfold3d(&ctx, &output, geometry, 0, &mut windows)
    })
    .unwrap();
    ctx.synchronize().unwrap();
    let changed =
        Tensor::from_bf16(vec![1, 2, 2, 2, 3], &vec![half::bf16::from_f32(7.); 24]).unwrap();
    transfers::copy_cpu_to_cuda(&changed, &input).unwrap();
    drop(input);
    drop(output);
    for _ in 0..3 {
        graph.replay().unwrap();
        ctx.synchronize().unwrap();
        let actual = transfers::to_cpu(&windows).unwrap().to_f32_vec().unwrap();
        let expected: Vec<_> = (0..12)
            .flat_map(|row| {
                if row < 6 {
                    [0., 0., 0., 0., 7., 7., 0., 0.]
                } else {
                    [7., 7., 0., 0., 7., 7., 0., 0.]
                }
            })
            .collect();
        assert_eq!(actual, expected);
    }
}

#[test]
fn unfold_geometry_rejects_empty_and_overflowing_extents() {
    let mut geometry = Unfold3dGeometry {
        kernel: [3; 3],
        stride: [1; 3],
        dilation: [1; 3],
        padding_before: [0; 3],
        padding_after: [0; 3],
    };
    assert!(geometry.output_spatial([1; 3]).is_err());
    geometry.padding_before = [usize::MAX; 3];
    assert!(geometry.output_spatial([4; 3]).is_err());
    geometry.padding_before = [0; 3];
    geometry.stride = [0; 3];
    assert!(geometry.output_spatial([4; 3]).is_err());
}

#[test]
fn blocked_unfold_padding_and_changed_input_capture_match_scalar_indexing() {
    let ctx = CudaContext::new(0).unwrap();
    let geometry = Unfold3dGeometry {
        kernel: [3; 3],
        stride: [1; 3],
        dilation: [1; 3],
        padding_before: [2, 1, 1],
        padding_after: [0, 1, 1],
    };
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        for (channels, block) in [(3usize, 2usize), (96, 64), (192, 64), (384, 64)] {
            let padded = channels.div_ceil(block) * block;
            let source: Vec<_> = (0..27 * channels)
                .map(|i| ((i % 101) as f32 - 50.) / 8.)
                .collect();
            let input = upload(&ctx, vec![1, 3, 3, 3, channels], &source, dtype);
            let mut output = upload(
                &ctx,
                vec![7, 27 * padded],
                &vec![0.; 7 * 27 * padded],
                dtype,
            );
            let expected = |sign: f32| {
                let mut result = Vec::new();
                for row in 4..11 {
                    for group in 0..padded / block {
                        for kt in 0..3 {
                            for kh in 0..3 {
                                for kw in 0..3 {
                                    for lane in 0..block {
                                        let c = group * block + lane;
                                        let t = (row / 9 + kt) as isize - 2;
                                        let h = ((row / 3) % 3 + kh) as isize - 1;
                                        let w = (row % 3 + kw) as isize - 1;
                                        result.push(
                                            if c < channels
                                                && (0..3).contains(&t)
                                                && (0..3).contains(&h)
                                                && (0..3).contains(&w)
                                            {
                                                sign * source[((t as usize * 3 + h as usize) * 3
                                                    + w as usize)
                                                    * channels
                                                    + c]
                                            } else {
                                                0.
                                            },
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
                result
            };
            unfold3d_blocked(&ctx, &input, geometry, 4, block, &mut output).unwrap();
            assert_eq!(
                transfers::to_cpu(&output).unwrap().to_f32_vec().unwrap(),
                expected(1.)
            );
            assert!(unfold3d_blocked(&ctx, &input, geometry, 4, 0, &mut output).is_err());
            assert!(unfold3d_blocked(&ctx, &input, geometry, 4, padded + 1, &mut output).is_err());
            let graph = crate::capture(&ctx, || {
                unfold3d_blocked(&ctx, &input, geometry, 4, block, &mut output)
            })
            .unwrap();
            let changed = upload(
                &ctx,
                vec![1, 3, 3, 3, channels],
                &source.iter().map(|v| -v).collect::<Vec<_>>(),
                dtype,
            );
            transfers::copy_cuda_to_cuda(&ctx, &changed, &input).unwrap();
            drop(input);
            drop(changed);
            for _ in 0..2 {
                graph.replay().unwrap();
                ctx.synchronize().unwrap();
                assert_eq!(
                    transfers::to_cpu(&output).unwrap().to_f32_vec().unwrap(),
                    expected(-1.)
                );
            }
        }
    }
}

#[test]
fn unfold_wide_coordinates_and_unaligned_views_preserve_values() {
    let ctx = CudaContext::new(0).unwrap();
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        let values: Vec<_> = (1..=16).map(|x| x as f32 / 8.).collect();
        let input = upload(&ctx, vec![1, 1, 1, 1, 8], &values[..8], dtype);
        let mut output = upload(&ctx, vec![3, 8], &[0.; 24], dtype);
        let width = i32::MAX as usize;
        let wide = Unfold3dGeometry {
            kernel: [1; 3],
            stride: [1, 1, width],
            dilation: [1; 3],
            padding_before: [0, 0, width],
            padding_after: [0, 0, width],
        };
        unfold3d(&ctx, &input, wide, 0, &mut output).unwrap();
        let expected: Vec<_> = [vec![0.; 8], values[..8].to_vec(), vec![0.; 8]].concat();
        assert_eq!(
            transfers::to_cpu(&output).unwrap().to_f32_vec().unwrap(),
            expected
        );

        let mut padded = vec![-7.];
        padded.extend_from_slice(&values);
        let source = upload(&ctx, vec![17], &padded, dtype);
        let destination = upload(&ctx, vec![17], &[-7.; 17], dtype);
        let element = dtype.size_in_bytes();
        let input = crate::CudaBuffer::from_tensor(&source)
            .unwrap()
            .view(element, 16 * element)
            .unwrap()
            .as_tensor(apxinf_core::Shape::new(vec![1, 1, 1, 2, 8]), dtype)
            .unwrap();
        let mut output = crate::CudaBuffer::from_tensor(&destination)
            .unwrap()
            .view(element, 16 * element)
            .unwrap()
            .as_tensor(apxinf_core::Shape::new(vec![2, 8]), dtype)
            .unwrap();
        let identity = Unfold3dGeometry {
            kernel: [1; 3],
            stride: [1; 3],
            dilation: [1; 3],
            padding_before: [0; 3],
            padding_after: [0; 3],
        };
        unfold3d(&ctx, &input, identity, 0, &mut output).unwrap();
        assert_eq!(
            transfers::to_cpu(&destination)
                .unwrap()
                .to_f32_vec()
                .unwrap(),
            padded
        );
    }
}
