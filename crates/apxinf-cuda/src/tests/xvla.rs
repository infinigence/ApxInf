//! X-VLA port operator tests: CPU-reference parity for the five custom kernels.

use crate::context::CudaContext;
use crate::kernels::xvla_ops::{
    dwconv3x3_bf16, gather_rows_bf16, gelu_erf_bf16, im2col_bf16, pack_action_tokens_bf16,
    prepare_row_gather, rgb_u8_to_normalized_tokens_bf16, sigmoid_bf16, sigmoid_columns_bf16,
    upload_bytes, XVLA_IMAGE_MEAN, XVLA_IMAGE_STD,
};
use crate::test_util::{
    assert_bf16_close, assert_bf16_close_elementwise, download_bf16_as_fp32, upload_fp32_as_bf16,
};

fn bf16_round(values: &[f32]) -> Vec<f32> {
    values
        .iter()
        .map(|value| half::bf16::from_f32(*value).to_f32())
        .collect()
}

/// Abramowitz & Stegun 7.1.26, |error| < 1.5e-7 — enough for BF16 parity.
fn erf_reference(z: f32) -> f32 {
    let x = z.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
        + 0.254829592)
        * t;
    let value = 1.0 - poly * (-x * x).exp();
    if z < 0.0 {
        -value
    } else {
        value
    }
}

#[test]
fn gelu_erf_matches_reference() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let values: Vec<f32> = (0..257)
        .map(|index| (index as f32 - 128.0) / 37.0)
        .collect();
    let input = upload_fp32_as_bf16(&ctx, &values, vec![257]).unwrap();
    let output = gelu_erf_bf16(&ctx, &input).unwrap();
    let actual = download_bf16_as_fp32(&output).unwrap();
    let expected: Vec<f32> = bf16_round(&values)
        .iter()
        .map(|x| 0.5 * x * (1.0 + erf_reference(x / std::f32::consts::SQRT_2)))
        .collect();
    assert_bf16_close(&actual, &expected, 2e-3, 6e-3);
}

#[test]
fn sigmoid_matches_reference() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let values: Vec<f32> = (0..129).map(|index| (index as f32 - 64.0) / 5.0).collect();
    let input = upload_fp32_as_bf16(&ctx, &values, vec![129]).unwrap();
    let output = sigmoid_bf16(&ctx, &input).unwrap();
    let actual = download_bf16_as_fp32(&output).unwrap();
    let expected: Vec<f32> = bf16_round(&values)
        .iter()
        .map(|x| 1.0 / (1.0 + (-x).exp()))
        .collect();
    assert_bf16_close(&actual, &expected, 2e-3, 6e-3);
}

#[test]
fn rgb_normalization_matches_reference() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (height, width) = (5usize, 7usize);
    let rgb: Vec<u8> = (0..height * width * 3)
        .map(|index| ((index * 37) % 251) as u8)
        .collect();
    let buffer = upload_bytes(&ctx, &rgb).unwrap();
    let output = rgb_u8_to_normalized_tokens_bf16(
        &ctx,
        &buffer,
        height,
        width,
        XVLA_IMAGE_MEAN,
        XVLA_IMAGE_STD,
    )
    .unwrap();
    let actual = download_bf16_as_fp32(&output).unwrap();
    let expected: Vec<f32> = rgb
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let channel = index % 3;
            ((*value as f32) * (1.0f32 / 255.0) - XVLA_IMAGE_MEAN[channel])
                / XVLA_IMAGE_STD[channel]
        })
        .collect();
    assert_bf16_close_elementwise(&actual, &bf16_round(&expected));
}

#[test]
fn im2col_matches_reference() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (channels, height, width) = (3usize, 6usize, 5usize);
    let (kh, kw, stride, pad) = (3usize, 3usize, 2usize, 1usize);
    let values: Vec<f32> = (0..channels * height * width)
        .map(|index| (index as f32 * 0.13) - 1.0)
        .collect();
    let input = upload_fp32_as_bf16(&ctx, &values, vec![height * width, channels]).unwrap();
    let output = im2col_bf16(
        &ctx, &input, height, width, kh, kw, stride, stride, pad, pad,
    )
    .unwrap();
    let actual = download_bf16_as_fp32(&output).unwrap();
    let rounded = bf16_round(&values);
    let out_h = (height + 2 * pad - kh) / stride + 1;
    let out_w = (width + 2 * pad - kw) / stride + 1;
    let mut expected = Vec::with_capacity(out_h * out_w * channels * kh * kw);
    for oy in 0..out_h {
        for ox in 0..out_w {
            for c in 0..channels {
                for ky in 0..kh {
                    for kx in 0..kw {
                        let iy = (oy * stride) as isize - pad as isize + ky as isize;
                        let ix = (ox * stride) as isize - pad as isize + kx as isize;
                        expected.push(
                            if iy >= 0 && iy < height as isize && ix >= 0 && ix < width as isize {
                                rounded[(iy as usize * width + ix as usize) * channels + c]
                            } else {
                                0.0
                            },
                        );
                    }
                }
            }
        }
    }
    assert_bf16_close_elementwise(&actual, &expected);
}

#[test]
fn dwconv3x3_matches_reference() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (channels, height, width) = (4usize, 5usize, 6usize);
    let values: Vec<f32> = (0..channels * height * width)
        .map(|index| (index as f32 * 0.07) - 0.5)
        .collect();
    let weights: Vec<f32> = (0..channels * 9)
        .map(|index| (index as f32 * 0.11) - 0.4)
        .collect();
    let biases: Vec<f32> = (0..channels)
        .map(|index| index as f32 * 0.25 - 0.3)
        .collect();
    let input = upload_fp32_as_bf16(&ctx, &values, vec![height * width, channels]).unwrap();
    let weight = upload_fp32_as_bf16(&ctx, &weights, vec![channels, 9]).unwrap();
    let bias = upload_fp32_as_bf16(&ctx, &biases, vec![channels]).unwrap();
    let output = dwconv3x3_bf16(&ctx, &input, &weight, Some(&bias), height, width).unwrap();
    let actual = download_bf16_as_fp32(&output).unwrap();
    let rounded = bf16_round(&values);
    let rounded_weights = bf16_round(&weights);
    let rounded_biases = bf16_round(&biases);
    let mut expected = vec![0.0f32; channels * height * width];
    for y in 0..height {
        for x in 0..width {
            for c in 0..channels {
                let mut accumulator = rounded_biases[c];
                for ky in 0..3usize {
                    for kx in 0..3usize {
                        let iy = y as isize + ky as isize - 1;
                        let ix = x as isize + kx as isize - 1;
                        if iy < 0 || iy >= height as isize || ix < 0 || ix >= width as isize {
                            continue;
                        }
                        let value = rounded[(iy as usize * width + ix as usize) * channels + c];
                        let tap = rounded_weights[c * 9 + ky * 3 + kx];
                        accumulator = value.mul_add(tap, accumulator);
                    }
                }
                expected[(y * width + x) * channels + c] = accumulator;
            }
        }
    }
    assert_bf16_close(&actual, &expected, 2e-3, 1e-2);
}

#[test]
fn gather_rows_unbounded_matches_reference() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let rows = 5usize;
    let cols = 3usize;
    let values: Vec<f32> = (0..rows * cols).map(|index| index as f32 + 0.25).collect();
    let input = upload_fp32_as_bf16(&ctx, &values, vec![rows, cols]).unwrap();
    let indices: [u32; 9] = [2, 0, 4, 4, 1, 2, 0, 0, 3];
    let plan = prepare_row_gather(&ctx, &indices, rows).unwrap();
    let output = gather_rows_bf16(&ctx, &input, &plan).unwrap();
    let actual = download_bf16_as_fp32(&output).unwrap();
    let rounded = bf16_round(&values);
    let mut expected = Vec::new();
    for row in indices {
        expected.extend_from_slice(&rounded[row as usize * cols..(row as usize + 1) * cols]);
    }
    assert_bf16_close_elementwise(&actual, &expected);

    assert!(prepare_row_gather(&ctx, &[0, rows as u32], rows).is_err());
}

#[test]
fn pack_action_tokens_zeroes_gripper_columns() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (rows, dim_action, dim_time) = (3usize, 20usize, 32usize);
    let action: Vec<f32> = (0..rows * dim_action)
        .map(|index| index as f32 + 0.5)
        .collect();
    let proprio: Vec<f32> = (0..dim_action).map(|index| index as f32 + 10.0).collect();
    let time: Vec<f32> = (0..dim_time).map(|index| index as f32 * 0.25).collect();
    let action = upload_fp32_as_bf16(&ctx, &action, vec![rows, dim_action]).unwrap();
    let proprio = upload_fp32_as_bf16(&ctx, &proprio, vec![dim_action]).unwrap();
    let time = upload_fp32_as_bf16(&ctx, &time, vec![dim_time]).unwrap();
    let output = pack_action_tokens_bf16(&ctx, &action, &proprio, &time, [9, 19]).unwrap();
    let actual = download_bf16_as_fp32(&output).unwrap();
    let columns = dim_action * 2 + dim_time;
    let mut expected = Vec::with_capacity(rows * columns);
    for row in 0..rows {
        for column in 0..dim_action {
            let value = if column == 9 || column == 19 {
                0.0
            } else {
                bf16_round(&[(row * dim_action + column) as f32 + 0.5])[0]
            };
            expected.push(value);
        }
        for column in 0..dim_action {
            let value = if column == 9 || column == 19 {
                0.0
            } else {
                bf16_round(&[column as f32 + 10.0])[0]
            };
            expected.push(value);
        }
        for column in 0..dim_time {
            expected.push(bf16_round(&[column as f32 * 0.25])[0]);
        }
    }
    assert_bf16_close_elementwise(&actual, &expected);
    assert_eq!(output.shape().dims(), [rows, columns]);
}

#[test]
fn sigmoid_columns_matches_reference() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (rows, columns) = (4usize, 20usize);
    let values: Vec<f32> = (0..rows * columns)
        .map(|index| (index as f32 - 30.0) / 7.0)
        .collect();
    let input = upload_fp32_as_bf16(&ctx, &values, vec![rows, columns]).unwrap();
    let output = sigmoid_columns_bf16(&ctx, &input, [9, 19]).unwrap();
    let actual = download_bf16_as_fp32(&output).unwrap();
    let rounded = bf16_round(&values);
    let mut expected = rounded.clone();
    for row in 0..rows {
        for column in [9usize, 19] {
            let index = row * columns + column;
            expected[index] = 1.0 / (1.0 + (-rounded[index]).exp());
        }
    }
    assert_bf16_close_elementwise(&actual, &expected);
}
