//! Helpers for Hugging Face `compressed-tensors` pack-quantized weights.
//!
//! The Qwen3.8-27B AWQ checkpoint stores W4A16 linear weights as four tensors
//! per logical weight:
//!
//! - `<prefix>.weight_packed`: `I32 [out_features, in_features / 8]`
//! - `<prefix>.weight_scale`: `BF16 [out_features, in_features / group_size]`
//! - `<prefix>.weight_zero_point`: `I32 [out_features / 8, in_features / group_size]`
//! - `<prefix>.weight_shape`: `I64 [2]`, logical `[out_features, in_features]`
//!
//! Each `I32` packs eight 4-bit unsigned nibbles, low nibble first. Zero-points
//! are packed across output rows for each input group; weights are packed across
//! input columns for each output row.

use std::collections::HashMap;

use apxinf_core::{DType, Tensor};

/// Dequantize a group-wise asymmetric INT4 weight to BF16.
///
/// This is a correctness/reference path for loading and validation. Fast CUDA
/// inference should consume the packed tensors directly or fuse this operation
/// with GEMM; materializing full BF16 weights is too memory-heavy for the target
/// 27B checkpoint.
pub fn dequantize_w4a16_grouped(
    tensors: &HashMap<String, Tensor>,
    prefix: &str,
    group_size: usize,
) -> Result<Tensor, String> {
    if group_size == 0 {
        return Err("group_size must be greater than zero".into());
    }
    let packed = get(tensors, prefix, "weight_packed")?;
    let scale = get(tensors, prefix, "weight_scale")?;
    let zero_point = get(tensors, prefix, "weight_zero_point")?;
    let shape = get(tensors, prefix, "weight_shape")?;

    expect_dtype(packed, DType::I32, "weight_packed")?;
    expect_dtype(scale, DType::BF16, "weight_scale")?;
    expect_dtype(zero_point, DType::I32, "weight_zero_point")?;
    expect_dtype(shape, DType::I64, "weight_shape")?;

    let logical_shape = shape.as_i64().map_err(|error| error.to_string())?;
    if logical_shape.len() != 2 || logical_shape.iter().any(|&dim| dim <= 0) {
        return Err(format!(
            "{prefix}.weight_shape must contain two positive dimensions, got {logical_shape:?}"
        ));
    }
    let rows = logical_shape[0] as usize;
    let cols = logical_shape[1] as usize;
    let packed_cols = cols.div_ceil(8);
    let groups = cols.div_ceil(group_size);
    let zp_rows = rows.div_ceil(8);

    if packed.shape().dims() != [rows, packed_cols] {
        return Err(format!(
            "{prefix}.weight_packed shape {:?} does not match expected [{rows}, {packed_cols}]",
            packed.shape().dims()
        ));
    }
    if scale.shape().dims() != [rows, groups] {
        return Err(format!(
            "{prefix}.weight_scale shape {:?} does not match expected [{rows}, {groups}]",
            scale.shape().dims()
        ));
    }
    if zero_point.shape().dims() != [zp_rows, groups] {
        return Err(format!(
            "{prefix}.weight_zero_point shape {:?} does not match expected [{zp_rows}, {groups}]",
            zero_point.shape().dims()
        ));
    }

    let packed = packed.as_i32().map_err(|error| error.to_string())?;
    let scales = scale.as_bf16().map_err(|error| error.to_string())?;
    let zero_points = zero_point.as_i32().map_err(|error| error.to_string())?;
    let mut output = Vec::with_capacity(rows * cols);

    for row in 0..rows {
        let zp_row = row / 8;
        let zp_shift = (row % 8) * 4;
        for col in 0..cols {
            let group = col / group_size;
            let word = packed[row * packed_cols + col / 8] as u32;
            let q = ((word >> ((col % 8) * 4)) & 0xF) as i32;
            let zp_word = zero_points[zp_row * groups + group] as u32;
            let zp = ((zp_word >> zp_shift) & 0xF) as i32;
            let value = (q - zp) as f32 * scales[row * groups + group].to_f32();
            output.push(half::bf16::from_f32(value));
        }
    }

    Tensor::from_bf16(vec![rows, cols], &output).map_err(|error| error.to_string())
}

fn get<'a>(
    tensors: &'a HashMap<String, Tensor>,
    prefix: &str,
    suffix: &str,
) -> Result<&'a Tensor, String> {
    let name = format!("{prefix}.{suffix}");
    tensors
        .get(&name)
        .ok_or_else(|| format!("missing compressed-tensors entry {name}"))
}

fn expect_dtype(tensor: &Tensor, expected: DType, suffix: &str) -> Result<(), String> {
    let actual = tensor.dtype();
    if actual != expected {
        return Err(format!(
            "{suffix} must be {expected}, got {actual}"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use apxinf_core::Shape;

    fn tensor_i32(shape: &[usize], values: &[i32]) -> Tensor {
        let bytes = values.iter().flat_map(|value| value.to_le_bytes()).collect();
        Tensor::from_raw(Shape::from(shape.to_vec()), DType::I32, apxinf_core::Device::Cpu, bytes)
            .unwrap()
    }

    fn tensor_i64(shape: &[usize], values: &[i64]) -> Tensor {
        let bytes = values.iter().flat_map(|value| value.to_le_bytes()).collect();
        Tensor::from_raw(Shape::from(shape.to_vec()), DType::I64, apxinf_core::Device::Cpu, bytes)
            .unwrap()
    }

    fn pack_nibbles(values: &[u8]) -> i32 {
        assert!(values.len() <= 8);
        values
            .iter()
            .enumerate()
            .fold(0u32, |word, (index, value)| {
                word | (((value & 0xF) as u32) << (index * 4))
            }) as i32
    }

    #[test]
    fn dequantizes_grouped_int4_weight() {
        let prefix = "layer.q_proj";
        let mut tensors = HashMap::new();
        let packed_rows = [
            pack_nibbles(&[1, 2, 3, 4, 5, 6, 7, 8]),
            pack_nibbles(&[2, 3, 4, 5, 6, 7, 8, 9]),
            pack_nibbles(&[3, 4, 5, 6, 7, 8, 9, 10]),
            pack_nibbles(&[4, 5, 6, 7, 8, 9, 10, 11]),
            pack_nibbles(&[5, 6, 7, 8, 9, 10, 11, 12]),
            pack_nibbles(&[6, 7, 8, 9, 10, 11, 12, 13]),
            pack_nibbles(&[7, 8, 9, 10, 11, 12, 13, 14]),
            pack_nibbles(&[8, 9, 10, 11, 12, 13, 14, 15]),
        ];
        tensors.insert(format!("{prefix}.weight_packed"), tensor_i32(&[8, 1], &packed_rows));
        tensors.insert(
            format!("{prefix}.weight_scale"),
            Tensor::from_bf16(vec![8, 1], &[half::bf16::from_f32(0.5); 8]).unwrap(),
        );
        tensors.insert(
            format!("{prefix}.weight_zero_point"),
            tensor_i32(&[1, 1], &[pack_nibbles(&[1, 2, 3, 4, 5, 6, 7, 8])]),
        );
        tensors.insert(format!("{prefix}.weight_shape"), tensor_i64(&[2], &[8, 8]));

        let dequantized = dequantize_w4a16_grouped(&tensors, prefix, 8).unwrap();
        assert_eq!(dequantized.shape().dims(), &[8, 8]);
        let values = dequantized.to_f32_vec().unwrap();
        assert_eq!(values[0], 0.0);
        assert_eq!(values[1], 0.5);
        assert_eq!(values[8], 0.0);
        assert_eq!(values[15], 3.5);
    }
}
