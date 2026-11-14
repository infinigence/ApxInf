//! Shared host matrix packing for device weight construction.
use std::borrow::Cow;

use apxinf_core::{DType, Device, Error, Result, Tensor};

fn matrix_geometry(tensors: &[&Tensor]) -> Result<(usize, Vec<usize>, usize)> {
    let first = tensors
        .first()
        .ok_or_else(|| Error::Other("empty tensor concatenation".into()))?;
    let dims = first.shape().dims();
    if dims.len() != 2 {
        return Err(Error::Other(format!("expected 2D weight, got {dims:?}")));
    }
    let rows = dims[0];
    let widths = tensors
        .iter()
        .map(|tensor| {
            let dims = tensor.shape().dims();
            if dims.len() != 2 || dims[0] != rows {
                return Err(Error::Other("packed linear input dimensions differ".into()));
            }
            Ok(dims[1])
        })
        .collect::<Result<Vec<_>>>()?;
    let total_cols = widths.iter().try_fold(0usize, |total, width| {
        total
            .checked_add(*width)
            .ok_or_else(|| Error::Other("packed linear output width overflow".into()))
    })?;
    Ok((rows, widths, total_cols))
}

pub(in crate::pi05) fn concat_host_2d(tensors: &[&Tensor]) -> Result<Tensor> {
    let (rows, widths, total_cols) = matrix_geometry(tensors)?;
    let sources = tensors
        .iter()
        .map(|tensor| tensor.to_f32_vec())
        .collect::<Result<Vec<_>>>()?;
    let mut output = vec![0.0f32; rows * total_cols];
    for row in 0..rows {
        let mut output_col = 0;
        for (source, width) in sources.iter().zip(&widths) {
            output[row * total_cols + output_col..row * total_cols + output_col + width]
                .copy_from_slice(&source[row * width..(row + 1) * width]);
            output_col += width;
        }
    }
    Tensor::from_f32(vec![rows, total_cols], &output)
}

/// Preserve native BF16 host storage instead of expanding it through F32.
/// Other floating inputs retain the same elementwise BF16 rounding.
pub(super) fn to_bf16_host(tensor: &Tensor) -> Result<Cow<'_, Tensor>> {
    if tensor.device() != Device::Cpu {
        return Err(Error::UnsupportedDevice(tensor.device()));
    }
    if tensor.dtype() == DType::BF16 {
        tensor.as_bf16()?;
        return Ok(Cow::Borrowed(tensor));
    }
    if tensor.dtype() == DType::F8E4M3 {
        return Err(Error::Other(
            "cannot convert scale-less E4M3 data to BF16".into(),
        ));
    }
    let values = match tensor.dtype() {
        DType::F32 => tensor
            .as_f32()?
            .iter()
            .copied()
            .map(half::bf16::from_f32)
            .collect::<Vec<_>>(),
        DType::F16 => tensor
            .as_f16()?
            .iter()
            .map(|value| half::bf16::from_f32(value.to_f32()))
            .collect::<Vec<_>>(),
        _ => tensor
            .to_f32_vec()?
            .into_iter()
            .map(half::bf16::from_f32)
            .collect::<Vec<_>>(),
    };
    Ok(Cow::Owned(Tensor::from_bf16(
        tensor.shape().dims().to_vec(),
        &values,
    )?))
}

/// Pack output columns directly in BF16. A single native BF16 projection is
/// borrowed; QKV and gate/up require only their final BF16 host allocation.
pub(super) fn concat_bf16_host<'a>(tensors: &[&'a Tensor]) -> Result<Cow<'a, Tensor>> {
    let (rows, widths, total_cols) = matrix_geometry(tensors)?;
    if tensors.len() == 1 {
        return to_bf16_host(tensors[0]);
    }
    let bytes = rows
        .checked_mul(total_cols)
        .and_then(|elements| elements.checked_mul(2))
        .ok_or_else(|| Error::Other("packed BF16 weight size overflow".into()))?;
    let converted = tensors
        .iter()
        .map(|tensor| to_bf16_host(tensor))
        .collect::<Result<Vec<_>>>()?;
    let sources = converted
        .iter()
        .map(|tensor| tensor.storage().as_cpu().expect("validated CPU tensor"))
        .collect::<Vec<_>>();
    let mut output = Vec::with_capacity(bytes);
    for row in 0..rows {
        for (source, width) in sources.iter().zip(&widths) {
            let start = row * width * 2;
            output.extend_from_slice(&source[start..start + width * 2]);
        }
    }
    Ok(Cow::Owned(Tensor::from_raw(
        vec![rows, total_cols].into(),
        DType::BF16,
        Device::Cpu,
        output,
    )?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::{bf16, f16};

    #[test]
    fn native_bf16_reuses_host_storage_including_single_projection() {
        let values = (0..=u16::MAX)
            .filter(|bits| bits & 0x7f80 != 0x7f80)
            .map(bf16::from_bits)
            .collect::<Vec<_>>();
        let input = Tensor::from_bf16(vec![1, values.len()], &values).unwrap();
        for output in [
            to_bf16_host(&input).unwrap(),
            concat_bf16_host(&[&input]).unwrap(),
        ] {
            assert_eq!(output.as_bf16().unwrap(), input.as_bf16().unwrap());
            assert_eq!(
                output.as_bf16().unwrap().as_ptr(),
                input.as_bf16().unwrap().as_ptr()
            );
        }
    }

    #[test]
    fn bf16_qkv_preserves_row_order_and_unequal_output_widths() {
        let matrix = |width, values: &[f32]| {
            Tensor::from_bf16(
                vec![2, width],
                &values
                    .iter()
                    .copied()
                    .map(bf16::from_f32)
                    .collect::<Vec<_>>(),
            )
            .unwrap()
        };
        let q = matrix(2, &[1., 2., 3., 4.]);
        let k = matrix(1, &[5., 6.]);
        let v = matrix(3, &[7., 8., 9., 10., 11., 12.]);
        let packed = concat_bf16_host(&[&q, &k, &v]).unwrap();
        assert_eq!(packed.shape().dims(), &[2, 6]);
        assert_eq!(
            packed.to_f32_vec().unwrap(),
            [1., 2., 5., 7., 8., 9., 3., 4., 6., 10., 11., 12.]
        );
        assert_eq!(q.to_f32_vec().unwrap(), [1., 2., 3., 4.]);
    }

    #[test]
    fn mixed_input_dtypes_match_f32_concat_then_bf16_rounding() {
        let a = Tensor::from_f32(vec![2, 2], &[1.00390625, 1.01171875, -0.0, 0.0000001]).unwrap();
        let b = Tensor::from_f16(vec![2, 1], &[f16::from_f32(-2.3), f16::from_f32(0.125)]).unwrap();
        let c =
            Tensor::from_bf16(vec![2, 1], &[bf16::from_f32(4.25), bf16::from_f32(-8.5)]).unwrap();
        let expected = concat_host_2d(&[&a, &b, &c])
            .unwrap()
            .to_f32_vec()
            .unwrap()
            .into_iter()
            .map(bf16::from_f32)
            .map(bf16::to_bits)
            .collect::<Vec<_>>();
        let actual = concat_bf16_host(&[&a, &b, &c]).unwrap();
        assert_eq!(
            actual
                .as_bf16()
                .unwrap()
                .iter()
                .copied()
                .map(bf16::to_bits)
                .collect::<Vec<_>>(),
            expected
        );
    }

    #[test]
    fn direct_f32_conversion_preserves_rounding_and_signed_zero() {
        let values = [
            -0.0,
            0.0,
            1.00390625,
            1.01171875,
            -1.00390625,
            -1.01171875,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::MAX,
        ];
        let input = Tensor::from_f32(vec![values.len()], &values).unwrap();
        let output = to_bf16_host(&input).unwrap();
        assert_eq!(
            output
                .as_bf16()
                .unwrap()
                .iter()
                .copied()
                .map(bf16::to_bits)
                .collect::<Vec<_>>(),
            values
                .iter()
                .copied()
                .map(bf16::from_f32)
                .map(bf16::to_bits)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn rejects_bad_geometry_and_unscaled_fp8() {
        let a = Tensor::zeros(vec![2, 1], DType::BF16);
        let b = Tensor::zeros(vec![3, 1], DType::BF16);
        let vector = Tensor::zeros(vec![2], DType::BF16);
        let fp8 = Tensor::from_f8_e4m3(vec![2, 1], &[0x38, 0x38]).unwrap();
        assert!(concat_bf16_host(&[]).is_err());
        assert!(concat_bf16_host(&[&a, &b]).is_err());
        assert!(concat_bf16_host(&[&vector]).is_err());
        assert!(concat_bf16_host(&[&fp8]).is_err());
        let huge_empty = Tensor::zeros(vec![0, usize::MAX], DType::BF16);
        let one_empty = Tensor::zeros(vec![0, 1], DType::BF16);
        assert!(concat_bf16_host(&[&huge_empty, &one_empty]).is_err());
    }
}
