//! Host-side packing helpers for π0-FAST device weights.
//!
//! QKV and gate/up projections are concatenated along the output dimension on
//! the host so each layer keeps one tensor-core GEMM, matching the execution
//! schedule. Packing happens before the device upload so the accelerator never
//! sees an unpacked group.

use apxinf_core::{Error, Result, Tensor};

/// Concatenate `[rows, w_i]` matrices into one `[rows, Σw_i]` row-major matrix.
pub(super) fn concat_host_2d(tensors: &[&Tensor]) -> Result<Tensor> {
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
    let total_cols = widths.iter().sum::<usize>();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concatenates_columns() {
        let a = Tensor::from_f32(vec![2, 2], &[1., 2., 3., 4.]).unwrap();
        let b = Tensor::from_f32(vec![2, 1], &[5., 6.]).unwrap();
        let packed = concat_host_2d(&[&a, &b]).unwrap();
        assert_eq!(packed.shape().dims(), &[2, 3]);
        assert_eq!(
            packed.to_f32_vec().unwrap(),
            vec![1., 2., 5., 3., 4., 6.]
        );
    }
}
