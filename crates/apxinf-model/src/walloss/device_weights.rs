//! Device-ready dynamic FP8 linear weights.

use apxinf_core::{Backend, Error, Result, Tensor};

#[cfg(feature = "cuda")]
use super::backend::{kernels, RuntimeBackend};

use super::{encode_e4m3, E4M3_MAX};

#[derive(Debug)]
pub struct DynamicFp8LinearWeights {
    /// Contiguous input-major physical `[input, output]` (`[K, N]`) E4M3
    /// matrix — cuda-new's canonical GEMM orientation. Quantization still
    /// happens per output channel in `[N, K]` orientation; the stored matrix
    /// is a load-time byte transpose with no numerical effect.
    pub weight: Tensor,
    /// FP32 scale vector with one element per output channel.
    pub channel_scales: Tensor,
    /// Logical input width before the resident matrix was aligned.
    pub input_features: usize,
    /// Logical output width before the resident matrix was aligned.
    pub output_features: usize,
}

impl DynamicFp8LinearWeights {
    /// Aligned input width (`K` of the resident `[K, N]` matrix).
    pub fn padded_input_features(&self) -> usize {
        self.weight.shape().dims()[0]
    }

    #[cfg(feature = "cuda")]
    pub fn as_kernel_view(&self) -> kernels::gemm::DynamicFp8WeightView<'_> {
        kernels::gemm::DynamicFp8WeightView {
            values_e4m3: &self.weight,
            channel_scales: &self.channel_scales,
        }
    }

    pub fn from_host(weight: &Tensor, backend: &dyn Backend) -> Result<Self> {
        let shape = weight.shape().dims();
        if shape.len() != 2 || shape[0] == 0 || shape[1] == 0 {
            return Err(Error::Other(format!(
                "dynamic FP8 weight must be a non-empty 2D matrix, got {shape:?}"
            )));
        }

        let input_features = shape[0];
        let output_features = shape[1];
        let padded_input_features = align_16(input_features);
        let padded_output_features = align_16(output_features);
        let padded_weight =
            pad_transpose_linear_weight(weight, padded_output_features, padded_input_features)?;

        #[cfg(feature = "cuda")]
        let (weight, channel_scales) = if let Some(cuda_backend) =
            backend.as_any().downcast_ref::<RuntimeBackend>()
        {
            let values = padded_weight
                .to_f32_vec()?
                .into_iter()
                .map(half::bf16::from_f32)
                .collect::<Vec<_>>();
            let host_bf16 =
                Tensor::from_bf16(vec![padded_output_features, padded_input_features], &values)?;
            let device_bf16 = backend.to_device(&host_bf16)?;
            let quantized = kernels::quantization::quantize_rows_bf16_e4m3(
                cuda_backend.context(),
                &device_bf16,
            )?;
            (quantized.values, quantized.scales)
        } else {
            quantize_rows_host(&padded_weight, backend)?
        };
        #[cfg(not(feature = "cuda"))]
        let (weight, channel_scales) = quantize_rows_host(&padded_weight, backend)?;

        // cuda-new's GEMM consumes the canonical [K, N] orientation; transpose
        // the quantized bytes once at load. E4M3 values are untouched, so the
        // quantization itself stays bit-identical to the legacy [N, K] path.
        let weight = transpose_e4m3(&weight, backend)?;

        Ok(Self {
            weight,
            channel_scales,
            input_features,
            output_features,
        })
    }
}

/// Transpose an `[N, K]` E4M3 matrix to `[K, N]` through the host — a pure
/// byte permutation performed once at weight-load time.
fn transpose_e4m3(weight: &Tensor, backend: &dyn Backend) -> Result<Tensor> {
    let dims = weight.shape().dims().to_vec();
    let host = backend.to_cpu(weight)?;
    let source = host.as_f8_e4m3()?;
    let (rows, cols) = (dims[0], dims[1]);
    let mut transposed = vec![0u8; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            transposed[col * rows + row] = source[row * cols + col];
        }
    }
    backend.to_device(&Tensor::from_f8_e4m3(vec![cols, rows], &transposed)?)
}

fn align_16(value: usize) -> usize {
    value.div_ceil(16) * 16
}

fn pad_transpose_linear_weight(
    weight: &Tensor,
    padded_rows: usize,
    padded_cols: usize,
) -> Result<Tensor> {
    let shape = weight.shape().dims();
    let (input_features, output_features) = (shape[0], shape[1]);
    let source = weight.to_f32_vec()?;
    let mut padded = vec![0.0f32; padded_rows * padded_cols];
    for input in 0..input_features {
        for output in 0..output_features {
            padded[output * padded_cols + input] = source[input * output_features + output];
        }
    }
    Tensor::from_f32(vec![padded_rows, padded_cols], &padded)
}

fn quantize_rows_host(weight: &Tensor, backend: &dyn Backend) -> Result<(Tensor, Tensor)> {
    let shape = weight.shape().dims();
    let (rows, cols) = (shape[0], shape[1]);
    let source = weight.to_f32_vec()?;
    let mut scales = vec![1.0e-12f32; rows];
    for row in 0..rows {
        for col in 0..cols {
            scales[row] = scales[row].max(source[row * cols + col].abs() / E4M3_MAX);
        }
    }
    let values = source
        .iter()
        .enumerate()
        .map(|(index, value)| encode_e4m3(*value / scales[index / cols]))
        .collect::<Vec<_>>();
    Ok((
        backend.to_device(&Tensor::from_f8_e4m3(shape.to_vec(), &values)?)?,
        backend.to_device(&Tensor::from_f32(vec![rows], &scales)?)?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use apxinf_core::{CpuBackend, DType};

    #[test]
    fn dynamic_fp8_weights_use_independent_output_channel_scales() {
        let source = Tensor::from_f32(vec![2, 3], &[1.0, 2.0, 4.0, -3.0, -8.0, 12.0]).unwrap();
        let quantized = DynamicFp8LinearWeights::from_host(&source, &CpuBackend).unwrap();
        assert_eq!(quantized.weight.shape().dims(), &[16, 16]);
        assert_eq!(quantized.input_features, 2);
        assert_eq!(quantized.output_features, 3);
        assert_eq!(quantized.weight.dtype(), DType::F8E4M3);
        let scales = quantized.channel_scales.to_f32_vec().unwrap();
        assert_eq!(
            &scales[..3],
            &[3.0 / E4M3_MAX, 8.0 / E4M3_MAX, 12.0 / E4M3_MAX]
        );
        assert!(scales[3..].iter().all(|scale| *scale == 1.0e-12));
        let values = quantized.weight.as_f8_e4m3().unwrap();
        for (output, expected) in [[1.0, -3.0], [2.0, -8.0], [4.0, 12.0]]
            .into_iter()
            .enumerate()
        {
            for input in 0..2 {
                // Resident layout is [K, N] = [input, output].
                let decoded =
                    crate::walloss::decode_e4m3(values[input * 16 + output]) * scales[output];
                assert!((decoded - expected[input]).abs() < expected[input].abs() * 0.05);
            }
        }
    }
}
