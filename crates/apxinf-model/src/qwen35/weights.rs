use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedLinearLayout {
    pub out_features: usize,
    pub in_features: usize,
    pub group_size: usize,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum WeightLayoutError {
    #[error("group size must be non-zero")]
    ZeroGroupSize,
    #[error("packed tensor shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    #[error("weight dtype must be I32 packed storage")]
    InvalidPackedDtype,
    #[error("scale dtype must be BF16")]
    InvalidScaleDtype,
    #[error("activation length must be {expected}, got {got}")]
    ActivationLength { expected: usize, got: usize },
    #[error("weight buffers have inconsistent lengths")]
    BufferLength,
    #[error("scale at index {index} is not finite")]
    NonFiniteScale { index: usize },
}

impl PackedLinearLayout {
    pub const fn new(out_features: usize, in_features: usize, group_size: usize) -> Self {
        Self {
            out_features,
            in_features,
            group_size,
        }
    }

    pub const fn groups(&self) -> usize {
        if self.group_size == 0 {
            0
        } else {
            self.in_features.div_ceil(self.group_size)
        }
    }

    pub const fn packed_k_columns(&self) -> usize {
        self.in_features.div_ceil(8)
    }

    pub const fn packed_n_rows(&self) -> usize {
        self.out_features.div_ceil(8)
    }

    pub fn validate_shapes(
        &self,
        weight_packed: &[usize],
        weight_scale: &[usize],
        weight_zero_point: &[usize],
    ) -> Result<(), WeightLayoutError> {
        if self.group_size == 0 {
            return Err(WeightLayoutError::ZeroGroupSize);
        }
        let expected_weight = [self.out_features, self.packed_k_columns()];
        let expected_scale = [self.out_features, self.groups()];
        let expected_zero_point = [self.packed_n_rows(), self.groups()];
        for (expected, got) in [
            (expected_weight.as_slice(), weight_packed),
            (expected_scale.as_slice(), weight_scale),
            (expected_zero_point.as_slice(), weight_zero_point),
        ] {
            if expected != got {
                return Err(WeightLayoutError::ShapeMismatch {
                    expected: expected.to_vec(),
                    got: got.to_vec(),
                });
            }
        }
        Ok(())
    }

    pub fn dequantize_value(
        &self,
        weight_packed: &[u32],
        scales: &[f32],
        zero_points: &[u32],
        out: usize,
        k: usize,
    ) -> Option<f32> {
        if out >= self.out_features || k >= self.in_features || self.group_size == 0 {
            return None;
        }
        let packed = *weight_packed.get(out * self.packed_k_columns() + k / 8)?;
        let q = ((packed >> ((k % 8) * 4)) & 0xF) as f32;
        let group = k / self.group_size;
        let scale = *scales.get(out * self.groups() + group)?;
        let packed_zp = *zero_points.get((out / 8) * self.groups() + group)?;
        let zp = ((packed_zp >> ((out % 8) * 4)) & 0xF) as f32;
        Some((q - zp) * scale)
    }

    /// Reference asymmetric W4 matrix-vector product. Packed storage follows
    /// the checkpoint contract: nibbles are packed along K and zero-points
    /// are packed along N, with group-32 scales along K.
    pub fn matvec_f32(
        &self,
        weight_packed: &[u32],
        scales: &[f32],
        zero_points: &[u32],
        activation: &[f32],
    ) -> Result<Vec<f32>, WeightLayoutError> {
        if activation.len() != self.in_features {
            return Err(WeightLayoutError::ActivationLength {
                expected: self.in_features,
                got: activation.len(),
            });
        }
        let expected_weight = self.out_features * self.packed_k_columns();
        let expected_scales = self.out_features * self.groups();
        let expected_zp = self.packed_n_rows() * self.groups();
        if weight_packed.len() != expected_weight
            || scales.len() != expected_scales
            || zero_points.len() != expected_zp
        {
            return Err(WeightLayoutError::BufferLength);
        }
        if let Some(index) = scales.iter().position(|scale| !scale.is_finite()) {
            return Err(WeightLayoutError::NonFiniteScale { index });
        }
        let mut output = vec![0.0; self.out_features];
        for (out, result) in output.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for (k, value) in activation.iter().enumerate() {
                let weight = self
                    .dequantize_value(weight_packed, scales, zero_points, out, k)
                    .ok_or(WeightLayoutError::BufferLength)?;
                sum += *value * weight;
            }
            *result = sum;
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_k_packed_weight_and_n_packed_zero_point_shapes() {
        let layout = PackedLinearLayout::new(1024, 5120, 32);
        assert_eq!(layout.groups(), 160);
        assert_eq!(layout.packed_k_columns(), 640);
        assert_eq!(layout.packed_n_rows(), 128);
        assert!(layout
            .validate_shapes(&[1024, 640], &[1024, 160], &[128, 160])
            .is_ok());
    }

    #[test]
    fn rejects_zero_point_packed_along_k_instead_of_n() {
        let layout = PackedLinearLayout::new(1024, 5120, 32);
        assert!(matches!(
            layout.validate_shapes(&[1024, 640], &[1024, 160], &[1024, 160]),
            Err(WeightLayoutError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn dequantizes_nibble_with_group_scale_and_n_packed_zero_point() {
        let layout = PackedLinearLayout::new(10, 9, 4);
        let mut packed = vec![0u32; layout.out_features * layout.packed_k_columns()];
        packed[0] = 0x0000_000f;
        packed[9 * layout.packed_k_columns() + 1] = 0;
        let scales = vec![2.0; layout.out_features * layout.groups()];
        let mut zero_points = vec![0u32; layout.packed_n_rows() * layout.groups()];
        zero_points[0] = 0x0000_0003;
        zero_points[layout.groups() + 2] = 0x0000_0030;
        assert_eq!(
            layout.dequantize_value(&packed, &scales, &zero_points, 0, 0),
            Some(24.0)
        );
        assert_eq!(
            layout.dequantize_value(&packed, &scales, &zero_points, 9, 8),
            Some(-6.0)
        );
    }

    #[test]
    fn reference_matvec_handles_group_boundary_and_tail() {
        let layout = PackedLinearLayout::new(9, 5, 4);
        let mut packed = vec![0u32; layout.out_features * layout.packed_k_columns()];
        for out in 0..layout.out_features {
            packed[out * layout.packed_k_columns()] = 0x0000_3210;
        }
        let scales = vec![1.0; layout.out_features * layout.groups()];
        let zero_points = vec![0u32; layout.packed_n_rows() * layout.groups()];
        let result = layout
            .matvec_f32(&packed, &scales, &zero_points, &[1.0; 5])
            .unwrap();
        assert_eq!(result.len(), 9);
        assert_eq!(result[0], 6.0);
        assert_eq!(result[8], 6.0);
    }

    #[test]
    fn reference_matvec_rejects_wrong_activation_or_buffers() {
        let layout = PackedLinearLayout::new(2, 8, 4);
        assert!(matches!(
            layout.matvec_f32(&[0; 2], &[1.0; 4], &[0; 1], &[1.0]),
            Err(WeightLayoutError::ActivationLength { .. })
        ));
        assert!(matches!(
            layout.matvec_f32(&[0], &[1.0; 4], &[0; 1], &[1.0; 8]),
            Err(WeightLayoutError::BufferLength)
        ));
        assert!(matches!(
            layout.matvec_f32(&[0; 2], &[f32::NAN; 4], &[0; 2], &[1.0; 8]),
            Err(WeightLayoutError::NonFiniteScale { .. })
        ));
    }
}
