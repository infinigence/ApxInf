use thiserror::Error;

use crate::manifest::{
    LoaderManifest, ManifestDType, ManifestError, PackAxis, QuantizationRole, TensorManifest,
};

#[derive(Debug, Error, Clone, PartialEq)]
pub enum W4Error {
    #[error("invalid loader manifest: {0}")]
    Manifest(#[from] ManifestError),
    #[error("missing required tensor `{0}`")]
    MissingTensor(String),
    #[error("tensor `{name}` expected {field} {expected}, got {actual}")]
    Inventory {
        name: String,
        field: &'static str,
        expected: String,
        actual: String,
    },
    #[error("nibble at index {index} must be in 0..=15, got {value}")]
    InvalidNibble { index: usize, value: u8 },
    #[error("zero-point nibble for group {group} must be in 0..=15, got {value}")]
    InvalidZeroPoint { group: usize, value: u8 },
    #[error("packed words contain {available} nibbles, fewer than logical length {logical_len}")]
    PackedLength {
        available: usize,
        logical_len: usize,
    },
    #[error("group_size must be positive")]
    ZeroGroupSize,
    #[error("expected {expected} scale and zero-point groups, got scales={scales}, zero_points={zero_points}")]
    GroupMetadata {
        expected: usize,
        scales: usize,
        zero_points: usize,
    },
}

struct ExpectedTensor {
    name: &'static str,
    shape: &'static [usize],
    dtype: ManifestDType,
    role: QuantizationRole,
    axis: PackAxis,
    group_size: Option<usize>,
}

const EXPECTED: &[ExpectedTensor] = &[
    ExpectedTensor {
        name: "k_proj.weight_packed",
        shape: &[1024, 640],
        dtype: ManifestDType::I32,
        role: QuantizationRole::PackedWeight,
        axis: PackAxis::K,
        group_size: None,
    },
    ExpectedTensor {
        name: "k_proj.weight_scale",
        shape: &[1024, 160],
        dtype: ManifestDType::BF16,
        role: QuantizationRole::Scale,
        axis: PackAxis::K,
        group_size: Some(32),
    },
    ExpectedTensor {
        name: "k_proj.weight_zero_point",
        shape: &[128, 160],
        dtype: ManifestDType::I32,
        role: QuantizationRole::ZeroPoint,
        axis: PackAxis::N,
        group_size: Some(32),
    },
    ExpectedTensor {
        name: "down_proj.weight_packed",
        shape: &[5120, 2176],
        dtype: ManifestDType::I32,
        role: QuantizationRole::PackedWeight,
        axis: PackAxis::K,
        group_size: None,
    },
    ExpectedTensor {
        name: "down_proj.weight_scale",
        shape: &[5120, 544],
        dtype: ManifestDType::BF16,
        role: QuantizationRole::Scale,
        axis: PackAxis::K,
        group_size: Some(32),
    },
    ExpectedTensor {
        name: "down_proj.weight_zero_point",
        shape: &[640, 544],
        dtype: ManifestDType::I32,
        role: QuantizationRole::ZeroPoint,
        axis: PackAxis::N,
        group_size: Some(32),
    },
];

pub fn validate_qwen35_w4_inventory(manifest: &LoaderManifest) -> Result<(), W4Error> {
    manifest.validate()?;
    for expected in EXPECTED {
        let actual = manifest
            .tensor(expected.name)
            .ok_or_else(|| W4Error::MissingTensor(expected.name.to_owned()))?;
        compare(expected, actual)?;
    }
    Ok(())
}

pub fn build_qwen35_w4_layer_manifest(
    prefix: &str,
    inventory: &[TensorManifest],
) -> Result<LoaderManifest, W4Error> {
    let mut tensors = Vec::with_capacity(EXPECTED.len());
    for expected in EXPECTED {
        let full_name = format!("{prefix}.{}", expected.name);
        let source = inventory
            .iter()
            .find(|tensor| tensor.name == full_name)
            .ok_or_else(|| W4Error::MissingTensor(full_name.clone()))?;
        let mut normalized = source.clone();
        normalized.name = expected.name.to_owned();
        normalized.pack_axis = Some(expected.axis);
        normalized.group_size = expected.group_size;
        normalized.quantization_role = Some(expected.role);
        tensors.push(normalized);
    }
    let manifest = LoaderManifest {
        schema: crate::manifest::LOADER_MANIFEST_SCHEMA.to_owned(),
        revision: crate::manifest::QWEN35_MODEL_REVISION.to_owned(),
        vocab_size: crate::manifest::QWEN35_MODEL_VOCAB_SIZE,
        tensors,
    };
    validate_qwen35_w4_inventory(&manifest)?;
    Ok(manifest)
}

fn compare(expected: &ExpectedTensor, actual: &TensorManifest) -> Result<(), W4Error> {
    if actual.shape != expected.shape {
        return Err(inventory_error(
            expected.name,
            "shape",
            expected.shape,
            &actual.shape,
        ));
    }
    if actual.dtype != expected.dtype {
        return Err(inventory_error(
            expected.name,
            "dtype",
            &expected.dtype,
            &actual.dtype,
        ));
    }
    if actual.quantization_role != Some(expected.role) {
        return Err(inventory_error(
            expected.name,
            "quantization_role",
            &Some(expected.role),
            &actual.quantization_role,
        ));
    }
    if actual.pack_axis != Some(expected.axis) {
        return Err(inventory_error(
            expected.name,
            "pack_axis",
            &Some(expected.axis),
            &actual.pack_axis,
        ));
    }
    if actual.group_size != expected.group_size {
        return Err(inventory_error(
            expected.name,
            "group_size",
            &expected.group_size,
            &actual.group_size,
        ));
    }
    Ok(())
}

fn inventory_error(
    name: &str,
    field: &'static str,
    expected: &(impl std::fmt::Debug + ?Sized),
    actual: &(impl std::fmt::Debug + ?Sized),
) -> W4Error {
    W4Error::Inventory {
        name: name.to_owned(),
        field,
        expected: format!("{expected:?}"),
        actual: format!("{actual:?}"),
    }
}

pub fn pack_nibbles(values: &[u8]) -> Result<Vec<u32>, W4Error> {
    let mut packed = vec![0u32; values.len().div_ceil(8)];
    for (index, value) in values.iter().copied().enumerate() {
        if value > 15 {
            return Err(W4Error::InvalidNibble { index, value });
        }
        packed[index / 8] |= u32::from(value) << ((index % 8) * 4);
    }
    Ok(packed)
}

pub fn unpack_nibbles(words: &[u32], logical_len: usize) -> Result<Vec<u8>, W4Error> {
    let available = words.len().saturating_mul(8);
    if logical_len > available {
        return Err(W4Error::PackedLength {
            available,
            logical_len,
        });
    }
    Ok((0..logical_len)
        .map(|index| ((words[index / 8] >> ((index % 8) * 4)) & 0x0f) as u8)
        .collect())
}

pub fn dequantize_grouped(
    values: &[u8],
    scales: &[f32],
    zero_points: &[u8],
    group_size: usize,
) -> Result<Vec<f32>, W4Error> {
    if group_size == 0 {
        return Err(W4Error::ZeroGroupSize);
    }
    let groups = values.len().div_ceil(group_size);
    if scales.len() != groups || zero_points.len() != groups {
        return Err(W4Error::GroupMetadata {
            expected: groups,
            scales: scales.len(),
            zero_points: zero_points.len(),
        });
    }
    for (group, value) in zero_points.iter().copied().enumerate() {
        if value > 15 {
            return Err(W4Error::InvalidZeroPoint { group, value });
        }
    }
    values
        .iter()
        .copied()
        .enumerate()
        .map(|(index, value)| {
            if value > 15 {
                return Err(W4Error::InvalidNibble { index, value });
            }
            let group = index / group_size;
            Ok((f32::from(value) - f32::from(zero_points[group])) * scales[group])
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{
        LoaderManifest, LOADER_MANIFEST_SCHEMA, QWEN35_MODEL_REVISION, QWEN35_MODEL_VOCAB_SIZE,
    };

    fn required_tensors() -> Vec<TensorManifest> {
        EXPECTED
            .iter()
            .map(|expected| TensorManifest {
                name: expected.name.to_owned(),
                shape: expected.shape.to_vec(),
                dtype: expected.dtype.clone(),
                quantization_role: Some(expected.role),
                pack_axis: Some(expected.axis),
                group_size: expected.group_size,
            })
            .collect()
    }

    fn manifest() -> LoaderManifest {
        LoaderManifest {
            schema: LOADER_MANIFEST_SCHEMA.to_owned(),
            revision: QWEN35_MODEL_REVISION.to_owned(),
            vocab_size: QWEN35_MODEL_VOCAB_SIZE,
            tensors: required_tensors(),
        }
    }

    #[test]
    fn inventory_accepts_exact_qwen35_w4_directions() {
        validate_qwen35_w4_inventory(&manifest()).unwrap();
    }

    #[test]
    fn inventory_rejects_swapped_n_k_axis() {
        let mut value = manifest();
        value
            .tensors
            .iter_mut()
            .find(|tensor| tensor.name == "k_proj.weight_zero_point")
            .unwrap()
            .pack_axis = Some(PackAxis::K);
        assert!(matches!(
            validate_qwen35_w4_inventory(&value),
            Err(W4Error::Inventory {
                field: "pack_axis",
                ..
            })
        ));
    }

    #[test]
    fn inventory_rejects_wrong_shape_dtype_and_group() {
        for (name, mutate, field) in [
            ("down_proj.weight_packed", 0usize, "shape"),
            ("k_proj.weight_scale", 1usize, "dtype"),
            ("down_proj.weight_zero_point", 2usize, "group_size"),
            ("k_proj.weight_packed", 3usize, "quantization_role"),
        ] {
            let mut value = manifest();
            let tensor = value
                .tensors
                .iter_mut()
                .find(|tensor| tensor.name == name)
                .unwrap();
            match mutate {
                0 => tensor.shape[1] += 1,
                1 => tensor.dtype = ManifestDType::F16,
                2 => tensor.group_size = Some(64),
                _ => tensor.quantization_role = Some(QuantizationRole::Scale),
            }
            assert!(matches!(
                validate_qwen35_w4_inventory(&value),
                Err(W4Error::Inventory { field: actual, .. }) if actual == field
            ));
        }
    }

    #[test]
    fn inventory_rejects_invalid_manifest_identity() {
        let mut value = manifest();
        value.vocab_size = 248_044;
        assert!(matches!(
            validate_qwen35_w4_inventory(&value),
            Err(W4Error::Manifest(_))
        ));
    }

    #[test]
    fn header_inventory_bridge_normalizes_one_layer_and_validates_it() {
        let prefix = "model.layers.0.self_attn";
        let raw = EXPECTED
            .iter()
            .map(|expected| TensorManifest {
                name: format!("{prefix}.{}", expected.name),
                shape: expected.shape.to_vec(),
                dtype: expected.dtype.clone(),
                quantization_role: None,
                pack_axis: None,
                group_size: None,
            })
            .collect::<Vec<_>>();
        let normalized = build_qwen35_w4_layer_manifest(prefix, &raw).unwrap();
        validate_qwen35_w4_inventory(&normalized).unwrap();
        assert_eq!(normalized.tensors[0].name, "k_proj.weight_packed");
        assert_eq!(
            normalized.tensors[0].quantization_role,
            Some(QuantizationRole::PackedWeight)
        );
    }

    #[test]
    fn synthetic_pack_round_trip_covers_tails_and_extremes() {
        for len in [7usize, 8, 9, 35] {
            let values = (0..len)
                .map(|index| if index % 2 == 0 { 0 } else { 15 })
                .collect::<Vec<_>>();
            let packed = pack_nibbles(&values).unwrap();
            assert_eq!(unpack_nibbles(&packed, len).unwrap(), values);
            if len % 8 != 0 {
                let used_bits = (len % 8) * 4;
                assert_eq!(packed.last().unwrap() >> used_bits, 0);
            }
        }
        assert_eq!(
            pack_nibbles(&[0, 16]),
            Err(W4Error::InvalidNibble {
                index: 1,
                value: 16
            })
        );
    }

    #[test]
    fn synthetic_group_boundary_uses_k_group_32() {
        let mut values = vec![8u8; 35];
        values[31] = 15;
        values[32] = 0;
        values[33] = 15;
        let output = dequantize_grouped(&values, &[0.5, 2.0], &[7, 1], 32).unwrap();
        assert_eq!(output[31], 4.0);
        assert_eq!(output[32], -2.0);
        assert_eq!(output[33], 28.0);
        assert_eq!(output.len(), 35);
    }

    #[test]
    fn synthetic_rejects_out_of_range_zero_point_nibble() {
        assert_eq!(
            dequantize_grouped(&[8], &[1.0], &[16], 32),
            Err(W4Error::InvalidZeroPoint {
                group: 0,
                value: 16,
            })
        );
    }
}
