use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const QWEN35_MODEL_VOCAB_SIZE: usize = 248_320;
pub const QWEN35_MODEL_REVISION: &str = "63768c10df38c0395e12ef49edac1bd539eaeeea";
pub const LOADER_MANIFEST_SCHEMA: &str = "apxinf.loader-manifest.v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ManifestDType {
    I32,
    BF16,
    F16,
    F32,
    Other(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PackAxis {
    N,
    K,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationRole {
    PackedWeight,
    Scale,
    ZeroPoint,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorManifest {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: ManifestDType,
    pub quantization_role: Option<QuantizationRole>,
    pub pack_axis: Option<PackAxis>,
    pub group_size: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoaderManifest {
    pub schema: String,
    pub revision: String,
    pub vocab_size: usize,
    pub tensors: Vec<TensorManifest>,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ManifestError {
    #[error("manifest schema must be {expected}, got {actual}")]
    Schema {
        expected: &'static str,
        actual: String,
    },
    #[error("manifest revision must be {expected}, got {actual}")]
    Revision {
        expected: &'static str,
        actual: String,
    },
    #[error("manifest vocab_size must be {expected}, got {actual}")]
    VocabSize { expected: usize, actual: usize },
    #[error("duplicate tensor name `{0}`")]
    DuplicateTensor(String),
    #[error("tensor `{0}` must have a non-empty shape with nonzero dimensions")]
    InvalidShape(String),
    #[error("tensor `{0}` group_size must be positive when present")]
    InvalidGroupSize(String),
}

impl LoaderManifest {
    pub fn validate(&self) -> Result<(), ManifestError> {
        if self.schema != LOADER_MANIFEST_SCHEMA {
            return Err(ManifestError::Schema {
                expected: LOADER_MANIFEST_SCHEMA,
                actual: self.schema.clone(),
            });
        }
        if self.revision != QWEN35_MODEL_REVISION {
            return Err(ManifestError::Revision {
                expected: QWEN35_MODEL_REVISION,
                actual: self.revision.clone(),
            });
        }
        if self.vocab_size != QWEN35_MODEL_VOCAB_SIZE {
            return Err(ManifestError::VocabSize {
                expected: QWEN35_MODEL_VOCAB_SIZE,
                actual: self.vocab_size,
            });
        }
        let mut names = BTreeSet::new();
        for tensor in &self.tensors {
            if !names.insert(tensor.name.as_str()) {
                return Err(ManifestError::DuplicateTensor(tensor.name.clone()));
            }
            if tensor.shape.is_empty() || tensor.shape.iter().any(|dimension| *dimension == 0) {
                return Err(ManifestError::InvalidShape(tensor.name.clone()));
            }
            if tensor.group_size == Some(0) {
                return Err(ManifestError::InvalidGroupSize(tensor.name.clone()));
            }
        }
        Ok(())
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorManifest> {
        self.tensors.iter().find(|tensor| tensor.name == name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(name: &str) -> TensorManifest {
        TensorManifest {
            name: name.to_owned(),
            shape: vec![2, 4],
            dtype: ManifestDType::I32,
            quantization_role: Some(QuantizationRole::PackedWeight),
            pack_axis: Some(PackAxis::K),
            group_size: None,
        }
    }

    fn manifest() -> LoaderManifest {
        LoaderManifest {
            schema: LOADER_MANIFEST_SCHEMA.to_owned(),
            revision: QWEN35_MODEL_REVISION.to_owned(),
            vocab_size: QWEN35_MODEL_VOCAB_SIZE,
            tensors: vec![tensor("a"), tensor("b")],
        }
    }

    #[test]
    fn manifest_validates_and_round_trips_stably() {
        let value = manifest();
        value.validate().unwrap();
        assert_eq!(value.tensor("b").unwrap().shape, [2, 4]);
        let json = serde_json::to_string(&value).unwrap();
        let decoded: LoaderManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, value);
    }

    #[test]
    fn manifest_rejects_identity_and_inventory_errors() {
        let mut value = manifest();
        value.revision = "wrong-revision".to_owned();
        assert!(matches!(
            value.validate(),
            Err(ManifestError::Revision { .. })
        ));

        let mut value = manifest();
        value.schema = "apxinf.loader-manifest.v2".to_owned();
        assert!(matches!(
            value.validate(),
            Err(ManifestError::Schema { .. })
        ));

        let mut value = manifest();
        value.vocab_size = 248_044;
        assert!(matches!(
            value.validate(),
            Err(ManifestError::VocabSize { .. })
        ));

        let mut value = manifest();
        value.tensors[1].name = "a".to_owned();
        assert_eq!(
            value.validate(),
            Err(ManifestError::DuplicateTensor("a".to_owned()))
        );

        let mut value = manifest();
        value.tensors[0].shape = vec![2, 0];
        assert_eq!(
            value.validate(),
            Err(ManifestError::InvalidShape("a".to_owned()))
        );
    }
}
