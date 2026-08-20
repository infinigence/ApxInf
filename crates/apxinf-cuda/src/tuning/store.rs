use std::collections::HashMap;

use apxinf_core::{Error, Result};

use super::key::{GemmBucketKey, GemmTuningKey};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TacticBackend {
    Cutlass,
    CublasLt,
    Vendor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TacticId {
    pub backend: TacticBackend,
    pub value: i32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GemmTuningRecord {
    pub key: GemmTuningKey,
    pub tactic: TacticId,
    pub milliseconds: Option<f64>,
}

/// Immutable, cross-model tactic lookup installed before graph capture.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TacticStore {
    exact_gemm: HashMap<GemmTuningKey, GemmTuningRecord>,
    bucket_gemm: HashMap<GemmBucketKey, GemmTuningRecord>,
}

impl TacticStore {
    pub fn from_gemm_records(records: impl IntoIterator<Item = GemmTuningRecord>) -> Result<Self> {
        let mut exact_gemm: HashMap<GemmTuningKey, GemmTuningRecord> = HashMap::new();
        let mut bucket_gemm: HashMap<GemmBucketKey, GemmTuningRecord> = HashMap::new();
        for record in records {
            if let Some(existing) = exact_gemm.get(&record.key) {
                if existing.tactic != record.tactic {
                    return Err(Error::Other(format!(
                        "conflicting tuning records for {:?}",
                        record.key
                    )));
                }
                if !is_faster(&record, existing) {
                    continue;
                }
            }
            exact_gemm.insert(record.key.clone(), record.clone());
            let bucket = record.key.bucket();
            match bucket_gemm.get(&bucket) {
                Some(existing) if !is_faster(&record, existing) => {}
                _ => {
                    bucket_gemm.insert(bucket, record);
                }
            }
        }
        Ok(Self {
            exact_gemm,
            bucket_gemm,
        })
    }

    /// Merge records loaded from several validated databases. Identical exact
    /// records are deduplicated; conflicting tactics for one physical key fail.
    pub fn merge(stores: impl IntoIterator<Item = Self>) -> Result<Self> {
        Self::from_gemm_records(
            stores
                .into_iter()
                .flat_map(|store| store.exact_gemm.into_values()),
        )
    }

    pub fn lookup_gemm(&self, key: &GemmTuningKey) -> Option<TacticId> {
        self.exact_gemm
            .get(key)
            .or_else(|| self.bucket_gemm.get(&key.bucket()))
            .map(|record| record.tactic)
    }

    pub fn lookup_gemm_exact(&self, key: &GemmTuningKey) -> Option<TacticId> {
        self.exact_gemm.get(key).map(|record| record.tactic)
    }

    pub fn gemm_records(&self) -> impl Iterator<Item = &GemmTuningRecord> {
        self.exact_gemm.values()
    }

    pub fn len(&self) -> usize {
        self.exact_gemm.len()
    }

    pub fn is_empty(&self) -> bool {
        self.exact_gemm.is_empty()
    }
}

fn is_faster(candidate: &GemmTuningRecord, current: &GemmTuningRecord) -> bool {
    match (candidate.milliseconds, current.milliseconds) {
        (Some(candidate), Some(current)) => candidate < current,
        (Some(_), None) => true,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tuning::{DeviceFingerprint, Epilogue, GemmLayout, GemmOp, ScaleMode, TuningDType};

    fn key(m: usize) -> GemmTuningKey {
        GemmTuningKey {
            op: GemmOp::Fp8F16,
            device: DeviceFingerprint {
                sm: 110,
                multiprocessor_count: 20,
            },
            m,
            n: 1024,
            k: 1024,
            activation_dtype: TuningDType::F8E4M3,
            weight_dtype: TuningDType::F8E4M3,
            output_dtype: TuningDType::F16,
            layout: GemmLayout::RowMajor,
            scale_mode: ScaleMode::PerTensor,
            epilogue: Epilogue::None,
            workspace_limit: usize::MAX,
        }
    }

    fn record(m: usize, value: i32, milliseconds: f64) -> GemmTuningRecord {
        GemmTuningRecord {
            key: key(m),
            tactic: TacticId {
                backend: TacticBackend::Cutlass,
                value,
            },
            milliseconds: Some(milliseconds),
        }
    }

    #[test]
    fn lookup_prefers_exact_then_fastest_bucket_then_none() {
        let store =
            TacticStore::from_gemm_records([record(10, 1, 0.03), record(12, 2, 0.01)]).unwrap();
        assert_eq!(store.lookup_gemm(&key(10)).unwrap().value, 1);
        assert_eq!(store.lookup_gemm(&key(11)).unwrap().value, 2);
        assert!(store.lookup_gemm(&key(17)).is_none());
    }

    #[test]
    fn merge_deduplicates_equal_records_and_rejects_conflicts() {
        let left = TacticStore::from_gemm_records([record(10, 1, 0.03)]).unwrap();
        let right = TacticStore::from_gemm_records([record(10, 1, 0.01)]).unwrap();
        let merged = TacticStore::merge([left, right]).unwrap();
        assert_eq!(merged.len(), 1);

        let left = TacticStore::from_gemm_records([record(10, 1, 0.03)]).unwrap();
        let conflict = TacticStore::from_gemm_records([record(10, 2, 0.01)]).unwrap();
        assert!(TacticStore::merge([left, conflict]).is_err());
    }
}
