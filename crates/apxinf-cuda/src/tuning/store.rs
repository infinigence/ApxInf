use std::collections::HashMap;

use apxinf_core::{Error, Result};

use super::key::{GemmBucketKey, GemmTuningKey};
use super::tactic::TacticId;

#[cfg(test)]
use super::tactic::TacticBackend;

#[derive(Clone, Debug, PartialEq)]
pub struct GemmTuningRecord {
    pub key: GemmTuningKey,
    pub tactic: TacticId,
    /// Missing on legacy records, which remain accepted. Newly generated
    /// records carry the selected provider family's compatibility revision.
    pub implementation_version: Option<u32>,
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
                if existing.tactic != record.tactic
                    && (existing.milliseconds.is_none() || record.milliseconds.is_none())
                {
                    return Err(Error::Other(format!(
                        "conflicting unmeasured tuning records for {:?}",
                        record.key
                    )));
                }
                if !is_faster(&record, existing) {
                    continue;
                }
            }
            exact_gemm.insert(record.key.clone(), record.clone());
            if !record.tactic.backend.bucket_eligible() {
                continue;
            }
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

    /// Merge records loaded from validated databases. Identical exact records
    /// are deduplicated; measured conflicts keep the faster winner, while an
    /// unmeasured conflict is rejected because it has no ordering evidence.
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

    pub fn lookup_gemm_bucket(&self, key: &GemmTuningKey) -> Option<TacticId> {
        self.bucket_gemm
            .get(&key.bucket())
            .map(|record| record.tactic)
    }

    /// Add or replace one exact winner and rebuild the derived bucket index.
    /// Returns whether the store changed.
    pub fn upsert_gemm(&mut self, record: GemmTuningRecord) -> bool {
        if let Some(existing) = self.exact_gemm.get(&record.key) {
            if existing == &record || !is_faster(&record, existing) {
                return false;
            }
        }
        self.exact_gemm.insert(record.key.clone(), record);
        self.rebuild_buckets();
        true
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

    fn rebuild_buckets(&mut self) {
        self.bucket_gemm.clear();
        for record in self.exact_gemm.values() {
            if !record.tactic.backend.bucket_eligible() {
                continue;
            }
            let bucket = record.key.bucket();
            match self.bucket_gemm.get(&bucket) {
                Some(existing) if !is_faster(record, existing) => {}
                _ => {
                    self.bucket_gemm.insert(bucket, record.clone());
                }
            }
        }
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
            implementation_version: Some(TacticBackend::Cutlass.implementation_version()),
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
    fn merge_deduplicates_and_keeps_the_fastest_measured_winner() {
        let left = TacticStore::from_gemm_records([record(10, 1, 0.03)]).unwrap();
        let right = TacticStore::from_gemm_records([record(10, 1, 0.01)]).unwrap();
        let merged = TacticStore::merge([left, right]).unwrap();
        assert_eq!(merged.len(), 1);

        let left = TacticStore::from_gemm_records([record(10, 1, 0.03)]).unwrap();
        let conflict = TacticStore::from_gemm_records([record(10, 2, 0.01)]).unwrap();
        let merged = TacticStore::merge([left, conflict]).unwrap();
        assert_eq!(merged.lookup_gemm_exact(&key(10)).unwrap().value, 2);
    }

    #[test]
    fn upsert_replaces_exact_and_rebuilds_bucket() {
        let mut store = TacticStore::from_gemm_records([record(10, 1, 0.03)]).unwrap();
        assert!(store.upsert_gemm(record(10, 4, 0.02)));
        assert_eq!(store.lookup_gemm_exact(&key(10)).unwrap().value, 4);
        assert_eq!(store.lookup_gemm_bucket(&key(11)).unwrap().value, 4);
        assert!(!store.upsert_gemm(record(10, 4, 0.02)));
        assert!(!store.upsert_gemm(record(10, 5, 0.04)));
    }
}
