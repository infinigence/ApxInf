//! Runtime-owned tuning state.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, RwLock};

use apxinf_core::{Error, Result};

use crate::context::CudaLibraryVersions;
use crate::device_caps::CudaDeviceCaps;

use super::{GemmTuningKey, GemmTuningRecord, TacticId, TacticStore, TuningDb, TuningOutcome};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TuningMode {
    #[default]
    Inference,
    AutoTune,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TacticMatch {
    Exact,
    Bucket,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResolvedTactic {
    pub tactic: TacticId,
    pub source: TacticMatch,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TuningPaths {
    pub directory: PathBuf,
    pub tactics: PathBuf,
    pub report: PathBuf,
}

impl TuningPaths {
    pub fn for_cuda(root: impl AsRef<Path>, caps: &CudaDeviceCaps) -> Self {
        let hardware = hardware_directory_name(caps);
        let directory = root.as_ref().join("nvidia").join(hardware);
        Self {
            tactics: directory.join("tactics.json"),
            report: directory.join("tuning_report.json"),
            directory,
        }
    }

    pub fn from_tactics(path: impl Into<PathBuf>) -> Self {
        let tactics = path.into();
        let directory = tactics
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        Self {
            report: directory.join("tuning_report.json"),
            directory,
            tactics,
        }
    }
}

/// Control-plane state owned by one CUDA runtime. GEMM plans retain the
/// resolved tactic, so graph replay never locks or reads this session.
#[derive(Debug)]
pub struct TuningSession {
    mode: TuningMode,
    store: RwLock<TacticStore>,
    tune_lock: Mutex<()>,
    generation: AtomicU64,
    paths: Option<TuningPaths>,
}

impl TuningSession {
    pub fn new(mode: TuningMode, store: TacticStore, paths: Option<TuningPaths>) -> Self {
        Self {
            mode,
            store: RwLock::new(store),
            tune_lock: Mutex::new(()),
            generation: AtomicU64::new(0),
            paths,
        }
    }

    pub fn inference(store: TacticStore) -> Self {
        Self::new(TuningMode::Inference, store, None)
    }

    pub fn mode(&self) -> TuningMode {
        self.mode
    }

    pub fn paths(&self) -> Option<&TuningPaths> {
        self.paths.as_ref()
    }

    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    pub fn lookup_gemm(&self, key: &GemmTuningKey) -> Option<ResolvedTactic> {
        let store = self.store.read().ok()?;
        store
            .lookup_gemm_exact(key)
            .map(|tactic| ResolvedTactic {
                tactic,
                source: TacticMatch::Exact,
            })
            .or_else(|| {
                store.lookup_gemm_bucket(key).map(|tactic| ResolvedTactic {
                    tactic,
                    source: TacticMatch::Bucket,
                })
            })
    }

    pub fn lookup_gemm_exact(&self, key: &GemmTuningKey) -> Option<TacticId> {
        self.store.read().ok()?.lookup_gemm_exact(key)
    }

    /// Tune one exact miss from the real operands which triggered it. Calls
    /// for the same runtime are serialized because native provider plan maps
    /// are mutated while candidates are evaluated.
    pub(crate) fn tune_gemm(
        &self,
        caps: &CudaDeviceCaps,
        versions: &CudaLibraryVersions,
        key: &GemmTuningKey,
        tune: impl FnOnce(Option<TacticId>) -> Result<TuningOutcome>,
    ) -> Result<ResolvedTactic> {
        if self.mode != TuningMode::AutoTune {
            return self.lookup_gemm(key).ok_or_else(|| {
                Error::Other("cannot tune a missing tactic in INFERENCE mode".into())
            });
        }
        let _guard = self
            .tune_lock
            .lock()
            .map_err(|_| Error::Other("CUDA autotune lock is poisoned".into()))?;
        if let Some(tactic) = self.lookup_gemm_exact(key) {
            return Ok(ResolvedTactic {
                tactic,
                source: TacticMatch::Exact,
            });
        }
        let preferred = self
            .store
            .read()
            .map_err(|_| Error::Other("CUDA tactic store lock is poisoned".into()))?
            .lookup_gemm_bucket(key);
        let outcome = tune(preferred)?;
        if outcome.winner.key != *key {
            return Err(Error::Other(
                "autotune outcome key does not match the requested GEMM".into(),
            ));
        }
        self.publish_gemm_persisted(caps, versions, outcome.winner.clone())?;
        if let Some(paths) = self.paths.as_ref() {
            super::report::append_outcome(&paths.report, caps, versions, &outcome)?;
        }
        let tactic = self.lookup_gemm_exact(key).ok_or_else(|| {
            Error::Other("persisted GEMM winner is missing after publication".into())
        })?;
        Ok(ResolvedTactic {
            tactic,
            source: TacticMatch::Exact,
        })
    }

    pub fn snapshot(&self) -> Result<TacticStore> {
        self.store
            .read()
            .map(|store| store.clone())
            .map_err(|_| Error::Other("CUDA tactic store lock is poisoned".into()))
    }

    /// Publish a newly verified exact winner. Persistence is performed by the
    /// caller after the report and database payload have both been assembled.
    pub fn publish_gemm(&self, record: GemmTuningRecord) -> Result<bool> {
        if self.mode != TuningMode::AutoTune {
            return Err(Error::Other(
                "cannot publish a tactic in INFERENCE mode".into(),
            ));
        }
        let mut store = self
            .store
            .write()
            .map_err(|_| Error::Other("CUDA tactic store lock is poisoned".into()))?;
        let changed = store.upsert_gemm(record);
        if changed {
            self.generation.fetch_add(1, Ordering::AcqRel);
        }
        Ok(changed)
    }

    /// Publish and durably merge a winner into the hardware database. The
    /// latest file is re-read while locked so concurrent model processes do
    /// not discard one another's newly discovered exact keys.
    pub fn publish_gemm_persisted(
        &self,
        caps: &CudaDeviceCaps,
        versions: &CudaLibraryVersions,
        record: GemmTuningRecord,
    ) -> Result<bool> {
        if self.mode != TuningMode::AutoTune {
            return Err(Error::Other(
                "cannot publish a tactic in INFERENCE mode".into(),
            ));
        }
        let Some(paths) = self.paths.as_ref() else {
            return self.publish_gemm(record);
        };
        let header = TuningDb::header_for_cuda(caps, versions);
        let merged =
            TuningDb::merge_record_atomic(&paths.tactics, &header, caps, versions, record)?;
        let mut store = self
            .store
            .write()
            .map_err(|_| Error::Other("CUDA tactic store lock is poisoned".into()))?;
        if *store == merged {
            return Ok(false);
        }
        *store = merged;
        self.generation.fetch_add(1, Ordering::AcqRel);
        Ok(true)
    }
}

fn hardware_directory_name(caps: &CudaDeviceCaps) -> String {
    let lower = caps.device_name.to_ascii_lowercase();
    let family = if lower.contains("thor") {
        "thor".to_owned()
    } else if lower.contains("orin") {
        "orin".to_owned()
    } else if lower.contains("4090") {
        "rtx4090".to_owned()
    } else {
        lower
            .strip_prefix("nvidia ")
            .unwrap_or(&lower)
            .chars()
            .map(|character| {
                if character.is_ascii_alphanumeric() {
                    character
                } else {
                    '-'
                }
            })
            .collect::<String>()
            .split('-')
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>()
            .join("-")
    };
    format!("{family}-sm{}", caps.sm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device_caps::CudaArchFamily;
    use crate::tuning::{
        DeviceFingerprint, Epilogue, GemmLayout, GemmOp, ScaleMode, TacticBackend, TuningDType,
    };

    fn caps(name: &str, sm: u32) -> CudaDeviceCaps {
        CudaDeviceCaps {
            device_name: name.into(),
            compute_major: sm / 10,
            compute_minor: sm % 10,
            sm,
            multiprocessor_count: 14,
            arch_family: CudaArchFamily::Sm100,
        }
    }

    fn key() -> GemmTuningKey {
        GemmTuningKey {
            op: GemmOp::Fp8F16,
            device: DeviceFingerprint {
                sm: 110,
                multiprocessor_count: 14,
            },
            m: 522,
            n: 32768,
            k: 2048,
            activation_dtype: TuningDType::F8E4M3,
            weight_dtype: TuningDType::F8E4M3,
            output_dtype: TuningDType::F16,
            layout: GemmLayout::RowMajor,
            scale_mode: ScaleMode::PerTensor,
            epilogue: Epilogue::None,
            workspace_limit: usize::MAX,
        }
    }

    fn record() -> GemmTuningRecord {
        GemmTuningRecord {
            key: key(),
            tactic: TacticId {
                backend: TacticBackend::Cutlass,
                value: 3,
            },
            implementation_version: Some(TacticBackend::Cutlass.implementation_version()),
            milliseconds: Some(0.1),
        }
    }

    #[test]
    fn resolves_hardware_paths_without_build_id() {
        let paths = TuningPaths::for_cuda("configs/tuning", &caps("NVIDIA Thor", 110));
        assert_eq!(
            paths.tactics,
            Path::new("configs/tuning/nvidia/thor-sm110/tactics.json")
        );
        assert_eq!(
            paths.report,
            Path::new("configs/tuning/nvidia/thor-sm110/tuning_report.json")
        );
        assert_eq!(
            TuningPaths::for_cuda("configs/tuning", &caps("NVIDIA Jetson Orin", 87)).tactics,
            Path::new("configs/tuning/nvidia/orin-sm87/tactics.json")
        );
        assert_eq!(
            TuningPaths::for_cuda("configs/tuning", &caps("NVIDIA GeForce RTX 4090", 89)).tactics,
            Path::new("configs/tuning/nvidia/rtx4090-sm89/tactics.json")
        );
    }

    #[test]
    fn inference_session_cannot_publish() {
        let session = TuningSession::inference(TacticStore::default());
        assert!(session.publish_gemm(record()).is_err());
    }

    #[test]
    fn autotune_publish_updates_generation_and_exact_lookup() {
        let session = TuningSession::new(TuningMode::AutoTune, TacticStore::default(), None);
        assert!(session.publish_gemm(record()).unwrap());
        assert_eq!(session.generation(), 1);
        assert_eq!(
            session.lookup_gemm(&key()).unwrap().source,
            TacticMatch::Exact
        );
    }
}
