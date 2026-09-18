//! Runtime-owned tuning state.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, RwLock};

use apxinf_core::{Error, Result};

use crate::context::CudaLibraryVersions;
use crate::device_caps::CudaDeviceCaps;

use super::db::{major_minor, versions_compatible};
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

    /// The store belonging to one CUDA/cuBLAS pair, under the hardware
    /// directory. The subdirectory name is built from the same major.minor
    /// truncation the loader uses to accept or reject a record, so a store
    /// found here is a store this toolkit can use in full.
    pub fn for_cuda_toolkit(
        root: impl AsRef<Path>,
        caps: &CudaDeviceCaps,
        versions: &CudaLibraryVersions,
    ) -> Self {
        let directory = root
            .as_ref()
            .join("nvidia")
            .join(hardware_directory_name(caps))
            .join(toolkit_directory_name(versions));
        Self {
            tactics: directory.join("tactics.json"),
            report: directory.join("tuning_report.json"),
            directory,
        }
    }

    /// Pick the store this toolkit can actually use, preferring a
    /// toolkit-qualified one and otherwise keeping the unqualified store that
    /// shipped before this existed.
    ///
    /// Tactics are only valid for the library that measured them: the loader
    /// drops every record whose recorded CUDA/cuBLAS major.minor differs from
    /// the running one, so a store recorded under another toolkit is loaded,
    /// rejected record by record, and silently replaced by the untuned
    /// heuristics. All three Qwen-Drive boards hit exactly that -- Orin
    /// 12.6 against 13.2, Thor 13.0 against 13.2, the RTX 4090 12.8 against
    /// 12.3 -- and the last of those additionally held another model's
    /// shapes. Routing each toolkit to its own subdirectory lets the stores
    /// coexist instead of overwriting one another.
    ///
    /// Resolution order:
    /// 1. `<hardware>/<toolkit>/tactics.json`, if it exists;
    /// 2. `<hardware>/tactics.json`, if it exists and its header matches the
    ///    running libraries -- the path every existing checkout uses;
    /// 3. `<hardware>/<toolkit>/tactics.json` otherwise, which is where an
    ///    autotune pass then writes without disturbing a store recorded for
    ///    another toolkit.
    ///
    /// An unreadable unqualified store is returned rather than stepped over,
    /// so the caller still reports the parse error instead of quietly
    /// starting from nothing.
    pub fn resolve_for_cuda(
        root: impl AsRef<Path>,
        caps: &CudaDeviceCaps,
        versions: &CudaLibraryVersions,
    ) -> Self {
        let root = root.as_ref();
        let toolkit = Self::for_cuda_toolkit(root, caps, versions);
        if toolkit.tactics.is_file() {
            return toolkit;
        }
        let unqualified = Self::for_cuda(root, caps);
        if unqualified.tactics.is_file() && tactics_usable_by(&unqualified.tactics, versions) {
            return unqualified;
        }
        toolkit
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

/// `cuda<major>.<minor>-cublas<major>.<minor>`, the exact pair the record
/// loader compares. Both are named because a toolkit can ship a cuBLAS
/// update without moving the CUDA runtime version.
fn toolkit_directory_name(versions: &CudaLibraryVersions) -> String {
    format!(
        "cuda{}-cublas{}",
        major_minor(&versions.cuda),
        major_minor(&versions.cublas)
    )
}

/// Whether every record in an existing store would survive the loader's
/// version check. A store that cannot be parsed counts as usable so the
/// caller reports the parse failure rather than this function hiding it.
fn tactics_usable_by(path: &Path, versions: &CudaLibraryVersions) -> bool {
    match TuningDb::from_json_file(path) {
        Ok(database) => {
            versions_compatible(database.header.cuda_version.as_deref(), &versions.cuda)
                && versions_compatible(database.header.cublas_version.as_deref(), &versions.cublas)
        }
        Err(_) => true,
    }
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

    fn versions(cuda: &str, cublas: &str) -> CudaLibraryVersions {
        CudaLibraryVersions {
            cuda: cuda.into(),
            cublas: cublas.into(),
        }
    }

    fn scratch_root(tag: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "apxinf-tuning-paths-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    fn write_store(path: &Path, cuda: &str, cublas: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(
            path,
            format!(
                r#"{{"cuda_version":"{cuda}","cublas_version":"{cublas}","device_name":"NVIDIA GeForce RTX 4090","sm":89,"records":[]}}"#
            ),
        )
        .unwrap();
    }

    #[test]
    fn toolkit_directory_names_both_libraries() {
        let paths =
            TuningPaths::for_cuda_toolkit("configs/tuning", &caps("NVIDIA Thor", 110), &versions("13.2", "13.2.1"));
        assert_eq!(
            paths.tactics,
            Path::new("configs/tuning/nvidia/thor-sm110/cuda13.2-cublas13.2/tactics.json")
        );
    }

    #[test]
    fn resolution_keeps_a_matching_unqualified_store() {
        let root = scratch_root("match");
        let device = caps("NVIDIA GeForce RTX 4090", 89);
        write_store(
            &root.join("nvidia/rtx4090-sm89/tactics.json"),
            "12.3",
            "12.3.02",
        );
        assert_eq!(
            TuningPaths::resolve_for_cuda(&root, &device, &versions("12.3", "12.3.02")).tactics,
            root.join("nvidia/rtx4090-sm89/tactics.json")
        );
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn resolution_steps_over_a_store_recorded_for_another_toolkit() {
        // The RTX 4090 case: a store recorded under 12.8 on a box running
        // 12.3, whose records the loader would reject one by one.
        let root = scratch_root("mismatch");
        let device = caps("NVIDIA GeForce RTX 4090", 89);
        let unqualified = root.join("nvidia/rtx4090-sm89/tactics.json");
        write_store(&unqualified, "12.8", "12.8.4");

        let running = versions("12.3", "12.3.02");
        let resolved = TuningPaths::resolve_for_cuda(&root, &device, &running);
        assert_eq!(
            resolved.tactics,
            root.join("nvidia/rtx4090-sm89/cuda12.3-cublas12.3/tactics.json")
        );
        assert!(!resolved.tactics.is_file());

        // Once that toolkit has a store of its own it wins outright, and the
        // 12.8 store is still there for the box it was recorded on.
        write_store(&resolved.tactics, "12.3", "12.3.02");
        assert_eq!(
            TuningPaths::resolve_for_cuda(&root, &device, &running).tactics,
            resolved.tactics
        );
        assert_eq!(
            TuningPaths::resolve_for_cuda(&root, &device, &versions("12.8", "12.8.4")).tactics,
            unqualified
        );
        std::fs::remove_dir_all(&root).ok();
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
