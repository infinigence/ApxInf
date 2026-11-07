//! Tuning seam: modes, tactic stores and where a persisted database lives.
//!
//! Native BF16 preparation compares muBLAS/muDNN on actual operands and binds
//! accepted records to hardware, KN layout, kernel build and vendor versions.
//! Every session and mutation has a process-unique generation: a prepared
//! graph cannot silently reuse resources from a replaced tactic store.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock, RwLock};

use apxinf_core::{Error, Result};

use crate::context::{MusaContext, MusaLibraryVersions};
use crate::device_caps::MusaDeviceCaps;
use crate::error::musa_error;

/// How aggressively execution resources may be selected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TuningMode {
    /// No measurement: use the provider's safe default.
    Inference,
    /// Measure candidates during prepare and persist the winner.
    AutoTune,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Bf16Provider {
    Mublas,
    Mudnn,
}

impl Bf16Provider {
    /// The spelling a record uses for this provider — one home for the two
    /// strings `parse_record` reads back and `record_bf16` writes.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Mublas => "mublas",
            Self::Mudnn => "mudnn",
        }
    }
}

static NEXT_GENERATION: AtomicU64 = AtomicU64::new(1);
fn next_generation() -> u64 {
    NEXT_GENERATION.fetch_add(1, Ordering::AcqRel)
}

/// Persisted selection for one workload.
///
/// Kept verbatim rather than decoded: the records in a real database describe
/// provider-specific tactics, and no MUSA provider consumes one yet. What is
/// checked here is that the file is a database at all, and that it does not
/// name a CUDA provider — a `cublaslt` tactic id cannot be executed on M1000
/// hardware, so silently accepting it would be the beginning of a wrong
/// answer.
#[derive(Clone, Debug, Default)]
pub struct TuningDb {
    records: Vec<serde_json::Value>,
    device_name: Option<String>,
    fingerprint: Fingerprint,
}

/// What a typed record is bound to. Kept as the checked pieces rather than as
/// the parsed document: the document also holds `records`, which the sibling
/// field already owns, so carrying it whole duplicated the file in memory.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Fingerprint {
    kernel_build_id: Option<String>,
    mp: Option<u64>,
    library_versions: Option<serde_json::Value>,
}

impl Fingerprint {
    fn from_document(value: &serde_json::Value) -> Self {
        Self {
            kernel_build_id: value["kernel_build_id"].as_str().map(str::to_string),
            mp: value["mp"].as_u64(),
            library_versions: match value.get("library_versions") {
                Some(versions) if versions.get("binaries").is_some() => Some(versions.clone()),
                _ => None,
            },
        }
    }
}

impl TuningDb {
    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|error| Error::Other(format!("MUSA tuning database {path:?}: {error}")))?;
        let value: serde_json::Value = serde_json::from_str(&text)
            .map_err(|error| Error::Other(format!("MUSA tuning database {path:?}: {error}")))?;
        let records = value
            .get("records")
            .and_then(|records| records.as_array())
            .ok_or_else(|| {
                Error::Other(format!(
                    "MUSA tuning database {path:?} has no `records` array"
                ))
            })?
            .clone();
        for record in &records {
            let backend = record
                .get("tactic")
                .and_then(|tactic| tactic.get("backend"))
                .or_else(|| record.get("backend"))
                .and_then(|backend| backend.as_str())
                .unwrap_or("");
            if matches!(backend, "cublas" | "cublaslt" | "cutlass") {
                return Err(Error::Other(format!(
                    "MUSA tuning database {path:?} names the CUDA provider `{backend}`; \
                     that tactic cannot be executed on MUSA hardware"
                )));
            }
        }
        // What was *read*, which is not the same as what will be accepted: a
        // versioned document can still be stale (`stale_reason`), and the
        // rejection says so on its own line. Between them, a latency number has
        // an answer to "which database produced this" that does not depend on
        // anyone remembering.
        eprintln!(
            "[apxinf] MUSA BF16 tactics: read {} records from {}",
            records.len(),
            path.display()
        );
        Ok(Self {
            records,
            device_name: value
                .get("device_name")
                .and_then(|name| name.as_str())
                .map(str::to_string),
            fingerprint: Fingerprint::from_document(&value),
        })
    }

    pub fn device_name(&self) -> Option<&str> {
        self.device_name.as_deref()
    }

    pub fn record_count(&self) -> usize {
        self.records.len()
    }

    /// Build the snapshot a session resolves against.
    ///
    /// Degrades instead of failing. A tuning database is an optimisation cache,
    /// so a record that no longer applies — the device, the kernel build or a
    /// mapped vendor binary changed under it — must cost the *tuning*, not the
    /// model. Every rejection lands on muBLAS, which is also what a missing
    /// record selects, and says so once so it is not a silent loss.
    ///
    /// `versions` is the running context's fingerprint. An empty `binaries` map
    /// means the fingerprint could not be taken at all (see
    /// `MusaContext::library_versions`); records that name one are then
    /// unverifiable and dropped, because accepting them would claim a
    /// validation that never happened.
    ///
    /// **A rejected database contributes nothing — not its tactics, and not its
    /// records.** `records` is what `record_bf16` writes back, stamped with the
    /// *current* kernel build id and the running context's fingerprint, so
    /// carrying a rejected database's records forward would re-publish a
    /// measurement this process did not make. A stale file's records must not
    /// ride along into the fresh document: the next process would accept them as
    /// measured for its own build, device and vendor libraries.
    pub fn build_store(
        &self,
        caps: &MusaDeviceCaps,
        versions: &MusaLibraryVersions,
    ) -> TacticStore {
        let mut store = TacticStore {
            records: self.records.clone(),
            bf16: HashMap::new(),
            // Rebuilt from `bf16` once the records below are in.
            bf16_buckets: HashMap::new(),
        };
        if !self.records.iter().any(|record| record["op"] == "bf16_gemm") {
            return store;
        }
        if let Some(reason) = self.stale_reason(caps, versions) {
            warn_ignored(reason);
            return TacticStore::default();
        }
        for record in &self.records {
            if record["op"] != "bf16_gemm" {
                continue;
            }
            // Past this point the database was written by this code and its
            // fingerprint matches, so anything malformed is corruption rather
            // than a changed environment. Drop the whole set — and the records
            // with it, or the malformed row would be written back, stay on
            // disk and keep rejecting the file: a partly-applied cache would be
            // harder to reason about than none.
            match parse_record(record) {
                Ok((shape, provider)) => {
                    if store.bf16.insert(shape, provider).is_some() {
                        warn_ignored("duplicate BF16 tactic shape");
                        return TacticStore::default();
                    }
                }
                Err(reason) => {
                    warn_ignored(reason);
                    return TacticStore::default();
                }
            }
        }
        store.rebuild_buckets();
        store
    }

    /// Why this database's BF16 records cannot be used, or `None` if they can.
    fn stale_reason(
        &self,
        caps: &MusaDeviceCaps,
        versions: &MusaLibraryVersions,
    ) -> Option<&'static str> {
        if self.fingerprint.kernel_build_id.as_deref() != Some(crate::KERNEL_BUILD_ID) {
            return Some("kernel build changed");
        }
        if self.fingerprint.mp != Some(caps.mp as u64) {
            return Some("MP generation changed");
        }
        if self.device_name() != Some(caps.device_name.as_str()) {
            return Some("device changed");
        }
        if versions.binaries.is_empty() {
            return Some("the running library fingerprint could not be taken");
        }
        let Some(stored) = &self.fingerprint.library_versions else {
            return Some("the database carries no library fingerprint");
        };
        let Ok(current) = serde_json::to_value(versions) else {
            return Some("the running library fingerprint could not be read");
        };
        for field in ["musa", "mublas_lt"] {
            if stored.get(field) != current.get(field) {
                return Some("vendor library changed");
            }
        }
        // The mapped-library fingerprints are compared **per name, and only
        // where both sides have that name**. `library_fingerprints` reads
        // `/proc/self/maps`, so it describes what this process loaded, and two
        // processes on one host do not load the same set: importing torch maps
        // `libmudnn.so.3.4.0.0`, a serving process that never imports it does
        // not. Comparing the two documents whole would therefore reject every
        // database across exactly the boundary a database has to cross to be
        // worth writing -- measured by the benchmark, consumed by the server.
        // A name this process has not mapped cannot take part in the run, so
        // its absence is not evidence that the vendor libraries changed; a
        // name both have, with different fingerprints, is.
        let Some(stored_binaries) = stored.get("binaries").and_then(|value| value.as_object())
        else {
            return Some("the database carries no library fingerprint");
        };
        for (name, recorded) in stored_binaries {
            let running = current.get("binaries").and_then(|value| value.get(name));
            if running.is_some_and(|running| running != recorded) {
                return Some("vendor library changed");
            }
        }
        None
    }
}

/// One BF16 record as a `(m, n, k)` key and the provider it names.
fn parse_record(record: &serde_json::Value) -> std::result::Result<((usize, usize, usize), Bf16Provider), &'static str> {
    let dimension = |name: &str| {
        record[name]
            .as_u64()
            .filter(|value| *value > 0 && *value <= i32::MAX as u64)
            .map(|value| value as usize)
            .ok_or("invalid BF16 tactic dimension")
    };
    let provider = match record["backend"].as_str() {
        Some("mublas") => Bf16Provider::Mublas,
        Some("mudnn") => Bf16Provider::Mudnn,
        _ => return Err("unsupported native BF16 tactic provider"),
    };
    if record["layout"] != "kn" {
        return Err("unsupported BF16 weight layout");
    }
    Ok((
        (dimension("m")?, dimension("n")?, dimension("k")?),
        provider,
    ))
}

/// The M a tactic record is keyed at, rounded up to its power-of-two class.
///
/// This is the rule CUDA applies (`crates/apxinf-cuda/src/tuning/key.rs:130`),
/// and the two backends have to answer the same way, because a `bf16_gemm`'s M
/// is the shape the *caller* decides: for PI0.5 it is `patches + token_count`
/// (`crates/apxinf-model/src/pi05/config.rs:244`), which moves with the prompt.
/// A table keyed on the exact M can therefore only ever serve the lengths
/// somebody happened to measure — the shipped database measured the benchmark
/// fixture's ten-token prompt, so every other prompt length silently left the
/// tuned providers on the floor. Rounding up to a power of two makes one
/// measured shape answer for every M within a factor of two of it, which is
/// what puts the whole 513..=712 prefix range (512 patch tokens plus up to
/// `max_token_len`) behind the four prefix records the database already
/// carries. `m_bucket_matches_the_cuda_rule` pins the values.
fn m_bucket(m: usize) -> usize {
    m.max(1).checked_next_power_of_two().unwrap_or(usize::MAX)
}

/// An `(m, n, k)` shape with `m` folded to its bucket.
///
/// A newtype rather than a tuple because the exact and bucket indexes are both
/// keyed on three `usize`s, and `BucketKey::of` is the only way into the bucket
/// one: a lookup that forgets to bucket is then a type error rather than a
/// quiet miss.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct BucketKey {
    m_bucket: usize,
    n: usize,
    k: usize,
}

impl BucketKey {
    fn of(m: usize, n: usize, k: usize) -> Self {
        Self {
            m_bucket: m_bucket(m),
            n,
            k,
        }
    }
}

/// How a provider was resolved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TacticMatch {
    /// A record measured at this very shape.
    Exact,
    /// A record measured at a neighbouring shape in the same bucket.
    Bucket,
}

/// A resolved provider, and where it came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ResolvedBf16 {
    pub provider: Bf16Provider,
    pub source: TacticMatch,
    /// The M of the record that answered: this shape's own for an exact hit,
    /// the neighbouring one for a bucket hit.
    pub recorded_m: usize,
}

/// The tactic decision for one BF16 GEMM shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Bf16Choice {
    /// A record measured at this shape.
    Exact(Bf16Provider),
    /// A record measured at a neighbouring shape in the same bucket.
    Bucket {
        provider: Bf16Provider,
        recorded_m: usize,
    },
    /// Measure this shape and persist the winner.
    Measure,
    /// No record covers this shape; use the safe default.
    Miss,
}

/// Decide how one BF16 GEMM shape is served.
///
/// Pure: the caller resolves the record and the autotune flag from the session,
/// which keeps the whole decision testable without a device. `require_exact` is
/// the caller's `allow_fallback = false`.
///
/// An exact record is the answer in every mode. Autotuning may turn a bucket or
/// a miss into a measurement, which is not a fallback: the measurement writes
/// the exact record the caller demanded. Without autotuning, a bucket is an
/// extrapolation and a miss is the vendor default; `require_exact` refuses both
/// rather than returning success for a constraint that did not hold.
pub(crate) fn choose_bf16(
    resolved: Option<ResolvedBf16>,
    autotune: bool,
    require_exact: bool,
    m: usize,
    n: usize,
    k: usize,
) -> Result<Bf16Choice> {
    match resolved {
        Some(resolved) if resolved.source == TacticMatch::Exact => {
            Ok(Bf16Choice::Exact(resolved.provider))
        }
        _ if autotune => Ok(Bf16Choice::Measure),
        Some(resolved) if !require_exact => Ok(Bf16Choice::Bucket {
            provider: resolved.provider,
            recorded_m: resolved.recorded_m,
        }),
        Some(resolved) => Err(musa_error(format!(
            "gemm {m}x{k}x{n} has no record measured at this shape; allow_fallback=false \
             forbids the neighbouring record at m={}",
            resolved.recorded_m
        ))),
        None if require_exact => Err(musa_error(format!(
            "gemm {m}x{k}x{n} has no tactic record; allow_fallback=false forbids the muBLAS \
             default. Enable autotuning or allow fallback"
        ))),
        None => Ok(Bf16Choice::Miss),
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum NoteKind {
    Bucket,
    Miss,
}

/// A shape and how it was reported, as the key of the once-per-shape notes.
type NoteKey = (NoteKind, usize, usize, usize);

static NOTES: OnceLock<Mutex<HashSet<NoteKey>>> = OnceLock::new();
static MISSES: AtomicUsize = AtomicUsize::new(0);
static DATABASE_REJECTED: AtomicBool = AtomicBool::new(false);

/// Distinct shapes that keep printing a miss line before the report is
/// summarized instead. A serving process meets a new prompt length per
/// request; the log should not.
const MISS_LINE_LIMIT: usize = 16;

/// Whether this `(kind, shape)` has not been reported yet.
fn note_once(kind: NoteKind, shape: (usize, usize, usize)) -> bool {
    NOTES
        .get_or_init(|| Mutex::new(HashSet::new()))
        .lock()
        .expect("MUSA tactic note lock is poisoned")
        .insert((kind, shape.0, shape.1, shape.2))
}

/// A provider came from a neighbouring record. Said once per bucket: the
/// bucket *is* the shape-generic answer, so one line covers every M in it, and
/// the number of lines is bounded by the number of records — not by traffic.
pub(crate) fn note_bf16_bucket(
    m: usize,
    n: usize,
    k: usize,
    recorded_m: usize,
    provider: Bf16Provider,
) {
    let bucket = m_bucket(m);
    if !note_once(NoteKind::Bucket, (bucket, n, k)) {
        return;
    }
    eprintln!(
        "[apxinf] MUSA BF16 tactics: bucket m={bucket} serves gemm (m={m},n={n},k={k}) \
         from recorded m={recorded_m}; using {}",
        provider.name()
    );
}

/// Nothing in the table covers this shape, so it runs muBLAS.
///
/// Said once per shape, up to a cap — and not at all once the whole database
/// was dropped: every miss then has the one cause that line already states,
/// and sixteen shape lines would bury it.
pub(crate) fn note_bf16_miss(m: usize, n: usize, k: usize) {
    if DATABASE_REJECTED.load(Ordering::Relaxed) {
        return;
    }
    if !note_once(NoteKind::Miss, (m, n, k)) {
        return;
    }
    let seen = MISSES.fetch_add(1, Ordering::Relaxed);
    if seen < MISS_LINE_LIMIT {
        eprintln!("[apxinf] MUSA BF16 tactics: miss for gemm (m={m},n={n},k={k}); using mublas");
    } else if seen == MISS_LINE_LIMIT {
        eprintln!(
            "[apxinf] MUSA BF16 tactics: {MISS_LINE_LIMIT} distinct GEMM shapes have missed; \
             further misses are not reported"
        );
    }
}

/// Say once that a database was dropped, so losing a tuning is never silent.
fn warn_ignored(reason: &str) {
    static WARNED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    WARNED.get_or_init(|| {
        // Every miss that follows shares this cause, which the line below
        // states; the per-shape notes stay quiet so it is not buried.
        DATABASE_REJECTED.store(true, Ordering::Relaxed);
        eprintln!("[apxinf] MUSA BF16 tactics ignored ({reason}); GEMM falls back to muBLAS");
    });
}

/// The tactic snapshot a session resolves against.
#[derive(Clone, Debug, Default)]
pub struct TacticStore {
    records: Vec<serde_json::Value>,
    bf16: HashMap<(usize, usize, usize), Bf16Provider>,
    /// Derived, never persisted: bucket -> recorded M -> provider. Nothing
    /// here reaches `records`, which is what keeps
    /// `shipped_tests::every_shipped_database_is_one_this_build_can_load`'s
    /// `bf16.len() == record_count()` true: a bucket entry is a way of
    /// *reading* the records, not a record.
    bf16_buckets: HashMap<BucketKey, BTreeMap<usize, Bf16Provider>>,
}

impl TacticStore {
    pub fn record_count(&self) -> usize {
        self.records.len()
    }

    fn lookup_bf16_exact(&self, m: usize, n: usize, k: usize) -> Option<Bf16Provider> {
        self.bf16.get(&(m, n, k)).copied()
    }

    /// The nearest recorded M in this shape's bucket, if the bucket has one.
    ///
    /// Nearest, not "the bucket's single winner": a MUSA record carries no
    /// timing (CUDA keeps one winner per bucket because it can rank two
    /// records by `milliseconds`), so the only quantity left to rank by is the
    /// distance between the record's M and the request's. Ties go to the
    /// smaller M — an arbitrary rule, but a written-down one.
    fn lookup_bf16_bucket(&self, m: usize, n: usize, k: usize) -> Option<(usize, Bf16Provider)> {
        let bucket = self.bf16_buckets.get(&BucketKey::of(m, n, k))?;
        let below = bucket.range(..=m).next_back();
        let above = bucket.range(m..).next();
        match (below, above) {
            (Some((&lower, &p)), Some((&upper, &q))) => {
                Some(if m - lower <= upper - m {
                    (lower, p)
                } else {
                    (upper, q)
                })
            }
            (Some((&lower, &p)), None) => Some((lower, p)),
            (None, Some((&upper, &q))) => Some((upper, q)),
            (None, None) => None,
        }
    }

    /// Exact first, then the bucket — the precedence `configs/tuning/README.md`
    /// states for both backends.
    fn lookup_bf16(&self, m: usize, n: usize, k: usize) -> Option<ResolvedBf16> {
        if let Some(provider) = self.lookup_bf16_exact(m, n, k) {
            return Some(ResolvedBf16 {
                provider,
                source: TacticMatch::Exact,
                recorded_m: m,
            });
        }
        self.lookup_bf16_bucket(m, n, k)
            .map(|(recorded_m, provider)| ResolvedBf16 {
                provider,
                source: TacticMatch::Bucket,
                recorded_m,
            })
    }

    /// Rebuild the derived bucket index from the exact map.
    ///
    /// Every record is eligible. CUDA excludes its shape-pinned GeGLU
    /// epilogues here; a MUSA record is `bf16_gemm` on a KN layout and nothing
    /// else, so there is no record kind a bucket would misrepresent. Written
    /// out because the *absence* of an eligibility predicate should read as a
    /// decision rather than as a port that forgot one.
    fn rebuild_buckets(&mut self) {
        self.bf16_buckets.clear();
        for (&(m, n, k), &provider) in &self.bf16 {
            self.bf16_buckets
                .entry(BucketKey::of(m, n, k))
                .or_default()
                .insert(m, provider);
        }
    }

    /// Insert one exact winner, and re-derive the bucket index. Returns
    /// whether anything changed.
    fn upsert_bf16(&mut self, m: usize, n: usize, k: usize, provider: Bf16Provider) -> bool {
        if self.bf16.get(&(m, n, k)) == Some(&provider) {
            return false;
        }
        self.bf16.insert((m, n, k), provider);
        self.rebuild_buckets();
        true
    }
}

/// Where a hardware domain's database and diagnostic report live.
#[derive(Clone, Debug)]
pub struct TuningPaths {
    pub directory: PathBuf,
    pub tactics: PathBuf,
    pub report: PathBuf,
}

impl TuningPaths {
    /// `configs/tuning/musa/<device>-mp<version>/`.
    pub fn for_device(root: impl AsRef<Path>, caps: &MusaDeviceCaps) -> Self {
        let hardware = format!(
            "{}-mp{}",
            caps.device_name.to_lowercase().replace(' ', "-"),
            caps.mp
        );
        Self::for_directory(root.as_ref().join("musa").join(hardware))
    }

    /// The version-qualified directory, for a database that pins library
    /// versions. Same two-step lookup shape as the CUDA backend.
    pub fn for_device_toolkit(
        root: impl AsRef<Path>,
        caps: &MusaDeviceCaps,
        versions: &MusaLibraryVersions,
    ) -> Self {
        let hardware = format!(
            "{}-mp{}",
            caps.device_name.to_lowercase().replace(' ', "-"),
            caps.mp
        );
        Self::for_directory(root.as_ref().join("musa").join(hardware).join(format!(
            "musa{}-mublaslt{}",
            versions.musa, versions.mublas_lt
        )))
    }

    /// The files a directory holds, for a caller that already knows which one.
    ///
    /// [`Self::for_device`] derives that directory from the running hardware,
    /// which is what a default should do; this is for the caller who named one,
    /// and it keeps the two file names in one place either way.
    pub fn for_directory(directory: impl Into<PathBuf>) -> Self {
        let directory = directory.into();
        Self {
            tactics: directory.join("tactics.json"),
            report: directory.join("tuning_report.json"),
            directory,
        }
    }

    /// Prefer a database pinned to the toolkit versions, else the plain one.
    pub fn resolve_for_device(
        root: impl AsRef<Path>,
        caps: &MusaDeviceCaps,
        versions: &MusaLibraryVersions,
    ) -> Self {
        let pinned = Self::for_device_toolkit(root.as_ref(), caps, versions);
        if pinned.tactics.is_file() {
            return pinned;
        }
        Self::for_device(root, caps)
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

/// Runtime-owned tuning state.
/// One session's mode, tactic store and identity.
///
/// **A poisoned `store` lock is treated as unrecoverable, and the two sides say
/// so differently on purpose.** The write in `record_bf16` is multi-step —
/// insert the shape, drop that shape's old record, push the new one, bump the
/// generation — so a panic inside it leaves the store describing a state that
/// never existed, and handing that back would be worse than stopping. Readers
/// therefore panic (`store`, `bf16_provider`); the one writer that can report
/// returns `Err` instead, because its caller is a prepare path that can abandon
/// the preparation rather than take the process down with it.
pub struct TuningSession {
    mode: TuningMode,
    store: RwLock<TacticStore>,
    generation: AtomicU64,
    paths: Option<TuningPaths>,
}

impl TuningSession {
    pub fn new(mode: TuningMode, store: TacticStore, paths: Option<TuningPaths>) -> Self {
        Self {
            mode,
            store: RwLock::new(store),
            generation: AtomicU64::new(next_generation()),
            paths,
        }
    }

    pub fn inference(store: TacticStore) -> Self {
        Self::new(TuningMode::Inference, store, None)
    }

    pub fn mode(&self) -> TuningMode {
        self.mode
    }

    /// Bumped whenever the resolved tactics change.
    ///
    /// `pi05`'s prepare loop warms the model and waits for this to stop moving
    /// before capture. With no provider resolving tactics it is constant, so
    /// the loop settles on its first comparison — the same observable
    /// behaviour as a CUDA run whose database already holds every key.
    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    pub fn store(&self) -> TacticStore {
        self.store
            .read()
            .expect("MUSA tuning store lock is poisoned")
            .clone()
    }

    pub(crate) fn bf16_provider(&self, m: usize, n: usize, k: usize) -> Option<ResolvedBf16> {
        self.store
            .read()
            .expect("MUSA tuning store lock is poisoned")
            .lookup_bf16(m, n, k)
    }

    pub(crate) fn record_bf16(
        &self,
        ctx: &MusaContext,
        m: usize,
        n: usize,
        k: usize,
        provider: Bf16Provider,
    ) -> Result<()> {
        if !self.autotune_enabled() {
            return Err(musa_error("autotuning is suppressed"));
        }
        let records = {
            let mut store = self
                .store
                .write()
                .map_err(|_| musa_error("tuning store poisoned"))?;
            if !store.upsert_bf16(m, n, k, provider) {
                return Ok(());
            }
            store
                .records
                .retain(|r| !(r["op"] == "bf16_gemm" && r["m"] == m && r["n"] == n && r["k"] == k));
            store.records.push(
                serde_json::json!({"op":"bf16_gemm","m":m,"n":n,"k":k,"layout":"kn",
                "backend":provider.name()}),
            );
            self.generation.store(next_generation(), Ordering::Release);
            store.records.clone()
        };
        if let Some(paths) = &self.paths {
            std::fs::create_dir_all(&paths.directory).map_err(Error::Io)?;
            let contents = serde_json::to_vec_pretty(
                &serde_json::json!({"device_name":ctx.caps().device_name,
                "mp":ctx.caps().mp,"kernel_build_id":crate::KERNEL_BUILD_ID,
                "library_versions":ctx.library_versions(),"records":records}),
            )
            .map_err(|e| musa_error(e.to_string()))?;
            let temporary = paths
                .tactics
                .with_extension(format!("json.{}.tmp", std::process::id()));
            std::fs::write(&temporary, contents).map_err(Error::Io)?;
            std::fs::rename(temporary, &paths.tactics).map_err(Error::Io)?;
        }
        Ok(())
    }

    pub fn paths(&self) -> Option<&TuningPaths> {
        self.paths.as_ref()
    }

    /// Whether autotuning may run here and now.
    pub fn autotune_enabled(&self) -> bool {
        self.mode == TuningMode::AutoTune && !autotune_suppressed()
    }

}

thread_local! {
    static AUTOTUNE_SUPPRESSED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Run `operation` with autotuning suppressed, restoring the previous state.
///
/// PI0.5 wraps the warmup and the capture in this: measuring inside a capture
/// is illegal, and measuring on placeholder inputs would persist a tactic
/// chosen for the wrong workload.
pub fn without_autotune<T>(operation: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            AUTOTUNE_SUPPRESSED.with(|state| state.set(self.0));
        }
    }
    let _restore = Restore(AUTOTUNE_SUPPRESSED.with(|state| state.replace(true)));
    operation()
}

pub(crate) fn autotune_suppressed() -> bool {
    AUTOTUNE_SUPPRESSED.with(std::cell::Cell::get)
}

/// Install a mode and a tactic snapshot on the context.
pub(crate) fn configure(
    ctx: &MusaContext,
    mode: TuningMode,
    databases: &[TuningDb],
    paths: Option<TuningPaths>,
) -> Result<()> {
    let mut store = TacticStore::default();
    for database in databases {
        let resolved = database.build_store(ctx.caps(), ctx.library_versions());
        store.bf16.extend(resolved.bf16);
        store.records.extend(resolved.records);
    }
    // After the merge, not per database: extending a bucket index one database
    // at a time would make a collision between them order-dependent.
    store.rebuild_buckets();
    ctx.install_tuning(TuningSession::new(mode, store, paths))
        .map_err(musa_error)
}

#[cfg(test)]
mod cache_tests {
    use super::*;
    fn caps() -> MusaDeviceCaps {
        MusaDeviceCaps {
            device_name: "M1000".into(),
            compute_major: 2,
            compute_minor: 2,
            mp: 22,
            multiprocessor_count: 8,
            arch_family: crate::device_caps::MusaArchFamily::Mp22,
            hardware_warp_size: 128,
        }
    }
    /// A taken fingerprint: non-empty `binaries` is what "taken" means.
    fn versions() -> MusaLibraryVersions {
        MusaLibraryVersions {
            musa: 1,
            mublas_lt: 0,
            binaries: [("libmublas.so".to_string(), "fingerprint".to_string())].into(),
        }
    }
    fn database(records: Vec<serde_json::Value>) -> TuningDb {
        TuningDb {
            records,
            device_name: Some("M1000".into()),
            fingerprint: Fingerprint {
                kernel_build_id: Some(crate::KERNEL_BUILD_ID.to_string()),
                mp: Some(22),
                library_versions: serde_json::to_value(versions()).ok(),
            },
        }
    }

    fn record() -> serde_json::Value {
        serde_json::json!(
            {"op":"bf16_gemm","m":10,"n":8192,"k":1024,"layout":"kn","backend":"mudnn"}
        )
    }

    #[test]
    fn a_matching_database_selects_its_provider() {
        let store = database(vec![record()]).build_store(&caps(), &versions());
        assert_eq!(store.bf16.get(&(10, 8192, 1024)), Some(&Bf16Provider::Mudnn));
    }

    /// Every rejection is a fallback to muBLAS, never a failure to load: the
    /// database is a cache, and the shapes it named stay runnable without it.
    #[test]
    fn every_rejection_degrades_to_an_empty_store() {
        let caps = caps();

        let mut stale = database(vec![record()]);
        stale.fingerprint.kernel_build_id = Some("stale".to_string());
        assert!(stale.build_store(&caps, &versions()).bf16.is_empty());

        let mut wrong_device = database(vec![record()]);
        wrong_device.device_name = Some("S5000".into());
        assert!(wrong_device.build_store(&caps, &versions()).bf16.is_empty());

        let mut wrong_layout = database(vec![record()]);
        wrong_layout.records[0]["layout"] = serde_json::json!("nk");
        assert!(wrong_layout.build_store(&caps, &versions()).bf16.is_empty());

        let mut unknown_provider = database(vec![record()]);
        unknown_provider.records[0]["backend"] = serde_json::json!("cublas");
        assert!(unknown_provider.build_store(&caps, &versions()).bf16.is_empty());

        let duplicate = database(vec![record(), record()]);
        assert!(duplicate.build_store(&caps, &versions()).bf16.is_empty());

        // A database with no typed records is not a rejection at all, so it
        // must not be reported as one.
        assert!(database(vec![serde_json::json!({"op":"norm"})])
            .build_store(&caps, &versions())
            .bf16
            .is_empty());
    }

    /// A rejected database must not leave its records behind.
    ///
    /// `record_bf16` writes `store.records` back stamped with the *current*
    /// kernel build id, the running device name and the running fingerprint, so
    /// records carried out of a rejected file would reach the next process as
    /// measurements made for *its* build — a provenance they never had.
    #[test]
    fn a_rejected_database_carries_no_records_forward() {
        let caps = caps();

        let mut stale = database(vec![record()]);
        stale.fingerprint.kernel_build_id = Some("stale".to_string());
        let store = stale.build_store(&caps, &versions());
        assert!(store.bf16.is_empty());
        assert!(
            store.bf16_buckets.is_empty(),
            "a rejected database left a derived bucket index behind, which would serve \
             shapes its records were never measured for"
        );
        assert!(
            store.records.is_empty(),
            "a database rejected for its fingerprint was carried forward, to be re-stamped \
             with the current one on the first autotune"
        );

        let mut wrong_device = database(vec![record()]);
        wrong_device.device_name = Some("S5000".into());
        assert!(wrong_device.build_store(&caps, &versions()).records.is_empty());

        // Malformed and duplicate rows are the other half of the same hole:
        // left in `records` they are written back, stay on disk and keep
        // rejecting the file on every load.
        let mut wrong_layout = database(vec![record()]);
        wrong_layout.records[0]["layout"] = serde_json::json!("nk");
        assert!(wrong_layout.build_store(&caps, &versions()).records.is_empty());

        let duplicate = database(vec![record(), record()]);
        assert!(duplicate.build_store(&caps, &versions()).records.is_empty());
        assert!(duplicate
            .build_store(&caps, &versions())
            .bf16_buckets
            .is_empty());

        // The one path that is not a rejection still carries its records: a
        // database with no typed record at all is passed through untouched.
        assert_eq!(
            database(vec![serde_json::json!({"op": "norm"})])
                .build_store(&caps, &versions())
                .record_count(),
            1
        );
    }

    /// An untakeable fingerprint means unverifiable records, not a free pass.
    #[test]
    fn records_are_dropped_when_the_library_fingerprint_is_unavailable() {
        let caps = caps();

        let blind = MusaLibraryVersions::default();
        assert!(blind.binaries.is_empty());
        assert!(database(vec![record()])
            .build_store(&caps, &blind)
            .bf16
            .is_empty());

        // A database whose stored fingerprint differs from the running one is
        // dropped for the same reason — and this is what a vendor upgrade
        // produces, so it must not be able to stop a load.
        let other = MusaLibraryVersions {
            musa: 2,
            mublas_lt: 0,
            binaries: [("libmublas.so".to_string(), "other".to_string())].into(),
        };
        assert!(database(vec![record()])
            .build_store(&caps, &other)
            .bf16
            .is_empty());

        // A database that names no library fingerprint at all is not trusted
        // either: it cannot say which vendor build it was measured against.
        let mut bare = database(vec![record()]);
        bare.fingerprint.library_versions = None;
        assert!(bare.build_store(&caps, &versions()).bf16.is_empty());
    }
    /// The fingerprint describes the process's own mapped set, and two
    /// processes on one host do not map the same libraries — the benchmark that
    /// writes a database imports torch, the server that reads it does not. A
    /// database has to survive that boundary, or the whole mechanism only ever
    /// feeds the process that produced it.
    #[test]
    fn a_database_survives_a_process_that_maps_fewer_libraries() {
        let caps = caps();

        // Written by a process that had one more MUSA library mapped.
        let mut written = versions();
        written
            .binaries
            .insert("libmudnn.so".to_string(), "torch-only".to_string());
        let mut wider = database(vec![record()]);
        wider.fingerprint.library_versions = serde_json::to_value(written.clone()).ok();
        assert_eq!(wider.build_store(&caps, &versions()).bf16.len(), 1);

        // Read by a process that has one the database never saw.
        let mut reading = versions();
        reading
            .binaries
            .insert("libmudnn_extra.so".to_string(), "new".to_string());
        let mut narrower = database(vec![record()]);
        narrower.fingerprint.library_versions = serde_json::to_value(written).ok();
        assert_eq!(narrower.build_store(&caps, &reading).bf16.len(), 1);
    }

    /// A library both sides have, and that changed between them, is the case
    /// the fingerprint exists for.
    #[test]
    fn a_changed_library_still_rejects_the_database() {
        let mut reading = versions();
        reading
            .binaries
            .insert("libmublas.so".to_string(), "changed".to_string());
        assert!(database(vec![record()])
            .build_store(&caps(), &reading)
            .bf16
            .is_empty());
    }

    /// Every session gets its own generation, so a plan prepared against one
    /// cannot be reused against a replacement that has different tactics.
    #[test]
    fn every_session_has_its_own_generation() {
        let first = TuningSession::new(TuningMode::Inference, TacticStore::default(), None);
        let other = TuningSession::new(TuningMode::Inference, TacticStore::default(), None);
        assert_ne!(first.generation(), other.generation());
        assert_eq!(first.generation(), first.generation());
    }

    // --- bucket resolution -------------------------------------------------
    //
    // The defect these exist for: PI0.5's `bf16_gemm` M is `patches +
    // token_count` (`crates/apxinf-model/src/pi05/config.rs:244`), so a table
    // keyed on the exact M serves only the prompt lengths somebody measured.
    // An idle-card sweep read 48.1 ms at every token count except the fixture's
    // own ten, which read 39.4 ms, with no line saying why.

    fn gemm(m: usize, n: usize, k: usize, backend: &str) -> serde_json::Value {
        serde_json::json!({"op":"bf16_gemm","m":m,"n":n,"k":k,"layout":"kn","backend":backend})
    }

    fn store(records: Vec<serde_json::Value>) -> TacticStore {
        database(records).build_store(&caps(), &versions())
    }

    /// The rule is CUDA's, and both backends have to answer the same way.
    #[test]
    fn m_bucket_matches_the_cuda_rule() {
        // The first five are `m_bucket_is_deterministic` in
        // crates/apxinf-cuda/src/tuning/key.rs; the rest are this model's.
        assert_eq!(m_bucket(0), 1);
        assert_eq!(m_bucket(1), 1);
        assert_eq!(m_bucket(10), 16);
        assert_eq!(m_bucket(512), 512);
        assert_eq!(m_bucket(522), 1024);
        // 512 patch tokens + `max_token_len` text tokens, the whole prefix
        // range a PI0.5 request can reach.
        assert_eq!(m_bucket(513), 1024);
        assert_eq!(m_bucket(712), 1024);
        assert_eq!(m_bucket(1025), 2048);
    }

    /// The defect, stated as an invariant: the database measured one shape and
    /// answers for the whole prefix range.
    #[test]
    fn a_bucket_serves_prefix_lengths_the_database_never_measured() {
        let store = store(vec![gemm(522, 2048, 2048, "mudnn")]);
        for m in 513..=712 {
            let resolved = store.lookup_bf16(m, 2048, 2048).expect("prefix range");
            assert_eq!(resolved.provider, Bf16Provider::Mudnn, "m={m}");
            assert_eq!(resolved.recorded_m, 522, "m={m}");
            match resolved.source {
                TacticMatch::Exact => assert_eq!(m, 522, "only its own M is a measurement"),
                TacticMatch::Bucket => assert_ne!(m, 522, "m={m} is the measured shape"),
            }
        }
    }

    #[test]
    fn an_exact_record_wins_over_the_bucket_that_contains_it() {
        let store = store(vec![
            gemm(10, 8192, 1024, "mudnn"),
            gemm(12, 8192, 1024, "mublas"),
        ]);
        // Both records sit in bucket 16 with the request, and the exact one wins.
        assert_eq!(m_bucket(10), m_bucket(12));
        let exact = store.lookup_bf16(10, 8192, 1024).expect("exact");
        assert_eq!(exact.source, TacticMatch::Exact);
        assert_eq!(exact.provider, Bf16Provider::Mudnn);
        assert_eq!(exact.recorded_m, 10);
        // Precedence is the lookup's doing, not the index's: the bucket answer
        // for a shape the database never measured is a neighbour's record.
        assert_eq!(
            store.lookup_bf16_bucket(13, 8192, 1024),
            Some((12, Bf16Provider::Mublas))
        );
    }

    /// Nearest recorded M; a tie goes to the smaller — arbitrary, but written
    /// down, and the only ranking a timing-free record can support.
    #[test]
    fn the_nearest_recorded_m_wins_and_a_tie_goes_to_the_smaller() {
        let store = store(vec![
            gemm(600, 2048, 2048, "mudnn"),
            gemm(700, 2048, 2048, "mublas"),
        ]);
        assert_eq!(m_bucket(600), 1024);
        assert_eq!(m_bucket(700), 1024);
        for (query, recorded) in [(601, 600), (649, 600), (650, 600), (651, 700), (900, 700)] {
            assert_eq!(
                store.lookup_bf16_bucket(query, 2048, 2048),
                Some((recorded, if recorded == 600 { Bf16Provider::Mudnn } else { Bf16Provider::Mublas })),
                "m={query}"
            );
        }
    }

    #[test]
    fn a_shape_outside_every_bucket_is_a_miss() {
        let store = store(vec![gemm(522, 2048, 2048, "mudnn")]);
        assert_eq!(store.lookup_bf16(2049, 2048, 2048), None);
        assert_eq!(store.lookup_bf16(518, 999, 999), None);
        assert_eq!(TacticStore::default().lookup_bf16(518, 2048, 2048), None);
    }

    /// The derived index is a way of reading the records, not a second record
    /// list — which is what keeps the shipped-database invariant true after an
    /// autotune adds a shape.
    #[test]
    fn an_upsert_derives_a_bucket_entry_without_adding_a_record() {
        let mut store = store(vec![gemm(10, 8192, 1024, "mudnn")]);
        assert_eq!(
            store.lookup_bf16_bucket(11, 8192, 1024),
            Some((10, Bf16Provider::Mudnn))
        );
        assert!(store.upsert_bf16(12, 8192, 1024, Bf16Provider::Mudnn));
        assert_eq!(
            store.lookup_bf16_bucket(13, 8192, 1024),
            Some((12, Bf16Provider::Mudnn)),
            "an upsert must re-derive the bucket index"
        );
        // The record list is the session's to write — `upsert_bf16` moved only
        // the exact map and the index derived from it.
        assert_eq!(store.records.len(), 1);
        assert_eq!(store.bf16.len(), 2);
        assert_eq!(
            store
                .bf16_buckets
                .get(&BucketKey::of(11, 8192, 1024))
                .map(BTreeMap::len),
            Some(2),
            "both records belong to one bucket"
        );
        // A repeat is not a change: no re-derivation, no write-back.
        assert!(!store.upsert_bf16(12, 8192, 1024, Bf16Provider::Mudnn));
        assert_eq!(store.records.len(), 1);
    }

    /// The kernel seam needs to tell an exact hit from a derived one — an
    /// autotune run must keep measuring a shape it only has a neighbour for.
    #[test]
    fn the_session_reports_where_the_provider_came_from() {
        let session = TuningSession::inference(store(vec![gemm(522, 2048, 2048, "mudnn")]));
        let exact = session.bf16_provider(522, 2048, 2048).expect("exact");
        assert_eq!(exact.source, TacticMatch::Exact);
        assert_eq!(exact.recorded_m, 522);
        let bucket = session.bf16_provider(518, 2048, 2048).expect("bucket");
        assert_eq!(bucket.source, TacticMatch::Bucket);
        assert_eq!(bucket.recorded_m, 522);
        assert_eq!(bucket.provider, Bf16Provider::Mudnn);
        assert_eq!(session.bf16_provider(518, 999, 999), None);
    }

    /// Notes are process-global, so one shape is reported once per kind. The
    /// keys are a shape no other test uses.
    #[test]
    fn a_shape_is_reported_once_per_kind() {
        let shape = (7, 7, 7);
        assert!(note_once(NoteKind::Miss, shape));
        assert!(!note_once(NoteKind::Miss, shape));
        assert!(
            note_once(NoteKind::Bucket, shape),
            "a bucket hit is not the miss it replaces"
        );
        assert!(!note_once(NoteKind::Bucket, shape));
    }
}

/// The databases this repository ships, checked against the build that ships
/// them.
///
/// The failure this exists for is a quiet one, and it has already happened once:
/// `adapters/` is inside the kernel build id, so a commit that edits an adapter
/// *and* carries a database invalidates the database it is carrying. Nothing
/// fails — `stale_reason` drops the file, `warn_ignored` prints one line, and
/// GEMM falls back to muBLAS. The pass is simply 1.7× slower from then on, and
/// the document next to it still advertises the tuned number. `cargo test` is
/// where that has to be caught, because nothing at run time can tell "tuned for
/// this build" from "tuned for the last one".
#[cfg(test)]
mod shipped_tests {
    use super::*;
    use crate::device_caps::MusaArchFamily;

    /// This repository's tuning tree, from the crate that has to find it.
    fn tuning_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/tuning")
    }

    /// The generation this binary was compiled for, as the directory spells it.
    ///
    /// A build with no toolkit has none, and then the id assertion below has
    /// nothing to compare against: `KERNEL_BUILD_ID` was hashed from
    /// `"no-toolkit"` and no database could have been produced for it. The rest
    /// of the checks still run.
    fn built_mp() -> Option<u32> {
        crate::MUSA_ARCH.strip_prefix("mp_")?.parse().ok()
    }

    /// The capabilities a database lists for itself.
    ///
    /// Only `device_name` and `mp` take part in the directory name; the rest is
    /// filled so the struct is whole, and a wrong value there would fail the
    /// same comparison for a different reason (`stale_reason` checks `mp`, not
    /// the minor).
    fn caps_of(db: &TuningDb) -> std::result::Result<MusaDeviceCaps, String> {
        let mp = db
            .fingerprint
            .mp
            .ok_or_else(|| "the database carries no mp".to_string())? as u32;
        let device_name = db
            .device_name()
            .ok_or_else(|| "the database names no device".to_string())?
            .to_string();
        Ok(MusaDeviceCaps {
            device_name,
            compute_major: mp / 10,
            compute_minor: mp % 10,
            mp,
            multiprocessor_count: 0,
            arch_family: MusaArchFamily::from_mp(mp),
            hardware_warp_size: 128,
        })
    }

    /// The fingerprint a database records, as the running side would report it.
    ///
    /// The library half is a self-comparison by construction and that is the
    /// point: this test is not about whether the host's libraries match (they
    /// are this test binary's, and `cache_tests` covers that logic) but about
    /// whether the *file* is one a build of this tree can load at all. Anything
    /// the parser rejects — an unknown provider, a non-`kn` layout, a duplicate
    /// shape — makes `build_store` return an empty store, which is what the
    /// count below reads.
    fn recorded_versions(db: &TuningDb) -> std::result::Result<MusaLibraryVersions, String> {
        let stored = db
            .fingerprint
            .library_versions
            .as_ref()
            .ok_or_else(|| "the database carries no library fingerprint".to_string())?;
        let binaries = stored["binaries"]
            .as_object()
            .ok_or_else(|| "the database carries no binaries map".to_string())?;
        Ok(MusaLibraryVersions {
            musa: stored["musa"].as_i64().unwrap_or_default() as i32,
            mublas_lt: stored["mublas_lt"].as_u64().unwrap_or_default() as usize,
            binaries: binaries
                .iter()
                .map(|(name, value)| (name.clone(), value.as_str().unwrap_or_default().to_string()))
                .collect(),
        })
    }

    /// Validate one database against the build that is reading it.
    ///
    /// Returns the reason rather than panicking so the fixture tests below can
    /// assert rejection without catching a panic.
    fn check_shipped_database(root: &Path, tactics: &Path) -> std::result::Result<(), String> {
        let db = TuningDb::from_json_file(tactics)
            .map_err(|error| format!("{tactics:?} does not parse: {error}"))?;
        let caps = caps_of(&db)?;

        // The directory name and the file name are `TuningPaths`', from the
        // device the database itself names — not a copy of the rule here.
        // A database filed under a name nothing looks up is a database that
        // is never loaded, and the reason would be in this test's comment
        // rather than in anyone's head.
        let expected = TuningPaths::for_device(root, &caps).tactics;
        if expected.canonicalize().ok() != tactics.canonicalize().ok() {
            return Err(format!(
                "{tactics:?} is not where `TuningPaths::for_device` looks for this device; a \
                 database filed under a name nothing derives is one nothing loads"
            ));
        }

        // A database for another generation is another binary's
        // (`KERNEL_BUILD_ID` hashes the offload arch), so only this
        // generation's can carry this build's id. Checked before the load
        // below, which a stale id empties out and so would report as "no
        // record loaded" — true, and not the reason.
        if Some(caps.mp) == built_mp()
            && db.fingerprint.kernel_build_id.as_deref() != Some(crate::KERNEL_BUILD_ID)
        {
            return Err(format!(
                "{tactics:?} was tuned for another kernel build. Editing anything under \
                 `crates/apxinf-musa/adapters/` — including a comment — rewrites the id; \
                 regenerate with `scripts/bench_pi05_musa.py --autotune`"
            ));
        }

        // Counted rather than hardcoded, so a database covering more shapes
        // is fine and a database that lost a record is not.
        if db.record_count() == 0 {
            return Err(format!("{tactics:?} carries no records"));
        }
        let store = db.build_store(&caps, &recorded_versions(&db)?);
        if store.bf16.len() != db.record_count() {
            return Err(format!(
                "{} of {} records in {tactics:?} did not load",
                db.record_count() - store.bf16.len(),
                db.record_count()
            ));
        }
        // The bucket index is derived from those records, so it can neither
        // invent one nor lose one: a bucket entry must never reach the
        // record list, which is what `record_bf16` would write back.
        if store.records.len() != db.record_count() {
            return Err(format!(
                "a derived entry reached the record list in {tactics:?}"
            ));
        }
        if store.bf16_buckets.len() > store.bf16.len() {
            return Err(format!("{tactics:?} has more buckets than records"));
        }
        Ok(())
    }

    #[test]
    fn every_shipped_database_is_one_this_build_can_load() {
        let root = tuning_root();
        let musa = root.join("musa");
        // A clean checkout ships no MUSA database: `--autotune` writes one under
        // `configs/tuning/musa`, which is a local artifact. Validate whatever is
        // present, and say so when there is nothing to validate, rather than
        // failing a checkout that has never been tuned.
        //
        // The same test covers a future shipped database (S5000, for example):
        // a `tactics.json` that appears here is checked by the rules below.
        let mut checked = 0;
        if musa.is_dir() {
            for entry in std::fs::read_dir(&musa).expect("a readable MUSA tuning tree") {
                let directory = entry.expect("a directory entry").path();
                if !directory.is_dir() {
                    continue;
                }
                let tactics = directory.join("tactics.json");
                if !tactics.is_file() {
                    continue;
                }
                if let Err(reason) = check_shipped_database(&root, &tactics) {
                    panic!("{reason}");
                }
                checked += 1;
            }
        }
        if checked == 0 {
            eprintln!(
                "no MUSA tuning database is shipped under {musa:?}; \
                 `scripts/bench_pi05_musa.py --autotune` writes one locally"
            );
        }
    }

    // ── Fixtures ────────────────────────────────────────────────────────────

    /// A throwaway tuning root for one fixture database.
    fn fixture_root(label: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "apxinf-musa-tuning-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("a clock after the epoch")
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).expect("a temporary tuning root");
        root
    }

    /// Write one M1000/mp22 database where `TuningPaths` derives its name.
    fn write_fixture_database(root: &Path, kernel_build_id: &str) -> PathBuf {
        let directory = root.join("musa").join("m1000-mp22");
        std::fs::create_dir_all(&directory).expect("the fixture directory");
        let tactics = directory.join("tactics.json");
        let document = serde_json::json!({
            "device_name": "M1000",
            "mp": 22,
            "kernel_build_id": kernel_build_id,
            "library_versions": {
                "musa": 1,
                "mublas_lt": 0,
                "binaries": {"libmublas.so": "fixture"}
            },
            "records": [
                {"op": "bf16_gemm", "m": 10, "n": 1024, "k": 1024,
                 "layout": "kn", "backend": "mudnn"}
            ]
        });
        std::fs::write(
            &tactics,
            serde_json::to_vec_pretty(&document).expect("serialisable"),
        )
        .expect("the fixture database");
        tactics
    }

    #[test]
    fn a_database_this_build_can_load_passes_the_shipped_check() {
        let root = fixture_root("valid");
        let tactics = write_fixture_database(&root, crate::KERNEL_BUILD_ID);
        let result = check_shipped_database(&root, &tactics);
        std::fs::remove_dir_all(&root).ok();
        result.unwrap_or_else(|reason| panic!("{reason}"));
    }

    #[test]
    fn a_database_from_another_kernel_build_is_refused() {
        let root = fixture_root("stale");
        let tactics = write_fixture_database(&root, "kb-another-build");
        let result = check_shipped_database(&root, &tactics);
        std::fs::remove_dir_all(&root).ok();
        let reason = result.expect_err("a database for another build must not load");
        // A build for another MP generation skips the id comparison above; the
        // load then reports the records that did not arrive. Both name the
        // cause closely enough for a maintainer to regenerate the database.
        assert!(
            reason.contains("another kernel build") || reason.contains("did not load"),
            "the refusal does not name the kernel build: {reason}"
        );
    }
}
