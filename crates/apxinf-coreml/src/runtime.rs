//! Compile, cache, load and run ML Program packages through Core ML.
use crate::program::Package;
use apxinf_core::{Error, Result};
use half::f16;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr, CString};
use std::fs::{File, TryLockError};
use std::io::ErrorKind;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::Mutex;
use std::time::SystemTime;

mod ffi {
    use std::ffi::{c_char, c_void};
    extern "C" {
        pub fn apx_coreml_error() -> *const c_char;
        pub fn apx_coreml_compile(package: *const c_char, compiled: *const c_char) -> i32;
        pub fn apx_coreml_load(compiled: *const c_char, units: i32, out: *mut *mut c_void) -> i32;
        pub fn apx_coreml_free(model: *mut c_void);
        #[allow(clippy::too_many_arguments)]
        pub fn apx_coreml_predict(
            model: *mut c_void,
            input_names: *const *const c_char,
            inputs: *const *const u16,
            input_shapes: *const *const i64,
            input_ranks: *const usize,
            input_count: usize,
            output_names: *const *const c_char,
            outputs: *const *mut u16,
            output_shapes: *const *const i64,
            output_ranks: *const usize,
            output_count: usize,
        ) -> i32;
    }
}

pub(crate) fn check(status: i32) -> Result<()> {
    if status == 0 {
        return Ok(());
    }
    // SAFETY: the bridge returns a NUL-terminated thread-local string.
    let message = unsafe { CStr::from_ptr(ffi::apx_coreml_error()) };
    Err(Error::Other(message.to_string_lossy().into_owned()))
}

fn c_path(path: &Path) -> Result<CString> {
    CString::new(path.to_string_lossy().as_bytes())
        .map_err(|_| Error::Contract("Core ML path contains a NUL byte"))
}

fn checked_fp16_count(shape: &[usize]) -> Result<usize> {
    let count = shape.iter().try_fold(1usize, |count, &dimension| {
        if dimension == 0 || i64::try_from(dimension).is_err() {
            return Err(Error::Contract(
                "Core ML shape needs positive i64 dimensions",
            ));
        }
        count
            .checked_mul(dimension)
            .ok_or(Error::Contract("Core ML shape element count overflows"))
    })?;
    count
        .checked_mul(std::mem::size_of::<f16>())
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or(Error::Contract(
            "Core ML FP16 shape exceeds addressable storage",
        ))?;
    Ok(count)
}

/// Compiles the `.mlpackage` directory `package` into `compiled`, which must
/// not exist. [`ProgramCache`] is the normal entry; this compiles a package
/// written elsewhere, such as a reference.
pub fn compile(package: &Path, compiled: &Path) -> Result<()> {
    let (source, target) = (c_path(package)?, c_path(compiled)?);
    // SAFETY: both paths are NUL-terminated.
    check(unsafe { ffi::apx_coreml_compile(source.as_ptr(), target.as_ptr()) })
}

/// Where Core ML may run a loaded program.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComputeUnits {
    All = 0,
    CpuOnly = 1,
    CpuAndGpu = 2,
    CpuAndNeuralEngine = 3,
}

/// A loaded Core ML program. Not `Send`: predictions run on the owning thread.
pub struct Model {
    raw: NonNull<c_void>,
}

impl Drop for Model {
    fn drop(&mut self) {
        // SAFETY: `raw` is the retained MLModel from `apx_coreml_load`.
        unsafe { ffi::apx_coreml_free(self.raw.as_ptr()) }
    }
}

impl Model {
    pub fn load(compiled: &Path, units: ComputeUnits) -> Result<Self> {
        let path = c_path(compiled)?;
        let mut raw = std::ptr::null_mut();
        // SAFETY: `path` is NUL-terminated; `raw` receives a retained object.
        check(unsafe { ffi::apx_coreml_load(path.as_ptr(), units as i32, &mut raw) })?;
        Ok(Self {
            raw: NonNull::new(raw).ok_or(Error::Other("Core ML returned no model".into()))?,
        })
    }

    /// Runs one prediction. Inputs and outputs are contiguous FP16 arrays
    /// with the program's static shapes.
    pub fn predict(
        &self,
        inputs: &[(&str, &[f16], &[usize])],
        outputs: &[(&str, &[usize])],
    ) -> Result<Vec<Vec<f16>>> {
        let names = |n: &[&str]| -> Result<Vec<CString>> {
            n.iter()
                .map(|s| CString::new(*s).map_err(|_| Error::Contract("Core ML name contains NUL")))
                .collect()
        };
        let shapes = |s: &[&[usize]]| -> Vec<Vec<i64>> {
            s.iter()
                .map(|d| d.iter().map(|&x| x as i64).collect())
                .collect()
        };
        let input_counts: Vec<usize> = inputs
            .iter()
            .map(|input| checked_fp16_count(input.2))
            .collect::<Result<_>>()?;
        let output_counts: Vec<usize> = outputs
            .iter()
            .map(|output| checked_fp16_count(output.1))
            .collect::<Result<_>>()?;
        if inputs
            .iter()
            .zip(&input_counts)
            .any(|((_, data, _), &count)| data.len() != count)
        {
            return Err(Error::Contract(
                "Core ML input length differs from its shape",
            ));
        }
        let in_names = names(&inputs.iter().map(|i| i.0).collect::<Vec<_>>())?;
        let in_shapes = shapes(&inputs.iter().map(|i| i.2).collect::<Vec<_>>());
        let out_names = names(&outputs.iter().map(|o| o.0).collect::<Vec<_>>())?;
        let out_shapes = shapes(&outputs.iter().map(|o| o.1).collect::<Vec<_>>());
        let mut results: Vec<Vec<f16>> = output_counts
            .iter()
            .map(|&count| vec![f16::ZERO; count])
            .collect();
        let ptrs = |v: &[CString]| v.iter().map(|s| s.as_ptr()).collect::<Vec<*const c_char>>();
        let (in_name_ptrs, out_name_ptrs) = (ptrs(&in_names), ptrs(&out_names));
        let in_data: Vec<*const u16> = inputs.iter().map(|i| i.1.as_ptr().cast()).collect();
        let in_shape_ptrs: Vec<*const i64> = in_shapes.iter().map(|s| s.as_ptr()).collect();
        let in_ranks: Vec<usize> = in_shapes.iter().map(Vec::len).collect();
        let out_data: Vec<*mut u16> = results.iter_mut().map(|r| r.as_mut_ptr().cast()).collect();
        let out_shape_ptrs: Vec<*const i64> = out_shapes.iter().map(|s| s.as_ptr()).collect();
        let out_ranks: Vec<usize> = out_shapes.iter().map(Vec::len).collect();
        // SAFETY: every pointer array has the stated count; each input buffer
        // holds the product of its shape, checked above, and each output
        // buffer was allocated with the product of its shape. The bridge
        // reads inputs and writes outputs only during this call.
        check(unsafe {
            ffi::apx_coreml_predict(
                self.raw.as_ptr(),
                in_name_ptrs.as_ptr(),
                in_data.as_ptr(),
                in_shape_ptrs.as_ptr(),
                in_ranks.as_ptr(),
                inputs.len(),
                out_name_ptrs.as_ptr(),
                out_data.as_ptr(),
                out_shape_ptrs.as_ptr(),
                out_ranks.as_ptr(),
                outputs.len(),
            )
        })?;
        Ok(results)
    }
}

fn cache_error(error: std::io::Error) -> Error {
    Error::Other(format!("Core ML cache: {error}"))
}

/// Eviction moves an entry under this prefix before deleting it, so an
/// interrupted deletion never leaves a partial program under its own name.
const EVICTED_PREFIX: &str = ".evicted-";

/// Opens the published entry at `compiled`, holds a shared lock on it and
/// refreshes its last-use time. `None` when no entry is there, including one
/// evicted while this waited for the lock.
fn use_entry(compiled: &Path) -> std::io::Result<Option<File>> {
    loop {
        let entry = match File::open(compiled) {
            Ok(entry) => entry,
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error),
        };
        entry.lock_shared()?;
        let current = match std::fs::metadata(compiled) {
            Ok(current) => current,
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error),
        };
        let locked = entry.metadata()?;
        if (locked.dev(), locked.ino()) == (current.dev(), current.ino()) {
            entry.set_modified(SystemTime::now())?;
            return Ok(Some(entry));
        }
        // Evicted and published again while this waited; lock the new one.
    }
}

fn remove_tree(path: &Path) -> std::io::Result<()> {
    match std::fs::remove_dir_all(path) {
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(()),
        other => other,
    }
}

fn tree_bytes(path: &Path) -> std::io::Result<u64> {
    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.is_dir() {
        return Ok(metadata.len());
    }
    let mut bytes = 0;
    for child in std::fs::read_dir(path)? {
        bytes += tree_bytes(&child?.path())?;
    }
    Ok(bytes)
}

/// Evicts compiled programs in `dir`, least recently used first, until the
/// rest take at most `budget` bytes. Programs that a live cache holds are
/// skipped. Leftovers of an interrupted eviction are deleted first, so
/// running this again after a crash converges on the same state.
fn evict_to_budget(
    dir: &Path,
    budget: u64,
    remove: impl Fn(&Path) -> std::io::Result<()>,
) -> Result<()> {
    let children = match std::fs::read_dir(dir) {
        Ok(children) => children,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(cache_error(error)),
    };
    let mut programs = Vec::new();
    for child in children {
        let path = child.map_err(cache_error)?.path();
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        if name.starts_with(EVICTED_PREFIX) {
            remove(&path).map_err(cache_error)?;
            continue;
        }
        if path
            .extension()
            .is_none_or(|extension| extension != "mlmodelc")
        {
            continue;
        }
        let sized = std::fs::metadata(&path)
            .and_then(|metadata| Ok((metadata.modified()?, tree_bytes(&path)?)));
        match sized {
            Ok((used, bytes)) => programs.push((used, bytes, path)),
            Err(error) if error.kind() == ErrorKind::NotFound => {}
            Err(error) => return Err(cache_error(error)),
        }
    }
    let mut total: u64 = programs.iter().map(|(_, bytes, _)| bytes).sum();
    programs.sort_by_key(|(used, _, _)| *used);
    for (_, bytes, path) in programs {
        if total <= budget {
            break;
        }
        if evict(dir, &path, &remove).map_err(cache_error)? {
            total -= bytes;
        }
    }
    Ok(())
}

/// Moves `compiled` out of the cache and deletes it, unless a live cache
/// holds it. Returns whether it was evicted.
fn evict(
    dir: &Path,
    compiled: &Path,
    remove: impl Fn(&Path) -> std::io::Result<()>,
) -> std::io::Result<bool> {
    let entry = match File::open(compiled) {
        Ok(entry) => entry,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error),
    };
    match entry.try_lock() {
        Ok(()) => {}
        Err(TryLockError::WouldBlock) => return Ok(false),
        Err(TryLockError::Error(error)) => return Err(error),
    }
    let stamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let name = compiled.file_name().unwrap_or_default().to_string_lossy();
    let trash = dir.join(format!(
        "{EVICTED_PREFIX}{name}-{}-{stamp}",
        std::process::id()
    ));
    std::fs::rename(compiled, &trash)?;
    drop(entry);
    remove(&trash)?;
    Ok(true)
}

/// A content-addressed directory of compiled programs. After [`seal`](Self::seal),
/// a program that is not yet compiled is an error, so no compilation happens
/// while requests run.
///
/// Every program a cache resolves stays in use until the cache is dropped:
/// it holds a shared lock on the entry, which eviction from any process
/// respects. Each hit also refreshes the entry's last-use time.
pub struct ProgramCache {
    dir: PathBuf,
    byte_budget: Option<u64>,
    sealed: bool,
    in_use: Mutex<HashMap<String, File>>,
}

impl ProgramCache {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self {
            dir: dir.into(),
            byte_budget: None,
            sealed: false,
            in_use: Mutex::default(),
        }
    }

    /// Bounds the directory to `bytes` of compiled programs. Without a
    /// budget the directory grows with every program and macOS build it has
    /// seen.
    pub fn with_byte_budget(mut self, bytes: u64) -> Self {
        self.byte_budget = Some(bytes);
        self
    }

    /// Seals the cache. With a byte budget, it then evicts the least recently
    /// used programs until the rest fit, skipping any program a live cache in
    /// any process is using. That includes every program this cache resolved,
    /// so a working set larger than the budget is kept whole. The cache is
    /// sealed even if eviction fails.
    pub fn seal(&mut self) -> Result<()> {
        self.sealed = true;
        match self.byte_budget {
            Some(budget) => evict_to_budget(&self.dir, budget, remove_tree),
            None => Ok(()),
        }
    }

    /// The compiled directory for `package`, compiling it on a miss. The key
    /// is the program and weight bytes and the macOS build.
    pub fn compiled(&self, package: &Package) -> Result<PathBuf> {
        self.compiled_with(package, compile)
    }

    fn compiled_with(
        &self,
        package: &Package,
        compiler: impl FnOnce(&Path, &Path) -> Result<()>,
    ) -> Result<PathBuf> {
        let mut hash = Sha256::new();
        hash.update(package.model_bytes());
        hash.update(&package.weights);
        hash.update(
            std::fs::read("/System/Library/CoreServices/SystemVersion.plist").unwrap_or_default(),
        );
        let key: String = hash.finalize()[..16]
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        let compiled = self.dir.join(format!("{key}.mlmodelc"));
        if let Some(entry) = use_entry(&compiled).map_err(cache_error)? {
            self.hold(key, entry);
            return Ok(compiled);
        }
        if self.sealed {
            return Err(Error::Contract(
                "Core ML programs are compiled before the cache is sealed",
            ));
        }
        std::fs::create_dir_all(&self.dir).map_err(cache_error)?;
        let staging = tempfile::Builder::new()
            .prefix(&format!(".{key}-"))
            .tempdir_in(&self.dir)
            .map_err(cache_error)?;
        let source = staging.path().join("program.mlpackage");
        let target = staging.path().join("program.mlmodelc");
        package.write(&source)?;
        compiler(&source, &target)?;
        // Locked before it is published, so no eviction can take it first.
        let entry = File::open(&target).map_err(cache_error)?;
        entry.lock_shared().map_err(cache_error)?;
        let entry = match std::fs::rename(&target, &compiled) {
            Ok(()) => {
                entry.set_modified(SystemTime::now()).map_err(cache_error)?;
                entry
            }
            // A concurrent loader may have renamed the same key first; use theirs.
            Err(_) => use_entry(&compiled)
                .map_err(cache_error)?
                .ok_or_else(|| Error::Other("cannot place the compiled Core ML program".into()))?,
        };
        self.hold(key, entry);
        Ok(compiled)
    }

    fn hold(&self, key: String, entry: File) {
        self.in_use
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .entry(key)
            .or_insert(entry);
    }

    pub fn load(&self, package: &Package, units: ComputeUnits) -> Result<Model> {
        Model::load(&self.compiled(package)?, units)
    }

    pub fn load_with_placement(
        &self,
        package: &Package,
        units: ComputeUnits,
        timeout: std::time::Duration,
        admit: impl FnOnce(&crate::placement::ExpectedPlacement) -> Result<()>,
    ) -> Result<Model> {
        let timeout_ms = crate::placement::timeout_millis(timeout)?;
        let compiled = self
            .compiled(package)?
            .canonicalize()
            .map_err(|error| Error::Other(format!("Core ML compiled path: {error}")))?;
        let facts = crate::placement::inspect(&compiled, units, timeout_ms).map_err(|error| {
            Error::Other(format!(
                "Core ML placement inspection at {}: {error}",
                compiled.display()
            ))
        })?;
        let model = Model::load(&compiled, units)?;
        admit(&facts)?;
        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ProgramBuilder;
    use std::mem::ManuallyDrop;
    use std::sync::mpsc;
    use std::time::Duration;

    fn package() -> Package {
        let mut builder = ProgramBuilder::new();
        let input = builder.input("x", &[1]).unwrap();
        builder.finish(&[&input]).unwrap()
    }

    fn write_compiled(target: &Path) -> Result<()> {
        std::fs::create_dir(target).unwrap();
        std::fs::write(target.join("compiled"), b"same program").unwrap();
        Ok(())
    }

    #[test]
    fn concurrent_cache_misses_keep_in_progress_packages_and_published_program() {
        let root = tempfile::tempdir().unwrap();
        let cache = ProgramCache::new(root.path());
        let package = package();
        let (staged_tx, staged_rx) = mpsc::channel();
        let (continue_tx, continue_rx) = mpsc::channel();
        std::thread::scope(|scope| {
            let cache = &cache;
            let package = &package;
            let first = scope.spawn(move || {
                cache.compiled_with(package, |source, target| {
                    let marker = source.join("in-progress");
                    std::fs::write(&marker, b"first caller").unwrap();
                    staged_tx.send(source.to_owned()).unwrap();
                    continue_rx.recv_timeout(Duration::from_secs(10)).unwrap();
                    assert_eq!(std::fs::read(&marker).unwrap(), b"first caller");
                    write_compiled(target)
                })
            });
            let first_source = staged_rx.recv_timeout(Duration::from_secs(10)).unwrap();
            let compiled = cache
                .compiled_with(package, |_, target| {
                    assert!(first_source.join("in-progress").is_file());
                    write_compiled(target)
                })
                .unwrap();
            assert!(first_source.join("in-progress").is_file());
            continue_tx.send(()).unwrap();
            assert_eq!(first.join().unwrap().unwrap(), compiled);
            assert_eq!(
                std::fs::read(compiled.join("compiled")).unwrap(),
                b"same program"
            );
            assert!(!first_source.exists());
            assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 1);
        });
    }

    #[test]
    fn compiler_error_removes_only_its_staging_and_allows_retry() {
        let root = tempfile::tempdir().unwrap();
        let cache = ProgramCache::new(root.path());
        let package = package();
        let unrelated = root.path().join("other-in-progress");
        std::fs::create_dir(&unrelated).unwrap();
        assert!(cache
            .compiled_with(&package, |_, target| {
                write_compiled(target)?;
                Err(Error::Contract("compiler failed"))
            })
            .is_err());
        assert!(unrelated.is_dir());
        assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 1);
        let compiled = cache
            .compiled_with(&package, |_, target| write_compiled(target))
            .unwrap();
        assert!(compiled.join("compiled").is_file());
        assert!(unrelated.is_dir());
        assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 2);
    }

    #[test]
    fn sealed_cache_returns_hits_and_rejects_misses_without_compiling() {
        let root = tempfile::tempdir().unwrap();
        let mut cache = ProgramCache::new(root.path());
        let package = package();
        let compiled = cache
            .compiled_with(&package, |_, target| write_compiled(target))
            .unwrap();
        cache.seal().unwrap();
        assert_eq!(cache.compiled(&package).unwrap(), compiled);
        let missing = tempfile::tempdir().unwrap();
        let mut cache = ProgramCache::new(missing.path());
        cache.seal().unwrap();
        assert!(cache
            .compiled_with(&package, |_, _| panic!("sealed miss compiled"))
            .is_err());
        assert_eq!(std::fs::read_dir(missing.path()).unwrap().count(), 0);
    }

    fn program(width: usize) -> Package {
        let mut builder = ProgramBuilder::new();
        let input = builder.input("x", &[width]).unwrap();
        builder.finish(&[&input]).unwrap()
    }

    /// Publishes a 4 KiB stand-in program and dates its last use `age_secs` ago.
    fn publish(dir: &Path, width: usize, age_secs: u64) -> PathBuf {
        let compiled = ProgramCache::new(dir)
            .compiled_with(&program(width), |_, target| {
                std::fs::create_dir(target).unwrap();
                std::fs::write(target.join("weight.bin"), vec![0u8; 4096]).unwrap();
                std::fs::write(target.join("model.mil"), b"program").unwrap();
                Ok(())
            })
            .unwrap();
        File::open(&compiled)
            .unwrap()
            .set_modified(SystemTime::now() - Duration::from_secs(age_secs))
            .unwrap();
        compiled
    }

    fn sealed_with_budget(dir: &Path, bytes: u64) {
        ProgramCache::new(dir)
            .with_byte_budget(bytes)
            .seal()
            .unwrap();
    }

    fn listing(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .map(|child| child.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }

    #[test]
    fn seal_evicts_least_recently_used_programs_until_the_budget_fits() {
        let root = tempfile::tempdir().unwrap();
        let oldest = publish(root.path(), 1, 300);
        let middle = publish(root.path(), 2, 200);
        let newest = publish(root.path(), 3, 100);
        let budget = tree_bytes(&middle).unwrap() + tree_bytes(&newest).unwrap();
        let mut unbounded = ProgramCache::new(root.path());
        unbounded.seal().unwrap();
        assert!(oldest.is_dir(), "a cache without a budget evicted");
        sealed_with_budget(root.path(), budget);
        assert!(!oldest.exists(), "the least recently used program stayed");
        assert!(
            middle.is_dir() && newest.is_dir(),
            "evicted past the budget"
        );
        assert!(tree_bytes(root.path()).unwrap() <= budget);
    }

    #[test]
    fn a_hit_makes_a_program_most_recently_used() {
        let root = tempfile::tempdir().unwrap();
        let first = publish(root.path(), 1, 200);
        let second = publish(root.path(), 2, 100);
        let mut hit = ProgramCache::new(root.path());
        hit.seal().unwrap();
        assert_eq!(hit.compiled(&program(1)).unwrap(), first);
        drop(hit);
        sealed_with_budget(root.path(), tree_bytes(&first).unwrap());
        assert!(first.is_dir(), "the program just hit was evicted");
        assert!(!second.exists(), "the program not used since stayed");
    }

    #[test]
    fn programs_held_by_a_live_cache_are_never_evicted() {
        let root = tempfile::tempdir().unwrap();
        let held = publish(root.path(), 1, 200);
        let idle = publish(root.path(), 2, 100);
        let mut loading = ProgramCache::new(root.path());
        loading.seal().unwrap();
        assert_eq!(loading.compiled(&program(1)).unwrap(), held);
        File::open(&held)
            .unwrap()
            .set_modified(SystemTime::now() - Duration::from_secs(300))
            .unwrap();
        sealed_with_budget(root.path(), 0);
        assert!(held.is_dir(), "a program another cache holds was evicted");
        assert!(!idle.exists());
        drop(loading);
        sealed_with_budget(root.path(), 0);
        assert!(!held.exists(), "a released program was kept over budget");
    }

    #[test]
    fn a_lookup_that_waited_on_an_eviction_holds_what_is_published_after_it() {
        let root = tempfile::tempdir().unwrap();
        let compiled = publish(root.path(), 1, 100);
        let lookup = |evict: &dyn Fn()| {
            let evicting = File::open(&compiled).unwrap();
            evicting.lock().unwrap();
            std::thread::scope(|scope| {
                let lookup = scope.spawn(|| use_entry(&compiled).unwrap());
                std::thread::sleep(Duration::from_millis(200));
                evict();
                drop(evicting);
                lookup.join().unwrap()
            })
        };
        let trash = root.path().join(".evicted-test");
        let moved = lookup(&|| std::fs::rename(&compiled, &trash).unwrap());
        assert!(moved.is_none(), "a moved entry counted as a hit");
        std::fs::remove_dir_all(&trash).unwrap();
        publish(root.path(), 1, 100);
        let republished = lookup(&|| {
            std::fs::rename(&compiled, &trash).unwrap();
            std::fs::create_dir(&compiled).unwrap();
        })
        .expect("the program published again was missed");
        let held = republished.metadata().unwrap();
        let current = std::fs::metadata(&compiled).unwrap();
        assert_eq!(
            (held.dev(), held.ino()),
            (current.dev(), current.ino()),
            "the lock is on the evicted copy, not the published one"
        );
    }

    #[test]
    fn an_interrupted_eviction_leaves_no_partial_program_and_a_rerun_converges() {
        let root = tempfile::tempdir().unwrap();
        let evicted = publish(root.path(), 1, 200);
        let kept = publish(root.path(), 2, 100);
        let budget = tree_bytes(&kept).unwrap();
        let crash = |trash: &Path| {
            std::fs::remove_file(trash.join("weight.bin"))?;
            Err(std::io::Error::other("killed mid-deletion"))
        };
        assert!(evict_to_budget(root.path(), budget, crash).is_err());
        assert!(!evicted.exists(), "a partial program kept its name");
        for _ in 0..2 {
            evict_to_budget(root.path(), budget, remove_tree).unwrap();
            assert_eq!(listing(root.path()), listing_of(&kept));
        }
        let mut sealed = ProgramCache::new(root.path());
        sealed.seal().unwrap();
        assert_eq!(sealed.compiled(&program(2)).unwrap(), kept);
    }

    fn listing_of(compiled: &Path) -> Vec<String> {
        vec![compiled.file_name().unwrap().to_string_lossy().into_owned()]
    }

    #[test]
    fn invalid_input_extent_returns_error_before_native_prediction() {
        let model = ManuallyDrop::new(Model {
            raw: NonNull::dangling(),
        });
        let shape = [usize::MAX, 2];
        assert!(model.predict(&[("x", &[], &shape)], &[]).is_err());
    }

    #[test]
    fn fp16_shape_count_preserves_scalars_and_clef_tiles() -> Result<()> {
        assert_eq!(checked_fp16_count(&[])?, 1);
        for tile in [128, 512] {
            assert_eq!(checked_fp16_count(&[1, 4096, 1, tile])?, 4096 * tile);
            assert_eq!(checked_fp16_count(&[1, 1, 1, tile])?, tile);
        }
        Ok(())
    }

    #[test]
    fn invalid_fp16_shapes_are_rejected_before_allocation() {
        for shape in [
            vec![0],
            vec![usize::MAX],
            vec![1usize << 32, 1usize << 32],
            vec![i64::MAX as usize, 2],
            vec![i64::MAX as usize, 2, 2],
            vec![isize::MAX as usize / 2 + 1],
        ] {
            assert!(checked_fp16_count(&shape).is_err(), "{shape:?}");
        }
    }

    #[test]
    fn invalid_output_extent_returns_error_before_native_prediction() {
        let model = ManuallyDrop::new(Model {
            raw: NonNull::dangling(),
        });
        assert!(model.predict(&[], &[("y", &[usize::MAX, 2])]).is_err());
    }
}
