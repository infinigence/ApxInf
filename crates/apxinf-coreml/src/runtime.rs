//! Compile, cache, load and run ML Program packages through Core ML.
use crate::program::Package;
use apxinf_core::{Error, Result};
use half::f16;
use sha2::{Digest, Sha256};
use std::ffi::{c_char, c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;

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

/// The cache directory size limit (`APXINF_COREML_CACHE_LIMIT`, human sizes such
/// as `8G`, `512M` or `0` for no limit). Defaults to 16 GiB.
fn cache_limit() -> u64 {
    const ENV: &str = "APXINF_COREML_CACHE_LIMIT";
    const DEFAULT: u64 = 16 << 30;
    match std::env::var(ENV) {
        Err(_) => DEFAULT,
        Ok(value) => parse_size(value.trim()).unwrap_or(DEFAULT),
    }
}

/// Parses `8G`, `512M`, `1024K` or a plain byte count into bytes.
fn parse_size(value: &str) -> Option<u64> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    let (digits, scale) = match value.as_bytes().last().copied() {
        Some(b'k') | Some(b'K') => (&value[..value.len() - 1], 1u64 << 10),
        Some(b'm') | Some(b'M') => (&value[..value.len() - 1], 1u64 << 20),
        Some(b'g') | Some(b'G') => (&value[..value.len() - 1], 1u64 << 30),
        Some(b't') | Some(b'T') => (&value[..value.len() - 1], 1u64 << 40),
        _ => (value, 1),
    };
    digits.trim().parse::<u64>().ok().map(|n| n * scale)
}

/// Recursive size of `path` in bytes, ignoring entries it cannot read.
fn dir_size(path: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(path) else {
        return 0;
    };
    entries
        .flatten()
        .map(|entry| {
            let path = entry.path();
            match entry.metadata() {
                Ok(meta) if meta.is_dir() => dir_size(&path),
                Ok(meta) => meta.len(),
                Err(_) => 0,
            }
        })
        .sum()
}

/// Evicts cached artifacts until `dir` fits in `limit` bytes.
///
/// Order: our own compiled `*.mlmodelc` programs oldest-first, then the ANE E5
/// bundle cache as a last resort (it is regenerated on the next load, so this
/// only costs one cold start). A `limit` of zero disables eviction.
fn prune_to_limit(dir: &Path, limit: u64) {
    if limit == 0 {
        return;
    }
    // Core ML keeps its ANE bundle cache under `~/Library/Caches/<process>/`,
    // not inside our own cache directory, and that is usually the bulk of the
    // space. Account for it here and let the last-resort step remove it.
    let mut bundles: Vec<PathBuf> = vec![dir.join("com.apple.e5rt.e5bundlecache")];
    if let Ok(exe) = std::env::current_exe() {
        if let Some(name) = exe.file_name() {
            if let Some(home) = std::env::var_os("HOME") {
                bundles.push(
                    PathBuf::from(home)
                        .join("Library/Caches")
                        .join(name)
                        .join("com.apple.e5rt.e5bundlecache"),
                );
            }
        }
    }
    bundles.retain(|path| path.is_dir());
    let mut total = dir_size(dir) + bundles.iter().map(|path| dir_size(path)).sum::<u64>();
    if total <= limit {
        return;
    }
    let mut programs: Vec<(std::time::SystemTime, PathBuf, u64)> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("mlmodelc") {
                continue;
            }
            let Ok(meta) = entry.metadata() else { continue };
            let modified = meta.modified().unwrap_or(std::time::UNIX_EPOCH);
            programs.push((modified, path, dir_size(&entry.path())));
        }
    }
    programs.sort_by_key(|(modified, _, _)| *modified);
    for (_, path, size) in programs {
        if total <= limit {
            return;
        }
        if std::fs::remove_dir_all(&path).is_ok() {
            total = total.saturating_sub(size);
        }
    }
    for bundle in bundles {
        if total <= limit {
            return;
        }
        let size = dir_size(&bundle);
        if std::fs::remove_dir_all(&bundle).is_ok() {
            total = total.saturating_sub(size);
        }
    }
}

/// A content-addressed directory of compiled programs. After [`seal`](Self::seal),
/// a program that is not yet compiled is an error, so no compilation happens
/// while requests run.
pub struct ProgramCache {
    dir: PathBuf,
    sealed: bool,
}

impl ProgramCache {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        let dir = dir.into();
        // Keep the cache directory bounded before anything is loaded: an
        // unbounded directory grows by a few GB for every composition it has
        // ever seen, and the ANE bundle cache inside it is the bulk.
        prune_to_limit(&dir, cache_limit());
        Self { dir, sealed: false }
    }

    /// Trims the cache directory to `APXINF_COREML_CACHE_LIMIT` now. Callers
    /// that finish a bulk load (which is when Core ML writes its E5 bundles)
    /// call this once the last program is in place.
    pub fn trim(&self) {
        prune_to_limit(&self.dir, cache_limit());
    }

    pub fn seal(&mut self) {
        self.sealed = true;
        // Loading runs the compile/write path and lets Core ML build its E5
        // bundle cache, so the directory can exceed the limit again after the
        // start-up prune. Trim once more now that nothing else will be written.
        prune_to_limit(&self.dir, cache_limit());
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
        let io = |e: std::io::Error| Error::Other(format!("Core ML cache: {e}"));
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
        if compiled.is_dir() {
            return Ok(compiled);
        }
        if self.sealed {
            return Err(Error::Contract(
                "Core ML programs are compiled before the cache is sealed",
            ));
        }
        std::fs::create_dir_all(&self.dir).map_err(io)?;
        let staging = tempfile::Builder::new()
            .prefix(&format!(".{key}-"))
            .tempdir_in(&self.dir)
            .map_err(io)?;
        let source = staging.path().join("program.mlpackage");
        let target = staging.path().join("program.mlmodelc");
        package.write(&source)?;
        compiler(&source, &target)?;
        // A concurrent loader may have renamed the same key first; keep either.
        if std::fs::rename(&target, &compiled).is_err() && !compiled.is_dir() {
            return Err(Error::Other(
                "cannot place the compiled Core ML program".into(),
            ));
        }
        Ok(compiled)
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
        cache.seal();
        assert_eq!(cache.compiled(&package).unwrap(), compiled);
        let missing = tempfile::tempdir().unwrap();
        let mut cache = ProgramCache::new(missing.path());
        cache.seal();
        assert!(cache
            .compiled_with(&package, |_, _| panic!("sealed miss compiled"))
            .is_err());
        assert_eq!(std::fs::read_dir(missing.path()).unwrap().count(), 0);
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

#[cfg(test)]
mod cache_limit_tests {
    use super::{dir_size, parse_size, prune_to_limit};

    #[test]
    fn parses_human_sizes() {
        assert_eq!(parse_size("1024"), Some(1024));
        assert_eq!(parse_size("8G"), Some(8 << 30));
        assert_eq!(parse_size("512M"), Some(512 << 20));
        assert_eq!(parse_size("1T"), Some(1 << 40));
        assert_eq!(parse_size(""), None);
        assert_eq!(parse_size("abc"), None);
    }

    #[test]
    fn evicts_oldest_programs_first() {
        let root = std::env::temp_dir().join(format!("apxinf-cache-limit-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        for index in 0..3 {
            let program = root.join(format!("{index:02}.mlmodelc"));
            std::fs::create_dir_all(&program).unwrap();
            std::fs::write(program.join("blob"), vec![0u8; 4096]).unwrap();
        }
        let before = dir_size(&root);
        assert!(before >= 3 * 4096);
        prune_to_limit(&root, 8192);
        let after = dir_size(&root);
        assert!(after <= 8192, "evicted size {after}");
        let _ = std::fs::remove_dir_all(&root);
    }
}
