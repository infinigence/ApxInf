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

/// A content-addressed directory of compiled programs. After [`seal`](Self::seal),
/// a program that is not yet compiled is an error, so no compilation happens
/// while requests run.
pub struct ProgramCache {
    dir: PathBuf,
    sealed: bool,
}

impl ProgramCache {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self {
            dir: dir.into(),
            sealed: false,
        }
    }

    pub fn seal(&mut self) {
        self.sealed = true;
    }

    /// The compiled directory for `package`, compiling it on a miss. The key
    /// is the program and weight bytes and the macOS build.
    pub fn compiled(&self, package: &Package) -> Result<PathBuf> {
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
        let staging = self.dir.join(format!(".{key}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&staging);
        let result = (|| {
            package.write(&staging.join("program.mlpackage"))?;
            let target = staging.join("program.mlmodelc");
            compile(&staging.join("program.mlpackage"), &target)?;
            // A concurrent loader may have renamed the same key first; keep either.
            if std::fs::rename(&target, &compiled).is_err() && !compiled.is_dir() {
                return Err(Error::Other(
                    "cannot place the compiled Core ML program".into(),
                ));
            }
            Ok(())
        })();
        let _ = std::fs::remove_dir_all(&staging);
        result.map(|()| compiled)
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
        let facts = crate::placement::inspect(&compiled, units, timeout_ms)?;
        let model = Model::load(&compiled, units)?;
        admit(&facts)?;
        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::ManuallyDrop;

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
