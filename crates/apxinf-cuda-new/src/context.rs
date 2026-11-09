//! CUDA device and stream context.

use std::sync::Arc;

use apxinf_core::{DType, Error, Result as CoreResult, Shape, Tensor};

use crate::ffi;
use crate::stream::CudaStream;
use crate::CudaDeviceCaps;

/// Owns the stream and opaque native runtime used by GEMM executions.
/// CUDA runtime and cuBLAS library versions, as the legacy context exposed
/// them. Used by gr00t's fixed-shape M41 FP8 gate.
#[derive(Clone, Debug)]
pub struct CudaLibraryVersions {
    pub cuda: String,
    pub cublas: String,
}

impl CudaLibraryVersions {
    fn query(cublas: &crate::cublas::CublasHandle) -> Result<Self, String> {
        let mut cuda = 0;
        unsafe {
            crate::ffi::raw::cuda_runtime::check_cuda(
                crate::ffi::raw::cuda_runtime::cudaRuntimeGetVersion(&mut cuda),
            )?;
        }
        Ok(Self {
            cuda: format_cuda_runtime_version(cuda)?,
            cublas: format_cublas_version(cublas.version()?)?,
        })
    }
}

fn format_version(major: i32, minor: i32, patch: i32) -> String {
    if patch == 0 {
        format!("{major}.{minor}")
    } else {
        format!("{major}.{minor}.{patch}")
    }
}

fn format_cuda_runtime_version(version: i32) -> Result<String, String> {
    if version <= 0 {
        return Err(format!("CUDA runtime returned invalid version {version}"));
    }
    Ok(format_version(
        version / 1000,
        (version % 1000) / 10,
        version % 10,
    ))
}

fn format_cublas_version(version: i32) -> Result<String, String> {
    if version <= 0 {
        return Err(format!("cuBLAS returned invalid version {version}"));
    }
    let (major, minor, patch) = if version >= 10_000 {
        (version / 10_000, (version % 10_000) / 100, version % 100)
    } else {
        (version / 1000, (version % 1000) / 100, version % 100)
    };
    Ok(format_version(major, minor, patch))
}

pub struct CudaContext {
    device_id: usize,
    stream: Arc<CudaStream>,
    runtime: crate::ffi::abi::types::Runtime,
    caps: CudaDeviceCaps,
    cublas: crate::cublas::CublasHandle,
    library_versions: CudaLibraryVersions,
    recipe_cache_dir: Option<String>,
    allow_online_tune: bool,
}

impl CudaContext {
    /// Create a context for the specified CUDA device.
    pub fn new(device_id: usize) -> Result<Self, String> {
        Self::new_with_autotune(device_id, true)
    }

    /// Create a context with an immutable permission for online tuning.
    ///
    /// Tuning requires both this permission and the operator policy's
    /// `online_tune`. Disabling it still permits reuse of persisted recipes.
    pub fn new_with_autotune(device_id: usize, allow_online_tune: bool) -> Result<Self, String> {
        Self::new_with_recipe_options(device_id, allow_online_tune, None)
    }

    /// Create a context with model-load recipe policy.
    ///
    /// An explicit directory becomes the default for GEMM and Attention when
    /// an individual operation does not provide a narrower override.
    pub fn new_with_recipe_options(
        device_id: usize,
        allow_online_tune: bool,
        recipe_cache_dir: Option<String>,
    ) -> Result<Self, String> {
        let device = i32::try_from(device_id)
            .map_err(|_| format!("CUDA device id {device_id} does not fit in i32"))?;
        unsafe {
            ffi::check_cuda(ffi::cudaSetDevice(device))?;
        }

        let stream = Arc::new(CudaStream::new_on(device_id)?);
        let mut runtime = std::ptr::null_mut();
        unsafe {
            crate::ffi::abi::status::check(
                crate::ffi::abi::runtime::apxinf_runtime_create_with_autotune(
                    device,
                    u32::from(allow_online_tune),
                    &mut runtime,
                ),
            )
            .map_err(|error| error.to_string())?;
        }
        let initialized = (|| {
            let caps = CudaDeviceCaps::query(runtime)?;
            let cublas = crate::cublas::CublasHandle::new()?;
            cublas.set_stream(&stream)?;
            let library_versions = CudaLibraryVersions::query(&cublas)?;
            Ok::<_, String>((caps, cublas, library_versions))
        })();
        let (caps, cublas, library_versions) = match initialized {
            Ok(initialized) => initialized,
            Err(error) => {
                unsafe { crate::ffi::abi::runtime::apxinf_runtime_destroy(runtime) };
                return Err(error);
            }
        };

        Ok(Self {
            device_id,
            stream,
            runtime,
            caps,
            library_versions,
            cublas,
            recipe_cache_dir,
            allow_online_tune,
        })
    }

    pub fn device_id(&self) -> usize {
        self.device_id
    }

    /// Whether this context may online-tune operators. A measurement traversal
    /// that launches tuning kernels writes into its output arena, so a
    /// virtual (null-backed) arena is only safe when this is `false`.
    pub fn allow_online_tune(&self) -> bool {
        self.allow_online_tune
    }
    /// Default recipe directory, relative to the process working directory.
    /// An explicit operator `cache_dir` overrides this hardware/toolkit path.
    pub fn default_cache_dir(&self) -> &str {
        if let Some(directory) = self.recipe_cache_dir.as_deref() {
            return directory;
        }
        unsafe {
            std::ffi::CStr::from_ptr(crate::ffi::abi::runtime::apxinf_runtime_default_cache_dir(
                self.runtime,
            ))
        }
        .to_str()
        .expect("native default cache directory is ASCII")
    }
    pub(crate) fn recipe_cache_dir(&self) -> Option<&str> {
        self.recipe_cache_dir.as_deref()
    }
    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }
    /// Raw cuBLAS handle bound to the context stream, for the direct-launch
    /// GEMM helpers that predate the tuned GEMM operator.
    pub fn cublas(&self) -> &crate::cublas::CublasHandle {
        &self.cublas
    }
    /// CUDA runtime and cuBLAS library versions.
    pub fn library_versions(&self) -> &CudaLibraryVersions {
        &self.library_versions
    }
    pub fn caps(&self) -> &CudaDeviceCaps {
        &self.caps
    }
    pub(crate) fn shared_stream(&self) -> Arc<CudaStream> {
        Arc::clone(&self.stream)
    }
    pub(crate) fn runtime(&self) -> crate::ffi::abi::types::Runtime {
        self.runtime
    }

    pub fn synchronize(&self) -> Result<(), String> {
        self.stream.synchronize()
    }

    /// Free and total device memory, in bytes.
    ///
    /// The counterpart to [`crate::GraphWorkspace::capacity`] for diagnostics
    /// that have to account for the whole process: graph instantiation and the
    /// native GEMM/Attention libraries allocate outside our arenas, so arena
    /// capacities alone cannot show whether a change moved the footprint.
    ///
    /// On unified-memory parts (Tegra/Thor) these are system-memory figures,
    /// not discrete VRAM. A caller should report the delta across the
    /// allocation it is measuring, not the absolute numbers.
    pub fn memory_info(&self) -> Result<(usize, usize), String> {
        self.stream.with_current_device(|| {
            let mut free = 0usize;
            let mut total = 0usize;
            unsafe { ffi::check_cuda(ffi::cudaMemGetInfo(&mut free, &mut total))? };
            Ok((free, total))
        })
    }

    /// Memory-aware bounds for operator workspace during online tuning.
    ///
    /// Returns `(workspace_limit, allow_tune)`. `workspace_limit` is clamped to
    /// the free device memory minus a reserve; when that budget falls below a
    /// floor, online tuning is disabled entirely so autotune never tips a
    /// nearly-exhausted device over. On unified-memory parts (Tegra/Thor) the
    /// figure is a system-memory estimate, so the clamp is advisory there.
    pub fn memory_aware_tuning_budget(&self) -> (usize, bool) {
        const RESERVE: usize = 1 << 30;
        const FLOOR: usize = 16 << 20;
        const DEFAULT_LIMIT: usize = 256 << 20;
        let free = self
            .memory_info()
            .map(|(free, _)| free)
            .unwrap_or(usize::MAX);
        let budget = free.saturating_sub(RESERVE);
        if budget < FLOOR {
            (0, false)
        } else {
            (budget.min(DEFAULT_LIMIT), true)
        }
    }

    /// Allocate caller-owned output storage for an L3 semantic operation.
    ///
    /// Inside an [`crate::ExecutionSession`] this sub-allocates from the
    /// deterministic graph arena. Outside a session it creates an ordinary
    /// zero-initialized device allocation. Operator selection remains in L3;
    /// this method only provides stable runtime-owned storage for its output
    /// binding.
    pub fn allocate_output(&self, shape: Shape, dtype: DType) -> CoreResult<Tensor> {
        let bytes = shape
            .numel()
            .checked_mul(dtype.size_in_bytes())
            .ok_or_else(|| Error::Other("CUDA output size overflow".into()))?;
        let buffer = crate::workspace::output_buffer(self, bytes)?;
        Ok(buffer.into_tensor(shape, dtype))
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        unsafe { crate::ffi::abi::runtime::apxinf_runtime_destroy(self.runtime) }
    }
}
