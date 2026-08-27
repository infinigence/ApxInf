//! Persistent CUDA graph workspace and deterministic sub-allocation.

use std::cell::Cell;
use std::sync::OnceLock;

use apxinf_core::{DType, Error, Result};

use crate::buffer::CudaBuffer;
use crate::context::CudaContext;
use crate::device_caps::CudaDeviceCaps;

const WORKSPACE_ALIGNMENT: usize = 256;

/// Persistent device arena used by a fixed-shape CUDA graph.
pub struct GraphWorkspace {
    storage: CudaBuffer,
    offset: Cell<usize>,
    fp8_emulation: Option<Fp8EmulationWorkspace>,
}

struct Fp8EmulationWorkspace {
    activation: CudaBuffer,
    weight: CudaBuffer,
}

impl GraphWorkspace {
    pub fn new(capacity_bytes: usize, device: usize) -> Result<Self> {
        if capacity_bytes == 0 {
            return Err(Error::Other(
                "static inference workspace capacity must be non-zero".into(),
            ));
        }
        Ok(Self {
            storage: CudaBuffer::alloc(capacity_bytes, device).map_err(Error::Cuda)?,
            offset: Cell::new(0),
            fp8_emulation: None,
        })
    }

    pub fn new_fp8(
        capacity_bytes: usize,
        max_activation_elements: usize,
        max_weight_elements: usize,
        device: usize,
    ) -> Result<Self> {
        let mut workspace = Self::new(capacity_bytes, device)?;
        let caps = CudaDeviceCaps::query(device).map_err(Error::Cuda)?;
        let native_fp8 =
            caps.compute_major > 8 || (caps.compute_major == 8 && caps.compute_minor >= 9);
        if !native_fp8 {
            if max_activation_elements == 0 || max_weight_elements == 0 {
                return Err(Error::Other(
                    "static inference FP8 emulation scratch capacities must be non-zero".into(),
                ));
            }
            let activation_bytes = max_activation_elements
                .checked_mul(DType::F16.size_in_bytes())
                .ok_or_else(|| {
                    Error::Other("static inference FP8 activation scratch overflow".into())
                })?;
            let weight_bytes = max_weight_elements
                .checked_mul(DType::F16.size_in_bytes())
                .ok_or_else(|| {
                    Error::Other("static inference FP8 weight scratch overflow".into())
                })?;
            workspace.fp8_emulation = Some(Fp8EmulationWorkspace {
                activation: CudaBuffer::alloc(activation_bytes, device).map_err(Error::Cuda)?,
                weight: CudaBuffer::alloc(weight_bytes, device).map_err(Error::Cuda)?,
            });
        }
        Ok(workspace)
    }

    pub fn capacity(&self) -> usize {
        self.storage.len()
    }

    pub fn used(&self) -> usize {
        self.offset.get()
    }

    fn reset(&self) {
        self.offset.set(0);
    }

    fn allocate(&self, bytes: usize, device: usize) -> Result<CudaBuffer> {
        if device != self.storage.device() {
            return Err(Error::Other(format!(
                "static inference workspace is on CUDA {}, but operation targets CUDA {device}",
                self.storage.device()
            )));
        }
        let start = self
            .offset
            .get()
            .checked_add(WORKSPACE_ALIGNMENT - 1)
            .ok_or_else(|| Error::Other("static inference workspace offset overflow".into()))?
            & !(WORKSPACE_ALIGNMENT - 1);
        let end = start
            .checked_add(bytes)
            .ok_or_else(|| Error::Other("static inference workspace size overflow".into()))?;
        if end > self.storage.len() {
            return Err(Error::Other(format!(
                "static inference workspace exhausted: need {end} bytes, capacity is {} bytes",
                self.storage.len()
            )));
        }
        self.offset.set(end);
        self.storage.view(start, bytes).map_err(Error::Cuda)
    }

    fn uses_fp8_emulation(&self) -> bool {
        self.fp8_emulation.is_some()
    }

    fn fp8_emulation_buffers(
        &self,
        activation_bytes: usize,
        weight_bytes: usize,
        device: usize,
    ) -> Result<(CudaBuffer, CudaBuffer)> {
        let scratch = self.fp8_emulation.as_ref().ok_or_else(|| {
            Error::Other(
                "static inference FP8 emulation requires GraphWorkspace::new_fp8 before graph capture".into(),
            )
        })?;
        if device != scratch.activation.device() {
            return Err(Error::Other(format!(
                "static inference FP8 emulation workspace is on CUDA {}, but operation targets CUDA {device}",
                scratch.activation.device()
            )));
        }
        if activation_bytes > scratch.activation.len() || weight_bytes > scratch.weight.len() {
            return Err(Error::Other(format!(
                "static inference FP8 emulation scratch exhausted: activation {activation_bytes}/{} bytes, weight {weight_bytes}/{} bytes",
                scratch.activation.len(),
                scratch.weight.len()
            )));
        }
        Ok((
            scratch
                .activation
                .view(0, activation_bytes)
                .map_err(Error::Cuda)?,
            scratch.weight.view(0, weight_bytes).map_err(Error::Cuda)?,
        ))
    }
}

thread_local! {
    static ACTIVE_WORKSPACE: Cell<*const GraphWorkspace> = const { Cell::new(std::ptr::null()) };
    static PREPARING: Cell<bool> = const { Cell::new(false) };
}

struct ActiveWorkspaceGuard {
    workspace: *const GraphWorkspace,
    preparing: bool,
}

impl Drop for ActiveWorkspaceGuard {
    fn drop(&mut self) {
        ACTIVE_WORKSPACE.with(|active| active.set(self.workspace));
        PREPARING.with(|preparing| preparing.set(self.preparing));
    }
}

fn with_workspace_phase<T>(
    workspace: &GraphWorkspace,
    prepare: bool,
    operation: impl FnOnce() -> Result<T>,
) -> Result<T> {
    workspace.reset();
    ACTIVE_WORKSPACE.with(|active| {
        if !active.get().is_null() {
            return Err(Error::Other(
                "nested static inference workspaces are not supported".into(),
            ));
        }
        let previous = active.replace(workspace as *const _);
        let previous_preparing = PREPARING.with(|preparing| preparing.replace(prepare));
        let _guard = ActiveWorkspaceGuard {
            workspace: previous,
            preparing: previous_preparing,
        };
        operation()
    })
}

pub(crate) fn prepare_with_workspace<T>(
    workspace: &GraphWorkspace,
    operation: impl FnOnce() -> Result<T>,
) -> Result<T> {
    with_workspace_phase(workspace, true, operation)
}

pub(crate) fn with_workspace<T>(
    workspace: &GraphWorkspace,
    operation: impl FnOnce() -> Result<T>,
) -> Result<T> {
    with_workspace_phase(workspace, false, operation)
}

/// Native execution resources may be installed only before capture or when an
/// operation is executed without a graph workspace.
pub(crate) fn may_prepare_native_resources() -> bool {
    PREPARING.with(Cell::get) || ACTIVE_WORKSPACE.with(|active| active.get().is_null())
}

fn stream_ordered_alloc_enabled() -> Result<bool> {
    static ENABLED: OnceLock<std::result::Result<bool, String>> = OnceLock::new();
    ENABLED
        .get_or_init(|| match std::env::var("APXINF_STREAM_ORDERED_ALLOC") {
            Err(std::env::VarError::NotPresent) => Ok(false),
            Ok(value) if value == "0" => Ok(false),
            Ok(value) if value == "1" => Ok(true),
            Ok(value) => Err(format!(
                "APXINF_STREAM_ORDERED_ALLOC must be 0 or 1, got `{value}`"
            )),
            Err(std::env::VarError::NotUnicode(_)) => {
                Err("APXINF_STREAM_ORDERED_ALLOC must be UTF-8".into())
            }
        })
        .clone()
        .map_err(Error::Other)
}

pub(crate) fn uninitialized_buffer(ctx: &CudaContext, bytes: usize) -> Result<CudaBuffer> {
    ACTIVE_WORKSPACE.with(|active| {
        let workspace = active.get();
        if workspace.is_null() {
            if stream_ordered_alloc_enabled()? {
                CudaBuffer::alloc_stream_ordered(bytes, ctx.device_id(), ctx.stream())
                    .map_err(Error::Cuda)
            } else {
                CudaBuffer::alloc(bytes, ctx.device_id()).map_err(Error::Cuda)
            }
        } else {
            unsafe { &*workspace }.allocate(bytes, ctx.device_id())
        }
    })
}

pub(crate) fn output_buffer(ctx: &CudaContext, bytes: usize) -> Result<CudaBuffer> {
    ACTIVE_WORKSPACE.with(|active| {
        let workspace = active.get();
        if workspace.is_null() {
            if stream_ordered_alloc_enabled()? {
                CudaBuffer::alloc_zeros_stream_ordered(bytes, ctx.device_id(), ctx.stream())
                    .map_err(Error::Cuda)
            } else {
                CudaBuffer::alloc_zeros(bytes, ctx.device_id()).map_err(Error::Cuda)
            }
        } else {
            unsafe { &*workspace }.allocate(bytes, ctx.device_id())
        }
    })
}

pub(crate) fn fp8_emulation_required(ctx: &CudaContext) -> Result<bool> {
    Ok(ACTIVE_WORKSPACE.with(|active| {
        let workspace = active.get();
        if workspace.is_null() {
            let caps = ctx.caps();
            !(caps.compute_major > 8 || (caps.compute_major == 8 && caps.compute_minor >= 9))
        } else {
            unsafe { &*workspace }.uses_fp8_emulation()
        }
    }))
}

pub(crate) fn fp8_emulation_buffers(
    ctx: &CudaContext,
    activation_bytes: usize,
    weight_bytes: usize,
) -> Result<(CudaBuffer, CudaBuffer)> {
    ACTIVE_WORKSPACE.with(|active| {
        let workspace = active.get();
        if workspace.is_null() {
            Ok((
                CudaBuffer::alloc(activation_bytes, ctx.device_id()).map_err(Error::Cuda)?,
                CudaBuffer::alloc(weight_bytes, ctx.device_id()).map_err(Error::Cuda)?,
            ))
        } else {
            unsafe { &*workspace }.fp8_emulation_buffers(
                activation_bytes,
                weight_bytes,
                ctx.device_id(),
            )
        }
    })
}
