//! Persistent CUDA graph workspace and deterministic sub-allocation.

use std::cell::Cell;

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
        // Diagnostic slack between arena allocations. The driver hands out
        // separate, non-adjacent blocks, so a kernel that writes slightly past
        // the end of its buffer lands in unused memory; the arena packs
        // allocations back to back, where the same overrun clobbers the next
        // one. If a slack makes a failure go away, that is the shape of it.
        let slack: usize = std::env::var("APXINF_CUDA_WORKSPACE_SLACK")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(0);
        let end = start
            .checked_add(bytes)
            .and_then(|value| value.checked_add(slack))
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

/// Whether execution is the synthetic eager traversal used only to prepare a
/// graph workspace. Autotuning must wait for a real request instead of using
/// these placeholder inputs.
pub(crate) fn is_preparing_workspace() -> bool {
    PREPARING.with(Cell::get)
}

/// Whether the workspace-less path may hand back an uncleared buffer.
///
/// The workspace path already makes the case: `GraphWorkspace::allocate`
/// returns a reused view into its arena and never clears it, so every consumer
/// of `output_buffer` already tolerates dirty memory. The fallback cleared
/// anyway, and it is not cheap -- one Qwen-Drive VQA inference on Orin issued
/// 33,545 memsets covering 41GB, 313ms of GPU time on top of 371ms of host API,
/// with single prefill outputs reaching 114MB.
///
/// Opt-in through `APXINF_CUDA_SKIP_OUTPUT_ZERO` until the model families that
/// have only ever taken the workspace-less path have been checked. Setting it
/// to `poison` fills the buffer with 0xFF instead, which is NaN for every float
/// width: that turns "nothing read uninitialized memory" from something a clean
/// run merely fails to disprove into something a run actively convicts, because
/// any such read propagates NaN into the output.
#[derive(Clone, Copy, PartialEq)]
enum OutputFill {
    Zero,
    Dirty,
    Poison,
}

fn output_fill() -> OutputFill {
    static FILL: std::sync::OnceLock<OutputFill> = std::sync::OnceLock::new();
    *FILL.get_or_init(
        || match std::env::var("APXINF_CUDA_SKIP_OUTPUT_ZERO").as_deref() {
            Err(_) => OutputFill::Zero,
            Ok("poison") => OutputFill::Poison,
            Ok(_) => OutputFill::Dirty,
        },
    )
}

pub(crate) fn output_buffer(ctx: &CudaContext, bytes: usize) -> Result<CudaBuffer> {
    ACTIVE_WORKSPACE.with(|active| {
        let workspace = active.get();
        if workspace.is_null() {
            match output_fill() {
                OutputFill::Zero => {
                    CudaBuffer::alloc_zeros(bytes, ctx.device_id()).map_err(Error::Cuda)
                }
                OutputFill::Dirty => CudaBuffer::alloc(bytes, ctx.device_id()).map_err(Error::Cuda),
                OutputFill::Poison => {
                    CudaBuffer::alloc_filled(bytes, ctx.device_id(), 0xFF).map_err(Error::Cuda)
                }
            }
        } else {
            unsafe { &*workspace }.allocate(bytes, ctx.device_id())
        }
    })
}

/// Like [`output_buffer`], but cleared.
///
/// A fresh driver allocation is not zero either, but the workspace hands back
/// a reused arena view, so a caller that needs zeros has to say so. Used for
/// the padded tails that a model's prep kernels leave untouched.
pub(crate) fn output_buffer_zeroed(ctx: &CudaContext, bytes: usize) -> Result<CudaBuffer> {
    ACTIVE_WORKSPACE.with(|active| {
        let workspace = active.get();
        if workspace.is_null() {
            CudaBuffer::alloc_zeros_async(bytes, ctx.device_id(), ctx.stream()).map_err(Error::Cuda)
        } else {
            let buffer = unsafe { &*workspace }.allocate(bytes, ctx.device_id())?;
            buffer
                .memset_async(0, bytes, ctx.stream())
                .map_err(Error::Cuda)?;
            Ok(buffer)
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
