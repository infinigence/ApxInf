//! Persistent CUDA graph workspace and deterministic sub-allocation.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use apxinf_core::{DType, Error, Result};

use crate::buffer::CudaBuffer;
use crate::context::CudaContext;
use crate::device_caps::CudaDeviceCaps;

const WORKSPACE_ALIGNMENT: usize = 256;

/// Persistent device arena used by a fixed-shape CUDA graph.
pub struct GraphWorkspace {
    storage: CudaBuffer,
    state: Arc<Mutex<WorkspaceState>>,
    fp8_emulation: Option<Fp8EmulationWorkspace>,
}

#[derive(Default)]
struct WorkspaceState {
    generation: u64,
    active_allocations: usize,
    live_requested_bytes: usize,
    peak_requested_bytes: usize,
    high_water_bytes: usize,
    free_blocks: BTreeMap<usize, usize>,
}

struct WorkspaceLease {
    state: Arc<Mutex<WorkspaceState>>,
    _storage: CudaBuffer,
    start: usize,
    extent: usize,
    requested_bytes: usize,
    generation: u64,
}

impl Drop for WorkspaceLease {
    fn drop(&mut self) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        if state.generation != self.generation || state.active_allocations == 0 {
            return;
        }
        state.active_allocations -= 1;
        state.live_requested_bytes = state
            .live_requested_bytes
            .saturating_sub(self.requested_bytes);
        if self.extent != 0 {
            state
                .free_blocks
                .entry(self.start)
                .or_insert(self.extent);
            merge_adjacent_free_blocks(&mut state.free_blocks);
        }
    }
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
            state: Arc::new(Mutex::new(WorkspaceState {
                free_blocks: BTreeMap::from([(0, capacity_bytes)]),
                ..WorkspaceState::default()
            })),
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
        self.lock_state()
            .map(|state| state.high_water_bytes)
            .unwrap_or(0)
    }

    pub fn peak_used(&self) -> usize {
        self.lock_state()
            .map(|state| state.peak_requested_bytes)
            .unwrap_or(0)
    }

    fn reset(&self) -> Result<()> {
        let capacity = self.storage.len();
        let mut state = self.lock_state()?;
        state.generation = state
            .generation
            .checked_add(1)
            .ok_or_else(|| Error::Other("workspace generation overflow".into()))?;
        state.active_allocations = 0;
        state.live_requested_bytes = 0;
        state.peak_requested_bytes = 0;
        state.high_water_bytes = 0;
        state.free_blocks.clear();
        state.free_blocks.insert(0, capacity);
        Ok(())
    }

    fn allocate(&self, bytes: usize, device: usize) -> Result<CudaBuffer> {
        if device != self.storage.device() {
            return Err(Error::Other(format!(
                "static inference workspace is on CUDA {}, but operation targets CUDA {device}",
                self.storage.device()
            )));
        }
        let extent = align_extent(bytes)?;
        let mut state = self.lock_state()?;
        let start = take_free_block(&mut state.free_blocks, extent).ok_or_else(|| {
            Error::Other(format!(
                "static inference workspace exhausted: need {} contiguous bytes, capacity is {} bytes",
                extent.max(bytes),
                self.storage.len()
            ))
        })?;
        state.active_allocations = state
            .active_allocations
            .checked_add(1)
            .ok_or_else(|| Error::Other("workspace active allocation overflow".into()))?;
        state.live_requested_bytes = state
            .live_requested_bytes
            .checked_add(bytes)
            .ok_or_else(|| Error::Other("workspace live byte overflow".into()))?;
        let allocation_end = start
            .checked_add(bytes)
            .ok_or_else(|| Error::Other("workspace high-water overflow".into()))?;
        state.high_water_bytes = state.high_water_bytes.max(allocation_end);
        state.peak_requested_bytes = state
            .peak_requested_bytes
            .max(state.live_requested_bytes);

        let lease = Arc::new(WorkspaceLease {
            state: Arc::clone(&self.state),
            _storage: self.storage.clone(),
            start,
            extent,
            requested_bytes: bytes,
            generation: state.generation,
        });
        self.storage
            .view(start, bytes)
            .map(|buffer| buffer.with_owner(lease))
            .map_err(Error::Cuda)
    }

    fn lock_state(&self) -> Result<std::sync::MutexGuard<'_, WorkspaceState>> {
        self.state
            .lock()
            .map_err(|_| Error::Other("static inference workspace state mutex is poisoned".into()))
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
    static EAGER: Cell<bool> = const { Cell::new(false) };
}

struct ActiveWorkspaceGuard {
    workspace: *const GraphWorkspace,
    preparing: bool,
    eager: bool,
}

impl Drop for ActiveWorkspaceGuard {
    fn drop(&mut self) {
        ACTIVE_WORKSPACE.with(|active| active.set(self.workspace));
        PREPARING.with(|preparing| preparing.set(self.preparing));
        EAGER.with(|eager| eager.set(self.eager));
    }
}

fn with_workspace_phase<T>(
    workspace: &GraphWorkspace,
    prepare: bool,
    eager: bool,
    operation: impl FnOnce() -> Result<T>,
) -> Result<T> {
    workspace.reset()?;
    ACTIVE_WORKSPACE.with(|active| {
        if !active.get().is_null() {
            return Err(Error::Other(
                "nested static inference workspaces are not supported".into(),
            ));
        }
        let previous = active.replace(workspace as *const _);
        let previous_preparing = PREPARING.with(|preparing| preparing.replace(prepare));
        let previous_eager = EAGER.with(|current| current.replace(eager));
        let _guard = ActiveWorkspaceGuard {
            workspace: previous,
            preparing: previous_preparing,
            eager: previous_eager,
        };
        operation()
    })
}

fn align_extent(bytes: usize) -> Result<usize> {
    if bytes == 0 {
        return Ok(0);
    }
    bytes
        .checked_add(WORKSPACE_ALIGNMENT - 1)
        .map(|end| end & !(WORKSPACE_ALIGNMENT - 1))
        .ok_or_else(|| Error::Other("static inference workspace extent overflow".into()))
}

fn take_free_block(free_blocks: &mut BTreeMap<usize, usize>, extent: usize) -> Option<usize> {
    let mut selected = None;
    for (start, block_extent) in free_blocks.iter() {
        if *block_extent >= extent
            && selected
                .is_none_or(|(_, selected_extent)| *block_extent < selected_extent)
        {
            selected = Some((*start, *block_extent));
        }
    }
    let (start, block_extent) = selected?;
    free_blocks.remove(&start);
    if block_extent > extent {
        free_blocks.insert(start + extent, block_extent - extent);
    }
    Some(start)
}

fn merge_adjacent_free_blocks(free_blocks: &mut BTreeMap<usize, usize>) {
    let mut merged = BTreeMap::new();
    for (&start, &extent) in free_blocks.iter() {
        if let Some((&last_start, &last_extent)) = merged.last_key_value() {
            if last_start + last_extent == start {
                *merged.get_mut(&last_start).expect("last key exists") =
                    last_extent + extent;
                continue;
            }
        }
        merged.insert(start, extent);
    }
    *free_blocks = merged;
}

pub(crate) fn prepare_with_workspace<T>(
    workspace: &GraphWorkspace,
    operation: impl FnOnce() -> Result<T>,
) -> Result<T> {
    with_workspace_phase(workspace, true, false, operation)
}

pub(crate) fn with_workspace<T>(
    workspace: &GraphWorkspace,
    operation: impl FnOnce() -> Result<T>,
) -> Result<T> {
    with_workspace_phase(workspace, false, false, operation)
}

/// Bind a workspace for an eager (non-captured) traversal.
///
/// Identical to [`with_workspace`] except that native execution resources stay
/// installable, so GEMM plan resolution and autotuning behave exactly as they
/// do without a workspace. Capture must keep using [`with_workspace`]: a
/// workspace-bound traversal cannot tell whether it is recording, and CUDA
/// forbids allocating or re-planning inside a capture.
pub(crate) fn with_workspace_eager<T>(
    workspace: &GraphWorkspace,
    operation: impl FnOnce() -> Result<T>,
) -> Result<T> {
    with_workspace_phase(workspace, false, true, operation)
}

/// Native execution resources may be installed only before capture or when an
/// operation is executed without a graph workspace.
pub(crate) fn may_prepare_native_resources() -> bool {
    PREPARING.with(Cell::get)
        || EAGER.with(Cell::get)
        || ACTIVE_WORKSPACE.with(|active| active.get().is_null())
}

/// Whether execution is the synthetic eager traversal used only to prepare a
/// graph workspace. Autotuning must wait for a real request instead of using
/// these placeholder inputs.
pub(crate) fn is_preparing_workspace() -> bool {
    PREPARING.with(Cell::get)
}

/// Allocate operator output storage; non-arena allocations start at zero.
pub(crate) fn output_buffer(ctx: &CudaContext, bytes: usize) -> Result<CudaBuffer> {
    ACTIVE_WORKSPACE.with(|active| {
        let workspace = active.get();
        if workspace.is_null() {
            // On the context's stream, so these are the blocks the reuse
            // cache may hand back: an operator output is written by a kernel
            // on that stream and read by the next one on the same stream, so
            // a recycled block is ordered behind whatever last used it. This
            // is the path the cache was measured on -- 43,811 malloc/free
            // pairs and 13.6 s of host time in one VQA inference.
            CudaBuffer::alloc_zeros_on(ctx, bytes).map_err(Error::Cuda)
        } else {
            unsafe { &*workspace }.allocate(bytes, ctx.device_id())
        }
    })
}

/// Allocate operator output storage without clearing it.
///
/// GEMM writes every output element with `beta = 0`, so clearing a fresh
/// allocation is avoidable host- and device-side work.
pub(crate) fn output_buffer_uninitialized(
    ctx: &CudaContext,
    bytes: usize,
) -> Result<CudaBuffer> {
    ACTIVE_WORKSPACE.with(|active| {
        let workspace = active.get();
        if workspace.is_null() {
            CudaBuffer::alloc_on(ctx, bytes).map_err(Error::Cuda)
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

/// As [`output_buffer_zeroed`], but only the trailing rows of each group.
///
/// The prep kernels that consume these buffers write every row a token maps to
/// and leave the padding that rounds the sequence up to a chunk multiple. Zeroing
/// the whole allocation to establish that padding costs the ratio between them,
/// which at the shipped prefill is 3392 rows written to clear 7.
pub(crate) fn output_buffer_tail_zeroed(
    ctx: &CudaContext,
    groups: usize,
    group_bytes: usize,
    used_bytes: usize,
) -> Result<CudaBuffer> {
    let bytes = groups
        .checked_mul(group_bytes)
        .ok_or_else(|| Error::Other("scratch extent overflow".into()))?;
    let tail = group_bytes
        .checked_sub(used_bytes)
        .ok_or_else(|| Error::Other("scratch used bytes exceed group extent".into()))?;
    ACTIVE_WORKSPACE.with(|active| {
        let workspace = active.get();
        let buffer = if workspace.is_null() {
            // A fresh driver allocation still has to establish the tail, but it
            // is not a reused arena, so the payload rows need no clearing.
            CudaBuffer::alloc(bytes, ctx.device_id()).map_err(Error::Cuda)?
        } else {
            unsafe { &*workspace }.allocate(bytes, ctx.device_id())?
        };
        if tail > 0 && groups > 0 {
            buffer
                .memset_2d_async(0, used_bytes, group_bytes, tail, groups, ctx.stream())
                .map_err(Error::Cuda)?;
        }
        Ok(buffer)
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
