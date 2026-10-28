//! Model-owned physical layout state, independent of execution IO addresses.

use std::sync::{Arc, Mutex, OnceLock, Weak};

use crate::CudaBuffer;
use apxinf_core::{Error, Result, Tensor};

// Weak owners keep registrations alive for all tensor/buffer aliases without
// keeping GPU allocations alive. Dropping a layout handle must not turn packed
// bytes back into an apparently canonical tensor. Purge only on allocation death.
struct Registration {
    owner: Weak<dyn std::any::Any + Send + Sync>,
    device: usize,
    start: usize,
    end: usize,
}

static ALLOCATIONS: OnceLock<Mutex<Vec<Registration>>> = OnceLock::new();

fn registrations() -> Result<std::sync::MutexGuard<'static, Vec<Registration>>> {
    let mut entries = ALLOCATIONS
        .get_or_init(Default::default)
        .lock()
        .map_err(|_| Error::Cuda("weight allocation registry poisoned".into()))?;
    entries.retain(|entry| entry.owner.strong_count() != 0);
    Ok(entries)
}

fn matches(entry: &Registration, buffer: &CudaBuffer) -> bool {
    let start = buffer.ptr() as usize;
    entry.device == buffer.device()
        && (entry
            .owner
            .ptr_eq(&Arc::downgrade(buffer.allocation_owner()))
            || (start < entry.end && entry.start < start.saturating_add(buffer.len())))
}

/// Check before cache lookup too: a cached canonical execution is not evidence
/// that its allocation is still canonical.
pub(crate) fn reject_canonical_alias(buffer: &CudaBuffer) -> Result<()> {
    if registrations()?.iter().any(|entry| matches(entry, buffer)) {
        return Err(Error::Cuda(
            "repackable allocation requires its shared weight layout state".into(),
        ));
    }
    Ok(())
}

/// One weight allocation whose layout may be committed during first preparation.
///
/// Every consumer of this allocation must attach this same state to its GEMM
/// arguments. The caller must not modify or read the tensor through canonical
/// tensor operations after preparation. Replacing a weight requires a new
/// allocation and state after retiring all executions/graphs that use the old one.
/// Clones share the layout decision; they do not allocate device memory.
#[derive(Clone)]
pub struct RepackedWeight {
    pub(crate) state: Arc<Mutex<WeightState>>,
}

pub(crate) struct WeightState {
    pub buffer: CudaBuffer,
    pub shape: Vec<usize>,
    pub dtype: apxinf_core::DType,
    pub layout: u32,
}

impl std::fmt::Debug for RepackedWeight {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RepackedWeight").finish_non_exhaustive()
    }
}

impl RepackedWeight {
    /// Adopt a canonical weight. No device allocation or packing occurs here.
    pub fn new(weight: &Tensor) -> Result<Self> {
        let buffer = CudaBuffer::from_tensor(weight).map_err(Error::Cuda)?;
        let mut entries = registrations()?;
        if entries.iter().any(|entry| matches(entry, &buffer)) {
            return Err(Error::Cuda(
                "allocation already has weight layout state; clone the existing state".into(),
            ));
        }
        let start = buffer.ptr() as usize;
        let end = start
            .checked_add(buffer.len())
            .ok_or_else(|| Error::Cuda("weight allocation range overflow".into()))?;
        entries.push(Registration {
            owner: Arc::downgrade(buffer.allocation_owner()),
            device: buffer.device(),
            start,
            end,
        });
        Ok(Self {
            state: Arc::new(Mutex::new(WeightState {
                buffer,
                shape: weight.shape().dims().to_vec(),
                dtype: weight.dtype(),
                layout: 0,
            })),
        })
    }
}
