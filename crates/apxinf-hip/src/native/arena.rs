//! Bump allocation of activation tensors from one device allocation.

use std::cell::Cell;
use std::sync::Arc;

use apxinf_core::storage::GpuStorageHandle;
use apxinf_core::{DType, Error, Result, Shape, Storage, Tensor};

use crate::runtime::{HipBuffer, HipContext};

/// Every tensor starts on a boundary this large, enough for any vector load.
const ALIGNMENT: usize = 256;

/// One device allocation handed out front to back as tensors.
///
/// An inference calls [`reset`](Self::reset) and then allocates every
/// activation in the same order as the previous run, so each tensor lands at
/// the same address every time — the property a captured HIP graph needs.
///
/// Tensors share ownership of the allocation, so it outlives the arena if a
/// tensor does. After `reset`, though, new tensors reuse the same bytes: a
/// caller must not read a tensor from an earlier run once the next one has
/// started allocating. Results that must survive go to the host or to memory
/// outside the arena.
pub struct HipArena {
    ctx: Arc<HipContext>,
    buffer: Arc<HipBuffer>,
    capacity: usize,
    offset: Cell<usize>,
    peak: Cell<usize>,
}

impl HipArena {
    pub fn new(ctx: &Arc<HipContext>, capacity: usize) -> Result<Self> {
        ctx.bind()?;
        let buffer = Arc::new(ctx.alloc(capacity.max(ALIGNMENT))?);
        Ok(Self {
            ctx: Arc::clone(ctx),
            buffer,
            capacity,
            offset: Cell::new(0),
            peak: Cell::new(0),
        })
    }

    /// An uninitialized tensor carved from the arena.
    pub fn tensor(&self, dims: impl Into<Vec<usize>>, dtype: DType) -> Result<Tensor> {
        let shape = Shape::new(dims.into());
        let bytes = shape
            .numel()
            .checked_mul(dtype.size_in_bytes())
            .ok_or(Error::Contract("apxinf-hip arena: tensor size overflows"))?;
        let start = self.offset.get().next_multiple_of(ALIGNMENT);
        let end = start
            .checked_add(bytes)
            .filter(|&end| end <= self.capacity)
            .ok_or_else(|| {
                Error::Other(format!(
                    "apxinf-hip arena: {bytes} bytes requested at offset {start}, capacity {}",
                    self.capacity
                ))
            })?;
        self.offset.set(end);
        self.peak.set(self.peak.get().max(end));
        let owner: Arc<dyn std::any::Any + Send + Sync> = self.buffer.clone();
        // SAFETY: `start..end` lies inside the arena's allocation, which the
        // handle keeps alive through `owner`.
        let handle = unsafe {
            GpuStorageHandle::from_raw_parts(self.buffer.ptr() as usize + start, bytes, Some(owner))
        };
        let device = self.ctx.device();
        // SAFETY: the view holds exactly `numel * size_of(dtype)` bytes.
        Ok(unsafe {
            Tensor::from_raw_parts_unchecked(shape, dtype, device, Storage::Gpu { device, handle })
        })
    }

    /// Start the next inference from the beginning of the arena.
    pub fn reset(&self) {
        self.offset.set(0);
    }

    pub fn used(&self) -> usize {
        self.offset.get()
    }

    /// The most bytes any run has used; size the next arena from this.
    pub fn peak(&self) -> usize {
        self.peak.get()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
}
