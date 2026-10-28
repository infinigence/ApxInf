//! Overlapped L2 prefetch of read-only weights on a side stream.
//!
//! `l2_prefetch_fork` records an event on the main stream, makes the lane's
//! side stream wait on it, and launches a bulk L2 prefetch there; the main
//! stream continues without waiting. `l2_prefetch_join` makes the main stream
//! wait for the side stream. Under stream capture both calls become graph
//! edges, so the prefetch node runs in parallel with main-stream nodes until
//! the join. The prefetch is a cache hint and never changes data.

use apxinf_core::{Error, Result, Tensor};

use crate::buffer::CudaBuffer;
use crate::context::{CudaContext, PrefetchLane, PrefetchLanes};
use crate::ffi;
use crate::stream::CudaStream;

const CHUNK_BYTES: u32 = 64 * 1024;
const BLOCKS: u32 = 1;
const THREADS: u32 = 64;

const LANES: usize = 8;

fn check(status: ffi::cudaError_t) -> Result<()> {
    ffi::check_cuda(status).map_err(Error::Cuda)
}

fn new_lane() -> Result<PrefetchLane> {
    let stream = CudaStream::new().map_err(Error::Cuda)?;
    let mut fork: ffi::cudaEvent_t = std::ptr::null_mut();
    let mut join: ffi::cudaEvent_t = std::ptr::null_mut();
    // cudaEventDisableTiming = 0x02
    unsafe {
        check(ffi::cudaEventCreateWithFlags(&mut fork, 0x02))?;
        check(ffi::cudaEventCreateWithFlags(&mut join, 0x02))?;
    }
    Ok(PrefetchLane { stream, fork, join, pending: false })
}

fn join_locked(ctx: &CudaContext, lane: &mut PrefetchLane) -> Result<()> {
    if lane.pending {
        unsafe {
            check(ffi::cudaEventRecord(lane.join, lane.stream.handle()))?;
            check(ffi::cudaStreamWaitEvent(ctx.stream().handle(), lane.join, 0))?;
        }
        lane.pending = false;
    }
    Ok(())
}

/// Start an overlapped L2 prefetch of `tensor`'s device bytes on the next
/// round-robin lane. Only that lane's previous prefetch is joined first.
pub fn l2_prefetch_fork(ctx: &CudaContext, tensor: &Tensor) -> Result<()> {
    l2_prefetch_fork_many(ctx, &[tensor])
}

/// Prefetch several tensors in order on one lane (sequential on the side
/// stream, so earlier tensors get DRAM bandwidth first).
pub fn l2_prefetch_fork_many(ctx: &CudaContext, tensors: &[&Tensor]) -> Result<()> {
    let mut regions = Vec::with_capacity(tensors.len());
    for tensor in tensors {
        let bytes = tensor.size_in_bytes() as u64;
        if bytes == 0 || bytes % 16 != 0 {
            continue;
        }
        let buffer = CudaBuffer::from_tensor(tensor).map_err(Error::Cuda)?;
        regions.push((buffer.ptr() as *const std::ffi::c_void, bytes));
    }
    if regions.is_empty() {
        return Ok(());
    }
    let mut guard = ctx
        .prefetch_lane()
        .lock()
        .map_err(|_| Error::Other("CUDA prefetch lane lock is poisoned".into()))?;
    if guard.is_none() {
        let lanes = (0..LANES).map(|_| new_lane()).collect::<Result<Vec<_>>>()?;
        *guard = Some(PrefetchLanes { lanes, next: 0 });
    }
    let set = guard.as_mut().expect("lanes initialized");
    let index = set.next;
    set.next = (set.next + 1) % set.lanes.len();
    let lane = &mut set.lanes[index];
    join_locked(ctx, lane)?;
    unsafe {
        check(ffi::cudaEventRecord(lane.fork, ctx.stream().handle()))?;
        check(ffi::cudaStreamWaitEvent(lane.stream.handle(), lane.fork, 0))?;
        for (pointer, bytes) in regions {
            check(ffi::apxinf_l2_bulk_prefetch(
                pointer,
                bytes,
                CHUNK_BYTES,
                BLOCKS,
                THREADS,
                lane.stream.handle(),
            ))?;
        }
    }
    lane.pending = true;
    Ok(())
}

/// Make the main stream wait for every outstanding prefetch.
pub fn l2_prefetch_join(ctx: &CudaContext) -> Result<()> {
    let mut guard = ctx
        .prefetch_lane()
        .lock()
        .map_err(|_| Error::Other("CUDA prefetch lane lock is poisoned".into()))?;
    if let Some(set) = guard.as_mut() {
        for lane in set.lanes.iter_mut() {
            join_locked(ctx, lane)?;
        }
    }
    Ok(())
}
