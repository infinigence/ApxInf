//! Safe wrapper around MUSA device memory.
//!
//! A port of `apxinf-cuda/src/buffer.rs`. The allocation policies carry over
//! unchanged because the reasons for them are properties of asynchronous
//! launches, not of NVIDIA hardware: reused blocks must be ordered behind
//! whatever last read them, zeroing is explicit because the workspace hands
//! back dirty arena views, and poisoning is how a read-before-write shows up.
//!
//! **Paired with `apxinf-cuda/src/buffer.rs`.** The two are separate files, not
//! two names for one: this is a port, and a port is allowed to be better in
//! places — the copies differ on most of their lines today, and that is the
//! intended state. What they are not allowed to be is *silently* different,
//! which is how `APXINF_MUSA_ALLOC_CACHE=0` came to mean the opposite of
//! `APXINF_CUDA_ALLOC_CACHE=0`. So a fix here is usually a fix there, and the
//! question before finishing one is whether the other side has the same defect.
//! `tests/backend_parity.rs` locks down the half of this a test can see — the
//! environment knobs — and says why it is only that half.

use std::ffi::c_void;
use std::sync::Arc;

use apxinf_core::storage::{GpuStorageHandle, Storage};
use apxinf_core::{DType, Device, Shape, Tensor};

use crate::ffi;

#[derive(Clone, Copy, Debug)]
pub struct MusaDeviceAddress {
    ptr: *mut c_void,
    len: usize,
    device: usize,
}

impl MusaDeviceAddress {
    /// Nothing on the bridged path reads the pointer — the operators that took
    /// one are `not_ported`, and a native kernel replacing them is what reads
    /// it. `len` and `device` are what the current validators need.
    #[allow(dead_code)]
    pub(crate) fn ptr(self) -> *mut c_void {
        self.ptr
    }

    pub fn len(self) -> usize {
        self.len
    }

    pub fn device(self) -> usize {
        self.device
    }
}

struct MusaAllocation {
    ptr: *mut c_void,
    len: usize,
    device: usize,
    /// The stream this block belongs to, as the raw handle. Zero means the
    /// block was allocated without one and is never cached; see `cache_put`.
    stream: usize,
}

/// Device-memory free list, keyed by device, stream and exact byte size.
///
/// A model's per-layer intermediates repeat exactly from layer to layer and
/// from solver step to solver step, so an exact-size free list turns almost
/// all of the driver round trips into a pointer pop. The CUDA backend measured
/// 43,811 `cudaMalloc`/`cudaFree` pairs in one Qwen-Drive VQA inference,
/// against 7.9s of kernel time; **that measurement is CUDA's and has not been
/// repeated here** — the mechanism is carried over, its benefit on M1000 is
/// not yet evidenced.
///
/// The free list is keyed by the **stream** as well as the device and size,
/// and that is a correctness requirement rather than a refinement. A launch is
/// asynchronous, so a block dropped by one stream can still be in flight when
/// the free list hands it out; if the taker is a different stream nothing
/// orders the two and the second kernel can write the block while the first is
/// still reading it. Within one stream the launches are ordered by definition,
/// so reuse there needs nothing else. A block allocated without a stream is
/// never cached, which is why [`MusaBuffer::alloc`] does not consult this at
/// all and [`MusaBuffer::alloc_on`] does.
///
/// Opt-in through `APXINF_MUSA_ALLOC_CACHE` until it has been measured.
/// `APXINF_MUSA_ALLOC_CACHE_MB` caps retained bytes, default 4096, so a
/// long-running process cannot grow without bound; past the cap a block is
/// released to the driver as before.
///
/// The variable is read the way `apxinf-cuda` reads its own: **present, and not
/// `0`, means on**. The default differs on purpose — CUDA's cache is on and
/// this one is off, because the measurement behind it is CUDA's and has not
/// been repeated here — but `=0` has to mean "off" in both or the same string
/// says two things in one repository; presence alone would make
/// `APXINF_MUSA_ALLOC_CACHE=0` *enable* the cache.
struct AllocCache {
    /// Pointers held as `usize`; `*mut c_void` is not `Send`.
    blocks: std::collections::HashMap<(usize, usize, usize), Vec<usize>>,
    retained: usize,
    cap: usize,
}

fn alloc_cache() -> Option<&'static std::sync::Mutex<AllocCache>> {
    static CACHE: std::sync::OnceLock<Option<std::sync::Mutex<AllocCache>>> =
        std::sync::OnceLock::new();
    CACHE
        .get_or_init(|| {
            let requested = std::env::var("APXINF_MUSA_ALLOC_CACHE").ok()?;
            if requested == "0" {
                return None;
            }
            let cap = std::env::var("APXINF_MUSA_ALLOC_CACHE_MB")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(4096)
                .saturating_mul(1024 * 1024);
            Some(std::sync::Mutex::new(AllocCache {
                blocks: std::collections::HashMap::new(),
                retained: 0,
                cap,
            }))
        })
        .as_ref()
}

fn poison_allocations() -> bool {
    static POISON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *POISON.get_or_init(|| std::env::var_os("APXINF_MUSA_POISON_ALLOC").is_some())
}

/// Take a cached block of exactly `num_bytes` held for `stream` on `device`.
fn cache_take(num_bytes: usize, device: usize, stream: usize) -> Option<*mut c_void> {
    if stream == 0 {
        return None;
    }
    let mut cache = alloc_cache()?.lock().ok()?;
    let ptr = cache.blocks.get_mut(&(device, stream, num_bytes))?.pop()?;
    cache.retained = cache.retained.saturating_sub(num_bytes);
    Some(ptr as *mut c_void)
}

/// Retain a block for reuse on the stream that owned it. Returns false when the
/// caller must free it — including for every block that has no stream, which is
/// the case that cannot be made safe by ordering.
fn cache_put(ptr: *mut c_void, num_bytes: usize, device: usize, stream: usize) -> bool {
    if stream == 0 {
        return false;
    }
    let Some(cache) = alloc_cache() else {
        return false;
    };
    let Ok(mut cache) = cache.lock() else {
        return false;
    };
    if cache.retained.saturating_add(num_bytes) > cache.cap {
        return false;
    }
    cache.retained += num_bytes;
    cache
        .blocks
        .entry((device, stream, num_bytes))
        .or_default()
        .push(ptr as usize);
    true
}

/// Release every block the free list holds for a stream that is going away.
///
/// The free list exists so a block can go back to the stream that ordered it,
/// which is what makes reuse safe without an event. When that stream is
/// destroyed the ordering goes with it — and so does the entry's key's meaning,
/// because the driver recycles stream handles: a later stream given the same
/// handle would be handed blocks it was never ordered against. So the blocks
/// are freed here rather than kept for a handle that may be someone else's next
/// time.
///
/// Called from [`crate::MusaStream`]'s `Drop`, before the stream itself is
/// destroyed. Blocks are only ever cached once their last owner has dropped
/// ([`MusaAllocation::drop`] is the only caller of `cache_put`), so nothing
/// live is released here.
pub(crate) fn cache_drop_stream(stream: usize) {
    if stream == 0 {
        return;
    }
    let Some(cache) = alloc_cache() else {
        return;
    };
    let Ok(mut cache) = cache.lock() else {
        return;
    };
    let held: Vec<(usize, usize, usize)> = cache
        .blocks
        .keys()
        .filter(|(_, key_stream, _)| *key_stream == stream)
        .copied()
        .collect();
    for key in held {
        let Some(pointers) = cache.blocks.remove(&key) else {
            continue;
        };
        let (device, _, num_bytes) = key;
        for pointer in pointers {
            cache.retained = cache.retained.saturating_sub(num_bytes);
            unsafe {
                // The owning device, not whichever is current — the same
                // reason `Drop for MusaAllocation` selects it.
                let _ = ffi::musa::musaSetDevice(device as i32);
                let freed = ffi::musa::musaFree(pointer as *mut c_void);
                if freed != ffi::musa::MUSA_SUCCESS {
                    // Consumed, not left pending: see `Drop for MusaAllocation`.
                    let _ = ffi::musa::musaGetLastError();
                }
            }
        }
    }
}

// SAFETY: this allocation is released through the MUSA runtime and its raw
// device address may be shared across host threads.
unsafe impl Send for MusaAllocation {}
unsafe impl Sync for MusaAllocation {}

impl Drop for MusaAllocation {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        if cache_put(self.ptr, self.len, self.device, self.stream) {
            return;
        }
        unsafe {
            // Select the owning device first, the way `alloc_with_stream` and
            // `zero` do. `musaFree` releases memory on whichever device is
            // current, so on a host with more than one card dropping a buffer
            // for an idle device while another is selected fails and leaks --
            // silently, since this result has nowhere to go. The error is
            // dropped rather than reported because a failing free has no
            // recovery and a `Drop` has no caller to return to.
            //
            // Dropped *and cleared*, which is the part that is easy to miss: a
            // failed `musaFree` leaves the runtime's sticky error set, and every
            // entry point in `adapters/custom_kernels.mu` reports the sticky
            // error after its own launch. An error parked here would surface as
            // the next operator's failure, attributed to a kernel that had
            // nothing to do with it. Nothing else in this crate calls the
            // runtime without consuming the status -- see
            // `ffi::musa::musaGetErrorString` for the other half of that
            // convention.
            let _ = ffi::musa::musaSetDevice(self.device as i32);
            let freed = ffi::musa::musaFree(self.ptr);
            if freed != ffi::musa::MUSA_SUCCESS {
                let _ = ffi::musa::musaGetLastError();
            }
        }
    }
}

/// Owns a block of GPU memory. Automatically freed on drop.
#[derive(Clone)]
pub struct MusaBuffer {
    ptr: *mut c_void,
    len: usize,
    device: usize,
    owner: Arc<dyn std::any::Any + Send + Sync>,
}

// SAFETY: MUSA device pointers can be sent between threads.
unsafe impl Send for MusaBuffer {}
unsafe impl Sync for MusaBuffer {}

impl MusaBuffer {
    /// Allocate `num_bytes` of device memory, outside any stream.
    ///
    /// Blocks allocated this way never enter the reuse cache, because nothing
    /// orders their release against whatever is still reading them. Use
    /// [`MusaBuffer::alloc_on`] for the per-operator buffers a model churns
    /// through; that is where the ordering exists to make reuse safe.
    pub fn alloc(num_bytes: usize, device: usize) -> Result<Self, String> {
        Self::alloc_with_stream(num_bytes, device, 0)
    }

    /// Allocate `num_bytes` on `ctx`'s stream, eligible for stream-ordered
    /// reuse.
    pub fn alloc_on(ctx: &crate::MusaContext, num_bytes: usize) -> Result<Self, String> {
        Self::alloc_with_stream(num_bytes, ctx.device_id(), ctx.stream().handle() as usize)
    }

    fn alloc_with_stream(num_bytes: usize, device: usize, stream: usize) -> Result<Self, String> {
        unsafe {
            ffi::check_runtime(ffi::musa::musaSetDevice(device as i32))?;
        }
        let ptr: *mut c_void = match cache_take(num_bytes, device, stream) {
            Some(cached) => cached,
            None => {
                let mut fresh: *mut c_void = std::ptr::null_mut();
                unsafe {
                    ffi::check_runtime(ffi::musa::musaMalloc(&mut fresh, num_bytes))?;
                }
                fresh
            }
        };
        let owner: Arc<dyn std::any::Any + Send + Sync> = Arc::new(MusaAllocation {
            ptr,
            len: num_bytes,
            device,
            stream,
        });
        let buf = Self {
            ptr,
            len: num_bytes,
            device,
            owner,
        };
        // Poison every allocation under APXINF_MUSA_POISON_ALLOC. 0xFF is NaN
        // at every float width, so anything that reads a buffer before writing
        // it shows up in the result. Unlike poisoning operator outputs alone,
        // this covers the allocations a model makes directly, which is where a
        // read-before-write would otherwise stay hidden behind whatever the
        // previous owner of the block left there.
        //
        // After the owner exists, so a failing memset releases the block on the
        // way out instead of orphaning it: this is the one allocation path whose
        // whole purpose is to be run when something is being diagnosed, and a
        // leak here distorts the measurement it exists to serve.
        if poison_allocations() {
            unsafe {
                ffi::check_runtime(ffi::musa::musaMemset(ptr, 0xFF, num_bytes))?;
            }
        }
        Ok(buf)
    }

    /// Allocate and zero-fill, outside any stream.
    pub fn alloc_zeros(num_bytes: usize, device: usize) -> Result<Self, String> {
        let buf = Self::alloc(num_bytes, device)?;
        buf.zero()?;
        Ok(buf)
    }

    /// Zero this buffer in place, preserving its allocation and shared aliases.
    pub fn zero(&self) -> Result<(), String> {
        // SAFETY: owner keeps the allocation alive, and len bounds this buffer's
        // valid storage. Select its device before submitting the memset.
        unsafe {
            ffi::check_runtime(ffi::musa::musaSetDevice(self.device as i32))?;
            ffi::check_runtime(ffi::musa::musaMemset(self.ptr, 0, self.len))?;
        }
        Ok(())
    }

    /// Copy `num_bytes` from another device buffer on the given stream.
    ///
    /// A captured graph reads and writes fixed addresses, so a caller that
    /// replays one has to move its inputs into the buffers the capture baked
    /// and its results back out. Stream-ordered, so it sequences with the
    /// replay without a host sync.
    pub fn copy_from_device_async(
        &self,
        source: &MusaBuffer,
        num_bytes: usize,
        stream: &crate::MusaStream,
    ) -> Result<(), String> {
        if num_bytes > self.len || num_bytes > source.len {
            return Err(format!(
                "device copy of {num_bytes} bytes exceeds {} or {}",
                self.len, source.len
            ));
        }
        unsafe {
            ffi::check_runtime(ffi::musa::musaMemcpyAsync(
                self.ptr,
                source.ptr,
                num_bytes,
                ffi::musa::MUSA_MEMCPY_DEVICE_TO_DEVICE,
                stream.handle(),
            ))?;
        }
        Ok(())
    }

    /// Fill `num_bytes` of this buffer with `value` on the given stream.
    ///
    /// Workspace views are reused rather than freshly allocated, so a caller
    /// that needs cleared memory has to clear it explicitly.
    pub fn memset_async(
        &self,
        value: i32,
        num_bytes: usize,
        stream: &crate::MusaStream,
    ) -> Result<(), String> {
        if num_bytes > self.len {
            return Err(format!(
                "memset of {num_bytes} bytes exceeds the {} byte buffer",
                self.len
            ));
        }
        unsafe {
            ffi::check_runtime(ffi::musa::musaMemsetAsync(
                self.ptr,
                value,
                num_bytes,
                stream.handle(),
            ))?;
        }
        Ok(())
    }

    /// Allocate and fill every byte with `value`.
    ///
    /// Used to poison operator outputs under test: 0xFF is NaN at every float
    /// width, so a consumer that reads memory the producing kernel did not
    /// write shows up in the result instead of passing unnoticed.
    pub fn alloc_filled(num_bytes: usize, device: usize, value: i32) -> Result<Self, String> {
        let buf = Self::alloc(num_bytes, device)?;
        unsafe {
            ffi::check_runtime(ffi::musa::musaMemset(buf.ptr, value, num_bytes))?;
        }
        Ok(buf)
    }

    /// Allocate and fill on `ctx`'s stream.
    ///
    /// Stream-ordered like [`MusaBuffer::alloc_zeros_async`], and for the same
    /// reason: this path is used for every operator output that is poisoned to
    /// prove nothing reads it unwritten, and a synchronous fill would serialise
    /// the host behind the device on every one of them.
    pub fn alloc_filled_on(
        ctx: &crate::MusaContext,
        num_bytes: usize,
        value: i32,
    ) -> Result<Self, String> {
        let stream = ctx.stream();
        let buf = Self::alloc_with_stream(num_bytes, ctx.device_id(), stream.handle() as usize)?;
        unsafe {
            ffi::check_runtime(ffi::musa::musaMemsetAsync(
                buf.ptr,
                value,
                num_bytes,
                stream.handle(),
            ))?;
        }
        Ok(buf)
    }

    /// Allocate and zero-fill asynchronously on the given stream.
    ///
    /// The fill is submitted on the stream rather than issued as a synchronous
    /// `musaMemset`: a synchronous memset drains the whole device queue before
    /// it returns, so every operator output would serialise the host behind the
    /// device. On the eager path that is the *whole* pass — instrumented device
    /// time and wall clock added up exactly (1031 ms of kernels, 222 ms of
    /// host, 1253 ms measured), which is what full serialisation looks like.
    /// Stream-ordered, the fill is one more launch on the queue the consumer is
    /// already behind, and the host runs ahead as it does everywhere else.
    pub fn alloc_zeros_async(
        num_bytes: usize,
        device: usize,
        stream: &crate::MusaStream,
    ) -> Result<Self, String> {
        // The stream is right here, so this block is eligible for reuse on it.
        let buf = Self::alloc_with_stream(num_bytes, device, stream.handle() as usize)?;
        unsafe {
            ffi::check_runtime(ffi::musa::musaMemsetAsync(
                buf.ptr,
                0,
                num_bytes,
                stream.handle(),
            ))?;
        }
        Ok(buf)
    }

    /// Copy data from host to this device buffer.
    pub fn copy_from_host(&self, src: &[u8]) -> Result<(), String> {
        if src.len() > self.len {
            return Err(format!(
                "source is {} bytes, buffer is {}",
                src.len(),
                self.len
            ));
        }
        unsafe {
            ffi::check_runtime(ffi::musa::musaMemcpy(
                self.ptr,
                src.as_ptr() as *const c_void,
                src.len(),
                ffi::musa::MUSA_MEMCPY_HOST_TO_DEVICE,
            ))
        }
    }

    /// Copy data from this device buffer to host.
    pub fn copy_to_host(&self, dst: &mut [u8]) -> Result<(), String> {
        if dst.len() > self.len {
            return Err(format!(
                "destination is {} bytes, buffer is {}",
                dst.len(),
                self.len
            ));
        }
        unsafe {
            ffi::check_runtime(ffi::musa::musaMemcpy(
                dst.as_mut_ptr() as *mut c_void,
                self.ptr,
                dst.len(),
                ffi::musa::MUSA_MEMCPY_DEVICE_TO_HOST,
            ))
        }
    }

    /// Raw device pointer for crate-internal launch code.
    pub(crate) fn ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// Number of allocated bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn device(&self) -> usize {
        self.device
    }

    pub fn address(&self) -> MusaDeviceAddress {
        MusaDeviceAddress {
            ptr: self.ptr,
            len: self.len,
            device: self.device,
        }
    }

    /// Create a bounds-checked view which keeps the parent allocation alive.
    pub fn view(&self, byte_offset: usize, len: usize) -> Result<Self, String> {
        let end = byte_offset
            .checked_add(len)
            .ok_or_else(|| "MUSA buffer view range overflow".to_string())?;
        if end > self.len {
            return Err(format!(
                "MUSA buffer view [{byte_offset}..{end}] exceeds {} bytes",
                self.len
            ));
        }
        let ptr = unsafe { (self.ptr as *mut u8).add(byte_offset) as *mut c_void };
        Ok(Self {
            ptr,
            len,
            device: self.device,
            owner: Arc::clone(&self.owner),
        })
    }

    /// Borrow MUSA tensor storage as a buffer while retaining its allocation.
    pub fn from_tensor(tensor: &Tensor) -> Result<Self, String> {
        let device = match tensor.device() {
            Device::Musa(device) => device,
            device => return Err(format!("expected MUSA tensor, got {device:?}")),
        };
        let handle = tensor
            .storage()
            .as_gpu()
            .ok_or_else(|| "MUSA tensor has no GPU storage".to_string())?;
        let owner = handle
            ._prevent_leak
            .clone()
            .ok_or_else(|| "MUSA tensor storage has no owning allocation".to_string())?;
        Ok(Self {
            ptr: handle.ptr as *mut c_void,
            len: handle.len,
            device,
            owner,
        })
    }

    /// Turn an owned MUSA allocation into a Tensor while preserving ownership.
    pub(crate) fn into_tensor(self, shape: Shape, dtype: DType) -> Tensor {
        let device = Device::Musa(self.device);
        let handle = GpuStorageHandle {
            ptr: self.ptr as usize,
            len: self.len,
            _prevent_leak: Some(Arc::new(self)),
        };
        Tensor::from_raw_parts(shape, dtype, device, Storage::Gpu { device, handle })
    }

    /// Borrow this allocation as a tensor while retaining shared ownership.
    /// The caller must ensure the requested shape and dtype exactly describe
    /// the underlying bytes.
    pub fn as_tensor(&self, shape: Shape, dtype: DType) -> Result<Tensor, String> {
        let expected = shape
            .numel()
            .checked_mul(dtype.size_in_bytes())
            .ok_or_else(|| "MUSA tensor byte size overflow".to_string())?;
        if expected != self.len {
            return Err(format!(
                "MUSA tensor view needs {expected} bytes, buffer has {}",
                self.len
            ));
        }
        Ok(self.clone().into_tensor(shape, dtype))
    }
}
