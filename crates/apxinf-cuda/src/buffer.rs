//! Safe wrapper around CUDA device memory.

use std::ffi::c_void;
use std::sync::Arc;

use apxinf_core::storage::{GpuStorageHandle, Storage};
use apxinf_core::{DType, Device, Shape, Tensor};

use crate::ffi;

#[derive(Clone, Copy, Debug)]
pub struct CudaDeviceAddress {
    ptr: *mut c_void,
    len: usize,
    device: usize,
}

impl CudaDeviceAddress {
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

struct CudaAllocation {
    ptr: *mut c_void,
    len: usize,
    device: usize,
}

/// Device-memory free list, keyed by device and exact byte size.
///
/// One Qwen-Drive VQA inference issued 43,811 cudaMalloc/cudaFree pairs and
/// spent 13.6s of host time inside them, against 7.9s of total kernel time:
/// every per-layer intermediate took a fresh block and returned it. The shapes
/// repeat exactly from layer to layer and from decode step to decode step, so
/// an exact-size free list turns nearly all of that into a pointer pop.
///
/// Opt-in through `APXINF_CUDA_ALLOC_CACHE` until it has been measured across
/// the other model families. `APXINF_CUDA_ALLOC_CACHE_MB` caps retained bytes,
/// default 4096, so a long-running process cannot grow without bound; past the
/// cap a block is released to the driver as before.
struct AllocCache {
    /// Pointers held as `usize`; `*mut c_void` is not `Send`.
    blocks: std::collections::HashMap<(usize, usize), Vec<usize>>,
    retained: usize,
    cap: usize,
}

fn alloc_cache() -> Option<&'static std::sync::Mutex<AllocCache>> {
    static CACHE: std::sync::OnceLock<Option<std::sync::Mutex<AllocCache>>> =
        std::sync::OnceLock::new();
    CACHE
        .get_or_init(|| {
            std::env::var_os("APXINF_CUDA_ALLOC_CACHE")?;
            let cap = std::env::var("APXINF_CUDA_ALLOC_CACHE_MB")
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
    *POISON.get_or_init(|| std::env::var_os("APXINF_CUDA_POISON_ALLOC").is_some())
}

/// Take a cached block of exactly `num_bytes` on `device`, if one is held.
fn cache_take(num_bytes: usize, device: usize) -> Option<*mut c_void> {
    let mut cache = alloc_cache()?.lock().ok()?;
    let ptr = cache.blocks.get_mut(&(device, num_bytes))?.pop()?;
    cache.retained = cache.retained.saturating_sub(num_bytes);
    Some(ptr as *mut c_void)
}

/// Retain a block for reuse. Returns false when the caller must free it.
fn cache_put(ptr: *mut c_void, num_bytes: usize, device: usize) -> bool {
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
        .entry((device, num_bytes))
        .or_default()
        .push(ptr as usize);
    true
}

// SAFETY: this allocation is released through the CUDA runtime and its raw
// device address may be shared across host threads.
unsafe impl Send for CudaAllocation {}
unsafe impl Sync for CudaAllocation {}

impl Drop for CudaAllocation {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        if cache_put(self.ptr, self.len, self.device) {
            return;
        }
        unsafe {
            let _ = ffi::cudaFree(self.ptr);
        }
    }
}

/// Owns a block of GPU memory. Automatically freed on drop.
#[derive(Clone)]
pub struct CudaBuffer {
    ptr: *mut c_void,
    len: usize,
    device: usize,
    owner: Arc<dyn std::any::Any + Send + Sync>,
}

// SAFETY: CUDA device pointers can be sent between threads.
unsafe impl Send for CudaBuffer {}
unsafe impl Sync for CudaBuffer {}

impl CudaBuffer {
    /// Allocate `num_bytes` of device memory.
    pub fn alloc(num_bytes: usize, device: usize) -> Result<Self, String> {
        unsafe {
            ffi::check_cuda(ffi::cudaSetDevice(device as i32))?;
        }
        let ptr: *mut c_void = match cache_take(num_bytes, device) {
            Some(cached) => cached,
            None => {
                let mut fresh: *mut c_void = std::ptr::null_mut();
                unsafe {
                    ffi::check_cuda(ffi::cudaMalloc(&mut fresh, num_bytes))?;
                }
                fresh
            }
        };
        // Poison every allocation under APXINF_CUDA_POISON_ALLOC. 0xFF is NaN at
        // every float width, so anything that reads a buffer before writing it
        // shows up in the result. Unlike poisoning operator outputs alone, this
        // covers the allocations a model makes directly, which is where a
        // read-before-write would otherwise stay hidden behind whatever the
        // previous owner of the block left there.
        if poison_allocations() {
            unsafe {
                ffi::check_cuda(ffi::cudaMemset(ptr, 0xFF, num_bytes))?;
            }
        }
        let owner: Arc<dyn std::any::Any + Send + Sync> = Arc::new(CudaAllocation {
            ptr,
            len: num_bytes,
            device,
        });
        Ok(Self {
            ptr,
            len: num_bytes,
            device,
            owner,
        })
    }

    /// Allocate and zero-fill.
    pub fn alloc_zeros(num_bytes: usize, device: usize) -> Result<Self, String> {
        let buf = Self::alloc(num_bytes, device)?;
        unsafe {
            ffi::check_cuda(ffi::cudaMemset(buf.ptr, 0, num_bytes))?;
        }
        Ok(buf)
    }

    /// Copy `num_bytes` from another device buffer on the given stream.
    ///
    /// A captured graph reads and writes fixed addresses, so a caller that
    /// replays one has to move its inputs into the buffers the capture baked
    /// and its results back out. Stream-ordered, so it sequences with the
    /// replay without a host sync.
    pub fn copy_from_device_async(
        &self,
        source: &CudaBuffer,
        num_bytes: usize,
        stream: &crate::CudaStream,
    ) -> Result<(), String> {
        if num_bytes > self.len || num_bytes > source.len {
            return Err(format!(
                "device copy of {num_bytes} bytes exceeds {} or {}",
                self.len, source.len
            ));
        }
        unsafe {
            ffi::check_cuda(ffi::cudaMemcpyAsync(
                self.ptr,
                source.ptr,
                num_bytes,
                ffi::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
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
        stream: &crate::CudaStream,
    ) -> Result<(), String> {
        if num_bytes > self.len {
            return Err(format!(
                "memset of {num_bytes} bytes exceeds the {} byte buffer",
                self.len
            ));
        }
        unsafe {
            ffi::check_cuda(ffi::cudaMemsetAsync(
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
            ffi::check_cuda(ffi::cudaMemset(buf.ptr, value, num_bytes))?;
        }
        Ok(buf)
    }

    /// Allocate and zero-fill asynchronously on the given stream.
    ///
    /// Takes the stream by reference: `CudaStream` owns its stream and is
    /// neither `Copy` nor `Clone`, so the by-value form this replaced could
    /// not be called with the `&CudaStream` a context hands out, and had no
    /// callers.
    pub fn alloc_zeros_async(
        num_bytes: usize,
        device: usize,
        stream: &crate::CudaStream,
    ) -> Result<Self, String> {
        let buf = Self::alloc(num_bytes, device)?;
        unsafe {
            ffi::check_cuda(ffi::cudaMemsetAsync(buf.ptr, 0, num_bytes, stream.handle()))?;
        }
        Ok(buf)
    }

    /// Copy data from host to this device buffer.
    pub fn copy_from_host(&self, src: &[u8]) -> Result<(), String> {
        assert!(src.len() <= self.len, "source exceeds buffer size");
        unsafe {
            ffi::check_cuda(ffi::cudaMemcpy(
                self.ptr,
                src.as_ptr() as *const c_void,
                src.len(),
                ffi::cudaMemcpyKind::cudaMemcpyHostToDevice,
            ))
        }
    }

    /// Copy data from this device buffer to host.
    pub fn copy_to_host(&self, dst: &mut [u8]) -> Result<(), String> {
        assert!(dst.len() <= self.len, "destination exceeds buffer size");
        unsafe {
            ffi::check_cuda(ffi::cudaMemcpy(
                dst.as_mut_ptr() as *mut c_void,
                self.ptr,
                dst.len(),
                ffi::cudaMemcpyKind::cudaMemcpyDeviceToHost,
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

    pub fn address(&self) -> CudaDeviceAddress {
        CudaDeviceAddress {
            ptr: self.ptr,
            len: self.len,
            device: self.device,
        }
    }

    /// Create a bounds-checked view which keeps the parent allocation alive.
    pub fn view(&self, byte_offset: usize, len: usize) -> Result<Self, String> {
        let end = byte_offset
            .checked_add(len)
            .ok_or_else(|| "CUDA buffer view range overflow".to_string())?;
        if end > self.len {
            return Err(format!(
                "CUDA buffer view [{byte_offset}..{end}] exceeds {} bytes",
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

    /// Borrow CUDA tensor storage as a buffer while retaining its allocation.
    pub fn from_tensor(tensor: &Tensor) -> Result<Self, String> {
        let device = match tensor.device() {
            Device::Cuda(device) => device,
            device => return Err(format!("expected CUDA tensor, got {device:?}")),
        };
        let handle = tensor
            .storage()
            .as_gpu()
            .ok_or_else(|| "CUDA tensor has no GPU storage".to_string())?;
        let owner = handle
            ._prevent_leak
            .clone()
            .ok_or_else(|| "CUDA tensor storage has no owning allocation".to_string())?;
        Ok(Self {
            ptr: handle.ptr as *mut c_void,
            len: handle.len,
            device,
            owner,
        })
    }

    /// Turn an owned CUDA allocation into a Tensor while preserving ownership.
    pub(crate) fn into_tensor(self, shape: Shape, dtype: DType) -> Tensor {
        let device = Device::Cuda(self.device);
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
            .ok_or_else(|| "CUDA tensor byte size overflow".to_string())?;
        if expected != self.len {
            return Err(format!(
                "CUDA tensor view needs {expected} bytes, buffer has {}",
                self.len
            ));
        }
        Ok(self.clone().into_tensor(shape, dtype))
    }
}

/// Page-locked host memory that is also mapped into the GPU's address
/// space (zero-copy). On unified-memory GPUs (Tegra/Thor) the host and
/// device pointers alias the same physical memory, so a CPU store is
/// visible to a kernel with no `cudaMemcpy` — useful for tiny per-token
/// control inputs (token id, position) where the `cudaMemcpyAsync` API
/// overhead dominates the actual transfer.
pub struct HostMappedBuffer {
    host_ptr: *mut c_void,
    dev_ptr: *mut c_void,
    len: usize,
    device: usize,
}

// SAFETY: the host pointer is page-locked and the device pointer is a
// normal GPU address; both are safe to share across threads.
unsafe impl Send for HostMappedBuffer {}
unsafe impl Sync for HostMappedBuffer {}

impl HostMappedBuffer {
    /// Allocate `len` bytes of pinned, mapped host memory.
    pub fn alloc(len: usize, device: usize) -> Result<Self, String> {
        unsafe {
            ffi::check_cuda(ffi::cudaSetDevice(device as i32))?;
        }
        let mut host_ptr: *mut c_void = std::ptr::null_mut();
        unsafe {
            ffi::check_cuda(ffi::cudaHostAlloc(
                &mut host_ptr,
                len,
                ffi::cudaHostAllocMapped | ffi::cudaHostAllocPortable,
            ))?;
            let mut dev_ptr: *mut c_void = std::ptr::null_mut();
            ffi::check_cuda(ffi::cudaHostGetDevicePointer(&mut dev_ptr, host_ptr, 0))?;
            // Zero the host side so the first kernel read sees 0s.
            std::ptr::write_bytes(host_ptr, 0u8, len);
            Ok(Self {
                host_ptr,
                dev_ptr,
                len,
                device,
            })
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn address(&self) -> CudaDeviceAddress {
        CudaDeviceAddress {
            ptr: self.dev_ptr,
            len: self.len,
            device: self.device,
        }
    }

    /// Publish one mapped u32 value to the device without exposing host raw
    /// pointers to model code.
    pub fn write_u32(&self, value: u32) -> Result<(), String> {
        self.write_u32s(&[value])
    }

    pub fn write_u32s(&self, values: &[u32]) -> Result<(), String> {
        let bytes = values
            .len()
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| "mapped u32 write size overflow".to_string())?;
        if self.len < bytes {
            return Err(format!(
                "mapped buffer is {} bytes, need {}",
                self.len, bytes
            ));
        }
        unsafe {
            for (index, value) in values.iter().copied().enumerate() {
                std::ptr::write_volatile((self.host_ptr as *mut u32).add(index), value);
            }
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }

    pub fn address_at(&self, byte_offset: usize, len: usize) -> Result<CudaDeviceAddress, String> {
        let end = byte_offset
            .checked_add(len)
            .ok_or_else(|| "mapped CUDA address range overflow".to_string())?;
        if end > self.len {
            return Err(format!(
                "mapped CUDA address [{byte_offset}..{end}] exceeds {} bytes",
                self.len
            ));
        }
        Ok(CudaDeviceAddress {
            ptr: unsafe { (self.dev_ptr as *mut u8).add(byte_offset) as *mut c_void },
            len,
            device: self.device,
        })
    }
}

impl Drop for HostMappedBuffer {
    fn drop(&mut self) {
        if !self.host_ptr.is_null() {
            unsafe {
                let _ = ffi::cudaFreeHost(self.host_ptr);
            }
        }
    }
}
