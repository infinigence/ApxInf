//! HIP device context, stream, allocations, and the bridge to `Tensor`.

use std::ffi::c_void;
use std::sync::Arc;

use apxinf_core::storage::GpuStorageHandle;
use apxinf_core::{DType, Device, Error, Result, Shape, Storage, Tensor};

use crate::ffi::{self, check};

/// Device properties the backend acts on, queried once per context.
#[derive(Clone, Debug)]
pub struct HipDeviceCaps {
    /// Full `gcnArchName`, e.g. `gfx1151` or `gfx90a:sramecc+:xnack-`.
    pub arch: String,
    pub name: String,
    /// 32 on RDNA, 64 on CDNA. Recorded rather than assumed; no kernel here
    /// depends on it, but tuning and later fused kernels will.
    pub warp_size: u32,
    pub compute_units: u32,
    pub total_memory: u64,
    /// Whether `hipMallocAsync` is available; see [`HipBuffer`].
    pub memory_pools: bool,
}

impl HipDeviceCaps {
    /// The architecture without feature flags, as `--offload-arch` spells it.
    pub fn base_arch(&self) -> &str {
        self.arch.split(':').next().unwrap_or(&self.arch)
    }
}

/// The one stream every operation is issued on.
///
/// A single in-order stream is what makes the rest of this crate simple: an
/// output is never read before the kernel writing it, and a stream-ordered
/// free cannot overtake work that still uses the memory.
pub(crate) struct HipStream {
    raw: *mut c_void,
}

// SAFETY: a HIP stream handle is an opaque token usable from any host thread.
// The backend is not designed for concurrent issue from several threads (same
// as apxinf-cuda); this only allows the handle to be owned by values that move
// between threads, such as tensor storage.
unsafe impl Send for HipStream {}
unsafe impl Sync for HipStream {}

impl HipStream {
    pub(crate) fn raw(&self) -> *mut c_void {
        self.raw
    }
}

impl Drop for HipStream {
    fn drop(&mut self) {
        // Nothing useful can be done with a failure while dropping.
        unsafe { ffi::apxinf_hip_stream_destroy(self.raw) };
    }
}

struct BlasHandle {
    raw: *mut c_void,
}

// SAFETY: as for `HipStream`; the handle is only used through the context.
unsafe impl Send for BlasHandle {}
unsafe impl Sync for BlasHandle {}

impl Drop for BlasHandle {
    fn drop(&mut self) {
        unsafe { ffi::apxinf_hip_blas_destroy(self.raw) };
    }
}

/// A device allocation, owned by the tensor storage that refers to it.
///
/// Holding an `Arc` to the stream keeps the stream alive as long as any tensor
/// allocated on it, so a tensor that outlives its backend still frees safely.
/// With memory pools the free is stream-ordered; otherwise `hipFree`
/// synchronizes the device first. Either way a pending kernel never loses the
/// memory it is reading.
pub(crate) struct HipBuffer {
    ptr: *mut c_void,
    bytes: usize,
    stream: Arc<HipStream>,
    stream_ordered: bool,
}

// SAFETY: the pointer is a device address, not host memory; it is freed exactly
// once, in `Drop`, on the stream that owns it.
unsafe impl Send for HipBuffer {}
unsafe impl Sync for HipBuffer {}

impl Drop for HipBuffer {
    fn drop(&mut self) {
        unsafe {
            ffi::apxinf_hip_free(self.ptr, self.stream_ordered as i32, self.stream.raw())
        };
    }
}

impl HipBuffer {
    pub(crate) fn ptr(&self) -> *mut c_void {
        self.ptr
    }
}

/// Everything one HIP device needs: its stream, BLAS handle and properties.
pub struct HipContext {
    device_id: usize,
    stream: Arc<HipStream>,
    blas: BlasHandle,
    caps: HipDeviceCaps,
}

impl HipContext {
    pub fn new(device_id: usize) -> Result<Self> {
        let device = i32::try_from(device_id)
            .map_err(|_| Error::Other(format!("apxinf-hip: device index {device_id} is too large")))?;

        let mut count = 0i32;
        let status = unsafe { ffi::apxinf_hip_device_count(&mut count) };
        if status == ffi::NOT_COMPILED {
            return Err(Error::Other(
                "apxinf-hip was built without ROCm (no hipcc found at build time); \
                 rebuild on a ROCm host to use a HIP device"
                    .into(),
            ));
        }
        check("hipGetDeviceCount", status)?;
        if device >= count {
            return Err(Error::Other(format!(
                "apxinf-hip: device hip:{device_id} requested, {count} HIP device(s) visible"
            )));
        }
        check("hipSetDevice", unsafe { ffi::apxinf_hip_set_device(device) })?;

        let caps = query_caps(device)?;
        let compiled = env!("APXINF_HIP_ARCH");
        if caps.base_arch() != compiled {
            return Err(Error::Other(format!(
                "apxinf-hip: kernels were built for {compiled}, but hip:{device_id} is {} ({}); \
                 rebuild with APXINF_HIP_ARCH={}",
                caps.base_arch(),
                caps.name,
                caps.base_arch()
            )));
        }

        let mut raw_stream = std::ptr::null_mut();
        check("hipStreamCreate", unsafe { ffi::apxinf_hip_stream_create(&mut raw_stream) })?;
        let stream = Arc::new(HipStream { raw: raw_stream });

        let mut raw_blas = std::ptr::null_mut();
        let status = unsafe { ffi::apxinf_hip_blas_create(&mut raw_blas, stream.raw()) };
        // Wrap before checking so a handle created before `hipblasSetStream`
        // failed is still destroyed.
        let blas = BlasHandle { raw: raw_blas };
        check("hipblasCreate", status)?;

        Ok(Self { device_id, stream, blas, caps })
    }

    pub fn device_id(&self) -> usize {
        self.device_id
    }

    pub fn device(&self) -> Device {
        Device::Hip(self.device_id)
    }

    pub fn caps(&self) -> &HipDeviceCaps {
        &self.caps
    }

    pub(crate) fn stream(&self) -> *mut c_void {
        self.stream.raw()
    }

    pub(crate) fn blas(&self) -> *mut c_void {
        self.blas.raw
    }

    /// Make this device current on the calling thread. HIP's current device is
    /// per thread, so every entry point binds before issuing work.
    pub(crate) fn bind(&self) -> Result<()> {
        check("hipSetDevice", unsafe { ffi::apxinf_hip_set_device(self.device_id as i32) })
    }

    pub fn synchronize(&self) -> Result<()> {
        self.bind()?;
        check("hipStreamSynchronize", unsafe {
            ffi::apxinf_hip_stream_synchronize(self.stream())
        })
    }

    pub(crate) fn alloc(&self, bytes: usize) -> Result<HipBuffer> {
        let mut ptr = std::ptr::null_mut();
        check("hipMalloc", unsafe {
            ffi::apxinf_hip_malloc(
                &mut ptr,
                bytes as u64,
                self.caps.memory_pools as i32,
                self.stream(),
            )
        })?;
        Ok(HipBuffer {
            ptr,
            bytes,
            stream: Arc::clone(&self.stream),
            stream_ordered: self.caps.memory_pools,
        })
    }

    /// Allocate an uninitialized device tensor and return it with its address.
    pub(crate) fn empty(&self, dims: Vec<usize>, dtype: DType) -> Result<(Tensor, *mut c_void)> {
        let shape = Shape::new(dims);
        let bytes = shape
            .numel()
            .checked_mul(dtype.size_in_bytes())
            .ok_or_else(|| Error::Other("apxinf-hip: tensor byte size overflows".into()))?;
        let buffer = self.alloc(bytes)?;
        let ptr = buffer.ptr();
        Ok((self.wrap(buffer, shape, dtype), ptr))
    }

    /// Hand an allocation to tensor storage, which owns it from here on.
    pub(crate) fn wrap(&self, buffer: HipBuffer, shape: Shape, dtype: DType) -> Tensor {
        let (ptr, bytes) = (buffer.ptr() as usize, buffer.bytes);
        // SAFETY: `ptr..ptr+bytes` is a live device allocation on this device,
        // and the owner frees it only after the last handle drops.
        let handle = unsafe { GpuStorageHandle::from_raw_parts(ptr, bytes, Some(Arc::new(buffer))) };
        let storage = Storage::Gpu { device: self.device(), handle };
        // SAFETY: the allocation holds exactly `numel * size_of(dtype)` bytes.
        unsafe { Tensor::from_raw_parts_unchecked(shape, dtype, self.device(), storage) }
    }

    /// Device address of a tensor that lives on this device.
    ///
    /// Rejects host tensors, tensors on another device (including CUDA ones),
    /// and storage too short for the declared shape — the checks that turn a
    /// misrouted tensor into an error instead of an illegal memory access.
    pub(crate) fn ptr(&self, tensor: &Tensor) -> Result<*mut c_void> {
        if tensor.device() != self.device() {
            return Err(Error::DeviceMismatch {
                expected: self.device(),
                got: tensor.device(),
            });
        }
        let handle = tensor.storage().as_gpu().ok_or(Error::DeviceMismatch {
            expected: self.device(),
            got: Device::Cpu,
        })?;
        if handle.len() < tensor.size_in_bytes() {
            return Err(Error::DataLengthMismatch {
                expected: tensor.size_in_bytes(),
                got: handle.len(),
            });
        }
        Ok(handle.ptr() as *mut c_void)
    }

    /// Copy host bytes into device memory and wait for the copy to finish.
    pub(crate) fn upload(&self, dst: *mut c_void, src: &[u8]) -> Result<()> {
        check("hipMemcpy H2D", unsafe {
            ffi::apxinf_hip_memcpy_htod(dst, src.as_ptr().cast(), src.len() as u64, self.stream())
        })
    }

    /// Copy device memory to the host after all queued work has completed.
    pub(crate) fn download(&self, dst: &mut [u8], src: *const c_void) -> Result<()> {
        check("hipMemcpy D2H", unsafe {
            ffi::apxinf_hip_memcpy_dtoh(dst.as_mut_ptr().cast(), src, dst.len() as u64, self.stream())
        })
    }

    pub(crate) fn copy_on_device(&self, dst: *mut c_void, src: *const c_void, bytes: usize) -> Result<()> {
        check("hipMemcpy D2D", unsafe {
            ffi::apxinf_hip_memcpy_dtod(dst, src, bytes as u64, self.stream())
        })
    }

    /// Upload a host tensor, or return a tensor already on this device as is.
    pub fn to_device(&self, tensor: &Tensor) -> Result<Tensor> {
        if tensor.device() == self.device() {
            return Ok(tensor.clone());
        }
        let bytes = tensor
            .storage()
            .as_cpu()
            .ok_or(Error::UnsupportedDevice(tensor.device()))?;
        let len = tensor.size_in_bytes();
        if bytes.len() < len {
            return Err(Error::DataLengthMismatch { expected: len, got: bytes.len() });
        }
        self.bind()?;
        let (out, ptr) = self.empty(tensor.shape().dims().to_vec(), tensor.dtype())?;
        self.upload(ptr, &bytes[..len])?;
        Ok(out)
    }

    /// Download a tensor on this device, or return a host tensor as is.
    pub fn to_cpu(&self, tensor: &Tensor) -> Result<Tensor> {
        if tensor.device() == Device::Cpu {
            return Ok(tensor.clone());
        }
        let src = self.ptr(tensor)?;
        self.bind()?;
        let mut bytes = vec![0u8; tensor.size_in_bytes()];
        self.download(&mut bytes, src)?;
        Tensor::from_raw(tensor.shape().clone(), tensor.dtype(), Device::Cpu, bytes)
    }
}

fn query_caps(device: i32) -> Result<HipDeviceCaps> {
    let mut arch = [0u8; 256];
    let mut name = [0u8; 256];
    let (mut warp, mut cus, mut memory, mut pools) = (0i32, 0i32, 0u64, 0i32);
    check("hipGetDeviceProperties", unsafe {
        ffi::apxinf_hip_device_info(
            device,
            arch.as_mut_ptr(),
            arch.len() as i32,
            name.as_mut_ptr(),
            name.len() as i32,
            &mut warp,
            &mut cus,
            &mut memory,
            &mut pools,
        )
    })?;
    Ok(HipDeviceCaps {
        arch: c_string(&arch),
        name: c_string(&name),
        warp_size: warp as u32,
        compute_units: cus as u32,
        total_memory: memory,
        memory_pools: pools != 0,
    })
}

fn c_string(buf: &[u8]) -> String {
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..end]).into_owned()
}
