//! Model-owned physical layout state, independent of execution IO addresses.

use std::collections::HashMap;
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

/// Model-owned weight allocation with an explicitly tracked physical layout.
///
/// Every consumer of this allocation must attach this same state to its GEMM
/// arguments. The caller must not modify or read the tensor through canonical
/// tensor operations after preparation. Replacing a weight requires a new
/// allocation and state after retiring all executions/graphs that use the old one.
/// Shared-layout mode can transition the base only after its execution leases
/// retire. Clones share state; they do not allocate device memory.
#[derive(Clone)]
pub struct RepackedWeight {
    pub(crate) state: Arc<Mutex<WeightState>>,
}

pub(crate) struct WeightState {
    pub buffer: CudaBuffer,
    pub shape: Vec<usize>,
    pub dtype: apxinf_core::DType,
    pub layout: u32,
    pub base_lease: Weak<WeightStorage>,
    // Recipe reuse is independent of permission to retain extra GPU layouts.
    // In particular, committing canonical layout 0 -> 4 must not retune the
    // same logical operation merely because it binds new activation/output IO.
    pub choices: HashMap<LayoutChoiceKey, crate::ffi::abi::gemm::Choice>,
    pub cache: Option<LayoutCache>,
}

/// A live execution/graph lease, independent of model metadata. Extra storage
/// dies with its last consumer; weak model entries cannot keep it allocated.
pub(crate) struct WeightStorage {
    pub buffer: CudaBuffer,
    pub layout: u32,
}

impl WeightState {
    pub fn prune_layouts(&mut self) {
        if let Some(cache) = self.cache.as_mut() {
            cache.buffers.retain(|_, buffer| buffer.strong_count() != 0);
        }
    }

    pub fn extra(&self, layout: u32) -> Option<Arc<WeightStorage>> {
        self.cache.as_ref()?.buffers.get(&layout)?.upgrade()
    }

    pub fn lease_base(&mut self) -> Arc<WeightStorage> {
        if let Some(lease) = self.base_lease.upgrade() {
            debug_assert_eq!(lease.layout, self.layout);
            return lease;
        }
        let lease = Arc::new(WeightStorage {
            buffer: self.buffer.clone(),
            layout: self.layout,
        });
        self.base_lease = Arc::downgrade(&lease);
        lease
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct LayoutChoiceKey {
    spec: crate::ffi::abi::gemm::Spec,
    workspace_limit: usize,
    graph_safe: bool,
    deterministic: bool,
    online_tune: bool,
    allow_fallback: bool,
    cache_dir: Option<String>,
}

impl LayoutChoiceKey {
    pub fn new(mut spec: crate::ffi::abi::gemm::Spec, policy: &super::GemmPolicy) -> Self {
        spec.b_layout = 0;
        Self {
            spec,
            workspace_limit: policy.workspace_limit,
            graph_safe: policy.graph_safe,
            deterministic: policy.deterministic,
            online_tune: policy.online_tune,
            allow_fallback: policy.allow_fallback,
            cache_dir: policy.cache_dir.clone(),
        }
    }
}

pub(crate) struct LayoutCache {
    pub budget: usize,
    pub buffers: HashMap<u32, Weak<WeightStorage>>,
}

impl LayoutCache {
    pub fn extra_bytes(&self) -> usize {
        self.buffers
            .values()
            .filter_map(Weak::upgrade)
            .map(|storage| storage.buffer.len())
            .sum()
    }
}

impl std::fmt::Debug for RepackedWeight {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RepackedWeight").finish_non_exhaustive()
    }
}

/// A reusable device allocation for W8A8 weights that arrive as checkpoint bytes, row-major `[N, K]`, one weight after
/// another through the same address (layer streaming).
///
/// Where the device has an output-major candidate for this K/N the slot's physical layout is the checkpoint's own, so
/// filling it is a plain DMA from the source bytes; otherwise the slot is canonical `[K, N]` and each fill transposes on
/// the host first. Consumers see the same logical `[K, N]` tensor and shared [`RepackedWeight`] state either way.
///
/// A slot may be refilled only while no execution that reads it is queued ahead of the fill on a stream the consumer does
/// not wait for: order the fill (on any stream) before its readers with a [`crate::CudaEvent`], and do not refill until
/// the readers' work has completed. Executions bind the slot's address, not a packed copy, so a refilled slot is read
/// fresh. Its contents are undefined until the first fill completes.
pub struct OutputMajorSlot {
    pub weight: Tensor,
    pub layout: RepackedWeight,
    n: usize,
    k: usize,
    direct: bool,
}

impl OutputMajorSlot {
    pub fn new(ctx: &crate::CudaContext, n: usize, k: usize) -> Result<Self> {
        if n == 0 || k == 0 {
            return Err(Error::Other("W8A8 slot dimensions must be positive".into()));
        }
        let bytes = n
            .checked_mul(k)
            .ok_or_else(|| Error::Other("W8A8 slot size overflow".into()))?;
        let direct = RepackedWeight::w8a8_candidate_layouts(ctx, n, k)? & (1 << 1) != 0;
        let buffer = CudaBuffer::alloc(bytes, ctx.device_id()).map_err(Error::Cuda)?;
        let weight =
            buffer.into_tensor(apxinf_core::Shape::new(vec![k, n]), apxinf_core::DType::I8);
        let layout = RepackedWeight::new(&weight)?;
        layout
            .state
            .lock()
            .map_err(|_| Error::Cuda("weight state lock poisoned".into()))?
            .layout = if direct { 1 } else { 4 };
        Ok(Self { weight, layout, n, k, direct })
    }

    /// Whether a fill is a plain copy of the source bytes (no host work).
    pub fn is_direct(&self) -> bool {
        self.direct
    }

    /// Enqueue the fill of this slot with the `[N, K]` bytes at `region[offset..offset + N * K]` on `stream`.
    /// A direct slot returns without waiting; a canonical slot transposes on the host and waits for the copy.
    pub fn fill(
        &self,
        region: &crate::PinnedRegion,
        offset: usize,
        stream: &crate::CudaStream,
    ) -> Result<()> {
        let len = self.n * self.k;
        let buffer = CudaBuffer::from_tensor(&self.weight).map_err(Error::Cuda)?;
        if self.direct {
            return buffer
                .copy_from_region_async(0, region, offset, len, stream)
                .map_err(Error::Cuda);
        }
        let source = region
            .bytes()
            .get(offset..offset + len)
            .ok_or_else(|| Error::Other("W8A8 slot fill is outside the source region".into()))?;
        buffer
            .copy_from_host_on(&transpose_to_canonical(source, self.n, self.k), stream)
            .map_err(Error::Cuda)
    }
}

/// `[N, K]` row-major bytes to canonical `[K, N]`, tiled to bound strided cache traffic.
fn transpose_to_canonical(source: &[u8], n: usize, k: usize) -> Vec<u8> {
    let mut canonical = vec![0u8; source.len()];
    for kb in (0..k).step_by(64) {
        for nb in (0..n).step_by(64) {
            for col in nb..(nb + 64).min(n) {
                for row in kb..(kb + 64).min(k) {
                    canonical[row * n + col] = source[col * k + row];
                }
            }
        }
    }
    canonical
}

impl RepackedWeight {
    /// Upload a W8A8 weight that is already stored output-major, physically `[N, K]`
    /// (the layout of checkpoint files), for row/channel GEMM.
    ///
    /// The tensor's logical shape stays `[K, N]` for every consumer, as with
    /// [`Self::upload_row_channel`]. When the device has an output-major candidate for
    /// this K/N the bytes are uploaded as they are, with no host transpose; otherwise they
    /// are transposed to canonical `[K, N]` first, so a caller never sees a layout the
    /// device cannot execute. `source` is the row-major `[N, K]` byte matrix.
    pub fn upload_row_channel_output_major(
        ctx: &crate::CudaContext,
        source: &[u8],
        n: usize,
        k: usize,
    ) -> Result<(Tensor, Self)> {
        if n == 0 || k == 0 || n.checked_mul(k) != Some(source.len()) {
            return Err(Error::Other(
                "W8A8 output-major upload needs a non-empty [N, K] INT8 matrix".into(),
            ));
        }
        let layouts = Self::w8a8_candidate_layouts(ctx, n, k)?;
        let (bytes, layout): (std::borrow::Cow<'_, [u8]>, u32) = if layouts & (1 << 1) != 0 {
            (std::borrow::Cow::Borrowed(source), 1)
        } else {
            (std::borrow::Cow::Owned(transpose_to_canonical(source, n, k)), 4)
        };
        let buffer = CudaBuffer::alloc(bytes.len(), ctx.device_id()).map_err(Error::Cuda)?;
        buffer
            .copy_from_host_on(&bytes, ctx.stream())
            .map_err(Error::Cuda)?;
        let tensor =
            buffer.into_tensor(apxinf_core::Shape::new(vec![k, n]), apxinf_core::DType::I8);
        let state = Self::new(&tensor)?;
        state
            .state
            .lock()
            .map_err(|_| Error::Cuda("weight state lock poisoned".into()))?
            .layout = layout;
        Ok((tensor, state))
    }

    /// Bit set of physical layouts some compiled candidate can execute for a W8A8 `[K, N]`
    /// weight on this device. A capability query, not a tactic; independent of M.
    fn w8a8_candidate_layouts(ctx: &crate::CudaContext, n: usize, k: usize) -> Result<u32> {
        use crate::ffi::abi::{gemm as abi, status};
        let n64 = i64::try_from(n).map_err(|_| Error::Other("W8A8 N overflow".into()))?;
        let k64 = i64::try_from(k).map_err(|_| Error::Other("W8A8 K overflow".into()))?;
        let spec = abi::Spec {
            version: 9,
            semantic: super::contracts::Semantic::Gemm as u32,
            a_dtype: 4,
            b_dtype: 4,
            accumulation_dtype: 5,
            output_dtype: 2,
            quantization: 3,
            b_layout: 0,
            numerics: 0,
            b_is_immutable: 1,
            a_alignment: 16,
            b_alignment: 16,
            output_alignment: 16,
            a_scales_alignment: 4,
            b_scales_alignment: 4,
            bias_alignment: 0,
            residual_alignment: 0,
            a_block_scales_alignment: 0,
            b_block_scales_alignment: 0,
            sf_vec_size: 0,
            m: 1,
            n: n64,
            k: k64,
            alpha_is_unit: 1,
            output_scale_is_unit: 1,
        };
        let policy = abi::Policy {
            workspace_limit: u64::MAX,
            online_tune: 0,
            allow_fallback: 1,
            graph_safe: 1,
            deterministic: 0,
            cache_dir: std::ptr::null(),
        };
        let mut layouts = 0;
        unsafe {
            status::check(abi::apxinf_gemm_candidate_layouts(
                ctx.runtime(),
                &spec,
                &policy,
                &mut layouts,
            ))?;
        }
        Ok(layouts)
    }

    /// Upload canonical CPU `[K, N]` INT8 weights for row/channel W8A8 GEMM.
    /// The compiled device backend fixes NK or canonical layout at load time,
    /// independently of activation rows. Only the final representation is
    /// uploaded; all consumers must attach the returned shared layout state.
    pub fn upload_row_channel(ctx: &crate::CudaContext, weight: &Tensor) -> Result<(Tensor, Self)> {
        use crate::ffi::abi::{gemm as abi, status};
        use apxinf_core::{DType, Device};

        if weight.device() != Device::Cpu || weight.dtype() != DType::I8 {
            return Err(Error::Other("W8A8 upload requires CPU INT8 weights".into()));
        }
        let [k, n] = weight.shape().dims() else {
            return Err(Error::Other("W8A8 upload requires a [K, N] matrix".into()));
        };
        let k64 = i64::try_from(*k).map_err(|_| Error::Other("W8A8 K overflow".into()))?;
        let n64 = i64::try_from(*n).map_err(|_| Error::Other("W8A8 N overflow".into()))?;
        if *k == 0 || *n == 0 {
            return Err(Error::Other("W8A8 dimensions must be positive".into()));
        }
        // This queries capability, not a tactic: the SM87 W8A8 layout contract
        // depends on dtype/K/N and device features, never M. Future providers
        // with shape-dependent layouts must not be added to this fixed policy.
        let spec = abi::Spec {
            version: 9,
            semantic: super::contracts::Semantic::Gemm as u32,
            a_dtype: 4,
            b_dtype: 4,
            accumulation_dtype: 5,
            output_dtype: 2,
            quantization: 3,
            b_layout: 0,
            numerics: 0,
            b_is_immutable: 1,
            a_alignment: 16,
            b_alignment: 16,
            output_alignment: 16,
            a_scales_alignment: 4,
            b_scales_alignment: 4,
            bias_alignment: 0,
            residual_alignment: 0,
            a_block_scales_alignment: 0,
            b_block_scales_alignment: 0,
            sf_vec_size: 0,
            m: 1,
            n: n64,
            k: k64,
            alpha_is_unit: 1,
            output_scale_is_unit: 1,
        };
        let policy = abi::Policy {
            workspace_limit: u64::MAX,
            online_tune: 0,
            allow_fallback: 1,
            graph_safe: 1,
            deterministic: 0,
            cache_dir: std::ptr::null(),
        };
        let mut layouts = 0;
        unsafe {
            status::check(abi::apxinf_gemm_candidate_layouts(
                ctx.runtime(),
                &spec,
                &policy,
                &mut layouts,
            ))?;
        }
        let layout = if layouts & (1 << 1) != 0 { 1 } else { 4 };
        let source = weight
            .storage()
            .as_cpu()
            .ok_or_else(|| Error::Other("expected CPU weight storage".into()))?;
        if k.checked_mul(*n) != Some(source.len()) {
            return Err(Error::Other("W8A8 weight storage size mismatch".into()));
        }
        let buffer = CudaBuffer::alloc(source.len(), ctx.device_id()).map_err(Error::Cuda)?;
        if layout == 1 {
            let mut packed = vec![0u8; source.len()];
            // Tile the host transpose to bound strided cache traffic for large
            // gate/up matrices. Preserve quantized bytes and channel scales.
            for kb in (0..*k).step_by(64) {
                for nb in (0..*n).step_by(64) {
                    for row in kb..(kb + 64).min(*k) {
                        for col in nb..(nb + 64).min(*n) {
                            packed[col * k + row] = source[row * n + col];
                        }
                    }
                }
            }
            buffer.copy_from_host(&packed).map_err(Error::Cuda)?;
        } else {
            buffer.copy_from_host(source).map_err(Error::Cuda)?;
        }
        let tensor = buffer.into_tensor(weight.shape().clone(), DType::I8);
        let state = Self::new(&tensor)?;
        state
            .state
            .lock()
            .map_err(|_| Error::Cuda("weight state lock poisoned".into()))?
            .layout = layout;
        Ok((tensor, state))
    }

    /// Allow independent shape/semantic choices while sharing each physical
    /// layout across live executions. A base with no live execution lease can
    /// change layout during prepare; leased allocations remain immutable.
    /// Extra representations are owned by their consumers, with weak model
    /// entries for reuse. `extra_bytes` bounds live copies, excluding trials.
    /// A budget smaller than one weight pins canonical at adoption, avoiding a
    /// first-caller decision when no second representation could be retained.
    pub fn with_layout_cache(weight: &Tensor, extra_bytes: usize) -> Result<Self> {
        let result = Self::new(weight)?;
        {
            let mut state = result
                .state
                .lock()
                .map_err(|_| Error::Cuda("weight state lock poisoned".into()))?;
            if extra_bytes < state.buffer.len() {
                state.layout = 4;
            }
            state.cache = Some(LayoutCache {
                budget: extra_bytes,
                buffers: HashMap::new(),
            });
        }
        Ok(result)
    }
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
                base_lease: Weak::new(),
                choices: HashMap::new(),
                cache: None,
            })),
        })
    }
}
