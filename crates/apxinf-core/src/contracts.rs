//! Portable operator contracts. No runtime handles, launch parameters or packed weights.
//!
//! All shapes describe dense contiguous row-major tensors. New portable operators
//! are functional: they never modify input storage (including aliased clones).
//! Layout operations preserve bits; arithmetic operates on F32/F16/BF16 only.
//! Backend implementations must reject unsupported dtype/device combinations;
//! they must not silently copy to CPU, change precision or ignore an option.

use crate::{DType, Device, Error, Result, Tensor};

fn shape_mismatch(expected: &[usize], got: &[usize]) -> Error {
    Error::ShapeMismatch {
        expected: format!("{expected:?}"),
        got: format!("{got:?}"),
    }
}

/// Validate device residency and storage extent for any dtype. Layout operators
/// preserve bits, so they accept every dtype; only arithmetic is float-only.
pub fn tensor_storage(input: &Tensor, device: Device) -> Result<()> {
    if input.device() != device {
        return Err(Error::DeviceMismatch {
            expected: device,
            got: input.device(),
        });
    }
    let bytes = checked_elements(input.shape().dims())?
        .checked_mul(input.dtype().size_in_bytes())
        .ok_or(Error::Contract("tensor byte size overflow"))?;
    if input.storage().len() < bytes {
        return Err(Error::DataLengthMismatch {
            expected: bytes,
            got: input.storage().len(),
        });
    }
    Ok(())
}

/// Validate a floating-point operand and its storage extent before launch.
pub fn float_tensor(input: &Tensor, device: Device) -> Result<()> {
    tensor_storage(input, device)?;
    if !input.dtype().is_float() {
        return Err(Error::UnsupportedDType {
            got: input.dtype(),
            allowed: "f32, f16, bf16",
        });
    }
    Ok(())
}

/// Byte size of a materialized output, rejecting element-count and byte overflow.
/// Callers validate the *output* extent before dispatch so a backend that trusts
/// "arguments are already validated" cannot compute an allocation size that wrapped.
pub fn checked_bytes(shape: &[usize], dtype: DType) -> Result<usize> {
    checked_elements(shape)?
        .checked_mul(dtype.size_in_bytes())
        .ok_or(Error::Contract("output byte size overflow"))
}

/// Byte size of an output whose dims are produced lazily, so callers on hot paths
/// never materialize the shape just to check it. Same rules as [`checked_bytes`].
pub fn checked_bytes_iter(mut dims: impl Iterator<Item = usize>, dtype: DType) -> Result<usize> {
    dims.try_fold(1usize, |n, d| {
        if d == 0 {
            return Err(Error::Contract(
                "portable operators reject empty dimensions",
            ));
        }
        n.checked_mul(d)
            .ok_or(Error::Contract("shape element count overflow"))
    })?
    .checked_mul(dtype.size_in_bytes())
    .ok_or(Error::Contract("output byte size overflow"))
}

/// Scalars are supported; empty dimensions and overflowing shapes are rejected.
pub fn checked_elements(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |n, &d| {
        if d == 0 {
            return Err(Error::Contract(
                "portable operators reject empty dimensions",
            ));
        }
        n.checked_mul(d)
            .ok_or(Error::Contract("shape element count overflow"))
    })
}

/// Tensor-valued attention uses [batch, tokens, heads, head_dim]. Unlike cached
/// decode, it has no implicit KV mutation or sequence-position advancement.
#[derive(Clone, Copy, Debug)]
pub enum AttentionMask<'a> {
    /// Every query may attend every key (PI0.5 prefix/action, independent vision batches).
    Full,
    /// A query at q_start+i can see key k_start+j iff k_start+j <= q_start+i.
    /// Explicit offsets remove ambiguity for unequal Q/K lengths and cached prefixes.
    Causal { q_start: usize, k_start: usize },
    /// Additive F32 bias with shape [B|1, Hq|1, Q|1, K|1], on the same device.
    /// Zero keeps an entry, negative infinity masks it. Finite values are valid
    /// additive biases; NaN and positive infinity are invalid. All-masked rows
    /// produce zero output. Validate values when constructing or updating a mask
    /// (see [`validate_mask_values`]), then reuse it without per-layer scans.
    /// Device-generated masks require a trusted generator or explicit device validation.
    Additive(&'a Tensor),
}

/// Explicit materialization precision for attention intermediates.
/// Accumulation, scale and softmax reductions use F32. QK scores are rounded to
/// `scores` after scaling/bias/masking, probabilities to `probabilities` after
/// softmax; PV accumulates in F32, then rounds once to the input/output dtype.
/// F32 requests do not permit TF32 or reduced-precision intermediates silently.
#[derive(Clone, Copy, Debug)]
pub struct AttentionOptions<'a> {
    pub scale: f32,
    pub mask: AttentionMask<'a>,
    pub scores: DType,
    pub probabilities: DType,
}

impl<'a> AttentionOptions<'a> {
    pub fn full(scale: f32) -> Self {
        Self {
            scale,
            mask: AttentionMask::Full,
            scores: DType::F32,
            probabilities: DType::F32,
        }
    }

    /// Return the output shape after structural and scalar-option validation.
    /// Checks shape, dtype, device and storage extent without reading tensor
    /// contents, copying data or launching kernels. Work scales with tensor rank,
    /// not element count. Mask values must be validated separately at construction
    /// or update time; see [`validate_mask_values`].
    /// Query head h uses KV head floor(h / (Hq/Hkv)) (MHA/GQA/MQA).
    pub fn validate(
        &self,
        device: Device,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<Vec<usize>> {
        for t in [q, k, v] {
            float_tensor(t, device)?;
        }
        if !self.scale.is_finite() || self.scale <= 0. {
            return Err(Error::Contract(
                "attention scale must be finite and positive",
            ));
        }
        for dtype in [k.dtype(), v.dtype()] {
            if dtype != q.dtype() {
                return Err(Error::DTypeMismatch {
                    expected: q.dtype(),
                    got: dtype,
                });
            }
        }
        for dtype in [self.scores, self.probabilities] {
            if dtype != DType::F32 && dtype != q.dtype() {
                return Err(Error::UnsupportedDType {
                    got: dtype,
                    allowed: "f32 or the Q/K/V dtype",
                });
            }
        }
        let qs = q.shape().dims();
        let ks = k.shape().dims();
        if qs.len() != 4 || ks.len() != 4 {
            return Err(Error::Contract(
                "attention expects Q[B,Q,Hq,D] and K/V[B,K,Hkv,D]",
            ));
        }
        if v.shape().dims() != ks {
            return Err(shape_mismatch(ks, v.shape().dims()));
        }
        // Report K against the shape Q implies, so the differing axis is visible.
        if qs[0] != ks[0] || qs[3] != ks[3] {
            return Err(shape_mismatch(&[qs[0], ks[1], ks[2], qs[3]], ks));
        }
        if !qs[2].is_multiple_of(ks[2]) {
            return Err(Error::Contract(
                "Hq must be a multiple of Hkv (MHA/GQA/MQA grouping)",
            ));
        }
        match self.mask {
            AttentionMask::Full => {}
            AttentionMask::Causal { q_start, k_start } => {
                q_start
                    .checked_add(qs[1] - 1)
                    .ok_or(Error::Contract("query position overflow"))?;
                k_start
                    .checked_add(ks[1] - 1)
                    .ok_or(Error::Contract("key position overflow"))?;
            }
            AttentionMask::Additive(mask) => {
                float_tensor(mask, device)?;
                if mask.dtype() != DType::F32 {
                    return Err(Error::DTypeMismatch {
                        expected: DType::F32,
                        got: mask.dtype(),
                    });
                }
                let target = [qs[0], qs[2], qs[1], ks[1]];
                if mask.ndim() != 4 {
                    return Err(Error::Contract("attention bias requires rank four"));
                }
                broadcast_shape(mask.shape().dims(), &target)?;
            }
        }
        Ok(qs.to_vec())
    }
}

/// Check additive F32 mask values on CPU in O(mask.numel()) time.
/// Finite biases and negative infinity are valid; NaN and positive infinity
/// are rejected. Only logical tensor elements are scanned, without broadcasting.
///
/// Call once when constructing a mask, and again after any content update, before
/// sharing it across attention layers. This is not a structural attention check:
/// [`AttentionOptions::validate`] still checks rank and broadcast compatibility.
/// Non-CPU masks return [`Error::UnsupportedDevice`]; this function never downloads
/// data or silently skips validation. Validate before upload, or use a trusted
/// device-side generator or explicit backend value-validation path.
pub fn validate_mask_values(mask: &Tensor) -> Result<()> {
    if mask.device() != Device::Cpu {
        return Err(Error::UnsupportedDevice(mask.device()));
    }
    if mask.dtype() != DType::F32 {
        return Err(Error::DTypeMismatch {
            expected: DType::F32,
            got: mask.dtype(),
        });
    }
    float_tensor(mask, Device::Cpu)?;
    let elements = checked_elements(mask.shape().dims())?;
    if mask.as_f32()?[..elements]
        .iter()
        .any(|x| x.is_nan() || *x == f32::INFINITY)
    {
        return Err(Error::Contract(
            "attention bias contains NaN or positive infinity",
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum PlannedMask {
    Full,
    Causal { q_start: usize, k_start: usize },
    Additive { dims: [usize; 4] },
}

/// A validated attention configuration, built once per execution plan.
///
/// Splits validation in two. Construction runs the **configuration-level** checks
/// — dtype sets, rank, head grouping, scale, mask broadcast compatibility, element
/// and byte overflow — which depend only on the model config and therefore cannot
/// change between calls that share a plan. Dispatch then re-checks only the
/// **instance-level** facts (device, exact shape, dtype, storage extent), which are
/// a handful of comparisons with no loops over dims and no allocation.
///
/// This keeps the hot path cheap without trusting the caller: a plan built for one
/// configuration cannot be silently used with tensors of another.
/// Mask kind and causal offsets are fixed by the plan. Changing either requires
/// a new plan so causal position overflow is checked again at construction.
///
/// The plan holds **no device**. Every fact it captures is device-independent, so
/// it stays valid on any backend whose operands match. The dispatching device is
/// passed to [`AttentionPlan::check_operands`] by the one authority that knows it
/// (`Backend::device()`), leaving no second copy to disagree with and therefore no
/// reconciliation check. The `device` argument to [`AttentionPlan::new`] validates
/// the operands at construction and is not retained.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttentionPlan {
    dtype: DType,
    scale_bits: u32,
    scores: DType,
    probabilities: DType,
    q_dims: [usize; 4],
    kv_dims: [usize; 4],
    mask: PlannedMask,
    q_bytes: usize,
    kv_bytes: usize,
    mask_bytes: usize,
}

impl AttentionPlan {
    /// Run the full structural validation once. Cost is the same as
    /// [`AttentionOptions::validate`]; amortize it over every call that reuses it.
    pub fn new(
        device: Device,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        options: &AttentionOptions<'_>,
    ) -> Result<Self> {
        let out = options.validate(device, q, k, v)?;
        let dtype = q.dtype();
        checked_bytes(&out, dtype)?;
        let dims = |t: &Tensor| -> [usize; 4] {
            let d = t.shape().dims();
            [d[0], d[1], d[2], d[3]]
        };
        let (mask, mask_bytes) = match options.mask {
            AttentionMask::Additive(m) => (
                PlannedMask::Additive { dims: dims(m) },
                checked_bytes(m.shape().dims(), DType::F32)?,
            ),
            AttentionMask::Full => (PlannedMask::Full, 0),
            AttentionMask::Causal { q_start, k_start } => {
                (PlannedMask::Causal { q_start, k_start }, 0)
            }
        };
        Ok(Self {
            dtype,
            scale_bits: options.scale.to_bits(),
            scores: options.scores,
            probabilities: options.probabilities,
            q_dims: dims(q),
            kv_dims: dims(k),
            mask,
            q_bytes: checked_bytes(q.shape().dims(), dtype)?,
            kv_bytes: checked_bytes(k.shape().dims(), dtype)?,
            mask_bytes,
        })
    }

    /// Output shape this plan guarantees (same as Q).
    #[inline]
    pub fn output_shape(&self) -> &[usize; 4] {
        &self.q_dims
    }

    #[inline]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Per-call check that these operands match the configuration this plan was
    /// validated for. Fixed number of comparisons, no allocation, no dim loops.
    pub fn check_operands(
        &self,
        device: Device,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        options: &AttentionOptions<'_>,
    ) -> Result<()> {
        if options.scale.to_bits() != self.scale_bits
            || options.scores != self.scores
            || options.probabilities != self.probabilities
        {
            return Err(Error::Contract("attention options differ from the plan"));
        }
        for (t, dims, bytes) in [
            (q, &self.q_dims, self.q_bytes),
            (k, &self.kv_dims, self.kv_bytes),
            (v, &self.kv_dims, self.kv_bytes),
        ] {
            operand_matches(t, dims, self.dtype, device, bytes)?;
        }
        match (options.mask, self.mask) {
            (AttentionMask::Additive(m), PlannedMask::Additive { dims }) => {
                operand_matches(m, &dims, DType::F32, device, self.mask_bytes)?
            }
            (AttentionMask::Full, PlannedMask::Full) => {}
            (
                AttentionMask::Causal { q_start, k_start },
                PlannedMask::Causal {
                    q_start: planned_q,
                    k_start: planned_k,
                },
            ) if q_start == planned_q && k_start == planned_k => {}
            _ => return Err(Error::Contract("attention mask differs from the plan")),
        }
        Ok(())
    }
}

/// Shape/dtype/device/extent equality against a plan entry. No allocation.
fn operand_matches(
    t: &Tensor,
    dims: &[usize; 4],
    dtype: DType,
    device: Device,
    bytes: usize,
) -> Result<()> {
    if t.device() != device {
        return Err(Error::DeviceMismatch {
            expected: device,
            got: t.device(),
        });
    }
    if t.dtype() != dtype {
        return Err(Error::DTypeMismatch {
            expected: dtype,
            got: t.dtype(),
        });
    }
    if t.shape().dims() != dims {
        return Err(shape_mismatch(dims, t.shape().dims()));
    }
    if t.storage().len() < bytes {
        return Err(Error::DataLengthMismatch {
            expected: bytes,
            got: t.storage().len(),
        });
    }
    Ok(())
}

/// A contiguous range along one axis; end is exclusive, no negative indexing.
#[derive(Clone, Copy, Debug)]
pub struct AxisSlice {
    pub axis: usize,
    pub start: usize,
    pub end: usize,
}
impl AxisSlice {
    /// Validate against `shape` and return the sliced axis length. Allocation-free,
    /// so hot paths can validate without materializing the output shape.
    pub fn validate(&self, shape: &[usize]) -> Result<usize> {
        checked_elements(shape)?;
        if self.axis >= shape.len() {
            return Err(Error::InvalidAxis {
                axis: self.axis,
                ndim: shape.len(),
            });
        }
        if self.start >= self.end || self.end > shape[self.axis] {
            return Err(Error::Contract(
                "slice must be a non-empty in-bounds [start,end)",
            ));
        }
        Ok(self.end - self.start)
    }

    /// Expected dim `i` of the slice output, without materializing the shape.
    #[inline]
    pub fn output_dim(&self, shape: &[usize], i: usize) -> usize {
        if i == self.axis {
            self.end - self.start
        } else {
            shape[i]
        }
    }

    pub fn output_shape(&self, shape: &[usize]) -> Result<Vec<usize>> {
        let len = self.validate(shape)?;
        let mut out = shape.to_vec();
        out[self.axis] = len;
        Ok(out)
    }
}

/// Right-aligned NumPy broadcasting. Materialized output must be contiguous.
pub fn broadcast_shape(source: &[usize], target: &[usize]) -> Result<()> {
    checked_elements(source)?;
    checked_elements(target)?;
    if source.len() > target.len()
        || source
            .iter()
            .rev()
            .zip(target.iter().rev())
            .any(|(&s, &t)| s != 1 && s != t)
    {
        return Err(shape_mismatch(target, source));
    }
    Ok(())
}

/// Validate that `axes` is a permutation of `0..shape.len()`. Allocation-free:
/// ranks are small, so membership uses a bitmask rather than a heap `Vec<bool>`.
pub fn validate_permutation(shape: &[usize], axes: &[usize]) -> Result<()> {
    checked_elements(shape)?;
    if axes.len() != shape.len() {
        return Err(shape_mismatch(shape, axes));
    }
    if shape.len() > usize::BITS as usize {
        return Err(Error::Contract("permutation rank exceeds usize::BITS"));
    }
    let mut seen = 0usize;
    for &a in axes {
        if a >= shape.len() || seen & (1 << a) != 0 {
            return Err(Error::Contract("axes must be a permutation of 0..rank"));
        }
        seen |= 1 << a;
    }
    Ok(())
}

/// Permute axes then materialize a contiguous result (not a strided view).
pub fn permuted_shape(shape: &[usize], axes: &[usize]) -> Result<Vec<usize>> {
    validate_permutation(shape, axes)?;
    Ok(axes.iter().map(|&a| shape[a]).collect())
}

/// Concatenate equal-rank shapes. Only the concatenation axis may differ.
/// Validate equal-rank concat inputs and return the summed axis length.
/// Allocation-free, so hot paths skip materializing the output shape.
pub fn validate_concat(shapes: &[&[usize]], axis: usize) -> Result<usize> {
    let first = *shapes
        .first()
        .ok_or(Error::Contract("concat requires at least one input"))?;
    checked_elements(first)?;
    if axis >= first.len() {
        return Err(Error::InvalidAxis {
            axis,
            ndim: first.len(),
        });
    }
    let mut total = 0usize;
    for shape in shapes {
        checked_elements(shape)?;
        if shape.len() != first.len()
            || shape
                .iter()
                .zip(first)
                .enumerate()
                .any(|(i, (a, b))| i != axis && a != b)
        {
            return Err(shape_mismatch(first, shape));
        }
        total = total
            .checked_add(shape[axis])
            .ok_or(Error::Contract("concat axis overflow"))?;
    }
    // Guard the materialized element count, which exceeds any single input.
    // Folded in place so the hot path never allocates a probe shape.
    first.iter().enumerate().try_fold(1usize, |n, (i, &d)| {
        let d = if i == axis { total } else { d };
        if d == 0 {
            return Err(Error::Contract(
                "portable operators reject empty dimensions",
            ));
        }
        n.checked_mul(d)
            .ok_or(Error::Contract("shape element count overflow"))
    })?;
    Ok(total)
}

/// Concatenate equal-rank shapes. Only the concatenation axis may differ.
pub fn concatenated_shape(shapes: &[&[usize]], axis: usize) -> Result<Vec<usize>> {
    let total = validate_concat(shapes, axis)?;
    let mut out = shapes[0].to_vec();
    out[axis] = total;
    Ok(out)
}
