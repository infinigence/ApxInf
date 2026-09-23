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

/// A contiguous range along one axis; end is exclusive, no negative indexing.
#[derive(Clone, Copy, Debug)]
pub struct AxisSlice {
    pub axis: usize,
    pub start: usize,
    pub end: usize,
}
impl AxisSlice {
    pub fn output_shape(&self, shape: &[usize]) -> Result<Vec<usize>> {
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
        let mut out = shape.to_vec();
        out[self.axis] = self.end - self.start;
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

/// Permute axes then materialize a contiguous result (not a strided view).
pub fn permuted_shape(shape: &[usize], axes: &[usize]) -> Result<Vec<usize>> {
    checked_elements(shape)?;
    if axes.len() != shape.len() {
        return Err(shape_mismatch(shape, axes));
    }
    let mut seen = vec![false; shape.len()];
    let mut out = Vec::with_capacity(shape.len());
    for &a in axes {
        if a >= shape.len() || seen[a] {
            return Err(Error::Contract("axes must be a permutation of 0..rank"));
        }
        seen[a] = true;
        out.push(shape[a]);
    }
    Ok(out)
}

/// Concatenate equal-rank shapes. Only the concatenation axis may differ.
pub fn concatenated_shape(shapes: &[&[usize]], axis: usize) -> Result<Vec<usize>> {
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
    let mut out = first.to_vec();
    out[axis] = 0;
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
        out[axis] = out[axis]
            .checked_add(shape[axis])
            .ok_or(Error::Contract("concat axis overflow"))?;
    }
    checked_elements(&out)?;
    Ok(out)
}
