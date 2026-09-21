//! Portable operator contracts. No runtime handles, launch parameters or packed weights.
//!
//! All shapes describe dense contiguous row-major tensors. New portable operators
//! are functional: they never modify input storage (including aliased clones).
//! Layout operations preserve bits; arithmetic operates on F32/F16/BF16 only.
//! Backend implementations must reject unsupported dtype/device combinations;
//! they must not silently copy to CPU, change precision or ignore an option.

use crate::{DType, Device, Error, Result, Tensor};

fn invalid(message: impl Into<String>) -> Error {
    Error::Other(message.into())
}

/// Validate a floating-point operand and its storage extent before launch.
pub fn float_tensor(input: &Tensor, device: Device) -> Result<()> {
    if input.device() != device {
        return Err(Error::DeviceMismatch {
            expected: device,
            got: input.device(),
        });
    }
    if !matches!(input.dtype(), DType::F32 | DType::F16 | DType::BF16) {
        return Err(invalid("portable arithmetic requires f32, f16 or bf16"));
    }
    let bytes = checked_elements(input.shape().dims())?
        .checked_mul(input.dtype().size_in_bytes())
        .ok_or_else(|| invalid("tensor byte size overflow"))?;
    if input.storage().len() < bytes {
        return Err(Error::DataLengthMismatch {
            expected: bytes,
            got: input.storage().len(),
        });
    }
    Ok(())
}

/// Scalars are supported; empty dimensions and overflowing shapes are rejected.
pub fn checked_elements(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |n, &d| {
        if d == 0 {
            return Err(invalid("portable operators reject empty dimensions"));
        }
        n.checked_mul(d)
            .ok_or_else(|| invalid("shape element count overflow"))
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
    /// produce zero output. A backend validates values before executing them.
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

    /// Return the output shape or a validation error; no kernel launch occurs.
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
            return Err(invalid("attention scale must be finite and positive"));
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
                return Err(invalid(
                    "attention intermediate dtype must be f32 or input dtype",
                ));
            }
        }
        let qs = q.shape().dims();
        let ks = k.shape().dims();
        if qs.len() != 4 || ks.len() != 4 || v.shape().dims() != ks {
            return Err(invalid("attention expects Q[B,Q,Hq,D], K/V[B,K,Hkv,D]"));
        }
        if qs[0] != ks[0] || qs[3] != ks[3] || qs[2] % ks[2] != 0 {
            return Err(invalid("attention batch/head dimensions do not match"));
        }
        match self.mask {
            AttentionMask::Full => {}
            AttentionMask::Causal { q_start, k_start } => {
                q_start
                    .checked_add(qs[1] - 1)
                    .ok_or_else(|| invalid("query position overflow"))?;
                k_start
                    .checked_add(ks[1] - 1)
                    .ok_or_else(|| invalid("key position overflow"))?;
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
                    return Err(invalid("attention bias requires rank four"));
                }
                broadcast_shape(mask.shape().dims(), &target)?;
                if device == Device::Cpu
                    && mask
                        .as_f32()?
                        .iter()
                        .any(|x| x.is_nan() || *x == f32::INFINITY)
                {
                    return Err(invalid("attention bias contains NaN or positive infinity"));
                }
            }
        }
        Ok(qs.to_vec())
    }
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
            return Err(invalid("invalid or empty slice"));
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
        return Err(invalid("incompatible broadcast shape"));
    }
    Ok(())
}

/// Permute axes then materialize a contiguous result (not a strided view).
pub fn permuted_shape(shape: &[usize], axes: &[usize]) -> Result<Vec<usize>> {
    checked_elements(shape)?;
    if axes.len() != shape.len() {
        return Err(invalid("permutation rank mismatch"));
    }
    let mut seen = vec![false; shape.len()];
    let mut out = Vec::with_capacity(shape.len());
    for &a in axes {
        if a >= shape.len() || seen[a] {
            return Err(invalid("axes must be a permutation"));
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
        .ok_or_else(|| invalid("concat requires at least one input"))?;
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
            return Err(invalid("concat shape mismatch"));
        }
        out[axis] = out[axis]
            .checked_add(shape[axis])
            .ok_or_else(|| invalid("concat axis overflow"))?;
    }
    checked_elements(&out)?;
    Ok(out)
}
