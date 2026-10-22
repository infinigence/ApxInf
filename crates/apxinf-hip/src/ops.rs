//! `Backend` operators: host-side validation, then one launch through the shim.
//!
//! Semantics follow apxinf-cuda, which is what models already run against:
//! F32 and BF16 storage, F32 arithmetic, one rounding at the store. Where the
//! CUDA backend leaves behaviour undefined — `add` and `mul` do not compare
//! operand sizes, `embedding` does not range-check ids — this backend returns
//! an error instead of reading out of bounds.

use apxinf_core::{DType, Error, Result, Tensor};

use crate::ffi::{self, check, DTYPE_BF16, DTYPE_F32};
use crate::runtime::HipContext;

pub(crate) fn dtype_code(op: &str, dtype: DType) -> Result<i32> {
    match dtype {
        DType::F32 => Ok(DTYPE_F32),
        DType::BF16 => Ok(DTYPE_BF16),
        other => Err(Error::Other(format!(
            "apxinf-hip: {op} supports f32 and bf16, got {other}"
        ))),
    }
}

fn same_dtype(expected: &Tensor, got: &Tensor) -> Result<()> {
    if expected.dtype() == got.dtype() {
        Ok(())
    } else {
        Err(Error::DTypeMismatch { expected: expected.dtype(), got: got.dtype() })
    }
}

/// Shim extents are `int`; refuse anything larger instead of truncating.
pub(crate) fn as_i32(what: &str, value: usize) -> Result<i32> {
    i32::try_from(value)
        .map_err(|_| Error::Other(format!("apxinf-hip: {what} = {value} exceeds the i32 range")))
}

pub(crate) fn rms_norm(ctx: &HipContext, input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let dtype = dtype_code("rms_norm", input.dtype())?;
    same_dtype(input, weight)?;
    let dims = input.shape().dims();
    let cols = *dims
        .last()
        .ok_or_else(|| Error::Other("apxinf-hip: rms_norm input must have at least one dimension".into()))?;
    if cols == 0 || weight.numel() != cols {
        return Err(Error::ShapeMismatch {
            expected: format!("weight [{cols}]"),
            got: weight.shape().to_string(),
        });
    }
    let rows = input.numel() / cols;
    let (x, w) = (ctx.ptr(input)?, ctx.ptr(weight)?);
    ctx.bind()?;
    let (out, y) = ctx.empty(dims.to_vec(), input.dtype())?;
    check("rms_norm", unsafe {
        ffi::apxinf_hip_rms_norm(dtype, x, w, y, rows as i64, cols as i64, eps, ctx.stream())
    })?;
    Ok(out)
}

pub(crate) fn silu(ctx: &HipContext, input: &Tensor) -> Result<Tensor> {
    let dtype = dtype_code("silu", input.dtype())?;
    let x = ctx.ptr(input)?;
    ctx.bind()?;
    let (out, y) = ctx.empty(input.shape().dims().to_vec(), input.dtype())?;
    check("silu", unsafe {
        ffi::apxinf_hip_silu(dtype, x, y, input.numel() as i64, ctx.stream())
    })?;
    Ok(out)
}

/// `op`: 0 adds, 1 multiplies. The output takes `a`'s shape, as in apxinf-cuda;
/// unlike it, operands of different sizes are an error rather than an
/// out-of-bounds read.
pub(crate) fn binary(ctx: &HipContext, name: &str, op: i32, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let dtype = dtype_code(name, a.dtype())?;
    same_dtype(a, b)?;
    if a.numel() != b.numel() {
        return Err(Error::ShapeMismatch {
            expected: a.shape().to_string(),
            got: b.shape().to_string(),
        });
    }
    let (pa, pb) = (ctx.ptr(a)?, ctx.ptr(b)?);
    ctx.bind()?;
    let (out, y) = ctx.empty(a.shape().dims().to_vec(), a.dtype())?;
    check(name, unsafe {
        ffi::apxinf_hip_binary(dtype, op, pa, pb, y, a.numel() as i64, ctx.stream())
    })?;
    Ok(out)
}

pub(crate) fn scale(ctx: &HipContext, input: &Tensor, factor: f32) -> Result<Tensor> {
    let dtype = dtype_code("scale", input.dtype())?;
    let x = ctx.ptr(input)?;
    ctx.bind()?;
    let (out, y) = ctx.empty(input.shape().dims().to_vec(), input.dtype())?;
    check("scale", unsafe {
        ffi::apxinf_hip_scale(dtype, x, y, input.numel() as i64, factor, ctx.stream())
    })?;
    Ok(out)
}

/// `[..., M, K] @ [K, N] -> [..., N]`. Leading dimensions of `a` fold into M,
/// the activation-times-weight shape every model here uses. A batched right
/// operand is rejected rather than silently treated as 2-D.
pub(crate) fn matmul(ctx: &HipContext, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let dtype = dtype_code("matmul", a.dtype())?;
    same_dtype(a, b)?;
    let (ad, bd) = (a.shape().dims(), b.shape().dims());
    if ad.len() < 2 || bd.len() != 2 {
        return Err(Error::Other(format!(
            "apxinf-hip: matmul expects [..., M, K] @ [K, N], got {} @ {}",
            a.shape(),
            b.shape()
        )));
    }
    let k = ad[ad.len() - 1];
    let n = bd[1];
    if bd[0] != k || k == 0 {
        return Err(Error::MatmulDimMismatch {
            m: ad[ad.len() - 2],
            k1: k,
            k2: bd[0],
            n,
        });
    }
    let m = a.numel() / k;
    let mut out_dims = ad.to_vec();
    *out_dims.last_mut().unwrap() = n;

    let (pa, pb) = (ctx.ptr(a)?, ctx.ptr(b)?);
    ctx.bind()?;
    let (out, pc) = ctx.empty(out_dims, a.dtype())?;
    if m == 0 || n == 0 {
        return Ok(out);
    }
    check("matmul", unsafe {
        ffi::apxinf_hip_gemm(
            ctx.blas(),
            dtype,
            as_i32("matmul M", m)?,
            as_i32("matmul N", n)?,
            as_i32("matmul K", k)?,
            pa,
            pb,
            pc,
        )
    })?;
    Ok(out)
}

/// Half-split RoPE over `[seq, n_heads, head_dim]`; a 2-D input is one token.
pub(crate) fn rope(
    ctx: &HipContext,
    input: &Tensor,
    n_heads: usize,
    head_dim: usize,
    theta: f32,
    pos_offset: u32,
) -> Result<Tensor> {
    let dtype = dtype_code("rope", input.dtype())?;
    let dims = input.shape().dims();
    let seq = if dims.len() == 2 { 1 } else { dims.first().copied().unwrap_or(0) };
    if head_dim == 0 || !head_dim.is_multiple_of(2) || input.numel() != seq * n_heads * head_dim {
        return Err(Error::Other(format!(
            "apxinf-hip: rope expects [seq, {n_heads}, {head_dim}] with an even head_dim, got {}",
            input.shape()
        )));
    }
    let x = ctx.ptr(input)?;
    ctx.bind()?;
    let (out, y) = ctx.empty(dims.to_vec(), input.dtype())?;
    check("rope", unsafe {
        ffi::apxinf_hip_rope(
            dtype,
            x,
            y,
            as_i32("rope seq", seq)?,
            as_i32("rope heads", n_heads)?,
            as_i32("rope head_dim", head_dim)?,
            theta,
            pos_offset,
            ctx.stream(),
        )
    })?;
    Ok(out)
}

/// Row lookup `table[ids] -> [ids.len(), dim]`, in the table's dtype.
pub(crate) fn embedding(ctx: &HipContext, table: &Tensor, ids: &[u32]) -> Result<Tensor> {
    dtype_code("embedding", table.dtype())?;
    let dims = table.shape().dims();
    if dims.len() != 2 {
        return Err(Error::Other(format!(
            "apxinf-hip: embedding table must be [vocab, dim], got {}",
            table.shape()
        )));
    }
    let (vocab, dim) = (dims[0], dims[1]);
    // The ids are on the host anyway, so checking them costs nothing and keeps
    // a bad token from becoming an out-of-bounds device read.
    if let Some(&bad) = ids.iter().find(|&&id| id as usize >= vocab) {
        return Err(Error::Other(format!(
            "apxinf-hip: embedding token id {bad} is out of range for vocabulary {vocab}"
        )));
    }
    let src = ctx.ptr(table)?;
    ctx.bind()?;
    let (out, y) = ctx.empty(vec![ids.len(), dim], table.dtype())?;
    if ids.is_empty() {
        return Ok(out);
    }
    let id_bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_ne_bytes()).collect();
    let id_buffer = ctx.alloc(id_bytes.len())?;
    ctx.upload(id_buffer.ptr(), &id_bytes)?;
    check("embedding", unsafe {
        ffi::apxinf_hip_embedding(
            table.dtype().size_in_bytes() as i32,
            src,
            id_buffer.ptr(),
            y,
            ids.len() as i64,
            dim as i64,
            ctx.stream(),
        )
    })?;
    // `id_buffer` drops here. Its free cannot overtake the launch above: it is
    // stream-ordered, or synchronizes the device first (see `HipBuffer`).
    Ok(out)
}
