//! Pointwise activations, gathers, QKV splits with RoPE, and row copies.
//!
//! Semantics are cuda-new's (see `kernels/elementwise.hip`).

use std::ffi::c_void;
use std::sync::Arc;

use apxinf_core::{Error, Result, Tensor};

use super::{bf16_ptr, int, long, optional_ptr, require_numel, rows_cols, same_shape};
use crate::ffi::{self, check};
use crate::runtime::{HipBuffer, HipContext};

/// Raw device bytes for inputs that are not BF16 tensors: RGB images (`u8`)
/// and token ids (`u32`). Allocated once at a fixed address and refilled
/// before each run, so a captured graph reads the new contents.
pub struct HipDeviceBytes {
    ctx: Arc<HipContext>,
    buffer: HipBuffer,
    len: usize,
}

impl HipDeviceBytes {
    pub fn new(ctx: &Arc<HipContext>, len: usize) -> Result<Self> {
        ctx.bind()?;
        Ok(Self {
            ctx: Arc::clone(ctx),
            buffer: ctx.alloc(len.max(1))?,
            len,
        })
    }

    /// Copy `bytes` to the start of the buffer and wait for the copy, so the
    /// caller may reuse its host memory immediately.
    pub fn upload(&self, bytes: &[u8]) -> Result<()> {
        if bytes.len() > self.len {
            return Err(Error::DataLengthMismatch {
                expected: self.len,
                got: bytes.len(),
            });
        }
        self.ctx.bind()?;
        self.ctx.upload(self.buffer.ptr(), bytes)
    }

    /// Upload `u32` values (token ids) in native byte order.
    pub fn upload_u32(&self, values: &[u32]) -> Result<()> {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
        self.upload(&bytes)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn ptr(&self) -> *const c_void {
        self.buffer.ptr().cast_const()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    None,
    /// GELU, tanh approximation.
    GeluTanh,
    Silu,
}

impl Activation {
    fn code(self) -> i32 {
        match self {
            Self::None => 0,
            Self::GeluTanh => 1,
            Self::Silu => 2,
        }
    }
}

/// `out = activation(x + bias)`, the bias per column and optional.
pub fn bias_activation(
    ctx: &HipContext,
    x: &Tensor,
    bias: Option<&Tensor>,
    activation: Activation,
    out: &Tensor,
) -> Result<()> {
    let (rows, cols) = rows_cols(x)?;
    same_shape(x, out)?;
    let pb = optional_ptr(ctx, bias, cols)?;
    let (px, py) = (bf16_ptr(ctx, x)?, bf16_ptr(ctx, out)?);
    check("bias_activation", unsafe {
        ffi::apxinf_hip_native_bias_activation(
            px,
            pb,
            py,
            long(rows)?,
            int(cols)?,
            activation.code(),
            ctx.stream(),
        )
    })
}

/// GeGLU over a packed `[rows, 2 * inner]` gate/up projection, gate half
/// first: `out[rows, inner] = gelu_tanh(gate) * up`.
pub fn geglu(ctx: &HipContext, gate_up: &Tensor, out: &Tensor) -> Result<()> {
    let (rows, packed) = rows_cols(gate_up)?;
    let (out_rows, inner) = rows_cols(out)?;
    if packed != 2 * inner || out_rows != rows {
        return Err(Error::ShapeMismatch {
            expected: format!("[{rows}, {}] from [{rows}, {packed}]", packed / 2),
            got: out.shape().to_string(),
        });
    }
    let (pg, py) = (bf16_ptr(ctx, gate_up)?, bf16_ptr(ctx, out)?);
    check("geglu", unsafe {
        ffi::apxinf_hip_native_geglu(pg, py, long(rows)?, int(inner)?, ctx.stream())
    })
}

/// One explicit-Euler flow step: `out = state + dt * velocity`.
pub fn euler_update(
    ctx: &HipContext,
    state: &Tensor,
    velocity: &Tensor,
    dt: f32,
    out: &Tensor,
) -> Result<()> {
    same_shape(state, velocity)?;
    same_shape(state, out)?;
    let (ps, pv, py) = (
        bf16_ptr(ctx, state)?,
        bf16_ptr(ctx, velocity)?,
        bf16_ptr(ctx, out)?,
    );
    check("euler_update", unsafe {
        ffi::apxinf_hip_native_euler_update(ps, pv, py, long(state.numel())?, dt, ctx.stream())
    })
}

/// Token-embedding lookup scaled by `sqrt(width)`: `out[t] = table[ids[t]] *
/// sqrt(width)` for the first `out.rows` ids in `ids` (`u32`, native order).
/// Ids outside the vocabulary produce zero rows.
pub fn embedding(
    ctx: &HipContext,
    table: &Tensor,
    ids: &HipDeviceBytes,
    out: &Tensor,
) -> Result<()> {
    let dims = table.shape().dims();
    if dims.len() != 2 {
        return Err(Error::Contract(
            "apxinf-hip embedding: table must be [vocab, width]",
        ));
    }
    let (vocab, width) = (dims[0], dims[1]);
    let (tokens, out_width) = rows_cols(out)?;
    if out_width != width {
        return Err(Error::ShapeMismatch {
            expected: format!("[tokens, {width}]"),
            got: out.shape().to_string(),
        });
    }
    if ids.len() < tokens * 4 {
        return Err(Error::DataLengthMismatch {
            expected: tokens * 4,
            got: ids.len(),
        });
    }
    let vocab = u32::try_from(vocab)
        .map_err(|_| Error::Contract("apxinf-hip embedding: vocabulary exceeds u32"))?;
    let (pt, py) = (bf16_ptr(ctx, table)?, bf16_ptr(ctx, out)?);
    check("embedding", unsafe {
        ffi::apxinf_hip_native_embedding(
            pt,
            ids.ptr(),
            py,
            int(tokens)?,
            int(width)?,
            vocab,
            ctx.stream(),
        )
    })
}

/// Vision patch-embedding epilogue: `out = projection + position (+ bias)`,
/// where `position` is `[tokens_per_view, cols]` and repeats across views.
pub fn add_position(
    ctx: &HipContext,
    projection: &Tensor,
    bias: Option<&Tensor>,
    position: &Tensor,
    tokens_per_view: usize,
    out: &Tensor,
) -> Result<()> {
    let (rows, cols) = rows_cols(projection)?;
    same_shape(projection, out)?;
    require_numel(position, tokens_per_view * cols, || {
        format!("[{tokens_per_view}, {cols}]")
    })?;
    if tokens_per_view == 0 || !rows.is_multiple_of(tokens_per_view) {
        return Err(Error::Contract(
            "apxinf-hip add_position: rows are not whole views",
        ));
    }
    let pb = optional_ptr(ctx, bias, cols)?;
    let (pp, ppos, py) = (
        bf16_ptr(ctx, projection)?,
        bf16_ptr(ctx, position)?,
        bf16_ptr(ctx, out)?,
    );
    check("add_position", unsafe {
        ffi::apxinf_hip_native_add_position(
            pp,
            pb,
            ppos,
            py,
            long(rows)?,
            int(cols)?,
            int(tokens_per_view)?,
            ctx.stream(),
        )
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageLayout {
    Nhwc,
    Nchw,
}

/// Cut `views` square RGB `u8` images into flattened patches
/// `[views * (image_size / patch_size)², 3 * patch_size²]`, channel-major
/// within a patch, mapping `[0, 255]` to `[-1, 1]`.
pub fn rgb_to_patches(
    ctx: &HipContext,
    images: &HipDeviceBytes,
    layout: ImageLayout,
    views: usize,
    image_size: usize,
    patch_size: usize,
    out: &Tensor,
) -> Result<()> {
    if patch_size == 0 || !image_size.is_multiple_of(patch_size) {
        return Err(Error::Contract(
            "apxinf-hip rgb_to_patches: image is not whole patches",
        ));
    }
    let image_bytes = views * image_size * image_size * 3;
    if images.len() < image_bytes {
        return Err(Error::DataLengthMismatch {
            expected: image_bytes,
            got: images.len(),
        });
    }
    let per_view = (image_size / patch_size).pow(2);
    let width = 3 * patch_size * patch_size;
    require_numel(out, views * per_view * width, || {
        format!("[{}, {width}]", views * per_view)
    })?;
    let py = bf16_ptr(ctx, out)?;
    check("rgb_to_patches", unsafe {
        ffi::apxinf_hip_native_rgb_to_patches(
            images.ptr(),
            py,
            int(views)?,
            int(image_size)?,
            int(patch_size)?,
            (layout == ImageLayout::Nhwc) as i32,
            ctx.stream(),
        )
    })
}

/// Geometry of a packed `[tokens, (q_heads + 2 kv_heads) * head_dim]` QKV
/// projection and the RoPE applied to its Q and K.
#[derive(Clone, Copy, Debug)]
pub struct QkvRope {
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub theta: f32,
    /// Position of the first token.
    pub position_offset: usize,
}

/// Split a packed QKV projection, add the optional bias, apply half-split RoPE
/// (frequency `theta^(-pair / (head_dim / 2))`) to Q and K, and copy V.
///
/// `q` receives `[tokens, q_heads, head_dim]`. `k` and `v` are
/// `[rows, kv_heads, head_dim]` buffers — typically a KV cache — and token `t`
/// lands at row `kv_row_offset + t`.
#[allow(clippy::too_many_arguments)]
pub fn split_qkv_rope(
    ctx: &HipContext,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    rope: &QkvRope,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    kv_row_offset: usize,
) -> Result<()> {
    let QkvRope {
        q_heads,
        kv_heads,
        head_dim,
        theta,
        position_offset,
    } = *rope;
    let (tokens, width) = rows_cols(qkv)?;
    let fused = (q_heads + 2 * kv_heads) * head_dim;
    if width != fused || head_dim % 2 != 0 {
        return Err(Error::ShapeMismatch {
            expected: format!("[tokens, {fused}] with an even head_dim"),
            got: qkv.shape().to_string(),
        });
    }
    require_numel(q, tokens * q_heads * head_dim, || {
        format!("[{tokens}, {q_heads}, {head_dim}]")
    })?;
    let kv_row = kv_heads * head_dim;
    for cache in [k, v] {
        if !cache.numel().is_multiple_of(kv_row) || cache.numel() / kv_row < kv_row_offset + tokens
        {
            return Err(Error::ShapeMismatch {
                expected: format!("[>= {}, {kv_heads}, {head_dim}]", kv_row_offset + tokens),
                got: cache.shape().to_string(),
            });
        }
    }
    let pb = optional_ptr(ctx, bias, fused)?;
    let pqkv = bf16_ptr(ctx, qkv)?;
    let (pq, pk, pv) = (bf16_ptr(ctx, q)?, bf16_ptr(ctx, k)?, bf16_ptr(ctx, v)?);
    check("split_qkv_rope", unsafe {
        ffi::apxinf_hip_native_split_qkv_rope(
            pqkv,
            pb,
            pq,
            pk,
            pv,
            int(tokens)?,
            int(q_heads)?,
            int(kv_heads)?,
            int(head_dim)?,
            theta,
            int(position_offset)?,
            int(kv_row_offset)?,
            ctx.stream(),
        )
    })
}

/// Split a packed `[tokens, 3 * width]` projection into `q`, `k`, `v` of
/// `[tokens, width]` elements each, adding the optional bias; no rotation.
pub fn split_qkv_bias(
    ctx: &HipContext,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
) -> Result<()> {
    let (tokens, fused) = rows_cols(qkv)?;
    if fused % 3 != 0 {
        return Err(Error::Contract(
            "apxinf-hip split_qkv_bias: width is not 3 * projection",
        ));
    }
    let width = fused / 3;
    for t in [q, k, v] {
        require_numel(t, tokens * width, || format!("{tokens} x {width} elements"))?;
    }
    let pb = optional_ptr(ctx, bias, fused)?;
    let pqkv = bf16_ptr(ctx, qkv)?;
    let (pq, pk, pv) = (bf16_ptr(ctx, q)?, bf16_ptr(ctx, k)?, bf16_ptr(ctx, v)?);
    check("split_qkv_bias", unsafe {
        ffi::apxinf_hip_native_split_qkv_bias(
            pqkv,
            pb,
            pq,
            pk,
            pv,
            int(tokens)?,
            int(width)?,
            ctx.stream(),
        )
    })
}

/// Copy all rows of `src` into `dst` starting at row `dst_row`; both are
/// `[.., cols]` with the same column count. Concatenation is two of these.
pub fn copy_rows(ctx: &HipContext, src: &Tensor, dst: &Tensor, dst_row: usize) -> Result<()> {
    let (rows, cols) = rows_cols(src)?;
    let (dst_rows, dst_cols) = rows_cols(dst)?;
    if cols != dst_cols || src.dtype() != dst.dtype() || dst_row + rows > dst_rows {
        return Err(Error::ShapeMismatch {
            expected: format!("[>= {}, {cols}] {}", dst_row + rows, src.dtype()),
            got: format!("{} {}", dst.shape(), dst.dtype()),
        });
    }
    let element = src.dtype().size_in_bytes();
    let (ps, pd) = (ctx.ptr(src)?, ctx.ptr(dst)?);
    // SAFETY: `dst_row + rows <= dst_rows`, so the destination range lies
    // inside `dst`'s storage, which `ptr` checked covers its shape.
    let pd = unsafe { pd.cast::<u8>().add(dst_row * cols * element) };
    ctx.copy_on_device(pd.cast(), ps, rows * cols * element)
}
