//! Fixed-shape mixture-of-experts helpers: the token router and the routing-
//! weighted SwiGLU. Both are direct launches with no recipe, tuning or
//! persistent state; enqueue allocates only the caller-visible output through
//! [`CudaContext::allocate_output`], so they run eagerly, inside an
//! `ExecutionSession`, and under CUDA Graph capture.

use apxinf_core::{DType, Error, Result, Shape, Tensor};

use crate::ffi::abi::moe as abi;
use crate::ops::validation::check_cuda;
use crate::{CudaBuffer, CudaContext};

/// Largest expert count, top-k and token count the router kernel supports.
pub const MOE_ROUTE_MAX_EXPERTS: usize = 64;
pub const MOE_ROUTE_MAX_TOP_K: usize = 8;
pub const MOE_ROUTE_MAX_TOKENS: usize = 4096;

fn invalid(message: impl Into<String>) -> Error {
    Error::Other(message.into())
}

fn storage(
    ctx: &CudaContext,
    tensor: &Tensor,
    name: &str,
    dtype: DType,
    dims: &[usize],
    alignment: usize,
) -> Result<CudaBuffer> {
    if tensor.dtype() != dtype || tensor.shape().dims() != dims {
        return Err(invalid(format!(
            "MoE {name} must be {dtype} {dims:?}, got {} {:?}",
            tensor.dtype(),
            tensor.shape().dims()
        )));
    }
    if tensor.device() != apxinf_core::Device::Cuda(ctx.device_id()) {
        return Err(invalid(format!(
            "MoE {name} is not on CUDA{}",
            ctx.device_id()
        )));
    }
    let buffer = CudaBuffer::from_tensor(tensor).map_err(Error::Cuda)?;
    let bytes = dims.iter().product::<usize>() * dtype.size_in_bytes();
    if buffer.len() < bytes {
        return Err(invalid(format!(
            "MoE {name} storage is smaller than its shape"
        )));
    }
    if !(buffer.ptr() as usize).is_multiple_of(alignment) {
        return Err(invalid(format!(
            "MoE {name} must be {alignment}-byte aligned"
        )));
    }
    Ok(buffer)
}

fn overlaps(a: &CudaBuffer, b: &CudaBuffer) -> bool {
    let (a0, b0) = (a.ptr() as usize, b.ptr() as usize);
    a0 < b0.saturating_add(b.len()) && b0 < a0.saturating_add(a.len())
}

fn dim_i32(value: usize, what: &str) -> Result<i32> {
    i32::try_from(value).map_err(|_| invalid(format!("MoE {what} exceeds the CUDA kernel range")))
}

/// Token-choice router with sigmoid scores.
///
/// `x` is BF16 `[T, D]`, `gate` BF16 `[E, D]`, `bias` F32 `[E]`. For every
/// token, `logit[e] = dot(x[t], gate[e])` (FP32 accumulation over BF16
/// products), `score = sigmoid(logit)`, and the `top_k` experts with the
/// largest `score + bias` are selected, ties going to the lower index. The
/// bias steers selection only. The selected scores are renormalised,
/// `w = score / (sum(selected scores) + 1e-20) * routed_scale`, and the result
/// is the dense F32 `[T, E]` matrix holding `w` at selected experts and 0
/// elsewhere. Requires `E <= 64`, `top_k <= min(8, E)`, `D % 8 == 0`,
/// `T <= 4096` and 16-byte aligned `x` and `gate`.
pub fn moe_route(
    ctx: &CudaContext,
    x: &Tensor,
    gate: &Tensor,
    bias: &Tensor,
    top_k: usize,
    routed_scale: f32,
) -> Result<Tensor> {
    let [tokens, dim] = *x.shape().dims() else {
        return Err(invalid("MoE router input must be a BF16 [T, D] matrix"));
    };
    let [experts, gate_dim] = *gate.shape().dims() else {
        return Err(invalid("MoE router gate must be a BF16 [E, D] matrix"));
    };
    if gate_dim != dim
        || tokens == 0
        || tokens > MOE_ROUTE_MAX_TOKENS
        || experts == 0
        || experts > MOE_ROUTE_MAX_EXPERTS
        || dim == 0
        || dim % 8 != 0
        || top_k == 0
        || top_k > MOE_ROUTE_MAX_TOP_K
        || top_k > experts
        || routed_scale.is_nan()
    {
        return Err(invalid(format!(
            "MoE router needs 1 <= T <= {MOE_ROUTE_MAX_TOKENS}, 1 <= E <= {MOE_ROUTE_MAX_EXPERTS}, D % 8 == 0 and 1 <= top_k <= min({MOE_ROUTE_MAX_TOP_K}, E); \
             got T={tokens} E={experts} D={dim} top_k={top_k} (gate D={gate_dim}, scale={routed_scale})"
        )));
    }
    let x_buffer = storage(ctx, x, "input", DType::BF16, &[tokens, dim], 16)?;
    let gate_buffer = storage(ctx, gate, "gate", DType::BF16, &[experts, dim], 16)?;
    let bias_buffer = storage(ctx, bias, "bias", DType::F32, &[experts], 4)?;
    let out = ctx.allocate_output(Shape::new(vec![tokens, experts]), DType::F32)?;
    let out_buffer = CudaBuffer::from_tensor(&out).map_err(Error::Cuda)?;
    check_cuda(unsafe {
        abi::apxinf_moe_route(
            x_buffer.ptr(),
            gate_buffer.ptr(),
            bias_buffer.ptr(),
            out_buffer.ptr(),
            dim_i32(tokens, "token count")?,
            dim_i32(experts, "expert count")?,
            dim_i32(dim, "width")?,
            dim_i32(top_k, "top_k")?,
            routed_scale,
            ctx.stream().handle(),
        )
    })?;
    Ok(out)
}

/// Routing-weighted SwiGLU over the concatenated expert intermediates.
///
/// `gate` and `up` are BF16 `[T, N]` with `N = E * I` (expert `e` owns columns
/// `e*I .. (e+1)*I`), `weights` is F32 `[T, E]`. The BF16 `[T, N]` result is
/// `silu(g) * u * w[t, c / I]` with every operation in FP32 and a single final
/// rounding. Multiplying each expert's block by its routing weight (zero for
/// unselected experts) lets one K = `E*I` down-projection sum the experts, so
/// the dense-masked MoE equals sparse dispatch. Inputs and output must not
/// overlap; an 8-element vector path is used when `N`, `I` and the pointers
/// are 8-element/16-byte aligned, otherwise a scalar path.
pub fn moe_swiglu_scaled(
    ctx: &CudaContext,
    gate: &Tensor,
    up: &Tensor,
    weights: &Tensor,
) -> Result<Tensor> {
    let [tokens, columns] = *gate.shape().dims() else {
        return Err(invalid("MoE SwiGLU gate must be a BF16 [T, N] matrix"));
    };
    let [weight_tokens, experts] = *weights.shape().dims() else {
        return Err(invalid("MoE SwiGLU weights must be an F32 [T, E] matrix"));
    };
    if tokens == 0
        || columns == 0
        || experts == 0
        || weight_tokens != tokens
        || columns % experts != 0
    {
        return Err(invalid(format!(
            "MoE SwiGLU needs non-empty [T, N] gate/up and [T, E] weights with N % E == 0; got gate {:?} weights {:?}",
            gate.shape().dims(),
            weights.shape().dims()
        )));
    }
    let gate_buffer = storage(ctx, gate, "gate", DType::BF16, &[tokens, columns], 2)?;
    let up_buffer = storage(ctx, up, "up", DType::BF16, &[tokens, columns], 2)?;
    let weight_buffer = storage(ctx, weights, "weights", DType::F32, &[tokens, experts], 4)?;
    let out = ctx.allocate_output(Shape::new(vec![tokens, columns]), DType::BF16)?;
    let out_buffer = CudaBuffer::from_tensor(&out).map_err(Error::Cuda)?;
    if [&gate_buffer, &up_buffer, &weight_buffer]
        .iter()
        .any(|input| overlaps(input, &out_buffer))
    {
        return Err(invalid("MoE SwiGLU output must not overlap its inputs"));
    }
    tokens
        .checked_mul(columns)
        .filter(|&n| n <= i32::MAX as usize)
        .ok_or_else(|| invalid("MoE SwiGLU element count exceeds the CUDA kernel range"))?;
    check_cuda(unsafe {
        abi::apxinf_moe_swiglu_scaled(
            gate_buffer.ptr(),
            up_buffer.ptr(),
            weight_buffer.ptr(),
            out_buffer.ptr(),
            dim_i32(tokens, "token count")?,
            dim_i32(columns, "width")?,
            dim_i32(experts, "expert count")?,
            ctx.stream().handle(),
        )
    })?;
    Ok(out)
}
