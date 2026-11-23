//! GEMM operations — `aclnnMatmul`, `aclnnFusedMatmul`, and `aclnnGeGlu`.
//!
//! Dispatch model mirrors `apxinf_cuda_new::ops::gemm`: semantic functions
//! (`gemm`, `gemm_bias`, `gemm_geglu`) split fused combinations at compile
//! time, while `GemmPolicy.impl_preference` selects the underlying aclnn
//! implementation at runtime (tactic dispatch).
//!
//! - `AclnnMatmul` (default): graph-capture safe, used by `bf16.rs`.
//! - `AclnnFusedMatmul`: fused matmul + bias + activation, used by
//!   the fused-op BF16 reference model to match the torch_npu FULL_FUSION path.

use apxinf_core::{Error, Result, Tensor};

use crate::l3::context::AscendNewContext;

// ── Policy / quantization types ───────────────────────────────────────

/// Underlying aclnn implementation tactic for a GEMM semantic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemmImpl {
    /// `aclnnMatmul` — standard interface, graph-capture safe (default).
    AclnnMatmul,
    /// `aclnnFusedMatmul` — fused matmul + bias + activation, matches the
    /// torch_npu FULL_FUSION path. Carries bias/activation via `GemmBiasArgs`.
    AclnnFusedMatmul,
}

impl Default for GemmImpl {
    fn default() -> Self {
        Self::AclnnMatmul
    }
}

/// GEMM tactic policy (simplified — no autotuning on Ascend).
#[derive(Clone, Debug)]
pub struct GemmPolicy {
    pub accumulation_dtype: apxinf_core::DType,
    pub workspace_limit: usize,
    pub online_tune: bool,
    pub allow_fallback: bool,
    pub graph_safe: bool,
    pub deterministic: bool,
    pub cache_dir: Option<String>,
    /// Underlying aclnn implementation tactic. `bf16.rs` leaves this at
    /// default (`AclnnMatmul`); the fused-op BF16 reference model sets `AclnnFusedMatmul`
    /// to use the fused matmul + bias path.
    pub impl_preference: GemmImpl,
}

impl Default for GemmPolicy {
    fn default() -> Self {
        Self {
            accumulation_dtype: apxinf_core::DType::F32,
            workspace_limit: 256 * 1024 * 1024,
            online_tune: false,
            allow_fallback: true,
            graph_safe: true,
            deterministic: false,
            cache_dir: None,
            impl_preference: GemmImpl::AclnnMatmul,
        }
    }
}

/// Weight version tag for immutable-weight caching.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WeightVersion(u64);

impl WeightVersion {
    pub fn new(v: u64) -> Self {
        Self(v)
    }
    pub fn get(&self) -> u64 {
        self.0
    }
}

/// Quantization mode for GEMM (only `None` is implemented for BF16).
#[derive(Clone, Debug)]
pub enum GemmQuantization<'a> {
    None,
    Fp8 {
        row_scales: &'a Tensor,
        channel_scales: &'a Tensor,
    },
    W8A8 {
        row_scales: &'a Tensor,
        channel_scales: &'a Tensor,
    },
    Fp8UnitScale,
    Nvfp4 {
        a_block_scales: &'a Tensor,
        b_block_scales: &'a Tensor,
        block_size: u32,
    },
}

impl Default for GemmQuantization<'_> {
    fn default() -> Self {
        Self::None
    }
}

// ── Args structs ──────────────────────────────────────────────────────

/// Arguments for a plain GEMM: `out = a @ b`.
pub struct GemmArgs<'a> {
    pub a: &'a Tensor,
    pub b: &'a Tensor,
    pub out: &'a mut Tensor,
    pub quantization: GemmQuantization<'a>,
    pub alpha: f32,
    pub output_scale: f32,
    pub policy: GemmPolicy,
    pub weight_version: Option<WeightVersion>,
}

impl<'a> GemmArgs<'a> {
    pub fn new(a: &'a Tensor, b: &'a Tensor, out: &'a mut Tensor) -> Self {
        Self {
            a,
            b,
            out,
            quantization: GemmQuantization::None,
            alpha: 1.0,
            output_scale: 1.0,
            policy: GemmPolicy::default(),
            weight_version: None,
        }
    }

    pub fn with_immutable_weight(self, version: WeightVersion) -> Self {
        Self {
            weight_version: Some(version),
            ..self
        }
    }
}

/// Arguments for fused GEMM + GeGLU.
pub struct GemmGegluArgs<'a> {
    pub gemm: GemmArgs<'a>,
}

/// Arguments for fused GEMM + bias: `out = a @ b + bias`.
///
/// Under `AclnnFusedMatmul`, bias is fused into the matmul kernel (no separate
/// `aclnnAdd` launch). Under `AclnnMatmul`, this falls back to `aclnnMatmul`
/// followed by `aclnnAdd` (2 launches).
pub struct GemmBiasArgs<'a> {
    pub gemm: GemmArgs<'a>,
    pub bias: &'a Tensor,
}

/// Fused-op-type string tags for `aclnnFusedMatmul`.
///
/// Empty string = no activation (just matmul + bias). These must match the
/// CANN `aclnnFusedMatmul` documented tags.
const FUSED_OP_NONE: &str = "";
const FUSED_OP_GELU: &str = "Gelu";
const FUSED_OP_GELU_TANH: &str = "GeluTanh";

// ── Implementations ───────────────────────────────────────────────────

/// Plain GEMM via `aclnnMatmul` (default) or `aclnnFusedMatmul` (no bias,
/// no activation — same result, but lets the fused-op BF16 reference model route through
/// the fused kernel for consistency with the torch_npu FULL_FUSION path).
pub fn gemm(ctx: &AscendNewContext, args: GemmArgs<'_>) -> Result<()> {
    match args.policy.impl_preference {
        GemmImpl::AclnnMatmul => gemm_matmul(ctx, args.a, args.b, args.out),
        GemmImpl::AclnnFusedMatmul => {
            // No bias, no activation: pass NULL bias + empty fused_op_type.
            gemm_fused(ctx, args.a, args.b, None, FUSED_OP_NONE, args.out)
        }
    }
}

/// `gemm_bias`: `out = a @ b + bias`.
///
/// Dispatches by `GemmImpl`:
/// - `AclnnMatmul` → `aclnnMatmul` + `aclnnAdd` (2 launches, graph-safe).
/// - `AclnnFusedMatmul` → `aclnnFusedMatmul` with bias (1 launch).
pub fn gemm_bias(ctx: &AscendNewContext, args: GemmBiasArgs<'_>) -> Result<()> {
    match args.gemm.policy.impl_preference {
        GemmImpl::AclnnMatmul => {
            gemm_matmul(ctx, args.gemm.a, args.gemm.b, args.gemm.out)?;
            // out += bias via aclnnAdd
            let stream = ctx.raw_stream();
            let dtype = args.gemm.out.dtype();
            let acl_out = crate::acl_tensor::AclTensor::from_device(
                device_ptr(args.gemm.out)?,
                args.gemm.out.shape(),
                dtype,
            );
            let acl_bias = crate::acl_tensor::AclTensor::from_device(
                device_ptr(args.bias)?,
                args.bias.shape(),
                dtype,
            );
            let acl_alpha = crate::acl_tensor::AclScalar::f32(1.0);
            crate::acl_tensor::run_op(
                |ws, exec| unsafe {
                    crate::ffi::aclnn::aclnnAddGetWorkspaceSize(
                        acl_out.raw(),
                        acl_bias.raw(),
                        acl_alpha.raw(),
                        acl_out.raw(),
                        ws,
                        exec,
                    )
                },
                |ws, ws_size, exec, s| unsafe {
                    crate::ffi::aclnn::aclnnAdd(ws, ws_size, exec, s)
                },
                stream,
                "aclnnAdd (gemm_bias)",
            )
            .map_err(|e| Error::Other(format!("gemm_bias add: {e}")))
        }
        GemmImpl::AclnnFusedMatmul => {
            gemm_fused(ctx, args.gemm.a, args.gemm.b, Some(args.bias), FUSED_OP_NONE, args.gemm.out)
        }
    }
}

/// Fused GEMM + GeGLU via `aclnnMatmul` → `aclnnGeGlu`.
///
/// The GEMM produces a [M, 2N] output, then GeGLU splits it into
/// [M, N] = A * Gelu(B).
pub fn gemm_geglu(ctx: &AscendNewContext, args: GemmGegluArgs<'_>) -> Result<()> {
    let stream = ctx.raw_stream();
    let dtype = args.gemm.a.dtype();

    // GEMM output shape: [M, 2N] (gate + up concatenated)
    let a_dims = args.gemm.a.shape().dims();
    let b_dims = args.gemm.b.shape().dims();
    let m = a_dims[a_dims.len() - 2];
    let n = b_dims[b_dims.len() - 1];
    let gemm_out_shape = apxinf_core::Shape::new(vec![m, n]);
    let geglu_out_shape = apxinf_core::Shape::new(vec![m, n / 2]);

    // Allocate intermediate GEMM output [M, 2N]
    let gemm_out = ctx.allocate_output(gemm_out_shape.clone(), dtype)?;

    // Run GEMM
    let a_ptr = device_ptr(args.gemm.a)?;
    let b_ptr = device_ptr(args.gemm.b)?;
    let gemm_out_ptr = device_ptr(&gemm_out)?;

    let acl_a = crate::acl_tensor::AclTensor::from_device(a_ptr, args.gemm.a.shape(), dtype);
    let acl_b = crate::acl_tensor::AclTensor::from_device(b_ptr, args.gemm.b.shape(), dtype);
    let acl_gemm_out = crate::acl_tensor::AclTensor::from_device(gemm_out_ptr, &gemm_out_shape, dtype);

    crate::acl_tensor::run_op(
        |ws, exec| unsafe {
            crate::ffi::aclnn::aclnnMatmulGetWorkspaceSize(
                acl_a.raw(),
                acl_b.raw(),
                acl_gemm_out.raw(),
                0,
                ws,
                exec,
            )
        },
        |ws, ws_size, exec, s| unsafe {
            crate::ffi::aclnn::aclnnMatmul(ws, ws_size, exec, s)
        },
        stream,
        "aclnnMatmul (geglu)",
    )
    .map_err(|e| Error::Other(format!("gemm_geglu matmul: {e}")))?;

    // Run GeGLU: input [M, 2N] → out [M, N], out_gelu [M, N]
    let out_ptr = device_ptr(args.gemm.out)?;
    // Allocate a throwaway out_gelu tensor (we only need the main output)
    let out_gelu = ctx.allocate_output(geglu_out_shape.clone(), dtype)?;
    let out_gelu_ptr = device_ptr(&out_gelu)?;

    let acl_input = crate::acl_tensor::AclTensor::from_device(gemm_out_ptr, &gemm_out_shape, dtype);
    let acl_out = crate::acl_tensor::AclTensor::from_device(out_ptr, &geglu_out_shape, dtype);
    let acl_out_gelu = crate::acl_tensor::AclTensor::from_device(out_gelu_ptr, &geglu_out_shape, dtype);

    crate::acl_tensor::run_op(
        |ws, exec| unsafe {
            crate::ffi::aclnn::aclnnGeGluGetWorkspaceSize(
                acl_input.raw(),
                1, // dim=1 (split along last dim)
                0, // approximate=0 (tanh)
                acl_out.raw(),
                acl_out_gelu.raw(),
                ws,
                exec,
            )
        },
        |ws, ws_size, exec, s| unsafe {
            crate::ffi::aclnn::aclnnGeGlu(ws, ws_size, exec, s)
        },
        stream,
        "aclnnGeGlu",
    )
    .map_err(|e| Error::Other(format!("gemm_geglu geglu: {e}")))?;

    Ok(())
}

// ── Helpers ───────────────────────────────────────────────────────────

/// `aclnnMatmul` core: `out = a @ b` (no bias, no activation).
fn gemm_matmul(
    ctx: &AscendNewContext,
    a: &Tensor,
    b: &Tensor,
    out: &mut Tensor,
) -> Result<()> {
    let stream = ctx.raw_stream();
    let dtype = a.dtype();

    let acl_a = crate::acl_tensor::AclTensor::from_device(device_ptr(a)?, a.shape(), dtype);
    let acl_b = crate::acl_tensor::AclTensor::from_device(device_ptr(b)?, b.shape(), dtype);
    let acl_out = crate::acl_tensor::AclTensor::from_device(device_ptr(out)?, out.shape(), dtype);

    crate::acl_tensor::run_op(
        |ws, exec| unsafe {
            crate::ffi::aclnn::aclnnMatmulGetWorkspaceSize(
                acl_a.raw(),
                acl_b.raw(),
                acl_out.raw(),
                0, // cube_math_type: 0 = default
                ws,
                exec,
            )
        },
        |ws, ws_size, exec, s| unsafe {
            crate::ffi::aclnn::aclnnMatmul(ws, ws_size, exec, s)
        },
        stream,
        "aclnnMatmul",
    )
    .map_err(|e| Error::Other(format!("gemm_matmul: {e}")))
}

/// `aclnnFusedMatmul` core: `out = a @ b + bias` with optional activation.
///
/// `bias` is None → NULL pointer (no bias). `fused_op` selects activation
/// (empty string = none). `x3` (extra operand for some activations) is NULL.
fn gemm_fused(
    ctx: &AscendNewContext,
    a: &Tensor,
    b: &Tensor,
    bias: Option<&Tensor>,
    fused_op: &str,
    out: &mut Tensor,
) -> Result<()> {
    let stream = ctx.raw_stream();
    let dtype = a.dtype();

    let acl_a = crate::acl_tensor::AclTensor::from_device(device_ptr(a)?, a.shape(), dtype);
    let acl_b = crate::acl_tensor::AclTensor::from_device(device_ptr(b)?, b.shape(), dtype);
    let acl_out = crate::acl_tensor::AclTensor::from_device(device_ptr(out)?, out.shape(), dtype);
    let acl_bias = bias.map(|t| {
        crate::acl_tensor::AclTensor::from_device(device_ptr(t).unwrap_or(0), t.shape(), dtype)
    });
    let fused_op_c = std::ffi::CString::new(fused_op).map_err(|_| Error::Other("fused_op contains NUL".into()))?;

    crate::acl_tensor::run_op(
        |ws, exec| unsafe {
            crate::ffi::aclnn::aclnnFusedMatmulGetWorkspaceSize(
                acl_a.raw(),
                acl_b.raw(),
                acl_bias.as_ref().map_or(std::ptr::null(), |t| t.raw()),
                std::ptr::null(), // x3 (extra operand, NULL for none/gelu)
                fused_op_c.as_ptr(),
                0, // cube_math_type: 0 = default
                acl_out.raw(),
                ws,
                exec,
            )
        },
        |ws, ws_size, exec, s| unsafe {
            crate::ffi::aclnn::aclnnFusedMatmul(ws, ws_size, exec, s)
        },
        stream,
        "aclnnFusedMatmul",
    )
    .map_err(|e| Error::Other(format!("gemm_fused: {e}")))
}

fn device_ptr(tensor: &Tensor) -> Result<usize> {
    match tensor.storage() {
        apxinf_core::Storage::Gpu { handle, .. } => Ok(handle.ptr()),
        _ => Err(Error::Other(format!(
            "expected NPU tensor, got {:?}",
            tensor.device()
        ))),
    }
}
