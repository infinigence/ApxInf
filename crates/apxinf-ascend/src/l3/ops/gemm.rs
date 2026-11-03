//! GEMM operations — `aclnnMatmul` and `aclnnGeGlu`.

use apxinf_core::{Error, Result, Tensor};

use crate::l3::context::AscendNewContext;

// ── Policy / quantization types ───────────────────────────────────────

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

// ── Implementations ───────────────────────────────────────────────────

/// Plain GEMM via `aclnnMatmul`.
pub fn gemm(ctx: &AscendNewContext, args: GemmArgs<'_>) -> Result<()> {
    let stream = ctx.raw_stream();
    let dtype = args.a.dtype();

    let a_ptr = device_ptr(args.a)?;
    let b_ptr = device_ptr(args.b)?;
    let out_ptr = device_ptr(args.out)?;

    let acl_a = crate::acl_tensor::AclTensor::from_device(a_ptr, args.a.shape(), dtype);
    let acl_b = crate::acl_tensor::AclTensor::from_device(b_ptr, args.b.shape(), dtype);
    let acl_out = crate::acl_tensor::AclTensor::from_device(out_ptr, args.out.shape(), dtype);

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
    .map_err(|e| Error::Other(format!("gemm: {e}")))?;

    Ok(())
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

fn device_ptr(tensor: &Tensor) -> Result<usize> {
    match tensor.storage() {
        apxinf_core::Storage::Gpu { handle, .. } => Ok(handle.ptr()),
        _ => Err(Error::Other(format!(
            "expected NPU tensor, got {:?}",
            tensor.device()
        ))),
    }
}
