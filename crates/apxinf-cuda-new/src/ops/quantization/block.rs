use apxinf_core::{Result, Tensor};

use crate::CudaContext;

/// Physical layout used for per-block scale values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScaleLayout {
    /// `[rows, k / block_size]`, as stored by checkpoints and consumed by
    /// single-token GEMV.
    RowMajor,
    /// Tensor-core atom layout consumed by block-scaled GEMM candidates.
    GemmAtom,
}

/// Mathematical producer whose output is block-quantized.
#[derive(Clone, Copy)]
pub enum BlockQuantizationSemantic<'a> {
    Activation,
    RmsNorm { weight: &'a Tensor, epsilon: f32 },
    SwiGlu,
}

/// Block quantization into caller-owned value and scale tensors.
///
/// The concrete packed dtype is inferred from `output` and `scales`; the
/// semantic producer and scale layout are data, not separate L3 names.
pub struct BlockQuantizationArgs<'a> {
    pub semantic: BlockQuantizationSemantic<'a>,
    pub input: &'a Tensor,
    pub output: &'a Tensor,
    pub scales: &'a Tensor,
    pub input_scale: f32,
    pub block_size: u32,
    pub scale_layout: ScaleLayout,
}

impl<'a> BlockQuantizationArgs<'a> {
    pub fn activation(
        input: &'a Tensor,
        output: &'a Tensor,
        scales: &'a Tensor,
        input_scale: f32,
        block_size: u32,
        scale_layout: ScaleLayout,
    ) -> Self {
        Self {
            semantic: BlockQuantizationSemantic::Activation,
            input,
            output,
            scales,
            input_scale,
            block_size,
            scale_layout,
        }
    }

    pub fn rms_norm(
        input: &'a Tensor,
        weight: &'a Tensor,
        output: &'a Tensor,
        scales: &'a Tensor,
        epsilon: f32,
        input_scale: f32,
        block_size: u32,
        scale_layout: ScaleLayout,
    ) -> Self {
        Self {
            semantic: BlockQuantizationSemantic::RmsNorm { weight, epsilon },
            input,
            output,
            scales,
            input_scale,
            block_size,
            scale_layout,
        }
    }

    pub fn swiglu(
        input: &'a Tensor,
        output: &'a Tensor,
        scales: &'a Tensor,
        input_scale: f32,
        block_size: u32,
        scale_layout: ScaleLayout,
    ) -> Self {
        Self {
            semantic: BlockQuantizationSemantic::SwiGlu,
            input,
            output,
            scales,
            input_scale,
            block_size,
            scale_layout,
        }
    }
}

pub fn block_quantize(ctx: &CudaContext, args: BlockQuantizationArgs<'_>) -> Result<()> {
    use crate::ops::gemm::nvfp4_scales as launch;

    match args.semantic {
        BlockQuantizationSemantic::Activation => launch::nvfp4_quantize_activation(
            ctx,
            args.input,
            args.output,
            args.scales,
            args.input_scale,
            args.block_size,
            args.scale_layout,
        ),
        BlockQuantizationSemantic::RmsNorm { weight, epsilon } => launch::nvfp4_quantize_rms_norm(
            ctx,
            args.input,
            weight,
            args.output,
            args.scales,
            epsilon,
            args.input_scale,
            args.block_size,
            args.scale_layout,
        ),
        BlockQuantizationSemantic::SwiGlu => launch::nvfp4_quantize_swiglu(
            ctx,
            args.input,
            args.output,
            args.scales,
            args.input_scale,
            args.block_size,
            args.scale_layout,
        ),
    }
}

pub struct PackBlockScalesArgs<'a> {
    pub source: &'a Tensor,
    pub output: &'a Tensor,
    pub rows: usize,
    pub k: usize,
    pub block_size: u32,
}

pub fn block_scale_buffer_bytes(rows: usize, k: usize, block_size: u32) -> Result<usize> {
    crate::ops::gemm::nvfp4_scales::nvfp4_scale_buffer_bytes(rows, k, block_size)
}

pub fn pack_block_scales(ctx: &CudaContext, args: PackBlockScalesArgs<'_>) -> Result<()> {
    crate::ops::gemm::nvfp4_scales::nvfp4_pack_block_scales(
        ctx,
        args.source,
        args.output,
        args.rows,
        args.k,
        args.block_size,
    )
}
