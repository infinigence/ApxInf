//! Safe CUDA compositions shared by the GPT-SoVITS components.

use apxinf_core::{DType, Error, Result, Tensor};
use apxinf_cuda_new::ops::{
    self, tts, AttentionArgs, BiasOrder, Conv1dAccumulation, Conv1dArgs, Conv1dSemantic, GemmArgs,
    NormArgs, PointwiseArgs, PointwiseSemantic, RelativeAttentionArgs, ResidualAddArgs,
    WeightVersion,
};
use apxinf_cuda_new::CudaContext as Context;
use half::{bf16, f16};

pub(crate) const DTYPE: DType = DType::F16;

pub(crate) fn alloc(ctx: &Context, dims: &[usize]) -> Result<Tensor> {
    ctx.allocate_output(apxinf_core::Shape::new(dims.to_vec()), DTYPE)
}

pub(crate) fn gemm(ctx: &Context, a: &Tensor, b: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    let a_dims = a.shape().dims();
    let b_dims = b.shape().dims();
    if a_dims.len() != 2 || b_dims.len() != 2 || a_dims[1] != b_dims[0] {
        return Err(Error::Other(format!(
            "gpt_sovits gemm shape mismatch: a={a_dims:?} b={b_dims:?}"
        )));
    }
    let expected_device = apxinf_core::Device::Cuda(ctx.device_id());
    if a.device() != expected_device || b.device() != expected_device {
        return Err(Error::Other(format!(
            "gemm device mismatch: a {:?}, b {:?}, expected {:?}",
            a.device(),
            b.device(),
            expected_device
        )));
    }
    if a.dtype() != DTYPE || b.dtype() != DTYPE {
        return Err(Error::Other(format!(
            "gemm dtype mismatch: a {:?} {:?}, b {:?} {:?}",
            a.dtype(),
            a_dims,
            b.dtype(),
            b_dims
        )));
    }
    if let Some(bias) = bias {
        if bias.dtype() != DTYPE
            || bias.device() != expected_device
            || bias.shape().dims() != [b_dims[1]]
        {
            return Err(Error::Other(format!(
                "gemm bias mismatch: {:?} {:?} for n={}",
                bias.dtype(),
                bias.shape().dims(),
                b_dims[1]
            )));
        }
    }
    let mut out = alloc(ctx, &[a_dims[0], b_dims[1]])?;
    if out.device() != expected_device
        || out.dtype() != DTYPE
        || out.shape().dims() != [a_dims[0], b_dims[1]]
    {
        return Err(Error::Other(format!(
            "gemm output mismatch: {:?} {:?} {:?}",
            out.device(),
            out.dtype(),
            out.shape().dims()
        )));
    }
    let out_dims = out.shape().dims().to_vec();
    let bias_dims = bias.map(|tensor| tensor.shape().dims().to_vec());
    let context = |error: Error| {
        Error::Other(format!(
            "gemm a={a_dims:?} b={b_dims:?} out={out_dims:?} bias={bias_dims:?}: {error}"
        ))
    };
    match bias {
        Some(bias) => ops::gemm_bias(
            ctx,
            ops::GemmBiasArgs {
                gemm: GemmArgs::new(a, b, &mut out),
                bias,
            },
        )
        .map_err(context)?,
        None => ops::gemm(ctx, GemmArgs::new(a, b, &mut out)).map_err(context)?,
    }
    Ok(out)
}

/// F16 linear using Torch's CUDA `F.linear` -> `aten::addmm` bias-before-cast
/// semantic. GPT-SoVITS routes every FP16 bias-carrying linear through this
/// operation so HuBERT, GPT, SoVITS, BERT, and CFM share one numerical contract.
pub(crate) fn linear_bias_accumulator(
    ctx: &Context,
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
) -> Result<Tensor> {
    let Some(bias) = bias else {
        return linear(ctx, input, weight, None);
    };
    let a_dims = input.shape().dims();
    let b_dims = weight.shape().dims();
    if a_dims.len() != 2 || b_dims.len() != 2 || a_dims[1] != b_dims[0] {
        return Err(Error::Other(format!(
            "gpt_sovits accumulator linear shape mismatch: input={a_dims:?} weight={b_dims:?}"
        )));
    }
    linear_bias_accumulator_with_workspace(ctx, input, weight, bias, None)
}

pub(crate) fn linear_bias_accumulator_with_workspace(
    ctx: &Context,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    workspace_limit: Option<usize>,
) -> Result<Tensor> {
    let a_dims = input.shape().dims().to_vec();
    let b_dims = weight.shape().dims().to_vec();
    if a_dims.len() != 2 || b_dims.len() != 2 || a_dims[1] != b_dims[0] {
        return Err(Error::Other("accumulator linear shape mismatch".into()));
    }
    if input.dtype() != DTYPE || weight.dtype() != DTYPE || bias.dtype() != DTYPE {
        return Err(Error::Other(
            "accumulator linear requires F16 input, weight, and bias".into(),
        ));
    }
    // The fused GEMM epilogue loads the F16 projection into FP32, adds the
    // F16 bias there, and rounds once on the final write, matching Torch's
    // addmm bias handling at the F16 output boundary.
    let mut out = alloc(ctx, &[a_dims[0], b_dims[1]])?;
    let mut gemm = GemmArgs::new(input, weight, &mut out);
    // These projections consume loaded checkpoint weights, immutable for the
    // model lifetime. Permit provider-owned preparation through the existing
    // allocation/version contract without choosing a provider or configuration.
    gemm = gemm.with_immutable_weight(WeightVersion::new(1));
    if let Some(limit) = workspace_limit {
        gemm.policy.workspace_limit = limit;
    }
    ops::gemm_bias(ctx, ops::GemmBiasArgs { gemm, bias })?;
    Ok(out)
}

/// Output frame count produced by the official `speed_factor` path in
/// `TextEncoder.forward`: `int(frames / speed) + 1`. The `+1` is part of the
/// reference implementation and is preserved verbatim.
pub(crate) fn speed_output_frames(in_frames: usize, speed_factor: f32) -> usize {
    ((in_frames as f32 / speed_factor).floor() as usize) + 1
}

/// Linear interpolation of row positions, matching PyTorch
/// `F.interpolate(x, size=out_rows, mode="linear")` with the default
/// `align_corners=False` mapping. The input is `[rows, cols]`; columns are
/// independent. Computation is FP32 and the result is rounded back to the
/// input dtype, matching the CUDA upsample kernel's `accscalar_t` contract.
pub(crate) fn linear_interpolate_rows(
    ctx: &Context,
    input: &Tensor,
    out_rows: usize,
) -> Result<Tensor> {
    let dims = input.shape().dims();
    if dims.len() != 2 {
        return Err(Error::Other(format!(
            "linear_interpolate_rows expects [rows, cols], got {dims:?}"
        )));
    }
    let in_rows = dims[0];
    let cols = dims[1];
    if in_rows == 0 || cols == 0 || out_rows == 0 {
        return Err(Error::Other(
            "linear_interpolate_rows needs non-empty input and output".into(),
        ));
    }
    if in_rows == out_rows {
        return Ok(input.clone());
    }
    let cpu = apxinf_cuda_new::transfers::to_cpu(input)?;
    let dtype = cpu.dtype();
    let values = cpu.to_f32_vec()?;
    let output = interpolate_rows_f32(&values, in_rows, cols, out_rows);
    let host = match dtype {
        DType::F16 => {
            let data = output
                .iter()
                .map(|value| f16::from_f32(*value))
                .collect::<Vec<_>>();
            Tensor::from_f16(vec![out_rows, cols], &data)?
        }
        DType::BF16 => {
            let data = output
                .iter()
                .map(|value| bf16::from_f32(*value))
                .collect::<Vec<_>>();
            Tensor::from_bf16(vec![out_rows, cols], &data)?
        }
        DType::F32 => Tensor::from_f32(vec![out_rows, cols], &output)?,
        other => {
            return Err(Error::Other(format!(
                "linear_interpolate_rows does not support dtype {other}"
            )))
        }
    };
    apxinf_cuda_new::transfers::to_cuda(&host, ctx.device_id())
}

/// PyTorch-compatible `align_corners=False` row interpolation on an
/// `[rows, cols]` FP32 buffer. Kept free of CUDA so the mapping can be
/// unit-tested on hosts without a GPU.
pub(crate) fn interpolate_rows_f32(
    values: &[f32],
    in_rows: usize,
    cols: usize,
    out_rows: usize,
) -> Vec<f32> {
    assert!(in_rows > 0 && cols > 0 && out_rows > 0);
    assert_eq!(values.len(), in_rows * cols);
    let scale = in_rows as f32 / out_rows as f32;
    let last = (in_rows - 1) as f32;
    let mut output = vec![0.0f32; out_rows * cols];
    for out_row in 0..out_rows {
        let mut source = scale * (out_row as f32 + 0.5) - 0.5;
        if source < 0.0 {
            source = 0.0;
        } else if source > last {
            source = last;
        }
        let lower = source.floor() as usize;
        let upper = (lower + 1).min(in_rows - 1);
        let weight = source - lower as f32;
        let lower_weight = 1.0f32 - weight;
        let lower_base = lower * cols;
        let upper_base = upper * cols;
        let output_base = out_row * cols;
        for column in 0..cols {
            output[output_base + column] =
                lower_weight * values[lower_base + column] + weight * values[upper_base + column];
        }
    }
    output
}

/// Linear over the last dimension of a contiguous `[rows, cols]` matrix.
pub(crate) fn linear(
    ctx: &Context,
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
) -> Result<Tensor> {
    gemm(ctx, input, weight, bias)
}

/// Linear over the last dimension of `[batch, time, channels]`.
pub(crate) fn linear_seq(
    ctx: &Context,
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
) -> Result<Tensor> {
    let dims = input.shape().dims().to_vec();
    if dims.len() != 3 {
        return Err(Error::Other("linear_seq expects rank-3 input".into()));
    }
    let rows = dims[0] * dims[1];
    let flat = input.reshape(apxinf_core::Shape::new(vec![rows, dims[2]]))?;
    let out = linear(ctx, &flat, weight, bias)?;
    out.reshape(apxinf_core::Shape::new(dims))
}

pub(crate) fn layer_norm(
    ctx: &Context,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let dims = input.shape().dims().to_vec();
    let cols = *dims
        .last()
        .ok_or_else(|| Error::Other("layer norm needs a rank".into()))?;
    let rows = input.numel() / cols;
    let flat = input.reshape(apxinf_core::Shape::new(vec![rows, cols]))?;
    let mut out = alloc(ctx, &[rows, cols])?;
    ops::norm::norm(ctx, NormArgs::layer(&flat, weight, bias, &out, eps))?;
    out.reshape(apxinf_core::Shape::new(dims))
}

pub(crate) fn residual_add(ctx: &Context, input: &Tensor, residual: &Tensor) -> Result<Tensor> {
    let dims = input.shape().dims().to_vec();
    let cols = *dims
        .last()
        .ok_or_else(|| Error::Other("residual needs a rank".into()))?;
    let rows = input.numel() / cols;
    if residual.numel() != input.numel() {
        return Err(Error::Other(format!(
            "residual shape mismatch: {:?} vs {:?}",
            input.shape().dims(),
            residual.shape().dims()
        )));
    }
    let flat = input.reshape(apxinf_core::Shape::new(vec![rows, cols]))?;
    let residual = residual.reshape(apxinf_core::Shape::new(vec![rows, cols]))?;
    let hidden = alloc(ctx, &[rows, cols])?;
    ops::residual_add(
        ctx,
        ResidualAddArgs {
            input: &flat,
            bias: None,
            residual: &residual,
            gate: None,
            hidden: &hidden,
            bias_order: BiasOrder::SingleRound,
        },
    )?;
    hidden.reshape(apxinf_core::Shape::new(dims))
}

pub(crate) fn affine(
    ctx: &Context,
    input: &Tensor,
    scale: &Tensor,
    shift: &Tensor,
) -> Result<Tensor> {
    // Pack scale||shift into the [2 * cols] binding expected by AFFINE.
    let dims = input.shape().dims().to_vec();
    let cols = *dims
        .last()
        .ok_or_else(|| Error::Other("affine needs a rank".into()))?;
    let scale_dims = scale.shape().dims();
    let shift_dims = shift.shape().dims();
    if scale_dims != [cols] || shift_dims != [cols] {
        return Err(Error::Other("affine style width mismatch".into()));
    }
    let combined = ops::concat_rows(
        ctx,
        &scale.reshape(apxinf_core::Shape::new(vec![1, cols]))?,
        &shift.reshape(apxinf_core::Shape::new(vec![1, cols]))?,
    )?;
    let combined = combined.reshape(apxinf_core::Shape::new(vec![2 * cols]))?;
    let rows = input.numel() / cols;
    let flat = input.reshape(apxinf_core::Shape::new(vec![rows, cols]))?;
    let out = alloc(ctx, &[rows, cols])?;
    tts::affine(ctx, &flat, &combined, &out)?;
    out.reshape(apxinf_core::Shape::new(dims))
}

pub(crate) fn gate_residual(
    ctx: &Context,
    input: &Tensor,
    residual: &Tensor,
    gate: &Tensor,
) -> Result<Tensor> {
    let dims = input.shape().dims().to_vec();
    let cols = *dims
        .last()
        .ok_or_else(|| Error::Other("gate needs a rank".into()))?;
    if gate.shape().dims() != [cols] {
        return Err(Error::Other("gate width mismatch".into()));
    }
    let zeros = alloc(ctx, &[cols])?;
    let ones = alloc(ctx, &[cols])?;
    // gate_style = [scale, shift, gate]; only the third segment is read.
    let first = ops::concat_rows(
        ctx,
        &zeros.reshape(apxinf_core::Shape::new(vec![1, cols]))?,
        &ones.reshape(apxinf_core::Shape::new(vec![1, cols]))?,
    )?;
    let second = gate.reshape(apxinf_core::Shape::new(vec![1, cols]))?;
    let style = ops::concat_rows(ctx, &first, &second)?;
    let style = style.reshape(apxinf_core::Shape::new(vec![3 * cols]))?;
    let rows = input.numel() / cols;
    let flat = input.reshape(apxinf_core::Shape::new(vec![rows, cols]))?;
    let residual_flat = residual.reshape(apxinf_core::Shape::new(vec![rows, cols]))?;
    let hidden = alloc(ctx, &[rows, cols])?;
    ops::residual_add(
        ctx,
        ResidualAddArgs {
            input: &flat,
            bias: None,
            residual: &residual_flat,
            gate: Some(ops::ResidualGate::PackedStyle(&style)),
            hidden: &hidden,
            bias_order: BiasOrder::SingleRound,
        },
    )?;
    hidden.reshape(apxinf_core::Shape::new(dims))
}

fn activation(
    ctx: &Context,
    input: &Tensor,
    bias: Option<&Tensor>,
    activation: tts::TtsActivation,
) -> Result<Tensor> {
    let dims = input.shape().dims().to_vec();
    let cols = *dims
        .last()
        .ok_or_else(|| Error::Other("activation needs a rank".into()))?;
    let rows = input.numel() / cols;
    let flat = input.reshape(apxinf_core::Shape::new(vec![rows, cols]))?;
    let out = alloc(ctx, &[rows, cols])?;
    tts::bias_activation(ctx, &flat, bias, &out, activation)?;
    out.reshape(apxinf_core::Shape::new(dims))
}

pub(crate) fn gelu_exact(ctx: &Context, input: &Tensor) -> Result<Tensor> {
    activation(ctx, input, None, tts::TtsActivation::GeluExact)
}

pub(crate) fn mish(ctx: &Context, input: &Tensor) -> Result<Tensor> {
    activation(ctx, input, None, tts::TtsActivation::Mish)
}

pub(crate) fn leaky_relu(ctx: &Context, input: &Tensor) -> Result<Tensor> {
    activation(ctx, input, None, tts::TtsActivation::LeakyRelu)
}

pub(crate) fn leaky_relu_one_percent(ctx: &Context, input: &Tensor) -> Result<Tensor> {
    activation(ctx, input, None, tts::TtsActivation::LeakyReluOnePercent)
}

pub(crate) fn tanh_gate(
    ctx: &Context,
    input: &Tensor,
    secondary: Option<&Tensor>,
) -> Result<Tensor> {
    let dims = input.shape().dims().to_vec();
    let cols = *dims
        .last()
        .ok_or_else(|| Error::Other("gate needs a rank".into()))?;
    if cols % 2 != 0 {
        return Err(Error::Other("gate input width must be even".into()));
    }
    let rows = input.numel() / cols;
    let flat = input.reshape(apxinf_core::Shape::new(vec![rows, cols]))?;
    let secondary_flat = match secondary {
        Some(tensor) => Some(tensor.reshape(apxinf_core::Shape::new(vec![rows, cols]))?),
        None => None,
    };
    let out = alloc(ctx, &[rows, cols / 2])?;
    tts::tanh_sigmoid_gate(ctx, &flat, secondary_flat.as_ref(), &out)?;
    let mut out_dims = dims.clone();
    *out_dims.last_mut().unwrap() /= 2;
    out.reshape(apxinf_core::Shape::new(out_dims))
}

pub(crate) fn glu_separate(ctx: &Context, first: &Tensor, second: &Tensor) -> Result<Tensor> {
    let dims = first.shape().dims().to_vec();
    let cols = *dims.last().unwrap();
    let rows = first.numel() / cols;
    let flat_first = first.reshape(apxinf_core::Shape::new(vec![rows, cols]))?;
    let flat_second = second.reshape(apxinf_core::Shape::new(vec![rows, cols]))?;
    let out = alloc(ctx, &[rows, cols])?;
    tts::glu_separate(ctx, &flat_first, &flat_second, &out)?;
    out.reshape(apxinf_core::Shape::new(dims))
}

pub(crate) fn column_norm(
    ctx: &Context,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
) -> Result<Tensor> {
    let dims = input.shape().dims().to_vec();
    let cols = *dims
        .last()
        .ok_or_else(|| Error::Other("column norm needs a rank".into()))?;
    let rows = input.numel() / cols;
    let flat = input.reshape(apxinf_core::Shape::new(vec![rows, cols]))?;
    let style = ops::concat_rows(
        ctx,
        &weight.reshape(apxinf_core::Shape::new(vec![1, cols]))?,
        &bias.reshape(apxinf_core::Shape::new(vec![1, cols]))?,
    )?;
    let style = style.reshape(apxinf_core::Shape::new(vec![2 * cols]))?;
    let out = alloc(ctx, &[rows, cols])?;
    tts::column_norm(ctx, &flat, &style, &out, 1e-5)?;
    out.reshape(apxinf_core::Shape::new(dims))
}

pub(crate) fn grn(ctx: &Context, input: &Tensor, gamma: &Tensor, beta: &Tensor) -> Result<Tensor> {
    let dims = input.shape().dims().to_vec();
    let cols = *dims
        .last()
        .ok_or_else(|| Error::Other("grn needs a rank".into()))?;
    let rows = input.numel() / cols;
    let flat = input.reshape(apxinf_core::Shape::new(vec![rows, cols]))?;
    let style = ops::concat_rows(
        ctx,
        &gamma.reshape(apxinf_core::Shape::new(vec![1, cols]))?,
        &beta.reshape(apxinf_core::Shape::new(vec![1, cols]))?,
    )?;
    let style = style.reshape(apxinf_core::Shape::new(vec![2 * cols]))?;
    let out = alloc(ctx, &[rows, cols])?;
    tts::grn(ctx, &flat, &style, &out)?;
    out.reshape(apxinf_core::Shape::new(dims))
}

pub(crate) fn embedding(ctx: &Context, table: &Tensor, ids: &Tensor) -> Result<Tensor> {
    let table_dims = table.shape().dims();
    let tokens = ids.shape().dims().first().copied().unwrap_or(0);
    let out = alloc(ctx, &[tokens, table_dims[1]])?;
    // The general CUDA embedding operator is F32/BF16 only. The speech models
    // keep every checkpoint tensor in F16, so gather through the tts adapter.
    tts::embedding_gather(ctx, table, ids, &out)?;
    Ok(out)
}

pub(crate) fn gather_rows(ctx: &Context, table: &Tensor, ids: &Tensor) -> Result<Tensor> {
    embedding(ctx, table, ids)
}

pub(crate) fn attention(
    ctx: &Context,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    heads: usize,
) -> Result<Tensor> {
    let dims = query.shape().dims().to_vec();
    if dims.len() != 4 || dims[2] != heads {
        return Err(Error::Other("attention expects [B,T,H,D]".into()));
    }
    let mut out = alloc(ctx, &dims)?;
    ops::attention(ctx, AttentionArgs::new(query, key, value, &mut out))?;
    Ok(out)
}

pub(crate) fn relative_attention(
    ctx: &Context,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    relative_key: &Tensor,
    relative_value: &Tensor,
    window: usize,
) -> Result<Tensor> {
    let dims = query.shape().dims().to_vec();
    let mut out = alloc(ctx, &dims)?;
    let mut args =
        RelativeAttentionArgs::new(query, key, value, relative_key, relative_value, &mut out);
    args.window = window;
    ops::relative_attention(ctx, args)?;
    Ok(out)
}

pub(crate) fn conv1d(
    ctx: &Context,
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<Tensor> {
    conv1d_impl(
        ctx,
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        Conv1dAccumulation::Default,
    )
}

pub(crate) fn conv1d_ordered(
    ctx: &Context,
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<Tensor> {
    conv1d_impl(
        ctx,
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        Conv1dAccumulation::OrderedF32,
    )
}

pub(crate) fn conv1d_impl(
    ctx: &Context,
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    accumulation: Conv1dAccumulation,
) -> Result<Tensor> {
    let dims = input.shape().dims().to_vec();
    if dims.len() != 3 {
        return Err(Error::Other("conv1d expects [B,T,C]".into()));
    }
    let out_channels = weight.shape().dims()[0];
    let span = dilation * (weight.shape().dims()[2] - 1) + 1;
    if dims[1] + 2 * padding < span {
        return Err(Error::Other(
            "conv1d input is shorter than its kernel".into(),
        ));
    }
    let out_length = (dims[1] + 2 * padding - span) / stride + 1;
    let mut out = alloc(ctx, &[dims[0], out_length, out_channels])?;
    let mut args = Conv1dArgs::new(Conv1dSemantic::Conv1d, input, weight, &mut out);
    if let Some(bias) = bias {
        args = args.with_bias(bias);
    }
    args.stride = stride;
    args.padding = padding;
    args.dilation = dilation;
    args.groups = groups;
    args.accumulation = accumulation;
    ops::conv1d(ctx, args)?;
    Ok(out)
}

pub(crate) fn conv_transpose1d(
    ctx: &Context,
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    padding: usize,
    output_padding: usize,
) -> Result<Tensor> {
    let dims = input.shape().dims().to_vec();
    if dims.len() != 3 {
        return Err(Error::Other("conv_transpose1d expects [B,T,C]".into()));
    }
    let out_channels = weight.shape().dims()[1];
    let kernel = weight.shape().dims()[2];
    let out_length = (dims[1] - 1) * stride - 2 * padding + (kernel - 1) + output_padding + 1;
    let mut out = alloc(ctx, &[dims[0], out_length, out_channels])?;
    let mut args = Conv1dArgs::new(Conv1dSemantic::ConvTranspose1d, input, weight, &mut out);
    if let Some(bias) = bias {
        args = args.with_bias(bias);
    }
    args.stride = stride;
    args.padding = padding;
    args.output_padding = output_padding;
    ops::conv1d(ctx, args)?;
    Ok(out)
}

pub(crate) fn slice_rows(
    ctx: &Context,
    input: &Tensor,
    offset: usize,
    rows: usize,
) -> Result<Tensor> {
    let dims = input.shape().dims().to_vec();
    if dims.len() != 2 || offset + rows > dims[0] {
        return Err(Error::Other("slice_rows range is invalid".into()));
    }
    let out = alloc(ctx, &[rows, dims[1]])?;
    tts::copy_rows(ctx, input, offset, rows, &out)?;
    Ok(out)
}

pub(crate) fn concat_rows(ctx: &Context, first: &Tensor, second: &Tensor) -> Result<Tensor> {
    ops::concat_rows(ctx, first, second)
}

pub(crate) fn concat_columns(ctx: &Context, first: &Tensor, second: &Tensor) -> Result<Tensor> {
    let first_dims = first.shape().dims().to_vec();
    let second_dims = second.shape().dims().to_vec();
    if first_dims.len() != 2 || second_dims.len() != 2 || first_dims[0] != second_dims[0] {
        return Err(Error::Other(
            "concat_columns expects equal-row rank-2 inputs".into(),
        ));
    }
    let out = alloc(ctx, &[first_dims[0], first_dims[1] + second_dims[1]])?;
    tts::concat_columns(ctx, first, second, &out)?;
    Ok(out)
}

pub(crate) fn slice_columns(
    ctx: &Context,
    input: &Tensor,
    column_offset: usize,
    columns: usize,
) -> Result<Tensor> {
    let dims = input.shape().dims().to_vec();
    if dims.len() != 2 || column_offset + columns > dims[1] {
        return Err(Error::Other("slice_columns range is invalid".into()));
    }
    let out = alloc(ctx, &[dims[0], columns])?;
    tts::slice_columns(ctx, input, column_offset, columns, &out)?;
    Ok(out)
}

pub(crate) fn upsample2(ctx: &Context, input: &Tensor) -> Result<Tensor> {
    let dims = input.shape().dims().to_vec();
    if dims.len() != 2 {
        return Err(Error::Other("upsample2 expects [T,C]".into()));
    }
    let mut ids = Vec::with_capacity(dims[0] * 2);
    for index in 0..dims[0] {
        ids.push(index as i32);
        ids.push(index as i32);
    }
    let ids = Tensor::from_i32(vec![ids.len()], &ids)?;
    let ids = apxinf_cuda_new::transfers::to_cuda(&ids, ctx.device_id())?;
    embedding(ctx, input, &ids)
}

pub(crate) fn rope_apply(
    ctx: &Context,
    input: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<Tensor> {
    let dims = input.shape().dims().to_vec();
    let out = alloc(ctx, &dims)?;
    tts::rope_apply(ctx, input, cos, sin, &out)?;
    Ok(out)
}

pub(crate) fn add_bias(ctx: &Context, input: &Tensor, bias: &Tensor) -> Result<Tensor> {
    activation(ctx, input, Some(bias), tts::TtsActivation::None)
}

pub(crate) fn relu(ctx: &Context, input: &Tensor) -> Result<Tensor> {
    activation(ctx, input, None, tts::TtsActivation::Relu)
}

pub(crate) fn silu(ctx: &Context, input: &Tensor) -> Result<Tensor> {
    activation(ctx, input, None, tts::TtsActivation::Silu)
}

pub(crate) fn zeros_like(ctx: &Context, input: &Tensor) -> Result<Tensor> {
    alloc(ctx, input.shape().dims())
}

pub(crate) fn ones_vector(ctx: &Context, cols: usize) -> Result<Tensor> {
    let values = vec![1.0f32; cols];
    super::weights::fp16_tensor(ctx, &[cols], &values)
}

pub(crate) fn mean_rows(ctx: &Context, input: &Tensor) -> Result<Tensor> {
    let dims = input.shape().dims().to_vec();
    let cols = *dims
        .last()
        .ok_or_else(|| Error::Other("mean_rows needs a rank".into()))?;
    let rows = input.numel() / cols;
    let flat = input.reshape(apxinf_core::Shape::new(vec![rows, cols]))?;
    let out = alloc(ctx, &[1, cols])?;
    tts::mean_rows(ctx, &flat, &out)?;
    out.reshape(apxinf_core::Shape::new(vec![cols]))
}

pub(crate) fn fill_zero(ctx: &Context, tensor: &Tensor) -> Result<()> {
    tts::fill_zero(ctx, tensor)
}

pub(crate) fn gelu_tanh(ctx: &Context, input: &Tensor) -> Result<Tensor> {
    activation(ctx, input, None, tts::TtsActivation::Gelu)
}

pub(crate) fn euler_update(
    ctx: &Context,
    state: &Tensor,
    velocity: &Tensor,
    dt: f32,
) -> Result<Tensor> {
    let dims = state.shape().dims().to_vec();
    let cols = *dims
        .last()
        .ok_or_else(|| Error::Other("euler needs a rank".into()))?;
    let rows = state.numel() / cols;
    let flat_state = state.reshape(apxinf_core::Shape::new(vec![rows, cols]))?;
    let flat_velocity = velocity.reshape(apxinf_core::Shape::new(vec![rows, cols]))?;
    let mut out = alloc(ctx, &[rows, cols])?;
    let mut args = PointwiseArgs::new(PointwiseSemantic::EulerUpdate, &flat_state, &mut out);
    args.secondary = Some(&flat_velocity);
    args.dt = dt;
    ops::pointwise(ctx, args)?;
    out.reshape(apxinf_core::Shape::new(dims))
}

pub(crate) fn tanh(ctx: &Context, input: &Tensor) -> Result<Tensor> {
    activation(ctx, input, None, tts::TtsActivation::Tanh)
}

pub(crate) fn sync(ctx: &Context) -> Result<()> {
    ctx.synchronize().map_err(Error::Cuda)
}


#[cfg(all(test, feature = "cuda"))]
mod cuda_reference_tests {
    use super::*;
    use apxinf_cuda_new::CudaContext as Context;

    fn ctx() -> Context {
        Context::new(0).unwrap()
    }
    fn dev(ctx: &Context, shape: &[usize], values: &[f32]) -> Tensor {
        super::super::weights::fp16_tensor(ctx, shape, values).unwrap()
    }
    fn host(tensor: &Tensor) -> Vec<f32> {
        apxinf_cuda_new::transfers::to_cpu(tensor)
            .unwrap()
            .to_f32_vec()
            .unwrap()
    }
    fn check(name: &str, actual: &[f32], expected: &[f32], tol: f32) {
        let error = actual
            .iter()
            .zip(expected)
            .map(|(a, e)| (a - e).abs())
            .fold(0.0f32, f32::max);
        println!("{name}: max_abs={error}");
        assert!(error <= tol, "{name}: max_abs={error}");
    }

    #[test]
    fn linear_interpolate_rows_matches_torch_align_corners_false() {
        let ctx = ctx();
        let values = [
            0.0f32, 1.0, 10.0, 2.0, 3.0, 11.0, 4.0, 5.0, 12.0, 6.0, 7.0, 13.0,
        ];
        let input = dev(&ctx, &[4, 3], &values);
        let out = linear_interpolate_rows(&ctx, &input, 7).unwrap();
        let got = host(&out);
        let expected = interpolate_rows_f32(&values, 4, 3, 7);
        // Inputs/outputs are FP16 here, so values around 10 carry ~4e-3
        // quantization steps; the mapping itself is FP32-exact.
        check("linear_interpolate_rows", &got, &expected, 1e-2);

        // align_corners=False clamps the first/last output rows to the
        // boundary samples rather than extrapolating.
        assert_eq!(&got[0..=2], &values[0..=2]);
        assert_eq!(&got[18..=20], &values[9..=11]);

        // Same shape is an identity fast path.
        let same = linear_interpolate_rows(&ctx, &input, 4).unwrap();
        check(
            "linear_interpolate_rows_identity",
            &host(&same),
            &values,
            0.0,
        );
    }

    #[test]
    fn layer_norm_linear_rope_attention_match_reference() {
        let ctx = ctx();
        let values: Vec<f32> = (0..32)
            .map(|i| (i as f32 * 0.37 - 5.0).sin() * 2.0)
            .collect();
        let input = dev(&ctx, &[4, 8], &values);
        let ones = vec![1.0f32; 8];
        let zeros = vec![0.0f32; 8];
        let w = dev(&ctx, &[8], &ones);
        let b = dev(&ctx, &[8], &zeros);
        let out = layer_norm(&ctx, &input, &w, &b, 1e-6).unwrap();
        let got = host(&out);
        let expected: Vec<f32> = (0..4)
            .map(|r| {
                let x = &values[r * 8..r * 8 + 8];
                let mean = x.iter().sum::<f32>() / 8.0;
                let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / 8.0;
                x.iter()
                    .map(|v| (v - mean) / (var + 1e-6).sqrt())
                    .collect::<Vec<_>>()
            })
            .flatten()
            .collect();
        check("layer_norm", &got, &expected, 2e-2);

        let a = dev(&ctx, &[4, 8], &values);
        let wv: Vec<f32> = (0..8 * 4)
            .map(|i| (i as f32 * 0.23 - 2.0).cos() * 0.3)
            .collect();
        let weight = dev(&ctx, &[8, 4], &wv);
        let out = linear(&ctx, &a, &weight, None).unwrap();
        let got = host(&out);
        let expected: Vec<f32> = (0..4)
            .map(|r| {
                (0..4)
                    .map(|c| {
                        (0..8)
                            .map(|k| values[r * 8 + k] * wv[k * 4 + c])
                            .sum::<f32>()
                    })
                    .collect::<Vec<_>>()
            })
            .flatten()
            .collect();
        check("linear", &got, &expected, 3e-2);

        let len = 4usize;
        let heads = 16usize;
        let dim = 64usize;
        let qv: Vec<f32> = (0..len * heads * dim)
            .map(|i| (i as f32 * 0.017 - 1.0).sin())
            .collect();
        let kv: Vec<f32> = (0..len * heads * dim)
            .map(|i| (i as f32 * 0.029 - 2.0).cos())
            .collect();
        let vv: Vec<f32> = (0..len * heads * dim)
            .map(|i| (i as f32 * 0.041 - 3.0).sin())
            .collect();
        let shape = vec![1, len, heads, dim];
        let q = dev(&ctx, &shape, &qv);
        let k = dev(&ctx, &shape, &kv);
        let v = dev(&ctx, &shape, &vv);
        let out = attention(&ctx, &q, &k, &v, heads).unwrap();
        let got = host(&out);
        let mut expected = vec![0.0f32; qv.len()];
        for t in 0..len {
            for h in 0..heads {
                let mut scores = Vec::with_capacity(len);
                for source in 0..len {
                    let dot = (0..dim)
                        .map(|d| qv[(t * heads + h) * dim + d] * kv[(source * heads + h) * dim + d])
                        .sum::<f32>();
                    scores.push(dot / (dim as f32).sqrt());
                }
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = scores.iter().map(|x| (x - max).exp()).collect();
                let sum = exps.iter().sum::<f32>();
                for d in 0..dim {
                    expected[(t * heads + h) * dim + d] = (0..len)
                        .map(|source| exps[source] / sum * vv[(source * heads + h) * dim + d])
                        .sum();
                }
            }
        }
        check("attention", &got, &expected, 5e-2);

        let cosv: Vec<f32> = (0..len * dim)
            .map(|index| {
                let d = index % dim;
                0.25 + 0.7 * (d as f32 / dim as f32)
            })
            .collect();
        let sinv: Vec<f32> = cosv
            .iter()
            .map(|value| (1.0 - value * value).sqrt())
            .collect();
        let q0 = dev(&ctx, &shape, &qv);
        let c = dev(&ctx, &[len, dim], &cosv);
        let sin = dev(&ctx, &[len, dim], &sinv);
        let roped = rope_apply(&ctx, &q0, &c, &sin).unwrap();
        let got = host(&roped);
        let mut expected = Vec::with_capacity(qv.len());
        for t in 0..len {
            for h in 0..heads {
                for d in (0..dim).step_by(2) {
                    let i = (t * heads + h) * dim + d;
                    let table = t * dim + d;
                    expected.push(qv[i] * cosv[table] - qv[i + 1] * sinv[table]);
                    expected.push(qv[i + 1] * cosv[table] + qv[i] * sinv[table]);
                }
            }
        }
        check("rope", &got, &expected, 2e-2);
    }
}

#[cfg(test)]
mod logic_tests {
    use super::speed_output_frames;

    #[test]
    fn speed_output_frames_matches_official_formula() {
        assert_eq!(speed_output_frames(100, 1.05), 96); // floor(95.238) + 1
        assert_eq!(speed_output_frames(100, 1.0), 101);
        assert_eq!(speed_output_frames(100, 1.5), 67); // floor(66.666) + 1
        assert_eq!(speed_output_frames(100, 0.5), 201);
        assert_eq!(speed_output_frames(1, 2.0), 1);
    }
}
