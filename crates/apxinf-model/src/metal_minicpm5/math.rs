//! MiniCPM's ordered BF16 arithmetic, not Qwen3's normalization contract.
//! Derived from EngineTailor (MIT), Copyright 2026 Haiyan Qin; see NOTICE.
use apxinf_core::Result;
use apxinf_mlx::{Array, MlxDType, Stream};

pub fn norm(x: &Array, weight: &Array, epsilon: &Array) -> Result<Array> {
    let f = x.cast(MlxDType::F32)?;
    let variance = f.mul(&f)?.mean(&[-1], true)?;
    // Transformers Llama rounds normalized values BEFORE multiplying weights.
    weight.mul(&f.mul(&variance.add(epsilon)?.rsqrt()?)?.cast(x.dtype())?)
}

pub fn rotate(x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
    let a = x.slice_axis(3, 64, 128)?.neg()?;
    let b = x.slice_axis(3, 0, 64)?;
    x.mul(cos)?.add(&Array::concat(&[&a, &b], 3)?.mul(sin)?)
}

pub fn tables(stream: &Stream, capacity: usize) -> Result<(Array, Array)> {
    let positions =
        Array::arange(stream, 0., capacity as f32, 1., MlxDType::F32)?.reshape(&[capacity, 1])?;
    let indices = Array::arange(stream, 0., 128., 2., MlxDType::F32)?;
    let base = Array::scalar(stream, 5_000_000., MlxDType::F32)?;
    let divisor = Array::scalar(stream, 128., MlxDType::F32)?;
    let one = Array::scalar(stream, 1., MlxDType::F32)?;
    let inverse = one
        .div(&base.pow(&indices.div(&divisor)?)?)?
        .reshape(&[1, 64])?;
    let frequencies = positions.mul(&inverse)?;
    let angles = Array::concat(&[&frequencies, &frequencies], 1)?;
    let cos = angles
        .cos()?
        .cast(MlxDType::BF16)?
        .reshape(&[1, 1, capacity, 128])?;
    let sin = angles
        .sin()?
        .cast(MlxDType::BF16)?
        .reshape(&[1, 1, capacity, 128])?;
    stream.eval(&[cos.clone(), sin.clone()])?;
    Ok((cos, sin))
}

pub fn swiglu(gate: &Array, up: &Array) -> Result<Array> {
    // Same operation order as mlx_lm.models.activations.swiglu.
    gate.mul(&gate.sigmoid()?)?.mul(up)
}

/// Preserve independent M1 projection reductions during tapped verification.
/// The extra owned padding element prevents MLX collapsing the batch to M8.
pub fn project(x: &Array, weight: &Array, preserve_m1: bool) -> Result<Array> {
    let rows = x.shape()[1];
    if !preserve_m1 || rows == 1 {
        return x.matmul(weight);
    }
    let k = x.shape()[2];
    let zeros = Array::zeros(x.stream(), &[rows, 1], x.dtype())?;
    let flat = x.reshape(&[rows, k])?;
    let padded = Array::concat(&[&flat, &zeros], 1)?.contiguous()?;
    let vectors = padded.as_strided(&[rows, 1, k], &[k + 1, k, 1], 0)?;
    vectors
        .matmul(&weight.reshape(&[1, k, weight.shape()[1]])?)?
        .reshape(&[1, rows, weight.shape()[1]])
}
