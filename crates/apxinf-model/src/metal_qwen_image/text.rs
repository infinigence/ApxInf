//! Prompt encoding: the fixed Qwen-Image-2.1 template and the Qwen3-VL text
//! stack (mflux `Qwen21TextEncoder`), read from disk one layer at a time.
use super::ops::Ops;
use super::weights::{
    TextEncoderFiles, TextLayer, DIM, HEADS, HEAD_DIM, TEXT_KV_HEADS, TEXT_LAYERS,
};
use apxinf_core::{Error, Result};
use apxinf_mlx::{Array, MlxDType};
use apxinf_tokenizer::Tokenizer;

const SYSTEM_PREFIX: &str =
    "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n";
/// The reference tokenizer truncates the templated prompt to this length.
const MAX_TOKENS: usize = 2048;
const EPS: f32 = 1e-6;

pub struct PromptTokens {
    pub ids: Vec<u32>,
    /// Leading system-prefix tokens whose states the DiT does not read.
    pub drop: usize,
}

pub fn tokenize(tokenizer: &Tokenizer, prompt: &str) -> Result<PromptTokens> {
    let text =
        format!("{SYSTEM_PREFIX}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n");
    let mut ids = tokenizer.encode_with_special_tokens(&text, true)?;
    ids.truncate(MAX_TOKENS);
    let drop = tokenizer
        .encode_with_special_tokens(SYSTEM_PREFIX, false)?
        .len();
    if ids.len() <= drop {
        return Err(Error::Other(
            "the Qwen-Image prompt template left no prompt tokens".into(),
        ));
    }
    Ok(PromptTokens { ids, drop })
}

/// `Qwen3VLRMSNorm`: FP32 normalization, FP32 weight multiply, cast back.
fn rms_norm(x: &Array, weight: &Array, ops: &Ops) -> Result<Array> {
    let x32 = x.cast(MlxDType::F32)?;
    let variance = x32.mul(&x32)?.mean(&[-1], true)?;
    let normalized = x32.mul(&variance.add(&ops.f32(EPS)?)?.rsqrt()?)?;
    weight
        .cast(MlxDType::F32)?
        .mul(&normalized)?
        .cast(x.dtype())
}

fn rotate_half(x: &Array) -> Result<Array> {
    let half = HEAD_DIM / 2;
    let x1 = x.slice_axis(3, 0, half)?;
    let x2 = x.slice_axis(3, half, HEAD_DIM)?;
    Array::concat(&[&x2.neg()?, &x1], 3)
}

/// `nn.Linear` without bias: `x @ W.T`.
fn linear(x: &Array, weight: &Array) -> Result<Array> {
    x.matmul(&weight.transpose(&[1, 0])?)
}

fn layer(
    h: &Array,
    w: &TextLayer,
    cos: &Array,
    sin: &Array,
    mask: &Array,
    ops: &Ops,
) -> Result<Array> {
    let length = h.shape()[1];
    let x = rms_norm(h, &w.input_norm, ops)?;
    let heads = |p: &Array, n: usize| p.reshape(&[1, length, n, HEAD_DIM]);
    let q = rms_norm(&heads(&linear(&x, &w.q)?, HEADS)?, &w.q_norm, ops)?;
    let k = rms_norm(&heads(&linear(&x, &w.k)?, TEXT_KV_HEADS)?, &w.k_norm, ops)?;
    let v = heads(&linear(&x, &w.v)?, TEXT_KV_HEADS)?;
    let (q, k, v) = (
        q.transpose(&[0, 2, 1, 3])?,
        k.transpose(&[0, 2, 1, 3])?,
        v.transpose(&[0, 2, 1, 3])?,
    );
    let q = q.mul(cos)?.add(&rotate_half(&q)?.mul(sin)?)?;
    let k = k.mul(cos)?.add(&rotate_half(&k)?.mul(sin)?)?;
    let repeat = |a: &Array| {
        let groups = HEADS / TEXT_KV_HEADS;
        a.reshape(&[1, TEXT_KV_HEADS, 1, length, HEAD_DIM])?
            .broadcast_to(&[1, TEXT_KV_HEADS, groups, length, HEAD_DIM])?
            .reshape(&[1, HEADS, length, HEAD_DIM])
    };
    let (k, v) = (repeat(&k)?, repeat(&v)?);
    let scale = (1.0 / (HEAD_DIM as f64).sqrt()) as f32;
    let attn = q
        .cast(MlxDType::F32)?
        .sdpa(
            &k.cast(MlxDType::F32)?,
            &v.cast(MlxDType::F32)?,
            scale,
            false,
            Some(mask),
        )?
        .cast(q.dtype())?
        .transpose(&[0, 2, 1, 3])?
        .reshape(&[1, length, DIM])?;
    let h = h.add(&linear(&attn, &w.o)?)?;
    let x = rms_norm(&h, &w.post_norm, ops)?;
    let mlp = ops.silu(&linear(&x, &w.gate)?)?.mul(&linear(&x, &w.up)?)?;
    h.add(&linear(&mlp, &w.down)?)
}

/// Prompt states after the final norm, without the system prefix: `[1, T, 4096]` BF16.
pub fn encode(files: &TextEncoderFiles, tokens: &PromptTokens, ops: &Ops) -> Result<Array> {
    let stream = ops.stream();
    let length = tokens.ids.len();
    let ids: Vec<i32> = tokens.ids.iter().map(|&id| id as i32).collect();
    let mut h = {
        let embedding = files.tensor(stream, "embed_tokens.weight")?;
        let h = embedding.take(&Array::from_i32(stream, &[1, length], &ids)?, 0)?;
        h.eval()?;
        h
    };
    // Additive causal mask; the unpadded prompt adds a zero padding mask.
    let mask: Vec<f32> = (0..length * length)
        .map(|i| {
            if i % length > i / length {
                f32::NEG_INFINITY
            } else {
                0.0
            }
        })
        .collect();
    let mask = Array::from_f32(stream, &[1, 1, length, length], &mask)?;
    // Text positions repeat on all three MRoPE axes, so the interleave is the identity.
    let inv_freq = files.tensor(stream, "rotary_emb.inv_freq")?;
    let positions: Vec<f32> = (0..length).map(|p| p as f32).collect();
    let freqs = inv_freq
        .reshape(&[HEAD_DIM / 2, 1])?
        .matmul(&Array::from_f32(stream, &[1, length], &positions)?)?
        .transpose(&[1, 0])?;
    let emb = Array::concat(&[&freqs, &freqs], 1)?.reshape(&[1, 1, length, HEAD_DIM])?;
    let cos = emb.cos()?.cast(h.dtype())?;
    let sin = emb.sin()?.cast(h.dtype())?;
    for i in 0..TEXT_LAYERS {
        let weights = files.layer(stream, i)?;
        h = layer(&h, &weights, &cos, &sin, &mask, ops)?;
        // Release this layer's weights before reading the next one.
        h.eval()?;
    }
    let norm = files.tensor(stream, "norm.weight")?;
    let h = rms_norm(&h, &norm, ops)?.slice_axis(1, tokens.drop, length)?;
    h.eval()?;
    Ok(h)
}
