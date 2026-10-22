use super::super::{
    backend::{transfers, Context},
    DreamZeroConfig as C,
};
use apxinf_core::{DType, Error, Result, Tensor};
use std::collections::HashMap;

pub(in crate::dreamzero) struct TextBlockWeights {
    pub norm1: Tensor,
    pub q: Tensor,
    pub k: Tensor,
    pub v: Tensor,
    pub o: Tensor,
    pub norm2: Tensor,
    pub gate: Tensor,
    pub up: Tensor,
    pub down: Tensor,
    /// Constant learned relative positions, expanded once at load time.
    pub position: Tensor,
}
pub(in crate::dreamzero) struct TextWeights {
    pub embedding: Tensor,
    pub blocks: Vec<TextBlockWeights>,
    pub norm: Tensor,
}

/// A layout transformation only: BF16 bits are preserved, with no F32 copy.
fn transpose(tensor: Tensor) -> Result<Tensor> {
    let dims = tensor.shape().dims();
    if dims.len() != 2 {
        return Err(Error::Other("DreamZero linear must be rank two".into()));
    }
    let (rows, cols) = (dims[0], dims[1]);
    let src = tensor.as_bf16()?;
    let mut dst = vec![half::bf16::ZERO; src.len()];
    // Tile the host-only transpose so large BF16 matrices do not touch a new
    // cache line/page for every scalar store during checkpoint loading.
    for r0 in (0..rows).step_by(32) {
        for c0 in (0..cols).step_by(32) {
            for r in r0..(r0 + 32).min(rows) {
                for c in c0..(c0 + 32).min(cols) {
                    dst[c * rows + r] = src[r * cols + c];
                }
            }
        }
    }
    Tensor::from_bf16(vec![cols, rows], &dst)
}

fn take(map: &mut HashMap<String, Tensor>, name: &str, shape: &[usize]) -> Result<Tensor> {
    let key = format!("action_head.text_encoder.{name}");
    let tensor = map
        .remove(&key)
        .ok_or_else(|| Error::Other(format!("missing DreamZero weight {key}")))?;
    if tensor.dtype() != DType::BF16 || tensor.shape().dims() != shape {
        return Err(Error::Other(format!(
            "DreamZero {key}: expected BF16 {shape:?}, got {} {:?}",
            tensor.dtype(),
            tensor.shape().dims()
        )));
    }
    Ok(tensor)
}

/// Bidirectional T5 buckets: 16 per direction, exact for distances below 8.
fn bucket(query: usize, key: usize) -> usize {
    let distance = query.abs_diff(key);
    let magnitude = if distance < 8 {
        distance
    } else {
        (8 + ((distance as f32 / 8.).ln() / 16f32.ln() * 8.) as usize).min(15)
    };
    magnitude + if key > query { 16 } else { 0 }
}

impl TextWeights {
    pub fn from_map(ctx: &Context, map: &mut HashMap<String, Tensor>) -> Result<Self> {
        let upload = |tensor: Tensor| transfers::to_cuda(&tensor, ctx.device_id());
        let embedding = upload(take(
            map,
            "token_embedding.weight",
            &[C::TEXT_VOCAB, C::TEXT_DIM],
        )?)?;
        let mut blocks = Vec::with_capacity(C::TEXT_LAYERS);
        for i in 0..C::TEXT_LAYERS {
            let p = format!("blocks.{i}");
            let norm1 = upload(take(map, &format!("{p}.norm1.weight"), &[C::TEXT_DIM])?)?;
            let norm2 = upload(take(map, &format!("{p}.norm2.weight"), &[C::TEXT_DIM])?)?;
            let table = take(
                map,
                &format!("{p}.pos_embedding.embedding.weight"),
                &[32, C::TEXT_HEADS],
            )?;
            let table = table.as_bf16()?;
            let n = C::TEXT_LEN;
            let mut position = vec![half::bf16::ZERO; C::TEXT_HEADS * n * n];
            for h in 0..C::TEXT_HEADS {
                for q in 0..n {
                    for k in 0..n {
                        position[(h * n + q) * n + k] = table[bucket(q, k) * C::TEXT_HEADS + h];
                    }
                }
            }
            let position = upload(Tensor::from_bf16(vec![1, C::TEXT_HEADS, n, n], &position)?)?;
            let mut projection = |name: &str, input: usize, output: usize| {
                upload(transpose(take(
                    map,
                    &format!("{p}.{name}.weight"),
                    &[output, input],
                )?)?)
            };
            blocks.push(TextBlockWeights {
                norm1,
                norm2,
                position,
                q: projection("attn.q", C::TEXT_DIM, C::TEXT_DIM)?,
                k: projection("attn.k", C::TEXT_DIM, C::TEXT_DIM)?,
                v: projection("attn.v", C::TEXT_DIM, C::TEXT_DIM)?,
                o: projection("attn.o", C::TEXT_DIM, C::TEXT_DIM)?,
                gate: projection("ffn.gate.0", C::TEXT_DIM, C::TEXT_FFN)?,
                up: projection("ffn.fc1", C::TEXT_DIM, C::TEXT_FFN)?,
                down: projection("ffn.fc2", C::TEXT_FFN, C::TEXT_DIM)?,
            });
        }
        let norm = upload(take(map, "norm.weight", &[C::TEXT_DIM])?)?;
        Ok(Self {
            embedding,
            blocks,
            norm,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn t5_bucket_boundaries_and_direction() {
        assert_eq!((bucket(0, 0), bucket(0, 1), bucket(1, 0)), (0, 17, 1));
        assert_eq!((bucket(0, 7), bucket(0, 8), bucket(0, 16)), (23, 24, 26));
        assert_eq!((bucket(0, 128), bucket(511, 0)), (31, 15));
    }
    #[test]
    fn projection_transpose_preserves_bf16_bits() {
        let data = [0x8000, 0x3f80, 0x7fc1, 0xbf80, 0x3f81, 0x0001].map(half::bf16::from_bits);
        let x = Tensor::from_bf16(vec![2, 3], &data).unwrap();
        let y = transpose(x).unwrap();
        assert_eq!(y.shape().dims(), &[3, 2]);
        assert_eq!(
            y.as_bf16()
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            [data[0], data[3], data[1], data[4], data[2], data[5]]
                .map(|v| v.to_bits())
                .to_vec()
        );
    }
}
