//! FP32 dense CPU operations. Heads are scheduled serially; BLAS owns GEMM threading.
use crate::{CpuBackend, Error, Result, Tensor};

impl CpuBackend {
    /// Non-causal GQA/MHA: Q [queries, heads, dim], K/V [keys, kv_heads, dim].
    /// All supplied keys are visible. Callers partition independent image batches.
    pub fn attention(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let qs = q.shape().dims();
        let ks = k.shape().dims();
        if qs.len() != 3
            || ks.len() != 3
            || k.shape() != v.shape()
            || qs.iter().any(|&n| n == 0)
            || ks.iter().any(|&n| n == 0)
            || qs[2] != ks[2]
            || qs[1] % ks[1] != 0
        {
            return Err(Error::Other("CPU attention: invalid Q/K/V geometry".into()));
        }
        let (nq, nh, d) = (qs[0], qs[1], qs[2]);
        let (nk, nkh) = (ks[0], ks[1]);
        let (q, k, v) = (q.as_f32()?, k.as_f32()?, v.as_f32()?);
        let mut output = vec![0.; nq * nh * d];
        let mut query = vec![0.; nq * d];
        let mut key_t = vec![0.; d * nk];
        let mut value = vec![0.; nk * d];
        let mut scores = vec![0.; nq * nk];
        let mut head_out = vec![0.; nq * d];
        for h in 0..nh {
            let kh = h / (nh / nkh);
            for t in 0..nq {
                query[t * d..(t + 1) * d]
                    .copy_from_slice(&q[(t * nh + h) * d..(t * nh + h + 1) * d]);
            }
            for t in 0..nk {
                for c in 0..d {
                    key_t[c * nk + t] = k[(t * nkh + kh) * d + c];
                }
                value[t * d..(t + 1) * d]
                    .copy_from_slice(&v[(t * nkh + kh) * d..(t * nkh + kh + 1) * d]);
            }
            crate::ops::sgemm(nq, d, nk, &query, &key_t, &mut scores);
            for row in scores.chunks_exact_mut(nk) {
                for x in row.iter_mut() {
                    *x /= (d as f32).sqrt();
                }
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.;
                for x in row.iter_mut() {
                    *x = (*x - max).exp();
                    sum += *x;
                }
                for x in row {
                    *x /= sum;
                }
            }
            crate::ops::sgemm(nq, nk, d, &scores, &value, &mut head_out);
            for t in 0..nq {
                output[(t * nh + h) * d..(t * nh + h + 1) * d]
                    .copy_from_slice(&head_out[t * d..(t + 1) * d]);
            }
        }
        Tensor::from_f32(vec![nq, nh, d], &output)
    }
}

pub(super) fn layer_norm(
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let dims = input.shape().dims();
    let cols = *dims
        .last()
        .ok_or_else(|| Error::Other("CPU LayerNorm: scalar input".into()))?;
    if cols == 0
        || weight.shape().dims() != [cols]
        || bias.shape().dims() != [cols]
        || !eps.is_finite()
        || eps <= 0.
    {
        return Err(Error::Other("CPU LayerNorm: invalid shape/epsilon".into()));
    }
    let (x, w, b) = (input.as_f32()?, weight.as_f32()?, bias.as_f32()?);
    let mut out = vec![0.; x.len()];
    for (row, dst) in x.chunks_exact(cols).zip(out.chunks_exact_mut(cols)) {
        let mean = row.iter().sum::<f32>() / cols as f32;
        let var = row.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / cols as f32;
        let inv = (var + eps).sqrt().recip();
        for c in 0..cols {
            dst[c] = (row[c] - mean) * inv * w[c] + b[c];
        }
    }
    Tensor::from_f32(dims.to_vec(), &out)
}

pub(super) fn gelu(input: &Tensor) -> Result<Tensor> {
    let out: Vec<_> = input
        .as_f32()?
        .iter()
        .map(|&x| {
            0.5 * x
                * (1. + (std::f32::consts::FRAC_2_PI.sqrt() * (x + 0.044715 * x * x * x)).tanh())
        })
        .collect();
    Tensor::from_f32(input.shape().clone(), &out)
}

pub(super) fn add_bias(input: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let cols = *input.shape().dims().last().unwrap_or(&0);
    if cols == 0 || bias.shape().dims() != [cols] {
        return Err(Error::Other("CPU bias: invalid shape".into()));
    }
    let b = bias.as_f32()?;
    let out: Vec<_> = input
        .as_f32()?
        .iter()
        .enumerate()
        .map(|(i, x)| x + b[i % cols])
        .collect();
    Tensor::from_f32(input.shape().clone(), &out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Backend;
    #[test]
    fn gqa_matches_scalar_f64_attention() {
        let (nq, nk, nh, nkh, d) = (3, 5, 4, 2, 4);
        let values = |n: usize, phase: f32| {
            (0..n)
                .map(|i| (i as f32 * 0.37 + phase).sin())
                .collect::<Vec<_>>()
        };
        let q = values(nq * nh * d, 0.);
        let k = values(nk * nkh * d, 0.4);
        let v = values(nk * nkh * d, 1.7);
        let out = CpuBackend
            .attention(
                &Tensor::from_f32(vec![nq, nh, d], &q).unwrap(),
                &Tensor::from_f32(vec![nk, nkh, d], &k).unwrap(),
                &Tensor::from_f32(vec![nk, nkh, d], &v).unwrap(),
            )
            .unwrap();
        let out = out.as_f32().unwrap();
        for t in 0..nq {
            for h in 0..nh {
                let kh = h / (nh / nkh);
                let scores: Vec<f64> = (0..nk)
                    .map(|s| {
                        (0..d)
                            .map(|c| {
                                q[(t * nh + h) * d + c] as f64 * k[(s * nkh + kh) * d + c] as f64
                            })
                            .sum::<f64>()
                            / (d as f64).sqrt()
                    })
                    .collect();
                let sum = scores.iter().map(|x| x.exp()).sum::<f64>();
                for c in 0..d {
                    let expected = (0..nk)
                        .map(|s| scores[s].exp() / sum * v[(s * nkh + kh) * d + c] as f64)
                        .sum::<f64>();
                    assert!((out[(t * nh + h) * d + c] as f64 - expected).abs() < 2e-6);
                }
            }
        }
    }
    #[test]
    fn dense_ops_validate_shapes_and_normalize_rows() {
        let x = Tensor::from_f32(vec![2, 2], &[1., 3., 5., 7.]).unwrap();
        let w = Tensor::from_f32(vec![2], &[2., 3.]).unwrap();
        let b = Tensor::from_f32(vec![2], &[1., -1.]).unwrap();
        let y = CpuBackend.layer_norm(&x, &w, &b, 1e-6).unwrap();
        for (a, b) in y.as_f32().unwrap().iter().zip([-1., 2., -1., 2.]) {
            assert!((a - b).abs() < 2e-6);
        }
        assert!(CpuBackend.layer_norm(&x, &x, &b, 1e-6).is_err());
        assert!(CpuBackend.attention(&x, &x, &x).is_err());
    }
}
