use crate::{Array, MlxDType, Stream};
use apxinf_core::{
    contracts::{self, AttentionMask, AttentionOptions, AxisSlice},
    Backend, DType, Device, Error, Graph, KvCache, NextTokenLogits, NormalGenerator, Result,
    SamplingBackend, Shape, Storage, Tensor, TokenPenalties, TokenSample, TokenSampler,
    TokenSamplingInit, TokenSamplingSpec, TokenSelection,
};
use std::{any::Any, rc::Rc};

fn err(s: impl Into<String>) -> Error {
    Error::Other(s.into())
}
fn mlx_dtype(d: DType) -> Result<MlxDType> {
    match d {
        DType::F32 => Ok(MlxDType::F32),
        DType::F16 => Ok(MlxDType::F16),
        DType::BF16 => Ok(MlxDType::BF16),
        DType::I32 => Ok(MlxDType::I32),
        _ => Err(Error::UnsupportedDType {
            got: d,
            allowed: "f32, f16, bf16, i32",
        }),
    }
}
fn core_dtype(d: MlxDType) -> Result<DType> {
    match d {
        MlxDType::F32 => Ok(DType::F32),
        MlxDType::F16 => Ok(DType::F16),
        MlxDType::BF16 => Ok(DType::BF16),
        MlxDType::I32 => Ok(DType::I32),
        _ => Err(err(
            "internal MLX index/mask dtype is not a public Tensor dtype",
        )),
    }
}
/// A concrete, thread-confined Metal backend. CPU-stream arrays are an explicit
/// low-level diagnostic facility, never an implicit fallback for this backend.
#[derive(Clone)]
pub struct MlxBackend {
    stream: Stream,
    index: usize,
}
impl MlxBackend {
    pub fn new(index: usize) -> Result<Self> {
        Ok(Self {
            stream: Stream::metal(index)?,
            index,
        })
    }
    pub fn stream(&self) -> &Stream {
        &self.stream
    }
    pub fn from_array(&self, array: Array) -> Result<Tensor> {
        if !self.stream.same(array.stream()) {
            return Err(err("MLX result belongs to another backend stream"));
        }
        let dtype = core_dtype(array.dtype())?;
        let array = array.contiguous()?;
        let shape = Shape::new(array.shape().to_vec());
        let bytes = contracts::checked_bytes(array.shape(), dtype)?;
        Tensor::from_opaque_parts(shape, dtype, self.device(), bytes, Rc::new(array))
    }
    pub fn array(&self, t: &Tensor) -> Result<Array> {
        contracts::tensor_storage(t, self.device())?;
        let Storage::Opaque { device, handle } = t.storage() else {
            return Err(err("Tensor is not backed by an MLX array"));
        };
        if *device != self.device() {
            return Err(err("opaque device metadata disagrees"));
        }
        let a = handle
            .owner()
            .downcast_ref::<Array>()
            .ok_or_else(|| err("opaque Tensor belongs to another backend"))?;
        if !self.stream.same(a.stream()) || a.stream().metal_index() != Some(self.index) {
            return Err(err("opaque array belongs to another MLX execution owner"));
        }
        if a.dtype() != mlx_dtype(t.dtype())? || a.numel() != t.numel() {
            return Err(err(
                "opaque array dtype/extent disagrees with Tensor metadata",
            ));
        }
        a.reshape(t.shape().dims())
    }
    fn float_array(&self, t: &Tensor) -> Result<Array> {
        contracts::float_tensor(t, self.device())?;
        self.array(t)
    }
    fn same_float(&self, a: &Tensor, b: &Tensor) -> Result<(Array, Array)> {
        if a.dtype() != b.dtype() {
            return Err(Error::DTypeMismatch {
                expected: a.dtype(),
                got: b.dtype(),
            });
        }
        Ok((self.float_array(a)?, self.float_array(b)?))
    }
    fn cached_attention(
        &self,
        q: &Tensor,
        kv: &mut dyn KvCache,
        layer: usize,
        heads: usize,
        kv_heads: usize,
        dim: usize,
        kv_len: usize,
        max_len: usize,
    ) -> Result<Tensor> {
        let cache = kv
            .as_any_mut()
            .downcast_mut::<MlxKvCache>()
            .ok_or_else(|| err("foreign KV cache"))?;
        cache.valid()?;
        if !cache.backend.stream.same(&self.stream)
            || cache.heads != kv_heads
            || cache.dim != dim
            || cache.max_len != max_len
        {
            return Err(err("KV geometry or owner mismatch"));
        }
        let pair = cache
            .pending
            .get(layer)
            .and_then(Option::as_ref)
            .or_else(|| cache.layers.get(layer).and_then(Option::as_ref))
            .ok_or_else(|| err("uninitialized KV layer"))?;
        let rows = q
            .shape()
            .dims()
            .first()
            .copied()
            .ok_or_else(|| err("invalid query rank"))?;
        if q.shape().dims() != [rows, heads, dim]
            || kv_len < rows
            || kv_len > max_len
            || pair.0.shape()[1] != kv_len
        {
            return Err(err("cached attention shape/length mismatch"));
        }
        let qt = self.from_array(self.array(q)?.reshape(&[1, rows, heads, dim])?)?;
        let kt = self.from_array(pair.0.clone())?;
        let vt = self.from_array(pair.1.clone())?;
        let options = AttentionOptions {
            scale: (dim as f32).sqrt().recip(),
            mask: AttentionMask::Causal {
                q_start: kv_len - rows,
                k_start: 0,
            },
            scores: DType::F32,
            probabilities: DType::F32,
        };
        let out = self.attention_impl(&qt, &kt, &vt, &options)?;
        self.from_array(self.array(&out)?.reshape(&[rows, heads * dim])?)
    }
}
impl Backend for MlxBackend {
    fn cast_impl(&self, t: &Tensor, d: DType) -> Result<Tensor> {
        let cast = self.array(t)?.cast(mlx_dtype(d)?)?;
        // MLX astype is allowed to return its input for equal dtype; the public
        // portable contract promises an independent functional result.
        self.from_array(if t.dtype() == d { cast.copy()? } else { cast })
    }
    fn slice_axis_impl(&self, t: &Tensor, s: AxisSlice) -> Result<Tensor> {
        s.validate(t.shape().dims())?;
        self.from_array(self.array(t)?.slice_axis(s.axis, s.start, s.end)?.copy()?)
    }
    fn concat_axis_impl(&self, t: &[&Tensor], axis: usize) -> Result<Tensor> {
        let a = t
            .iter()
            .map(|t| self.array(t))
            .collect::<Result<Vec<_>>>()?;
        self.from_array(Array::concat(&a.iter().collect::<Vec<_>>(), axis)?)
    }
    fn permute_impl(&self, t: &Tensor, axes: &[usize]) -> Result<Tensor> {
        self.from_array(self.array(t)?.transpose(axes)?.copy()?)
    }
    fn broadcast_to_impl(&self, t: &Tensor, shape: &[usize]) -> Result<Tensor> {
        self.from_array(self.array(t)?.broadcast_to(shape)?.copy()?)
    }
    fn attention_impl(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        o: &AttentionOptions<'_>,
    ) -> Result<Tensor> {
        o.validate(self.device(), q, k, v)?;
        let qs = q.shape().dims();
        let ks = k.shape().dims();
        let (batch, qlen, heads, dim) = (qs[0], qs[1], qs[2], qs[3]);
        let (klen, kh) = (ks[1], ks[2]);
        let qa = self
            .array(q)?
            .cast(MlxDType::F32)?
            .transpose(&[0, 2, 1, 3])?;
        let expand = |t: &Tensor| -> Result<Array> {
            self.array(t)?
                .cast(MlxDType::F32)?
                .transpose(&[0, 2, 1, 3])?
                .reshape(&[batch, kh, 1, klen, dim])?
                .broadcast_to(&[batch, kh, heads / kh, klen, dim])?
                .reshape(&[batch, heads, klen, dim])
        };
        let ka = expand(k)?;
        let va = expand(v)?;
        let mut scores = qa
            .matmul(&ka.transpose(&[0, 1, 3, 2])?)?
            .mul(&Array::scalar(&self.stream, o.scale, MlxDType::F32)?)?;
        match o.mask {
            AttentionMask::Full => {}
            AttentionMask::Additive(t) => {
                scores = scores.add(&self.array(t)?)?;
            }
            AttentionMask::Causal { q_start, k_start } => {
                // arange's ABI is float-valued; reject positions that lose exact integers.
                if q_start.checked_add(qlen).is_none_or(|n| n > 16_777_216)
                    || k_start.checked_add(klen).is_none_or(|n| n > 16_777_216)
                {
                    return Err(err("causal positions exceed exact MLX arange domain"));
                }
                let qp = Array::arange(
                    &self.stream,
                    q_start as f32,
                    (q_start + qlen) as f32,
                    1.,
                    MlxDType::I32,
                )?
                .reshape(&[qlen, 1])?;
                let kp = Array::arange(
                    &self.stream,
                    k_start as f32,
                    (k_start + klen) as f32,
                    1.,
                    MlxDType::I32,
                )?
                .reshape(&[1, klen])?;
                scores = kp.less_equal(&qp)?.where_select(
                    &scores,
                    &Array::scalar(&self.stream, f32::NEG_INFINITY, MlxDType::F32)?,
                )?;
            }
        }
        scores = scores.cast(mlx_dtype(o.scores)?)?.cast(MlxDType::F32)?;
        let valid = scores.max(-1)?.isfinite()?;
        let zero = Array::scalar(&self.stream, 0., MlxDType::F32)?;
        let probabilities = valid.where_select(&scores, &zero)?.softmax(-1)?;
        let probabilities = valid
            .where_select(&probabilities, &zero)?
            .cast(mlx_dtype(o.probabilities)?)?
            .cast(MlxDType::F32)?;
        self.from_array(
            probabilities
                .matmul(&va)?
                .cast(mlx_dtype(q.dtype())?)?
                .transpose(&[0, 2, 1, 3])?,
        )
    }
    fn rms_norm(&self, input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        if !eps.is_finite()
            || eps <= 0.
            || weight.shape().dims()
                != [input
                    .shape()
                    .dims()
                    .last()
                    .copied()
                    .ok_or_else(|| err("norm scalar unsupported"))?]
        {
            return Err(err("invalid RMSNorm contract"));
        }
        let (x, w) = self.same_float(input, weight)?;
        let x = x.cast(MlxDType::F32)?;
        let var = x.mul(&x)?.mean(&[-1], true)?;
        let norm = x.mul(
            &var.add(&Array::scalar(&self.stream, eps, MlxDType::F32)?)?
                .rsqrt()?,
        )?;
        self.from_array(
            norm.mul(&w.cast(MlxDType::F32)?)?
                .cast(mlx_dtype(input.dtype())?)?,
        )
    }
    fn silu(&self, t: &Tensor) -> Result<Tensor> {
        let a = self.float_array(t)?.cast(MlxDType::F32)?;
        self.from_array(a.mul(&a.sigmoid()?)?.cast(mlx_dtype(t.dtype())?)?)
    }
    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        if a.shape() != b.shape() {
            return Err(err("portable add requires equal shapes"));
        }
        let (x, y) = self.same_float(a, b)?;
        self.from_array(
            x.cast(MlxDType::F32)?
                .add(&y.cast(MlxDType::F32)?)?
                .cast(mlx_dtype(a.dtype())?)?,
        )
    }
    fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        if a.shape() != b.shape() {
            return Err(err("portable mul requires equal shapes"));
        }
        let (x, y) = self.same_float(a, b)?;
        self.from_array(
            x.cast(MlxDType::F32)?
                .mul(&y.cast(MlxDType::F32)?)?
                .cast(mlx_dtype(a.dtype())?)?,
        )
    }
    fn scale(&self, t: &Tensor, f: f32) -> Result<Tensor> {
        self.from_array(
            self.float_array(t)?
                .cast(MlxDType::F32)?
                .mul(&Array::scalar(&self.stream, f, MlxDType::F32)?)?
                .cast(mlx_dtype(t.dtype())?)?,
        )
    }
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let (x, y) = self.same_float(a, b)?;
        self.from_array(
            x.cast(MlxDType::F32)?
                .matmul(&y.cast(MlxDType::F32)?)?
                .cast(mlx_dtype(a.dtype())?)?,
        )
    }
    fn rope(&self, t: &Tensor, heads: usize, dim: usize, theta: f32, pos: u32) -> Result<Tensor> {
        let seq = t.shape().dims().first().copied().unwrap_or(0);
        if dim == 0
            || dim % 2 != 0
            || t.shape().dims() != [seq, heads, dim]
            || !theta.is_finite()
            || theta <= 0.
            || u64::from(pos) + seq as u64 > 16_777_216
        {
            return Err(err("invalid RoPE contract"));
        }
        let x = self.float_array(t)?.cast(MlxDType::F32)?;
        let frequency = Array::arange(&self.stream, 0., dim as f32, 2., MlxDType::F32)?
            .div(&Array::scalar(&self.stream, dim as f32, MlxDType::F32)?)?;
        let inv = Array::scalar(&self.stream, theta, MlxDType::F32)?
            .pow(&frequency)?
            .pow(&Array::scalar(&self.stream, -1., MlxDType::F32)?)?;
        let p = Array::arange(
            &self.stream,
            pos as f32,
            pos as f32 + seq as f32,
            1.,
            MlxDType::F32,
        )?
        .reshape(&[seq, 1, 1])?;
        let freq = p.mul(&inv.reshape(&[1, 1, dim / 2])?)?;
        let angles = Array::concat(&[&freq, &freq], 2)?;
        let rot = Array::concat(
            &[
                &x.slice_axis(2, dim / 2, dim)?.neg()?,
                &x.slice_axis(2, 0, dim / 2)?,
            ],
            2,
        )?;
        self.from_array(
            x.mul(&angles.cos()?)?
                .add(&rot.mul(&angles.sin()?)?)?
                .cast(mlx_dtype(t.dtype())?)?,
        )
    }
    fn embedding(&self, table: &Tensor, ids: &[u32]) -> Result<Tensor> {
        if table.ndim() != 2
            || ids.is_empty()
            || ids
                .iter()
                .any(|&x| x as usize >= table.shape().dims()[0] || x > i32::MAX as u32)
        {
            return Err(err("invalid embedding ids/shape"));
        }
        let index = Array::from_i32(
            &self.stream,
            &[ids.len()],
            &ids.iter().map(|&x| x as i32).collect::<Vec<_>>(),
        )?;
        self.from_array(self.array(table)?.take(&index, 0)?)
    }
    fn sdpa_decode(
        &self,
        q: &Tensor,
        kv: &mut dyn KvCache,
        l: usize,
        h: usize,
        kh: usize,
        d: usize,
        n: usize,
        max: usize,
    ) -> Result<Tensor> {
        self.cached_attention(q, kv, l, h, kh, d, n, max)
    }
    fn sdpa_prefill(
        &self,
        q: &Tensor,
        kv: &mut dyn KvCache,
        l: usize,
        h: usize,
        kh: usize,
        d: usize,
        n: usize,
        max: usize,
    ) -> Result<Tensor> {
        self.cached_attention(q, kv, l, h, kh, d, n, max)
    }
    fn create_kv_cache(
        &self,
        layers: usize,
        heads: usize,
        dim: usize,
        max_len: usize,
    ) -> Box<dyn KvCache> {
        Box::new(MlxKvCache {
            backend: self.clone(),
            layers: vec![None; layers],
            pending: vec![None; layers],
            seq_len: 0,
            heads,
            dim,
            max_len,
            invalid: (layers == 0 || heads == 0 || dim == 0 || max_len == 0)
                .then(|| "invalid KV dimensions".into()),
        })
    }
    fn kv_append(
        &self,
        kv: &mut dyn KvCache,
        l: usize,
        k: &Tensor,
        v: &Tensor,
        n: usize,
    ) -> Result<()> {
        let cache = kv
            .as_any_mut()
            .downcast_mut::<MlxKvCache>()
            .ok_or_else(|| err("foreign KV cache"))?;
        if !cache.backend.stream.same(&self.stream) {
            return Err(err("KV belongs to another backend"));
        }
        cache.append(l, k, v, n)
    }
    fn synchronize(&self) -> Result<()> {
        self.stream.synchronize()
    }
    fn begin_capture(&self) -> Result<()> {
        Err(Error::UnsupportedOp("CUDA-style graph capture on MLX"))
    }
    fn end_capture(&self) -> Result<Box<dyn Graph>> {
        Err(Error::UnsupportedOp("CUDA-style graph capture on MLX"))
    }
    fn device(&self) -> Device {
        Device::Metal(self.index)
    }
    fn to_device(&self, t: &Tensor) -> Result<Tensor> {
        if t.device() == self.device() {
            return self.from_array(self.array(t)?);
        }
        if t.device() != Device::Cpu {
            return Err(Error::UnsupportedDevice(t.device()));
        }
        let bytes = t
            .storage()
            .as_cpu()
            .ok_or_else(|| err("CPU tensor has no CPU bytes"))?;
        self.from_array(Array::from_bytes(
            &self.stream,
            t.shape().dims(),
            mlx_dtype(t.dtype())?,
            bytes,
        )?)
    }
    fn to_cpu(&self, t: &Tensor) -> Result<Tensor> {
        let a = self.array(t)?;
        Tensor::from_raw(t.shape().clone(), t.dtype(), Device::Cpu, a.to_bytes()?)
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}
impl SamplingBackend for MlxBackend {
    fn create_token_sampler(&self, spec: TokenSamplingSpec) -> Result<Box<dyn TokenSampler>> {
        spec.validate()?;
        Ok(Box::new(Greedy {
            backend: self.clone(),
            spec,
            remaining: None,
        }))
    }
    fn create_normal_generator(&self, _output: Tensor) -> Result<Box<dyn NormalGenerator>> {
        Err(Error::UnsupportedOp("MLX normal generator"))
    }
}
struct Greedy {
    backend: MlxBackend,
    spec: TokenSamplingSpec,
    remaining: Option<usize>,
}
impl TokenSampler for Greedy {
    fn spec(&self) -> TokenSamplingSpec {
        self.spec
    }
    fn begin(&mut self, init: TokenSamplingInit<'_>) -> Result<()> {
        self.remaining = None;
        init.params.validate(self.spec.vocab_size)?;
        if init.params.selection != TokenSelection::Greedy
            || init.params.penalties != TokenPenalties::default()
            || init.params.return_logprob
        {
            return Err(Error::UnsupportedOp("MLX sampling options beyond greedy"));
        }
        if init
            .prompt_token_ids
            .iter()
            .any(|&x| x as usize >= self.spec.vocab_size)
        {
            return Err(err("prompt token exceeds vocabulary"));
        }
        self.remaining = Some(
            self.spec
                .max_sequence_len
                .checked_sub(init.prompt_token_ids.len())
                .ok_or_else(|| err("prompt exceeds sampler capacity"))?,
        );
        Ok(())
    }
    fn sample(&mut self, logits: NextTokenLogits<'_>) -> Result<TokenSample> {
        let remaining = self
            .remaining
            .ok_or_else(|| err("sampler requires begin"))?;
        if remaining == 0 {
            return Err(err("sampler capacity exhausted"));
        }
        if logits.vocab_size() != self.spec.vocab_size {
            return Err(err("sampler vocabulary mismatch"));
        }
        let a = self
            .backend
            .array(logits.tensor())?
            .reshape(&[
                logits.tensor().numel() / self.spec.vocab_size,
                self.spec.vocab_size,
            ])?
            .slice_axis(0, logits.row_index(), logits.row_index() + 1)?
            .cast(MlxDType::F32)?;
        // Match core NaN/+inf/-inf normalization; argmax resolves ties to first.
        let token = greedy_token(&a)?;
        self.remaining = Some(remaining - 1);
        Ok(TokenSample {
            token_id: token,
            logprob: None,
        })
    }
}
fn greedy_token(a: &Array) -> Result<u32> {
    let adjusted = a.nan_to_num(f32::NEG_INFINITY, f32::MAX, f32::NEG_INFINITY)?;
    let valid = adjusted.max(-1)?.isfinite()?.reshape(&[1])?;
    let invalid = Array::from_i32(a.stream(), &[1], &[-1])?.cast(MlxDType::U32)?;
    let token = valid
        .where_select(&adjusted.argmax(-1)?.reshape(&[1])?, &invalid)?
        .to_u32_scalar()?;
    if token == u32::MAX {
        return Err(err("all token logits are invalid"));
    }
    Ok(token)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn greedy_matches_core_invalid_and_tie_rules() -> Result<()> {
        let s = Stream::cpu()?;
        assert_eq!(
            greedy_token(&Array::from_f32(
                &s,
                &[1, 4],
                &[f32::NAN, 4., 4., f32::NEG_INFINITY]
            )?)?,
            1
        );
        assert_eq!(
            greedy_token(&Array::from_f32(&s, &[1, 3], &[1., f32::INFINITY, 2.])?)?,
            1
        );
        assert!(greedy_token(&Array::from_f32(
            &s,
            &[1, 2],
            &[f32::NAN, f32::NEG_INFINITY]
        )?)
        .is_err());
        Ok(())
    }
}
type Pair = (Array, Array);
struct MlxKvCache {
    backend: MlxBackend,
    layers: Vec<Option<Pair>>,
    pending: Vec<Option<Pair>>,
    seq_len: usize,
    heads: usize,
    dim: usize,
    max_len: usize,
    invalid: Option<String>,
}
impl MlxKvCache {
    fn valid(&self) -> Result<()> {
        if let Some(e) = &self.invalid {
            Err(err(e.clone()))
        } else {
            Ok(())
        }
    }
}
impl KvCache for MlxKvCache {
    fn append(&mut self, l: usize, k: &Tensor, v: &Tensor, n: usize) -> Result<()> {
        self.valid()?;
        if l >= self.layers.len()
            || n == 0
            || self.seq_len.checked_add(n).is_none_or(|v| v > self.max_len)
            || self.pending[l].is_some()
            || k.shape().dims() != [n, self.heads, self.dim]
            || v.shape() != k.shape()
            || v.dtype() != k.dtype()
        {
            return Err(err("invalid KV append"));
        }
        let k = self
            .backend
            .array(k)?
            .reshape(&[1, n, self.heads, self.dim])?;
        let v = self
            .backend
            .array(v)?
            .reshape(&[1, n, self.heads, self.dim])?;
        self.pending[l] = Some(if let Some((oldk, oldv)) = &self.layers[l] {
            (
                Array::concat(&[oldk, &k], 1)?,
                Array::concat(&[oldv, &v], 1)?,
            )
        } else {
            (k, v)
        });
        Ok(())
    }
    fn advance(&mut self, n: usize) {
        let result = (|| -> Result<()> {
            self.valid()?;
            let next = self
                .seq_len
                .checked_add(n)
                .filter(|&x| x <= self.max_len)
                .ok_or_else(|| err("KV advance exceeds capacity"))?;
            if self
                .pending
                .iter()
                .any(|x| x.as_ref().is_none_or(|p| p.0.shape()[1] != next))
            {
                return Err(err("all KV layers must append before advance"));
            }
            let all: Vec<_> = self
                .pending
                .iter()
                .filter_map(Option::as_ref)
                .flat_map(|(k, v)| [k.clone(), v.clone()])
                .collect();
            self.backend.stream.eval(&all)?;
            Ok(())
        })();
        match result {
            Ok(()) => {
                for (old, new) in self.layers.iter_mut().zip(&mut self.pending) {
                    *old = new.take()
                }
                self.seq_len += n
            }
            Err(e) => {
                self.invalid = Some(e.to_string());
                self.pending.fill(None)
            }
        }
    }
    fn seq_len(&self) -> usize {
        self.seq_len
    }
    fn clear(&mut self) -> Result<()> {
        self.layers.fill(None);
        self.pending.fill(None);
        self.seq_len = 0;
        self.invalid = None;
        Ok(())
    }
    fn n_layers(&self) -> usize {
        self.layers.len()
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
