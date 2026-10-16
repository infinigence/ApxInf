//! mlx-audio `_sample_token` with the family-owned MLX key sequence.
//! `mx.random.seed` sets the key; each sample splits it, uses the second key
//! and keeps the first, exactly as MLX's global sequence does.
use apxinf_core::Result;
use apxinf_mlx::{Array, Compiled, MlxDType, Stream};

const TEMPERATURE: f64 = 0.9;
const TOP_K: usize = 50;
const REPETITION_PENALTY: f64 = 1.05;

pub struct Sampler {
    stream: Stream,
    /// `apply_top_k`, compiled like the reference.
    top_k: Compiled,
    /// `categorical_sampling(logits, 1.0)`: inputs logits and key; outputs
    /// the token and the next key.
    categorical: Compiled,
    key: Option<Array>,
}

impl Sampler {
    pub fn new(stream: &Stream) -> Result<Self> {
        let s = stream.clone();
        let top_k = Compiled::new(stream, 1, move |a| {
            let logprobs = &a[0];
            let width = logprobs.shape()[1];
            let order = logprobs.neg()?.argpartition(TOP_K - 1, -1)?;
            let mask = order.slice(&[0, TOP_K], &[1, width])?;
            let inf = Array::scalar(&s, f32::NEG_INFINITY, logprobs.dtype())?;
            Ok(vec![logprobs.put_along_axis(&mask, &inf, -1)?])
        })?;
        let s = stream.clone();
        let categorical = Compiled::new(stream, 2, move |a| {
            let (next, sub) = Array::random_split(&a[1])?;
            let scaled = a[0].mul(&Array::scalar(&s, 1.0, a[0].dtype())?)?;
            Ok(vec![scaled.categorical(-1, &sub)?, next])
        })?;
        Ok(Self {
            stream: stream.clone(),
            top_k,
            categorical,
            key: None,
        })
    }

    /// `mx.random.seed(seed)`.
    pub fn seed(&mut self, seed: u64) -> Result<()> {
        self.key = Some(Array::random_key(&self.stream, seed)?);
        Ok(())
    }

    fn scalar(&self, value: f64) -> Result<Array> {
        Array::scalar(&self.stream, value as f32, MlxDType::F32)
    }

    /// Sample one token `[1, 1]` (uint32) from the last row of `logits`
    /// `[1, L, V]`. `history` enables the repetition penalty over its unique
    /// tokens; `suppress` lists ids set to -inf first.
    pub fn sample(&mut self, logits: &Array, history: &[u32], suppress: &[u32]) -> Result<Array> {
        let [1, l, width] = *logits.shape() else {
            unreachable!("logits are [1, L, V]")
        };
        let mut x = logits
            .cast(MlxDType::F32)?
            .slice(&[0, l - 1, 0], &[1, l, width])?
            .reshape(&[1, width])?;
        let ids = |v: &[u32]| -> Result<Array> {
            let v: Vec<i32> = v.iter().map(|&t| t as i32).collect();
            Array::from_i32(&self.stream, &[1, v.len()], &v)
        };
        if !suppress.is_empty() {
            let inf = self.scalar(f64::NEG_INFINITY)?;
            x = x.put_along_axis(&ids(suppress)?, &inf, -1)?;
        }
        let mut unique: Vec<u32> = history
            .iter()
            .copied()
            .filter(|&t| (t as usize) < width)
            .collect();
        unique.sort_unstable();
        unique.dedup();
        if !unique.is_empty() {
            let index = ids(&unique)?;
            let selected = x.take(&index.reshape(&[unique.len()])?, -1)?;
            let penalty = self.scalar(REPETITION_PENALTY)?;
            // where(x < 0, x * p, x / p); 0 <= x selects the division for -0 too.
            let penalized = self
                .scalar(0.0)?
                .less_equal(&selected)?
                .where_select(&selected.div(&penalty)?, &selected.mul(&penalty)?)?;
            x = x.put_along_axis(&index, &penalized, -1)?;
        }
        x = x.div(&self.scalar(TEMPERATURE)?)?;
        x = self.top_k.call(&[x])?.remove(0);
        let key = self.key.take().expect("sampler is seeded per request");
        let mut out = self.categorical.call(&[x, key])?;
        self.key = Some(out.pop().unwrap());
        out.pop().unwrap().reshape(&[1, 1])
    }
}
