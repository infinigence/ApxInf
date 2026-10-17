//! mlx-audio `_sample_token` with the family-owned MLX key sequence.
//! `mx.random.seed` sets the key; each sample splits it, uses the second key
//! and keeps the first, exactly as MLX's global sequence does.
use apxinf_core::{Error, Result};
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
        let top_k = Compiled::new(stream, 1, |inputs| {
            Ok(vec![inputs[0].top_k_mask(TOP_K, -1)?])
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
        let indices = |tokens: &[u32]| -> Result<Vec<i32>> {
            tokens
                .iter()
                .map(|&token| {
                    i32::try_from(token)
                        .map_err(|_| Error::Contract("sampling token id exceeds i32"))
                })
                .collect()
        };
        if !suppress.is_empty() {
            let inf = self.scalar(f64::NEG_INFINITY)?;
            x = x.put_along_axis(&indices(suppress)?, &[1, suppress.len()], &inf, -1)?;
        }
        let mut unique: Vec<u32> = history
            .iter()
            .copied()
            .filter(|&t| (t as usize) < width)
            .collect();
        unique.sort_unstable();
        unique.dedup();
        if !unique.is_empty() {
            let token_indices = indices(&unique)?;
            let index = Array::from_i32(&self.stream, &[unique.len()], &token_indices)?;
            let selected = x.take(&index, -1)?;
            let penalty = self.scalar(REPETITION_PENALTY)?;
            // where(x < 0, x * p, x / p); 0 <= x selects the division for -0 too.
            let penalized = self
                .scalar(0.0)?
                .less_equal(&selected)?
                .where_select(&selected.div(&penalty)?, &selected.mul(&penalty)?)?;
            x = x.put_along_axis(&token_indices, &[1, unique.len()], &penalized, -1)?;
        }
        x = x.div(&self.scalar(TEMPERATURE)?)?;
        x = self.top_k.call(&[x])?.remove(0);
        let key = self.key.take().expect("sampler is seeded per request");
        let mut out = self.categorical.call(&[x, key])?;
        self.key = Some(out.pop().unwrap());
        out.pop().unwrap().reshape(&[1, 1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_sampling_matches_seeded_reference_after_rejection() -> Result<()> {
        let stream = Stream::cpu()?;
        let values: Vec<f32> = (0..64).map(|index| index as f32 * 0.11 - 3.1).collect();
        let logits = Array::from_f32(&stream, &[1, 1, values.len()], &values)?;
        let mut expected = values;
        expected[63] = f32::NEG_INFINITY;
        for token in [12, 62] {
            expected[token] = if expected[token] < 0.0 {
                expected[token] * REPETITION_PENALTY as f32
            } else {
                expected[token] / REPETITION_PENALTY as f32
            };
        }
        for value in &mut expected {
            *value /= TEMPERATURE as f32;
        }
        let mut ranked: Vec<usize> = (0..expected.len()).collect();
        ranked.sort_by(|&left, &right| expected[right].total_cmp(&expected[left]));
        for &index in &ranked[TOP_K..] {
            expected[index] = f32::NEG_INFINITY;
        }
        let expected = Array::from_f32(&stream, &[1, expected.len()], &expected)?;
        let mut sampler = Sampler::new(&stream)?;
        for seed in [0, 42, 43] {
            sampler.seed(seed)?;
            for invalid in [64, u32::MAX] {
                assert!(sampler.sample(&logits, &[], &[invalid]).is_err());
            }
            let (_, subkey) = Array::random_split(&Array::random_key(&stream, seed)?)?;
            let reference = expected.categorical(-1, &subkey)?;
            let actual = sampler.sample(&logits, &[12, 62, 62], &[63])?;
            assert_eq!(actual.to_bytes()?, reference.to_bytes()?);
            sampler.seed(seed)?;
            let repeated = sampler.sample(&logits, &[12, 62], &[63])?;
            assert_eq!(repeated.to_bytes()?, actual.to_bytes()?);
        }
        Ok(())
    }
}
