//! mlx-audio `_sample_token` with the family-owned MLX key sequence.
//! `mx.random.seed` sets the key; each sample splits it, uses the second key
//! and keeps the first, exactly as MLX's global sequence does.
use apxinf_core::{Error, Result};
use apxinf_mlx::{Array, Compiled, MlxDType, Stream};

const TEMPERATURE: f64 = 0.9;
const TOP_K: usize = 50;
const REPETITION_PENALTY: f64 = 1.05;

pub struct History {
    seen: Array,
    ids: Array,
}

impl History {
    pub fn new(stream: &Stream, width: usize) -> Result<Self> {
        Ok(Self {
            seen: Array::zeros(stream, &[1, width], MlxDType::U32)?,
            ids: Array::arange(stream, 0.0, width as f32, 1.0, MlxDType::U32)?,
        })
    }

    pub fn push(&mut self, token: &Array) -> Result<()> {
        let seen = self
            .seen
            .maximum(&token.equal(&self.ids)?.cast(MlxDType::U32)?)?;
        seen.eval()?;
        self.seen = seen;
        Ok(())
    }

    fn penalize(&self, logits: &Array) -> Result<Array> {
        if self.seen.shape() != logits.shape() {
            return Err(Error::Contract(
                "sampling history vocabulary does not match logits",
            ));
        }
        let penalty = Array::scalar(logits.stream(), REPETITION_PENALTY as f32, MlxDType::F32)?;
        let zero = Array::scalar(logits.stream(), 0.0, MlxDType::F32)?;
        let penalized = zero
            .less_equal(logits)?
            .where_select(&logits.div(&penalty)?, &logits.mul(&penalty)?)?;
        self.seen.where_select(&penalized, logits)
    }
}

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
        let sampler = Self {
            stream: stream.clone(),
            top_k,
            categorical,
            key: None,
        };
        for width in [super::config::VOCAB, super::config::PREDICTOR_VOCAB] {
            let logits = Array::zeros(stream, &[1, width], MlxDType::F32)?;
            drop(sampler.top_k.prepare(&[logits.clone()])?);
            drop(
                sampler
                    .categorical
                    .prepare(&[logits, Array::random_key(stream, 0)?])?,
            );
        }
        Ok(sampler)
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
    pub fn sample(
        &mut self,
        logits: &Array,
        history: Option<&History>,
        suppress: &[u32],
    ) -> Result<Array> {
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
        if let Some(history) = history {
            x = history.penalize(&x)?;
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

    fn check_history(stream: &Stream) -> Result<()> {
        let values = [-0.0, 0.0, -3.25, 2.75, f32::NEG_INFINITY, f32::MIN_POSITIVE];
        let logits = Array::from_f32(&stream, &[1, values.len()], &values)?;
        let mut history = History::new(&stream, values.len())?;
        let retained = history.seen.clone();
        for token in [0u32, 1, 2, 3, 4, 2, 6, u32::MAX] {
            let token = Array::from_bytes(&stream, &[1, 1], MlxDType::U32, &token.to_ne_bytes())?;
            token.eval()?;
            let before = stream.counters();
            history.push(&token)?;
            let after = stream.counters();
            assert_eq!(after.uploads_bytes, before.uploads_bytes);
            assert_eq!(after.downloads_bytes, before.downloads_bytes);
        }
        assert_eq!(retained.to_bytes()?, vec![0; values.len() * 4]);
        let expected: Vec<f32> = values
            .iter()
            .enumerate()
            .map(|(index, &value)| {
                if index == 5 {
                    value
                } else if value < 0.0 {
                    value * REPETITION_PENALTY as f32
                } else {
                    value / REPETITION_PENALTY as f32
                }
            })
            .collect();
        let actual = history.penalize(&logits)?.to_f32_vec()?;
        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        Ok(())
    }

    fn check_production_sampling(stream: &Stream) -> Result<()> {
        let mut sampler = Sampler::new(&stream)?;
        for width in [
            super::super::config::VOCAB,
            super::super::config::PREDICTOR_VOCAB,
        ] {
            let values: Vec<f32> = (0..width)
                .map(|index| (index % 67) as f32 * 0.13 - 3.0)
                .collect();
            let logits = Array::from_f32(&stream, &[1, 1, width], &values)?;
            logits.eval()?;
            for seed in [0, 42] {
                sampler.seed(seed)?;
                let before = stream.counters().trace_callbacks;
                let token = sampler.sample(&logits, None, &[])?;
                token.eval()?;
                assert_eq!(
                    stream.counters().trace_callbacks,
                    before,
                    "vocabulary {width}"
                );
                let scaled = logits
                    .reshape(&[1, width])?
                    .div(&sampler.scalar(TEMPERATURE)?)?;
                let masked = scaled.top_k_mask(TOP_K, -1)?;
                let (next, subkey) = Array::random_split(&Array::random_key(&stream, seed)?)?;
                assert_eq!(
                    token.to_bytes()?,
                    masked.categorical(-1, &subkey)?.to_bytes()?
                );
                assert_eq!(sampler.key.as_ref().unwrap().to_bytes()?, next.to_bytes()?);
            }
        }
        Ok(())
    }

    #[test]
    fn device_history_preserves_unique_membership_and_penalty_bits() -> Result<()> {
        check_history(&Stream::cpu()?)
    }

    #[test]
    fn production_sampling_profiles_are_prepared_without_consuming_request_keys() -> Result<()> {
        check_production_sampling(&Stream::cpu()?)
    }

    #[test]
    #[ignore = "requires native Apple Metal"]
    fn history_and_sampling_match_reference_metal() -> Result<()> {
        let stream = Stream::metal(0)?;
        check_history(&stream)?;
        check_production_sampling(&stream)
    }

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
                assert!(sampler.sample(&logits, None, &[invalid]).is_err());
            }
            let (_, subkey) = Array::random_split(&Array::random_key(&stream, seed)?)?;
            let reference = expected.categorical(-1, &subkey)?;
            let mut history = History::new(&stream, logits.shape()[2])?;
            for token in [12i32, 62, 62] {
                history.push(&Array::from_i32(&stream, &[1, 1], &[token])?.cast(MlxDType::U32)?)?;
            }
            let actual = sampler.sample(&logits, Some(&history), &[63])?;
            assert_eq!(actual.to_bytes()?, reference.to_bytes()?);
            sampler.seed(seed)?;
            let repeated = sampler.sample(&logits, Some(&history), &[63])?;
            assert_eq!(repeated.to_bytes()?, actual.to_bytes()?);
        }
        Ok(())
    }
}
