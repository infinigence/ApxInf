use apxinf_core::{
    standard_normal_f32, uniform_f32, DType, Device, Error, NextTokenLogits, NormalGenerator,
    Result, RngKey, Tensor, TokenPenalties, TokenSample, TokenSampler, TokenSamplingInit,
    TokenSamplingParams, TokenSamplingSpec, TokenSelection,
};
use half::{bf16, f16};

/// Apply history penalties to one logit using vLLM-compatible sign handling
/// for the repetition penalty.
pub(crate) fn adjusted_logit(mut value: f32, occurrences: u32, penalties: TokenPenalties) -> f32 {
    if value.is_nan() {
        return f32::NEG_INFINITY;
    }
    if value == f32::INFINITY {
        value = f32::MAX;
    }
    if occurrences != 0 {
        if penalties.repetition != 1.0 {
            value = if value < 0.0 {
                value * penalties.repetition
            } else {
                value / penalties.repetition
            };
        }
        value -= penalties.frequency * occurrences as f32;
        value -= penalties.presence;
    }
    value
}

pub(crate) struct CpuTokenSampler {
    spec: TokenSamplingSpec,
    counts: Vec<u32>,
    sequence_len: usize,
    params: Option<TokenSamplingParams>,
    rng: RngKey,
}

impl CpuTokenSampler {
    pub(crate) fn new(spec: TokenSamplingSpec) -> Result<Self> {
        spec.validate()?;
        Ok(Self {
            spec,
            counts: vec![0; spec.vocab_size],
            sequence_len: 0,
            params: None,
            rng: RngKey::default(),
        })
    }

    fn adjusted_row(&self, logits: NextTokenLogits<'_>) -> Result<Vec<f32>> {
        if logits.vocab_size() != self.spec.vocab_size {
            return Err(Error::Other(format!(
                "sampler vocabulary is {}, logits vocabulary is {}",
                self.spec.vocab_size,
                logits.vocab_size()
            )));
        }
        if logits.tensor().device() != Device::Cpu {
            return Err(Error::DeviceMismatch {
                expected: Device::Cpu,
                got: logits.tensor().device(),
            });
        }
        let data = logits.tensor().to_f32_vec()?;
        let start = logits.row_offset();
        let penalties = self.params.as_ref().expect("sampler initialized").penalties;
        Ok(data[start..start + self.spec.vocab_size]
            .iter()
            .zip(&self.counts)
            .map(|(&value, &count)| adjusted_logit(value, count, penalties))
            .collect())
    }

    fn greedy(&self, logits: &[f32], return_logprob: bool) -> Result<TokenSample> {
        let mut best_value = f32::NEG_INFINITY;
        let mut best_id = None;
        for (token_id, &value) in logits.iter().enumerate() {
            if value > best_value {
                best_value = value;
                best_id = Some(token_id as u32);
            }
        }
        let token_id =
            best_id.ok_or_else(|| Error::Other("all token logits are invalid".into()))?;
        if !best_value.is_finite() {
            return Err(Error::Other("all token logits are invalid".into()));
        }
        let logprob = return_logprob.then(|| {
            let sum = logits
                .iter()
                .filter(|value| value.is_finite())
                .map(|value| (*value - best_value).exp())
                .sum::<f32>();
            -sum.ln()
        });
        Ok(TokenSample { token_id, logprob })
    }

    fn random(
        &self,
        logits: &[f32],
        temperature: f32,
        top_k: Option<usize>,
        top_p: f32,
        return_logprob: bool,
    ) -> Result<TokenSample> {
        let mut ranked = logits
            .iter()
            .enumerate()
            .filter_map(|(token_id, &value)| {
                value
                    .is_finite()
                    .then_some((token_id as u32, value / temperature))
            })
            .collect::<Vec<_>>();
        ranked.sort_by(|(left_id, left), (right_id, right)| {
            right.total_cmp(left).then_with(|| left_id.cmp(right_id))
        });
        if let Some(top_k) = top_k {
            ranked.truncate(top_k);
        }
        let max = ranked
            .first()
            .map(|(_, value)| *value)
            .ok_or_else(|| Error::Other("all token logits are invalid".into()))?;
        let mut candidates = ranked
            .into_iter()
            .map(|(token_id, value)| (token_id, (value - max).exp()))
            .collect::<Vec<_>>();
        let total = candidates.iter().map(|(_, weight)| *weight).sum::<f32>();
        if !total.is_finite() || total <= 0.0 {
            return Err(Error::Other("token probability mass is invalid".into()));
        }

        let nucleus_target = top_p * total;
        let mut nucleus_len = candidates.len();
        let mut cumulative = 0.0f32;
        for (index, (_, weight)) in candidates.iter().enumerate() {
            cumulative += *weight;
            if cumulative >= nucleus_target {
                nucleus_len = index + 1;
                break;
            }
        }
        candidates.truncate(nucleus_len);
        let nucleus_total = candidates.iter().map(|(_, weight)| *weight).sum::<f32>();
        let target = uniform_f32(self.rng, 0) * nucleus_total;
        let mut cumulative = 0.0f32;
        let mut selected = *candidates.last().expect("non-empty nucleus");
        for candidate in &candidates {
            cumulative += candidate.1;
            if target < cumulative {
                selected = *candidate;
                break;
            }
        }
        Ok(TokenSample {
            token_id: selected.0,
            logprob: return_logprob.then(|| (selected.1 / nucleus_total).ln()),
        })
    }
}

impl TokenSampler for CpuTokenSampler {
    fn spec(&self) -> TokenSamplingSpec {
        self.spec
    }

    fn begin(&mut self, init: TokenSamplingInit<'_>) -> Result<()> {
        init.params.validate(self.spec.vocab_size)?;
        if init.prompt_token_ids.len() > self.spec.max_sequence_len {
            return Err(Error::Other(format!(
                "prompt length {} exceeds sampler capacity {}",
                init.prompt_token_ids.len(),
                self.spec.max_sequence_len
            )));
        }
        self.counts.fill(0);
        for &token_id in init.prompt_token_ids {
            let count = self.counts.get_mut(token_id as usize).ok_or_else(|| {
                Error::Other(format!(
                    "prompt token {token_id} is outside vocabulary {}",
                    self.spec.vocab_size
                ))
            })?;
            *count = count
                .checked_add(1)
                .ok_or_else(|| Error::Other("token occurrence count overflow".into()))?;
        }
        self.sequence_len = init.prompt_token_ids.len();
        self.params = Some(init.params.clone());
        self.rng = init.rng;
        Ok(())
    }

    fn sample(&mut self, logits: NextTokenLogits<'_>) -> Result<TokenSample> {
        if self.params.is_none() {
            return Err(Error::Other(
                "token sampler must be initialized with begin()".into(),
            ));
        }
        if self.sequence_len >= self.spec.max_sequence_len {
            return Err(Error::Other(format!(
                "token sampler reached sequence capacity {}",
                self.spec.max_sequence_len
            )));
        }
        let mut next_rng = self.rng;
        next_rng.advance()?;
        let row = self.adjusted_row(logits)?;
        let params = self.params.as_ref().expect("sampler initialized");
        let sample = match params.selection {
            TokenSelection::Greedy => self.greedy(&row, params.return_logprob)?,
            TokenSelection::Random {
                temperature,
                top_k,
                top_p,
            } => self.random(&row, temperature, top_k, top_p, params.return_logprob)?,
        };
        self.counts[sample.token_id as usize] = self.counts[sample.token_id as usize]
            .checked_add(1)
            .ok_or_else(|| Error::Other("token occurrence count overflow".into()))?;
        self.sequence_len += 1;
        self.rng = next_rng;
        Ok(sample)
    }
}

pub(crate) struct CpuNormalGenerator {
    output: Tensor,
}

impl CpuNormalGenerator {
    pub(crate) fn new(output: Tensor) -> Result<Self> {
        if output.device() != Device::Cpu {
            return Err(Error::DeviceMismatch {
                expected: Device::Cpu,
                got: output.device(),
            });
        }
        if !matches!(output.dtype(), DType::F32 | DType::F16 | DType::BF16) {
            return Err(Error::Other(format!(
                "standard-normal generation does not support {}",
                output.dtype()
            )));
        }
        Ok(Self { output })
    }
}

impl NormalGenerator for CpuNormalGenerator {
    fn output(&self) -> &Tensor {
        &self.output
    }

    fn generate(&mut self, rng: RngKey) -> Result<&Tensor> {
        let values = standard_normal_f32(self.output.numel(), rng);
        match self.output.dtype() {
            DType::F32 => self.output.as_f32_mut()?.copy_from_slice(&values),
            DType::F16 => {
                let bytes = self.output.as_cpu_bytes_mut()?;
                bytes
                    .chunks_exact_mut(2)
                    .zip(values)
                    .for_each(|(output, value)| {
                        output.copy_from_slice(&f16::from_f32(value).to_bits().to_ne_bytes())
                    });
            }
            DType::BF16 => {
                let bytes = self.output.as_cpu_bytes_mut()?;
                bytes
                    .chunks_exact_mut(2)
                    .zip(values)
                    .for_each(|(output, value)| {
                        output.copy_from_slice(&bf16::from_f32(value).to_bits().to_ne_bytes())
                    });
            }
            dtype => {
                return Err(Error::Other(format!(
                    "standard-normal generation does not support {dtype}"
                )))
            }
        }
        Ok(&self.output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sampler(vocab_size: usize, max_sequence_len: usize) -> Box<dyn TokenSampler> {
        Box::new(
            CpuTokenSampler::new(TokenSamplingSpec {
                vocab_size,
                max_sequence_len,
            })
            .unwrap(),
        )
    }

    #[test]
    fn greedy_uses_last_row_and_lowest_id_for_ties() {
        let tensor =
            Tensor::from_f32(vec![2, 4], &[100.0, 0.0, 0.0, 0.0, 1.0, 4.0, 4.0, 2.0]).unwrap();
        let params = TokenSamplingParams::greedy();
        let mut sampler = sampler(4, 8);
        sampler
            .begin(TokenSamplingInit {
                prompt_token_ids: &[0],
                params: &params,
                rng: RngKey::default(),
            })
            .unwrap();
        assert_eq!(
            sampler
                .sample(NextTokenLogits::last(&tensor, 4).unwrap())
                .unwrap()
                .token_id,
            1
        );
    }

    #[test]
    fn greedy_logprob_is_stable_for_large_logits() {
        let tensor = Tensor::from_f32(vec![1, 3], &[f32::MAX, f32::MAX, 0.0]).unwrap();
        let params = TokenSamplingParams {
            return_logprob: true,
            ..TokenSamplingParams::greedy()
        };
        let mut sampler = sampler(3, 4);
        sampler
            .begin(TokenSamplingInit {
                prompt_token_ids: &[2],
                params: &params,
                rng: RngKey::default(),
            })
            .unwrap();
        let sample = sampler
            .sample(NextTokenLogits::last(&tensor, 3).unwrap())
            .unwrap();
        assert_eq!(sample.token_id, 0);
        assert!((sample.logprob.unwrap() + 2.0f32.ln()).abs() < 1e-6);
    }

    #[test]
    fn repetition_frequency_and_presence_penalties_affect_selection() {
        let tensor = Tensor::from_f32(vec![1, 3], &[4.0, 3.5, 1.0]).unwrap();
        let params = TokenSamplingParams {
            selection: TokenSelection::Greedy,
            penalties: TokenPenalties {
                repetition: 2.0,
                frequency: 0.5,
                presence: 0.25,
            },
            return_logprob: false,
        };
        let mut sampler = sampler(3, 8);
        sampler
            .begin(TokenSamplingInit {
                prompt_token_ids: &[0, 0],
                params: &params,
                rng: RngKey::default(),
            })
            .unwrap();
        assert_eq!(
            sampler
                .sample(NextTokenLogits::last(&tensor, 3).unwrap())
                .unwrap()
                .token_id,
            1
        );
    }

    #[test]
    fn generated_tokens_are_added_to_penalty_history() {
        let tensor = Tensor::from_f32(vec![1, 3], &[3.0, 2.5, 0.0]).unwrap();
        let params = TokenSamplingParams {
            selection: TokenSelection::Greedy,
            penalties: TokenPenalties {
                repetition: 2.0,
                ..TokenPenalties::default()
            },
            return_logprob: false,
        };
        let mut sampler = sampler(3, 4);
        sampler
            .begin(TokenSamplingInit {
                prompt_token_ids: &[2],
                params: &params,
                rng: RngKey::default(),
            })
            .unwrap();

        let logits = NextTokenLogits::last(&tensor, 3).unwrap();
        assert_eq!(sampler.sample(logits).unwrap().token_id, 0);
        assert_eq!(sampler.sample(logits).unwrap().token_id, 1);
    }

    #[test]
    fn top_p_excludes_the_probability_tail() {
        let tensor = Tensor::from_f32(vec![1, 4], &[0.0, 0.0, 0.0, 0.0]).unwrap();
        let params = TokenSamplingParams {
            selection: TokenSelection::Random {
                temperature: 1.0,
                top_k: None,
                top_p: 0.25,
            },
            penalties: TokenPenalties::default(),
            return_logprob: true,
        };
        let mut sampler = sampler(4, 4);
        sampler
            .begin(TokenSamplingInit {
                prompt_token_ids: &[3],
                params: &params,
                rng: RngKey::new(99, 7, 0),
            })
            .unwrap();

        let sample = sampler
            .sample(NextTokenLogits::last(&tensor, 4).unwrap())
            .unwrap();
        assert_eq!(sample.token_id, 0);
        assert_eq!(sample.logprob, Some(0.0));
    }

    #[test]
    fn top_k_one_matches_greedy() {
        let tensor = Tensor::from_f32(vec![1, 4], &[0.0, 3.0, 2.0, 1.0]).unwrap();
        let params = TokenSamplingParams {
            selection: TokenSelection::Random {
                temperature: 0.7,
                top_k: Some(1),
                top_p: 1.0,
            },
            penalties: TokenPenalties::default(),
            return_logprob: true,
        };
        let mut sampler = sampler(4, 8);
        sampler
            .begin(TokenSamplingInit {
                prompt_token_ids: &[0],
                params: &params,
                rng: RngKey::new(7, 3, 1),
            })
            .unwrap();
        let sample = sampler
            .sample(NextTokenLogits::last(&tensor, 4).unwrap())
            .unwrap();
        assert_eq!(sample.token_id, 1);
        assert_eq!(sample.logprob, Some(0.0));
    }

    #[test]
    fn seeded_sampling_is_reproducible_and_sequence_isolated() {
        let tensor = Tensor::from_f32(vec![1, 4], &[0.0, 0.0, 0.0, 0.0]).unwrap();
        let params = TokenSamplingParams {
            selection: TokenSelection::Random {
                temperature: 1.0,
                top_k: None,
                top_p: 1.0,
            },
            penalties: TokenPenalties::default(),
            return_logprob: false,
        };
        let run = |key| {
            let mut sampler = sampler(4, 8);
            sampler
                .begin(TokenSamplingInit {
                    prompt_token_ids: &[0],
                    params: &params,
                    rng: key,
                })
                .unwrap();
            (0..4)
                .map(|_| {
                    sampler
                        .sample(NextTokenLogits::last(&tensor, 4).unwrap())
                        .unwrap()
                        .token_id
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(run(RngKey::new(42, 9, 0)), run(RngKey::new(42, 9, 0)));
        assert_ne!(run(RngKey::new(42, 9, 0)), run(RngKey::new(42, 10, 0)));
    }

    #[test]
    fn parameter_and_history_validation_is_strict() {
        let invalid = TokenSamplingParams {
            selection: TokenSelection::Random {
                temperature: 0.0,
                top_k: Some(0),
                top_p: 1.5,
            },
            ..TokenSamplingParams::default()
        };
        assert!(invalid.validate(4).is_err());

        let params = TokenSamplingParams::default();
        let mut sampler = sampler(4, 2);
        assert!(sampler
            .begin(TokenSamplingInit {
                prompt_token_ids: &[4],
                params: &params,
                rng: RngKey::default(),
            })
            .is_err());
    }

    #[test]
    fn normal_generator_is_reproducible_and_well_formed() {
        let mut generator =
            CpuNormalGenerator::new(Tensor::zeros(vec![100_000], DType::F32)).unwrap();
        let output_address = generator.output().as_f32().unwrap().as_ptr();
        let key = RngKey::new(123, 4, 5);
        let first = generator.generate(key).unwrap().as_f32().unwrap().to_vec();
        assert_eq!(
            generator.output().as_f32().unwrap().as_ptr(),
            output_address
        );
        let second = generator.generate(key).unwrap().as_f32().unwrap().to_vec();
        assert_eq!(first, second);
        assert!(first.iter().all(|value| value.is_finite()));
        let mean = first.iter().sum::<f32>() / first.len() as f32;
        let variance = first
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f32>()
            / first.len() as f32;
        assert!(mean.abs() < 0.02, "mean={mean}");
        assert!((variance - 1.0).abs() < 0.03, "variance={variance}");
    }

    #[test]
    fn normal_generator_supports_fp16_and_bf16_outputs() {
        let key = RngKey::new(5, 6, 7);
        let expected = standard_normal_f32(17, key);
        for dtype in [DType::F16, DType::BF16] {
            let mut generator = CpuNormalGenerator::new(Tensor::zeros(vec![17], dtype)).unwrap();
            let actual = generator.generate(key).unwrap().to_f32_vec().unwrap();
            for (actual, expected) in actual.iter().zip(&expected) {
                let tolerance = if dtype == DType::F16 { 0.002 } else { 0.02 };
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "dtype={dtype}, actual={actual}, expected={expected}"
                );
            }
        }
    }

    #[test]
    fn all_invalid_logits_are_rejected() {
        let tensor =
            Tensor::from_f32(vec![1, 3], &[f32::NAN, f32::NEG_INFINITY, f32::NAN]).unwrap();
        let params = TokenSamplingParams::default();
        let mut sampler = sampler(3, 4);
        sampler
            .begin(TokenSamplingInit {
                prompt_token_ids: &[0],
                params: &params,
                rng: RngKey::default(),
            })
            .unwrap();
        assert!(sampler
            .sample(NextTokenLogits::last(&tensor, 3).unwrap())
            .is_err());
    }
}
