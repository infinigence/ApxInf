use apxinf_core::{
    CpuBackend, Error, NextTokenLogits, SamplingBackend, Tensor, TokenPenalties, TokenSample,
    TokenSampler, TokenSamplingInit, TokenSamplingSpec, TokenSelection,
};
use apxinf_tokenizer::Tokenizer;
use serde_json::Value;

const BUDGET: usize = 512;
const CLOSE_TOKEN: u32 = 9;
const CLOSE_TEXT: &str = "</think>";

pub(super) fn qualify_tokenizer(
    tokenizer: &Tokenizer,
    document: &Value,
    eos: &[u32],
) -> apxinf_core::Result<()> {
    let added = document["added_tokens"]
        .as_array()
        .ok_or_else(|| Error::Other("Bounded thinking requires added token 9".into()))?;
    let entries: Vec<_> = added
        .iter()
        .filter(|entry| entry["id"] == CLOSE_TOKEN)
        .collect();
    if entries.len() != 1
        || entries[0]["content"] != CLOSE_TEXT
        || entries[0]["special"] != false
        || CLOSE_TOKEN as usize >= tokenizer.vocab_size()
        || eos.contains(&CLOSE_TOKEN)
        || tokenizer.encode_with_special_tokens(CLOSE_TEXT, false)? != [CLOSE_TOKEN]
        || tokenizer.decode_with_options(&[CLOSE_TOKEN], false)? != CLOSE_TEXT
    {
        return Err(Error::Other(
            "Bounded thinking tokenizer identity differs".into(),
        ));
    }
    Ok(())
}

enum InitialReasoning {
    Reasoning { selected: usize, ids: Vec<u32> },
    Closed,
}

pub(super) struct BoundedThinkingSampler<'a> {
    cpu: Box<dyn TokenSampler>,
    tokenizer: &'a Tokenizer,
    eos: &'a [u32],
    constrained: Tensor,
    phase: InitialReasoning,
}

impl<'a> BoundedThinkingSampler<'a> {
    pub(super) fn new(
        spec: TokenSamplingSpec,
        tokenizer: &'a Tokenizer,
        eos: &'a [u32],
    ) -> apxinf_core::Result<Self> {
        if spec.vocab_size != tokenizer.vocab_size()
            || CLOSE_TOKEN as usize >= spec.vocab_size
            || eos.contains(&CLOSE_TOKEN)
        {
            return Err(Error::Other("Bounded thinking vocabulary differs".into()));
        }
        let mut row = vec![-f32::MAX; spec.vocab_size];
        row[CLOSE_TOKEN as usize] = 0.0;
        Ok(Self {
            cpu: CpuBackend.create_token_sampler(spec)?,
            tokenizer,
            eos,
            constrained: Tensor::from_f32(vec![spec.vocab_size], &row)?,
            phase: InitialReasoning::Reasoning {
                selected: 0,
                ids: Vec::with_capacity(BUDGET + 1),
            },
        })
    }
}

impl TokenSampler for BoundedThinkingSampler<'_> {
    fn spec(&self) -> TokenSamplingSpec {
        self.cpu.spec()
    }

    fn begin(&mut self, init: TokenSamplingInit<'_>) -> apxinf_core::Result<()> {
        init.params.validate(self.spec().vocab_size)?;
        if init.params.penalties != TokenPenalties::default() || init.params.return_logprob {
            return Err(Error::Other(
                "Bounded thinking requires neutral penalties and no log probabilities".into(),
            ));
        }
        if let TokenSelection::Random {
            temperature, top_k, ..
        } = init.params.selection
        {
            if temperature > 2.0 || top_k.is_some() {
                return Err(Error::Other(
                    "Bounded thinking requires temperature at most two and no top-k".into(),
                ));
            }
        }
        self.cpu.begin(init)?;
        self.phase = InitialReasoning::Reasoning {
            selected: 0,
            ids: Vec::with_capacity(BUDGET + 1),
        };
        Ok(())
    }

    fn sample(&mut self, logits: NextTokenLogits<'_>) -> apxinf_core::Result<TokenSample> {
        let forced = matches!(
            self.phase,
            InitialReasoning::Reasoning {
                selected: BUDGET,
                ..
            }
        );
        let sample = if forced {
            self.cpu.sample(NextTokenLogits::last(
                &self.constrained,
                self.spec().vocab_size,
            )?)?
        } else {
            self.cpu.sample(logits)?
        };
        if let InitialReasoning::Reasoning { selected, ids } = &mut self.phase {
            *selected += 1;
            if !self.eos.contains(&sample.token_id) {
                ids.push(sample.token_id);
            }
            let closed = self
                .tokenizer
                .decode_with_options(ids, false)?
                .contains(CLOSE_TEXT);
            if forced && (sample.token_id != CLOSE_TOKEN || !closed) {
                return Err(Error::Other(
                    "Bounded thinking failed to select a complete closing delimiter".into(),
                ));
            }
            if closed {
                self.phase = InitialReasoning::Closed;
            }
        }
        Ok(sample)
    }
}

#[cfg(test)]
#[path = "native_bounded_thinking_tests.rs"]
mod tests;
