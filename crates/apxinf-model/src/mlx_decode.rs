use crate::llm_trait::{GeneratedToken, TokenSequence};
use apxinf_core::{Error, Result, TokenPenalties, TokenSamplingParams, TokenSelection};
use apxinf_mlx::{greedy_token_arrays, Array, Stream};

pub(crate) fn enabled(params: &TokenSamplingParams) -> bool {
    params.selection == TokenSelection::Greedy
        && params.penalties == TokenPenalties::default()
        && !params.return_logprob
        && std::env::var_os("APXINF_MLX_SYNC_DECODE").is_none_or(|value| value != "1")
}

pub(crate) enum DecodeFinish {
    PublishedPosition(usize),
    Invalidated,
}

pub(crate) trait DeviceDecode {
    fn stream(&self) -> &Stream;
    /// Evaluating the returned logits must evaluate every state array updated by this step.
    fn decode(&mut self, token: &Array) -> Result<Array>;
    fn finish_decode(&mut self, outcome: DecodeFinish) -> Result<()>;
}

struct PendingToken {
    feed: Array,
    checked: Array,
}

impl PendingToken {
    fn new(logits: &Array) -> Result<Self> {
        let (feed, checked) = greedy_token_arrays(logits)?;
        Ok(Self { feed, checked })
    }

    fn read(&self) -> Result<GeneratedToken> {
        let bytes = self.checked.to_bytes_ready()?;
        let token_id = u32::from_ne_bytes(
            bytes
                .try_into()
                .map_err(|_| Error::Contract("greedy token must contain one u32"))?,
        );
        if token_id == u32::MAX {
            return Err(Error::Other("all token logits are invalid".into()));
        }
        Ok(GeneratedToken {
            token_id,
            logprob: None,
        })
    }
}

pub(crate) struct GreedySequence<'a, M: DeviceDecode> {
    model: &'a mut M,
    current: Option<PendingToken>,
    successor: Option<PendingToken>,
    prompt_end: usize,
    failed: bool,
    finished: bool,
}

impl<'a, M: DeviceDecode> GreedySequence<'a, M> {
    pub(crate) fn new(model: &'a mut M, first: Array, prompt_end: u32) -> Result<Self> {
        let mut sequence = Self {
            model,
            current: None,
            successor: None,
            prompt_end: prompt_end as usize,
            failed: false,
            finished: false,
        };
        sequence.current = Some(PendingToken::new(&first)?);
        sequence.model.stream().async_eval(&[sequence
            .current
            .as_ref()
            .unwrap()
            .checked
            .clone()])?;
        Ok(sequence)
    }

    fn advance(&mut self, lookahead: bool) -> Result<GeneratedToken> {
        let current = self
            .current
            .as_ref()
            .ok_or(Error::Contract("token sequence exhausted"))?;
        if lookahead {
            let logits = self.model.decode(&current.feed)?;
            self.successor = Some(PendingToken::new(&logits)?);
            let outputs = [self.successor.as_ref().unwrap().checked.clone()];
            self.model.stream().async_eval(&outputs)?;
        }
        let token = current.read()?;
        self.current = self.successor.take();
        Ok(token)
    }
}

impl<M: DeviceDecode> TokenSequence for GreedySequence<'_, M> {
    fn next(&mut self, lookahead: bool) -> Result<GeneratedToken> {
        let result = self.advance(lookahead);
        self.failed |= result.is_err();
        result
    }

    fn finish(mut self: Box<Self>, published: usize) -> Result<()> {
        let outcome = if self.failed {
            DecodeFinish::Invalidated
        } else {
            DecodeFinish::PublishedPosition(self.prompt_end + published.saturating_sub(1))
        };
        let result = self.model.finish_decode(outcome);
        self.finished = true;
        result
    }

    fn overlaps_callbacks(&self) -> bool {
        true
    }
}

impl<M: DeviceDecode> Drop for GreedySequence<'_, M> {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.model.finish_decode(DecodeFinish::Invalidated);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GenerationCancellation, GenerationOptions, LlmInput, LlmTrait, LoadedModel};
    use apxinf_core::{Backend, CpuBackend, Device, Tensor, TokenSampler};
    use apxinf_loader::ModelConfig;
    use std::{cell::RefCell, collections::HashMap, rc::Rc};

    #[derive(Default)]
    struct Observed {
        position: usize,
        enqueues: Vec<(usize, usize)>,
        finishes: Vec<Option<usize>>,
        resets: usize,
    }

    struct Model {
        stream: Stream,
        observed: Rc<RefCell<Observed>>,
        prompt: usize,
        tokens: Vec<Option<u32>>,
        fail_decode: bool,
    }

    impl Model {
        fn logits(&self, index: usize) -> Result<Array> {
            let mut values = [f32::NEG_INFINITY; 4];
            if let Some(token) = self.tokens[index.min(self.tokens.len() - 1)] {
                values[token as usize] = 1.;
            }
            Array::from_f32(&self.stream, &[1, 4], &values)
        }
    }

    impl DeviceDecode for Model {
        fn stream(&self) -> &Stream {
            &self.stream
        }
        fn decode(&mut self, _token: &Array) -> Result<Array> {
            let position = self.observed.borrow().position;
            self.observed
                .borrow_mut()
                .enqueues
                .push((position, self.stream.counters().downloads_bytes));
            if self.fail_decode {
                return Err(Error::Other("injected decode failure".into()));
            }
            self.observed.borrow_mut().position += 1;
            self.logits(position - self.prompt + 1)
        }
        fn finish_decode(&mut self, outcome: DecodeFinish) -> Result<()> {
            self.stream.synchronize()?;
            let position = match outcome {
                DecodeFinish::PublishedPosition(position) => Some(position),
                DecodeFinish::Invalidated => None,
            };
            let mut observed = self.observed.borrow_mut();
            observed.finishes.push(position);
            if let Some(position) = position {
                assert!(position <= observed.position && observed.position - position <= 1);
                observed.position = position;
            }
            Ok(())
        }
    }

    impl LlmTrait for Model {
        fn load(_: ModelConfig, _: HashMap<String, Tensor>, _: Device) -> Result<Self> {
            unreachable!()
        }
        fn forward(&mut self, _: &[u32], _: u32) -> Result<Tensor> {
            unreachable!()
        }
        fn backend(&self) -> &dyn Backend {
            &CpuBackend
        }
        fn reset(&mut self) {
            self.observed.borrow_mut().resets += 1;
        }
        fn vocab_size(&self) -> usize {
            4
        }
        fn prefill(&mut self, input: LlmInput<'_>) -> Result<Tensor> {
            self.prompt = input.token_ids.len();
            self.observed.borrow_mut().position = self.prompt;
            Tensor::from_f32(vec![1, 4], &[0.; 4])
        }
        fn token_sequence<'a>(
            &'a mut self,
            _: Tensor,
            _: Box<dyn TokenSampler>,
            _: &TokenSamplingParams,
            prompt_end: u32,
        ) -> Result<Box<dyn TokenSequence + 'a>> {
            let first = self.logits(0)?;
            Ok(Box::new(GreedySequence::new(self, first, prompt_end)?))
        }
    }

    fn loaded(tokens: Vec<Option<u32>>, fail_decode: bool) -> (LoadedModel, Rc<RefCell<Observed>>) {
        let observed = Rc::new(RefCell::new(Observed::default()));
        let model = Model {
            stream: Stream::cpu().unwrap(),
            observed: observed.clone(),
            prompt: 0,
            tokens,
            fail_decode,
        };
        (LoadedModel::text(Box::new(model)), observed)
    }

    #[test]
    fn cpu_pipeline_submits_before_read_and_stops_at_budget() {
        let (mut model, observed) = loaded(vec![Some(2), Some(3), Some(1)], false);
        let mut callbacks = vec![];
        let output = model
            .generate_streaming_with_options(
                LlmInput::text(&[0, 1]),
                &GenerationOptions::greedy(3, None),
                |token| callbacks.push(token.token_id),
            )
            .unwrap();
        assert_eq!(output.token_ids(), vec![2, 3, 1]);
        assert_eq!(callbacks, output.token_ids());
        let state = observed.borrow();
        assert_eq!(state.enqueues, vec![(2, 0), (3, 4)]);
        assert_eq!(state.finishes, vec![Some(4)]);
        assert_eq!(state.position, 4);
    }

    #[test]
    fn cpu_pipeline_eos_discards_one_step_and_ignores_unused_invalid_logits() {
        for eos in [2, 3] {
            let tokens = if eos == 2 {
                vec![Some(2), None]
            } else {
                vec![Some(2), Some(3), None]
            };
            let (mut model, observed) = loaded(tokens, false);
            let output = model
                .generate_streaming_with_options(
                    LlmInput::text(&[0, 1]),
                    &GenerationOptions::greedy(8, Some(eos)),
                    |_| {},
                )
                .unwrap();
            let expected = if eos == 2 { vec![2] } else { vec![2, 3] };
            assert_eq!(output.token_ids(), expected);
            let state = observed.borrow();
            assert_eq!(state.enqueues.len(), expected.len());
            assert_eq!(state.finishes, vec![Some(2 + expected.len() - 1)]);
            assert_eq!(state.position, 2 + expected.len() - 1);
        }
    }

    #[test]
    fn cpu_pipeline_cancel_returns_only_published_prefix() {
        let (mut model, observed) = loaded(vec![Some(2), Some(3), Some(1)], false);
        let cancel = GenerationCancellation::default();
        let mut callbacks = vec![];
        let output = model
            .generate_streaming_cancellable(
                LlmInput::text(&[0, 1]),
                &GenerationOptions::greedy(8, None),
                &cancel,
                |token| {
                    callbacks.push(token.token_id);
                    let external = cancel.clone();
                    std::thread::spawn(move || external.cancel())
                        .join()
                        .unwrap();
                },
            )
            .unwrap();
        assert_eq!(callbacks, vec![2]);
        assert_eq!(output.token_ids(), callbacks);
        assert_eq!(observed.borrow().finishes, vec![Some(2)]);
        let output = model
            .generate_streaming_cancellable(
                LlmInput::text(&[0, 1]),
                &GenerationOptions::greedy(8, None),
                &cancel,
                |_| panic!("cancelled callback"),
            )
            .unwrap();
        assert!(output.tokens.is_empty());
        assert_eq!(observed.borrow().resets, 1);
    }

    #[test]
    fn cpu_pipeline_errors_drain_and_invalidate_once() {
        for fail_decode in [false, true] {
            let (mut model, observed) = loaded(vec![None, Some(2)], fail_decode);
            let error = model
                .generate_streaming_with_options(
                    LlmInput::text(&[0, 1]),
                    &GenerationOptions::greedy(8, None),
                    |_| panic!("invalid token published"),
                )
                .err()
                .unwrap();
            assert!(error.to_string().contains(if fail_decode {
                "injected decode failure"
            } else {
                "all token logits are invalid"
            }));
            assert_eq!(observed.borrow().finishes, vec![None]);
        }
    }
}
