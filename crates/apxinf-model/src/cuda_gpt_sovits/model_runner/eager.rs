//! Eager execution of the synthesis stages: operator calls plus the
//! conditioning cache.

use std::cell::RefCell;
use std::rc::Rc;

use apxinf_core::{Error, Result, Tensor};
use apxinf_cuda_new::transfers;

use super::cache::ConditioningCache;
use super::gpt_decode::{generate_semantic, GptExecutionState};
use crate::cuda_gpt_sovits::model::conditioning::{ReferenceFeatures, ReferenceKey, TextKey};
use crate::cuda_gpt_sovits::model::{ChunkArgs, GptSovitsModel, SynthesisExecutor};
use crate::cuda_gpt_sovits::ops;
use crate::cuda_gpt_sovits::weights::fp16_tensor;
use crate::tts_trait::GptSovitsInputs;

pub(crate) struct EagerExecutor<'a> {
    pub(crate) model: &'a GptSovitsModel,
    pub(crate) cache: &'a Rc<RefCell<ConditioningCache>>,
    pub(crate) gpt_state: &'a RefCell<GptExecutionState>,
}

impl EagerExecutor<'_> {
    fn bert_phone_features_inner(
        &self,
        token_ids: &[u32],
        word2ph: &[u32],
        phone_count: usize,
        cache_text: bool,
    ) -> Result<Tensor> {
        let key = TextKey::new(token_ids, word2ph, phone_count, false);
        if cache_text {
            if let Some(value) = self.cache.borrow_mut().text(&key) {
                return Ok(value);
            }
        }
        let hidden = self.model.bert.forward(&self.model.ctx, token_ids)?;
        if hidden.shape().dims()[0] != word2ph.len() {
            return Err(Error::Other(format!(
                "BERT token/word2ph mismatch: {} tokens vs {} entries",
                hidden.shape().dims()[0],
                word2ph.len()
            )));
        }
        let mut map: Vec<i32> = Vec::with_capacity(phone_count);
        for (index, count) in word2ph.iter().enumerate() {
            for _ in 0..*count {
                map.push(index as i32);
            }
        }
        if map.len() != phone_count {
            return Err(Error::Other(format!(
                "word2ph expands to {} phones, expected {phone_count}",
                map.len()
            )));
        }
        let ids = Tensor::from_i32(vec![map.len()], &map)?;
        let ids = transfers::to_cuda(&ids, self.model.ctx.device_id())?;
        let value = ops::gather_rows(&self.model.ctx, &hidden, &ids)?;
        if cache_text {
            self.cache.borrow_mut().insert_text(key, value.clone());
        }
        Ok(value)
    }

    fn codes_to_vec(&self, codes: &Tensor) -> Result<Vec<u32>> {
        let host = transfers::to_cpu(codes)?;
        let values = host.as_i32()?;
        Ok(values.iter().map(|value| *value as u32).collect())
    }
}

impl SynthesisExecutor for EagerExecutor<'_> {
    fn reference_features(&self, inputs: &GptSovitsInputs<'_>) -> Result<Rc<ReferenceFeatures>> {
        let key = ReferenceKey::new(inputs, false);
        if let Some(value) = self.cache.borrow_mut().reference(&key) {
            return Ok(value);
        }
        let frames = inputs.reference_mel.len() / 100;
        let spec_frames = inputs.reference_spec.len() / 704;
        let mel = fp16_tensor(&self.model.ctx, &[frames, 100], inputs.reference_mel)?;
        let spec = fp16_tensor(&self.model.ctx, &[spec_frames, 704], inputs.reference_spec)?;
        let bert = self.bert_phone_features_inner(
            inputs.reference_bert_token_ids,
            inputs.reference_word2ph,
            inputs.reference_phone_ids.len(),
            false,
        )?;
        let codes = self
            .model
            .hubert
            .encode(&self.model.ctx, inputs.reference_audio_16k)?;
        let prompt = self.codes_to_vec(&codes)?;
        let style = self.model.sovits.style(&self.model.ctx, &spec)?;
        let features = self.model.sovits.decode_features(
            &self.model.ctx,
            &codes,
            inputs.reference_phone_ids,
            &style,
            1.0,
        )?;
        let value = Rc::new(ReferenceFeatures {
            bert,
            prompt,
            style,
            features,
            mel,
        });
        self.cache
            .borrow_mut()
            .insert_reference(key, Rc::clone(&value));
        Ok(value)
    }

    fn target_bert(&self, inputs: &GptSovitsInputs<'_>) -> Result<Tensor> {
        self.bert_phone_features_inner(
            inputs.bert_token_ids,
            inputs.word2ph,
            inputs.phone_ids.len(),
            true,
        )
    }

    fn expected_target_frames(&self) -> Option<usize> {
        None
    }

    fn generate_semantic(
        &self,
        phones: &[u32],
        bert: &Tensor,
        prompt: &[u32],
        seed: u64,
        top_k: usize,
        top_p: f32,
        temperature: f32,
        repetition_penalty: f32,
        max_new_tokens: usize,
    ) -> Result<Vec<u32>> {
        generate_semantic(
            self.model.gpt.as_ref(),
            &self.model.ctx,
            &mut self.gpt_state.borrow_mut(),
            phones,
            bert,
            prompt,
            seed,
            top_k,
            top_p,
            temperature,
            repetition_penalty,
            max_new_tokens,
        )
    }

    fn target_features(
        &self,
        codes: &Tensor,
        phones: &[u32],
        style: &Tensor,
        speed_factor: f32,
    ) -> Result<Tensor> {
        self.model
            .sovits
            .decode_features(&self.model.ctx, codes, phones, style, speed_factor)
    }

    fn cfm_chunk(&self, args: ChunkArgs<'_>, steps: usize, seed: u64) -> Result<Tensor> {
        self.model.cfm.run_chunk(
            &self.model.ctx,
            args.prompt_features,
            args.target,
            args.prompt_mel,
            steps,
            seed,
        )
    }

    fn vocode(&self, mel: &Tensor) -> Result<Vec<f32>> {
        self.model.vocoder.synth(&self.model.ctx, mel)
    }
}
