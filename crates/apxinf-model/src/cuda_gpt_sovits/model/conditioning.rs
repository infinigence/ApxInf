//! Conditioning data shared between the eager and prepared execution paths.

use apxinf_core::Tensor;

use crate::tts_trait::GptSovitsInputs;

pub(crate) const TEXT_DEVICE_BUDGET: usize = 64 * 1024 * 1024;

#[derive(Clone, PartialEq, Eq)]
pub(crate) struct TextKey {
    ids: Vec<u32>,
    word2ph: Vec<u32>,
    phones: usize,
    prepared: bool,
}
impl TextKey {
    pub fn new(ids: &[u32], word2ph: &[u32], phones: usize, prepared: bool) -> Self {
        Self {
            ids: ids.to_vec(),
            word2ph: word2ph.to_vec(),
            phones,
            prepared,
        }
    }
    pub(crate) fn bytes(&self) -> usize {
        (self.ids.len() + self.word2ph.len()) * 4
    }
}

#[derive(Clone, PartialEq, Eq)]
pub(crate) struct ReferenceKey {
    phones: Vec<u32>,
    text: TextKey,
    audio: Vec<u32>,
    mel: Vec<u32>,
    spec: Vec<u32>,
}
impl ReferenceKey {
    pub fn new(request: &GptSovitsInputs<'_>, prepared: bool) -> Self {
        Self {
            phones: request.reference_phone_ids.to_vec(),
            text: TextKey::new(
                request.reference_bert_token_ids,
                request.reference_word2ph,
                request.reference_phone_ids.len(),
                prepared,
            ),
            audio: request
                .reference_audio_16k
                .iter()
                .map(|x| x.to_bits())
                .collect(),
            mel: request.reference_mel.iter().map(|x| x.to_bits()).collect(),
            spec: request.reference_spec.iter().map(|x| x.to_bits()).collect(),
        }
    }
    pub(crate) fn bytes(&self) -> usize {
        (self.phones.len() + self.audio.len() + self.mel.len() + self.spec.len()) * 4
            + self.text.bytes()
    }
}
pub(crate) struct ReferenceFeatures {
    pub bert: Tensor,
    pub prompt: Vec<u32>,
    pub style: Tensor,
    pub features: Tensor,
    pub mel: Tensor,
}
pub(crate) fn tensor_bytes(value: &Tensor) -> usize {
    value.shape().dims().iter().product::<usize>() * value.dtype().size_in_bytes()
}
impl ReferenceFeatures {
    pub(crate) fn device_bytes(&self) -> usize {
        [&self.bert, &self.style, &self.features, &self.mel]
            .into_iter()
            .map(tensor_bytes)
            .sum()
    }
}
