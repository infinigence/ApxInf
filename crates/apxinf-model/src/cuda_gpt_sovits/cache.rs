//! Bounded conditioning reuse owned by one loaded TTS instance.
use crate::tts_trait::{TtsCacheConfig, TtsCacheInfo, TtsRequest};
use apxinf_core::Tensor;
use std::collections::VecDeque;
use std::rc::Rc;

const TEXT_DEVICE_BUDGET: usize = 64 * 1024 * 1024;

#[derive(Clone, PartialEq, Eq)]
pub(super) struct TextKey {
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
    fn bytes(&self) -> usize {
        (self.ids.len() + self.word2ph.len()) * 4
    }
}

#[derive(Clone, PartialEq, Eq)]
pub(super) struct ReferenceKey {
    phones: Vec<u32>,
    text: TextKey,
    audio: Vec<u32>,
    mel: Vec<u32>,
    spec: Vec<u32>,
}
impl ReferenceKey {
    pub fn new(request: &TtsRequest<'_>, prepared: bool) -> Self {
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
    fn bytes(&self) -> usize {
        (self.phones.len() + self.audio.len() + self.mel.len() + self.spec.len()) * 4
            + self.text.bytes()
    }
}
pub(super) struct ReferenceFeatures {
    pub bert: Tensor,
    pub prompt: Vec<u32>,
    pub style: Tensor,
    pub features: Tensor,
    pub mel: Tensor,
}
fn tensor_bytes(value: &Tensor) -> usize {
    value.shape().dims().iter().product::<usize>() * value.dtype().size_in_bytes()
}
impl ReferenceFeatures {
    fn device_bytes(&self) -> usize {
        [&self.bert, &self.style, &self.features, &self.mel]
            .into_iter()
            .map(tensor_bytes)
            .sum()
    }
}
pub(super) struct ConditioningCache {
    config: TtsCacheConfig,
    reference: Option<(ReferenceKey, Rc<ReferenceFeatures>)>,
    texts: VecDeque<(TextKey, Tensor)>,
    prompt_hits: u64,
    prompt_misses: u64,
    text_hits: u64,
    text_misses: u64,
}
impl Default for ConditioningCache {
    fn default() -> Self {
        Self {
            config: TtsCacheConfig::default(),
            reference: None,
            texts: VecDeque::new(),
            prompt_hits: 0,
            prompt_misses: 0,
            text_hits: 0,
            text_misses: 0,
        }
    }
}
impl ConditioningCache {
    pub fn configure(&mut self, config: TtsCacheConfig) {
        if config.prompt != self.config.prompt {
            self.reference = None;
        }
        if config.text != self.config.text {
            self.texts.clear();
        }
        self.config = config;
        while self.texts.len() > config.max_text_entries {
            self.texts.pop_front();
        }
    }
    pub fn clear(&mut self, reset_stats: bool) {
        self.reference = None;
        self.texts.clear();
        if reset_stats {
            self.prompt_hits = 0;
            self.prompt_misses = 0;
            self.text_hits = 0;
            self.text_misses = 0;
        }
    }
    pub fn reference(&mut self, key: &ReferenceKey) -> Option<Rc<ReferenceFeatures>> {
        if self.config.prompt {
            if let Some((saved, features)) = &self.reference {
                if saved == key {
                    self.prompt_hits += 1;
                    return Some(Rc::clone(features));
                }
            }
        }
        self.prompt_misses += 1;
        None
    }
    pub fn insert_reference(&mut self, key: ReferenceKey, features: Rc<ReferenceFeatures>) {
        if self.config.prompt {
            self.reference = Some((key, features));
        }
    }
    pub fn text(&mut self, key: &TextKey) -> Option<Tensor> {
        if self.config.text {
            if let Some(index) = self.texts.iter().position(|(saved, _)| saved == key) {
                let entry = self.texts.remove(index).expect("text cache entry");
                let value = entry.1.clone();
                self.texts.push_back(entry);
                self.text_hits += 1;
                return Some(value);
            }
        }
        self.text_misses += 1;
        None
    }
    pub fn insert_text(&mut self, key: TextKey, value: Tensor) {
        if !self.config.text
            || self.config.max_text_entries == 0
            || tensor_bytes(&value) > TEXT_DEVICE_BUDGET
        {
            return;
        }
        if let Some(index) = self.texts.iter().position(|(saved, _)| saved == &key) {
            self.texts.remove(index);
        }
        self.texts.push_back((key, value));
        while self.texts.len() > self.config.max_text_entries
            || self
                .texts
                .iter()
                .map(|(_, x)| tensor_bytes(x))
                .sum::<usize>()
                > TEXT_DEVICE_BUDGET
        {
            self.texts.pop_front();
        }
    }
    pub fn info(&self) -> TtsCacheInfo {
        let (ref_host, ref_device) = self.reference.as_ref().map_or((0, 0), |(key, value)| {
            (key.bytes() + value.prompt.len() * 4, value.device_bytes())
        });
        TtsCacheInfo {
            supported: true,
            prompt_enabled: self.config.prompt,
            text_enabled: self.config.text,
            prompt_entries: usize::from(self.reference.is_some()),
            text_entries: self.texts.len(),
            max_text_entries: self.config.max_text_entries,
            prompt_hits: self.prompt_hits,
            prompt_misses: self.prompt_misses,
            text_hits: self.text_hits,
            text_misses: self.text_misses,
            host_bytes: ref_host + self.texts.iter().map(|(key, _)| key.bytes()).sum::<usize>(),
            device_bytes: ref_device
                + self
                    .texts
                    .iter()
                    .map(|(_, x)| tensor_bytes(x))
                    .sum::<usize>(),
            text_device_budget_bytes: TEXT_DEVICE_BUDGET,
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn text_eviction_configuration_and_clear_preserve_accounting() {
        let mut cache = ConditioningCache::default();
        cache.configure(TtsCacheConfig {
            prompt: true,
            text: true,
            max_text_entries: 2,
        });
        let keys: Vec<_> = (0..3)
            .map(|id| TextKey::new(&[id], &[1], 1, false))
            .collect();
        for key in &keys[..2] {
            cache.insert_text(key.clone(), Tensor::from_f32(vec![1], &[1.]).unwrap());
        }
        assert!(cache.text(&keys[0]).is_some());
        cache.insert_text(keys[2].clone(), Tensor::from_f32(vec![1], &[1.]).unwrap());
        assert!(cache.text(&keys[1]).is_none());
        assert!(cache.text(&keys[0]).is_some());
        let info = cache.info();
        assert_eq!(info.text_hits, 2);
        assert_eq!(info.text_misses, 1);
        assert_eq!(info.device_bytes, 8);
        cache.configure(TtsCacheConfig {
            prompt: true,
            text: false,
            max_text_entries: 2,
        });
        assert_eq!(cache.info().text_entries, 0);
        assert!(cache.text(&keys[0]).is_none());
        cache.clear(false);
        assert_eq!(cache.info().text_hits, 2);
        cache.clear(true);
        assert_eq!(cache.info().text_hits, 0);
        assert_eq!(cache.info().device_bytes, 0);
        assert!(TextKey::new(&[0], &[1], 1, false) != TextKey::new(&[0], &[1], 1, true));
    }
}
