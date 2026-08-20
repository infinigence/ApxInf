//! Model hyperparameters parsed from weight file metadata.

/// Configuration for a Llama-family model.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
}

impl ModelConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.n_heads
    }

    /// TinyLlama 1.1B defaults.
    pub fn tinyllama() -> Self {
        Self {
            hidden_size: 2048,
            intermediate_size: 5632,
            n_layers: 22,
            n_heads: 32,
            n_kv_heads: 4,
            vocab_size: 32000,
            max_seq_len: 2048,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
        }
    }
}
