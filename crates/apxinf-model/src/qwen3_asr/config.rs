//! qwen3-asr-1.7B: fixed checkpoint geometry (validated at load by shapes).

/// Audio sample rate (Hz).
pub const SAMPLE_RATE: u32 = 16_000;
/// One 30 s window of canonical input.
pub const SAMPLES: usize = 480_000;
/// Mel bins (also the conv H axis input size).
pub const MEL_BINS: usize = 128;
/// Mel frames per window (fixed 30 s contract).
pub const FRAMES: usize = 3_000;
/// Mel frames per encoder chunk (`n_window * 2`).
pub const CHUNK_FRAMES: usize = 100;
/// Number of chunks in one window.
pub const N_CHUNKS: usize = FRAMES / CHUNK_FRAMES;
/// Inference attention window, in raw mel frames.
pub const N_WINDOW_INFER: usize = 800;

/// Post-conv3 time steps per full chunk.
pub const POST_CNN_TIME: usize = 13;
/// Post-conv3 frequency bins per full chunk (128 -> 64 -> 32 -> 16).
pub const FREQ_BINS: usize = MEL_BINS / 8;
/// Conv channel width (downsample_hidden_size).
pub const CONV_CH: usize = 480;
/// Conv kernel (square, k=3 s=2 p=1 on both axes).
pub const CONV_K: usize = 3;
/// conv_out input width = CONV_CH * freq bins per timestep (480 * 16).
pub const CONV_OUT_IN: usize = CONV_CH * FREQ_BINS;

pub const ENC_DMODEL: usize = 1_024;
pub const ENC_LAYERS: usize = 24;
pub const ENC_HEADS: usize = 16;
pub const ENC_HEAD_DIM: usize = ENC_DMODEL / ENC_HEADS;
pub const ENC_FF: usize = 4_096;
/// nn.LayerNorm default eps (config carries no override).
pub const LN_EPS: f32 = 1e-5;

/// Projector output width == text hidden size.
pub const OUTPUT_DIM: usize = 2_048;

pub const TXT_HIDDEN: usize = 2_048;
pub const TXT_LAYERS: usize = 28;
pub const TXT_HEADS: usize = 16;
pub const TXT_KV_HEADS: usize = 8;
pub const TXT_HEAD_DIM: usize = 128;
pub const TXT_INTER: usize = 6_144;
pub const VOCAB: usize = 151_936;
pub const RMS_EPS: f32 = 1e-6;
pub const ROPE_THETA: f32 = 1e6;
/// KV cache capacity: 30 s prompt (408) + max generation, rounded up.
pub const KV_CAPACITY: usize = 1_024;

/// `<|audio_pad|>` — spliced audio embedding placeholder.
pub const AUDIO_TOKEN: u32 = 151_676;
/// `<|audio_start|>`.
pub const AUDIO_START: u32 = 151_669;
/// `<|audio_end|>`.
pub const AUDIO_END: u32 = 151_670;
/// `<|im_end|>` and `<|endoftext|>`.
pub const EOS_TOKENS: [u32; 2] = [151_645, 151_643];
/// Default generation bound (generation_config max_new_tokens).
pub const MAX_NEW_TOKENS: usize = 512;

/// Chat-template prompt for a transcription request:
/// `<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<audio_start>` +
/// audio pads + `<|audio_end|><|im_end|>\n<|im_start|>assistant\n` +
/// `language {LANG}<asr_text>`.
pub fn prompt_prefix() -> Vec<u32> {
    // <|im_start|>system \n <|im_end|> \n <|im_start|>user \n <|audio_start|>
    vec![151_644, 8948, 198, 151_645, 198, 151_644, 872, 198, AUDIO_START]
}

/// Suffix tokens for a supported language ("language Chinese<asr_text>").
pub fn prompt_suffix(language: &str) -> Option<Vec<u32>> {
    match language {
        // "Chinese" BPE ids: "language"=11528, "Chinese"=8453, <|asr_text|>=151704
        "Chinese" => Some(vec![
            AUDIO_END,
            151_645,
            198,
            151_644,
            77_091,
            198,
            11_528,
            8_453,
            151_704,
        ]),
        _ => None,
    }
}

/// Mel frames HF computes for `samples` waveform values (reflect-padded
/// STFT, last frame dropped, capped at the 30 s window).
pub fn valid_frames(samples: usize) -> usize {
    (samples / 160 + 1).min(FRAMES)
}

/// Post-conv length of one chunk with `frames` valid mel frames, applying
/// three (k=3, s=2, p=1) convolutions: `(x - 1) / 2 + 1`, zero stays zero.
pub fn post_cnn_len(frames: usize) -> usize {
    let mut len = frames;
    for _ in 0..3 {
        len = if len > 0 { (len - 1) / 2 + 1 } else { 0 };
    }
    len
}

/// Attention window lengths over the packed post-CNN sequence, mirroring
/// `get_audio_cu_seqlens`: windows of `max_chunk_len * n_window_ratio`
/// positions, remainder last.
pub fn attention_windows(n_positions: usize, valid_frames: usize) -> Vec<usize> {
    let max_chunk = (1..=N_CHUNKS)
        .map(|i| {
            let start = (i - 1) * CHUNK_FRAMES;
            let frames = valid_frames.saturating_sub(start).min(CHUNK_FRAMES);
            post_cnn_len(frames)
        })
        .max()
        .unwrap_or(0);
    let ratio = N_WINDOW_INFER / CHUNK_FRAMES;
    let window = (max_chunk * ratio).max(1);
    let mut windows = Vec::new();
    let mut remaining = n_positions;
    while remaining > window {
        windows.push(window);
        remaining -= window;
    }
    if remaining > 0 {
        windows.push(remaining);
    }
    windows
}
