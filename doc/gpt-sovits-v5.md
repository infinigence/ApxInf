# GPT-SoVITS V5

ApxInf provides a CUDA-native GPT-SoVITS V5 Turbo speech runtime for NVIDIA
devices. The public input is target text, reference text, and a 3–10 second
reference audio file. The output is mono `float32` PCM at 48 kHz.

## Checkpoint conversion

Convert the five official components with
`scripts/convert_gpt_sovits_weights.py`. The converter writes FP16 safetensors,
`config.json`, and `manifest.json`. The native runtime follows the official V5
half-precision path, so load it with `precision="auto"`; other precision
selections are rejected.

```bash
python scripts/convert_gpt_sovits_weights.py \
  --gpt /path/to/s1v3.ckpt \
  --sovits /path/to/s2Gv5turbo.pth \
  --vocoder /path/to/vocoder.pth \
  --hubert /path/to/chinese-hubert-base/pytorch_model.bin \
  --roberta /path/to/chinese-roberta-wwm-ext-large/pytorch_model.bin \
  --out /path/to/gpt-sovits-v5-apxinf
```

The Python text frontend needs the GPT-SoVITS `text` and
`module.mel_processing` packages, PyTorch, torchaudio, resampy, Transformers, and the
`chinese-roberta-wwm-ext-large` tokenizer. Put those resources under
`CHECKPOINT/frontend`, pass `frontend_root=`, or set `GPT_SOVITS_ROOT`. Runtime
model computation stays in ApxInf; Python owns text cleaning, tokenization,
resampling, mel extraction, and spectrogram extraction. The HuBERT 16 kHz
branch follows the official GPU `kaiser_best` resampler; the 32 kHz mel and
spectrogram branch uses torchaudio.

The Rust model contract is `apxinf_model::TtsTrait`, defined in
`tts_trait.rs`, alongside the separate `LlmTrait` language-model contract.
`LoadedModel::Speech` owns the implementation, and `LoadedModel::tts()`
returns the trait object. `SpeechRequest`, `SpeechSpec`, `PreparedSpeech`,
and `PcmAudio` describe synthesis input, fixed profiles, prepared plans, and
output. The Python binding remains `SpeechRunner`.

## Python API

```python
from apxinf import GptSovitsPolicy

policy = GptSovitsPolicy.from_pretrained(
    "/path/to/gpt-sovits-v5-apxinf",
    device="cuda:0",
    precision="auto",
    autotune=False,
    frontend_root="/path/to/GPT-SoVITS",
)

request = {
    "text": "要合成的文本。",
    "reference_text": "参考音频对应的文本。",
    "reference_audio": "/path/to/reference.wav",
    "seed": 1234,
    "sample_steps": 4,
    "speed_factor": 1.0,
    "text_split_method": "cut5",
    "fragment_interval": 0.3,
}
result = policy.infer(request)
pcm = result["samples"]
sample_rate = result["sample_rate"]
```

`autotune=False` is the default for both `GptSovitsPolicy.from_pretrained`
and `SpeechRunner.from_pretrained`. Valid hardware/toolkit recipes are reused;
recipe misses use the backend fallback without benchmarking. Set
`autotune=True` to tune missing GEMM/attention keys during eager execution or
preparation. Steady graph replay does not tune. Operator build changes can
invalidate recipes, so numerical and latency evidence must be recorded again
for the resulting selection.

`apxinf.SpeechRunner` is the lower-level canonical tensor API. It accepts phone
IDs, RoBERTa IDs, `word2ph`, 16 kHz HuBERT audio, normalized 100-bin mel, and a
704-bin spectrogram. Most applications should use `GptSovitsPolicy`.

## Speed, segmentation and fragment gaps

`speed_factor` follows the official `TextEncoder.forward` semantics: the
encoded text latent is interpolated to `int(frames / speed_factor) + 1` frames
with PyTorch `F.interpolate(mode="linear", align_corners=False)` before the
projection and the 2x bridge upsample. It is not waveform resampling, so pitch
and formants are preserved. Prepared CUDA Graph profiles are shape-specialized
at native speed; requests with `speed_factor != 1.0` automatically use the
eager path.

`text_split_method` accepts `cut0` through `cut5` and uses the official
GPT-SoVITS `text_segmentation_method.py`. After every fragment, including the
last, the policy appends `int(32000 * fragment_interval)` zero samples, matching
the official `audio_postprocess` sample-rate base, normalizes only when the
fragment peak exceeds 1, and concatenates the results. The engine itself adds
no implicit trailing pause; the only gap is the requested fragment interval.
`result["fragments"]` reports each fragment's text, sample count and duration.

## Conditioning caches

Caches are enabled by default and owned by one policy/model instance:

```python
policy.configure_cache(prompt=True, text=True, max_text_entries=32)
print(policy.cache_info())  # separate frontend and native statistics
policy.clear_cache()        # retains prepared graphs
policy.configure_cache(prompt=False, text=False)  # uncached comparison
```

`prompt=True` retains the latest reference text preprocessing, reference audio
preprocessing, reference BERT features, HuBERT codes, style, SoVITS features and
mel. A new target text reuses these reference values. Reference file identity
and content SHA256 are checked on each request: replacing a file at the same
path invalidates its frontend entry even when timestamps remain unchanged.
Native reference keys compare the complete audio/mel/spec and text inputs.

`text=True` retains target G2P/tokenization outputs and target BERT features in
an LRU, limited to 32 entries by default. Frontend text storage is capped at
8 MiB; native BERT tensors are capped at 64 MiB. Reference frontend audio is
limited to one entry and 64 MiB. Language, text version and frontend identity
participate in text keys. Changing switches clears the affected entries;
`clear_cache(reset_stats=False)` retains hit counters.

Stored frontend arrays are immutable. Prepared graph outputs are copied to
independent device buffers before being retained. Eager/prepared native entries
use separate keys. Generated semantic tokens, generated audio and target-dependent
GPT prefix values are never reused as prompt conditioning. `close()` releases
policy caches and its runner.

Measure cache effects with the same binary, inputs, seed, graph profile and PCM
hash. Separate first misses, warm reference/new text and repeated text; load and
graph preparation are separate setup costs. A new text whose shapes or generated
length differ requires eager execution or another explicit prepared profile.

## Fixed-shape CUDA Graph preparation

`GptSovitsPolicy.prepare()` captures RoBERTa, HuBERT, reference style,
reference/target SoVITS and vocoder CUDA Graphs. Each CFM chunk captures its
conditioning/noise initialization and one reusable solver step. Replay resets
the state, binds the precomputed time modulation on the device, and replays that
step for each Euler iteration. The state, conditioning, modulation and output
addresses remain fixed; the prepared object retains the state and conditioning
buffers as well as each step's attention and projection resources. GPT uses
per-request KV contents and retains one decode executable, KV allocation and
workspace for the latest exact capacity across requests. Each request recomputes
and overwrites its complete text/audio prefix before decode. Fixed-address
metadata supplies the current cache length and write position on each replay.
Token embedding, logits transfer and FP16 exponential-race sampling occur outside
the decode graph. Sampling currently downloads logits and filters them on CPU;
its counter RNG is independent of PyTorch's generator offset mapping. Frontend/acoustic graph inputs are rebound before replay;
fixed row maps, weights and workspaces remain owned by the prepared object.
The caller must supply `target_frames`, four times the expected semantic
token count:

```python
profile = policy.prepare(request, target_frames=100, sample_steps=4)
assert policy.is_prepared
result = policy.infer({**request, "max_seconds": 1.0})
policy.clear_prepared()
```

Every allocation- or dispatch-relevant input size is part of the prepared
profile. A later request with different text/reference shapes or `sample_steps`
is rejected. If GPT produces a different number of frames, inference reports
the expected and actual lengths rather than silently falling back. Values,
including IDs, samples, reference mel, seed, temperature, and penalties, may
change while the shape profile remains fixed.

The capture acceptance checks eager-versus-graph parity, repeat replay,
same-shape input propagation, seed propagation, and incompatible-shape
rejection.

The CFM schedule admits 32 fixed timestep indices. Their embeddings and block
scale/shift/gate vectors are computed on GPU during loading and retained with
the model. Repeated chunk inference reuses these immutable vectors, preserving
the original projection shapes and FP16 rounding.

HuBERT's per-channel normalization accumulates moments in FP32, then explicitly
rounds epsilon, mean and reciprocal standard deviation to FP16 before forming
FP32 scale/shift parameters. Other callers retain the default FP32 statistics
contract. Both paths support fixed-shape graph replay. Large FP16 bias/activation
operations use four-element loads and stores with the same FP32 arithmetic.

## Supported scope

- GPT-SoVITS V5 Turbo with Chinese-first frontend semantics.
- FP16 converted weights on CUDA.
- 48 kHz mono PCM.
- Four CFM steps by default.
- `cfg_rate` must be zero; V5 Turbo classifier-free guidance is unsupported.
- Reference audio must be 3–10 seconds through the Python policy.
- GPT caches the joint text/audio prefix and processes one audio token per
  decode step. Compatible requests reuse the graph and buffers, recomputing
  prefix values; another capacity replaces the retained resources.
- Convolution preparation retains cuDNN handles, descriptors, layout buffers
  and algorithm workspace when the development library is available. Without
  cuDNN, the native direct implementation remains available.

Component comparisons against official PyTorch use captured identical inputs
and noise. Component tolerances and discrete-token/PCM acceptance are separate:
captured GPT inputs isolate decoder checks from conditioning errors. Current
native HuBERT-to-code and public BERT/HuBERT/GPT conditioning-to-token checks
against the original fixtures remain unresolved; historical passing results at
another build/library tuple do not establish current acceptance.
The sampler's FP16 filtering and argmax boundaries have independent reference
tests, including top-p prefix rounding and ties, but its CUDA RNG mapping and
device sampling path remain incomplete.
## Acceptance tiers

Required for acceptance:

- Converted weights and model structure match the official checkpoint.
- Frontend outputs match the official frontend exactly on the fixture set.
- Each component meets its declared tolerance on captured official inputs
  (`abs_error <= 0.02 + 0.02 * abs(reference)`).
- With teacher-forced official tokens, GPT logits meet
  `abs_error <= 0.05 + 0.05 * abs(reference)` and greedy tokens match exactly.
- The public Python path, eager execution and CUDA Graph replay produce
  identical outputs for the same request.
- Performance is measured against official PyTorch eager and CUDA Graph with
  identical inputs, tokens, noise and output length.

Required end-to-end quality check: on a fixed text and reference-audio set,
ASR transcription error, speaker similarity and duration must stay within the
spread between official PyTorch eager and official PyTorch CUDA Graph runs.

Diagnostic only: final PCM bit equality or a waveform SNR threshold after
injecting official random numbers and CFM noise. Sampling, iterative CFM and
the vocoder amplify last-bit differences into phase differences, and official
eager and graph runs differ from each other in the same way, so waveform SNR is
not an acceptance gate. Passing graph replay, same-library acoustic checks or
isolated vocoder parity does not establish component acceptance.

Numerical evidence must identify the CUDA libraries actually loaded at runtime,
including cuBLAS/cuBLASLt and cuDNN; a compiler toolkit version alone is insufficient.
The current original PyTorch captures use cuBLASLt 13.1 while the native system
runtime uses 13.4. Same-input final-projection differences were reproduced by
running official PyTorch against the native library version. Such environment
controls are diagnostic evidence and do not replace the original output acceptance.
The CFM plain LayerNorm now uses a Welford reduction; isolated same-input checks
reproduce all 22 captured attention normalization outputs exactly. Final PCM and
public conditioning-to-token acceptance remain unresolved.

The converter also emits cfm.text_pe, the reference CPU FP32 nonpersistent
sinusoidal positional buffer rounded to the inference dtype. Native loading
uploads this fixed [4096,512] table once. Reconvert earlier development
checkpoints that lack this buffer; model loading does not regenerate it with a
different host trigonometric implementation.

## End-to-end quality acceptance (2026-11-06)

Run: `devlocal/gpt-sovits-v5/results/e2e-quality-20261106/` (testset.json,
gen_official.py, gen_apxinf.py, eval_metrics.py, report.py; logs in pipeline.log).
5 reference speakers x 3 texts (long/short/medium), seed 1234, top_k 5,
repetition_penalty 1.35, sample_steps 4, max_seconds 22. Judges on orin:
funasr seaco-paraformer-large (CER), microsoft/wavlm-base-plus-sv (speaker
cosine), wav duration. HF access via HF_ENDPOINT=https://hf-mirror.com.

Result: 12/15 pairs (F02, F07, S01701, Y03) are at parity with official eager —
mean CER 0.050 vs 0.051, mean speaker similarity 0.965 vs 0.964, duration
within 0.05 s. ApxInf eager and prepared(CUDA Graph) outputs are bit-identical
for the same request and seed, so the graph path adds no quality change.

F09 (fast, expressive movie-commentary reference) regresses systematically:
speaker similarity 0.69-0.81 vs official 0.93-0.98, stable across seeds
1234/42/2026 (official is likewise stable at 0.93-0.98, so this is not seed
luck), and CER on the short text reaches 0.29-0.52 vs official 0.0-0.15.
The suspected conditioning/HuBERT-to-code source was later ruled out: the
actual cause was the reference-window offset, fixed on 2026-11-10 below.

The formal "within official eager-vs-graph spread" gate degenerates on this
host: official graph and eager runs are near bit-identical (spread ~0), so the
strict pass counts (cer 11/15, speaker 1/15, duration 3/15) measure bit-exact
reproduction rather than quality. The 4-reference parity summary above is the
meaningful reading. Cross-seed controls live in wav-f09-seeds/ and
wav-f09-official-seeds/ with corrected sims in f09-seed-metrics.json (the
helper scripts initially fed 48k/32k audio to WavLM labelled 16k; funasr CER
was never affected).

E2E verdict (2026-11-06): PASS for 4/5 references, FAIL for F09. The F09
regression was later traced to reference-window slicing, not HuBERT codes; see
the 2026-11-10 follow-up below.

## Native speed/segmentation A/B (2026-11-10)

Run: `tmp/apxinf-e04-compare/` on orin; three E04 texts, Y03 reference,
`seed=1234`, `top_k=15`, `sample_steps=4`, `repetition_penalty=1.35`,
`speed_factor=1.05`, `text_split_method=cut5`, `fragment_interval=0.15`.
Both backends were judged with funasr seaco-paraformer-large and
microsoft/wavlm-base-plus-sv.

| id | official CER | apxinf CER | official speaker | apxinf speaker | official dur | apxinf dur |
|----|--------------|------------|------------------|----------------|--------------|------------|
| E04-s01 | 0.042 | 0.000 | 0.976 | 0.967 | 10.14 s | 9.72 s |
| E04-s08 | 0.042 | 0.021 | 0.969 | 0.977 | 8.78 s | 8.74 s |
| E04-s13 | 0.017 | 0.000 | 0.972 | 0.977 | 10.20 s | 9.68 s |

The earlier apxinf inbox batches simulated the official speed by polyphase
resampling and trimmed fragment edges before inserting fixed gaps. That
external processing shifted pitch/formants ("气虚") and removed natural
pauses ("断句不对"). With speed and segmentation handled inside the engine,
the three texts are at or above official CER and speaker similarity, and
duration is within ~0.5 s.


## F09 root cause and re-acceptance (2026-11-10)

F09 is the only test reference shorter than 5.0 s. With a 4.95 s reference the
prompt mel is shorter than 500 frames, so `feature_offset` in the reference
alignment became zero. The eager and prepared paths treated a zero offset as
"no slice needed" and passed the full decoded feature stream to CFM. CFM then
paired the *tail* of those features with the *head* of the mel, shifting the
prompt by ~60 frames (0.6 s). HuBERT feature and semantic-code parity were
verified separately: features match official at relative error 0.0012, and
prompt codes match 97.8-100% across all nine references, F09 at 99.28%. The
defect was therefore purely in the reference window selection.

The fix slices `[feature_offset, feature_offset + reference_frames)` even when
`feature_offset == 0`, and the prepared path no longer assumes a 500-frame
prompt. Re-running the full 5-reference x 3-text set with `cut0`,
`fragment_interval=0.3`, seed 1234, top_k 5, sample_steps 4:

| reference | CER official | CER apxinf | speaker official | speaker apxinf | max duration delta |
|-----------|--------------|------------|------------------|----------------|--------------------|
| F02 | 0.056 | 0.050 | 0.942 | 0.943 | 0.04 s |
| F07 | 0.050 | 0.050 | 0.965 | 0.960 | 1.12 s |
| F09 | 0.050 | 0.050 | 0.968 | 0.961 | 0.40 s |
| S01701 | 0.050 | 0.050 | 0.973 | 0.976 | 0.52 s |
| Y03 | 0.050 | 0.050 | 0.975 | 0.981 | 0.20 s |

All 15 texts now match official CER; speaker similarity is within 0.9 points
of official for every reference, and F09 improves from 0.70-0.79 to
0.948-0.984. Prepared CUDA Graph and eager output are bit-identical for the
same request. Raw wavs and metrics are under `tmp/e2e-new/` and
`tmp/f09-e2e/` on orin.
