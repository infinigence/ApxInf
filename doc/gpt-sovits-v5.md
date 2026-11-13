# GPT-SoVITS V5

ApxInf provides a CUDA-native GPT-SoVITS V5 speech runtime for NVIDIA
devices, loaded from a checkpoint directory in the `apxinf-gpt-sovits-v5`
format. The public input is target text, reference text, and a 3–10 second
reference audio file. The output is mono `float32` PCM at 48 kHz.

## Checkpoint format

A converted directory is identified by the `format` field of `manifest.json`,
which is `apxinf-gpt-sovits-v5`. The complete manifest for the V5 release:

```json
{
  "format": "apxinf-gpt-sovits-v5",
  "source_sha256": {
    "gpt": "87133414860ea14ff6620c483a3db5ed07b44be42e2c3fcdad65523a729a745a",
    "sovits": "287b32a06b7d486f5303ec520468b0728fb01aac8b0f5d86fd5455c9a1d3e022",
    "vocoder": "b7543c83222aa2105bb4efb56845086bc5f2eeba61243d723ffa062052b3221f",
    "hubert": "24164f129c66499d1346e2aa55f183250c223161ec2770c0da3d3b08cf432d3c",
    "roberta": "e53a693acc59ace251d143d068096ae0d7b79e4b1b503fa84c9dcf576448c1d8"
  },
  "dtype": "float16",
  "version": "v5turbo"
}
```

- `format` names the checkpoint layout. `apxinf-gpt-sovits-v5` is the only
  layout this runtime loads; it covers the five-component V5 model.
- `version` records the upstream release tag, `v5turbo`; the model itself is
  **GPT-SoVITS V5**. The upstream GPT file is still called `s1v3.ckpt` for
  historical reasons; that `v3` is a file name and not a model generation.
- `dtype` is `float16`, the only precision the runtime accepts.
- `source_sha256` records the five official inputs the directory was converted
  from, so a checkpoint is traceable to exact upstream files across machines.

The directory also holds `config.json` (model geometry, inference defaults and
the `assets` map) and the five safetensors files named there: `gpt`, `sovits`,
`vocoder`, `hubert` and `roberta`.

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

The Rust model contract is `apxinf_model::TtsTrait`, defined in `tts_trait.rs`,
alongside the separate `LlmTrait` language-model contract. One contract covers
every TTS family (GPT-SoVITS V5 and Qwen3-TTS share it); `LoadedModel::Tts`
owns the implementation and `LoadedModel::tts_mut()` returns the runtime.
`TtsRequest` carries the shared core (text, seed), optional `TtsSampling` knobs,
and a typed per-family `TtsExtension`; GPT-SoVITS passes its phone/BERT/reference
inputs through `TtsExtension::GptSovits`. `TtsSpec`, `PreparedTts` and
`GeneratedAudio` describe fixed profiles, prepared plans and output. The Python
binding remains `SpeechRunner`.

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
Token embedding and FP16 exponential-race sampling run outside the decode graph.
When `top_p >= 1.0` and `top_k <= 128` the sampler runs on the device: the
repetition-penalty flags stay resident there, and each step reads back only the four-word
sampler output (sampled token, adjusted-logits argmax, status) instead of the logits.
Requests outside that range (`top_p < 1.0`, `top_k > 128`, or an unsupported
logits dtype) fall back to the reference host sampler, which downloads logits and
filters them on CPU. Both paths use the same counter RNG, which is independent of
PyTorch's generator offset mapping. Frontend/acoustic graph inputs are rebound before replay;
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

A profile covers one text fragment. `prepare()` applies the request's
`text_split_method` and rejects text that splits into more than one fragment;
use `cut0` or prepare per fragment.

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

- GPT-SoVITS V5 with Chinese-first frontend semantics.
- FP16 converted weights on CUDA.
- 48 kHz mono PCM.
- Four CFM steps by default.
- `cfg_rate` must be zero; V5 classifier-free guidance is unsupported.
- Reference audio must be 3–10 seconds through the Python policy.
- GPT caches the joint text/audio prefix and processes one audio token per
  decode step. Compatible requests reuse the graph and buffers, recomputing
  prefix values; another capacity replaces the retained resources.
- Convolution preparation retains cuDNN handles, descriptors, layout buffers
  and algorithm workspace. The vocoder's channel-last convolutions have no
  direct fallback, so GPT-SoVITS requires a build with cuDNN available.

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
  the same reference audio, text, seed and request parameters, and reported as
  real-time factor (synthesis time / audio duration), because output lengths
  differ slightly between implementations.

End-to-end quality is judged on a fixed text and reference-audio set by ASR
character error rate, speaker similarity and duration, compared with the
official PyTorch pipeline under the same request parameters.

Diagnostic only: final PCM bit equality or a waveform SNR threshold after
injecting official random numbers and CFM noise. Sampling, iterative CFM and
the vocoder amplify last-bit differences into phase differences, and official
eager and graph runs differ from each other in the same way, so waveform SNR is
not an acceptance gate.

Numerical evidence must identify the CUDA libraries actually loaded at runtime,
including cuBLAS/cuBLASLt and cuDNN; a compiler toolkit version alone is
insufficient.

The converter also emits `cfm.text_pe`, the reference CPU FP32 nonpersistent
sinusoidal positional buffer rounded to the inference dtype. Native loading
uploads this fixed [4096,512] table once and does not regenerate it with a
different host trigonometric implementation.
