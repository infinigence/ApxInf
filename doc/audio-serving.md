# Run the file audio service

The standalone service implements completed-file ASR and completed-WAV TTS.
Each deployment keeps one configured model resident for one selected task.
The [audio contract](audio-serving-contract-v0.1.md) defines limits and ownership.
CPU tests exercise the implementation without a speech-quality claim.
Hardware quality and device-allocation qualification remain open.

## Prepare an isolated runtime

Use a separate Python environment for audio.
Do not install audio dependencies into an existing text environment.
The approved adapter calls MLX Audio public APIs at the revision below.

| Dependency | Pin | Purpose |
| --- | --- | --- |
| MLX Audio | `0.6.1`, commit `f976eab8b47a3cada5aeb3b52a78451ee06117bc` | Whisper loading and transcription, Kokoro loading and synthesis |
| MLX | `0.32.4` | Device arrays, synchronization, and cache cleanup |
| NumPy | Exact installed version in the manifest | PCM conversion through its public array API |
| English frontend | Exact installed versions and file digests in the manifest | Kokoro pronunciation and text preparation |

The runtime manifest fixes the exact Python version and all selected frontend packages.
The operator must provision dependencies and model resources before service startup.
The service does not download resources.
MLX Audio must retain its installation `direct_url.json` commit record.
An installation without that record fails identity checks.

An example installation command for an operator-managed Python 3.12 environment follows.
This command installs dependencies and does not qualify a model.

```sh
python3.12 -m venv /absolute/audio-env
/absolute/audio-env/bin/python -m pip install 'mlx==0.32.4' 'mlx-audio @ git+https://github.com/Blaizzy/mlx-audio.git@f976eab8b47a3cada5aeb3b52a78451ee06117bc'
```

Provision the selected English frontend separately for Kokoro.
Record its package versions, package data, voice file, and preprocessing assets in the manifest.
The worker checks every regular model file and every selected frontend package file.
The worker denies network access and subprocess fallback.
Missing phonemizer or frontend resources must fail instead of invoking a download.
Startup constructs the public Misaki English frontend before model readiness.

## Freeze the manifest

The manifest is a UTF-8 JSON object with no duplicate or unknown fields.
Its exact bytes define both `model_revision` and `manifest_revision` through SHA-256.
The supervisor passes that expected digest before worker provider construction.
The capability digest uses compact JSON with sorted keys and exact Unicode values.
The capability fields are `task`, `languages`, `voices`, `limits`, and `cancellation`.

| Field | Required value |
| --- | --- |
| `schema` | `apxinf-audio-manifest/1` |
| `task` | `asr` or `tts` |
| `model` | Public model alias |
| `provider` | `mlx_audio` |
| `family` | `whisper` for ASR, `kokoro` for TTS |
| `precision` | `frozen_artifact` |
| `model_dir` | Canonical absolute local model directory |
| `runtime` | Exact runtime identity object |
| `artifacts` | Frozen artifact entries, at most 256 |
| `languages` | Explicit language list, `en` only for the Kokoro profile |
| `voices` | Object from public voice alias to canonical absolute `.safetensors` file |

`runtime` has exactly `python`, `mlx`, `mlx_audio`, `mlx_audio_commit`, and `packages`.
The `packages` object maps distribution names to exact installed versions.
It cannot override `mlx` or `mlx-audio`, including normalized aliases.
Every production profile requires the NumPy package identity.
The Kokoro profile also requires `misaki`, `spacy`, and `en_core_web_sm` package identities.
ASR declares an empty `voices` object.

Each artifact entry has exactly `path`, `sha256`, and `role`.
The path names a canonical absolute regular file.
The role is `model`, `config`, `preprocess`, `voice`, or `frontend`.
The list must include model, configuration, and preprocessing identities.
Kokoro frontend artifacts also define its preprocessing identity.
Every model file must occur in the list.
Each voice file must have the `voice` role.
Kokoro frontend package files must have the `frontend` role.
File hashes use bounded reads, including large model weights.

The ASR adapter calls `mlx_audio.stt.utils.load_model(Path, lazy=False, strict=True)`.
It calls `model.generate(audio=array, language=language, stream=False, temperature=0, return_timestamps=False)`.
It reads the returned `STTOutput.text` attribute.
Startup calls `model.get_tokenizer(language=language, task="transcribe")` for each declared language.
Missing processor resources or unsupported languages prevent readiness.
The TTS adapter calls `mlx_audio.tts.utils.load_model(Path, lazy=False, strict=True)`.
It calls `model.generate(text=text, voice=local_safetensors_path, speed=1.0, lang_code="a")`.
It checks each yielded waveform shape and cumulative output before host conversion or WAV writing.
These adapter calls do not establish an allocation bound inside one provider generation step.

## Start one deployment

Use one shared lock path for all cooperating Metal applications.
The service requires an explicit production lock path.
The lock remains held through worker reaping and service shutdown.
The lock cannot arbitrate applications that do not use it.
Host pressure must remain normal at startup and admission.
The service does not increase wired limits or bypass pressure checks.

```sh
cargo build -p apxinf-audio-serving --bin apxinf-audio-serve
target/debug/apxinf-audio-serve --manifest /absolute/asr-manifest.json --python /absolute/audio-env/bin/python --metal-lock /absolute/metal-measure.lock --listen 127.0.0.1:8010
```

Use a separate deployment and port for TTS.
The service has no shared text-and-audio device allocator.
Do not infer safe simultaneous model residency from separate process limits.

```sh
curl http://127.0.0.1:8010/readyz
curl -F model=local-asr -F file=@recording.wav http://127.0.0.1:8010/v1/audio/transcriptions
curl -H 'Content-Type: application/json' -d '{"model":"local-tts","input":"A completed speech request.","voice":"heart","response_format":"wav"}' http://127.0.0.1:8011/v1/audio/speech --output speech.wav
```

SIGINT and SIGTERM close admission and reap the worker.
An unexpected worker exit or protocol fault disables readiness and fails queued requests.
The service does not replay requests or restart a failed model automatically.
Restart the deployment after you fix the reported configuration or provider fault.

## Check the implementation

The original shared corpus covers strict frames, WAV bytes, encoding, text bounds, and lifecycle transitions.
The HTTP tests start the actual binary and persistent worker on local TCP.
The CPU fixture requires both explicit test flags.
The fixture reports `cpu_fixture` and `disabled_cpu_fixture` in readiness.
The production provider cannot use those flags.

```sh
CARGO_TARGET_DIR=/tmp/apxinf-audio-serving-target cargo test -p apxinf-audio-serving
CARGO_TARGET_DIR=/tmp/apxinf-audio-serving-target cargo build -p apxinf-audio-serving --bin apxinf-audio-serve
APXINF_AUDIO_BIN=/tmp/apxinf-audio-serving-target/debug/apxinf-audio-serve python3 -m unittest discover -s tests/audio_serving -p 'test_*.py' -v
```

The public API maps common serving errors consistently.

| Condition | HTTP status and public code |
| --- | --- |
| Wrong model alias | 404 `model_not_found` |
| Host pressure or media capacity | 503 `capacity_unavailable` |
| Worker not ready | 503 `model_unavailable` |
| Worker channel EOF | 503 `worker_lost` |
| Malformed worker protocol | 500 `protocol_fault` |
| Queue or original deadline | 504 `deadline_exceeded` |
| Waiting queue full | 429 `queue_full` |

The HTTP implementation uses Hyper and Tokio through public APIs.
`http-body-util` bounds incoming body collection.
`multer` parses bounded multipart bodies.
`bytes` retains the media lease until the final response buffer reference disappears.
`futures-util` adapts the completed multipart body to the parser's stream API.
The Cargo lockfile fixes their resolved versions.

## Qualification limits

The implementation does not expose streaming PCM, SSE, realtime transcription, or incremental text input.
The implementation does not claim full upstream audio API compatibility.
The output ceiling applies at each yielded provider segment before host copies.
The provider can allocate device tensors before it yields that segment.
That internal allocation requires model-specific measurement and qualification.

The hardware gate requires fixed runtime, model, voice, and frontend identities.
The gate must measure ASR quality, TTS quality, peak allocation, pressure, cancellation, and cleanup.
CPU fixture success cannot close that gate.
No GPU inference or model-quality acceptance formed part of the CPU implementation checks.
Retain hardware observations, runtime snapshots, and complete outputs in the ignored development workspace.
Historical observations do not qualify a later source, runtime, model, or frontend identity.

The reusable qualification command records a real production observation without granting quality approval.
Run this command only against the selected offline deployment.
Keep output within an ignored local directory.

```sh
python3 benchmarks/serving/audio_qualify.py --service-url http://127.0.0.1:8010 --task asr --model local-asr --expected-revision MANIFEST_SHA256 --input-wav /absolute/recording.wav --reference-text /absolute/reference.txt --output-dir devlocal/audio-serving/qualification-asr
python3 benchmarks/serving/audio_qualify.py --service-url http://127.0.0.1:8011 --task tts --model local-tts --expected-revision MANIFEST_SHA256 --text 'An original qualification sentence.' --voice heart --output-dir devlocal/audio-serving/qualification-tts
```

The command checks service identity and completed output.
Optional WER uses case-sensitive Unicode whitespace word splitting without other normalization.
TTS waveform validity does not measure intelligibility, pronunciation, or voice consistency.
The command leaves quality and device-allocation approval open in its report.
