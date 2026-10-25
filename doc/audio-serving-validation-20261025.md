# File audio serving validation

Date: 2026-10-25.
Status: Draft implementation with open hardware qualification.

The [contract](audio-serving-contract-v0.1.md) defines the A0 and A1 file profile.
The [operator guide](audio-serving.md) defines runtime preparation and service commands.
The [observation record](audio-serving-validation-20261025.json) contains selected results from actual service requests.

## Implemented behavior

The standalone audio service keeps one model resident for one selected task.
File ASR accepts bounded PCM16 mono WAV input at 16 kHz.
TTS returns one complete PCM16 mono WAV file after WAV checks.
The worker separates a terminal result from resource settlement.
The gateway retains its media reservation through the last response buffer reference.

All new implementation code, tests, and conformance fixtures use independent handwritten work.
The adapter calls pinned dependency APIs.
This change contains no copied external implementation or test corpus.
The implementation has no dependency on the unmerged text serving changes.

## CPU checks

| Check | Result |
| --- | --- |
| Rust crate tests | 5 passed |
| Python contracts, adapters, worker, and actual TCP tests | 30 passed |
| Rust Clippy with warnings denied | Passed |
| Rust formatting and whitespace checks | Passed |
| English structural lint | Zero findings |

Both languages consume the same original conformance corpus.
The corpus contains 35 frames, 10 WAV cases, seven JSON cases, five text cases, five encoding cases, and seven state sequences.
The HTTP tests start the actual Rust binary and persistent Python worker.
The explicit CPU provider tests transport behavior without a speech-quality claim.
CI includes Linux and macOS CPU jobs.
This local run does not establish the result of either remote CI job.

The checks cover these observable behaviors:

- Exact schemas, UTF-8 encoding, integer limits, Unicode text rules, and WAV boundaries.
- Model and task errors, unsupported fields, normalized content lengths, and queue limits.
- Original deadlines, queue expiry, disconnect cancellation, and retained settlement reservations.
- Worker epochs, malformed frames, duplicate terminals, missing settlement, and channel loss.
- Loading and execution signals, bounded diagnostics, complete WAV responses, and final buffer ownership.
- Startup identity checks, frontend failures, special-file rejection, and retained failure diagnostics after reaping.

The structural lint does not establish full ASD-STE100 dictionary review or certification.

## Fixed findings

| Finding | Correction and evidence |
| --- | --- |
| Python accepted encodings that Rust rejected | Both validators require UTF-8. Shared encoding cases pass. |
| Whitespace and numeric parsing differed | Both validators use explicit rules. Shared boundary cases pass. |
| Idle HTTP flush consumed the response deadline | The send deadline starts after final response readiness. A 16-second CPU request succeeds. |
| Missing Whisper processor could permit readiness | Startup checks the public tokenizer and each declared language. Original failure tests pass. |
| Kokoro deferred its full English frontend check | Startup constructs the public frontend and phonemizer fallback. An original failure test passes. |
| Public and worker error names differed | The contract lists exact HTTP mappings and three worker failure codes. Both corpus consumers pass. |
| An open stdin could cause interpreter abort during startup failure | The control reader uses bounded, unbuffered descriptor reads. The original failure test passes. |
| A FIFO output could block settlement and shutdown | Nonblocking opens precede regular-file checks. The actual CLI rejects the FIFO, reaps the worker, and releases quota. |
| Worker reaping replaced the recorded failure cause | Reaping preserves the cause. The actual HTTP fault test checks final readiness diagnostics. |

Independent reviewers checked the implementation and corrective commits.
The final corrective review found no new material issue.
The comment review removed 14 narrative lines.
It required no constraint encoding, comment restoration, or architectural change.

## Actual model observations

The host uses Apple M4 hardware with 16 GiB of memory.
The runtime uses Python 3.12.10, MLX 0.32.4, and MLX Audio 0.6.1.
The MLX Audio commit is `f976eab8b47a3cada5aeb3b52a78451ee06117bc`.
The selected English frontend uses Misaki 0.9.4, spaCy 3.8.7, preshed 3.0.10, and `en_core_web_sm` 3.8.0.
The local dependency check reports no broken requirements.

The service held the configured shared Metal lock during each deployment.
The host pressure sensor reported normal pressure before each deployment.
The TTS worker exited before the ASR deployment started.
Each service completed one request, reported zero active requests, and returned its media reservation to zero.
SIGTERM then produced service exit code zero and worker process exit.

The original sentence was: "The local audio service returns one complete recording."

| Task | Observed result |
| --- | --- |
| Kokoro TTS with the frozen `af_heart` voice | Complete 24 kHz WAV, 96,000 frames, four seconds, zero saturated samples |
| Whisper ASR with the fixed local test profile | The transcript matches the reference after Unicode whitespace splitting |

The ASR input uses an earlier successful TTS recording from the same original sentence.
An explicit offline step uses SciPy 1.18.1 to resample that recording from 24 kHz to 16 kHz.
The service performs no hidden resampling.
This single synthetic utterance does not estimate general recognition quality.
WAV validity does not establish intelligibility, pronunciation, or voice consistency.

### Frozen test artifacts

The Kokoro source is `mlx-community/Kokoro-82M-bf16` at revision `a71e4d38b236d968966a2002c4c895dbd12b1c3c`.
Its stored weights contain 548 F32 tensors, despite the repository name.
The weight file SHA-256 is `4e9ecdf03b8b6cf906070390237feda473dc13327cb8d56a43deaa374c02acd8`.
The voice SHA-256 is `2c1c733b0e6576c810e268d3e440c21dea4e0f0131a3ba4cfc98d7fe6136d094`.

The Whisper weight source is `mlx-community/whisper-tiny-fp16` at revision `c8c5417d20aca5049e6a6b07bede74a316618542`.
The original weight SHA-256 is `64a626bc42762297c27ca3d5ddb5b730b48cba6b031072fe33a9a4eae3dec211`.
Processor resources use `openai/whisper-tiny` revision `169d4a4341b33bc18d8881c4b69c2e104e1cc0af`.

The strict loader rejected the original auxiliary `alignment_heads` weight.
The explicit test profile removes only that weight before deployment identity checks.
All 166 remaining F16 tensors retain their exact dtype, shape, and values.
The derived weight SHA-256 is `20192882a6684f84d9e318c9acb2d83aabbbfbdba08d7553e7aa1eafa9f2c301`.
The service retains strict loading and does not perform this conversion.

The manifest freezes model files, voice files, frontend resources, and selected runtime versions.
Manifest digests identify these local deployments.
Absolute local paths contribute to each digest.
The observation record includes those deployment digests.
Model weights and local environments do not form part of this PR.

### Preserved failed observations

The initial English frontend failed inside spaCy tokenizer loading before any model waveform conversion.
The failure also occurred through the public English model loader without MLX inference.
With spaCy 3.8.7, preshed 3.0.14 still produced the overflow.
Changing preshed to 3.0.10 made that fixed frontend profile load successfully.
The runtime and manifest identities record this explicit dependency choice.

The initial Whisper bundle also failed strict loading and remained unready.
These failures prompted frontend preflight and process-exit regression checks.

## Open qualification

Complete these checks before a model profile receives deployment approval:

1. Measure recognition quality with fixed human recordings, accents, noise, silence, and declared languages.
2. Evaluate TTS intelligibility, pronunciation, voice consistency, punctuation, and maximum accepted text.
3. Measure peak device allocation during loading, generation, cancellation, and settlement.
4. Test hardware cancellation and deadlines during model loading and active generation.
5. Exercise maximum-duration input, maximum output, slow receivers, and repeated requests on fixed hardware.
6. Check controlled pressure failures and unsuccessful process-reaping paths without unsafe host pressure experiments.
7. Define measured acceptance thresholds before performance or memory-safety claims.

Provider generation can allocate device tensors before it yields a bounded output segment.
The current media quota does not establish an internal device-allocation ceiling.
Streaming PCM, realtime ASR, incremental TTS input, and shared text/audio scheduling remain future stages.
