# Serving validation on 2026-10-11

This report records checks of the current serial text service and its calibration tools.
The checked HTTP, lifecycle, and Claude Code tasks passed within the scope below.
Memory coverage remains partial.
These results do not complete the full P1 gate or approve a memory admission estimate.
The [local service profile](local-service-v0.1.md) defines the implemented capabilities.

## Evidence and identity

The evidence directory is `devlocal/serving-upstream/validation-20261011/` in the serving worktree.
Paths below are relative to that directory.
Git ignores the raw reports, logs, model outputs, and isolated client directories.
This maintained report records their results without adding those artifacts to the repository.

The live run is `live-acceptance-01`.
Its `report.json` records the binary hash, launch arguments, host samples, subprocess results, and cleanup.
Its `ready.json` records the checked worker manifest and capabilities.

| Item | Recorded value |
| --- | --- |
| Host | Mac16,10, arm64, 17179869184 bytes of physical memory |
| Operating system | macOS 27.0.1 |
| Model | Qwen3.5-2B, local bundle precision, MLX-LM text path |
| Bundle snapshot | `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| Runtime | Python 3.14.3, MLX 0.32.1, MLX-LM 0.31.3 |
| Claude Code | 2.1.289 |
| Service profile | `apxinf-serial-v0.1`, one active sequence |
| Deployment limits | 16384 context tokens, 2048 output tokens, two waiting requests |
| Generation settings | Greedy, prefill chunks of 256 tokens, one token per output event |
| Runtime memory limit | 10737418240 bytes |
| Pressure policy | `macos` |

The live binary SHA-256 is:

```text
1cb378163e520615e72bcfd821da158dc60c22ff5ed396ce342c0433f481f38f
```

The model revision is:

```text
d2629f6e6865a95be54a7b61ea3c0c6f29cbc9813c2e0ee02137c649a8faebda
```

The capability revision is:

```text
0c1d9b5ec6d82811ed1dc50dcd9a9749a19bc2560945471469ad7dd42fe550dd
```

The live worker epoch is `ee41dc0f-18a6-422d-bc03-1a779ca1e53d`.
The calibration stages below record the same model revision.
Their reports identify their separate processes and measurement tools.

The launcher held the existing Metal coordination lock during model execution.
The recorded live host samples reported normal pressure.
System swap use was 451018752 bytes in the first and final live samples.
These samples do not prove continuous normal pressure, available memory headroom, or isolation from desktop workloads.

## CPU and host checks

| Check | Result | Evidence |
| --- | --- | --- |
| Rust library suite | 120 passed, zero failed, one ignored | `rust-full.log` |
| Rust HTTP integration entry | One passed, zero failed | `rust-full.log` |
| CPU HTTP scenarios | 12 passed through the real service executable | `http-gateway-final-02.log` |
| Actual macOS pressure sensor | One passed, observed state `normal` | `host-sensor.log` |
| Final Python suite | 240 passed, six skipped, 246 total | `python-final.log` |
| Production collector CPU suite | 12 passed | `production-cpu-tests.log` |

The Rust suite completed successfully, including its executable and documentation test targets.
The HTTP integration entry runs the 12 CPU scenarios.
These counts describe the same test group.
The actual sensor check separately executed the test that the default library suite ignores.

The Python total includes the 12 collector tests.
Its six skipped tests require an explicitly configured live service.
The separate live HTTP run below executed those six tests.

The new supervisor regressions check faults before admission, faults after a full content buffer, and worker exit after output.
They require the original error, one public terminal result, withdrawn readiness, and observed process cleanup.
The full suite also checks cancellation grace before acceptance and after a worker terminal.

The CPU HTTP fixture checks both public APIs, usage modes, tools, stop sequences, rejection, deadlines, disconnects, and shutdown.
It also checks a preparation protocol fault through token counting and a streaming protocol fault through both APIs.
Those faults retain `protocol_fault`, return unavailable readiness, and produce no success terminal.
The fixture uses no model, MLX package, or GPU.

## Live service and client checks

| Check | Result | Evidence |
| --- | --- | --- |
| HTTP test module | 37 passed, including six tests against the live service | `live-acceptance-01/live-http.log` |
| Disconnect cancellation | Passed, followed by resource settlement and ready status | `live-acceptance-01/lifecycle.json` |
| Queue overflow | Passed with queue capacity two, followed by resource settlement and ready status | `live-acceptance-01/lifecycle.json` |
| Claude Code read, edit, check, and cancel tasks | Four passed | `live-acceptance-01/claude-code/report.json` |
| Idle worker termination | Passed, unavailable readiness and retained gateway health | `live-acceptance-01/worker-failure.json` |
| Final process cleanup | No remaining members in the owned process group | `live-acceptance-01/report.json` |

The HTTP total includes 31 offline tests.
It does not represent 37 model requests or 37 separate live scenarios.
The live tests check both streaming APIs, token counting, service observations, and error responses.

Claude Code used isolated configuration and restricted tool permissions.
Its context limit was 16384 tokens, and its request output limit was 512 tokens.
The cancellation task observed content before cancellation and subsequently recorded resource settlement.
These small tasks establish observed client compatibility, not general coding quality or long conversation reliability.

The worker termination check changed readiness to HTTP 503 while `/healthz` retained HTTP 200.
The service exited with code one after the deliberate worker fault.
The parent observed no remaining process group members.
This result establishes cleanup after the injected fault, not a normal shutdown with exit code zero.

## Calibration coverage

The [memory evidence plan](memory-evidence-plan-20261010.md) defines these stages.
The [coverage profile](memory-coverage-v0.1.md) defines a reached target.
An executed case can finish at EOS without reaching its planned output allowance.
Each stage contains `calibration.json`, `process.json`, `samples.jsonl`, frozen inputs, and `coverage.json`.

| Stage | Planned | Executed | Generation observed | Reached | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| 02, cache and prefill boundaries | 12 | 12 | 12 | 7 | Five targets ended at early EOS. |
| 03, output depth | 4 | 4 | 4 | 0 | All four targets ended at early EOS. |
| 04, repeated long input | 3 | 3 | 3 | 3 | Each case consumed 8192 input tokens and produced 32 output tokens. |
| 05, context boundary | 5 | 0 | 0 | 0 | Loaded pressure was `warning`, so the probe skipped every case. |

Stage 03 requested 256 or 2048 output tokens with inputs of 32 or 512 tokens.
Actual output was one token for each 32-token input and eight tokens for each 512-token input.
Those observations do not establish either planned output depth.

Stage 04 used one process for its first case and two repetitions.
Each case recorded an owned allocator peak of 4194703532 bytes.
That value belongs to the recorded synchronized measurement interval.
It is not an RSS value, a production memory upper bound, or an approved reservation.

Stages 02, 03, and 04 completed with successful request cleanup and process exit code zero.
Their parent records show reaping and no remaining process group members.
Stage 05 reported successful run cleanup, process exit code one, observed reaping, and no remaining members.
Successful cleanup does not turn its skipped requests into coverage.

All four coverage reports passed their internal consistency checks.
Each report retains `admission_approved=false` and `full_memory_domain_established=false`.
Their input checks do not establish source authenticity or complete host identity.

Calibration uses synthetic tokens from a repeated prepared seed.
It adds synchronization before memory observations.
Those observations can change allocation overlap and timing.
They do not establish equivalent coverage through normal production HTTP requests.

## Production collector startup block

The `production-smoke-01` attempt used the same live binary and model profile.
The service reported `Host memory pressure prevents model loading: warning.` during startup.
It did not reach readiness, execute requests, or run the production collector.
This attempt supplies no real collector measurements.

The `production-smoke-01/launcher-report.json` status is `failed` because startup exited with code one.
The cause is a startup pressure block, not a completed collector check.
The parent cleanup record contains no remaining process group members.
The service diagnostic remains in `production-smoke-01/service.log`.

Host samples before and after startup reported normal pressure.
Those separate samples do not invalidate the service's warning observation during startup.
No retry followed this block in the reported run.
The 12 CPU collector tests establish fixture behavior only.

## Remaining limits

This run establishes the 8192-token calibration shape only with 32 output tokens and the recorded seed.
It provides no production HTTP coverage of long contexts or complete coverage near the 16384-token context limit.
The long output allowances remain unresolved.

Stages 06 and 07 did not run after the stage 05 pressure block.
The planned later cancellation positions and repeated context boundary remain unobserved.
Stage 08 has no frozen estimator candidate and supplies no independent estimator validation.
No result here approves a memory admission estimate, margin, or numerical envelope.

Batching, persistent caches, sessions, images, sampling, and multiple resident models remain outside this validation scope.
The [production memory profile](production-memory-validation-v0.1.md) defines a separate evidence procedure without calibration sampling synchronization.
Real production collector measurements remain absent after the startup block.

This document applies ASD-STE100 structural principles.
Structural checks do not establish a complete dictionary review or certification.
