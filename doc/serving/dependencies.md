# Serving dependency record

Date: 2026-10-09. Scope: serial profile 0.1.

This record describes dependency calls, not imported implementation code.
All new ApxInf source, tests, and fixtures are handwritten.
The [serial profile](serial-profile-v0.1.md) defines the wire interface.
`Cargo.lock` records the resolved Rust dependency versions.

## Rust gateway

| Dependency | Version constraint | Role | Replacement boundary |
| --- | --- | --- | --- |
| `axum` | `=0.8.9` | HTTP routing, bounded bodies, responses, and SSE transport | `gateway.rs` |
| `tokio` | `=1.53.1` | Process pipes, timers, bounded channels, sockets, and task execution | `supervisor.rs` and the binary |
| `futures-util` | `=0.3.33` | Public response stream trait | `gateway.rs` |
| `serde` | `1` | Visitor interface for duplicate-key rejection | `contracts.rs` |
| `serde_json` | `1` | JSON parsing and values | Validators and public API adapters |
| `sha2` | `=0.10.9` | SHA-256 for identity and token digests | `contracts.rs` and artifact inspection |
| `uuid` | `=1.24.0` | New command, request, epoch, and lease IDs | Supervisor and response adapters |
| `clap` | `=4.5.23` | Command-line arguments | Serving binary |

The gateway implements admission, event order, and resource settlement itself.
These dependencies do not supply the serving scheduler or wire semantics.

The diagnostic module uses Rust standard library threads, bounded channels, atomics, and fallible stderr writes.
Tokio reads worker stderr through a separate pipe with bounded chunks.
These calls add no package dependency and do not change the worker protocol.

## Host pressure sensor

The macOS policy calls the system `sysctlbyname` C interface without an additional package dependency.
The call reads `kern.memorystatus_vm_pressure_level` and does not change system settings.
`host_pressure.rs` owns the platform boundary and rejects unsupported sensor results.
Other platforms require an explicit disabled policy in this profile.
The service does not use this signal as an allocation estimate.

## Python worker

The production worker requires Python 3.14.3, MLX 0.32.1, and MLX-LM 0.31.3.
It rejects a different runtime before model loading.
It also checks the package versions from the existing ApxInf runtime pin helper.
The model manifest records those package versions.

| Public dependency API | Role |
| --- | --- |
| `mlx_lm.load` | Load the approved local model bundle and tokenizer |
| `mlx_lm.generate.generate_step` | Execute serial prompt processing and greedy generation |
| `generate_step.prompt_progress_callback` | Observe prefill progress and check cancellation between prefill chunks |
| `generate_step.prefill_step_size` | Set the configured prefill chunk size |
| `generate_step.prompt_cache` | Use a request-owned cache with no session retention |
| `generate_step.sampler` | Select the token with the highest score |
| `mlx_lm.models.cache.make_prompt_cache` | Construct the model-specific request cache |
| `tokenizer.apply_chat_template` | Convert normalized messages and tools to authoritative prompt tokens |
| `tokenizer.detokenizer` | Produce incremental text and the terminal text tail |
| `mlx.core.array`, `mlx.core.argmax` | Create token arrays and select greedy token IDs |
| `mlx.core.reset_peak_memory` | Reset the request peak counter |
| `mlx.core.default_stream`, `mlx.core.default_device` | Capture load, request, and generation streams through their active contexts |
| `mlx.core.synchronize`, `mlx.core.clear_cache` | Finish device work and clear reusable allocator storage during cleanup |
| `mlx.core.get_active_memory`, `get_cache_memory`, `get_peak_memory` | Measure worker memory |
| `mlx.core.set_memory_limit` | Set the configured allocator limit |

The adapter uses the existing first-party helper `_pinned_toolchain_versions` in `scripts/apxinf_mlx_generate.py`.
This helper is an ApxInf internal dependency, not a public upstream API.
The manifest includes its file digest.

The manifest records each model artifact path, byte count, and SHA-256 digest.
It also records runtime versions, adapter revision, adapter digest, Python digest, and validator digest.
The execution record includes provider, precision, prefill chunk size, output batch size, and memory limit.
The supervisor checks local artifact digests before it accepts the worker identity.

The Python validators use only the standard library.
The original CPU tests inject a runtime stub without loading MLX.
Passing these tests does not prove device execution or model quality.

## Offline memory calibration

`benchmarks/serving/memory_calibration.py` calls the existing ApxInf `MLXRuntime` adapter.
It records allocator counters and cache metadata through that adapter's observation interface.
Cache `nbytes` and optional `offset` properties describe logical arrays and supported sequence positions.
The observer does not evaluate cache arrays or add their payload to allocator memory.

The handwritten `host_memory.py` helper uses Python `ctypes`, `struct`, and `resource` standard library APIs.
It reads macOS pressure, swap usage, hardware identity, and process peak RSS.
The driver uses `fcntl.flock` and inherited descriptors for the existing Metal measurement lock.
It reuses the first-party `metal_matrix.stop_owned_group` helper to reclaim its isolated child process.
These calls add no package dependency and preserve worker protocol 2.0.

`memory_coverage.py` reads those artifacts with Python standard library APIs.
It calls existing first-party canonical identity and token digest functions from `serving/contracts.py`.
It does not import MLX or introduce a runtime dependency.

Explicit calibration inputs use the existing first-party `prepare_input` validator for seed messages.
The parent records bounded JSON inputs and their SHA-256 digests before it starts the probe.
These operations add no package dependency or worker fields.
