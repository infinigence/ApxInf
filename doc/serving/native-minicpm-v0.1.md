# Native MiniCPM5 serial profile

This profile uses `apxinf-worker/2.1`.
The Python worker retains `apxinf-worker/2.0`.
The service must select the protocol before worker launch.
The service must reject another protocol for that worker.

## Runtime identity

The native `runtime` object contains exactly these string fields:
`kind`, `apxinf`, `mlx`, `executable_sha256`, and `metal_library_sha256`.
`kind` is `native_minicpm5`.
`apxinf` is `0.1.0`.
Current native builds use MLX `0.32.3`.
Validators retain compatibility with earlier `0.31.2` runtime records.
The loaded library must match the native build's SDK version.
This 2026-10-29 amendment does not transfer earlier device qualification to the upgraded runtime.
Both digest fields contain lowercase SHA-256 hexadecimal strings.
The manifest repeats this runtime object and includes the sorted bundle artifact records.
The service checks each artifact and runtime file digest before readiness.
The selected library filename must be `libmlx.dylib`.
The worker checks its loaded library path before it loads the model.
A different loaded library prevents readiness.

The manifest declares `adapter_revision=apxinf-native-minicpm5-v1`.
Its execution object declares `provider=metal_minicpm5`, `precision=bf16`, and `variant=bf16-compiled`.
It declares `parser=minicpm5_xml` and `enable_thinking=false`.
Its `memory_limit_bytes`, `max_context`, and `max_output_tokens` match the launch configuration.

## Capabilities and preparation

The capability fields retain their 2.0 meanings.
The native worker declares `token_or_prefill_chunk` cancellation at the family plan's evaluated chunk boundaries.
The shared `--prefill-step-size 256` launch value is retained for compatibility; it does not set native chunk lengths.
This candidate permits at most 16384 positions.
Long-context acceptance requires the separate quality and memory gates in the context profile.
Validators also accept earlier 2.1 workers with `end_prefill_or_token` and at most 4096 positions.
Only one greedy text sequence can execute.
The worker does not support state reuse, image input, or sampling.

The worker renders the checkpoint template with structured tool calls and tool messages.
Tool arguments remain JSON objects during rendering.
The worker returns exact token IDs, their count, and their digest through `prepared_input`.
The coordinator checks this result before capacity reservation and submission.
The worker must consume the submitted tokens without another template operation.
The native profile requires all checkpoint EOS tokens, including 1 and 130073.
The worker must not emit EOS text.

## Controlled execution and settlement

`GenerationBoundary` identifies a control boundary, completed prefill progress, or an emitted token.
`PrefillStart` checks controls without emitting a progress frame.
A callback returns false to stop generation at that boundary.
The existing generation wrapper always continues through this callback.
The worker checks controls before generation and at each boundary.
A reader thread accepts bounded commands while the execution owner runs the model.
Only the execution owner can access model objects.

The worker synchronizes device work before releasing request state.
It releases KV arrays, decode functions, and request-dependent compilation traces.
Fixed weights and position tables remain resident model resources.
The worker clears allocator cache before reporting `resources_released`.
A failed synchronization or release closes the worker without releasing leases.
The supervisor retains reservations until successful worker recovery.
