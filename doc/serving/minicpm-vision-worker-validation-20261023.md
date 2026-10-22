# Serial MiniCPM-V worker observations

Status: partial P2c evidence. Date: 2026-10-23.
The Python worker completes repeated text, image, and cancellation requests.
The explicit HTTP profile passes selected image, policy, context-rejection, and cancellation checks.
Broader media coverage and Claude Code acceptance remain open.
The [vision profile](minicpm-vision-v0.1.md) defines the tested interface and candidate limits.

## Selected execution

The model is `openbmb/MiniCPM-V-4.7-1B`, revision `379f120c2ff3588e6e06559ec6b9ec15cc05114e`, with 1554476272 BF16 parameters.
The worker uses the separate pinned MLX-VLM environment in the [dependency record](dependencies.md).
The host has an Apple M4 and 16 GiB of unified memory.
The configured ceilings are 16384 context positions and 8192 output tokens.
The allocator limit is 8589934592 bytes. These settings do not qualify every permitted request.
Thinking, sessions, batching, and state reuse remain disabled.

The tested model identity is `b9edde971c128b843f332f768abc2ac328cbe1a6b3a0a5ea2f5c091619e1c50a`.
The ready manifest records checkpoint artifacts, seven adapter sources, sixteen dependency modules, and the loaded MLX library.
One original observation driver owns the Metal measurement lock.
The worker inherits that lock descriptor. The driver closes its descriptor after worker reaping.

## Actual serial requests

One resident worker executes these requests in order.
Counts include every yielded ID, including EOS. Internal lookahead is not measured.

| Request | Input IDs | Output IDs | Observed result |
| --- | ---: | ---: | --- |
| Short text | 18 | 3 | Answers `ready.` and settles. |
| Failed drawing PNG | 108 | 164 | Describes green abstract objects and rejects the requested subject. |
| Red-square PNG | 108 | 136 | Describes a red square on white and rejects the requested subject. |
| Cancellation after progress | 21 | 1 | Reports cancellation and settles. |
| Short text after cancellation | 18 | 3 | Answers `ready.` and settles again. |

The driver sends cancellation after zero-progress notification, before it observes output.
The worker yields one ID before it applies that control.
This observation does not establish cancellation inside prefill or before output.
Each terminal reports a null consumed position and complete yielded-token usage.
The attempt tracker checks event order, usage, source bytes, and capacity release.
The worker exits with code zero after shutdown. The driver reaps that process.
The host pressure sensor reports normal before each request and after process exit.

The two image requests have identical prepared IDs and different source bytes.
Their answers differ with the pixels. Human inspection supports the broad contrast.
The failed drawing has overlapping green shapes and gray blocks, without a recognizable pelican or bicycle.
The control has a red square on white. Some detailed descriptions of the failed drawing remain imprecise.
This pair supports bounded image grounding. It does not establish drawing quality or general visual accuracy.

Source accounting reports 38005 and 4792 bytes for the respective PNG files.
Each image attempt measures 2409760 retained host-array bytes.
The maximum observed device allocator peak is 3486174386 bytes.
This observation does not establish a general memory bound.
It does not measure process RSS or post-settlement active allocations.

## Demonstrated defects and repairs

The initial source review reproduces model-held position arrays after generator disposal.
The repaired owner restores position baselines and removes eighteen derived convolution caches after synchronization.
Original tests check partial installation, repeated restoration, resident handles, and permanent fencing after uncertainty.

An independent reviewer reproduces a separate one-shot synchronization failure during generator cleanup.
The old path permits later successful synchronization and resource release.
The repair fences both dependency-close and synchronization failures before further settlement.
Original regressions reproduce the old release error at exhaustion and before the first yield.
Both regressions pass after the repair. Fresh independent review accepts the change.

The first loaded-model request answers `ready.` but fails settlement.
Its inventory encounters integer keys in six rotary compilation dictionaries.
The worker withholds resource release and exits. This failed run does not count as successful settlement.
Loaded-model diagnostics identify key `3` and callable `mlx.core._gc_func` in each dictionary.
The profile now defines these dictionaries, exact module classes, wrapper type, and source identity.
Restoration clears qualified entries and preserves each original dictionary identity.
Unknown entries and unrelated state changes still fence the runtime before restoration.
The successful serial run above uses this repair. Global dependency compilation caches remain intact.

## Automated and preparation checks

The shared Python environment runs 323 serving checks with seven conditional skips.
Its older MLX wrapper cannot execute one new positive compilation-cache check.
The pinned vision environment passes all 35 vision checks without skips.
Twelve Rust contract checks pass, including paired fixture exchange.
The structural English check reports zero findings in the profile and dependency record.
This result is separate from a full ASD-STE100 dictionary review or certification.

Actual public processing covers text, portrait, landscape, and four-image histories without model execution.
Replay reproduces full typed outputs, token IDs, image spans, source order, and host-storage counts.
Changed arrays and preparation identities reject before device transfer.
Prepared host arrays disappear after their context closes.
Weight-free long preparation accepts 530, 4114, 8210, and 15018 input IDs.
An oversized preparation rejects at the configured context bound.
These preparation checks do not qualify inference at those lengths.

## Direct reference and selected context checks

One separate process compares the same model identity against the successful worker capture.
It calls the public processor and `generate_step` with matching prompt bytes, images, options, and selection controls.
The product's private state owner and synchronization methods protect repeated model use in this diagnostic.
The text request matches all 3 yielded IDs. The two image requests match all 164 and 136 IDs.
All three distinct attempts reach EOS in both paths.
Two additional repetitions of the failed-drawing request match all 164 IDs again.
These repetitions reuse one recorded worker attempt. They are not five distinct worker attempts.
This comparison does not reproduce worker cancellation.

The same process tests three independently generated sets of random facts.
Each prompt places one fact near its start, middle, and end.
Calibration uses the full prepared ID count and requires at least 95 percent of each target.

| Target input IDs | Actual input IDs | Output allowance | Observed result |
| ---: | ---: | ---: | --- |
| 4096 | 4094 | 256 | Exact three-fact JSON and EOS. |
| 8192 | 8180 | 256 | Exact three-fact JSON and EOS. |
| 15000 | 14999 | 256 | Exact three-fact JSON and EOS. |

These results qualify the selected retrieval prompts. They do not qualify arbitrary tasks or larger output reservations at those input lengths.
The model remains within the configured 16384-position total window.

After the first image, every remaining request reports 3109662078 active allocator bytes and zero cache bytes after settlement.
The largest peak in this sequence is 3866523146 bytes, during the 14999-ID request.
After model disposal and garbage collection, the process reports 1054 active bytes and zero cache bytes.
The process exits zero with no recorded cleanup errors. The host pressure sensor remains normal at its sampled boundaries.
These values describe one sequence and the selected device allocator.
They do not establish a memory ceiling, process RSS, or a speed comparison.

## Selected HTTP and Claude checks

The explicit provider loads through the production service and publishes `apxinf-minicpm-vision-v0.1` readiness.
The ready model revision and full manifest equal the qualified runtime identity above.
The observation checks worker parentage, process group, epoch, model path, and service listener ownership.
An independent source review accepts the Rust integration. Its complete CPU suite passes 186 checks with one ignored check.
The root's native-feature suite passes 209 checks with one ignored check. The release binary builds successfully.

The selected red-square request counts 89 input IDs and generates with the same count.
Its response identifies a red square on white and ends after 132 yielded IDs.
Temperature 0.001 rejects with `invalid_request`. A two-field URL source also rejects with `invalid_request`.
A long text input plus an 8192-token output reservation rejects with `context_limit`.
Counting and context rejection leave the public generated-token counter unchanged after settlement.
The fake-worker regressions separately prove that these paths do not submit generation.

A separate streaming request disconnects after its first public content event.
The worker reports cancellation after three output IDs. The service returns to idle and remains ready.
The observer records a reader exception after closing its socket.
Server events and capacity observations establish cancellation. The observer does not receive a clean stream end.
These observations do not establish interruption inside prefill.

The first real Claude vision revision fails before reading an image or changing the SVG.
The model repeatedly changes `/Users` to `/users`, which fails the client's restricted path check.
It then reads PNG bytes through Bash and eventually selects an undeclared tool.
The service rejects that invalid tool selection. No successful image read or Edit occurs.
The unchanged original drawing retains its failed visual verdict.
The owned service exits zero, its process group becomes empty, and the worker disappears.

Some separate startup attempts encounter host pressure warning after the initial normal sample.
The service withholds readiness and exits. These attempts do not qualify HTTP behavior.
One other attempt stops because the observer incorrectly compares the complete `lsof` output with a PID-only list.
A local listener reproduction demonstrates the additional file-descriptor field and verifies the repaired PID filter.

## Evidence and open gates

Private evidence remains under `devlocal/serving-minicpm/vision-grounding/`.
`worker-actual-001/` preserves the failed settlement. `runtime-state-diagnostic-003.log` identifies the rotary dictionaries.
`worker-actual-002/` contains exact frames, prepared inputs, output text, and the reaped process result.
The capture omits original submit commands. Lease equality checks rely on the driver's runtime attempt tracker.
The independent reports are `vision-worker-source-review.md`, `vision-state-final-review.md`, `vision-cleanup-rereview.md`, and `vision-rotary-cache-review.md`.
`profile-api/` preserves preparation, identity, decoder, and tiny public-state observations.
`runtime-qualification-001/` preserves direct comparisons, exact yielded IDs, context prompts, outputs, and allocator observations.
`service-actual-004/` preserves the selected HTTP checks, cancellation, failed Claude run, and process cleanup.
`service-actual-001/`, `service-actual-002/`, `service-actual-003/`, and `service-actual-005/` preserve earlier observation or startup failures.

Broader media shapes, maximum output reservations, and memory coverage outside the selected sequence remain open.
The public service still requires an explicit vision provider and checked preparation replay.
Real Claude Code must read the rendered drawing and perform a useful revision.
The original pelican-riding-a-bicycle visual acceptance remains unsatisfied.
