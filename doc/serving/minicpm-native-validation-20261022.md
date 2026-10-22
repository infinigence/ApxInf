# Native MiniCPM serving validation

Status: long-context gates passed. Final Claude Code drawing acceptance remains pending.

## Configuration

The test computer has an Apple M4 and 16 GiB of physical memory.
The model is `openbmb/MiniCPM5-2B`, revision `f97400052a43d642bbc6e9975e2397e3ae6a6b52`.
Execution uses the native `metal_minicpm5` provider, BF16 weights, and `bf16-compiled`.
The runtime uses MLX 0.31.2 and worker protocol 2.1.
The client is Claude Code 2.1.289.

The service admits one active sequence and two queued requests in these checks.
The context allowance is 16384 positions, including input and reserved output.
Prefill uses evaluated chunks of at most 256 tokens.
The memory budget is 10737418240 bytes, with a 3221225472-byte sequence reservation.
The MLX allocator limit is a guideline, not a physical memory cap.
Host pressure policy remains enabled.

## HTTP and state checks

The initial 4096-position service passed seven HTTP checks.
They covered three text requests, Read output, tool-result history, client disconnection, and queue overflow.
Rendered input counts matched response usage.
Every settled request returned to 5035610116 active allocator bytes and zero cached bytes.

The larger profile retains position tables for 16384 positions.
Its settled active allocator count was 5041901572 bytes.
Every measured long request returned to this value with zero cached bytes.
These allocator observations do not establish total process RSS or total Metal memory.

A separate 16384-position service passed client-disconnect and idle-worker-failure checks.
The disconnect released request resources and restored the settled allocator baseline.
The fault probe verified process ownership before it terminated the idle native worker.
The service stopped with exit code 1 after the worker fault, as its failure policy requires.
The probe did not require a forced process-group kill.

## Long context

Each prompt contained random facts near its beginning, middle, and end.
The response had to return all three values exactly in a JSON object.
The probe checked rendered input counts and runtime identity before and after each request.

| Measured input tokens | Reserved output tokens | Generated output tokens | Result |
| --- | --- | --- | --- |
| 2040 | 128 | 46 | All three facts correct |
| 8190 | 128 | 52 | All three facts correct |
| 15998 | 128 | 47 | All three facts correct |

The maximum observed active allocator peak was 6596364464 bytes.
All 256 host samples in the combined context and first client run reported normal pressure.
Swap usage changed from 2259877888 to 2251489280 bytes during that run.
This evidence does not establish an upper bound for every prompt or a throughput claim.

A separate request used the same 15998-token prompt to test cancellation.
The probe sent cancellation after the first 256-token progress event.
The worker stopped after 512 consumed tokens, with zero output tokens.
It then reported resource release.
The same process and epoch answered a subsequent short request correctly.
Both cleanups restored 5041901572 active bytes and zero cached bytes.

## Client failures and corrections

The first client run completed Read and Edit exchanges but failed to save the drawing.
The model edited the brief and later emitted the Boolean literal `False`.
The gateway rejected that literal under its original strict JSON parameter rule.
A separate native worker replay reproduced the complete 328-token output and the same literal.
The reconstructed input contained 1335 tokens, compared with 1337 in the original request.
The available records cannot explain that two-token difference.

The MiniCPM parser now accepts exact `True` and `False` literals for declared Boolean properties.
It still rejects expressions, other spellings, and values with incorrect types.
The Qwen parser retains its existing behavior.

This client exposed Read and Edit when the requested tool list also contained Write.
The revised drawing task provides an empty SVG and identifies the Edit target explicitly.
The model must supply all drawing content.
The next run reached SVG generation but exhausted its 2048-token output allowance.
The operator stopped its repeated attempt and retained that failure record.
The 4096-token run also exhausted its allowance and left the drawing empty.
A native replay exposed identical SVG elements repeated from the first few hundred tokens.
A separate replay added explicit limits on element count and repetition.
It still repeated an identical path until the probe stopped it.
These observations reject output allowance as the only cause of this failure.

The checkpoint generation configuration enables sampling with temperature 1.0 and top-p 0.95.
The current native worker explicitly uses greedy selection.
Sampling comparison remains diagnostic work and does not change the advertised execution policy.

## Generation comparison

A separate native diagnostic used the same 1140 input tokens and 4096-position output reservation.
It sent native model logits to the existing ApxInf CPU sampler.
Greedy selection reproduced all 769 captured worker tokens.
CPU and Metal greedy choices matched at all 1024 measured steps.
The output repeated identical SVG elements and did not reach EOS within that probe.

Random selection used temperature 1.0, top-p 0.95, and seed 42.
It produced a complete Edit call and reached EOS after 535 output tokens.
The extracted SVG rendered successfully but failed visual acceptance.
The image did not show a recognizable pelican riding a bicycle.
This result supports sampling as a loop correction, not as sufficient task acceptance.

A comparison used external MLX-LM 0.31.3 with MLX 0.32.1 and the same prepared token input.
Its greedy output stopped after 773 tokens and also failed drawing quality inspection.
This separate reference did not replace the native provider or establish numerical equivalence.
The native serving profile remains unchanged while generation diagnostics continue.

A native diagnostic enabled the checkpoint's thinking template and retained the same random policy.
It reached EOS after 1031 tokens but failed the rendered drawing check.
A separate variant added composition instructions and exhausted 4096 output tokens.
Neither diagnostic establishes an adequate budget for general reasoning tasks.

## Model and integration boundaries

The selected MiniCPM5-2B checkpoint is a text model with a Llama causal architecture.
It does not contain the vision components of MiniCPM-V.
The native model rejects image input, and the service advertises no image capability.
The current client task reads SVG source. It does not receive rendered image pixels.
A visual revision loop requires a vision-capable model and image transport through the serving and tool interfaces.

The checkpoint supports its thinking template option.
The greedy and sampled native profiles set that option to false.
The separate native thinking profile enables the template and hides the initial reasoning prefix.
All three profiles reject client requests that enable Anthropic extended thinking.
Internal model reasoning and the Anthropic thinking API are separate integration contracts.
The HTTP checks below establish initial hidden reasoning support for the explicit thinking profile.

The official [model card](https://huggingface.co/openbmb/MiniCPM5-2B) recommends temperature 1.0 and top-p 0.95.
Its example enables the thinking template.
These reference notes explain the diagnostic choices. They are not imported implementation instructions.

## Current limits

The tested context limit is 16384 positions.
The longest factual input contained 15998 tokens with 128 reserved output tokens.
This test does not establish general reasoning quality across that context range.
Ordinary BF16 variants permit this capacity.
DSpark retains its separate 4096-position limit.

Requests do not reuse an earlier request's KV state.
Cleanup releases request arrays and request-dependent compiled functions before capacity release.
A bounded idle prefix remains a later optimization.
Context shifting, disk snapshots, continuous batching, and multiple resident models remain deferred.
A worker fault requires a service restart in this local profile.

## Explicit sampled profile follow-up

The separate 2.2 profile retains the checkpoint, BF16 precision, and 16384-position limit.
It adds explicit temperature and top-p selection through the existing CPU sampler.
The [sampling contract](native-sampling-v0.1.md) defines its identity and request behavior.
These runs use different executable identities from the earlier greedy measurements.

Eight HTTP policy checks passed across the Anthropic and OpenAI endpoints.
The first real sampled Claude run completed Read, Read, and Edit.
It wrote valid SVG, then failed after repeated automatic context compaction.
The rendered artifact did not show a recognizable pelican riding a bicycle.

The bounded test harness then disabled automatic client compaction.
It retained the real 16384-position deployment limit and service admission checks.
The next run completed four client turns and exited successfully.
Its SVG contained an unmatched closing group tag, so the artifact check failed.
This result establishes a complete tool interaction, but it does not pass the drawing task.
These bounded tests do not establish automatic compaction or indefinite session support.

Independent review found an overflow at very small positive sampling temperatures.
An original regression test reproduced the failure before the correction.
The sampler now centers and scales logits with binary64 intermediate values before computing binary32 weights.
This also handles extreme finite logits and large temperatures in the general core interface.
The final core suite passed 32 unit checks and one integration check.
The integrated serving suite passed 161 checks and skipped one.
The Python serving suite ran 269 checks, with six skipped.
Structural writing checks passed separately from implementation tests.
The final executable passed ten HTTP policy cases, including temperature `1e-40`, across both endpoints.
Client disconnect released capacity, and the verified owned worker fault correctly ended the service.
The fault test required no forced process-group termination.

## Explicit thinking profile follow-up

The separate 2.3 profile enables the checkpoint's thinking template without changing the model revision or BF16 precision.
The [thinking contract](native-thinking-v0.1.md) defines its output boundary, token accounting, and identity.
The integrated Rust serving suite passed 177 checks and skipped one.
The Python serving suite ran 270 checks, with six skipped.
Independent reviews found no remaining correctness or comment issues in this integration.

The real native service passed ten sampling policy cases across both HTTP endpoints.
A one-token output allowance returned empty content, `max_tokens`, and one output token in usage.
A larger allowance returned the requested marker without initial reasoning text or tags.
These checks exercise the actual model template, native worker, and HTTP output path.

A client disconnected before the first public content event, after the worker generated 16 internal tokens.
The request ended as cancelled, released capacity, and left the same worker ready.
The older cancellation probe waits for public content before disconnecting.
That trigger cannot test a response that spends its entire output allowance on hidden reasoning.
The separate scratch probe records its different trigger explicitly.

The real Claude Code task used an 8192-token output allowance within the 16384-position context limit.
The client completed Read, Read, Edit, and Read, then exited successfully after 172.37 seconds.
The artifact passed the structural SVG checks.
Its rendered image did not show a recognizable pelican riding a bicycle, so visual acceptance failed.
The recorded visual verdict identifies the SVG by its SHA-256 digest.
This single run does not establish a quality improvement from thinking or a larger output allowance.

The subsequent verified worker fault correctly ended the service without forced process-group termination.
The HTTP, cancellation, and tool checks therefore pass for this executable.
The pelican task remains incomplete because its visual gate failed.

## Executable feedback diagnostic

A separate scratch variant exposes Read, Edit, and a narrowly allowed Bash command to Claude Code.
An original Python checker reports XML errors and checks the SVG root, viewBox, and title.
The variant asks the model to run the checker after editing and correct reported errors.
It retains the original drawing brief and verifies that the checker remains unchanged.
This diagnostic does not replace the baseline pelican task.

The first checked run completed Read, Read, Edit, and Bash in 132.11 seconds.
The checker reported success, and the parent runner independently accepted the SVG structure.
The model did not change the checker or the brief.
The first artifact passed, so this run does not establish error correction.
Rendered inspection found a large green disk with overlapping shapes, without a recognizable pelican or bicycle.
Structural feedback alone did not satisfy the visual gate in this run.

A further scratch run supplied the failed baseline SVG and an external visual review as text.
The model read the review, checked the original file, and edited the SVG.
Claude Code then stopped locally with `Prompt is too long` and terminal reason `blocking_limit`.
The client did not send the next generation request to ApxInf.
The successful checker call preceded the edit, so the model did not complete the required final check.
The revised SVG also failed rendered visual inspection.

An original local-response probe reproduced the client limit without a large response body.
Both cases returned the same short Read call and reported 2695 input tokens.
Output usage of 50 permitted the next request, but output usage of 6986 caused a local block.
The latter count matches the real run's final generation, which includes hidden reasoning.
These results distinguish generated token usage from retained visible context.

## Bounded client compatibility mode

The runner now provides the explicit `--acceptance-claude-context-compat` option.
It requires Claude Code 2.1.289 and a positive context limit from successful deployment readiness.
The internal blocking threshold equals the effective client window, which cannot exceed the deployment window.
The default mode remains unchanged.
The [runner documentation](../../benchmarks/serving/README.md) records the internal client dependency and limits.

The original local-response probe also exercised the full acceptance runner, including discovery, readiness, version validation, task execution, and reporting.
The default run stopped after one request with `blocking_limit`.
The compatibility run completed two requests and returned the fixture's original marker through real Claude Code.
Both runs used a 16384-position window and an 8192-token output allowance.
The synthetic response supplied the same 6986-token output usage in both runs.
This probe checks client compatibility, not model generation quality.

A separate native run counted 9011 input tokens and rejected an additional 8192-token output allowance with HTTP 400 and `context_limit`.
The same executable passed hidden-output cancellation and the verified worker fault check.
No forced process-group termination occurred.
The final Python serving suite ran 272 checks, with six skipped.
Independent review found no material correctness issues in the client compatibility change.
The full pelican revision task still requires a successful visual result and final checker execution.

## Local evidence records

The working directory retains complete diagnostic records under `devlocal/serving-minicpm/`.
This ignored directory is not part of the source checkout distributed through Git.

| Record | Purpose |
| --- | --- |
| `native-http-4k/report.json` | Seven native HTTP checks |
| `native-acceptance-16k/report.json` | Long-context results, host samples, and initial client failure |
| `raw-tool-probe/raw-output.txt` | Native Boolean failure reproduction |
| `long-cancel/result.json` | Long prefill cancellation and subsequent request |
| `native-claude-retry/` | Truncated 2048-token drawing attempt |
| `native-claude-4k-output/` | Truncated 4096-token drawing attempt |
| `svg-raw-probe/` | Raw repeated elements and constrained-prompt comparison |
| `sampling-probe/` | Greedy equivalence, random generation, and separate reference comparison |
| `native-fault-16k/report.json` | Client disconnect and owned worker fault shutdown |
| `native-sampled-claude/` | Successful file edit, compaction failure, and failed visual review |
| `native-sampled-no-compact/` | Complete client tool interaction and invalid generated SVG |
| `native-sampled-final-fault/` | Final executable HTTP policy, disconnect, and worker fault checks |
| `sampling-temperature-repro.log` | Tiny-temperature regression before correction |
| `sampling-core-tests-final.log` | Final CPU sampling checks |
| `sampling-tests-final.log` | Integrated Rust serving checks |
| `thinking-tests.log` | Integrated Rust thinking checks |
| `thinking-python-tests.log` | Python serving checks after thinking integration |
| `native-thinking-claude/` | Hidden output checks and incompatible content-triggered cancellation probe |
| `native-thinking-claude-recheck/` | Native thinking HTTP, cancellation, and real client evidence |
| `native-thinking-checked-pelican/` | Checker tool execution and failed visual review |
| `checked_pelican.py` | Scratch runner for executable and external visual feedback |
| `native-thinking-visual-feedback/` | Model revision, local client block, and failed visual review |
| `client-context-probe-74f03bd5/` | Local client reproduction with controlled output usage |
| `client-context-probe-5b75b7e4/` | Full runner default and compatibility comparison |
| `native-thinking-context-rejection/` | Exact native context rejection, cancellation, and worker fault checks |
| `client-compat-tests.log` | Python checks for the bounded client option |
| `chunk-cli-smoke.log` | Native CLI regression after chunk integration |

The [context profile](minicpm-context-v0.1.md) records the separate numerical equivalence gate.
The [pelican task](pelican-acceptance-v0.1.md) defines structural and visual acceptance separately.
Structural writing checks do not establish a complete dictionary review or certification.
