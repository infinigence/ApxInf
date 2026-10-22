# MiniCPM default client context validation

Date: 2026-10-23.
Source revision for native runs: `4970e4b`.
Native binary implementation revision: `b92f330`.
The [bounded profile record](native-bounded-thinking-validation-20261023.md) retains the earlier 8192-token runs.
This record tests a 4096-token output allowance without the private client blocking-limit override.
The original task still fails visual acceptance.

## Qualified scope

The provider is `native-minicpm5-thinking-bounded`, with worker protocol 2.4.
The deployment retains 16384 total positions and 512 initial reasoning tokens.
The checkpoint remains MiniCPM5-2B revision `f97400052a43d642bbc6e9975e2397e3ae6a6b52`.
The native runtime remains MLX 0.31.2 with BF16 compiled execution.
The computer has an Apple M4 and 16 GiB of memory.
The actual Claude Code version is 2.1.289.

The client uses the existing isolated bare and restricted profile.
It exposes Read and Edit and disables automatic compaction.
It sets the context window from successful service readiness.
It omits `CLAUDE_CODE_BLOCKING_LIMIT_OVERRIDE` and the acceptance compatibility option.
This configuration does not qualify unrestricted tools, interactive sessions, or arbitrary long conversations.

## Original client task

The original drawing brief and task prompt remain unchanged.
The task asks Claude to read the brief and empty file, then save a complete SVG through Edit.
The native service and client both select 4096 output tokens explicitly.
This allowance includes hidden initial reasoning and the closing token.

The Claude process completed four Read calls and one Edit, then exited zero.
The first two Reads omitted the final task directory and failed the client's directory restriction.
The model corrected both paths and read the actual brief and empty SVG.
The Edit then saved the file successfully.
This result demonstrates recovery from these path errors, not recovery from a failed literal replacement.

| Request | Input tokens | Output tokens |
| --- | ---: | ---: |
| Initial Read calls | 806 | 190 |
| Corrected Read calls | 1166 | 680 |
| Edit | 1530 | 1173 |
| Final response | 2291 | 201 |

The client elapsed time was 170.5 seconds in this single run.
The run establishes no latency improvement or general success rate.
No request reached the context limit or the client's blocking limit.
The task runner returned two because visual review remained separate.
The service settled to idle and exited zero without forced termination.

## Artifact verdict

The saved SVG contains 1409 bytes.
Its SHA-256 is `b3c0a325a4a2bf71e5563f6e7a89aa9c0fb565841191fcd696c6416b672e9739`.
The existing structural checks passed.
They check XML, the allowed element subset, required title, viewBox, and external-content exclusions.
They do not check all SVG attribute semantics or drawing geometry.

Direct inspection of the rendered PNG failed visual acceptance.
The image has a dark rectangular block, two vertical lines, detached yellow shapes, and a lower pale oval.
It has no bicycle wheels or connected bicycle frame.
It has no recognizable pelican head, neck, and long beak connected to its body.
Client exit zero and a successful Edit therefore do not satisfy the original task.

## Diagnostic question

The completed tool cycle still produces an unrecognizable drawing.
Earlier runs also failed visual inspection with other output allowances.
The diagnostic tested whether the failure also occurs without Claude or tool argument encoding.
It sent the same original brief directly through Messages and asked for the complete SVG.
It retained the bounded native profile and 4096-token output allowance.

The service returned HTTP 200 with 158 input tokens and 4096 output tokens.
The stop reason was `max_tokens`.
The public response contained continued planning prose and a 301-byte SVG framework without graphical elements.
The extracted framework also contained `defs`, outside the task's allowed element subset.
Structural acceptance therefore failed before a visual verdict could qualify the output.
The complete response remains unchanged in the evidence directory.

The model, profile, and complete worker identity remained unchanged before and after the request.
The service settled to idle and exited zero without forced termination.
This failure rejects tool mediation as a sufficient explanation for the observed task failures.
This diagnostic does not isolate the reasoning policy, native arithmetic, or checkpoint quality.
It cannot establish original task acceptance because it executes no real client tools.

The initial-prefix budget controls a token boundary.
It does not guarantee a useful final answer or classify later public text.
This observation prevents treating the 512-token mechanism as a complete task-quality solution.

## Unavailable Write workflow

A separate revision supplied text from direct inspection of the failed original client image.
The requested workflow retained the original brief and asked for complete replacement through Write.
The launch requested Read, Edit, Write, and Bash.
The actual client advertised only Bash, Edit, and Read.
Write was unavailable, so this configuration could not test the intended workflow.

The client read the brief, source SVG, and review.
It attempted Bash replacements that the client denied.
Those attempts do not prove a model preference for Bash over an available Write tool.
The owner interrupted this invalid configuration after inspecting the actual initial tool list.
The original capture remains incomplete, and no completed client report exists for this run.

The active native request then settled as cancelled after client disconnect.
The service exited zero without forced termination, and all owned processes ended.
This interrupted run has no observed idle-metrics settlement check.
The SVG retained the original failed digest `b3c0a325a4a2bf71e5563f6e7a89aa9c0fb565841191fcd696c6416b672e9739`.
The workflow result is invalid configuration, not a completed revision or an accepted drawing.

This setup error also exposed a capture defect.
The runner trusted requested tools without checking the client's actual inventory.
The existing [task definition](pelican-acceptance-v0.1.md) already records Write's absence in the earlier client profile.
The new check prevents another long run with the same mismatch.

## Capture correction

The capture now checks named tools from its comma-separated `--tools` argument against the initial `system/init` event.
It records requested, advertised, and missing tool names in `tool_inventory`.
A missing tool or an invalid initial list stops only the owned client process group.
Generated output before the initial list also fails this check.
A later list cannot change that failed verdict.
New task acceptance requires a verified inventory when the command requests named tools.

The original regression expected missing Write, but the old capture returned an empty missing list and reached its deadline.
The corrected focused suite passes 18 tests.
The full Python serving suite runs 281 tests, with six existing conditional HTTP skips.
Its remaining 275 tests pass.
No native implementation changed during this correction, so the Rust suite was not repeated.

A real client-only probe requested Read, Edit, and Write with and without `--restricted`.
Both invocations advertised only Edit and Read, and the capture recorded missing Write.
Both stopped without a capture timeout.
The client returned process status zero after interruption, which does not establish task success.
The task verdict rejects the inventory mismatch independently from that status.

The Read and Edit control reported a verified inventory.
Its process exited one because the probe deliberately provided no inference backend.
These probes check client tool metadata and capture behavior only.
They do not qualify native inference, file changes, permissions, or arbitrary client configurations.

The next model-quality investigation needs a controlled reference comparison.
The direct failure does not justify assigning every remaining failure to model capability or to ApxInf arithmetic.

## Independent review

Independent review found no blocking code or evidence issues and repeated all 18 focused tests successfully.
The reviewer inspected the actual failed PNG and checked the reported digests and process records.
Comment review found no new comments, suppressions, or open constraints.
The reviewer configuration selected `gpt-6-astra`.
Its runtime did not expose the exact model identity.
The review used the current-turn summary and local evidence rather than the complete chat transcript.
It excluded the separately identified malformed historical decision rows.

## Evidence

Local records remain under `devlocal/serving-minicpm/`.
`native-bounded-default4096/report.json` records readiness, process status, and settlement.
`native-bounded-default4096/claude/report.json` records the original task and actual tool results.
`native-bounded-default4096/visual-review.json` binds the failed verdict to the SVG digest.
The task directory retains the actual SVG, rendered PNG, brief, and client streams.
`native-bounded-direct4096/` retains the separate direct-model diagnostic.
`native-bounded-default4096-write-revision/invalidation.json` records the interrupted invalid configuration.
`client-tool-inventory/report.json` records the real client-only inventory probes.
`client-tools-before.log` retains the original failing regression.
`client-tools-after.log` and `client-tools-python-full.log` retain the passing checks.

Structural writing checks do not establish a complete dictionary review or certification.
