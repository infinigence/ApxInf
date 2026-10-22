# Local context correctness probe

Status: Probe definition 0.1. Date: 2026-10-22.

This probe checks fact retrieval through the local Anthropic Messages API.
It does not measure latency, memory capacity, or Claude Code task success.
The [serving contract](contracts-v0.1.md) defines token counts and model identity.

## Input and limit

The command accepts `--base-url`, `--input-tokens`, `--max-tokens`, `--output-dir`, and `--metal-lock`.
`--input-tokens` accepts one or more positive target counts.
The URL must identify a loopback HTTP service.
The output directory contains one `report.json` file.
The optional Metal lock uses the existing measurement lock file.

The probe reads `/readyz` before each target.
It requires a declared supported profile, a ready worker, and positive context and output limits.
The supported profiles are `apxinf-serial-v0.1`, `apxinf-native-sampled-v0.1`, `apxinf-native-thinking-v0.1`, and `apxinf-native-bounded-thinking-v0.1`.
An unknown profile fails the probe.
It uses the smaller context limit reported by the ready worker.
It rejects a target when the target plus `--max-tokens` exceeds that limit.
It rejects `--max-tokens` above the reported output limit.

The probe creates three random nonce facts for each target.
It places them before, within, and after deterministic distractor notes.
The final user instruction asks for one exact JSON object with `start`, `middle`, and `end` values.
The nonce values appear only once in the request.

The probe calls `/v1/messages/count_tokens` with each candidate request body.
It searches a bounded note count and selects the largest measured count at or below the target.
The selected count must reach at least 95 percent of the target.
The generation request uses the selected body without changing its content.
The probe does not trim input or lower the output allowance to make a target fit.

## Record

`report.json` contains `format`, `created_at`, `base_url`, `max_tokens`, and `cases`.
Each case contains `target_input_tokens`, `status`, `reason`, `nonce_facts`, and `notes`.
Each case also contains `ready_before`, `ready_after`, `metrics_before`, and `metrics_after` when available.
The ready snapshots retain the full worker manifest, capabilities, runtime, and revisions.

For a calibrated case, the record contains `body_sha256`, `actual_input_tokens`, and `count_calls`.
It also contains `http_status`, `raw_response`, `response_input_tokens`, `stop_reason`, and `observed_facts` when available.
The digest covers compact UTF-8 JSON with sorted object keys.
The raw response retains a service error or the complete response body.

`status` is `passed`, `failed`, or `unsupported`.
The probe passes only when all three nonce values match exactly.
It also requires `stop_reason=end_turn` and equal counted and reported input tokens.
It requires the response model and the model, capability, and worker revisions to match readiness.
The profile must remain unchanged between the ready snapshots.
Markdown fences fail the required plain JSON format, even when all values are correct.
A missing field, truncated response, HTTP error, identity change, or wrong nonce fails the case.
Unsupported limits and targets receive `unsupported` with a reason.
The process exits with a nonzero status unless every requested target passes.

The report supports a bounded local context observation only.
It does not approve a larger capability profile or establish general model quality.
