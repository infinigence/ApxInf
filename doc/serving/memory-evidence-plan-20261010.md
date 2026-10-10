# Memory evidence plan, 2026-10-10

Status: planned measurements, with no measurement results or acceptance claim.
This plan uses [calibration v1](memory-calibration-v0.1.md) and [coverage v1](memory-coverage-v0.1.md).
It adds no worker fields and approves no admission budget.

The plan files and input messages are original, handwritten fixtures.
They select cases without requiring exhaustive shape enumeration.
A case in this plan does not establish measured coverage until its artifact passes the evidence checks.

## Fixed profile

Use the same approved model directory for all stages.
Record the exact model revision before comparing runs.
Do not substitute another model, provider, or precision profile.

| Field | Required value |
| --- | --- |
| Provider | `mlx-lm` |
| Precision | `bundle` |
| Context limit | 16384 tokens |
| Output limit | 2048 tokens |
| Prefill step | 256 tokens |
| Output batch | 1 token |
| MLX memory guideline | 10737418240 bytes |
| Host pressure policy | `macos` |
| Thinking | Disabled |
| Tools | None |
| Calibration synchronization | `runtime_owned_streams` |

`P` means the exact input token count after synthetic tiling and truncation.
`G` means the maximum output token count for a case.
Each case requires `P + G <= 16384`.
The model's real EOS behavior remains active.
Do not suppress EOS to satisfy an output target.

## Input files

The `--plan-file` argument accepts the exact ordered case array from calibration v1.
Each case has six fields: `case_id`, `input_tokens`, `max_tokens`, `stop_after_prefill`, `stop_after_output`, and `repetition`.
Both unused stop fields contain null.
The `--seed-messages` argument accepts an original messages array.

The parent freezes the selected plan and messages in the new artifact directory.
The probe reads those frozen inputs instead of the original source paths.
Record their content digests with each result.
Do not combine `--plan-file` with automatic matrix selection arguments.

Use [the primary input](../../benchmarks/serving/plans/seed-primary.json) for stages 01 through 07.
Reserve [the prose input](../../benchmarks/serving/plans/seed-held-out-prose.json) and [the code input](../../benchmarks/serving/plans/seed-held-out-code.json) for stage 08.
Different messages produce different prepared token sequences.
Their tiled forms remain synthetic inputs, not conversational quality tests.

## Stage order

Run one stage at a time under the existing Metal device lock.
Each invocation requires a new artifact directory and a new probe process.
Do not run competing GPU workloads during a measurement.
The stage number defines the execution order.

| Stage and plan | Cases | Purpose |
| --- | --- | --- |
| [01 smoke](../../benchmarks/serving/plans/01-smoke.json) | 5 | Check zero output and short generation before larger cases. |
| [02 default boundaries](../../benchmarks/serving/plans/02-default-boundaries.json) | 12 | Separate KV capacity changes from additional prefill calls. |
| [03 output depth](../../benchmarks/serving/plans/03-output-depth.json) | 4 | Check allowances 256 and 2048 with input lengths 32 and 512. |
| [04 fresh long request](../../benchmarks/serving/plans/04-fresh-long.json) | 3 | Start with `P=8192, G=32`, then repeat that case twice. |
| [05 context edge](../../benchmarks/serving/plans/05-context-edge.json) | 5 | Check `P+G=16384` with allowances 0, 1, 32, 256, and 2048. |
| [06 late stops](../../benchmarks/serving/plans/06-late-stops.json) | 5 | Check late prefill stops, the final prompt callback, and late output stops. |
| [07 fresh context edge](../../benchmarks/serving/plans/07-fresh-context-edge.json) | 2 | Start with `P=14336, G=2048`, then repeat it once. |
| [08 independent inputs](../../benchmarks/serving/plans/08-held-out.json) | 4 per input | Compare the reserved inputs at shapes from earlier stages. |

Stages 01 through 07 contain 36 planned cases.
Stage 08 contains four cases for each of two separate input runs.
The complete schedule contains 44 case executions across nine probe processes.
These counts describe fixtures, not completed measurements.

Stage 02 examines boundaries around the default prefill step and nearby cache growth positions.
The journal must retain actual cache offsets separately from planned input lengths.
A requested boundary does not establish a physical allocation boundary.

The [static geometry analysis](qwen35-memory-geometry-20261010.md) selects three additional cases.
`P=254, G=1` precedes the predicted KV boundary.
`P=255, G=2` isolates decode across that boundary.
`P=258, G=1` requires a second prefill loop call with the pinned generator and chunk size 256.
These predictions require measurements before acceptance.

Stage 04 makes the long case the first request in its process.
Stage 07 makes the context edge case the first request in another process.
Later repetitions remain in the same process as their first case.
A fresh process does not establish a cold GPU hardware cache.

Stage 05 includes `G=0` to check the service's no-generation path at the context limit.
That case does not measure prefill for 16384 tokens.
Only generation cases establish generation observations.

Stage 06 uses prefill stop positions 3840 of 4096, 14080 of 14336, and 14336 of 14336.
Its output stop positions are 1024 and 2047 under an allowance of 2048.
The final prompt callback can follow computation beyond the prompt.
Do not describe that callback as a pure prefill measurement.
Stop cases check cleanup and partial execution, not full-allowance shape coverage.

## Execution and progression

The following command illustrates stage 01 from the repository root.
Replace both absolute paths before execution.
Use a new output directory for every invocation.

```sh
python3 benchmarks/serving/memory_calibration.py \
  --model /absolute/path/to/model \
  --output-dir /absolute/path/to/new-run \
  --plan-file benchmarks/serving/plans/01-smoke.json \
  --seed-messages benchmarks/serving/plans/seed-primary.json \
  --max-context 16384 \
  --max-output-tokens 2048 \
  --prefill-step-size 256 \
  --output-batch-tokens 1 \
  --memory-budget-bytes 10737418240 \
  --timeout 1800
```

Before the next stage, check the current artifacts:

1. Check the frozen plan and input digests.
2. Check model identity and the fixed profile.
3. Check every planned case and its recorded outcome.
4. Check generator closure, request settlement, and process reaping separately.
5. Run the CPU coverage analyzer against the artifact directory.
6. Record reached targets and unresolved targets separately.

Stop progression after a pressure block, execution failure, invalid artifact, or unconfirmed cleanup.
Resolve the cause before another stage starts.
Keep failed artifacts and partial journals.
Do not weaken the pressure policy or change system memory settings to continue.

Early EOS can complete a request without reaching its requested allowance.
Record that allowance as unresolved.
An early EOS alone does not invalidate earlier measurements or forbid the next stage.
Do not change the requested allowance in the retained artifact.

## Independent input gate

Freeze an estimator candidate before using stage 08 as held-out validation.
The candidate must identify its inputs, supported domain, margins, and source artifacts.
Do not fit that candidate with measurements from the reserved inputs.
Run the prose and code inputs in separate probe processes.

If stage 08 precedes candidate definition, label its results exploratory input coverage.
Those results cannot later become independent validation for a candidate that uses them.
Use new reserved inputs after validation results change the candidate.
Two inputs do not establish coverage of all model inputs.

## Evidence and acceptance gates

Preserve the calibration report, process report, complete journal, coverage report, frozen inputs, and their digests.
Record host identity, operating-system build, runtime versions, environment policy, and observation policy with the run report.
Existing source fields alone do not contain every required identity field.

A reached target requires the coverage rules, actual observations, successful cleanup, and evidence of process reaping.
Unknown cache properties remain null.
Unobserved positions remain outside the measured evidence.
Do not extrapolate a smaller output count to an unresolved allowance.

Keep allocator active memory, allocator cache memory, allocator peaks, logical cache payload, process RSS, and system swap separate.
Compare peaks only within their recorded reset intervals.
Inherited peaks do not become new case peaks.
Normal host pressure does not prove memory headroom.
System swap changes do not identify the service's individual contribution.

Calibration synchronizes runtime streams before observations.
That synchronization can change allocation overlap and measured peaks.
Production validation without these sampling synchronizations is a separate acceptance gate.
It must preserve normal production synchronization for generation and cleanup.
This plan contains no result for that gate.

An approved admission envelope requires a separate versioned contract and successful independent validation.
Its domain must exclude unresolved inputs, allowances, execution modes, and resource states.
Its margins require measured justification.
Neither these fixtures nor complete execution automatically approves an envelope.
Batch execution and persistent caches require separate evidence.

This document applies structural ASD-STE100 principles.
Structural checks do not establish full dictionary compliance or certification.
