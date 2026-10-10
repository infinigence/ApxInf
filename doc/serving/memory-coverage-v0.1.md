# Memory coverage profile 0.1

This profile defines a CPU-only analysis of existing memory calibration artifacts.
The source format is [memory calibration v1](memory-calibration-v0.1.md).
The analyzer does not load models, call MLX, change source files, or approve admission budgets.

The analyzer uses the existing canonical identity and token digest APIs.
It does not add dependencies or change the worker protocol.

## Terms

| Term | Meaning |
| --- | --- |
| Shape | An exact pair of input length `P` and output allowance `G`. |
| Planned | The source plan contains the case. |
| Executed | The case has an input digest or journal records. |
| Generation observed | A new peak epoch belongs to the case. |
| Reached | The case reaches its requested boundary and satisfies the evidence conditions below. |
| Inherited observation | A sample reads an epoch that a previous operation owns. |
| Shape domain | Integer pairs with `P >= 1`, `G >= 0`, and `P + G <= max_context`. |

The shape domain also requires `G <= max_output_tokens`.
A reached shape does not establish an upper bound for memory.
Different inputs with the same shape can produce different observations.

## Input and output

The command accepts one artifact directory.
It reads `calibration.json`, `process.json`, and `samples.jsonl` from that directory.
When `measurement_inputs` is not null, it also reads the fixed file `measurement-inputs.json`.
It does not follow filenames supplied inside those documents.
An optional `--output` path creates a new report file exclusively.
Without that option, the command writes JSON to standard output.

The output format is `apxinf-memory-coverage-v1`.
Its artifact kind is `coverage_evidence`.
`admission_approved` and `full_memory_domain_established` always remain false.

| Field | Meaning |
| --- | --- |
| `sources` | Actual source filenames, byte counts, and SHA-256 digests. |
| `validation` | `valid` or `invalid`, with errors and limitations. |
| `frozen_input_validation` | `verified`, `unavailable`, or `invalid`, as defined below. |
| `run_id` | The source run identity. |
| `identity` | The recorded model identity, manifest, and runtime identity. |
| `requested_profile` | The source execution limits and observation policy. |
| `hardware` | The recorded hardware fields. |
| `process` | Parent process evidence, including reaping. |
| `cases` | Planned boundaries, actual evidence, reached decisions, and reasons. |
| `peak_epochs` | Peak groups keyed by `run_id` and `peak_epoch`. |
| `shape_domain` | Exact planned and reached shapes, domain counts, and missing intervals. |
| `summary` | Case counts for planned, executed, generation observed, and reached. |

Exit status 0 means the source evidence passes validation.
It does not mean the source run completed or the shape domain is sufficient.
Exit status 1 means the input is invalid, unreadable, or unsupported.
Invalid input produces no derived cases, peak groups, or domain claims.
The report retains source digests and the validation error.

## Evidence validation

The analyzer rejects duplicate JSON keys, non-finite JSON numbers, and incomplete journal lines.
It checks document types, integer ranges, journal indices, monotonic timestamps, and journal digests.
Journal timestamps use the calibration format's unsigned 64-bit nanosecond range, independent of the worker protocol's safe-integer limit.

It checks plan identifiers, case order, contiguous sample ranges, and run sample references.
It rejects unknown case references and more than 4096 records for one case.
It rejects source recovery errors.
It rejects execution after a previous case stops the matrix.
Validation checks internal consistency, not source authenticity or current model files.

Executed cases require loaded identity, input evidence, and normal startup pressure observations.
Every requested profile requires `host_pressure_policy: macos`, including legacy artifacts without frozen inputs.
Every pressure observation requires `state`, `dispatch_value`, and `error`.
Known states require the matching dispatch value and a null error.
Unknown pressure requires a nonempty error and either a null dispatch value or an unknown unsigned 32-bit value.

The analyzer rejects observations that contradict `load_started` or continue generation after pressure blocks execution.
Case phases follow execution order, with terminal observations before settlement observations.
Completed generation requires initial progress and matching generation epochs through settlement.
Completed positive outputs also require complete prompt progress and a first-output observation.

The analyzer checks the model revision with the existing canonical identity API.
It checks repeated runtime and execution fields against the requested profile.
It checks seed and case input digests with the existing token digest API.
Output token digests remain reported evidence because the journal does not contain every output token ID.
An empty output digest must match the existing empty-token digest.

A reaped parent record must match the process record in `calibration.json`.
An unreaped parent record cannot establish reached cases.
An unreaped record can differ from the probe's earlier process snapshot.
An absent child can establish no runtime observations or reached cases.
A reaped process cannot retain recorded process-group members.
A completed, reaped source run requires successful process exit and complete, settled case results.
A failed or blocked run can retain earlier reached cases after successful reaping.

The calibration and process documents each have a 16 MiB limit.
The frozen input document has the calibration tool's 4 MiB limit.
The journal has a 64 MiB limit and a 4 MiB limit per line.
The analyzer accepts at most 256 cases and 65536 journal records.
These analyzer limits can reject a larger calibration artifact without changing its original evidence.

## Frozen input validation

The analyzer accepts legacy artifacts without `measurement_inputs`, including a null value.
Their `frozen_input_validation` is `unavailable`, even when overall validation is `valid`.
This status cannot establish input provenance for a held-out experiment.
The analyzer does not read an unreferenced frozen file.

A non-null `measurement_inputs` requires the calibration v1 frozen format.
The metadata contains exactly `file`, `sha256`, `plan_source`, `seed_source`, and `seed_messages`.
The `file` value must equal `measurement-inputs.json`.
The analyzer checks the actual byte digest before it parses the UTF-8 document.
It rejects missing files, oversized files, duplicate keys, and non-finite numbers.

The document contains exactly `format`, `profile`, `plan`, `seed_messages`, `plan_source`, and `seed_source`.
Its format must equal `apxinf-calibration-inputs-v1`.
Its profile and plan must equal the calibration report's requested profile and plan.
Its seed messages and source records must equal the corresponding metadata fields.
These comparisons preserve JSON types, including Boolean and integer differences.

The profile uses the calibration v1 fields and fixed policies, including `host_pressure_policy: macos`.
Each plan entry has the six calibration fields, including both explicit stop fields.
Case identifiers match `[A-Za-z0-9][A-Za-z0-9._-]{0,127}` and remain unique.
The analyzer checks integer bounds, context bounds, repetition bounds, and mutually exclusive stop conditions.
It checks seed messages with the public `prepare_input` validator, empty tools, and `enable_thinking: false`.

Each source record contains exactly `kind`, `path`, and `sha256`.
A `default` source has a null path and a digest of its canonical frozen content.
Canonical JSON uses sorted keys, compact separators, UTF-8 characters, and no final newline.
A `file` source has an absolute path and the recorded digest of the original source bytes.
The analyzer does not read that external path or equate its digest with a canonical content digest.

When `input_source` exists, its messages must equal the frozen seed messages.
Its template options must equal exactly `{"enable_thinking": false}`.
A blocked run can satisfy these checks with a frozen snapshot and no prepared input.
Successful checks set `frozen_input_validation` to `verified`.
A failed check sets it to `invalid` and makes the complete coverage report invalid.
Other validation failures can still make a report invalid after the frozen checks succeed.

Verified frozen inputs establish internal agreement between these artifacts.
They do not establish source authenticity, tokenizer correctness, held-out independence, or an admission budget.

## Case decisions

Each result retains its source status, reason, actual output count, EOS flag, stop flag, and cleanup fields.
Each result also records `planned`, `executed`, `generation_observed`, `reached`, and `unreached_reasons`.
Executed does not mean the model computed a token.
Zero-output cases can execute without generation.
`full_allowance_observed` requires case samples and an actual output count equal to the allowance.

A reached case requires all these conditions:

- The source case status is `completed`.
- The generator closure and settlement fields are true.
- The case has no execution or cleanup error.
- The parent records successful process reaping.
- Required terminal and settled samples exist.
- The case reaches its requested boundary.

A case without a stop boundary reaches its target only when actual output count equals its allowance.
Early EOS does not reach the full allowance.
EOS at the final allowed output does not reduce shape coverage.
A zero-output case requires zero actual outputs and no generation epoch.

A prefill-stop case requires the stop flag and a progress sample at or beyond its requested position.
An output-stop case requires the stop flag and enough actual outputs.
Stop cases do not establish full-allowance shape coverage.
Failed, blocked, skipped, and interrupted cases retain evidence but do not establish reached targets.

`cache_layers` records each observed layer index, type, maximum offset, and maximum logical payload size.
It also records whether any offset or payload size was unknown.
`max_observed_cache_offset` summarizes supported layer offsets only.
Unknown offsets remain null.
The analyzer does not replace them with runtime positions or output counts.
It does not require physical offsets to equal the planned endpoint.

## Peak ownership

Peak epoch zero belongs to loading.
A new positive epoch belongs to the generating case that first observes it.
Each generation case can own at most one epoch.
Epochs increase by one and never decrease.
Peak counters never decrease within an epoch.
After observation failures, settlement can reveal a failed case's new epoch for the first time.
That partial evidence cannot establish a reached case.

A `before_request` sample cannot start a new epoch.
A zero-output case cannot start a new epoch.
Progress and output samples must belong to their case's generation epoch.
Terminal observations from a case blocked before generation can inherit a previous epoch.

Each group retains all sample indices and its inherited sample indices.
`observed_peak_bytes` is the maximum counter across all group samples.
`owned_peak_bytes` excludes inherited samples.
A case reports only its own epoch's `owned_peak_bytes`.
Inherited counters never become a new case peak.
The analyzer does not add peaks, active bytes, allocator cache bytes, or logical cache payloads.

## Domain limits

The report lists exact planned and reached shapes without interpolation.
Only cases without stop boundaries contribute to these lists.
Repeated cases contribute one shape but retain separate case evidence.
The domain report counts all legal integer pairs.
Missing output allowances use inclusive integer intervals.
For each represented allowance, missing input lengths also use inclusive integer intervals.

`exact_shape_domain_exhausted` only describes enumeration of those integer pairs.
It does not establish coverage of model inputs, execution schedules, allocation overlap, or host conditions.
The default calibration allowances 1 and 32 do not cover a service allowance of 2048.
The report does not scale serial observations to multiple sequences.

Synchronized observations can change allocation overlap and timing.
The source identity lacks an explicit operating-system build and device identity.
It also lacks an explicit environment policy and sampling-policy identity.
Tool and adapter hashes retain the available implementation identity.
These limitations remain explicit even when every planned case completes.

This document applies structural ASD-STE100 principles.
Structural checks do not establish full dictionary compliance or certification.
