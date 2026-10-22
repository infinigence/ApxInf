# MiniCPM reasoning closure comparison

Date: 2026-10-23.
Product source revision: `cfc6f88`.
Native worker implementation revision: `b92f330`.
This diagnostic compares existing worker protocols without changing product policy.
The original Claude drawing task remains unaccepted.

## Question

The [calculation record](minicpm-reference-validation-20261023.md) identifies an abrupt forced closure in a diagnostic composition request.
That request has 1253 input tokens and extra guidance.
It does not establish how the original drawing request responds to the closure policy.
This comparison uses the original direct drawing brief, with 158 prepared input tokens.

The HTTP profiles derive random seeds from request identifiers.
Two fresh HTTP requests therefore do not provide a fixed-seed policy comparison.
The raw worker interface accepts an explicit seed and exposes every selected token.
This diagnostic uses that interface for both policies.
It executes no Claude tools and cannot substitute for original client acceptance.

## Matched controls

Both workers load MiniCPM5-2B revision `f97400052a43d642bbc6e9975e2397e3ae6a6b52`.
They use the same worker executable, artifact manifest, MLX library, BF16 compiled variant, and prefill schedule.
The worker executable SHA-256 is `5b7cd2a0af5800321efc1d0eaa3f32e9b848ebd32537724ade8106c966a1ca2a`.
The native runtime remains MLX 0.31.2 on an Apple M4 with 16 GiB of memory.

Both preparations contain identical messages, empty tool lists, thinking options, token IDs, and prompt digests.
Both submissions use temperature 1, top-p 0.95, seed 42, and 4096 output tokens.
Their request timeouts are 600000 milliseconds.
Both worker deployments permit 16384 total positions and at most 8192 output tokens.
The selected request allowance yields a calculated native capacity of 4352 positions.

The comparer checks complete manifests after checking and removing only the intended policy differences.
These differences are the adapter revision, output policy, and bounded reasoning object.
It also checks launch arguments after removing protocol and worker-epoch differences.
Derived model revisions differ intentionally. Capability revisions match.
Each submission uses its own worker's declared revisions and EOS IDs.
Outgoing commands and incoming frames remain separate retained records.

## Observed result

The first 512 selected IDs match exactly.
At index 512, the natural profile selects token 1385, while the bounded profile selects closing token 9.
Neither profile closes its initial prefix before this boundary.
The comparison therefore observes the intended policy intervention with identical preceding token history.

| Observation | Natural profile, protocol 2.3 | Bounded profile, protocol 2.4 |
| --- | --- | --- |
| Selected output tokens | 4096 | 1333, including EOS |
| First closing delimiter | Absent | After selected token 513 |
| Terminal cause | Length | EOS |
| Complete SVG after initial closure | Absent | Present |
| Extracted SVG structural check | Failed because file is absent | Passed |
| Plain SVG response | No | No |
| Visual acceptance | No delivered candidate | Failed |

The bounded tail includes continued planning, Markdown fences, and malformed markup before its complete SVG.
The extracted SVG contains two circles, overlapping green ellipses, detached shapes, and rectangular bicycle components.
Direct inspection of its rendered PNG finds no recognizable pelican riding a bicycle.
Its SHA-256 is `40f1144ac3a067a8f8534b56e536d9fba613393f573647ebba67a6cb3699d016`.
Structural validity does not establish recognizable drawing geometry.

The inspected tail follows the first initial closing delimiter.
The diagnostic does not reproduce later supervisor filtering, stop matching, or tool parsing.
Its extracted SVG therefore does not qualify the complete HTTP response.

## Interpretation and remaining work

For this request and seed, natural closure produces no final content within 4096 output tokens.
The fixed boundary enables an extracted candidate but does not solve its drawing quality.
These observations supply no demonstrated task-quality benefit from removing the boundary alone.
They do not establish an optimal reasoning allowance or general model capability.
The earlier 1253-token calculation case remains separate from this 158-token request.

The [pinned model card](https://huggingface.co/openbmb/MiniCPM5-2B/blob/f97400052a43d642bbc6e9975e2397e3ae6a6b52/README.md) suggests repetition penalty 1.05 for repeated output.
The paired worker observation uses neutral repetition.
A separate public-API probe tests that suggestion without changing a serving profile.
The original real Claude task and rendered drawing remain mandatory acceptance gates.

## Repetition and output allowance

The private probe uses the existing native model and CPU sampler APIs.
Its neutral control reproduces all 158 prepared IDs and all 4096 selected IDs from the natural worker.
This control checks one request and seed, rather than general provider equivalence.
The separate repetition run changes only the penalty to the nearest F32 value of 1.05.
That value is 1.0499999523162842.

| Repetition penalty | Output allowance | Native capacity | Selected tokens | Initial closure or EOS |
| --- | --- | --- | --- | --- |
| 1 | 4096 | 4352 | 4096 | Neither |
| 1.05, rounded to F32 | 4096 | 4352 | 4096 | Neither |
| 1.05, rounded to F32 | 8192 | 8448 | 8192 | Neither |

The repetition run first differs from the neutral run at generated index 10.
The larger run uses a fresh process and increases both output allowance and reserved capacity.
Its first 4096 selected IDs exactly match the smaller repetition run.
The changed capacity therefore causes no observed prefix divergence in this pair.
Neither repetition run delivers final content after initial reasoning.

All three probes use the pinned BF16 model, seed 42, temperature 1, and top-p 0.95.
They use the same probe executable and loaded MLX library.
Their final summaries follow request-state release.
The parent observes all three wrappers exit zero, and the independent reviewer checks completed records and child absence.
These observations do not measure allocator release or qualify KV reuse.

The independent reviewer checks complete IDs, source and library hashes, preparation, and counting semantics.
The event stream includes EOS, while the old probe summary count excludes EOS.
No EOS occurs in these runs, so both counts agree here.
The reviewer labels wrapper exit codes and lock serialization as parent observations.
The records reside in the three `repetition-direct*` directories and `repetition-evidence-review.md` under the ignored feature directory.

For this request and seed, the optional penalty and larger allowance do not resolve missing final output.
These results reject another budget increase as a demonstrated fix.
They do not establish general model inability or identify a native calculation defect.
The next comparison must test a qualified reference generation path before another production policy change.

## Lifecycle, review, and records

Both attempts pass the existing frame validators and attempt tracker through resource release.
Both acknowledge shutdown, report drained state, and exit zero without forced termination.
The measurement wrapper checks each child termination before it releases the lock.
No concurrent model execution occurs within this pair.
These raw-worker outcomes do not qualify HTTP admission or supervisor recovery.

The private probe and comparer use existing first-party APIs and standard-library facilities.
They implement no replacement model, sampler, or external code example.
This diagnostic adds no dependency or product interface.
The independent code review finds no blocker and names the control and lifecycle checks required for interpretation.
The comparer subsequently checks those conditions against the completed pair.

The independent evidence review checks matched controls, complete transcripts, accounting, and the rendered PNG.
It accepts the narrow comparison and agrees with the failed visual verdict.
It identifies an incorrect capability-revision sentence and a misplaced visual record.
The subsequent receipt checks both corrections and the digest-bound visual record at its intended path.

Ignored records reside under `devlocal/serving-minicpm/`.
The `natural-direct4096-seed42/` and `bounded-direct4096-seed42/` directories retain complete command and event transcripts.
The `closure-comparison4096-seed42/` directory retains checked identities, output checks, the extracted SVG, and its PNG.
Its `visual-review.json` binds the failed verdict to the SVG digest.
The private source, design, and review remain in the same feature directory.
These local records do not accompany a fresh Git checkout.

Structural writing checks do not establish a full dictionary review or certification.
