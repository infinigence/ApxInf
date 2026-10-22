# MiniCPM native and reference calculation validation

Date: 2026-10-23.
Source revision: `13d3eb7`.
This diagnostic compares native model output with an explicit CPU reference.
The original Claude drawing task remains unaccepted.

## Question and scope

The [default client record](minicpm-default-context-validation-20261023.md) retains failed tool and direct drawing attempts.
Those results do not isolate native calculation, reasoning policy, or checkpoint quality.
This diagnostic checks calculation at fixed histories before another task-policy change.

A logit is an unnormalized score for one possible next token.
Each observed row contains one score for each of 130560 tokens.
Both providers receive identical prepared input IDs and fixed continuation IDs.
They do not sample their own continuations during this comparison.

The input has 1253 tokens and includes extra composition guidance.
It comes from a retained diagnostic worker request, not the original Claude task.
Its original output allowance is 2048 tokens.
The native capacity remains 3328 positions, including when the capture observes only two rows.
This record does not qualify the 16K context range, tool execution, or visual quality.

The selected source request is `9b70c269-3409-48cc-8c93-3b115d87a403`.
It completed and released its request resources.
The old source driver later ended with an EOF transport fault and worker exit one.
This diagnostic preserves that failure and does not qualify the old driver's complete lifecycle.

## Execution identity

The checkpoint is `openbmb/MiniCPM5-2B`, revision `f97400052a43d642bbc6e9975e2397e3ae6a6b52`.
Both providers load the same local snapshot and BF16 parameters.
The run manifest records configuration, weight-index, and weight-file hashes.
The computer has an Apple M4 and 16 GiB of memory.

| Property | Native capture | Reference capture |
| --- | --- | --- |
| Entry point | Public ApxInf model APIs | Public Transformers model APIs |
| Execution | Metal, `bf16-compiled` | CPU, BF16 parameters, eager attention |
| Runtime | MLX 0.31.2 | Torch 2.12.1, Transformers 5.4.0 |
| Prefill | Evaluated chunks of at most 256 tokens | Chunks of at most 256 tokens |
| Decode | One recorded token per call | One recorded token per call |
| Output projection | Last token row | `logits_to_keep=1` |
| Cache | Native padded capacity | Public dynamic cache |

The native capture checks actual loaded MLX identity through the public runtime query.
The library SHA-256 is `2ee6fbd32ff22e22e1301ebe3c3bece95584104ff9cbc900513d41a095211bbd`.
The reference capture records mapped Torch libraries through the operating system's image API.
The parent records their file hashes separately from package versions.
Reference parameters occupy CPU storage and contain 2516756480 BF16 elements.

The checkpoint configuration declares Transformers 5.6.2, while the installed reference uses 5.4.0.
CPU arithmetic and dynamic cache layout also differ from native Metal execution.
These differences limit attribution when providers disagree.
Neither provider silently changes its device, precision, snapshot, or implementation after failure.

## Row alignment

Row 0 follows the complete prepared input.
Row 1 follows its first recorded output token.
Later row j follows exactly j recorded output tokens.
The decode call receives the preceding recorded token at its original absolute position.

The bounded source selected closing token 9 at output index 512.
Row 512 contains the model's raw scores before that selection.
The diagnostic does not substitute the sampler's synthetic forced-choice row.
Row 513 follows the recorded closing token.

The retained source output and driver report identify the request and intervention.
The original submit commands are absent from the retained output.
The allowance therefore comes from the driver report, not a reconstructed submit frame.

## First observation

Both fresh capture processes completed rows 0 and 1, then exited zero.
The parent checked both process reaps without termination signals.
Each row contains 522240 bytes of little-endian F32 data.
All 130560 scores are finite in every row.

Total variation measures the probability mass that must move between two distributions.
The comparator uses the complete vocabulary and stable F64 probabilities at temperature 1.
It also retains raw error, centered error, margins, top-20 overlap, and recorded-token log probabilities.
No historical scalar tolerance defines a universal equivalence gate.

| Row | Native greedy ID | Reference greedy ID | Total variation | Top-20 overlap | Maximum raw error |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 5899 | 5899 | 0.000646049 | 20 | 0.40625 |
| 1 | 354 | 354 | 0.000027835 | 19 | 0.28125 |

These observations show matching greedy choices and nearby distributions at two fixed histories.
They supply no positive evidence of a large calculation fault at those positions.
They do not establish general numerical equivalence or adequate drawing ability.

## Extended observation

The extended capture selects rows 0, 1, 511, 512, and 513 and replays every intervening token.
Both processes completed, exited zero, and reached successful parent reaps without termination signals.
The current offline comparer checked their retained artifacts after completion.
All scores remain finite in all ten selected rows.

| Row | Native greedy ID | Reference greedy ID | Total variation | Top-20 overlap | Maximum raw error |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 5899 | 5899 | 0.000646049 | 20 | 0.40625 |
| 1 | 354 | 354 | 0.000027835 | 19 | 0.28125 |
| 511 | 880 | 880 | 0.018167516 | 20 | 0.34375 |
| 512 | 33 | 33 | 0.00000008043 | 20 | 0.3125 |
| 513 | 62 | 62 | 0.052540276 | 19 | 0.625 |

Every selected greedy ID matches, but row 513 moves approximately 5.25 percent of probability mass.
Matching greedy IDs therefore does not establish numerical equivalence.
Rows 0 and 1 repeat bitwise within each provider across the initial and extended captures.

The second extended capture also completes and passes current offline qualification.
All five raw rows repeat bitwise within each provider across both extended captures.
Both native and CPU processes exit zero, and their parents check successful reaps without termination signals.
The cross-provider differences therefore remain stable across these two repetitions.
This stability does not identify which arithmetic differences produce them.

The original source closes its reasoning after the incomplete text `Left wheel center: (200`.
At row 512, both raw distributions instead prefer comma token 33.
Both give closing token 9 a probability of approximately 6.70e-13 before the policy intervention.
The next recorded token is 62, which decodes to `I`.
The source then starts a public statement about creating the complete SVG.

These observations identify an abrupt policy boundary inside a planned coordinate.
They do not prove that this boundary causes the failed drawing.
A controlled reasoning-policy comparison remains necessary.

## Process and artifact checks

The existing measurement wrapper owns the lock throughout serial native and CPU execution.
The coordinator owns both capture children and waits for each process before starting the next.
The child owns row publication, its execution manifest, and its cleanup acknowledgement.
The parent owns command, deadline, signals, exit status, successful reap, and artifact hashes.

The native first capture reports successful request release and allocator-cache clearance.
After model drop, its allocator reports zero active bytes and zero cached bytes.
These allocator observations are separate from process memory and the parent's reap result.
The CPU capture drops model, output, and cache references.
It does not measure CPU allocator release.
Its parent separately checks process exit and reap.

Both validators pass the same 13 original case fixtures.
Eight local checks cover distributions, process timeout, coordinator interruption, malformed manifests, row inventories, and execution schedules.
They also reject offline qualification after an interrupted child exits zero.

The independent initial audit checks all four first-run raw rows and reproduces their greedy choices and probability distances.
It also identifies coordinator validation gaps, which the subsequent implementation closes.

The independent final code review accepts the bounded diagnostic and passes all eight local checks.
It checks both 13-fixture validators and rechecks the initial raw-row calculations.
Its approval does not cover production serving or the extended observations.
The independent evidence review reproduces all five probability distances and forced-token probabilities.
It checks source provenance, six process outcomes, and all repeated rows.
It accepts this bounded report and retains the original drawing failure.

Each run retains its original diagnostic source hashes.
The initial run and first extended run precede some coordinator validation corrections.
The stricter current offline comparer subsequently checks their retained artifacts.
This later qualification does not rewrite the original source provenance.

## Evidence and remaining gate

Ignored local records reside under `devlocal/serving-minicpm/native-reference/`.
The `first/` directory contains initial rows, manifests, parent records, and metrics.
The `closing/` directory contains the extended attempt.
The `closing-repeat/` directory contains its separate repetition.
The two `*-repeatability.json` files retain the within-provider comparisons.
The separate `native-reference-design/` directory contains independent candidates, the judge, selected interface, and reviews.
These local artifacts do not accompany a fresh Git checkout.

The [acceptance plan](minicpm-claude-priorities-20261022.md) retains the original drawing and client gates.
Calculation checks cannot substitute for those gates.
Any reasoning-policy or model-capability conclusion needs its own controlled task observation.
The next unit tests the reasoning boundary before changing the serving policy.
Structural writing checks do not establish a full dictionary review or certification.
