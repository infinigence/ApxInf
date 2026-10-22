# Bounded native reasoning validation

Date: 2026-10-23.
Native worker revision: `b92f330`.
Context probe revision: `f22255a`.
The [profile contract](native-bounded-thinking-v0.1.md) defines the interface.
Native control checks passed.
The original pelican task failed visual acceptance despite a completed client tool cycle.
An external-feedback revision also failed without changing the SVG.

## Scope and rationale

The existing thinking profile has no independent initial reasoning budget.
A prior compatible Claude revision reached its 1200-second capture deadline without a completed Edit.
Its SVG remained unchanged.
The [earlier evidence](minicpm-native-validation-20261022.md) retains that failure.

The new explicit provider is `native-minicpm5-thinking-bounded`.
It retains MiniCPM5-2B, BF16, MLX 0.31.2, 16384 total positions, and one active sequence.
The model snapshot remains `f97400052a43d642bbc6e9975e2397e3ae6a6b52`.
Worker protocol 2.4 declares a fixed initial reasoning budget of 512 tokens.
Every selected token, including the constrained close and EOS, consumes the ordinary output allowance.
The profile retains exact public usage and context admission.

Two designs compared worker selection ownership with a model generation hook.
The selected design adds a private worker sampler and preserves the core sampler interface.
The constrained close calls the existing CPU sampler once.
That call preserves history, capacity, and random-counter advancement.
The scratch prototype omitted that call and cannot qualify the production random stream.

## CPU checks

The complete native serving test command passed 189 tests and ignored one existing macOS sensor test.
The totals comprise 159 library tests, 23 worker tests, one HTTP integration test, and six shutdown tests.
Python ran 276 serving tests and skipped six existing conditional HTTP tests.
The remaining 270 Python tests passed.

```sh
cargo test -p apxinf-serving --features native-minicpm5
python3 -m unittest discover -s tests/python -p 'test_serving*.py'
```

The original 2.4 corpus contains 85 frame cases, ten event transcripts, and three digest vectors.
Rust and Python consume that same corpus.
Earlier fixture files remain unchanged.
The new identity rejects changed budgets, closing IDs, selection policies, and use under earlier protocols.
The launcher checks the complete runtime and execution identity.
Readiness advertises the fixed budget without claiming Anthropic extended thinking.

The worker tests use an original CPU toy model and tokenizer.
They cover natural closure across multiple IDs, partial delimiters, later literal delimiters, and per-request reset.
A 600-step reference sequence checks random continuation through the constrained selection.
Generation tests cover allowances 0, 512, 513, and 514.
They check EOS usage, callback refusal, and ordinary consumption of token 9 by the next forward operation.
Supervisor tests check raw stop offsets, hidden length completion, incomplete EOS, cancellation, expiry, and release.
The context probe accepts the declared text profiles and rejects a profile change during a case.
Its format diagnostic distinguishes invalid JSON structure from incorrect fact values.
Code fences still fail the strict JSON check.

Independent interface and worker reviews found no blocking correctness issues.
The interface review identified an unqualified closure promise in the local guide.
The guide now states the shorter-allowance and earlier-termination exceptions.
Comment review found no new comments or suppressions.

## Native checks

The direct native probe used 1253 input tokens and seed 42 for random selection.
It used the diagnostic composition prompt from the earlier prototype.
That prompt includes additional drawing guidance and is not the original Claude task.

| Selection | Output allowance | Selected tokens | First complete close | Terminal cause |
| --- | ---: | ---: | ---: | --- |
| Random | 512 | 512 | None | Length |
| Random | 513 | 513 | 513 | Length |
| Random | 2048 | 1253 | 513 | EOS |
| Greedy | 1024 | 514 | 513 | EOS |

All three closing selections at index 512 returned token 9.
Worker usage equals every observed selected token, including EOS.
Each request received a resource release event.
These results establish control and accounting, not useful final content.

The first diagnostic driver closed stdin without the required shutdown command.
The worker correctly reported an input transport fault and exited one after all four requests settled.
The original failed driver record remains unchanged.
A separate one-token request then checked explicit shutdown acknowledgement, drain, and worker exit zero.
Neither run required forced termination.

Both HTTP APIs passed ten sampling-policy cases.
Hidden one-token completion and a visible marker response passed.
The service counted 9011 input tokens and rejected an additional 8192-token output reservation.
This rejection preserves the 16384-position limit.

The long-context case used 15354 input tokens and a 1024-token output allowance.
It returned all three expected nonce values with 202 output tokens.
The response added Markdown fences around its JSON object.
The strict format verdict therefore remains failed.
The diagnostic review checks the values without changing that verdict.
Readiness retained the same profile and worker identity.
This result does not activate the 513th-token constrained selection.

A separate lifecycle run cancelled before public content and observed capacity release.
The same worker then reported ready.
The owned worker failure check also passed.
That fault test deliberately produces service exit one.
Normal service runs exited zero without forced group termination.

## Client and visual checks

One original pelican task used real Claude Code 2.1.289 with the declared acceptance compatibility option.
The option retains the real 16384-position window and exact usage.
It uses a private client hook and is not a stable production API.
The Claude process completed Read, Read, and Edit in 196.6 seconds and exited zero.
The task runner returned two to request a separate visual verdict.
That runner status does not establish task success.
The four requests reported output counts 201, 514, 1473, and 149.
This single run establishes no performance comparison.

The 2246-byte SVG passed structural validation.
Its SHA-256 is `a46ce14c5f6b422ad9817d1b35d8ea9741566f85355c212fafe67f30c731b2d9`.
Direct inspection found horizontal bars, vertical supports, small circles, and disconnected curved shapes.
The image has no recognizable pelican or connected bicycle.
Its visual verdict remains failed.
File creation, XML validity, and client exit zero do not establish visual success.
The current MiniCPM5-2B deployment remains text-only.
It cannot inspect the rendered image through an unsupported image input.

## External-feedback revision

The revision supplied text from direct inspection of the failed original drawing.
The model read the brief, SVG, and review before its modification attempts.
The checker and review remained unchanged.
The launch declared `Bash(python3 check_svg.py)` as an allowed rule.
The client also permitted read-only Bash commands.

The client made twelve tool calls, including three unsuccessful Edit calls.
The first Edit supplied tab indentation where the source used two spaces.
Its `old_string` did not match the source.
The client returned that error, and the model requested Read again.
That Read reported an unchanged file instead of returning its content again.
The model then attempted inspection through Bash.
The `cat -A` command failed on macOS.
The later `head -50 pelican.svg | cat -v` command returned file content.
Later Edit calls also failed to match.
The client denied a Bash attempt to rewrite the file with shell redirection.
The checker passed before any successful Edit.
That result did not satisfy the final-edit requirement.

The final admitted request used 8121 input tokens and an 8192-token output reservation.
Its subsequent history exceeded the available input capacity for that reservation.
The service rejected the next request with HTTP 400 `context_limit`.
The Claude process exited one after 681.7 seconds, without reaching its capture timeout.
Its final result had `subtype=success` but `is_error=true` and the API error text.
Acceptance uses the error flag, process status, tool results, and artifact state together.
It does not trust the subtype alone.

The SVG retained SHA-256 `a46ce14c5f6b422ad9817d1b35d8ea9741566f85355c212fafe67f30c731b2d9`.
Its unchanged bytes retain the original failed visual verdict.
No checker ran after a successful final Edit because no Edit succeeded.
The service settled to idle and exited zero without forced termination.
The launcher released the Metal lock and reaped its owned processes.

This run identifies two remaining constraints.
The model needs accurate literal Edit parameters and recovery from tool errors.
A fixed 8192-token output reservation reduces the input capacity for later tool turns.
The next candidate uses a 4096-token output allowance with the separate 512-token initial reasoning budget.
That candidate needs its own real client and default-policy validation.
It has no accepted quality or compatibility result yet.

## Evidence

Local raw records remain under `devlocal/serving-minicpm/`.
The test logs are `bounded-full-tests.log` and `bounded-python-final.log`.
The release build log is `bounded-build.log`.
The native record directory is `native-bounded-worker-raw/`.
Explicit shutdown evidence is in `native-bounded-worker-shutdown/`.
HTTP and context records are in `native-bounded-claude/`.
Native cancellation and owned fault records are in `native-bounded-lifecycle/`.
The original client artifact and visual verdict are in `native-bounded-pelican/`.
The failed feedback revision is in `native-bounded-revision/`.
Each runtime result must retain its complete worker identity and actual terminal usage.

Structural writing checks do not establish a complete dictionary review or certification.
