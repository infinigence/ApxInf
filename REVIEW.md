# Review instructions

These rules govern `/ai-review` (GitHub workflow), the `apxinf-pre-pr` skill,
and human review. Local `/code-review` does not read this file; the skill and
the workflow both tell the agent to read it first.

## What Important means here

Reserve 🔴 Important for:

- Violating the model / model_runner / weights responsibility split or the
  model→backend dependency direction (doc/reference/model-layer-architecture.md).
- New kernels or L3 candidates added to legacy `apxinf-cuda` instead of
  `apxinf-cuda-new`.
- An L3 candidate without its independent Torch-oracle reference test
  (see the `l3-reference-testing` skill).
- Numeric or performance claims without board, arch, commit, and the just
  recipe used — or that compare numbers from different boards/builds.
- A behavior change without the matching authoritative doc update in the
  same PR.
- Reporting "pass" for something that ran without the required device or
  checkpoint.
- A new `APXINF_*` environment variable not registered in
  doc/reference/env-vars.md.
- Host-specific absolute paths written into maintained docs.

## Nits

Style, naming, and phrasing are Nit at most. Post at most 5 inline Nits per
review; if you found more, say "plus N similar items" in the summary. If
everything found is a Nit, lead the summary with "No blocking issues."

## Do not report

- Anything CI or `just pre-commit` already enforces (fmt, doc-lint,
  boundaries, dead links).
- Vendored CUTLASS / fa2 sources, `*.lock`, `devlocal/`, `doc/archive/`.
- Pre-existing issues you did not verify against current code; label them
  "pre-existing, unverified" instead.

## Always check

- New public API surface has at least one caller test.
- Shared binding/backend changes list affected public callers and their measured
  or unverified status; private harnesses are not public-path acceptance.
- Model evidence identifies effective grids, actual graph/eager mode, full output
  comparison and loaded build. Final source/harness edits invalidate old evidence.
- `just --list` covers any new command the PR introduces.
- Docs touched by the PR still pass `just doc-lint`.

## Verification bar

Behavior claims need a `file:line` citation in the source, not an inference
from naming. If you cannot point at the line, downgrade the finding or drop it.

## Convergence

On the second and later reviews of the same PR, report Important findings
only — do not re-litigate Nits.
