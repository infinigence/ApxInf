# Review Report Format

Simplified Chinese version: [report-format.zh-CN.md](report-format.zh-CN.md).

Start with version metadata so freshness can be checked mechanically:

```yaml
---
repo: infinigence/ApxInf
pr: 123
author: login
base_sha: full-sha
head_sha: full-sha
merge_base_sha: full-sha
changed_files: integer
additions: integer
deletions: integer
reviewed_at: RFC3339 timestamp
coverage: complete | qualified
coverage_notes: empty or concise limitation
verdict: block | needs-work | lgtm
---
```

Use only the sections that add information:

1. Summary and motivation.
2. Architecture or execution-path change.
3. Changed modules and affected models, devices, formats and APIs.
4. Validation evidence: tests, CI, benchmarks and missing qualification.
5. Findings ordered by priority.
6. Overall impact and verdict.
7. Re-review delta when the head changed.

For each finding include:

```text
[P1] Imperative title
File: repository/relative/path:changed-line
Trigger: concrete input, platform, configuration or schedule
Evidence: relevant control/data flow from the pinned comparison
Impact: observable failure or compatibility consequence
Direction: concise repair or validation direction
```

Anchor removed code to the original side. State when the patch or exact source
could not be inspected. Do not claim “no findings” when coverage is qualified by
missing files, inaccessible commits, or unreconciled counts.

GitHub comments should omit report metadata and retain title, trigger, evidence,
impact and repair direction. Avoid posting summary-only comments or duplicate
comments on every PR in a stack; comment at the root introduction when the issue
is inherited unchanged and mention affected descendants in the private report.
