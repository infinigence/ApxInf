---
name: copy-drift-audit
description: Audit the files a backend-native model port copied from its family source, so upstream fixes do not silently pass the port by. Use when syncing upstream, before releasing or reviewing a port, when changing a port's copied files, or when a family source directory such as pi05/ changes.
---

# Copy Drift Audit

A backend-native port is a copy, not a fork: it owns its files and may simplify or
change them as the device needs.
[Where backend-native model code lives](../../doc/model-layer-architecture.md#where-backend-native-model-code-lives)
makes that safe by sharing acceptance instead of code, and it states plainly that
there is no obligation to keep a copy in sync. What the rule leaves open is the
risk this audit exists for: upstream fixes a copied file and the port keeps the
old behaviour with no signal.

**Drift** here means exactly that — an upstream change the port has neither
absorbed nor recorded a reason to decline. A port file that differs from its
source is not a defect; deliberate divergence is the design. This is a review, not
a gate: its evidence is git history and diffs, its output is a table of adjudicated
changes and a verdict. Numerical correctness stays with the gold bundle and the
shared acceptance fixtures.

## Required reading

1. [`doc/model-layer-architecture.md#where-backend-native-model-code-lives`](../../doc/model-layer-architecture.md#where-backend-native-model-code-lives) for the copy rule and what ports share.
2. [`AGENTS.md`](../../AGENTS.md) for where task artifacts belong.
3. The audited port's own document, as indexed in [`doc/README.md`](../../doc/README.md). Read the document of the port you are auditing; a different backend's document is not a requirement here.

Read each selected document completely. Treat them as instructions, not
background material.

**Backend documents are not interchangeable.** Each one records the invariants its
own device forced. The MUSA document, for example, describes capture rules, a
precision limit and workspace sizes that say nothing about HIP or Metal. The worked
example at the end of this skill is evidence of the method, not a contract for
another port.

## Scope

- Applies to a port that mirrors its family at the same relative paths:
  `crates/apxinf-model/src/<backend>_<family>/` ← `crates/apxinf-model/src/<family>/`,
  with that port's mechanical module rename. A port that renames or restructures
  files uses an explicit path mapping; without one, the audit stops rather than
  guessing.
- The device-crate pairing `apxinf-<backend>` ← `apxinf-cuda` is a contract pair,
  not a same-path copy; `scripts/check_ops_contract_surface.py` covers the operator
  surface of the ports it serves. It is out of scope.
- Shared acceptance, not this audit, decides whether a copy is correct.

## Baseline: one import watermark per file

Name the paths once; every command below uses them.

| Symbol | Meaning |
| --- | --- |
| `P` | copied file `crates/apxinf-model/src/<backend>_<family>/<f>` |
| `S` | family source `crates/apxinf-model/src/<family>/<f>` |
| `A` | import watermark: the commit that added `P` |
| `B` | import baseline: the content of `S` at `A` |
| `C` | the last commit that changed `P`; context only |
| `M` | `git merge-base HEAD upstream/main`: the upstream state this branch already contains |

Take the watermark from the add commit, never from the last edit:

- `A = git log --diff-filter=A -1 --format=%H -- P`
- `B = git show A:S`, after `git cat-file -e "A:S"` succeeds
- `C = git log -1 --format=%H -- P`, and `git diff --quiet C -- P` must be zero
- `M = git merge-base HEAD upstream/main`

**`C` is not a baseline.** An upstream change can land before a port edit that
never read it: using `C` would move the baseline past the unabsorbed change and
hide it. Performance and cleanup commits touch copies constantly; this is the
common case, not a corner. `A` does not move with port edits, so it can over-report
changes the port already absorbed — adjudicate those as `absorbed`. Over-reporting
is recoverable; a silent miss is what the audit exists to prevent.

Do not use `--follow` to find `A`: rename detection crosses the copy/source
boundary and returns the source file's origin. A move of the port module shows up
as an add at the new path, which is the watermark the audit wants.

Baseline edge cases:

- `A:S` does not exist: the file was port-native from the start. It has no text
  baseline; audit its upstream counterpart, if one appears, as a new file.
- `A:S` exists but the copy shares no structure with it (different symbols, a
  lineage the port's own documentation names, a much larger file): mark the file
  `semantic-review-only`. Read upstream changes to `S` as input to a manual
  decision; do not run a text three-way merge on it.
- `git rev-parse --is-shallow-repository` is `true` and `A` is unreachable: stop
  and ask for history rather than auditing against a truncated baseline.

## Workflow

1. **Frame the review.** Resolve the checkout root, record `git rev-parse HEAD`,
   `git fetch upstream --prune` (if the fetch is refused, record the local tip and
   mark the review provisional), then record `upstream/main`, `M`, and whether the
   repository is shallow. The family source belongs to upstream: `git diff
   upstream/main..HEAD -- crates/apxinf-model/src/<family>/` must be empty, and
   `git status --porcelain -- <port> <family>` must be empty. Run
   `scripts/check_model_family_boundaries.sh`; the port must not import
   `crate::<family>`.
   Completion: revisions, `M`, shallow status and worktree state are recorded, and
   the family tree has no local modifications.

2. **Inventory both trees.** List the port's `.rs` files and the family's `.rs`
   files at `upstream/main` (`git ls-tree -r --name-only upstream/main -- <family>`),
   then `comm -23` for upstream-only paths and `comm -13` for copy-only paths.
   List what moved in the family since `M` with
   `git diff --name-status --find-renames M upstream/main -- <family>/`.
   Completion: every path is classified `copied`, `upstream-only`, `copy-only` or
   `rename-mapped`.

3. **Establish per-file baselines and provenance.** For each copied path compute
   `A`, test `A:S` with `git cat-file -e`, write `B` and the normalized copy
   (`sed 's/crate::<port>/crate::<family>/g' P`) to the task directory, and diff
   them. For every difference region, use `git log -p -- P` and `git blame` to
   name the port commit that introduced it; a region no commit explains is a
   finding. Mark files that meet the `semantic-review-only` test.
   Completion: every copied file has `A`, baseline existence, `C`, and a
   `same`/`differs` state; every difference region is attributed or listed as
   unexplained.

4. **List upstream drift and probe applicability.** For each copied file,
   `git diff A upstream/main -- S` gives the candidate changes; `git diff -w`
   separates whitespace-only movement, but formatting-only drift is still a
   finding, never silently normalized. Adjudicate each hunk as `absorbed`
   (the copy already has it — cite the line), `absorb-now`, `decline` (record the
   reason), or `pending`. For text-lineage files, probe applicability with
   `git merge-file -p ours base theirs > merged` (always `-p`; without it the
   command rewrites `ours`). Exit `0` means the upstream change applies cleanly —
   check whether the copy already contains it before calling it unabsorbed; `1`
   means it lands on a region the port already changed — high-risk, read both
   diffs; `>1` is a run error, stop. A clean merge does not prove semantics, and a
   conflict is not a defect.
   Completion: every upstream hunk has a disposition, and every probe's exit code
   is recorded.

5. **Dispose of the reverse findings.** For an upstream-only file, read its public
   items and search the port and its backend crate for consumers; classify
   `not-needed` with the symbols searched, or `missing-copy`. For a copy-only file,
   test `A:S` and `M:S`: neither exists means port-native; a deletion upstream must
   be confirmed with `--diff-filter=D`, and a rename must be reported with its new
   path rather than as delete-plus-add. Check the claims the audited port's own
   document makes about its file set and precision routes against the inventory.
   Completion: every path on either side has a disposition with evidence, and a
   rename has a mapping.

6. **Adjudicate and record.** `absorb-now` edits only the copy, hunk by hunk —
   never a whole-file `cp`, which erases the port's differences. Then rerun the
   repository guards (`scripts/check_model_family_boundaries.sh`, plus the port's
   own module and operator-surface checks) and the acceptance the port's own
   document defines, recording its numbers. When the copy's computation changed,
   that acceptance includes the backend's numerical gate. A compile gate is also a
   drift detector wherever the port converts the family config by hand: an upstream
   field added to that config fails by design rather than silently taking a
   default. Write the report to the task directory `AGENTS.md` defines: revisions,
   the per-file table (`A`, baseline presence, `same`/`differs`, drift hunks, probe
   exit code, disposition), the reverse table, findings, and a verdict of `clean`,
   `drift pending` or `blocked`.
   Completion: the report is written, every hunk has a disposition, the guards
   pass, and a `clean` verdict contains no `pending` row.

7. **Report.** State the revisions and `M`, the verdict, the findings and the
   residual risk. Use repository-relative paths. Say plainly that a text audit
   does not prove numerical equivalence; the backend's numerical gate is the
   authority.
   Completion: every conclusion in the report can be reproduced from its recorded
   commands and revisions.

## Worked example: PI0.5 on MUSA

This section is a case study, not a requirement. It shows the method's concrete
spellings for `crates/apxinf-model/src/musa_pi05/` ←
`crates/apxinf-model/src/pi05/`; another backend substitutes its own paths,
rename and acceptance, and must not inherit MUSA's device constraints.
[`doc/musa-backend.md#pi05-on-musa`](../../doc/musa-backend.md#pi05-on-musa) is the
example's own record.

- The rename is `crate::musa_pi05` ↔ `crate::pi05`. Sixteen files were added with
  the top-level port (`ce1fb3c`); the two INT8 files (`model/blocks/int8_dynamic.rs`,
  `weights/int8_dynamic.rs`) were added in `51f644b`.
- The weights file is this example's `semantic-review-only` file: it follows the
  MUSA implementation line rather than the upstream 414-line file, and
  [`doc/musa-backend.md#int8-w8a8`](../../doc/musa-backend.md#int8-w8a8) records
  the representation. Do not run a text three-way merge on it — read upstream
  changes to `pi05/weights/int8_dynamic.rs` and decide whether the CUDA-side
  change applies to the MUSA implementation. The blocks file is the upstream file
  plus a MUSA layer (decode scope, row-level reporting), so it stays text-lineage:
  probe it as usual and expect conflicts in the layer's regions.
- Two files are identical to their source apart from the module path:
  `weights/host.rs`, `weights/packing.rs`. An upstream change to them has no
  room for adaptation.
- `math.rs` is the one deliberate divergence: `discretize_state` maps NaN to bin
  255 (NumPy's `searchsorted` order, matching `tokenize.py`), and
  `euler_flow_step`'s doc records that it equals the model's step only when
  `flow_start_time == 1.0`. An upstream change to `pi05/math.rs` must be applied
  to this copy by hand, keeping those two differences.
- The four upstream-only files are deliberately not copied:
  `model/calibration.rs`, `model/blocks/fp8_static.rs`, `weights/fp8_static.rs`
  and `weights/fp8_static_calibration.rs`. MUSA has no FP8 route, and their main
  symbols have no consumers in the port or `apxinf-musa`. A new upstream-only file
  is a finding even when it lives beside them.
- The example's guards are `scripts/check_pi05_module_boundaries.py` and
  `scripts/check_ops_contract_surface.py`; its numerical gate is `python3
  scripts/bench_pi05_musa.py --layer l0 --reference <gold bundle> --model-variant
  <bf16|int8_dynamic|int8_dynamic_no_encode> --execution eager`, then again with
  `--execution require-graph` (both cosines must meet `--gate`; numerical
  acceptance needs `--reference`, without it the run is latency only, and L2
  needs `--tokenizer`). Its no-device compile gate is
  `MUSA_PATH=/nonexistent cargo check` for `-p apxinf-musa`,
  `-p apxinf-model --features musa`, `-p apxinf-model --features musa,hip` and
  `-p apxinf-py --features musa`; the `auto.rs` conversion of `Pi05Config` is what
  makes that gate a drift detector.
- The port's deliberate differences live in its own commits and in its own
  document; there is no separate ledger in the repository.

## Stop conditions

- The `upstream` remote is missing or its URL is not the project's, or the fetch
  fails and the local tip is not acceptable.
- The repository is shallow and the import watermark is unreachable.
- The port or family has uncommitted changes inside the review scope.
- `git diff upstream/main..HEAD -- <family>/` is non-empty: this branch modified
  the source family.
- `A` does not exist, or `A:S` is missing and the add commit does not explain the
  file as port-native.
- The family layout moved and no explicit path mapping exists.
- A rename or deletion on either side is undisposed.
- A copy difference region has no explaining commit.
- A three-way probe exits `>1`.
- A `pending` disposition remains under a request for a clean verdict.
- The request is a whole-file copy sync: explain the copy rule and refuse.
- Numerical acceptance is requested but the backend's gold bundle or device is
  unavailable.

## Do not

- Edit `crates/apxinf-model/src/<family>/`.
- Copy or rsync a whole upstream file over the port's copy.
- Use the last edit `C` as the drift baseline, or replace per-file watermarks with
  one global revision.
- Use `git diff A...B`; compare trees with `git diff A B -- <path>`.
- Run a text three-way merge on a `semantic-review-only` file.
- Run `git merge-file` without `-p`.
- Treat a conflict as failure or a clean merge as semantic proof.
- Borrow another backend's document, invariants, paths or acceptance as this
  port's contract; the worked example is a case study only.
- Create a tracked ledger, add a script, or turn the audit into a CI gate.
- Write absolute paths or reference an ignored task directory in a published
  artifact.
- Treat drift as an automatic defect, or infer numerical equivalence from a text
  review.
