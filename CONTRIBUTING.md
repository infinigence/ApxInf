# Contributing to ApxInf

## Environment setup

- Linux with an NVIDIA driver and CUDA toolkit (see the "NVIDIA build
  environment" table in the README), or macOS for CPU-only Rust work.
- Rust stable via [rustup](https://rustup.rs).
- [just](https://just.systems): `cargo install just`. All commands run
  through the justfile — run `just --list` first.
- [actionlint](https://github.com/rhysd/actionlint/blob/main/docs/install.md)
  for workflow validation (`brew install actionlint` on macOS).
  CI installs a pinned official release and verifies its SHA-256 checksum.
- Create a Python virtual environment and install the CPU test dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e 'python/apxinf[test]'
```

```bash
git config core.hooksPath .githooks   # commit-time quick checks
just pre-pr                           # full local check; CI runs the same
```

## Development loop

1. Branch from `main`: `git checkout -b <user>/<topic> upstream/main`.
2. Make changes; run `just check` and `just test-cpu` as you go. GPU work
   needs `just test-gpu <family>` or `just bench <model>` on a real board.
3. Docs change with the code in the same PR. New `APXINF_*` env vars go
   into `doc/reference/env-vars.md` in the same PR.
4. Before opening the PR, run `just pre-pr` and let the `apxinf-pre-pr`
   skill review the diff against `REVIEW.md`.
5. Open the PR from your fork and fill in the evidence section of the
   template.

## PR evidence

Every performance, accuracy, or numerics claim needs: the just recipe run,
the board/arch, the commit, and what was skipped and why. "Not verified" is
an acceptable answer; "pass" without running is not.

## Documentation policy

The taxonomy lives in `doc/README.md#how-docs-are-organized`. Rules of
thumb:

- Commands live in the justfile; docs name recipes.
- Contracts and catalogs live in `doc/reference/`; procedures in
  `doc/guides/`; one file per model in `doc/models/`; dated, frozen
  measurements in `doc/reports/`; proposals with a `Status:` line in
  `doc/design/`.
- Reports record commit + board + recipe and are never edited after
  publication.
- Don't write what a type, test, or script already enforces; don't write
  per-file tours; don't keep TODOs in docs (use GitHub issues).

## AI assistance policy

- AI tools (Claude Code, Codex, Copilot, …) are welcome for development.
- The human submitter is responsible for every line: review it, understand
  it, and be able to explain it. Do not submit code you cannot explain.
- Declare AI assistance in the PR template's "AI assistance" field.
- Do not add AI attribution trailers to commits (`Assisted-by`,
  `Co-Authored-By: Claude`, …) — attribution lives in the PR, not the
  commit history.
- Agents must not `git push`, open PRs, or merge on their own; a human
  performs those steps.

## Review

- With the Git hook enabled above, `just pre-commit` runs on local commits;
  CI runs `just pre-pr`.
- `just test-tooling` tests recipe dispatch, documentation links and anchors,
  the Claude commit guard, and workflow configuration without a GPU or API key.
  `just workflow-lint` validates workflows with actionlint. Both run in
  `just pre-pr` and CI.
- The committed `.claude/settings.json` automatically installs a Bash
  `PreToolUse` commit guard for Claude Code users of this checkout. It runs
  the quick check when a command may invoke `git commit`: failed checks block
  the tool (exit 2); missing `just` emits a warning and allows the tool without
  claiming checks passed. The separate Stop hook only enforces acceptance when
  `APXINF_AGENT_ACCEPTANCE_PLAN` is set.
- To bypass only the Claude commit guard, start a session with
  `APXINF_SKIP_COMMIT_GUARD=1 claude`; remove that variable and restart to restore
  it. This does not disable the Git hook or CI. Document the reason for bypassing
  and run the checks before submission.
- The optional Git hook covers aliases, other clients and human shells.
  `git commit --no-verify` skips that Git hook for one commit;
  `git config --local --unset core.hooksPath` removes this checkout's explicit
  hook-path setting. Neither bypasses Claude's PreToolUse guard. Unsetting
  `APXINF_AGENT_ACCEPTANCE_PLAN` disables only the Stop acceptance guard.
- Maintainers can additionally request an AI review by commenting
  `/ai-review` on a PR (see `.github/workflows/ai-review.yml`). Its output
  is advisory, like any other review; a human still approves.
- Reviewers: follow `REVIEW.md`.
