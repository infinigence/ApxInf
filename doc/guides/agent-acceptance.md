# Agent acceptance

Use an acceptance plan for model changes or long agent jobs with multiple
deliverables. Ordinary documentation edits need only the existing checks.
The plan declares what must run; receipts record what actually ran. It is an
audit aid, not a numerical oracle, permission grant or security sandbox.

## Declare the task before measuring

Start with `just acceptance init <feature>`. It creates
`devlocal/<feature>/acceptance.json` even on a fresh clone with no `devlocal/`.
The generated plan is deliberately incomplete and cannot pass until its task
criteria and commands are filled in; existing plans are never overwritten.
Every case has an
id, criterion, named entry point and an argv command. Commands run from the
worktree root. `inputs` pins ignored harnesses/fixtures that Git cannot track.

```json
{
  "version": 1,
  "cases": [
    {
      "id": "public-policy",
      "criterion": "Assert effective grids and compare full actions with the baseline for default and rectangular inputs",
      "entry_point": "AutoPolicy.from_pretrained(...).infer(...)",
      "command": ["python3", "devlocal/example/scripts/public_policy_check.py"],
      "inputs": ["devlocal/example/scripts/public_policy_check.py"]
    }
  ]
}
```

Adapt cases to the task. Shared RGB changes need affected public callers, not
only the edited family's Rust harness. Reference comparisons need the known
fixture anchor before new shapes. Preparation changes need explicit/implicit
execution, shape changes, rejected-input behavior and required-graph policy.
Keep thresholds and expected shapes in executable harness assertions. Harnesses
must fail on missing required hardware/checkpoints, zero selected tests,
unexpected fallback or skipped mandatory cases. Exit zero alone does not prove
those properties; review the assertions.

Use `just acceptance run <plan> <case>` to execute the declared command and
capture its full output. Receipts and uniquely named logs live beside the plan.
`just acceptance check <plan>` returns zero only when all cases have successful,
current receipts. It rejects changes to the plan, source files (including new
nonignored files), pinned inputs or logs. Commits alone do not invalidate an
unchanged source fingerprint. Changing any source invalidates all cases.
Installed wheels, submodule contents and remote machines are not fingerprinted:
harnesses must verify and record their loaded extension hash, remote revision,
board, checkpoint, tactics and effective execution mode.

Freeze source and harnesses before final measurements. Later optimizations need
fresh evidence. Performance harnesses assert actual graph/eager mode, record
fallback, reconcile timing scopes, and prove tuning created keys. Full tensor
comparison is required; matching stdout tails is not identity evidence.

## Blocked is a result, not a pass

When a check cannot run, use `just acceptance block <plan> <case> --reason <reason>`.
This records an unresolved item and exits 2. The check command also exits 2
when only blocked items remain (1 for missing, failed or stale evidence).
Report affected capabilities as UNVERIFIED. A blocked receipt is not a waiver
to claim readiness; changing acceptance scope needs an explicit task decision.

For Claude Code, the repository Stop hook is opt-in via
`APXINF_AGENT_ACCEPTANCE_PLAN=<absolute-plan-path>` in the launch environment.
Without it, ordinary sessions are unchanged. Missing/failed/stale evidence causes
one corrective turn. A repeated Stop or explicitly blocked cases can end the
session with an INCOMPLETE diagnostic; neither writes a success marker. External
launchers must use the acceptance command's exit status to classify success,
not report headings, file existence, CLI process exit zero or absence of a Stop
block. Codex and other agents use the same CLI; this Claude hook does not
automatically integrate with those runtimes. The private headless launcher is
not shipped in this repository.

## Operating long jobs

Keep evaluator notes and expected diagnoses outside agent-visible job directories.
Expose only the brief, operator inbox and worktree evidence directory. Use
OS/container mounts and remote account permissions to make checkpoints and
reference repositories read-only. Shell keyword hooks cannot enforce permissions
across SSH, Python, aliases or scripts; this hook does not try to.

After a board crash or operator prohibition, stop the affected workload and
record the blocker. Resume only after the operator clears that workload; a board
responding to SSH is not permission to repeat the crash. Timebox investigations
and checkpoint evidence before compaction. Never interpret compaction, retry
limits or an interrupted agent as acceptance success.
