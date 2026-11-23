---
name: finish-work-session
description: Report recorded token consumption for a Codex or Claude Code work session and all descendant agents (sub-agents), with input cache hits, input cache misses, output tokens, and a per-model breakdown. Use when a developer says "finish this session", "wrap up this session", or asks for session token accounting.
---

Produce a usage report for the entire current main thread, including earlier
turns, resumed work, and recursively spawned agents. This invocation requests
accounting; it does not itself request a commit, push, archive, or agent shutdown.

## Collect usage

Use the bundled Python 3 helper from the main session. It reads local session
logs only: Codex rollouts (plus the optional state database's thread paths and
spawn edges) or Claude Code transcripts (plus their sub-agent transcripts). It
needs no packages, credentials, network calls, or additional model requests.

Resolve the project/worktree being worked on and the absolute skill directory
(Claude Code shows it as the skill's base directory). Run from that project,
substituting the actual skill path and naming the agent you are running in:

```bash
# Codex
python3 /absolute/path/to/finish-work-session/scripts/session_usage.py --agent codex
# Claude Code
python3 /absolute/path/to/finish-work-session/scripts/session_usage.py --agent claude-code
```

Without `--agent`, the helper detects the format from `--session-file` or the
environment. Codex identity comes from `CODEX_THREAD_ID`, falling back to
`CODEX_SESSION_ID`, under `CODEX_HOME` or `~/.codex` (`--codex-home` overrides).
Claude Code identity comes from `CLAUDE_CODE_SESSION_ID`, under
`CLAUDE_CONFIG_DIR` or `~/.claude` (`--claude-home` overrides). If identity is
unavailable, use a known main session ID with `--session-id ID` or an explicit
log with `--session-file /path/to/file.jsonl`. Do not guess by choosing the
newest log or a session with the same working directory. Do not substitute
goal-scoped usage for whole-session usage.

Run accounting near the end of the work, after already-authorized agent work has
settled where practical. Do not spawn agents just to prepare this report. If work
is still running, report a snapshot and say so; do not interrupt it merely to get
a total. A live snapshot cannot include its own final answer or later requests.

The helper prints Markdown without creating files. For saved reports, follow the
target checkout's `AGENTS.md`, reuse the active feature's directory, and preserve
prior runs. In this repository the default is
`<project-root>/devlocal/<feat-name>/reports/`; if there is no active feature, use
`finish-work-session`. For example, `--output-dir` writes a uniquely named pair
of Markdown and JSON reports in the supplied directory. Never place task reports
inside the skill installation. `--format json` prints structured data instead.

## Present the result

Return the helper's overview and per-model table, with integer token counts:

- **Upward / input:** cache hit, cache miss, and total input.
- **Downward / output:** recorded output tokens.
- **Total:** input plus output, followed by the input cache-hit rate.
- **Models:** the same counts grouped by recorded model, across all agents.

Include the root thread ID, snapshot time, latest usage timestamp, and coverage:
main thread plus the number of descendants, including any missing or unmetered
agents. Keep the per-session rows in saved reports for auditing; include them in
chat when useful or requested. Surface the helper's warnings and unknown values;
do not describe partial coverage as complete accounting or missing usage as zero.

Input already includes cached input: cache miss is input minus cache hit. Cache
writes remain part of cache misses. For Claude Code, the helper already adds the
API's separate cache-read and cache-creation counters into input; do not add
them again. Output is the recorded output counter; do
not add reasoning output again. Preserve model identifiers, including unknown
attribution, without guessing a model from the current settings or its nickname.

Read [references/accounting.md](references/accounting.md) when interpreting
warnings, investigating discrepancies, or adapting to a changed log schema.
