# Accounting contract

This helper supports local Codex JSONL rollouts and Claude Code JSONL
transcripts (see [Claude Code transcripts](#claude-code-transcripts)). These
persisted schemas vary by runtime; they are not a billing API. The formats below were verified against the
local installation on 2026-11-16. Unknown or absent data must remain visible.

## Scope and ancestry

The first `session_meta.payload.id` identifies a rollout. Later metadata may be
copied parent history. Direct `parent_thread_id` and
`source.subagent.thread_spawn.parent_thread_id` establish ancestry. A
`forked_from_id` is an ancestry edge only for a subagent source; an ordinary user
fork is a separate work session. The optional read-only `state_*.sqlite` index
supplies `threads(id, rollout_path)` and
`thread_spawn_edges(parent_thread_id, child_thread_id)` so missing child files
can be reported. A database edge describes ancestry, not current liveness.

Scan both `sessions/` and `archived_sessions/`, plus indexed rollout paths.
Count each thread once. If more than one file claims a thread, choose the largest
snapshot (mtime and path break ties) and flag the ambiguity instead of adding
overlapping copies. Explicit `--session-file` wins for its thread.

Copied ancestor turn IDs and records with another `thread_id` are excluded.
When a selected thread was forked, load ancestor turn identities only to identify
copied history; do not add ancestor usage to that selected thread. The new input
charged for supplying inherited context to a child is still real consumption.

## Counters and model attribution

Prefer `token_usage_record.payload.usage`, which records one response.
Deduplicate by `response_id` within a thread; contradictory duplicates warn.
Use `thread_id` for ownership, not `session_id` (the latter may name the root).
`turn_token_usage` and `thread_token_usage` are cumulative, not additional usage.

Take a record's explicit `model` if present; otherwise use the matching
`turn_context.payload.turn_id` and `model`, or the preceding turn context when
no turn ID is available. Use `unknown` if a supplied turn ID has no model.
These are recorded/requested model identifiers, not proof of an undisclosed
provider-side routing choice. Do not relabel them from a catalog or the current
configuration.

For legacy rollouts without response records, difference successive
`event_msg` / `token_count` / `info.total_token_usage` snapshots. Repeated totals
and `info: null` add nothing. When the first snapshot includes multiple responses,
assign it to the current model only if it agrees with `last_token_usage`; otherwise
place it in `unknown` because earlier model attribution is unavailable. A larger
delta than `last_token_usage` is similarly unattributed. Counter regressions are
ambiguous: keep the previous high-water mark and flag partial coverage instead
of inventing a reset baseline or charging a replay again.

For rollouts containing response records, compare their sum with the latest
authoritative cumulative checkpoint (`thread_token_usage` or a token-count
snapshot). A positive, consistent gap goes into `unknown` with a warning; never
sum cumulative snapshots on top of the response records. Lagging token-count
events are superseded by newer response checkpoints. Conflicting counters are
reported as partial coverage. Missing cache fields produce unknown cache totals,
not a guessed 0% hit rate. Cache writes are a subset of uncached input and are not
added separately. Reasoning output is not added to recorded output.

## Claude Code transcripts

The root transcript is `<claude-home>/projects/<encoded-cwd>/<session-id>.jsonl`;
if more than one project directory claims the ID, the largest is selected and
flagged. Sub-agent transcripts live under `<session-id>/subagents/*.jsonl` next
to it (any project directory is scanned for that session ID). Older versions
wrote sub-agent turns inline with `isSidechain: true`; those are reported as
`sidechain:<agentId>` rows of the root transcript.

Only `type: "assistant"` entries carry usage, in `message.usage`. One API
response is persisted as one entry per content block, all sharing `message.id`
and the same usage, so deduplicate by `message.id` (fall back to `requestId`).
If a duplicate has a larger `output_tokens` (streamed partial blocks), keep the
larger; a differing input count warns. Message IDs are deduplicated across the
whole tree, so history copied into a forked sub-agent is counted once.
Entries with model `<synthetic>` are client-generated and skipped.

Anthropic `input_tokens` excludes cache traffic. Input total is
`input_tokens + cache_creation_input_tokens + cache_read_input_tokens`; cache hit
is `cache_read_input_tokens`; cache miss therefore includes cache writes, as for
Codex. Missing cache fields make the cache split unknown. The model is
`message.model`, verbatim.

Claude Code does not persist every auxiliary request (for example, session
titles or other background side queries) as an assistant entry, so those are
absent. A resumed session that copies earlier history into a new transcript
counts that history once as part of the resumed session.

## Limits and verification

Reports are snapshots of durable records, not invoice totals. A running request,
the report's final answer, unflushed events, missing files, and agents without
usage telemetry may be absent. The helper never estimates tokens from text.
It checks file sizes before/after parsing and flags changes; this is not an
atomic snapshot of all threads. A malformed record or incomplete final JSONL
line warns and makes coverage partial while retaining other readable records.

Run the maintained regression suite from the project root:

```bash
python3 -B -m unittest discover -s skills/finish-work-session/tests -v
```

The suite uses synthetic logs and databases in
`devlocal/finish-work-session/` in this checkout, with no live model requests.
For a finished real session, also compare the report with the sum of its unique
response records and the final cumulative counters of each descendant.
