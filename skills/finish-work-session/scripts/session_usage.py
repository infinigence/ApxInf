#!/usr/bin/env python3
"""Report local Codex or Claude Code session-tree usage without reading message text into output."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import sys
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class Usage:
    input: int = 0
    cached: int | None = 0
    output: int = 0

    def __add__(self, other: Usage) -> Usage:
        cached = None if self.cached is None or other.cached is None else self.cached + other.cached
        return Usage(self.input + other.input, cached, self.output + other.output)

    def difference(self, other: Usage) -> Usage | None:
        cached = None if self.cached is None or other.cached is None else self.cached - other.cached
        result = Usage(self.input - other.input, cached, self.output - other.output)
        if result.input < 0 or result.output < 0:
            return None
        if cached is not None and not 0 <= cached <= result.input:
            return None
        return result

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input,
            "cache_hit_tokens": self.cached,
            "cache_miss_tokens": None if self.cached is None else self.input - self.cached,
            "output_tokens": self.output,
            "total_tokens": self.input + self.output,
            "cache_hit_rate": self.cached / self.input if self.input and self.cached is not None else None,
        }

    def nonzero(self) -> bool:
        return bool(self.input or self.output)


def usage(value: Any) -> Usage:
    if not isinstance(value, dict):
        raise ValueError("missing usage object")
    for key in ("input_tokens", "output_tokens"):
        if type(value.get(key)) is not int or value[key] < 0:
            raise ValueError(f"invalid or missing {key}")
    cached = value.get("cached_input_tokens")
    if cached is not None and (type(cached) is not int or not 0 <= cached <= value["input_tokens"]):
        raise ValueError("invalid cached_input_tokens")
    return Usage(value["input_tokens"], cached, value["output_tokens"])


def add(target: dict[str, Usage], model: str, value: Usage) -> None:
    target[model] = target.get(model, Usage()) + value


def total(values) -> Usage:
    result = Usage()
    for value in values:
        result += value
    return result


def read_header(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        first = json.loads(stream.readline())
    meta = first.get("payload", {})
    if first.get("type") != "session_meta" or not isinstance(meta.get("id"), str):
        raise ValueError("first record is not session metadata with a thread ID")
    return meta


def parent_id(meta: dict) -> str | None:
    source = meta.get("source")
    subagent = source.get("subagent") if isinstance(source, dict) else None
    spawn = subagent.get("thread_spawn", {}) if isinstance(subagent, dict) else {}
    return meta.get("parent_thread_id") or spawn.get("parent_thread_id") or (
        meta.get("forked_from_id") if subagent is not None else None
    )


def discover(home: Path, explicit: Path | None):
    paths = set()
    edges: dict[str, set[str]] = defaultdict(set)
    notices = []
    for dirname in ("sessions", "archived_sessions"):
        directory = home / dirname
        if directory.is_dir():
            paths.update(directory.rglob("*.jsonl"))
    databases = [p for p in home.glob("state_*.sqlite") if p.stem[6:].isdigit()]
    if databases:
        database = max(databases, key=lambda p: int(p.stem[6:]))
        try:
            with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as db:
                tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if "thread_spawn_edges" in tables:
                    for parent, child in db.execute("SELECT parent_thread_id, child_thread_id FROM thread_spawn_edges"):
                        edges[parent].add(child)
                if "threads" in tables:
                    for (raw_path,) in db.execute("SELECT rollout_path FROM threads"):
                        path = Path(raw_path).expanduser()
                        if not path.is_absolute():
                            path = home / path
                        if path.is_file():
                            paths.add(path)
        except (OSError, sqlite3.Error, TypeError) as exc:
            notices.append(f"Could not read optional thread index {database.name}: {exc}")
    if explicit:
        paths.add(explicit)
    candidates: dict[str, list[tuple[Path, dict]]] = defaultdict(list)
    for path in sorted({p.resolve() for p in paths}):
        try:
            meta = read_header(path)
            candidates[meta["id"]].append((path, meta))
            parent = parent_id(meta)
            if parent:
                edges[parent].add(meta["id"])
        except (OSError, ValueError, TypeError) as exc:
            notices.append(f"Could not index rollout {path}: {exc}")
    selected = {}
    duplicates = set()
    for thread, entries in candidates.items():
        if len(entries) > 1:
            duplicates.add(thread)
        selected[thread] = max(entries, key=lambda item: (
            item[0] == explicit, item[0].stat().st_size, item[0].stat().st_mtime_ns, str(item[0])
        ))
    return selected, edges, duplicates, notices


def read_events(path: Path):
    events, warnings = [], []
    before = path.stat().st_size
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                event = json.loads(line)
                if not isinstance(event, dict) or not isinstance(event.get("payload"), dict):
                    raise ValueError("record or payload is not an object")
                # Retain only accounting metadata, never messages or tool bodies.
                if event.get("type") in ("session_meta", "turn_context", "token_usage_record"):
                    events.append((line_number, event))
                elif event.get("type") == "event_msg" and event["payload"].get("type") in (
                    "token_count", "task_started", "turn_started"
                ):
                    events.append((line_number, event))
            except (ValueError, TypeError) as exc:
                warnings.append(f"Line {line_number}: unreadable record ({exc}).")
    if before != path.stat().st_size:
        warnings.append("Rollout changed while being read; rerun for a later snapshot.")
    return events, warnings


def analyze(thread: str, events, inherited_turns: set[str], warnings: list[str]):
    models = {}
    own_turns = set()
    by_model: dict[str, Usage] = {}
    records = []
    snapshots = []
    seen = {}
    current_model = "unknown"
    current_turn = None
    inherited_baseline = Usage()
    last_timestamp = None
    for line_number, event in events:
        payload, kind = event["payload"], event["type"]
        if kind == "turn_context":
            current_turn = payload.get("turn_id")
            current_model = payload.get("model") or "unknown"
            if current_turn and current_turn not in inherited_turns:
                own_turns.add(current_turn)
                models[current_turn] = current_model
            continue
        if kind == "event_msg" and payload.get("type") in ("task_started", "turn_started"):
            current_turn = payload.get("turn_id")
            current_model = models.get(current_turn, "unknown")
            continue
        is_record = kind == "token_usage_record"
        info = payload.get("info") if kind == "event_msg" and payload.get("type") == "token_count" else None
        if not is_record and not info:
            continue
        if is_record and payload.get("thread_id", thread) != thread:
            continue
        turn = payload.get("turn_id", current_turn)
        if turn in inherited_turns and (not is_record or payload.get("thread_id") != thread):
            if info:
                try:
                    inherited_baseline = usage(info.get("total_token_usage"))
                except ValueError:
                    pass
            continue
        model = payload.get("model") or (models.get(turn, "unknown") if turn else current_model)
        try:
            if is_record:
                value = usage(payload.get("usage"))
                response = payload.get("response_id")
                if response and response in seen:
                    if seen[response] != value:
                        warnings.append(f"Line {line_number}: conflicting duplicate response usage.")
                    continue
                if response:
                    seen[response] = value
                else:
                    warnings.append(f"Line {line_number}: response ID missing; duplicate detection unavailable.")
                records.append((line_number, model, value))
                if payload.get("thread_token_usage") is not None:
                    snapshots.append((line_number, model, usage(payload["thread_token_usage"]), value))
            else:
                last = usage(info["last_token_usage"]) if info.get("last_token_usage") is not None else None
                snapshots.append((line_number, model, usage(info.get("total_token_usage")), last))
            last_timestamp = event.get("timestamp") or last_timestamp
        except (ValueError, TypeError) as exc:
            warnings.append(f"Line {line_number}: invalid usage ({exc}).")

    if records:
        for _, model, value in records:
            add(by_model, model, value)
        if snapshots:
            checkpoint_line, _, checkpoint, _ = snapshots[-1]
            measured = total(value for line, _, value in records if line <= checkpoint_line)
            gap = checkpoint.difference(measured)
            if gap is None:
                warnings.append("Cumulative usage disagrees with response records; response records retained.")
            elif gap.nonzero():
                add(by_model, "unknown", gap)
                warnings.append("Cumulative usage contains additional tokens with unknown model attribution.")
    else:
        baseline = inherited_baseline
        for index, (line_number, model, cumulative, last) in enumerate(snapshots):
            if index == 0 and baseline.nonzero() and cumulative.difference(baseline) is None:
                baseline = Usage()  # Native child counters restarted after copied history.
            delta = cumulative.difference(baseline)
            if delta is None:
                warnings.append(f"Line {line_number}: cumulative counters regressed; previous high-water mark retained.")
                continue
            baseline = cumulative
            if not delta.nonzero():
                continue
            if last != delta:
                model = "unknown"
                warnings.append(f"Line {line_number}: cumulative delta has unknown model attribution.")
            add(by_model, model, delta)
    if not records and not snapshots:
        warnings.append("No recorded token usage; consumption is unknown, not zero.")
    for model, value in by_model.items():
        if value.cached is None:
            warnings.append(f"Cache-hit counts unavailable for model {model}; cache split is unknown.")
        if model == "unknown":
            warnings.append("Some recorded usage has no reliable model identifier.")
    return by_model, own_turns, last_timestamp, bool(records or snapshots)


def build_report(home: Path, session_id: str | None, explicit: Path | None = None) -> dict:
    snapshot_time = datetime.now(timezone.utc).isoformat()
    explicit = explicit.expanduser().resolve() if explicit else None
    if explicit:
        identity = read_header(explicit)["id"]
        if session_id and session_id != identity:
            raise ValueError("--session-id disagrees with --session-file")
        session_id = identity
    if not session_id:
        raise ValueError("No current thread ID; supply --session-id or --session-file (never guess the newest log).")
    selected, edges, duplicates, notices = discover(home, explicit)
    if session_id not in selected:
        raise ValueError(f"No readable rollout for root thread {session_id} under {home} or its thread index.")
    event_cache = {}

    def load(thread):
        if thread not in event_cache:
            try:
                event_cache[thread] = read_events(selected[thread][0])
            except (OSError, UnicodeError) as exc:
                event_cache[thread] = ([], [f"Could not read rollout: {exc}"])
        return event_cache[thread]

    def ancestor_turns(thread):
        turns, visited = set(), {thread}
        meta = selected[thread][1]
        ancestor = parent_id(meta) or meta.get("forked_from_id")
        while ancestor and ancestor not in visited and ancestor in selected:
            visited.add(ancestor)
            for _, event in load(ancestor)[0]:
                if event["type"] == "turn_context" and event["payload"].get("turn_id"):
                    turns.add(event["payload"]["turn_id"])
            meta = selected[ancestor][1]
            ancestor = parent_id(meta) or meta.get("forked_from_id")
        return turns

    sessions, all_models = [], {}
    visited = set()
    pending = [(session_id, None)]
    while pending:
        thread, parent = pending.pop(0)
        if thread in visited:
            notices.append(f"Repeated/cyclic ancestry for thread {thread}; counted once.")
            continue
        visited.add(thread)
        pending.extend((child, thread) for child in sorted(edges.get(thread, ())))
        warnings = []
        row = {"thread_id": thread, "parent_thread_id": parent}
        if thread not in selected:
            row.update(status="missing", rollout_path=None, totals=None, models={}, last_usage_at=None,
                       observed_models=[],
                       warnings=["Known descendant rollout is missing; consumption is unknown."])
        else:
            path, meta = selected[thread]
            events, read_warnings = load(thread)
            warnings.extend(read_warnings)
            if thread in duplicates:
                warnings.append("Multiple rollout files claim this thread; selected one snapshot without summing copies.")
            inherited = ancestor_turns(thread)
            if meta.get("forked_from_id") and meta["forked_from_id"] not in selected:
                warnings.append("Fork source unavailable; copied history cannot be fully verified.")
            models, _, timestamp, metered = analyze(thread, events, inherited, warnings)
            observed_models = set(models) | {
                event["payload"]["model"] for _, event in events
                if event["type"] == "turn_context" and event["payload"].get("model")
                and event["payload"].get("turn_id") not in inherited
            }
            for model, value in models.items():
                add(all_models, model, value)
            row.update(status="partial" if warnings and metered else "recorded" if metered else "unmetered",
                       rollout_path=str(path), totals=total(models.values()).as_dict() if metered else None,
                       observed_models=sorted(observed_models),
                       models={k: v.as_dict() for k, v in sorted(models.items())}, last_usage_at=timestamp,
                       warnings=list(dict.fromkeys(warnings)))
        sessions.append(row)
    timestamps = [s["last_usage_at"] for s in sessions if s["last_usage_at"]]
    return {
        "schema_version": 1,
        "agent": "codex",
        "root_thread_id": session_id,
        "snapshot_at": snapshot_time,
        "last_usage_at": max(timestamps) if timestamps else None,
        "coverage": "partial" if notices or any(s["status"] != "recorded" for s in sessions) else "recorded",
        "session_count": len(sessions),
        "descendant_count": len(sessions) - 1,
        "metered_session_count": sum(s["totals"] is not None for s in sessions),
        "totals": total(all_models.values()).as_dict() if any(s["totals"] is not None for s in sessions) else None,
        "models": {k: v.as_dict() for k, v in sorted(all_models.items())},
        "sessions": sessions,
        "warnings": notices,
        "scope_note": "Snapshot of recorded usage; excludes unflushed/running requests, this report's final answer, and later work. Missing usage is not zero.",
    }


# --- Claude Code transcripts -------------------------------------------------

CLAUDE_SKIPPED_MODELS = {"<synthetic>"}


def claude_usage(value: Any) -> Usage:
    """Map Anthropic usage, whose input_tokens exclude cache reads and writes."""
    if not isinstance(value, dict):
        raise ValueError("missing usage object")
    fields = {}
    for key in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"):
        raw = value.get(key)
        if raw is None and key.startswith("cache_"):
            fields[key] = None
            continue
        if type(raw) is not int or raw < 0:
            raise ValueError(f"invalid or missing {key}")
        fields[key] = raw
    read, write = fields["cache_read_input_tokens"], fields["cache_creation_input_tokens"]
    total_input = fields["input_tokens"] + (read or 0) + (write or 0)
    return Usage(total_input, read, fields["output_tokens"])


def claude_root_transcript(home: Path, session_id: str) -> tuple[Path, list[str]]:
    matches = sorted((home / "projects").glob(f"*/{session_id}.jsonl"))
    if not matches:
        raise ValueError(f"No Claude Code transcript for session {session_id} under {home / 'projects'}.")
    notices = []
    if len(matches) > 1:
        notices.append(f"Multiple transcripts claim session {session_id}; selected the largest without summing copies.")
    return max(matches, key=lambda p: (p.stat().st_size, p.stat().st_mtime_ns, str(p))), notices


def claude_session_id(path: Path) -> str:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if isinstance(entry, dict) and isinstance(entry.get("sessionId"), str):
                return entry["sessionId"]
    return path.stem


def claude_descendants(home: Path, root: Path, session_id: str) -> list[Path]:
    """Subagent transcripts live in <project>/<session-id>/subagents/*.jsonl."""
    found = set(root.parent.joinpath(session_id).rglob("*.jsonl"))
    found.update((home / "projects").glob(f"*/{session_id}/**/*.jsonl"))
    return sorted({p.resolve() for p in found} - {root.resolve()})


def analyze_claude(path: Path, seen_tree: set[str], split_sidechains: bool):
    """Return {row_key: (by_model, last_timestamp, metered, observed)} plus warnings."""
    rows: dict[str, dict] = {}
    pending: dict[str, tuple[str, str, Usage, Any]] = {}
    order: list[str] = []
    warnings: list[str] = []
    copied = 0
    before = path.stat().st_size
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                entry = json.loads(line)
                if not isinstance(entry, dict):
                    raise ValueError("record is not an object")
            except ValueError as exc:
                warnings.append(f"Line {line_number}: unreadable record ({exc}).")
                continue
            if entry.get("type") != "assistant" or not isinstance(entry.get("message"), dict):
                continue
            message = entry["message"]
            model = message.get("model") or "unknown"
            if model in CLAUDE_SKIPPED_MODELS:
                continue
            key = f"sidechain:{entry.get('agentId') or 'unknown'}" if split_sidechains and entry.get("isSidechain") else ""
            row = rows.setdefault(key, {"models": {}, "timestamp": None, "observed": set()})
            row["observed"].add(model)
            try:
                value = claude_usage(message.get("usage"))
            except (ValueError, TypeError) as exc:
                warnings.append(f"Line {line_number}: invalid usage ({exc}).")
                continue
            response = message.get("id") or entry.get("requestId")
            if not response:
                warnings.append(f"Line {line_number}: message ID missing; duplicate detection unavailable.")
                response = f"{path}:{line_number}"
            if response in seen_tree:
                copied += 1
                continue
            if response in pending:
                previous = pending[response]
                if previous[2].input != value.input:
                    warnings.append(f"Line {line_number}: conflicting duplicate message usage.")
                # Streaming may persist partial blocks first; keep the most complete output count.
                if value.output < previous[2].output:
                    continue
            else:
                order.append(response)
            pending[response] = (key, model, value, entry.get("timestamp"))
    if before != path.stat().st_size:
        warnings.append("Transcript changed while being read; rerun for a later snapshot.")
    for response in order:
        key, model, value, timestamp = pending[response]
        seen_tree.add(response)
        row = rows[key]
        add(row["models"], model, value)
        row["timestamp"] = max(filter(None, (row["timestamp"], timestamp)), default=None)
    if copied:
        warnings.append(f"Skipped {copied} copied message records already counted elsewhere in this tree.")
    return rows, warnings


def build_claude_report(home: Path, session_id: str | None, explicit: Path | None = None) -> dict:
    snapshot_time = datetime.now(timezone.utc).isoformat()
    notices: list[str] = []
    if explicit:
        root = explicit.expanduser().resolve()
        identity = claude_session_id(root)
        if session_id and session_id != identity:
            raise ValueError("--session-id disagrees with --session-file")
        session_id = identity
    elif not session_id:
        raise ValueError("No current session ID; supply --session-id or --session-file (never guess the newest log).")
    else:
        root, notices = claude_root_transcript(home, session_id)
    seen_tree: set[str] = set()
    sessions, all_models = [], {}

    def append(thread, parent, path, row, warnings):
        metered = bool(row and row["models"])
        models = row["models"] if row else {}
        for model, value in models.items():
            add(all_models, model, value)
            if value.cached is None:
                warnings.append(f"Cache-hit counts unavailable for model {model}; cache split is unknown.")
        if not metered:
            warnings.append("No recorded token usage; consumption is unknown, not zero.")
        warnings = list(dict.fromkeys(warnings))
        sessions.append({
            "thread_id": thread, "parent_thread_id": parent, "rollout_path": str(path),
            "status": "partial" if warnings and metered else "recorded" if metered else "unmetered",
            "totals": total(models.values()).as_dict() if metered else None,
            "observed_models": sorted(row["observed"]) if row else [],
            "models": {k: v.as_dict() for k, v in sorted(models.items())},
            "last_usage_at": row["timestamp"] if row else None, "warnings": warnings,
        })

    def ignorable(warnings):
        return [w for w in warnings if not w.startswith("Skipped ")]

    rows, warnings = analyze_claude(root, seen_tree, split_sidechains=True)
    append(session_id, None, root, rows.pop("", None), ignorable(warnings))
    for key in sorted(rows):  # legacy in-file sidechains
        append(key, session_id, root, rows[key], [])
    for path in claude_descendants(home, root, session_id):
        try:
            child_rows, child_warnings = analyze_claude(path, seen_tree, split_sidechains=False)
        except (OSError, UnicodeError) as exc:
            child_rows, child_warnings = {}, [f"Could not read transcript: {exc}"]
        notes = [w for w in child_warnings if w.startswith("Skipped ")]
        notices.extend(f"{path.stem}: {note}" for note in notes)
        append(path.stem, session_id, path, child_rows.get(""), ignorable(child_warnings))
    timestamps = [s["last_usage_at"] for s in sessions if s["last_usage_at"]]
    return {
        "schema_version": 1,
        "agent": "claude-code",
        "root_thread_id": session_id,
        "snapshot_at": snapshot_time,
        "last_usage_at": max(timestamps) if timestamps else None,
        "coverage": "partial" if any(s["status"] != "recorded" for s in sessions)
        or any(not n.split(": ", 1)[-1].startswith("Skipped ") for n in notices) else "recorded",
        "session_count": len(sessions),
        "descendant_count": len(sessions) - 1,
        "metered_session_count": sum(s["totals"] is not None for s in sessions),
        "totals": total(all_models.values()).as_dict() if all_models else None,
        "models": {k: v.as_dict() for k, v in sorted(all_models.items())},
        "sessions": sessions,
        "warnings": notices,
        "scope_note": "Snapshot of recorded usage; excludes unflushed/running requests, this report's final answer, "
                      "later work, and auxiliary requests Claude Code does not persist in transcripts. "
                      "Missing usage is not zero.",
    }


def detect_agent(requested: str, session_id: str | None, session_file: Path | None, claude_home: Path) -> str:
    if requested != "auto":
        return requested
    if session_file:
        try:
            with session_file.expanduser().open(encoding="utf-8") as stream:
                first = json.loads(stream.readline())
            return "codex" if isinstance(first, dict) and first.get("type") == "session_meta" else "claude-code"
        except (OSError, ValueError):
            return "codex"
    if session_id:
        return "claude-code" if any((claude_home / "projects").glob(f"*/{session_id}.jsonl")) else "codex"
    if os.environ.get("CODEX_THREAD_ID") or os.environ.get("CODEX_SESSION_ID"):
        return "codex"
    if os.environ.get("CLAUDE_CODE_SESSION_ID"):
        return "claude-code"
    return "codex"


def markdown(report: dict) -> str:
    def number(value):
        return "unknown" if value is None else f"{value:,}"

    def escape(value):
        return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")

    def cells(value):
        return " | ".join(number(value.get(key)) for key in (
            "cache_hit_tokens", "cache_miss_tokens", "input_tokens", "output_tokens", "total_tokens"
        ))

    header = "| Scope / model | Input: cache hit | Input: cache miss | Input total | Output | Total |"
    separator = "|---|---:|---:|---:|---:|---:|"
    lines = [
        "# Session token usage", "",
        f"Agent: {report.get('agent', 'codex')}. Root thread: `{report['root_thread_id']}`. "
        f"Snapshot: {report['snapshot_at']}.",
        f"Latest recorded usage: {report['last_usage_at'] or 'unavailable'}.", "",
        f"Coverage: **{report['coverage']}** — main thread + {report['descendant_count']} descendants; "
        f"{report['metered_session_count']}/{report['session_count']} sessions have usage records.", "",
        "Upward = input; downward = output. Input includes cache hits.", "", header, separator,
        f"| All recorded usage | {cells(report['totals'] or {})} |",
    ]
    for model, value in report["models"].items():
        lines.append(f"| {escape(model)} | {cells(value)} |")
    rate = (report["totals"] or {}).get("cache_hit_rate")
    lines.extend(["", f"Input cache-hit rate: {rate:.2%}." if rate is not None else "Input cache-hit rate: unavailable.",
                  "", "## Session coverage", "", "| Thread | Models observed | Status | Input | Output |", "|---|---|---|---:|---:|"])
    for session in report["sessions"]:
        value = session["totals"] or {}
        observed = ", ".join(session["observed_models"]) or "unknown"
        lines.append(f"| {escape(session['thread_id'])} | {escape(observed)} | {session['status']} | "
                     f"{number(value.get('input_tokens'))} | {number(value.get('output_tokens'))} |")
    lines.extend(["", report["scope_note"]])
    warnings = list(report["warnings"])
    for session in report["sessions"]:
        warnings.extend(f"{session['thread_id']}: {warning}" for warning in session["warnings"])
    if warnings:
        lines.extend(["", "Warnings:", ""] + [f"- {escape(warning)}" for warning in warnings])
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=("auto", "codex", "claude-code"), default="auto",
                        help="Log format (default: detect from --session-file or the environment)")
    parser.add_argument("--session-id", help="Root thread/session ID (default: CODEX_THREAD_ID, CODEX_SESSION_ID, "
                                             "or CLAUDE_CODE_SESSION_ID)")
    parser.add_argument("--session-file", type=Path, help="Explicit root rollout/transcript; overrides environment identity")
    parser.add_argument("--codex-home", type=Path, default=Path(os.environ.get("CODEX_HOME", "~/.codex")).expanduser())
    parser.add_argument("--claude-home", type=Path,
                        default=Path(os.environ.get("CLAUDE_CONFIG_DIR", "~/.claude")).expanduser())
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output-dir", type=Path, help="Save a unique Markdown/JSON pair here, respecting project artifact policy")
    args = parser.parse_args(argv)
    claude_home = args.claude_home.expanduser().resolve()
    agent = detect_agent(args.agent, args.session_id, args.session_file, claude_home)
    identity = args.session_id
    if not identity and not args.session_file:
        if agent == "codex":
            identity = os.environ.get("CODEX_THREAD_ID") or os.environ.get("CODEX_SESSION_ID")
        else:
            identity = os.environ.get("CLAUDE_CODE_SESSION_ID")
    try:
        if agent == "codex":
            report = build_report(args.codex_home.expanduser().resolve(), identity, args.session_file)
        else:
            report = build_claude_report(claude_home, identity, args.session_file)
        rendered = markdown(report)
        structured = json.dumps(report, indent=2) + "\n"
        if args.output_dir:
            directory = args.output_dir.expanduser().resolve()
            directory.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            stem = f"session-usage-{stamp}-{uuid4().hex[:8]}"
            for suffix, content in (("md", rendered), ("json", structured)):
                path = directory / f"{stem}.{suffix}"
                with path.open("x", encoding="utf-8") as output:
                    output.write(content)
                print(f"Saved {path}", file=sys.stderr)
        print(structured if args.format == "json" else rendered, end="")
        return 0
    except (OSError, ValueError, sqlite3.Error) as exc:
        print(f"session_usage: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
