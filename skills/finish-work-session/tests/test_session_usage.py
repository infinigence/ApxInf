"""Accounting regression tests; all fixtures stay in the checkout's devlocal/."""

import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest


SKILL = Path(__file__).resolve().parents[1]
SCRIPT = SKILL / "scripts" / "session_usage.py"
spec = importlib.util.spec_from_file_location("session_usage", SCRIPT)
accounting = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = accounting
spec.loader.exec_module(accounting)


def event(kind, payload, timestamp="2026-11-16T08:00:00Z"):
    return {"type": kind, "timestamp": timestamp, "payload": payload}


def counters(input_tokens, cached, output, **extra):
    value = {"input_tokens": input_tokens, "output_tokens": output,
             "total_tokens": input_tokens + output, **extra}
    if cached is not None:
        value["cached_input_tokens"] = cached
    return value


def context(model="model-a", turn="turn-a"):
    return event("turn_context", {"model": model, "turn_id": turn})


def record(value, thread="root", turn="turn-a", response="response-1", cumulative=None, **extra):
    payload = {"thread_id": thread, "turn_id": turn, "response_id": response, "usage": value, **extra}
    if cumulative is not None:
        payload["thread_token_usage"] = cumulative
    return event("token_usage_record", payload)


def snapshot(value, last=None):
    return event("event_msg", {"type": "token_count", "info": {
        "total_token_usage": value, "last_token_usage": last if last is not None else value,
    }})


class SessionUsageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        results = SKILL.parents[1] / "devlocal" / "finish-work-session" / "results"
        results.mkdir(parents=True, exist_ok=True)
        cls.run_dir = Path(tempfile.mkdtemp(prefix="tests-", dir=results))

    def setUp(self):
        self.home = self.run_dir / self._testMethodName
        self.home.mkdir()

    def rollout(self, thread="root", events=(), parent=None, archived=False, name=None, **metadata):
        meta = {"id": thread, "source": "cli", **metadata}
        if parent:
            meta["parent_thread_id"] = parent
        directory = self.home / ("archived_sessions" if archived else "sessions")
        directory.mkdir(exist_ok=True)
        path = directory / (name or f"{thread}.jsonl")
        path.write_text("".join(json.dumps(e) + "\n" for e in [event("session_meta", meta), *events]))
        return path

    def report(self, thread="root"):
        return accounting.build_report(self.home, thread)

    def test_recursive_tree_models_and_archived_children(self):
        self.rollout(events=[context(), record(counters(100, 60, 10))])
        self.rollout("child", [context("model-b", "child-turn"),
                     record(counters(200, 50, 20), "child", "child-turn", session_id="root")], parent="root")
        self.rollout("grandchild", [context("model-a", "grand-turn"),
                     record(counters(40, 0, 5), "grandchild", "grand-turn")], parent="child", archived=True)
        self.rollout("unrelated", [context(), record(counters(9999, 0, 0), "unrelated")])
        result = self.report()
        self.assertEqual(result["descendant_count"], 2)
        self.assertEqual(result["totals"]["input_tokens"], 340)
        self.assertEqual(result["totals"]["cache_hit_tokens"], 110)
        self.assertEqual(result["totals"]["cache_miss_tokens"], 230)
        self.assertEqual(result["totals"]["output_tokens"], 35)
        self.assertEqual(result["models"]["model-a"]["total_tokens"], 155)
        self.assertEqual(result["models"]["model-b"]["total_tokens"], 220)
        self.assertEqual(result["coverage"], "recorded")

    def test_ledger_and_cumulative_events_are_not_added_twice(self):
        first, second = counters(100, 30, 10), counters(50, 40, 5)
        both = counters(150, 70, 15)
        self.rollout(events=[context(), record(first, cumulative=first), snapshot(first), snapshot(first),
                     record(second, response="response-2", cumulative=both), snapshot(both, second),
                     record(second, response="response-2", cumulative=both)])
        result = self.report()
        self.assertEqual(result["totals"]["total_tokens"], 165)
        self.assertEqual(result["coverage"], "recorded")

    def test_model_changes_and_explicit_record_model(self):
        one = counters(10, 2, 1)
        self.rollout(events=[context(), record(one), context("model-b", "turn-b"),
                     record(one, turn="turn-b", response="r2"), context("model-c", "turn-b"),
                     record(one, turn="turn-b", response="r3"),
                     record(one, turn="turn-b", response="r4", model="routed-model")])
        result = self.report()
        self.assertEqual(set(result["models"]), {"model-a", "model-b", "model-c", "routed-model"})
        self.assertTrue(all(v["total_tokens"] == 11 for v in result["models"].values()))

    def test_legacy_repeated_events_and_model_switch(self):
        first, second = counters(100, 60, 10), counters(80, 40, 5)
        self.rollout(events=[context(), snapshot(first), snapshot(first),
                     event("event_msg", {"type": "token_count", "info": None}),
                     context("model-b", "turn-b"), snapshot(counters(180, 100, 15), second)])
        result = self.report()
        self.assertEqual(result["totals"]["total_tokens"], 195)
        self.assertEqual(result["models"]["model-a"]["total_tokens"], 110)
        self.assertEqual(result["models"]["model-b"]["total_tokens"], 85)

    def test_legacy_unattributable_history_is_unknown(self):
        self.rollout(events=[context(), snapshot(counters(100, 60, 10), counters(20, 5, 2))])
        result = self.report()
        self.assertEqual(set(result["models"]), {"unknown"})
        self.assertEqual(result["totals"]["total_tokens"], 110)
        self.assertEqual(result["coverage"], "partial")

    def test_partial_ledger_is_reconciled_without_guessing_models(self):
        self.rollout(events=[context(), snapshot(counters(100, 60, 10)),
                     record(counters(50, 20, 5), cumulative=counters(150, 80, 15))])
        result = self.report()
        self.assertEqual(result["totals"]["total_tokens"], 165)
        self.assertEqual(result["models"]["model-a"]["total_tokens"], 55)
        self.assertEqual(result["models"]["unknown"]["total_tokens"], 110)

    def test_legacy_prefix_and_ledger_without_cumulative_field(self):
        self.rollout(events=[context(), snapshot(counters(100, 60, 10)),
                     record(counters(50, 20, 5))])
        result = self.report()
        self.assertEqual(result["totals"]["total_tokens"], 165)
        self.assertEqual(result["models"]["unknown"]["total_tokens"], 110)

    def test_lagging_event_does_not_override_new_response_checkpoint(self):
        self.rollout(events=[context(), snapshot(counters(100, 20, 10)),
                     record(counters(100, 20, 10), cumulative=counters(100, 20, 10)),
                     record(counters(20, 10, 5), response="r2", cumulative=counters(120, 30, 15))])
        result = self.report()
        self.assertEqual(result["totals"]["total_tokens"], 135)
        self.assertEqual(result["coverage"], "recorded")

    def test_copied_parent_history_is_excluded(self):
        first = counters(100, 60, 10)
        parent_events = [context(), record(first), snapshot(first)]
        self.rollout(events=parent_events)
        self.rollout("child", [event("session_meta", {"id": "root"}), *parent_events,
                     context("model-b", "child-turn"), record(counters(20, 10, 2), "child", "child-turn")],
                     parent="root", forked_from_id="root")
        result = self.report()
        self.assertEqual(result["totals"]["total_tokens"], 132)
        self.assertEqual(self.report("child")["totals"]["total_tokens"], 22)

    def test_legacy_child_inherited_counter_baseline(self):
        inherited = [context(), snapshot(counters(100, 60, 10))]
        self.rollout(events=inherited)
        self.rollout("child", [*inherited, context("model-b", "child-turn"),
                     snapshot(counters(120, 70, 12), counters(20, 10, 2))], parent="root")
        self.assertEqual(self.report("child")["totals"]["total_tokens"], 22)

    def test_user_fork_is_not_a_spawned_descendant(self):
        self.rollout(events=[context(), record(counters(100, 60, 10))])
        self.rollout("user-fork", [context(), record(counters(50, 10, 3), "user-fork")], forked_from_id="root")
        self.assertEqual(self.report()["descendant_count"], 0)

    def test_nested_spawn_metadata(self):
        self.rollout(events=[context(), record(counters(100, 60, 10))])
        self.rollout("child", [context(), record(counters(10, 0, 1), "child")],
                     source={"subagent": {"thread_spawn": {"parent_thread_id": "root"}}})
        self.assertEqual(self.report()["descendant_count"], 1)

    def test_missing_and_unmetered_children_are_visible(self):
        self.rollout(events=[context(), record(counters(100, 60, 10))])
        self.rollout("guardian", [context("review-model", "guardian-turn")], parent="root")
        with sqlite3.connect(self.home / "state_5.sqlite") as db:
            db.execute("CREATE TABLE thread_spawn_edges (parent_thread_id TEXT, child_thread_id TEXT)")
            db.execute("INSERT INTO thread_spawn_edges VALUES ('root', 'missing-child')")
        result = self.report()
        self.assertEqual(result["session_count"], 3)
        self.assertEqual(result["metered_session_count"], 1)
        self.assertEqual(result["coverage"], "partial")
        self.assertEqual({s["status"] for s in result["sessions"]}, {"recorded", "missing", "unmetered"})
        self.assertTrue(all(s["totals"] is None for s in result["sessions"][1:]))
        guardian = next(s for s in result["sessions"] if s["thread_id"] == "guardian")
        self.assertEqual(guardian["observed_models"], ["review-model"])

    def test_database_rollout_paths_are_read(self):
        path = self.rollout(events=[context(), record(counters(100, 60, 10))])
        moved = self.home / "external.jsonl"
        path.rename(moved)
        with sqlite3.connect(self.home / "state_5.sqlite") as db:
            db.execute("CREATE TABLE threads (id TEXT, rollout_path TEXT)")
            db.execute("INSERT INTO threads VALUES (?, ?)", ("root", str(moved)))
        self.assertEqual(self.report()["totals"]["total_tokens"], 110)

    def test_archive_duplicate_counts_one_thread(self):
        events = [context(), record(counters(100, 60, 10))]
        self.rollout(events=events)
        self.rollout(events=events, archived=True)
        result = self.report()
        self.assertEqual(result["totals"]["total_tokens"], 110)
        self.assertEqual(result["session_count"], 1)
        self.assertEqual(result["coverage"], "partial")

    def test_missing_cache_is_unknown_and_reasoning_is_not_added(self):
        self.rollout(events=[context(), record(counters(100, None, 10, reasoning_output_tokens=8,
                                                       cache_write_input_tokens=50))])
        result = self.report()
        self.assertIsNone(result["totals"]["cache_hit_tokens"])
        self.assertIsNone(result["totals"]["cache_miss_tokens"])
        self.assertEqual(result["totals"]["output_tokens"], 10)
        self.assertEqual(result["totals"]["total_tokens"], 110)

    def test_cache_writes_stay_inside_cache_misses(self):
        self.rollout(events=[context(), record(counters(100, 40, 10, cache_write_input_tokens=50))])
        self.assertEqual(self.report()["totals"]["cache_miss_tokens"], 60)

    def test_truncated_tail_keeps_valid_usage_and_warns(self):
        path = self.rollout(events=[context(), record(counters(100, 60, 10))])
        with path.open("a") as stream:
            stream.write('{"type":')
        result = self.report()
        self.assertEqual(result["totals"]["total_tokens"], 110)
        self.assertEqual(result["coverage"], "partial")

    def test_counter_regression_never_charges_a_replay(self):
        self.rollout(events=[context(), snapshot(counters(100, 60, 10)), snapshot(counters(20, 10, 2)),
                     snapshot(counters(100, 60, 10))])
        result = self.report()
        self.assertEqual(result["totals"]["total_tokens"], 110)
        self.assertEqual(result["coverage"], "partial")

    def test_unmetered_root_does_not_print_zero_usage(self):
        self.rollout(events=[context()])
        result = self.report()
        self.assertIsNone(result["totals"])
        self.assertIn("| All recorded usage | unknown |", accounting.markdown(result))

    def test_invalid_cache_count_is_not_silently_clamped(self):
        self.rollout(events=[context(), record(counters(10, 11, 1))])
        result = self.report()
        self.assertIsNone(result["totals"])
        self.assertEqual(result["coverage"], "partial")

    def test_conflicting_duplicate_warns(self):
        self.rollout(events=[context(), record(counters(100, 60, 10)), record(counters(100, 60, 12))])
        result = self.report()
        self.assertEqual(result["totals"]["total_tokens"], 110)
        self.assertEqual(result["coverage"], "partial")

    def test_unknown_model_and_unmatched_turn_are_preserved(self):
        self.rollout(events=[context(), record(counters(100, 60, 10), turn="unmatched")])
        self.assertEqual(set(self.report()["models"]), {"unknown"})

    def test_cli_identity_and_saved_outputs(self):
        path = self.rollout(events=[context(), record(counters(100, 60, 10))])
        output = self.home / "reports"
        environment = dict(os.environ, CODEX_THREAD_ID="wrong-thread", CODEX_SESSION_ID="also-wrong")
        result = subprocess.run([sys.executable, "-B", str(SCRIPT), "--session-file", str(path),
                                 "--codex-home", str(self.home), "--format", "json", "--output-dir", str(output)],
                                env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        parsed = json.loads(result.stdout)
        self.assertEqual(parsed["root_thread_id"], "root")
        self.assertEqual(len(list(output.glob("*.md"))), 1)
        self.assertEqual(json.loads(next(output.glob("*.json")).read_text()), parsed)
        mismatch = subprocess.run([sys.executable, "-B", str(SCRIPT), "--session-id", "other",
                                   "--session-file", str(path)], capture_output=True, text=True)
        self.assertEqual(mismatch.returncode, 2)

    def test_no_identity_does_not_guess_newest_log(self):
        self.rollout(events=[context(), record(counters(100, 60, 10))])
        with self.assertRaisesRegex(ValueError, "No current thread ID"):
            accounting.build_report(self.home, None)


def claude_message(message_id, model="claude-model-a", input_tokens=10, read=80, write=5, output=7,
                   sidechain=False, agent=None, session="claude-root", timestamp="2026-11-16T08:00:00Z"):
    usage = {"input_tokens": input_tokens, "output_tokens": output}
    if read is not None:
        usage["cache_read_input_tokens"] = read
    if write is not None:
        usage["cache_creation_input_tokens"] = write
    entry = {"type": "assistant", "sessionId": session, "isSidechain": sidechain, "timestamp": timestamp,
             "message": {"id": message_id, "model": model, "role": "assistant", "usage": usage,
                         "content": [{"type": "text", "text": "secret body"}]}}
    if agent:
        entry["agentId"] = agent
    return entry


class ClaudeCodeUsageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        results = SKILL.parents[1] / "devlocal" / "finish-work-session" / "results"
        results.mkdir(parents=True, exist_ok=True)
        cls.run_dir = Path(tempfile.mkdtemp(prefix="claude-tests-", dir=results))

    def setUp(self):
        self.home = self.run_dir / self._testMethodName
        self.project = self.home / "projects" / "-repo"
        self.project.mkdir(parents=True)

    def transcript(self, entries, session="claude-root", child=None):
        path = self.project / f"{session}.jsonl"
        if child:
            path = self.project / session / "subagents" / f"{child}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
        lines = [{"type": "user", "sessionId": session, "message": {"role": "user", "content": "hi"}}, *entries]
        path.write_text("".join(json.dumps(e) + "\n" for e in lines))
        return path

    def report(self, session="claude-root"):
        return accounting.build_claude_report(self.home, session)

    def test_cache_fields_map_to_input_and_content_blocks_dedupe(self):
        self.transcript([claude_message("m1"), claude_message("m1"), claude_message("m2", output=3),
                         claude_message("synthetic", model="<synthetic>", input_tokens=0, read=0, write=0, output=0)])
        result = self.report()
        self.assertEqual(result["agent"], "claude-code")
        self.assertEqual(result["totals"]["input_tokens"], 190)
        self.assertEqual(result["totals"]["cache_hit_tokens"], 160)
        self.assertEqual(result["totals"]["cache_miss_tokens"], 30)
        self.assertEqual(result["totals"]["output_tokens"], 10)
        self.assertEqual(set(result["models"]), {"claude-model-a"})
        self.assertEqual(result["coverage"], "recorded")
        self.assertNotIn("secret body", json.dumps(result))

    def test_streamed_partial_output_keeps_most_complete_count(self):
        self.transcript([claude_message("m1", output=1), claude_message("m1", output=40)])
        self.assertEqual(self.report()["totals"]["output_tokens"], 40)

    def test_subagents_models_and_copied_history(self):
        self.transcript([claude_message("m1")])
        self.transcript([claude_message("m1"), claude_message("s1", model="claude-model-b", sidechain=True)],
                        child="agent-abc")
        self.transcript([claude_message("other")], session="unrelated")
        result = self.report()
        self.assertEqual(result["descendant_count"], 1)
        self.assertEqual(result["models"]["claude-model-a"]["total_tokens"], 102)
        self.assertEqual(result["models"]["claude-model-b"]["total_tokens"], 102)
        self.assertEqual(result["coverage"], "recorded")

    def test_legacy_inline_sidechains_get_rows(self):
        self.transcript([claude_message("m1"), claude_message("s1", sidechain=True, agent="a1")])
        result = self.report()
        self.assertEqual([s["thread_id"] for s in result["sessions"]], ["claude-root", "sidechain:a1"])
        self.assertEqual(result["totals"]["total_tokens"], 204)

    def test_missing_cache_fields_are_unknown(self):
        self.transcript([claude_message("m1", read=None, write=None)])
        result = self.report()
        self.assertIsNone(result["totals"]["cache_hit_tokens"])
        self.assertEqual(result["coverage"], "partial")

    def test_unmetered_child_makes_coverage_partial(self):
        self.transcript([claude_message("m1")])
        self.transcript([], child="agent-empty")
        self.assertEqual(self.report()["coverage"], "partial")

    def test_cli_detects_claude_environment(self):
        self.transcript([claude_message("m1")])
        environment = {k: v for k, v in os.environ.items() if not k.startswith("CODEX_")}
        environment.update(CLAUDE_CODE_SESSION_ID="claude-root", CLAUDE_CONFIG_DIR=str(self.home))
        result = subprocess.run([sys.executable, "-B", str(SCRIPT), "--format", "json"],
                                env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["totals"]["total_tokens"], 102)

    def test_cli_session_file_sniffs_claude_format(self):
        path = self.transcript([claude_message("m1")])
        result = subprocess.run([sys.executable, "-B", str(SCRIPT), "--session-file", str(path),
                                 "--claude-home", str(self.home), "--format", "json"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["root_thread_id"], "claude-root")

    def test_unknown_session_fails_without_guessing(self):
        self.transcript([claude_message("m1")])
        with self.assertRaisesRegex(ValueError, "No Claude Code transcript"):
            self.report("missing")


if __name__ == "__main__":
    unittest.main()
