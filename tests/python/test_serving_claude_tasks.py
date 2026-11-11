from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "serving"))
import claude_code_tasks as client
from metal_matrix import group_members


VALID_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 480">'
    '<title>Pelican Riding a Bicycle</title>'
    '<ellipse cx="240" cy="220" rx="100" ry="60" fill="white"/>'
    '<path d="M 280 220 L 420 240 L 300 260 Z" fill="orange"/>'
    '</svg>'
)


def event(kind, blocks=None, **fields):
    value = {"type": kind, **fields}
    if blocks is not None:
        value["message"] = {"content": blocks}
    return {"elapsed_s": 0.1, "event": value}


def tool_call(tool_id, name, path=None):
    path = path or ("drawing-brief.txt" if name == "Read" else "pelican.svg")
    return event("assistant", [{"type": "tool_use", "id": tool_id, "name": name,
                                "input": {"file_path": path}}])


def tool_result(tool_id, failed=False):
    return event("user", [{"type": "tool_result", "tool_use_id": tool_id,
                           "is_error": failed, "content": "Tool completed."}])


def capture(events=None, subtype="success"):
    return {"events": (events or []) + [event("result", result="Done.", subtype=subtype,
                                               is_error=False, usage={"input_tokens": 123})],
            "exit_code": 0}


def successful_events():
    return [tool_call("read-1", "Read"), tool_result("read-1"),
            tool_call("write-1", "Write"), tool_result("write-1")]


class ClientCaptureTests(unittest.TestCase):
    def capture_owned_probe(self, body, timeout=.2):
        producer = '''import os, pathlib, subprocess, sys, time
pathlib.Path('leader.pid').write_text(str(os.getpid()))
''' + body
        with tempfile.TemporaryDirectory() as temporary:
            workspace = client.TaskWorkspace.create(Path(temporary) / "attempt")
            directory = workspace.task
            directory.mkdir()
            try:
                result = client.capture_client([sys.executable, "-c", producer], workspace,
                                               dict(os.environ), timeout, False)
                pgid = int((directory / "leader.pid").read_text())
                live = [member for member in group_members(pgid)
                        if not member["state"].startswith("Z")]
                return result, live
            finally:
                pid_file = directory / "leader.pid"
                if pid_file.exists():
                    try:
                        os.killpg(int(pid_file.read_text()), signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_exited_leader_does_not_disable_inherited_pipe_deadline(self):
        result, live = self.capture_owned_probe('''
subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(3)'])
print('{"type":"result","subtype":"success","is_error":false,"result":"Done."}', flush=True)
''')
        self.assertTrue(result["timed_out"], result)
        self.assertEqual(result["exit_code"], 0)
        self.assertLess(result["elapsed_s"], 1.5)
        self.assertEqual(live, [])

    def test_closed_pipes_do_not_complete_a_live_leader(self):
        result, live = self.capture_owned_probe('''
os.close(1)
os.close(2)
time.sleep(3)
''')
        self.assertTrue(result["timed_out"], result)
        self.assertLess(result["elapsed_s"], 1.5)
        self.assertEqual(live, [])

    def test_descendant_ignoring_interrupt_does_not_escape_cleanup(self):
        result, live = self.capture_owned_probe('''
child = "import pathlib, signal, time; signal.signal(signal.SIGINT, signal.SIG_IGN); pathlib.Path('child-ready').touch(); time.sleep(.6)"
subprocess.Popen([sys.executable, '-c', child])
while not pathlib.Path('child-ready').exists():
    time.sleep(.01)
''')
        self.assertTrue(result["timed_out"], result)
        self.assertLess(result["elapsed_s"], 1.5)
        self.assertEqual(live, [])

    def test_normal_parent_remains_successful(self):
        result, live = self.capture_owned_probe('''
print('{"type":"result","subtype":"success","is_error":false,"result":"Done."}', flush=True)
''', timeout=2)
        self.assertEqual(result["exit_code"], 0)
        self.assertFalse(result["timed_out"])
        self.assertEqual(live, [])

    def test_escaped_pipe_holder_fails_bounded_drain_without_being_signaled(self):
        producer = '''import pathlib, subprocess, sys
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(3)'], start_new_session=True)
pathlib.Path('escaped.pid').write_text(str(child.pid))
'''
        with tempfile.TemporaryDirectory() as temporary:
            workspace = client.TaskWorkspace.create(Path(temporary) / "attempt")
            directory = workspace.task
            directory.mkdir()
            started = time.monotonic()
            try:
                with self.assertRaisesRegex(RuntimeError, "pipes did not close"):
                    client.capture_client([sys.executable, "-c", producer], workspace,
                                          dict(os.environ), .2, False)
                self.assertLess(time.monotonic() - started, 1.5)
                escaped = int((directory / "escaped.pid").read_text())
                self.assertTrue(any(not member["state"].startswith("Z")
                                    for member in group_members(escaped)))
            finally:
                pid_file = directory / "escaped.pid"
                if pid_file.exists():
                    try:
                        os.killpg(int(pid_file.read_text()), signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_cleanup_failure_cannot_return_a_successful_capture(self):
        with patch.object(client, "stop_child", side_effect=RuntimeError("Unconfirmed cleanup.")):
            with self.assertRaisesRegex(RuntimeError, "Unconfirmed cleanup"):
                self.capture_owned_probe("print('{}', flush=True)", timeout=2)

    def test_permission_race_recovers_only_a_confirmed_exited_group(self):
        for alive in (False, True):
            with self.subTest(alive=alive):
                process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3)" if alive else "pass"],
                                           start_new_session=True)
                try:
                    if not alive:
                        process.wait(timeout=2)
                    with patch.object(client, "stop_owned_group", side_effect=PermissionError("Signal denied.")):
                        if alive:
                            with self.assertRaises(PermissionError):
                                client.stop_child(process)
                        else:
                            client.stop_child(process)
                            self.assertEqual(process.returncode, 0)
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=2)

    def test_missing_requested_tool_stops_before_the_deadline(self):
        producer = '''import time
print('{"type":"system","subtype":"init","tools":["Read","Edit"]}', flush=True)
time.sleep(15)
'''
        with tempfile.TemporaryDirectory() as temporary:
            workspace = client.TaskWorkspace.create(Path(temporary) / "attempt")
            workspace.task.mkdir()
            result = client.capture_client(
                [sys.executable, "-c", producer, "--tools", "Read,Edit,Write"],
                workspace, dict(os.environ), 2, False)
        self.assertEqual(result.get("tool_inventory", {}).get("missing", []), ["Write"])
        self.assertFalse(result["timed_out"])
        self.assertEqual(result["tool_inventory"]["status"], "mismatch")
        self.assertEqual(result["events"][0]["event"]["tools"], ["Read", "Edit"])

    def test_available_named_tools_preserve_normal_capture(self):
        producer = '''print('{"type":"system","subtype":"init","tools":["Edit","Read"]}', flush=True)
print('{"type":"result","subtype":"success","is_error":false,"result":"Done."}', flush=True)
'''
        with tempfile.TemporaryDirectory() as temporary:
            workspace = client.TaskWorkspace.create(Path(temporary) / "attempt")
            workspace.task.mkdir()
            result = client.capture_client(
                [sys.executable, "-c", producer, "--tools", "Read,Edit"],
                workspace, dict(os.environ), 2, False)
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["tool_inventory"], {"requested": ["Edit", "Read"],
                         "advertised": ["Edit", "Read"], "missing": [], "status": "verified"})

    def test_malformed_initial_tool_list_stops_the_client(self):
        producer = '''import time
print('{"type":"system","subtype":"init","tools":null}', flush=True)
time.sleep(15)
'''
        with tempfile.TemporaryDirectory() as temporary:
            workspace = client.TaskWorkspace.create(Path(temporary) / "attempt")
            workspace.task.mkdir()
            result = client.capture_client(
                [sys.executable, "-c", producer, "--tools", "Read"],
                workspace, dict(os.environ), 2, False)
        self.assertEqual(result["tool_inventory"]["status"], "invalid")
        self.assertIsNone(result["tool_inventory"]["advertised"])
        self.assertFalse(result["timed_out"])

    def test_late_inventory_cannot_repair_output_before_initialization(self):
        producer = '''print('{"type":"result","subtype":"success","is_error":false,"result":"Done."}', flush=True)
print('{"type":"system","subtype":"init","tools":["Read"]}', flush=True)
'''
        with tempfile.TemporaryDirectory() as temporary:
            workspace = client.TaskWorkspace.create(Path(temporary) / "attempt")
            workspace.task.mkdir()
            result = client.capture_client(
                [sys.executable, "-c", producer, "--tools", "Read"],
                workspace, dict(os.environ), 2, False)
        self.assertEqual(result["tool_inventory"]["status"], "invalid")
        self.assertIn("before advertising", " ".join(result["diagnostics"]))

    def test_output_is_visible_before_the_client_exits(self):
        producer = '''import pathlib, sys, time
print('{"type":"system","subtype":"capture_probe"}', flush=True)
print('capture stderr probe', file=sys.stderr, flush=True)
pathlib.Path('producer-ready').touch()
while not pathlib.Path('release-producer').exists():
    time.sleep(0.01)
'''
        with tempfile.TemporaryDirectory() as temporary, ThreadPoolExecutor(max_workers=1) as executor:
            attempts = (Path(temporary) / "tasks").resolve()
            workspace = client.TaskWorkspace.create(attempts / "attempt")
            directory = workspace.task
            directory.mkdir()
            future = executor.submit(client.capture_client, [sys.executable, "-c", producer],
                                     workspace, dict(os.environ), 15, False)
            try:
                deadline = time.monotonic() + 5
                stdout = workspace.stdout
                stderr = workspace.stderr
                while time.monotonic() < deadline:
                    if ((directory / "producer-ready").exists() and stdout.read_bytes() and stderr.read_bytes()):
                        break
                    time.sleep(.01)
                self.assertFalse(future.done())
                self.assertEqual(list(attempts.glob("*/client.process.json")), [workspace.process])
                identity = json.loads(workspace.process.read_text())
                self.assertGreater(identity["pid"], 0)
                self.assertEqual(identity["pgid"], identity["pid"])
                self.assertEqual(os.getpgid(identity["pid"]), identity["pgid"])
                self.assertIn(b'"capture_probe"', stdout.read_bytes())
                self.assertEqual(stderr.read_text(), "capture stderr probe\n")
                for path in (stdout, stderr, workspace.process, workspace.config):
                    self.assertFalse(path.is_relative_to(directory), str(path))
                self.assertEqual(set(path.name for path in directory.iterdir()), {"producer-ready"})
            finally:
                (directory / "release-producer").touch()
                result = future.result(timeout=5)
            self.assertEqual(result["exit_code"], 0)
            self.assertEqual(result["pid"], identity["pid"])
            self.assertEqual(result["pgid"], identity["pgid"])
            self.assertEqual(result["events"][0]["event"]["subtype"], "capture_probe")
            self.assertEqual(result["environment_revision"], "task-workspace-v2")
            self.assertEqual(result["attempt_directory"], str(workspace.root))


class TaskWorkspaceTests(unittest.TestCase):
    def test_attempt_reservation_preserves_evidence_and_separates_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            first = client.TaskWorkspace.create(directory / "first")
            self.assertTrue(first.root.is_absolute())
            self.assertFalse(first.task.exists())
            first.stdout.write_bytes(b"original capture\n")
            with self.assertRaises(FileExistsError):
                client.TaskWorkspace.create(first.root)
            self.assertEqual(first.stdout.read_bytes(), b"original capture\n")
            second = client.TaskWorkspace.create(directory / "second")
            self.assertNotEqual(first.config, second.config)
            self.assertTrue(first.config.is_dir())
            self.assertTrue(second.config.is_dir())

    def test_run_exposes_only_case_files_in_client_working_directory(self):
        producer = '''import json, os, pathlib, sys
print(json.dumps({"type": "system", "subtype": "init", "tools": ["Read", "Edit", "Bash"]}), flush=True)
print("workspace probe stderr", file=sys.stderr, flush=True)
observation = {"cwd": str(pathlib.Path.cwd()), "entries": sorted(os.listdir(".")),
               "config_dir": os.environ["CLAUDE_CONFIG_DIR"]}
print(json.dumps({"type": "result", "subtype": "success", "is_error": False,
                  "result": json.dumps(observation)}), flush=True)
'''
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            executable = directory / "workspace-probe"
            executable.write_text(f"#!{sys.executable}\n" + producer)
            executable.chmod(0o700)
            output = directory / "run"
            args = client.parser().parse_args([
                "--claude", str(executable), "--output-dir", str(output),
                "--tasks", "expenses-tools-cny", "--timeout", "5"])
            with patch.object(client, "request_json", side_effect=[
                    (200, {"data": [{"id": "workspace-probe-model"}]}),
                    (200, {"worker": {"capabilities": {"max_context": 4096}}})]), \
                    patch.object(client, "metrics_snapshot", return_value={"values": {}}):
                self.assertEqual(client.run(args), 1)
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["environment_revision"], "task-workspace-v2")
            self.assertEqual(len(report["tasks"]), 1)
            record = report["tasks"][0]
            self.assertEqual(record["environment_revision"], "task-workspace-v2")
            self.assertEqual(record["exit_code"], 0)
            self.assertFalse(record["timed_out"])
            self.assertFalse(record["passed"])
            self.assertTrue(record["inputs_unchanged"])
            self.assertEqual(record["tool_inventory"]["status"], "verified")
            observation = json.loads(record["result"]["result"])
            workspace = Path(observation["cwd"])
            self.assertEqual(workspace, Path(record["directory"]))
            attempt = Path(record["attempt_directory"])
            self.assertEqual(workspace, attempt / "task")
            self.assertTrue(attempt.is_absolute())
            config = Path(observation["config_dir"])
            self.assertTrue(config.is_dir())
            for field in ("stdout_path", "stderr_path", "process_path"):
                self.assertTrue(Path(record[field]).is_file(), record[field])
                self.assertEqual(Path(record[field]).parent, attempt)
            self.assertEqual(config, attempt / "client-config")
            self.assertEqual(Path(record["stderr_path"]).read_text(), "workspace probe stderr\n")
            identity = json.loads(Path(record["process_path"]).read_text())
            self.assertEqual(identity, {"pid": record["pid"], "pgid": record["pgid"]})
            self.assertEqual(set(observation["entries"]), {
                "transactions.csv", "expense-rules.txt", "calculator-usage.txt",
                "expense_calculator.py", "calculation-plan.json", "expenses.json"})
            for path in (config, *(Path(record[field]) for field in (
                    "stdout_path", "stderr_path", "process_path"))):
                self.assertFalse(path.is_relative_to(workspace), str(path))


class PelicanTaskTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / "task"
        self.spec = client.prepare_task("pelican", self.directory)

    def outcome(self, events=None, subtype="success"):
        return client.task_outcome("pelican", self.spec, self.directory, capture(events, subtype))

    def save_svg(self, text=VALID_SVG):
        (self.directory / "pelican.svg").write_text(text)

    def test_task_creates_an_original_brief_and_narrow_tool_set(self):
        brief = (self.directory / "drawing-brief.txt").read_text()
        self.assertIn("pelican", brief)
        self.assertIn("riding a bicycle", brief)
        self.assertIn("feet on the pedals", brief)
        self.assertIn('viewBox="0 0 640 480"', brief)
        self.assertEqual(self.spec["tools"], ["Read", "Edit"])
        self.assertEqual(self.spec["allowed"], ["Read", "Edit"])
        self.assertEqual((self.directory / "pelican.svg").read_text(), "")
        self.assertIn(str(self.directory / "pelican.svg"), self.spec["prompt"])

    def test_valid_svg_and_matched_read_write_cycle_pass_structurally(self):
        self.save_svg()
        result = self.outcome(successful_events())
        self.assertTrue(result["automated_passed"], result)
        self.assertIsNone(result["passed"])
        self.assertEqual(result["visual_review"], "pending")
        self.assertEqual(result["observed_input_usage"], {"input_tokens": 123})

    def test_edit_result_also_completes_the_save_cycle(self):
        self.save_svg()
        events = [tool_call("read-1", "Read"), tool_result("read-1"),
                  tool_call("edit-1", "Edit"), tool_result("edit-1")]
        self.assertTrue(self.outcome(events)["automated_passed"])

    def test_inventory_failure_rejects_an_otherwise_successful_capture(self):
        self.save_svg()
        for status in ("unreported", "mismatch", "invalid", "unknown"):
            with self.subTest(status=status):
                recorded = capture(successful_events())
                recorded["tool_inventory"] = {"status": status}
                result = client.task_outcome("pelican", self.spec, self.directory, recorded)
                self.assertFalse(result["automated_passed"])
                self.assertFalse(result["passed"])

    def test_timeout_or_cancellation_rejects_a_successful_result(self):
        self.save_svg()
        for flag in ("timed_out", "cancel_sent"):
            with self.subTest(flag=flag):
                recorded = capture(successful_events())
                recorded[flag] = True
                result = client.task_outcome("pelican", self.spec, self.directory, recorded)
                self.assertFalse(result["automated_passed"])
                self.assertFalse(result["passed"])

    def test_standard_xml_declaration_does_not_reject_svg(self):
        self.save_svg('<?xml version="1.0" encoding="UTF-8"?>' + VALID_SVG)
        self.assertTrue(self.outcome(successful_events())["automated_passed"])

    def test_a_write_before_the_brief_read_does_not_count(self):
        self.save_svg()
        events = [tool_call("write-1", "Write"), tool_result("write-1"),
                  tool_call("read-1", "Read"), tool_result("read-1")]
        self.assertFalse(self.outcome(events)["passed"])

    def test_wrong_tool_result_id_does_not_complete_write(self):
        self.save_svg()
        for events in (
            successful_events()[:-1] + [tool_result("another-id")],
            [tool_call("read-1", "Read"), tool_result("another-id"),
             tool_call("write-1", "Write"), tool_result("write-1")],
        ):
            with self.subTest(events=events):
                self.assertFalse(self.outcome(events)["passed"])

    def test_wrong_tool_paths_do_not_complete_the_cycle(self):
        self.save_svg()
        for events in (
            [tool_call("read-1", "Read", "another-brief.txt"), tool_result("read-1"),
             tool_call("write-1", "Write"), tool_result("write-1")],
            [tool_call("read-1", "Read"), tool_result("read-1"),
             tool_call("write-1", "Write", "another.svg"), tool_result("write-1")],
        ):
            with self.subTest(events=events):
                self.assertFalse(self.outcome(events)["passed"])

    def test_failed_tool_result_does_not_complete_read_or_write(self):
        self.save_svg()
        for events in (
            [tool_call("read-1", "Read"), tool_result("read-1", True),
             tool_call("write-1", "Write"), tool_result("write-1")],
            successful_events()[:-1] + [tool_result("write-1", True)],
        ):
            with self.subTest(events=events):
                self.assertFalse(self.outcome(events)["passed"])

    def test_empty_and_malformed_artifacts_fail(self):
        self.assertFalse(self.outcome(successful_events())["passed"])
        for source in ("", "<svg", "not xml"):
            with self.subTest(source=source):
                self.save_svg(source)
                result = self.outcome(successful_events())
                self.assertFalse(result["passed"])
                self.assertFalse(result["svg_check"]["valid"])

    def test_false_success_without_a_tool_cycle_or_valid_artifact_fails(self):
        self.save_svg()
        self.assertFalse(self.outcome([])["passed"])
        self.assertFalse(self.outcome(successful_events(), "error")["passed"])
        self.save_svg("<svg/>")
        self.assertFalse(self.outcome(successful_events())["passed"])

    def test_active_content_and_external_references_fail(self):
        cases = (
            VALID_SVG.replace("</svg>", '<script>alert(1)</script></svg>'),
            VALID_SVG.replace("</svg>", '<image href="https://example.com/x.png"/></svg>'),
            VALID_SVG.replace('fill="white"', 'fill="url(https://example.com/fill.svg)"'),
            '<!DOCTYPE svg [<!ENTITY x "unsafe">]>' + VALID_SVG,
            '<?xml-stylesheet href="https://example.com/x.css"?>' + VALID_SVG,
        )
        for source in cases:
            with self.subTest(source=source):
                self.save_svg(source)
                self.assertFalse(self.outcome(successful_events())["passed"])

    def test_title_and_viewbox_are_required(self):
        for source in (VALID_SVG.replace("Pelican Riding a Bicycle", "Other title"),
                       VALID_SVG.replace("0 0 640 480", "0 0 600 480")):
            with self.subTest(source=source):
                self.save_svg(source)
                self.assertFalse(self.outcome(successful_events())["passed"])


if __name__ == "__main__":
    unittest.main()
