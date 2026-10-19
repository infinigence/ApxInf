"""Exercise real receipts, source changes and Stop events in isolated Git trees."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/agent_acceptance.py"
HOOK = REPO / "scripts/hooks/guard_agent_stop.py"


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        parent = REPO / "devlocal/agent-acceptance/tests"
        parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.git("init", "-q")
        self.git("config", "user.email", "test@example.invalid")
        self.git("config", "user.name", "Test")
        (self.root / ".gitignore").write_text("devlocal/\n")
        (self.root / "model.py").write_text("original\n")
        self.git("add", ".")
        self.git("commit", "-qm", "fixture")
        self.plan = self.root / "devlocal/task/acceptance.json"
        self.plan.parent.mkdir(parents=True)
        self.harness = self.plan.parent / "check.py"
        self.harness.write_text("print('full tensor assertions passed')\n")
        self.document = {"version": 1, "cases": [{
            "id": "public-policy", "criterion": "full output identity",
            "entry_point": "Policy.infer", "command": [sys.executable, str(self.harness)],
            "inputs": [str(self.harness.relative_to(self.root))],
        }]}
        self.save()

    def git(self, *args):
        return subprocess.run(["git", *args], cwd=self.root, check=True,
                              capture_output=True, text=True)

    def save(self):
        self.plan.write_text(json.dumps(self.document))

    def cli(self, *args):
        return subprocess.run([sys.executable, str(SCRIPT), args[0], str(self.plan), *args[1:]],
                              cwd=self.root, capture_output=True, text=True)

    def hook(self, active=False, enabled=True):
        env = dict(os.environ)
        env.pop("APXINF_AGENT_ACCEPTANCE_PLAN", None)
        if enabled:
            env["APXINF_AGENT_ACCEPTANCE_PLAN"] = str(self.plan)
        return subprocess.run([sys.executable, str(HOOK)], env=env,
                              input=json.dumps({"stop_hook_active": active}),
                              capture_output=True, text=True)

    def test_success_and_commit_only_change(self):
        self.assertEqual(self.cli("run", "public-policy").returncode, 0)
        self.assertEqual(self.cli("check").returncode, 0)
        self.git("commit", "--allow-empty", "-qm", "metadata only")
        self.assertEqual(self.cli("check").returncode, 0)
        self.assertEqual(self.hook().stdout, "")

    def test_failure_is_not_a_receipt_of_success(self):
        self.harness.write_text("raise SystemExit(7)\n")
        self.assertEqual(self.cli("run", "public-policy").returncode, 7)
        self.assertEqual(self.cli("check").returncode, 1)

    def test_interrupted_rerun_cannot_reuse_previous_success(self):
        self.harness.write_text("import time\nfrom pathlib import Path\n"
                                "if Path('devlocal/pause').exists(): time.sleep(1)\n")
        self.assertEqual(self.cli("run", "public-policy").returncode, 0)
        (self.root / "devlocal/pause").touch()
        process = subprocess.Popen([sys.executable, str(SCRIPT), "run", str(self.plan),
                                    "public-policy"], stdout=subprocess.DEVNULL)
        receipt = self.plan.parent / "acceptance-evidence/public-policy.json"
        try:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                try:
                    if json.loads(receipt.read_text())["status"] == "running":
                        break
                except ValueError:
                    pass
                time.sleep(0.01)
            else:
                self.fail("rerun did not invalidate the previous receipt")
            process.terminate()
            process.wait(timeout=5)
            self.assertEqual(self.cli("check").returncode, 1)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()

    def test_source_harness_plan_and_logs_invalidate_evidence(self):
        changes = [
            lambda: (self.root / "model.py").write_text("changed\n"),
            lambda: (self.root / "new.py").write_text("new\n"),
            lambda: self.harness.write_text("print('changed harness')\n"),
            lambda: self.plan.write_text(self.plan.read_text() + "\n"),
            lambda: next((self.plan.parent / "acceptance-evidence").glob("*.log")).write_text("tampered"),
        ]
        for change in changes:
            with self.subTest(change=change):
                self.assertEqual(self.cli("run", "public-policy").returncode, 0)
                change()
                self.assertEqual(self.cli("check").returncode, 1)
                # Remove old run logs so the next log-mutation targets the latest run.
                for log in (self.plan.parent / "acceptance-evidence").glob("*.log"):
                    log.unlink()

    def test_mutating_source_during_command_fails(self):
        self.harness.write_text("from pathlib import Path\nPath('model.py').write_text('mutated')\n")
        self.assertEqual(self.cli("run", "public-policy").returncode, 1)
        self.assertEqual(self.cli("check").returncode, 1)

    def test_blocked_and_bounded_stop_never_pass(self):
        self.assertEqual(self.hook(enabled=False).stdout, "")
        self.assertEqual(json.loads(self.hook().stdout)["decision"], "block")
        self.assertEqual(self.hook(active=True).stdout, "")
        self.assertIn("INCOMPLETE", self.hook(active=True).stderr)
        self.assertEqual(self.cli("block", "public-policy", "--reason", "board unavailable").returncode, 2)
        self.assertEqual(self.cli("check").returncode, 2)
        self.assertEqual(self.hook().stdout, "")
        self.assertIn("board unavailable", self.hook().stderr)

    def test_adding_case_invalidates_old_receipts_and_requires_new_case(self):
        self.cli("run", "public-policy")
        self.document["cases"].append(dict(self.document["cases"][0], id="other-family"))
        self.save()
        result = self.cli("check")
        self.assertEqual(result.returncode, 1)
        self.assertIn("other-family", result.stdout)


if __name__ == "__main__":
    unittest.main()
