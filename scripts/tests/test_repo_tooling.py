"""Exercise repository entry points with captured subprocesses, not a GPU/API."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import yaml


REPO = Path(__file__).resolve().parents[2]


class UniqueLoader(yaml.BaseLoader):
    def construct_mapping(self, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in result:
                raise ValueError(f"Duplicate YAML key: {key}")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


class ToolingTests(unittest.TestCase):
    def setUp(self):
        (REPO / "devlocal").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="tooling-test-", dir=REPO / "devlocal")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.log = self.root / "calls.jsonl"
        self.env = dict(os.environ, PATH=f"{self.bin}:{os.environ['PATH']}", TOOL_LOG=str(self.log))
        self.env.pop("APXINF_PLATFORM", None)

    def stub(self, name):
        script = self.bin / name
        script.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "with open(os.environ['TOOL_LOG'], 'a') as f:\n"
            "    f.write(json.dumps({'args': sys.argv, 'arch': os.getenv('APXINF_CUDA_ARCH'), "
            "'repo': os.getenv('GH_REPO'), 'cwd': os.getcwd()}) + '\\n')\n"
            "sys.exit(int(os.getenv('TOOL_EXIT', '0')))\n"
        )
        script.chmod(0o755)

    def run_command(self, command, **kwargs):
        return subprocess.run(command, cwd=REPO, env=self.env, text=True,
                              capture_output=True, **kwargs)

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def test_gpu_recipe_uses_real_features_and_library_targets(self):
        self.stub("cargo")
        result = self.run_command(["just", "test-gpu", "pi05"])
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = [call["args"][1:] for call in self.calls()]
        self.assertEqual(calls, [
            ["test", "--release", "-p", "apxinf-model", "--lib", "--features", "cuda", "pi05"],
            ["test", "--release", "-p", "apxinf-cuda-new", "--lib", "pi05"],
        ])

    def test_l3_preserves_arguments_and_arch_override(self):
        self.stub("cargo")
        self.env["APXINF_CUDA_ARCH"] = "sm_89"
        result = self.run_command(["just", "test-l3", "--lib", "a test name", "--", "--nocapture"])
        self.assertEqual(result.returncode, 0, result.stderr)
        call = self.calls()[0]
        self.assertEqual(call["args"][1:], ["test", "-p", "apxinf-cuda-new", "--lib", "a test name", "--", "--nocapture"])
        self.assertEqual(call["arch"], "sm_89")

    def test_legacy_l3_shim_consumes_cargo_subcommand(self):
        self.stub("cargo")
        result = self.run_command(["bash", "crates/apxinf-cuda-new/test-new.sh", "test", "--lib", "--", "--test-threads=1"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls()[0]["args"][1:], ["test", "-p", "apxinf-cuda-new", "--lib", "--", "--test-threads=1"])

    def test_qwen_benchmark_routes_to_rust_and_preserves_checkpoint_path(self):
        self.stub("cargo")
        result = self.run_command(["just", "bench", "qwen38", "/models/checkpoint with spaces", "--repeats", "1"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls()[0]["args"][1:], ["run", "--release", "-p", "apxinf-model", "--features", "cuda", "--example", "qwen38_bench", "--", "/models/checkpoint with spaces", "--repeats", "1"])

    def test_python_benchmark_routes_only_to_existing_scripts(self):
        self.stub("python3")
        for family in ("pi05", "pi0_fast", "smolvla", "gr00t"):
            with self.subTest(family=family):
                result = self.run_command(["just", "bench", family, "--help"])
                self.assertEqual(result.returncode, 0, result.stderr)
                target = self.calls()[-1]["args"][1]
                self.assertTrue((REPO / target).is_file(), target)
        result = self.run_command(["just", "bench", "nonexistent"])
        self.assertNotEqual(result.returncode, 0)

    def test_claude_hook_skips_reads_and_blocks_failed_commit_checks(self):
        self.stub("just")
        settings = json.loads((REPO / ".claude/settings.json").read_text())
        command = settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
        self.env["CLAUDE_PROJECT_DIR"] = str(REPO)
        for shell_command, exit_code in (("git status", 0), ("echo 'git commit'", 0),
                                          ("git commit -m update", 2),
                                          ("git -C . commit -m update", 2),
                                          ("git status && git commit -m update", 2)):
            with self.subTest(command=shell_command):
                self.env["TOOL_EXIT"] = "1"
                event = json.dumps({"tool_name": "Bash", "tool_input": {"command": shell_command}})
                result = self.run_command(["bash", "-c", command], input=event)
                self.assertEqual(result.returncode, exit_code, result.stderr)
        self.assertEqual(len(self.calls()), 3)
        self.env["TOOL_EXIT"] = "0"
        result = self.run_command(["bash", "-c", command], input=event)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_workflow_yaml_has_unique_keys_and_missing_key_skips_ai(self):
        workflows = {}
        for path in (REPO / ".github/workflows").glob("*.yml"):
            workflows[path.name] = yaml.load(path.read_text(), Loader=UniqueLoader)
        steps = workflows["ai-review.yml"]["jobs"]["review"]["steps"]
        config = steps[0]
        self.assertEqual(config["env"]["GH_REPO"], "${{ github.repository }}")
        self.stub("gh")
        output = self.root / "outputs"
        self.env.update(AI_KEY="", PR="128", GH_REPO="example/repo", GITHUB_OUTPUT=str(output))
        result = self.run_command(["bash", "-eu", "-c", config["run"]])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(output.read_text(), "enabled=false\n")
        self.assertEqual(self.calls()[0]["args"][1:3], ["pr", "comment"])
        self.assertEqual(self.calls()[0]["repo"], "example/repo")
        for step in steps[1:]:
            self.assertEqual(step["if"], "steps.configuration.outputs.enabled == 'true'")
        output.unlink()
        self.log.unlink()
        self.env["AI_KEY"] = "test-placeholder"
        result = self.run_command(["bash", "-eu", "-c", config["run"]])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(output.read_text().endswith("enabled=true\n"))
        self.assertEqual(self.calls(), [])

    def test_doc_lint_checks_skills_and_same_file_anchors(self):
        fixture = self.root / "repo"
        (fixture / "scripts").mkdir(parents=True)
        (fixture / "doc/reference").mkdir(parents=True)
        (fixture / ".agents/skills/example").mkdir(parents=True)
        shutil.copy(REPO / "scripts/check_docs.py", fixture / "scripts/check_docs.py")
        (fixture / "justfile").write_text("default:\n    @echo ok\n")
        (fixture / "doc/reference/env-vars.md").write_text("# Environment\n")
        skill = fixture / ".agents/skills/example/SKILL.md"
        skill.write_text("# Entry\n[Doc](../../../doc/reference/env-vars.md#environment)\n[Self](#entry)\n")
        subprocess.run(["git", "init", "-q", str(fixture)], check=True)
        def lint():
            return subprocess.run([sys.executable, str(fixture / "scripts/check_docs.py")], text=True, capture_output=True)
        self.assertEqual(lint().returncode, 0)
        for target in ("../../doc/reference/env-vars.md", "#missing", "../../../doc/reference/env-vars.md#absent"):
            with self.subTest(target=target):
                skill.write_text(f"# Entry\n[Broken]({target})\n")
                result = lint()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(".agents/skills/example/SKILL.md", result.stdout)
        skill.write_text("# Entry\n## Repeat\n## Repeat\n[Second](#repeat-1)\n")
        self.assertEqual(lint().returncode, 0)


if __name__ == "__main__":
    unittest.main()
