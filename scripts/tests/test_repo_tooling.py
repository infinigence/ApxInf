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
        evidence = REPO / "devlocal" / "agent-tooling"
        evidence.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="run-", dir=evidence)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.log = self.root / "calls.jsonl"
        self.env = dict(os.environ, PATH=f"{self.bin}:{os.environ['PATH']}", TOOL_LOG=str(self.log))
        for name in ("APXINF_PLATFORM", "APXINF_CUDA_ARCH", "APXINF_NVCC_PARALLELISM",
                     "APXINF_SKIP_COMMIT_GUARD"):
            self.env.pop(name, None)

    def stub(self, name):
        script = self.bin / name
        script.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "with open(os.environ['TOOL_LOG'], 'a') as f:\n"
            "    f.write(json.dumps({'args': sys.argv, 'arch': os.getenv('APXINF_CUDA_ARCH'), "
            "'parallelism': os.getenv('APXINF_NVCC_PARALLELISM'), "
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
        result = self.run_command(["just", "test-gpu", "pi05", "--", "--ignored", "--test-threads=1"])
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = [call["args"][1:] for call in self.calls()]
        self.assertEqual(calls, [
            ["test", "--release", "-p", "apxinf-model", "--lib", "--features", "cuda", "pi05", "--", "--ignored", "--test-threads=1"],
            ["test", "--release", "-p", "apxinf-cuda-new", "--lib", "pi05", "--", "--ignored", "--test-threads=1"],
        ])

        self.log.unlink()
        self.env["TOOL_EXIT"] = "3"
        failed = self.run_command(["just", "test-gpu", "pi05"])
        self.assertNotEqual(failed.returncode, 0)
        self.assertEqual(len(self.calls()), 1, "stop after the first failed crate")

    def test_l3_preserves_arguments_and_arch_override(self):
        self.stub("cargo")
        self.env["APXINF_CUDA_ARCH"] = "sm_89"
        result = self.run_command(["just", "test-l3", "--lib", "a test name", "--", "--nocapture"])
        self.assertEqual(result.returncode, 0, result.stderr)
        call = self.calls()[0]
        self.assertEqual(call["args"][1:], ["test", "-p", "apxinf-cuda-new", "--lib", "a test name", "--", "--nocapture"])
        self.assertEqual(call["arch"], "sm_89")

    def test_platform_table_reaches_build_and_l3_commands(self):
        for tool in ("maturin", "pip", "cargo"):
            self.stub(tool)
        for platform, arch, parallelism in (("thor", "sm_110", "4"),
                                           ("thor-u", "sm_101", "4"),
                                           ("orin", "sm_87", "4"),
                                           ("rtx4090", "sm_89", "9"),
                                           ("gb10", "sm_120", "9")):
            for recipe in ("build-py", "test-l3"):
                with self.subTest(platform=platform, recipe=recipe):
                    self.log.unlink(missing_ok=True)
                    self.env.update(APXINF_PLATFORM=platform, APXINF_CUDA_ARCH="sm_80",
                                    APXINF_NVCC_PARALLELISM="9")
                    args = ["--lib", "test name", "--", "--nocapture"] if recipe == "test-l3" else []
                    result = self.run_command(["just", recipe, *args])
                    self.assertEqual(result.returncode, 0, result.stderr)
                    calls = self.calls()
                    self.assertEqual(len(calls), 1 if recipe == "test-l3" else 3)
                    for call in calls:
                        self.assertEqual(call["arch"], arch)
                        self.assertEqual(call["parallelism"], parallelism)
                    if args:
                        self.assertEqual(calls[0]["args"][-len(args):], args)

    def test_platform_unset_preserves_environment_and_defaults(self):
        for tool in ("maturin", "pip", "cargo"):
            self.stub(tool)
        for recipe, default_arch in (("build-py", None), ("test-l3", "sm_110")):
            for inherited in (False, True):
                with self.subTest(recipe=recipe, inherited=inherited):
                    self.log.unlink(missing_ok=True)
                    for key in ("APXINF_CUDA_ARCH", "APXINF_NVCC_PARALLELISM"):
                        self.env.pop(key, None)
                    if inherited:
                        self.env.update(APXINF_CUDA_ARCH="sm_89", APXINF_NVCC_PARALLELISM="2")
                    result = self.run_command(["just", recipe])
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(self.calls()[0]["arch"], "sm_89" if inherited else default_arch)
                    self.assertEqual(self.calls()[0]["parallelism"], "2" if inherited else None)

    def test_platform_errors_stop_before_build_or_install(self):
        for tool in ("maturin", "pip", "cargo"):
            self.stub(tool)
        for recipe in ("build-py", "test-l3"):
            with self.subTest(recipe=recipe):
                self.env["APXINF_PLATFORM"] = "unknown; exit 0"
                result = self.run_command(["just", recipe])
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Unknown APXINF_PLATFORM", result.stderr)
                self.assertEqual(self.calls(), [])
                self.env.update(APXINF_PLATFORM="thor", TOOL_EXIT="7")
                result = self.run_command(["just", recipe])
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(len(self.calls()), 1, "failed build must not install a wheel")
                self.log.unlink()
                self.env.pop("TOOL_EXIT")

    def test_malformed_platform_row_fails_before_export(self):
        (self.root / "scripts").mkdir()
        (self.root / "configs").mkdir()
        script = self.root / "scripts/load_platform.sh"
        shutil.copy(REPO / "scripts/load_platform.sh", script)
        self.env["APXINF_PLATFORM"] = "thor"
        for row in ("thor invalid 4", "thor sm_110 0", "thor sm_110 4 extra"):
            with self.subTest(row=row):
                (self.root / "configs/platforms.tsv").write_text(row + "\n")
                result = self.run_command(["bash", "-eu", "-c", 'source "$1"', "test", str(script)])
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Invalid build preset", result.stderr)

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

    def test_commit_guard_missing_just_warns_and_allows_commit(self):
        # Invoke real Python directly with an empty PATH: no mocked which/run.
        self.env["PATH"] = str(self.bin)
        event = json.dumps({"tool_name": "Bash", "tool_input": {"command": "git commit -m test"}})
        result = self.run_command([sys.executable, str(REPO / "scripts/hooks/guard_git_commit.py")], input=event)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("just is not on PATH; checks not run", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_commit_guard_explicit_bypass_is_separate_from_git_no_verify(self):
        self.stub("just")
        self.env["TOOL_EXIT"] = "1"
        event = json.dumps({"tool_name": "Bash", "tool_input": {"command": "git commit --no-verify -m test"}})
        command = [sys.executable, str(REPO / "scripts/hooks/guard_git_commit.py")]
        for value in ("0", "true", "1"):
            with self.subTest(value=value):
                self.log.unlink(missing_ok=True)
                self.env["APXINF_SKIP_COMMIT_GUARD"] = value
                result = self.run_command(command, input=event)
                self.assertEqual(result.returncode, 0 if value == "1" else 2, result.stderr)
                self.assertEqual(len(self.calls()), 0 if value == "1" else 1)
                if value == "1":
                    self.assertIn("checks not run", result.stderr)
                self.assertEqual(result.stdout, "")

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
