from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "python/apxinf/apxinf/serving"
sys.path.insert(0, str(SOURCE))
import clef_identity as identity
import clef_contracts as c
from clef_runtime import ClefRuntime


class ClefIdentityTests(unittest.TestCase):
    def artifacts(self, root):
        documents = {"config.json": {"text_config": {"vocab_size": 128}, "quantization": {"bits": 4}},
                     "tokenizer.json": {}, "tokenizer_config.json": {}, "joint_head_config.json": {},
                     "model.safetensors.index.json": {"weight_map": {"test.weight": "model-1.safetensors"}}}
        for name, value in documents.items():
            (root / name).write_text(json.dumps(value))
        for name in ("model-1.safetensors", "joint_head.safetensors", "chat_template.jinja"):
            (root / name).write_bytes(b"original local identity fixture")

    def test_contracts_and_runtime_import_without_optional_dependencies(self):
        result = subprocess.run([sys.executable, "-S", "-c",
            "import clef_contracts, clef_identity, clef_runtime, clef_worker, sys; "
            "assert not any(k == 'mlx' or k.startswith('mlx.') or k == 'PIL' for k in sys.modules)"],
            cwd=SOURCE, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_artifact_set_is_sorted_complete_and_excludes_unselected_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.artifacts(root)
            (root / "README.md").write_text("unselected")
            (root / "clef_mlx.py").write_text("unselected")
            (root / "__pycache__").mkdir()
            (root / "optional.json").symlink_to(root / "config.json")
            artifacts, vocab = identity.model_artifacts(root)
            self.assertEqual(vocab, 128)
            names = [entry["path"] for entry in artifacts]
            self.assertEqual(names, sorted(names))
            self.assertNotIn("README.md", names)
            self.assertNotIn("optional.json", names)
            self.assertTrue(all(entry["sha256"] == identity.file_digest(root / entry["path"]) for entry in artifacts))

    def test_required_symlink_is_not_a_checked_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.artifacts(root)
            (root / "joint_head.safetensors").unlink()
            (root / "joint_head.safetensors").symlink_to(root / "model-1.safetensors")
            with self.assertRaises(identity.ClefError):
                identity.model_artifacts(root)

    def test_index_escape_and_missing_shard_rejected(self):
        for name in ("../model-1.safetensors", "missing.safetensors", "/outside.safetensors"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.artifacts(root)
                (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": name}}))
                with self.assertRaises(identity.ClefError):
                    identity.model_artifacts(root)

    def test_nested_remote_code_mapping_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.artifacts(root)
            (root / "tokenizer_config.json").write_text(json.dumps({"nested": {"auto_map": {}}}))
            with self.assertRaises(identity.ClefError):
                identity.model_artifacts(root)

    def test_interpreter_record_retains_selected_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "python-real"
            executable.write_bytes(b"original interpreter identity fixture")
            selected = root / "python"
            selected.symlink_to(executable)
            record = identity.file_record(selected, preserve_path=True)
            self.assertEqual(record["path"], str(selected))
            self.assertEqual(record["sha256"], identity.file_digest(executable))

    def test_wrong_dependency_hash_never_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            loader = Path(directory) / "loader.py"
            loader.write_text("raise AssertionError('must not import')")
            with self.assertRaises(identity.ClefError):
                identity.load_dependency(loader)

    def test_wrong_runtime_never_imports_device_modules(self):
        with patch.object(identity.platform, "system", return_value="Darwin"), \
             patch.object(identity.platform, "python_version", return_value="3.12.10"), \
             patch.object(identity.importlib.metadata, "version", return_value="0"), \
             patch.object(identity.importlib, "import_module") as imported:
            with self.assertRaises(identity.ClefError):
                identity.check_runtime()
            imported.assert_not_called()

    def test_changed_checkpoint_never_imports_device_modules(self):
        with patch.object(identity, "model_artifacts", return_value=([], 128)), \
             patch.object(identity, "check_runtime") as imported:
            with self.assertRaises(identity.ClefError):
                ClefRuntime("/selected/model", "/selected/loader.py", 4096, 9000000000)
            imported.assert_not_called()


if __name__ == "__main__":
    unittest.main()
