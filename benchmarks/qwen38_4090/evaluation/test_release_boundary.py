import hashlib
import json
import subprocess
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
MANIFEST_PATH = HERE / "RELEASE_MANIFEST.public.json"


class PublicReleaseBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_tracked_evaluator_files_match_manifest(self) -> None:
        expected = set(self.manifest["public_evaluator_files"])
        expected.add(MANIFEST_PATH.name)
        output = subprocess.run(
            [
                "git",
                "-C",
                str(REPOSITORY),
                "ls-files",
                "--",
                str(HERE.relative_to(REPOSITORY)),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        actual = {str(Path(path).relative_to(HERE.relative_to(REPOSITORY))) for path in output}
        self.assertEqual(actual, expected)

    def test_public_file_hashes_match_manifest(self) -> None:
        for relative, expected in self.manifest["public_evaluator_files"].items():
            with self.subTest(path=relative):
                actual = hashlib.sha256((HERE / relative).read_bytes()).hexdigest()
                self.assertEqual(actual, expected)

    def test_release_state_has_frozen_provenance(self) -> None:
        self.assertIn(self.manifest["status"], {"candidate", "released"})
        starter = self.manifest["starter"]
        if self.manifest["status"] == "candidate":
            self.assertIsNone(starter["release_revision"])
            revision = starter["base_revision"]
            expected_tree = starter["base_tree"]
        else:
            self.assertEqual(self.manifest["release_blockers"], [])
            self.assertRegex(self.manifest["teacher_release_manifest_sha256"], r"^[0-9a-f]{64}$")
            revision = starter["revision"]
            expected_tree = starter["tree"]
        actual_tree = subprocess.run(
            ["git", "-C", str(REPOSITORY), "rev-parse", f"{revision}^{{tree}}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        self.assertEqual(actual_tree, expected_tree)
        hidden = self.manifest["datasets"]["multimodal_hidden"]
        self.assertRegex(hidden["manifest_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(hidden["status"], "frozen-private")


if __name__ == "__main__":
    unittest.main()
