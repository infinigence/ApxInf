from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import compare_pi05_openpi as parity


class FakeModel:
    def infer(self, case):
        value = float(case["noise"][0, 0]) + len(case["token_ids"]) / 100
        return np.full((10, 32), value, dtype=np.float32)


class Pi05OpenPiParityTest(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / "devlocal/pi05-openpi-parity/tests"
        scratch.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        parity.prepare(parity.parse_args(["prepare", "--suite-dir", str(self.root)]))

    def test_seven_cases_are_deterministic(self):
        first = parity.load_suite(self.root)
        self.assertEqual([case["name"] for case in first["cases"]], [
            "typical", "second-noise", "long-language", "dark-zero-noise",
            "bright-negative-noise", "float-chw", "view-order-contrast",
        ])
        parity.prepare(parity.parse_args(["prepare", "--suite-dir", str(self.root), "--force"]))
        self.assertEqual(first["cases"], parity.load_suite(self.root)["cases"])
        case = parity.load_case(self.root, first["cases"][0])
        self.assertEqual(case["noise"].shape, (10, 32))
        self.assertEqual(case["images"].shape, (2, 224, 224, 3))

    def test_three_view_source_uses_second_scene(self):
        keys = ["base", "left", "right"]
        sources = []
        for scene in range(2):
            path = self.root / f"scene-{scene}.npz"
            np.savez(path, **{key: np.full((224, 224, 3), scene + view, np.uint8)
                              for view, key in enumerate(keys)})
            sources.append(path)
        other = self.root / "three-view"
        parity.prepare(parity.parse_args([
            "prepare", "--suite-dir", str(other), "--image-keys", ",".join(keys),
            "--source-npz", str(sources[0]), "--source-npz", str(sources[1]),
        ]))
        manifest = parity.load_suite(other)
        self.assertTrue(manifest["representative"])
        self.assertEqual(manifest["cases"][1]["name"], "second-scene")
        self.assertEqual(parity.load_case(other, manifest["cases"][0])["images"].shape,
                         (3, 224, 224, 3))

    def test_compare_fails_a_changed_output(self):
        manifest = parity.load_suite(self.root)
        parity.collect(self.root, manifest, FakeModel(), "openpi",
                       {"checkpoint_sha256": "same", "revision": "test"}, force=False)
        parity.collect(self.root, manifest, FakeModel(), "apxinf",
                       {"checkpoint_sha256": "same", "precision": "bf16"}, force=False)
        args = parity.parse_args(["compare", "--suite-dir", str(self.root)])
        self.assertEqual(parity.compare(args), 0)
        result_path = self.root / "apxinf.json"
        result = json.loads(result_path.read_text())
        result["cases"][2]["actions"] = np.negative(result["cases"][2]["actions"]).tolist()
        parity.write_json(result_path, result, force=True)
        args.force = True
        self.assertEqual(parity.compare(args), 1)

    def test_rejects_changed_input_and_checkpoint(self):
        manifest = parity.load_suite(self.root)
        parity.collect(self.root, manifest, FakeModel(), "openpi",
                       {"checkpoint_sha256": "one", "revision": "test"}, force=False)
        parity.collect(self.root, manifest, FakeModel(), "apxinf",
                       {"checkpoint_sha256": "two", "precision": "bf16"}, force=False)
        with self.assertRaisesRegex(ValueError, "different model.safetensors"):
            parity.compare(parity.parse_args(["compare", "--suite-dir", str(self.root)]))
        with (self.root / manifest["cases"][0]["path"]).open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "case input changed"):
            parity.load_suite(self.root)

    def test_zero_reference_does_not_false_pass(self):
        result = parity.metrics(np.zeros((10, 32)), np.ones((10, 32)))
        self.assertEqual(result["cosine"], 0.0)
        self.assertEqual(result["relative_l2"], np.finfo(np.float64).max)


if __name__ == "__main__":
    unittest.main()
