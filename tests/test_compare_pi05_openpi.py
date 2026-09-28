from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

import numpy as np

from scripts import compare_pi05_openpi as parity


class FakePolicy:
    def __init__(self, offset: float = 0.0):
        self.offset = offset

    def infer(self, observation, *, noise):
        value = float(noise[0, 0]) + len(observation["prompt"]) / 100
        return {"actions": np.full((10, 7), value + self.offset, dtype=np.float32)}


class Pi05OpenPiParityTest(unittest.TestCase):
    def setUp(self):
        scratch = pathlib.Path(__file__).resolve().parents[1] / "devlocal/pi05-openpi-parity/tests"
        scratch.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        args = parity.parse_args(["prepare", "--suite-dir", str(self.root), "--interface", "policy"])
        parity.prepare(args)

    def test_bounded_cases_are_deterministic_and_cover_edges(self):
        first = parity.load_suite(self.root)
        self.assertEqual(len(first["cases"]), 7)
        self.assertFalse(first["representative"])
        self.assertEqual(
            [case["name"] for case in first["cases"]],
            ["typical", "second-noise", "long-language", "dark-zero-noise",
             "bright-negative-noise", "float-chw-resize", "view-order-contrast"],
        )
        # The default suite has seven distinct, fixed inputs.
        self.assertEqual(len({case["sha256"] for case in first["cases"]}), len(first["cases"]))
        parity.prepare(parity.parse_args(["prepare", "--suite-dir", str(self.root), "--interface", "policy", "--force"]))
        self.assertEqual(first["cases"], parity.load_suite(self.root)["cases"])

    def test_collect_and_compare_reject_a_bad_case(self):
        manifest = parity.load_suite(self.root)
        meta = {"checkpoint_sha256": "same", "revision": "test"}
        parity.collect(self.root, manifest, FakePolicy(), "openpi", {**meta, "config_name": "pi05_libero"}, force=False)
        parity.collect(self.root, manifest, FakePolicy(), "apxinf", {**meta, "precision": "fp8"}, force=False)
        args = parity.parse_args(["compare", "--suite-dir", str(self.root)])
        self.assertEqual(parity.compare(args), 0)

        result_path = self.root / "apxinf.json"
        result = json.loads(result_path.read_text())
        result["cases"][2]["actions"] = np.negative(result["cases"][2]["actions"]).tolist()
        parity.write_json(result_path, result, force=True)
        args.force = True
        self.assertEqual(parity.compare(args), 1)
        report = json.loads((self.root / "report.json").read_text())
        self.assertFalse(report["passed"])
        self.assertFalse(report["cases"][2]["passed"])

    def test_changed_input_is_rejected_before_inference(self):
        manifest = parity.load_suite(self.root)
        path = self.root / manifest["cases"][0]["path"]
        with path.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "case input changed"):
            parity.load_suite(self.root)

    def test_compare_rejects_different_checkpoint_bytes(self):
        manifest = parity.load_suite(self.root)
        parity.collect(
            self.root, manifest, FakePolicy(), "openpi",
            {"checkpoint_sha256": "reference", "revision": "test", "config_name": "pi05_libero"},
            force=False,
        )
        parity.collect(
            self.root, manifest, FakePolicy(), "apxinf",
            {"checkpoint_sha256": "other", "precision": "fp8"},
            force=False,
        )
        with self.assertRaisesRegex(ValueError, "identical model.safetensors"):
            parity.compare(parity.parse_args(["compare", "--suite-dir", str(self.root)]))

    def test_zero_vector_cosine_is_not_a_false_pass(self):
        reference = np.zeros((10, 7), dtype=np.float32)
        actual = np.ones((10, 7), dtype=np.float32)
        self.assertEqual(parity.metrics(reference, actual)["cosine"], 0.0)
        self.assertEqual(
            parity.metrics(reference, actual)["relative_l2"], np.finfo(np.float64).max
        )

    def test_two_sources_replace_repeat_noise_and_support_three_views(self):
        keys = ["observation/image", "observation/wrist_image", "observation/right_image"]
        sources = []
        for index in range(2):
            path = self.root / f"source-{index}.npz"
            np.savez(path, **{key: np.full((224, 224, 3), index + view, np.uint8)
                              for view, key in enumerate(keys)},
                     **{"observation/state": np.arange(8, dtype=np.float32)})
            sources.append(path)
        other = self.root / "three-view"
        args = parity.parse_args([
            "prepare", "--suite-dir", str(other), "--image-keys", ",".join(keys),
            "--source-npz", str(sources[0]), "--source-npz", str(sources[1]),
            "--discrete-state", "--interface", "policy",
        ])
        parity.prepare(args)
        manifest = parity.load_suite(other)
        self.assertTrue(manifest["representative"])
        self.assertTrue(manifest["discrete_state"])
        self.assertEqual(len(manifest["cases"]), 7)
        self.assertEqual(manifest["cases"][1]["name"], "second-scene")

    def test_default_bare_suite_has_full_action_width_and_pinned_tokens(self):
        other = self.root / "bare"
        parity.prepare(parity.parse_args(["prepare", "--suite-dir", str(other)]))
        manifest = parity.load_suite(other)
        self.assertEqual(manifest["interface"], "bare")
        self.assertEqual(manifest["action_dim"], 32)
        observation, noise = parity.load_case(other, manifest["cases"][0])
        self.assertEqual(tuple(observation["token_ids"]), parity.SHORT_TOKENS)
        self.assertEqual(noise.shape, (10, 32))


if __name__ == "__main__":
    unittest.main()
