import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration_data import select_samples
import calibrate_gr00t_fp8 as collector
from calibrate_gr00t_fp8 import fixture_identity, merge_amax, parse_args


class CalibrationTests(unittest.TestCase):
    def test_collector_passes_float32_to_legacy_native_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tactics = root / "tactics.json"
            tactics.write_text("{}")
            bits = np.array([0x3F80, 0xBF80, 0x8000, 0x0001], dtype=np.uint16)
            expected = (bits.astype(np.uint32) << 16).view(np.float32)
            for source in (bits, expected):
                with self.subTest(dtype=source.dtype):
                    output = root / f"profile-{source.dtype}.json"
                    args = parse_args(["--output", str(output), "--tactics", str(tactics), str(root)])
                    arrays = {name: source.copy() for name in ("pixel_values", "state", "noise")}
                    arrays.update(image_grid_thw=np.array([[1, 1, 2]], dtype=np.uint32),
                                  token_ids=np.array([1], dtype=np.uint32),
                                  attention_mask=np.ones(1, dtype=np.uint8))
                    calls = []

                    def calibrate(*inputs):
                        for index in (0, 4, 6):
                            self.assertEqual(inputs[index].dtype, np.float32)
                            self.assertTrue(inputs[index].flags.c_contiguous)
                            np.testing.assert_array_equal(inputs[index].view(np.uint32), expected.view(np.uint32))
                        calls.append(inputs)
                        return {"test.input": 1.0}

                    model = SimpleNamespace(_calibration_plan=lambda: ["test.input"],
                                            _calibrate_preprocessed=calibrate)
                    extension = SimpleNamespace(ModelRunner=SimpleNamespace(load=lambda *args, **kwargs: model))
                    with patch.object(collector, "parse_args", return_value=args), \
                         patch.object(collector.Gr00tPolicy, "checkpoint_identity", return_value="test-checkpoint"), \
                         patch.object(collector, "fixture_identity", return_value="test-fixture"), \
                         patch.object(collector, "load_fixture", return_value=({"embodiment_id": 2}, arrays)), \
                         patch.object(collector, "build_calibration_document", return_value={"scales": {}, "calibration_data": {}}), \
                         patch.dict(sys.modules, {"apxinf_py": extension}):
                        collector.main()
                    self.assertEqual(len(calls), 1)

    def test_task_balanced_distinct_and_reproducible(self):
        tasks = [{"task_index": index, "task": f"task-{index}"} for index in range(3)]
        episodes = [{"episode_index": task * 10 + episode,
                     "tasks": [f"task-{task}"], "length": 101}
                    for task in range(3) for episode in range(5)]
        samples = select_samples(episodes, tasks, 3, 3, 9000)
        self.assertEqual(samples, select_samples(episodes[::-1], tasks[::-1], 3, 3, 9000))
        self.assertEqual(len(samples), 27)
        self.assertEqual(len({(sample["episode_index"], sample["frame_index"]) for sample in samples}), 27)
        self.assertEqual({sample["frame_index"] for sample in samples}, {10, 50, 90})
        for task in tasks:
            self.assertEqual(sum(sample["task_index"] == task["task_index"] for sample in samples), 9)
        with self.assertRaises(ValueError):
            select_samples(episodes, tasks, 6, 3, 9000)

    def test_each_sample_requires_complete_finite_amax(self):
        records = {}
        merge_amax(records, {"left": 2, "right": 3}, ["left", "right"])
        merge_amax(records, {"left": 4, "right": 1}, ["left", "right"])
        self.assertEqual(records, {"left": 4, "right": 3})
        for invalid in ({"left": 8}, {"left": np.nan, "right": 1}, {"left": -1, "right": 1}):
            with self.assertRaises(ValueError):
                merge_amax(records, invalid, ["left", "right"])
        self.assertEqual(records, {"left": 4, "right": 3})

    def test_no_implicit_official_fixture_and_exclusive_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            output = str(Path(directory) / "profile.json")
            with self.assertRaises(SystemExit):
                parse_args(["--output", output])
            with self.assertRaises(SystemExit):
                parse_args(["--output", output, "--dataset", "data", "official.fixture"])
            args = parse_args(["--output", output, "--dataset", "data"])
            self.assertEqual(args.fixtures, [])
            self.assertEqual(args.episodes_per_task * args.frames_per_episode, 9)

    def test_fixture_identity_includes_tensor_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory)
            (fixture / "manifest.json").write_text(json.dumps({"tensors": {"state": {"file": "state.bin"}}}))
            (fixture / "state.bin").write_bytes(b"first")
            before = fixture_identity([fixture])
            (fixture / "state.bin").write_bytes(b"other")
            self.assertNotEqual(before, fixture_identity([fixture]))


if __name__ == "__main__":
    unittest.main()
