# 格式与脚本入口的 CPU 回归测试：调用 formats.py 验证 BF16 位解码、
# fixture/noise/参考文件读取及校验，并检查脚本复用、精度对比入口、
# 重复输出发布检查和参数提前拒绝；不另行实现格式解析逻辑。
# 使用人工数组、临时文件和 mock，不加载模型、不使用 GPU。

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bench_apxinf_musa_rust as rust
import bench_apxinf_musa_wheel as wheel
import bench_torch_eager_musa as torch_bench
import compare_outputs
import formats
import prompt_fixture


class FormatTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def fixture(self):
        arrays = {
            "pixel_values": np.full((4, 8), 0x3F80, dtype="<u2"),
            "image_grid_thw": np.array([[1, 2, 2]], dtype="<u4"),
            "token_ids": np.array([1, 2, 3], dtype="<u4"),
            "attention_mask": np.array([1, 1, 1], dtype="u1"),
            "state": np.full((1, 1, 132), 0xBF80, dtype="<u2"),
            "noise": np.full((1, 40, 132), 0x3F00, dtype="<u2"),
        }
        manifest = {"schema": formats.FIXTURE_SCHEMA, "fixture": "test", "embodiment_id": 2, "tensors": {}}
        for name, values in arrays.items():
            values.tofile(self.root / f"{name}.bin")
            manifest["tensors"][name] = {
                "file": f"{name}.bin", "dtype": formats.TENSOR_DTYPES[name], "shape": list(values.shape),
            }
        (self.root / "manifest.json").write_text(json.dumps(manifest))
        return manifest

    def test_repeat_export_uses_first_identical_sample(self):
        measured = [{name: np.ones((40, 132), dtype=np.float32)
                     for name in ("normalized_actions", "actions")} for unused in range(3)]
        self.assertIs(torch_bench.validate_repeat_outputs(measured), measured[0])

    def test_repeat_export_rejects_different_outputs(self):
        for name in ("normalized_actions", "actions"):
            for changed_index in (1, 2):
                with self.subTest(name=name, changed_index=changed_index):
                    measured = [{key: np.ones((40, 132), dtype=np.float32)
                                 for key in ("normalized_actions", "actions")} for unused in range(3)]
                    measured[changed_index][name][0, 0] = np.nextafter(np.float32(1), np.float32(2))
                    with self.assertRaisesRegex(RuntimeError, "refusing to publish"):
                        torch_bench.validate_repeat_outputs(measured)

    def test_repeat_export_rejects_nonfinite_outputs(self):
        for name in ("normalized_actions", "actions"):
            for value in (np.nan, np.inf):
                with self.subTest(name=name, value=value):
                    measured = [{key: np.ones((40, 132), dtype=np.float32)
                                 for key in ("normalized_actions", "actions")} for unused in range(3)]
                    measured[1][name][0, 0] = value
                    with self.assertRaisesRegex(ValueError, "finite"):
                        torch_bench.validate_repeat_outputs(measured)

    def test_repeat_export_requires_multiple_runs_before_loading(self):
        for repeat in (0, 1):
            with self.subTest(repeat=repeat):
                with self.assertRaisesRegex(ValueError, "at least two"):
                    torch_bench.validate_repeat_outputs([{}] * repeat)
                args = SimpleNamespace(warmup=0, repeat=repeat)
                with patch.object(torch_bench, "parse_args", return_value=args):
                    with patch.object(torch_bench, "require_path") as require_path:
                        with self.assertRaisesRegex(ValueError, "at least 2"):
                            torch_bench.main()
                        require_path.assert_not_called()

    def test_rust_benchmark_rejects_zero_warmup_before_launch(self):
        args = SimpleNamespace(warmup=0, iterations=50)
        with patch.object(rust, "parse_args", return_value=args):
            with patch.object(rust.subprocess, "run") as run:
                with self.assertRaisesRegex(SystemExit, "--warmup.*positive"):
                    rust.main()
                run.assert_not_called()

    def test_all_scripts_share_readers(self):
        self.assertIs(wheel.prepare_fixture, prompt_fixture.prepare_fixture)
        self.assertIs(rust.prepare_fixture, prompt_fixture.prepare_fixture)
        self.assertIs(prompt_fixture.load_fixture, formats.load_fixture)
        self.assertIs(torch_bench.load_noise, formats.load_noise)
        self.assertIs(compare_outputs.load_archive, formats.load_archive)
        self.assertIs(wheel.load_archive, formats.load_archive)
        self.assertIs(rust.load_archive, formats.load_archive)

    def test_bf16_bits_are_not_integer_values(self):
        for dtype in ("<u2", ">u2"):
            with self.subTest(dtype=dtype):
                raw = np.array([0x3F80, 0xBF80, 0x3F00], dtype=dtype)
                decoded = formats.decode_float_array(raw, "noise")
                np.testing.assert_array_equal(decoded, [1.0, -1.0, 0.5])
                self.assertEqual(decoded.dtype, np.float32)
                self.assertTrue(decoded.flags.c_contiguous)

    def test_float_dtypes_and_noncontiguous_arrays(self):
        for dtype in (np.float16, np.float32, np.float64):
            values = np.arange(12, dtype=dtype).reshape(3, 4)[:, ::2]
            decoded = formats.decode_float_array(values, "output")
            np.testing.assert_array_equal(decoded, values)
            self.assertEqual(decoded.dtype, np.float32)
            self.assertTrue(decoded.flags.c_contiguous)

    def test_invalid_numeric_encodings_are_rejected(self):
        for values in (
            np.array([1], dtype=np.int16), np.array([1], dtype=np.uint32),
            np.array([True]), np.array(["1"]), np.array([1], dtype=object),
            np.array([1j]), np.array([]), np.array([np.nan]), np.array([np.inf]),
            np.array([1e300]), np.array([0x7F80], dtype=np.uint16),
            np.array([0x7FC0], dtype=np.uint16),
        ):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    formats.decode_float_array(values, "input")

    def test_noise_npy_npz_dtype_and_batch_variants(self):
        for extension in (".npy", ".npz"):
            for shape in ((40, 132), (1, 40, 132)):
                for dtype in (np.uint16, np.float16, np.float32, np.float64):
                    with self.subTest(extension=extension, shape=shape, dtype=dtype):
                        values = np.full(shape, 0xBF80 if dtype == np.uint16 else -1, dtype=dtype)
                        path = self.root / f"noise{extension}"
                        if extension == ".npy":
                            np.save(path, values)
                        else:
                            np.savez(path, noise=values)
                        decoded = torch_bench.load_noise(path)
                        self.assertEqual(decoded.shape, (1, 40, 132))
                        np.testing.assert_array_equal(decoded, np.full((1, 40, 132), -1.0))

    def test_noise_requires_key_shape_and_no_pickle(self):
        path = self.root / "noise.npz"
        np.savez(path, normalized_actions=np.ones((40, 132)))
        with self.assertRaisesRegex(ValueError, "no 'noise'"):
            formats.load_noise(path)
        for values in (np.ones((2, 40, 132)), np.ones(5280), np.ones((132, 40)), np.ones((40, 132), dtype=object)):
            np.savez(path, noise=values)
            with self.assertRaises(ValueError):
                formats.load_noise(path)

    def test_npz_keeps_normalized_and_decoded_actions_separate(self):
        path = self.root / "gold.npz"
        np.savez(path, normalized_actions=np.ones((2, 3)), actions=np.zeros((2, 3)))
        archive = formats.load_archive(path)
        np.testing.assert_array_equal(archive["normalized_actions"], np.ones((2, 3)))
        np.testing.assert_array_equal(archive["actions"], np.zeros((2, 3)))

    def test_json_preserves_output_shape(self):
        path = self.root / "gold.json"
        path.write_text(json.dumps({"output": {"shape": [2, 3], "values": list(range(6))}}))
        np.testing.assert_array_equal(
            formats.load_archive(path)["normalized_actions"], np.arange(6).reshape(2, 3)
        )

    def test_references_share_bf16_and_batch_decoding(self):
        for extension in (".npy", ".npz", ".json"):
            path = self.root / f"reference{extension}"
            if extension == ".npy":
                np.save(path, np.full((1, 2, 3), 0xBF80, dtype=np.uint16))
            elif extension == ".npz":
                np.savez(path, normalized_actions=np.full((1, 2, 3), 0xBF80, dtype=np.uint16))
            else:
                path.write_text(json.dumps({"output": {"shape": [1, 2, 3], "values": [-1] * 6}}))
            np.testing.assert_array_equal(rust.load_reference(path), -np.ones((2, 3)))
            np.testing.assert_array_equal(wheel.load_archive(path)["normalized_actions"], -np.ones((2, 3)))

    def test_reference_noise_encodings_compare_equal(self):
        path = self.root / "reference.npz"
        np.savez(path, normalized_actions=np.ones((40, 132)), noise=np.full((40, 132), 0x3F00, dtype=np.uint16))
        decoded = formats.load_archive(path)
        np.testing.assert_array_equal(decoded["noise"], np.full((1, 40, 132), 0.5))

    def test_decoded_actions_and_legacy_aliases_are_not_normalized_actions(self):
        path = self.root / "reference.npz"
        for key in ("actions", "final_action", "output"):
            np.savez(path, **{key: np.ones((40, 132))})
            with self.assertRaisesRegex(ValueError, "missing normalized_actions"):
                rust.load_reference(path)

    def test_json_requires_explicit_valid_shape_and_numeric_values(self):
        path = self.root / "reference.json"
        documents = [[], {"actions": [[1]]}, {"output": {"values": [1]}}]
        for shape, values in (([2, 3], [1]), ([-1, 3], [1] * 6), ([True, 3], [1] * 3),
                              ([1.0, 3], [1] * 3), ([1, 1], ["1"]), ([1, 1], [float("nan")])):
            documents.append({"output": {"shape": shape, "values": values}})
        for document in documents:
            path.write_text(json.dumps(document))
            with self.subTest(document=document), self.assertRaises(ValueError):
                formats.load_archive(path)

    def test_fixture_decodes_only_floating_tensors(self):
        self.fixture()
        _, arrays = prompt_fixture.load_fixture(self.root)
        np.testing.assert_array_equal(arrays["pixel_values"], np.ones((4, 8)))
        np.testing.assert_array_equal(arrays["state"], -np.ones((1, 1, 132)))
        np.testing.assert_array_equal(arrays["noise"], np.full((1, 40, 132), 0.5))
        self.assertEqual(arrays["token_ids"].dtype, np.uint32)
        self.assertEqual(arrays["attention_mask"].dtype, np.uint8)

    def test_fixture_rejects_bad_size_dtype_shape_and_paths(self):
        for problem in ("size", "dtype", "shape", "escape", "nan", "mask", "embodiment"):
            with self.subTest(problem=problem):
                manifest = self.fixture()
                entry = manifest["tensors"]["noise"]
                if problem == "size":
                    with (self.root / "noise.bin").open("ab") as handle:
                        handle.write(b"x")
                elif problem == "dtype":
                    entry["dtype"] = "float32"
                elif problem == "shape":
                    entry["shape"] = [-1, 40, 132]
                elif problem == "escape":
                    entry["file"] = "../outside.bin"
                elif problem == "nan":
                    np.full((1, 40, 132), 0x7FC0, dtype="<u2").tofile(self.root / "noise.bin")
                elif problem == "mask":
                    np.array([1, 2, 1], dtype=np.uint8).tofile(self.root / "attention_mask.bin")
                else:
                    manifest["embodiment_id"] = True
                (self.root / "manifest.json").write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    formats.load_fixture(self.root)

    def test_noise_mismatch_fails_before_launching_a_model(self):
        self.fixture()
        reference = self.root / "reference.npz"
        np.savez(reference, normalized_actions=np.ones((40, 132)), noise=np.ones((40, 132)))
        args = SimpleNamespace(fixture=self.root, reference=reference, warmup=1, iterations=1,
                               output=self.root / "output.json", prompt=None,
                               reference_noise=None, allow_missing_noise=False,
                               expect_model="nvidia/GR00T-N1.7-LIBERO")
        for module in (rust, wheel):
            with patch.object(module, "parse_args", return_value=args):
                with self.assertRaisesRegex(ValueError, "noise differ"):
                    module.main()

    def prompt_args(self, prompt="new prompt"):
        manifest = self.fixture()
        manifest["prompt"] = "original prompt"
        (self.root / "manifest.json").write_text(json.dumps(manifest))
        return SimpleNamespace(fixture=self.root, prompt=prompt, checkpoint=self.root,
                               backbone=self.root, reference=None,
                               output=self.root / "result.json")

    def test_prompt_fixture_preserves_non_language_bytes(self):
        args = self.prompt_args()
        original = (self.root / "manifest.json").read_bytes()
        with patch.object(prompt_fixture, "encode_prompt", side_effect=[
            np.array([1, 2, 3], dtype="<u4"), np.array([1, 2, 3, 4], dtype="<u4")
        ]):
            destination, manifest, arrays = prompt_fixture.prepare_fixture(args)
        self.assertEqual(manifest["prompt"], "new prompt")
        self.assertEqual(arrays["token_ids"].size, 4)
        np.testing.assert_array_equal(arrays["attention_mask"], [1, 1, 1, 1])
        self.assertEqual(original, (self.root / "manifest.json").read_bytes())
        for name in ("pixel_values", "image_grid_thw", "state", "noise"):
            self.assertEqual((self.root / f"{name}.bin").read_bytes(),
                             (destination / f"{name}.bin").read_bytes())

    def test_prompt_requires_exact_original_tokens(self):
        args = self.prompt_args()
        with patch.object(prompt_fixture, "encode_prompt", return_value=np.array([9])):
            with self.assertRaisesRegex(ValueError, "does not reproduce"):
                prompt_fixture.prepare_fixture(args)
        self.assertFalse(args.output.with_suffix(".fixture").exists())

    def test_changed_prompt_refuses_stale_reference(self):
        args = self.prompt_args()
        args.reference = self.root / "reference.npz"
        with patch.object(prompt_fixture, "encode_prompt", side_effect=[
            np.array([1, 2, 3]), np.array([1, 2, 3, 4])
        ]):
            with self.assertRaisesRegex(ValueError, "omit --reference"):
                prompt_fixture.prepare_fixture(args)

    def test_unchanged_prompt_reuses_fixture(self):
        args = self.prompt_args("original prompt")
        with patch.object(prompt_fixture, "encode_prompt", return_value=np.array([1, 2, 3])):
            destination, _, _ = prompt_fixture.prepare_fixture(args)
        self.assertEqual(destination, self.root)

    def test_wheel_statistics_use_internal_not_api_time(self):
        self.fixture()
        args = SimpleNamespace(fixture=self.root, prompt=None, reference=None,
                               reference_noise=None, allow_missing_noise=False,
                               expect_model="nvidia/GR00T-N1.7-LIBERO",
                               checkpoint=self.root, backbone=self.root, device=0,
                               precision="bf16", calibration=None, tactics=None, autotune=False,
                               execution="require-graph", warmup=1, iterations=2,
                               output=self.root / "result.json")
        output = np.ones((40, 132), dtype=np.float32)
        calls = []

        def infer(*inputs):
            for index in (0, 4, 6):
                self.assertEqual(inputs[index].dtype, np.uint16)
            calls.append(True)
            return output, 20.0

        model = SimpleNamespace(_infer_preprocessed_timed=infer, execution_mode="musa-graph")
        extension = SimpleNamespace(__file__="test-wheel", ModelRunner=SimpleNamespace(
            load=lambda *arguments, **options: model))
        with patch.object(wheel, "parse_args", return_value=args), \
             patch.dict(sys.modules, {"apxinf_py": extension}), \
             patch.object(wheel.time, "perf_counter", side_effect=[0, 1, 2, 2.1, 3, 3.1]), \
             contextlib.redirect_stdout(io.StringIO()):
            wheel.main()
        report = json.loads(args.output.read_text())
        self.assertEqual(report["latency_ms"]["p50"], 20.0)
        self.assertAlmostEqual(report["api_latency_ms"]["p50"], 100.0)
        self.assertEqual(report["model_hz"], 50.0)
        self.assertEqual(calls, [True, True, True])

        extension.ModelRunner.load = lambda *arguments, **options: SimpleNamespace(
            _infer_preprocessed=lambda *inputs: self.fail("must not fall back to legacy API timing"))
        with patch.object(wheel, "parse_args", return_value=args), \
             patch.dict(sys.modules, {"apxinf_py": extension}), \
             self.assertRaisesRegex(RuntimeError, "rebuild the wheel"):
            wheel.main()

    def test_fixture_bf16_bits_roundtrip_without_quantization(self):
        bits = np.array([0x0000, 0x8000, 0x3F80, 0xBF80, 0x0001, 0x7F7F], dtype=np.uint16)
        values = formats.decode_float_array(bits, "test")
        np.testing.assert_array_equal(formats.fixture_bf16_bits(values), bits)
        np.testing.assert_array_equal(formats.fixture_bf16_bits(bits), bits)
        with self.assertRaisesRegex(ValueError, "exactly representable"):
            formats.fixture_bf16_bits(np.array([1.0001], dtype=np.float32))

    def test_comparison_cli_decodes_noise_and_normalized_actions(self):
        actual = self.root / "actual.npz"
        reference = self.root / "reference.npz"
        output = self.root / "parity.json"
        np.savez(actual, normalized_actions=np.ones((40, 132), dtype=np.float32),
                 noise=np.full((1, 40, 132), 0.5, dtype=np.float32))
        np.savez(reference, normalized_actions=np.full((1, 40, 132), 0x3F80, dtype=np.uint16),
                 noise=np.full((40, 132), 0x3F00, dtype=np.uint16))
        args = SimpleNamespace(apxinf=actual, reference=reference, output=output, precision="bf16",
                               reference_noise=None, allow_missing_noise=False,
                               expect_model="nvidia/GR00T-N1.7-LIBERO")
        with patch.object(compare_outputs, "parse_args", return_value=args), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(compare_outputs.main(), 0)
        report = json.loads(output.read_text())
        self.assertTrue(report["passed"])
        self.assertTrue(report["noise_exact_match"])
        self.assertEqual(report["comparisons"]["normalized_actions"]["max_abs"], 0.0)
        np.savez(reference, normalized_actions=np.full((40, 132), 0x3F80, dtype=np.uint16),
                 noise=np.full((40, 132), 0xBF00, dtype=np.uint16))
        with patch.object(compare_outputs, "parse_args", return_value=args):
            with self.assertRaisesRegex(SystemExit, "noise differ"):
                compare_outputs.main()

    def test_json_archive_reads_noise_and_metadata(self):
        document = {
            "model": "nvidia/GR00T-N1.7-LIBERO",
            "schema": "apxinf.gr00t-n1.7.benchmark.v2",
            "output": {"shape": [40, 132], "values": np.zeros((40, 132), dtype=np.float32).tolist()},
            "noise": {"shape": [1, 40, 132], "values": np.zeros((1, 40, 132), dtype=np.float32).tolist()},
        }
        path = self.root / "archive.json"
        path.write_text(json.dumps(document))
        archive = formats.load_archive(
            path,
            expect_model=document["model"],
            expect_schema_prefix="apxinf.gr00t-n1.7.benchmark.",
        )
        self.assertEqual(archive["noise"].shape, (1, 40, 132))
        self.assertEqual(
            formats.load_metadata(path),
            {"model": document["model"], "schema": document["schema"]},
        )
        with self.assertRaisesRegex(ValueError, "model"):
            formats.load_archive(path, expect_model="other")
        with self.assertRaisesRegex(ValueError, "schema"):
            formats.load_archive(path, expect_schema_prefix="other")

    def test_json_archive_rejects_bad_noise(self):
        for noise in (
            {"shape": [1, 40], "values": [0.0] * 40},
            {"shape": [1, 40, 132], "values": [float("nan")] + [0.0] * 5279},
            {"shape": [1, 40, 132], "values": [0.0] * 5279},
        ):
            with self.subTest(noise=noise):
                document = {
                    "output": {"shape": [40, 132], "values": [0.0] * 5280},
                    "noise": noise,
                }
                path = self.root / "archive.json"
                path.write_text(json.dumps(document))
                with self.assertRaises(ValueError):
                    formats.load_archive(path)

    def test_comparison_cli_enforces_max_abs_and_noise_policy(self):
        actual = self.root / "actual.json"
        reference = self.root / "reference.json"
        output = self.root / "parity.json"
        base = np.ones((40, 132), dtype=np.float32)
        noise = np.zeros((1, 40, 132), dtype=np.float32)

        def write(path, values, model, schema=None, with_noise=True):
            document = {
                "model": model,
                "output": {"shape": [40, 132], "values": values.tolist()},
            }
            if schema is not None:
                document["schema"] = schema
            if with_noise:
                document["noise"] = {"shape": [1, 40, 132], "values": noise.tolist()}
            path.write_text(json.dumps(document))

        write(actual, base, "gr00t", "apxinf.gr00t-n1.7.benchmark.v2")
        write(reference, base, "nvidia/GR00T-N1.7-LIBERO")
        args = SimpleNamespace(
            apxinf=actual,
            reference=reference,
            output=output,
            precision="bf16",
            reference_noise=None,
            allow_missing_noise=False,
            expect_model="nvidia/GR00T-N1.7-LIBERO",
        )
        with patch.object(compare_outputs, "parse_args", return_value=args), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(compare_outputs.main(), 0)

        drifted = base.copy()
        drifted[0, 0] += 0.126
        write(actual, drifted, "gr00t", "apxinf.gr00t-n1.7.benchmark.v2")
        with patch.object(compare_outputs, "parse_args", return_value=args):
            with self.assertRaisesRegex(SystemExit, "max_abs"):
                compare_outputs.main()

        write(actual, base, "gr00t", "apxinf.gr00t-n1.7.benchmark.v2", with_noise=False)
        write(reference, base, "nvidia/GR00T-N1.7-LIBERO", with_noise=False)
        with patch.object(compare_outputs, "parse_args", return_value=args):
            with self.assertRaisesRegex(SystemExit, "noise is required"):
                compare_outputs.main()
        args.allow_missing_noise = True
        with patch.object(compare_outputs, "parse_args", return_value=args), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(compare_outputs.main(), 0)


if __name__ == "__main__":
    unittest.main()
