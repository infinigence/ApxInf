import importlib.util
import struct
import unittest
from unittest import mock
from pathlib import Path


SCRIPT = Path(__file__).with_name("qwen35_checkpoint_runner.py")
SPEC = importlib.util.spec_from_file_location("qwen35_checkpoint_runner", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)


class RunnerUnitTests(unittest.TestCase):
    def test_greedy_stop_includes_eos_and_reports_eos(self):
        tokens, reason = runner.greedy_token_ids([11, 248046, 12], 8)
        self.assertEqual(tokens, [11, 248046])
        self.assertEqual(reason, "eos")

    def test_greedy_stop_consumes_budget_without_eos(self):
        tokens, reason = runner.greedy_token_ids([11, 12, 13], 3)
        self.assertEqual(tokens, [11, 12, 13])
        self.assertEqual(reason, "budget")

    def test_artifact_path_rejects_escape(self):
        with self.assertRaises(ValueError):
            runner.safe_artifact_path(Path("/tmp/apxinf-runner-test"), "../escape.bin")

    def test_f32_writer_is_little_endian_and_finite(self):
        path = Path("/tmp/apxinf-runner-f32.bin")
        try:
            runner.write_f32(path, [1.0, -2.5])
            self.assertEqual(path.read_bytes(), struct.pack("<ff", 1.0, -2.5))
        finally:
            path.unlink(missing_ok=True)

    def test_ephemeral_decompression_hooks_restore_packed_state_each_module(self):
        self.assertIsNotNone(getattr(runner, "install_ephemeral_decompression_hooks", None))

        class FakeModule:
            quantization_scheme = mock.sentinel.scheme

            def register_forward_pre_hook(self, callback):
                self.pre = callback
                return mock.sentinel.pre_handle

            def register_forward_hook(self, callback, *, always_call):
                self.always_call = always_call
                self.post = callback
                return mock.sentinel.post_handle

        first = FakeModule()
        second = FakeModule()
        class FakeModel:
            def named_modules(self):
                return [("first", first), ("second", second)]

        with mock.patch.object(runner, "snapshot_compressed_module", create=True,
                               return_value=mock.sentinel.state) as snapshot, mock.patch.object(
            runner, "decompress_module", create=True
        ) as decompress, mock.patch.object(
            runner, "restore_compressed_module", create=True
        ) as restore:
            handles = runner.install_ephemeral_decompression_hooks(FakeModel())
            self.assertEqual(handles, [mock.sentinel.pre_handle, mock.sentinel.post_handle,
                                       mock.sentinel.pre_handle, mock.sentinel.post_handle])
            first.pre(first, ())
            first.post(first, (), None)
            snapshot.assert_called_once_with(first)
            decompress.assert_called_once_with(first)
            restore.assert_called_once_with(first, mock.sentinel.state)
            self.assertTrue(first.always_call)

    def test_gdn_required_dimensions_are_copied_from_job_model(self):
        self.assertIsNotNone(getattr(runner, "gdn_required_dimensions", None))
        self.assertEqual(
            runner.gdn_required_dimensions(
                {
                    "linear_conv_kernel_dim": 4,
                    "linear_key_head_dim": 128,
                    "linear_num_key_heads": 16,
                    "linear_num_value_heads": 48,
                    "linear_value_head_dim": 128,
                }
            ),
            {
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 128,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 48,
                "linear_value_head_dim": 128,
            },
        )


if __name__ == "__main__":
    unittest.main()
