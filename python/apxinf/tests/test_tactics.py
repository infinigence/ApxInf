import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

from apxinf import _tactics
from apxinf.policies.impls import pi05


class Pi05TacticSelectionTest(unittest.TestCase):
    def test_selects_validated_source_tactics(self):
        cases = [
            (87, "bf16", "orin-sm87"),
            (89, "bf16", "rtx4090-sm89"),
            (101, "fp8", "thor-sm101"),
            (110, "bf16", "thor-sm110"),
            (110, "fp8", "thor-sm110"),
        ]
        for sm, precision, directory in cases:
            with self.subTest(sm=sm, precision=precision), tempfile.TemporaryDirectory() as root:
                root = pathlib.Path(root)
                path = root / "configs" / "tuning" / "nvidia" / directory / "tactics.json"
                path.parent.mkdir(parents=True)
                path.touch()
                with mock.patch.object(_tactics, "_SOURCE_ROOT", root), mock.patch.object(
                    _tactics, "cuda_sm", return_value=sm
                ):
                    selected = _tactics.resolve_pi05_tactics("cuda:0", precision)
                self.assertEqual(selected, path)

    def test_checkpoint_tactics_precede_source_default(self):
        with tempfile.TemporaryDirectory() as root:
            path = pathlib.Path(root) / "tactics.json"
            path.touch()
            with mock.patch.object(_tactics, "cuda_sm") as cuda_sm:
                selected = _tactics.resolve_pi05_tactics(
                    "cuda:0", "bf16", model_dir=path.parent
                )
            self.assertEqual(selected, path)
            cuda_sm.assert_not_called()

    def test_precision_modes_share_one_hardware_database(self):
        with tempfile.TemporaryDirectory() as root:
            root = pathlib.Path(root)
            path = root / "configs" / "tuning" / "nvidia" / "orin-sm87" / "tactics.json"
            path.parent.mkdir(parents=True)
            path.touch()
            with mock.patch.object(_tactics, "_SOURCE_ROOT", root), mock.patch.object(
                _tactics, "cuda_sm", return_value=87
            ):
                self.assertEqual(_tactics.resolve_pi05_tactics("cuda:0", "int8"), path)

    def test_autotune_can_create_missing_hardware_database(self):
        with tempfile.TemporaryDirectory() as root:
            root = pathlib.Path(root)
            expected = root / "configs" / "tuning" / "nvidia" / "thor-sm110" / "tactics.json"
            with mock.patch.object(_tactics, "_SOURCE_ROOT", root), mock.patch.object(
                _tactics, "cuda_sm", return_value=110
            ):
                self.assertEqual(
                    _tactics.resolve_pi05_tactics(
                        "cuda:0", "fp8", allow_missing=True
                    ),
                    expected,
                )

    def test_hidden_override_takes_precedence(self):
        override = pathlib.Path("custom.json")
        with mock.patch.object(_tactics, "cuda_sm") as cuda_sm:
            selected = _tactics.resolve_pi05_tactics(
                "not-a-cuda-device", "bf16", override=override
            )
        self.assertEqual(selected, override)
        cuda_sm.assert_not_called()

    def test_policy_loader_passes_automatic_tactics_to_binding(self):
        selected = pathlib.Path("automatic.json")
        captured = {}

        class FakeModel:
            action_horizon = 10
            action_dim = 32
            num_views = 2
            image_size = 224
            max_token_len = 200

            def reset_sampling(self, seed=None):
                pass

        class FakeBindingModel:
            @staticmethod
            def load(*args, **kwargs):
                captured.update(kwargs)
                return FakeModel()

        class FakePipeline:
            names = []

            def __getitem__(self, name):
                raise KeyError(name)

        fake_unnormalizer = types.SimpleNamespace(width=7)
        with tempfile.TemporaryDirectory() as model_dir, mock.patch.dict(
            sys.modules, {"apxinf_py": types.SimpleNamespace(Model=FakeBindingModel)}
        ), mock.patch.object(
            pi05, "resolve_pi05_tactics", return_value=selected
        ) as resolve, mock.patch.object(
            pi05, "PromptTokenizer", return_value=object()
        ), mock.patch.object(
            pi05.Unnormalizer, "from_norm_stats", return_value=fake_unnormalizer
        ), mock.patch.object(
            pi05.Pi05Policy,
            "default_pipelines",
            return_value=(FakePipeline(), FakePipeline()),
        ):
            pi05.Pi05Policy.from_pretrained(
                model_dir,
                device="cuda:0",
                precision="bf16",
                autotune=True,
                tokenizer_path="unused-tokenizer.model",
            )

        resolve.assert_called_once_with(
            "cuda:0",
            "bf16",
            model_dir=pathlib.Path(model_dir),
            override=None,
            allow_missing=True,
        )
        self.assertEqual(captured["tactics"], str(selected))
        self.assertTrue(captured["autotune"])


if __name__ == "__main__":
    unittest.main()
