from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/apxinf_mlx_generate.py"


def write_fake_runtime(
    root: Path,
    tokens: list[int],
    *,
    eos_token_ids: tuple[int, ...] = (9,),
) -> Path:
    packages = root / "packages"
    (packages / "mlx").mkdir(parents=True)
    (packages / "mlx_lm").mkdir()
    (packages / "mlx_lm/models").mkdir()
    (packages / "mlx/__init__.py").write_text("", encoding="utf-8")
    (packages / "mlx_lm/models/__init__.py").write_text("", encoding="utf-8")
    (packages / "mlx_lm/models/cache.py").write_text(
        "def make_prompt_cache(model, max_kv_size=None):\n"
        "    if max_kv_size is not None:\n"
        "        raise ValueError('fake does not support rotating cache')\n"
        "    return model.make_cache()\n",
        encoding="utf-8",
    )
    (packages / "mlx/core.py").write_text(
        textwrap.dedent(
            """
            SYNCHRONIZED = False
            TOKENS = __TOKENS__
            TOKEN_INDEX = 0

            def reset_peak_memory():
                global SYNCHRONIZED
                SYNCHRONIZED = False
                return None

            def get_peak_memory():
                if not SYNCHRONIZED:
                    raise RuntimeError("peak read before MLX synchronization")
                return 123456

            def synchronize():
                global SYNCHRONIZED
                SYNCHRONIZED = True

            def clear_cache():
                return None

            class Array(list):
                pass

            class Scalar:
                def __init__(self, value):
                    self.value = value

                def item(self):
                    return self.value

            class Logits:
                def __init__(self, token):
                    self.token = token

                def __getitem__(self, _key):
                    return self

                def astype(self, _dtype):
                    return self

            def array(values):
                return Array(values)

            def eval(*_values):
                return None

            def argmax(logits, axis=-1):
                if axis != -1:
                    raise ValueError("unexpected argmax axis")
                return Scalar(logits.token)

            float32 = "float32"

            def next_logits():
                global TOKEN_INDEX
                if TOKEN_INDEX >= len(TOKENS):
                    raise RuntimeError("generation ended early")
                token = TOKENS[TOKEN_INDEX]
                TOKEN_INDEX += 1
                return Logits(token)
            """
        ).replace("__TOKENS__", repr(tokens)),
        encoding="utf-8",
    )
    (packages / "mlx_lm/__init__.py").write_text(
        textwrap.dedent(
            """
            import json
            import os
            import sys
            import mlx.core as mx

            class Model:
                def make_cache(self):
                    return []

                def __call__(self, _tokens, *, cache):
                    raise RuntimeError("ApxInf worker must use generate_step")

            class Tokenizer:
                eos_token_ids = EOS_TOKEN_IDS

            def load(path, **kwargs):
                print("dependency stdout noise")
                print("dependency stderr noise", file=sys.stderr)
                with open(os.environ["FAKE_MLX_CALL_LOG"], "w", encoding="utf-8") as handle:
                    json.dump(
                        {
                            "path": path,
                            "kwargs": kwargs,
                            "offline": {
                                key: os.environ.get(key)
                                for key in (
                                    "HF_HUB_OFFLINE",
                                    "TRANSFORMERS_OFFLINE",
                                    "HF_HUB_DISABLE_TELEMETRY",
                                )
                            },
                        },
                        handle,
                        sort_keys=True,
                    )
                return Model(), Tokenizer(), {
                    "model_type": "qwen3_5",
                    "quantization": {"bits": 8, "group_size": 64},
                }
            """
        ).replace("EOS_TOKEN_IDS", repr(list(eos_token_ids))),
        encoding="utf-8",
    )
    (packages / "mlx_lm/generate.py").write_text(
        "TOKENS = "
        + repr(tokens)
        + "\n\ndef generate_step(prompt, model, *, max_tokens, sampler=None, prompt_cache=None):\n"
        + "    if type(prompt).__name__ != 'Array':\n"
        + "        raise TypeError('prompt must be an MLX array')\n"
        + "    if sampler is None:\n"
        + "        raise TypeError('ApxInf must bind its explicit greedy sampler')\n"
        + "    for token in TOKENS[:max_tokens]:\n"
        + "        yield token, None\n",
        encoding="utf-8",
    )
    for dirname, name, version in (
        ("mlx-0.32.1.dist-info", "mlx", "0.32.1"),
        ("mlx_lm-0.31.3.dist-info", "mlx-lm", "0.31.3"),
        ("mlx_metal-0.32.1.dist-info", "mlx-metal", "0.32.1"),
        ("huggingface_hub-1.28.0.dist-info", "huggingface-hub", "1.28.0"),
        ("numpy-2.5.2.dist-info", "numpy", "2.5.2"),
        ("safetensors-0.8.0.dist-info", "safetensors", "0.8.0"),
        ("tokenizers-0.22.2.dist-info", "tokenizers", "0.22.2"),
        ("transformers-5.15.1.dist-info", "transformers", "5.15.1"),
    ):
        metadata = packages / dirname
        metadata.mkdir()
        (metadata / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
            encoding="utf-8",
        )
    return packages


def run_worker(
    model_dir: Path,
    packages: Path,
    call_log: Path,
    request: object,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(packages)
    environment["FAKE_MLX_CALL_LOG"] = str(call_log)
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--model-dir", str(model_dir)],
        input=json.dumps(request, separators=(",", ":")) + "\n",
        text=True,
        capture_output=True,
        env=environment,
        check=False,
    )


def run_raw_worker(
    model_dir: Path,
    standard_input: str,
    *,
    packages: Path | None = None,
    extra_environment: dict[str, str] | None = None,
    arguments: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    if packages is not None:
        environment["PYTHONPATH"] = str(packages)
    if extra_environment:
        environment.update(extra_environment)
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            *(arguments if arguments is not None else ["--model-dir", str(model_dir)]),
        ],
        input=standard_input,
        text=True,
        capture_output=True,
        env=environment,
        check=False,
    )


class MlxGenerateWorkerTests(unittest.TestCase):
    def test_generates_from_raw_ids_and_emits_one_strict_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen3_5",
                        "quantization": {"bits": 8, "group_size": 64},
                    }
                ),
                encoding="utf-8",
            )
            packages = write_fake_runtime(root, [7, 9, 11, 12])
            call_log = root / "calls.json"
            request = {
                "format": "apxinf-mlx-generation-request-v1",
                "prompt_token_ids": [1, 2, 3],
                "max_tokens": 4,
            }
            completed = run_worker(model_dir, packages, call_log, request)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stderr, "")
            self.assertEqual(completed.stdout.count("\n"), 1)
            receipt = json.loads(completed.stdout)
            self.assertEqual(
                set(receipt),
                {
                    "format",
                    "request",
                    "model",
                    "packages",
                    "runtime",
                    "metrics",
                    "generation",
                },
            )
            self.assertEqual(
                set(receipt["metrics"]),
                {
                    "load_ms",
                    "ttft_ms",
                    "tpot_ms",
                    "tps",
                    "timed_decode_tokens",
                    "mlx_peak_memory_bytes",
                },
            )
            self.assertEqual(receipt["format"], "apxinf-mlx-generation-receipt-v1")
            self.assertEqual(receipt["generation"]["generated_token_ids"], [7, 9])
            self.assertEqual(receipt["generation"]["stop_reason"], "eos")
            self.assertEqual(
                receipt["request"]["prompt_token_ids_sha256"],
                "a615eeaee21de5179de080de8c3052c8da901138406ba71c38c032845f7d54f4",
            )
            self.assertEqual(
                receipt["request"]["greedy_strategy"],
                "mlx-generate-step-argmax-v1",
            )
            self.assertEqual(
                receipt["packages"],
                {
                    "huggingface-hub": "1.28.0",
                    "mlx": "0.32.1",
                    "mlx-lm": "0.31.3",
                    "mlx-metal": "0.32.1",
                    "numpy": "2.5.2",
                    "safetensors": "0.8.0",
                    "tokenizers": "0.22.2",
                    "transformers": "5.15.1",
                },
            )
            self.assertEqual(receipt["runtime"]["python_version"], "3.14.3")
            self.assertEqual(
                receipt["runtime"]["python_executable"],
                str(Path(sys.executable).resolve(strict=True)),
            )
            self.assertEqual(
                receipt["runtime"]["python_executable_sha256"],
                hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),
            )
            self.assertEqual(
                receipt["runtime"]["runner"], str(SCRIPT.resolve(strict=True))
            )
            self.assertEqual(
                receipt["runtime"]["runner_sha256"],
                hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
            )
            self.assertEqual(receipt["model"]["model_type"], "qwen3_5")
            self.assertEqual(
                receipt["model"]["quantization"], {"bits": 8, "group_size": 64}
            )
            self.assertEqual(receipt["metrics"]["mlx_peak_memory_bytes"], 123456)
            self.assertEqual(receipt["metrics"]["timed_decode_tokens"], 1)
            self.assertGreaterEqual(receipt["metrics"]["load_ms"], 0.0)
            self.assertGreaterEqual(receipt["metrics"]["ttft_ms"], 0.0)
            self.assertGreaterEqual(receipt["metrics"]["tpot_ms"], 0.0)
            self.assertGreaterEqual(receipt["metrics"]["tps"], 0.0)
            call = json.loads(call_log.read_text(encoding="utf-8"))
            self.assertEqual(call["path"], receipt["model"]["model_dir"])
            # mlx-lm 0.31.3 exposes trust_remote_code only through the
            # tokenizer_config mapping; passing it as a top-level keyword is
            # rejected by the real load() signature.
            self.assertNotIn("trust_remote_code", call["kwargs"])
            self.assertTrue(call["kwargs"]["tokenizer_config"]["local_files_only"])
            self.assertFalse(call["kwargs"]["tokenizer_config"]["trust_remote_code"])
            self.assertEqual(
                call["offline"],
                {
                    "HF_HUB_DISABLE_TELEMETRY": "1",
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                },
            )

    def test_stop_on_eos_false_forces_exact_max_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text(
                json.dumps({"model_type": "qwen3_5"}), encoding="utf-8"
            )
            packages = write_fake_runtime(root, [7, 9, 11, 12])
            request = {
                "format": "apxinf-mlx-generation-request-v1",
                "prompt_token_ids": [1, 2, 3],
                "max_tokens": 4,
                "stop_on_eos": False,
            }

            completed = run_worker(model_dir, packages, root / "calls.json", request)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            receipt = json.loads(completed.stdout)
            self.assertEqual(
                receipt["generation"]["generated_token_ids"], [7, 9, 11, 12]
            )
            self.assertEqual(receipt["generation"]["stop_reason"], "length")
            self.assertFalse(receipt["request"]["stop_on_eos"])

    def test_zero_token_budget_returns_an_empty_valid_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text(
                '{"model_type":"qwen3_5"}', encoding="utf-8"
            )
            packages = write_fake_runtime(root, [7], eos_token_ids=())
            request = {
                "format": "apxinf-mlx-generation-request-v1",
                "prompt_token_ids": [1],
                "max_tokens": 0,
                "stop_on_eos": True,
            }

            completed = run_worker(model_dir, packages, root / "calls.json", request)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            receipt = json.loads(completed.stdout)
            self.assertEqual(receipt["generation"]["generated_token_ids"], [])
            self.assertEqual(receipt["generation"]["generated_token_count"], 0)
            self.assertEqual(receipt["generation"]["stop_reason"], "length")
            self.assertEqual(receipt["metrics"]["timed_decode_tokens"], 0)
            self.assertEqual(receipt["metrics"]["ttft_ms"], 0.0)
            self.assertEqual(receipt["metrics"]["tpot_ms"], 0.0)
            self.assertEqual(receipt["metrics"]["tps"], 0.0)

    def test_zero_eos_token_overrides_the_model_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text(
                json.dumps({"model_type": "qwen3_5"}), encoding="utf-8"
            )
            packages = write_fake_runtime(root, [7, 0, 9])
            request = {
                "format": "apxinf-mlx-generation-request-v1",
                "prompt_token_ids": [1],
                "max_tokens": 3,
                "eos_token_id": 0,
                "stop_on_eos": True,
            }

            completed = run_worker(model_dir, packages, root / "calls.json", request)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            receipt = json.loads(completed.stdout)
            self.assertEqual(receipt["generation"]["generated_token_ids"], [7, 0])
            self.assertEqual(receipt["generation"]["stop_reason"], "eos")
            self.assertEqual(receipt["request"]["effective_eos_token_ids"], [0])

    def test_rejects_a_toolchain_version_outside_the_frozen_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text(
                '{"model_type":"qwen3_5"}', encoding="utf-8"
            )
            packages = write_fake_runtime(root, [7])
            imported = root / "runtime-imported"
            (packages / "mlx/__init__.py").write_text(
                "from pathlib import Path\n"
                f"Path({str(imported)!r}).write_text('yes', encoding='utf-8')\n",
                encoding="utf-8",
            )
            (packages / "mlx-0.32.1.dist-info/METADATA").write_text(
                "Metadata-Version: 2.1\nName: mlx\nVersion: 9.9.9\n",
                encoding="utf-8",
            )
            request = {
                "format": "apxinf-mlx-generation-request-v1",
                "prompt_token_ids": [1],
                "max_tokens": 1,
            }

            completed = run_worker(model_dir, packages, root / "calls.json", request)

            self.assertEqual(completed.returncode, 2)
            self.assertEqual(completed.stdout, "")
            error = json.loads(completed.stderr)
            self.assertEqual(error["error"]["code"], "unsupported_toolchain")
            self.assertIn("mlx=9.9.9", error["error"]["message"])
            self.assertFalse(imported.exists())

    def test_invalid_request_is_one_json_error_and_does_not_import_mlx(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text(
                '{"model_type":"qwen3_5"}', encoding="utf-8"
            )
            packages = root / "packages"
            (packages / "mlx").mkdir(parents=True)
            marker = root / "imported"
            (packages / "mlx/__init__.py").write_text(
                "from pathlib import Path\n"
                f"Path({str(marker)!r}).write_text('imported', encoding='utf-8')\n",
                encoding="utf-8",
            )
            invalid = (
                '{"format":"apxinf-mlx-generation-request-v1",'
                '"prompt_token_ids":[1],"max_tokens":1,"unknown":true}\n'
            )

            completed = run_raw_worker(model_dir, invalid, packages=packages)

            self.assertEqual(completed.returncode, 2)
            self.assertEqual(completed.stdout, "")
            self.assertEqual(completed.stderr.count("\n"), 1)
            error = json.loads(completed.stderr)
            self.assertEqual(error["format"], "apxinf-mlx-generation-error-v1")
            self.assertEqual(set(error), {"format", "error"})
            self.assertEqual(error["error"]["code"], "invalid_request")
            self.assertFalse(marker.exists())

    def test_malformed_requests_fail_closed_before_runtime_import(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text(
                '{"model_type":"qwen3_5"}', encoding="utf-8"
            )
            packages = root / "packages"
            (packages / "mlx").mkdir(parents=True)
            marker = root / "imported"
            (packages / "mlx/__init__.py").write_text(
                "from pathlib import Path\n"
                f"Path({str(marker)!r}).write_text('imported', encoding='utf-8')\n",
                encoding="utf-8",
            )
            cases = {
                "duplicate key": (
                    '{"format":"apxinf-mlx-generation-request-v1",'
                    '"prompt_token_ids":[1],"max_tokens":1,"max_tokens":2}\n',
                    "invalid_json",
                ),
                "non finite": (
                    '{"format":"apxinf-mlx-generation-request-v1",'
                    '"prompt_token_ids":[1],"max_tokens":NaN}\n',
                    "invalid_json",
                ),
                "two lines": ("{}\n{}\n", "invalid_request"),
                "boolean token": (
                    '{"format":"apxinf-mlx-generation-request-v1",'
                    '"prompt_token_ids":[true],"max_tokens":1}\n',
                    "invalid_request",
                ),
                "empty prompt": (
                    '{"format":"apxinf-mlx-generation-request-v1",'
                    '"prompt_token_ids":[],"max_tokens":1}\n',
                    "invalid_request",
                ),
                "null eos": (
                    '{"format":"apxinf-mlx-generation-request-v1",'
                    '"prompt_token_ids":[1],"max_tokens":1,"eos_token_id":null}\n',
                    "invalid_request",
                ),
                "unbounded output": (
                    '{"format":"apxinf-mlx-generation-request-v1",'
                    '"prompt_token_ids":[1],"max_tokens":65537}\n',
                    "invalid_request",
                ),
                "request byte limit": (" " * (1024 * 1024 + 1), "invalid_request"),
            }

            for label, (payload, expected_code) in cases.items():
                with self.subTest(label=label):
                    completed = run_raw_worker(model_dir, payload, packages=packages)
                    self.assertEqual(completed.returncode, 2)
                    self.assertEqual(completed.stdout, "")
                    self.assertEqual(completed.stderr.count("\n"), 1)
                    error = json.loads(completed.stderr)
                    self.assertEqual(error["error"]["code"], expected_code)
                    self.assertFalse(marker.exists())

    def test_cli_failures_also_use_the_one_line_json_error_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for arguments in ([], ["--help"]):
                with self.subTest(arguments=arguments):
                    completed = run_raw_worker(
                        root,
                        "",
                        arguments=arguments,
                    )
                    self.assertNotEqual(completed.returncode, 0)
                    self.assertEqual(completed.stdout, "")
                    self.assertEqual(completed.stderr.count("\n"), 1)
                    error = json.loads(completed.stderr)
                    self.assertEqual(error["error"]["code"], "invalid_arguments")

    def test_model_must_be_an_absolute_local_directory_without_remote_code(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packages = root / "packages"
            (packages / "mlx").mkdir(parents=True)
            marker = root / "imported"
            (packages / "mlx/__init__.py").write_text(
                "from pathlib import Path\n"
                f"Path({str(marker)!r}).write_text('imported', encoding='utf-8')\n",
                encoding="utf-8",
            )
            request = (
                json.dumps(
                    {
                        "format": "apxinf-mlx-generation-request-v1",
                        "prompt_token_ids": [1],
                        "max_tokens": 1,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )

            real_model = root / "real-model"
            real_model.mkdir()
            (real_model / "config.json").write_text(
                '{"model_type":"qwen3_5"}', encoding="utf-8"
            )
            model_link = root / "model-link"
            model_link.symlink_to(real_model, target_is_directory=True)
            remote_model = root / "remote-model"
            remote_model.mkdir()
            (remote_model / "config.json").write_text(
                '{"model_type":"qwen3_5","model_file":"modeling.py"}',
                encoding="utf-8",
            )
            cases = {
                "relative": ["--model-dir", "real-model"],
                "directory symlink": ["--model-dir", str(model_link)],
                "remote code": ["--model-dir", str(remote_model)],
            }

            for label, arguments in cases.items():
                with self.subTest(label=label):
                    completed = run_raw_worker(
                        root,
                        request,
                        packages=packages,
                        arguments=arguments,
                    )
                    self.assertEqual(completed.returncode, 2)
                    self.assertEqual(completed.stdout, "")
                    error = json.loads(completed.stderr)
                    self.assertEqual(error["error"]["code"], "invalid_model")
                    self.assertFalse(marker.exists())

    def test_runtime_failures_never_publish_a_partial_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = {
                "format": "apxinf-mlx-generation-request-v1",
                "prompt_token_ids": [1],
                "max_tokens": 2,
                "stop_on_eos": False,
            }

            load_root = root / "load"
            load_root.mkdir()
            load_model = load_root / "model"
            load_model.mkdir()
            (load_model / "config.json").write_text(
                '{"model_type":"qwen3_5"}', encoding="utf-8"
            )
            load_packages = write_fake_runtime(load_root, [7, 8])
            (load_packages / "mlx_lm/__init__.py").write_text(
                "def load(path, **kwargs):\n"
                "    print('load noise')\n"
                "    raise RuntimeError('load failed\\nwith detail')\n",
                encoding="utf-8",
            )

            early_root = root / "early"
            early_root.mkdir()
            early_model = early_root / "model"
            early_model.mkdir()
            (early_model / "config.json").write_text(
                '{"model_type":"qwen3_5"}', encoding="utf-8"
            )
            early_packages = write_fake_runtime(early_root, [7])

            cases = (
                (
                    "load failure",
                    run_worker(
                        load_model,
                        load_packages,
                        load_root / "calls.json",
                        request,
                    ),
                    "model_load_failed",
                ),
                (
                    "early generation end",
                    run_worker(
                        early_model,
                        early_packages,
                        early_root / "calls.json",
                        request,
                    ),
                    "generation_failed",
                ),
            )
            for label, completed, expected_code in cases:
                with self.subTest(label=label):
                    self.assertEqual(completed.returncode, 2)
                    self.assertEqual(completed.stdout, "")
                    self.assertEqual(completed.stderr.count("\n"), 1)
                    error = json.loads(completed.stderr)
                    self.assertEqual(error["error"]["code"], expected_code)


if __name__ == "__main__":
    unittest.main()
