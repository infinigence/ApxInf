import hashlib
import json
from pathlib import Path
import stat
import sys
import tempfile
import unittest

from tools.oracle import generate_golden as oracle


REVISION = "63768c10df38c0395e12ef49edac1bd539eaeeea"


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


class OracleGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.model_dir = self.root / "model"
        self.model_dir.mkdir()
        write_json(
            self.model_dir / "config.json",
            {
                "model_type": "qwen3_5",
                "text_config": {
                    "vocab_size": 248320,
                    "hidden_size": 5120,
                    "num_hidden_layers": 64,
                    "num_attention_heads": 24,
                    "num_key_value_heads": 4,
                    "head_dim": 256,
                    "linear_conv_kernel_dim": 4,
                    "linear_key_head_dim": 128,
                    "linear_num_key_heads": 16,
                    "linear_num_value_heads": 48,
                    "linear_value_head_dim": 128,
                    "layer_types": [
                        "linear_attention" if index % 4 != 3 else "full_attention"
                        for index in range(64)
                    ],
                },
            },
        )
        write_json(
            self.model_dir / "generation_config.json",
            {"eos_token_id": [248046, 248044], "do_sample": False},
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_build_job_normalizes_selection_and_freezes_model_contract(self) -> None:
        job = oracle.build_job(
            model_dir=self.model_dir,
            revision=REVISION,
            layers=[63, 0, 31, 31],
            stages=["tokens", "layer_hidden", "logits", "tokens"],
            input_manifest=None,
            max_new_tokens=4,
        )

        self.assertEqual(job["selection"]["layers"], [0, 31, 63])
        self.assertEqual(
            job["selection"]["stages"], ["layer_hidden", "logits", "tokens"]
        )
        self.assertEqual(job["model"]["vocab_size"], 248320)
        self.assertEqual(job["model"]["revision"], REVISION)
        self.assertEqual(job["generation"]["eos_token_ids"], [248046, 248044])
        self.assertEqual(job["generation"]["temperature"], 0)
        self.assertEqual(job["generation"]["max_new_tokens"], 4)
        self.assertEqual(job["input"]["input_ids"], [1, 2, 3, 4, 5, 6, 7, 8])
        self.assertNotIn("output_token_ids", job)
        self.assertTrue(all(item["status"] == "pending" for item in job["artifacts"]))
        artifact_names = [item["file"] for item in job["artifacts"]]
        self.assertIn("output-tokens.json", artifact_names)
        self.assertIn("logits.f32.bin", artifact_names)
        self.assertIn("layer-000-hidden.f32.bin", artifact_names)
        self.assertIn("layer-063-hidden.f32.bin", artifact_names)

    def test_bundle_is_canonical_and_hashes_every_control_manifest(self) -> None:
        input_path = self.root / "input.json"
        write_json(input_path, {"schema": "apxinf.oracle-input.v1", "input_ids": [248056, 7]})
        job = oracle.build_job(
            model_dir=self.model_dir,
            revision=REVISION,
            layers=[0, 3],
            stages=["embedding", "gdn_state", "kv_state"],
            input_manifest=input_path,
            max_new_tokens=2,
        )
        output_dir = self.root / "bundle"
        manifest_path = oracle.write_manifest_bundle(job, output_dir)

        self.assertEqual(manifest_path, output_dir / "job-manifest.json")
        expected = {
            "input-manifest.json",
            "selection.json",
            "generation.json",
            "golden-schema.json",
            "artifact-manifest.json",
            "job-manifest.json",
        }
        self.assertEqual({path.name for path in output_dir.iterdir()}, expected | {"artifacts"})
        job_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name in expected - {"job-manifest.json"}:
            raw = (output_dir / name).read_bytes()
            self.assertEqual(job_manifest["control_files"][name], hashlib.sha256(raw).hexdigest())
            self.assertTrue(raw.endswith(b"\n"))
        self.assertEqual(job_manifest["status"], "manifest_only")
        self.assertEqual(job_manifest["input_manifest_sha256"], job["input"]["sha256"])
        self.assertEqual(list((output_dir / "artifacts").iterdir()), [])

    def test_layers_default_to_hidden_and_state_artifacts_follow_layer_type(self) -> None:
        hidden_job = oracle.build_job(
            self.model_dir, REVISION, [0], [], None, 1
        )
        self.assertEqual(hidden_job["selection"]["stages"], ["layer_hidden"])
        self.assertEqual(
            [item["file"] for item in hidden_job["artifacts"]],
            ["layer-000-hidden.f32.bin"],
        )

        state_job = oracle.build_job(
            self.model_dir,
            REVISION,
            [0, 3],
            ["gdn_state", "kv_state"],
            None,
            1,
        )
        self.assertEqual(
            [item["file"] for item in state_job["artifacts"]],
            [
                "layer-000-gdn-state.f32.bin",
                "layer-003-kv-key.f32.bin",
                "layer-003-kv-value.f32.bin",
            ],
        )

    def test_artifact_identity_changes_with_generation_parameters(self) -> None:
        short = oracle.build_job(
            self.model_dir, REVISION, [], ["tokens"], None, 1
        )
        longer = oracle.build_job(
            self.model_dir, REVISION, [], ["tokens"], None, 2
        )
        self.assertNotEqual(
            short["artifact_identity_sha256"], longer["artifact_identity_sha256"]
        )

    def test_symbolic_shapes_and_f32_values_fail_closed(self) -> None:
        symbols = {
            "prompt_tokens": 8,
            "completion_tokens": 1,
            "trajectory_tokens": 9,
        }
        self.assertEqual(
            oracle._validate_shape(
                [9, 5120], ["trajectory_tokens", 5120], symbols, "hidden.bin"
            ),
            [9, 5120],
        )
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            oracle._validate_shape(
                [8, 5120], ["trajectory_tokens", 5120], symbols, "hidden.bin"
            )
        nan_path = self.root / "nan.f32.bin"
        nan_path.write_bytes(__import__("struct").pack("<f", float("nan")))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            oracle._validate_f32_artifact(nan_path, [1], "nan.f32.bin")

    def test_rejects_wrong_vocab_eos_empty_revision_and_invalid_selection(self) -> None:
        config_path = self.model_dir / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["text_config"]["vocab_size"] = 248044
        write_json(config_path, config)
        with self.assertRaisesRegex(ValueError, "248320"):
            oracle.build_job(self.model_dir, REVISION, [0], ["tokens"], None, 1)

        config["text_config"]["vocab_size"] = 248320
        write_json(config_path, config)
        write_json(self.model_dir / "generation_config.json", {"eos_token_id": [248044]})
        with self.assertRaisesRegex(ValueError, "248046"):
            oracle.build_job(self.model_dir, REVISION, [0], ["tokens"], None, 1)

        write_json(
            self.model_dir / "generation_config.json",
            {"eos_token_id": [248046, 248044]},
        )
        for revision, layers, stages, message in [
            ("", [0], ["tokens"], "revision"),
            ("wrong-revision", [0], ["tokens"], "revision"),
            (REVISION, [64], ["tokens"], "layer"),
            (REVISION, [], ["unknown"], "stage"),
            (REVISION, [], [], "selection"),
        ]:
            with self.subTest(revision=revision, layers=layers, stages=stages):
                with self.assertRaisesRegex(ValueError, message):
                    oracle.build_job(
                        self.model_dir, revision, layers, stages, None, 1
                    )


class RunnerTests(OracleGeneratorTests):
    def setUp(self) -> None:
        super().setUp()
        self.bundle_index = 0

    def _make_bundle(self, max_new_tokens: int = 1) -> Path:
        job = oracle.build_job(
            self.model_dir, REVISION, [], ["tokens"], None, max_new_tokens
        )
        self.bundle_index += 1
        output_dir = self.root / f"runner-bundle-{self.bundle_index}"
        oracle.write_manifest_bundle(job, output_dir)
        return output_dir

    def _write_runner(self, body: str) -> Path:
        path = self.root / "runner.py"
        path.write_text(
            "#!/usr/bin/env python3\n"
            "import hashlib, json, os\n"
            "from pathlib import Path\n"
            "out = Path(os.environ['APXINF_ORACLE_OUTPUT_DIR'])\n"
            "job = json.loads(Path(os.environ['APXINF_ORACLE_JOB_MANIFEST']).read_text())\n"
            + body,
            encoding="utf-8",
        )
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        return path

    def test_runner_validates_metadata_and_records_artifact_hashes(self) -> None:
        bundle = self._make_bundle()
        runner = self._write_runner(
            "payload = json.dumps({'schema':'apxinf.oracle-tokens.v1','output_token_ids':[7],"
            "'decoded_text':'fixture'}, sort_keys=True).encode() + b'\\n'\n"
            "(out / 'output-tokens.json').write_bytes(payload)\n"
            "report = {'schema':'apxinf.oracle-artifact-report.v1','generation':{'completion_tokens':1,'stop_reason':'budget'},'artifacts':["
            "{'file':'output-tokens.json','schema_ref':'tokens','dtype':'json','shape':[1],"
            "'sha256':hashlib.sha256(payload).hexdigest()}]}\n"
            "(out / 'artifact-report.json').write_text(json.dumps(report))\n"
        )

        oracle.run_runner(bundle, [sys.executable, str(runner)], [])

        artifact_manifest = json.loads(
            (bundle / "artifact-manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(artifact_manifest["status"], "complete")
        self.assertRegex(artifact_manifest["artifacts"][0]["sha256"], r"^[0-9a-f]{64}$")
        job_manifest = json.loads((bundle / "job-manifest.json").read_text())
        self.assertEqual(job_manifest["status"], "complete")

    def test_runner_rejects_missing_extra_invalid_metadata_and_nonzero_exit(self) -> None:
        cases = {
            "missing": "(out / 'artifact-report.json').write_text(json.dumps({'schema':'apxinf.oracle-artifact-report.v1','artifacts':[]}))\n",
            "extra": "(out / 'output-tokens.json').write_text('{}')\n(out / 'extra.bin').write_bytes(b'x')\n(out / 'artifact-report.json').write_text(json.dumps({'schema':'apxinf.oracle-artifact-report.v1','artifacts':[]}))\n",
            "metadata": "payload=b'{}\\n'\n(out / 'output-tokens.json').write_bytes(payload)\n(out / 'artifact-report.json').write_text(json.dumps({'schema':'wrong','artifacts':[]}))\n",
            "schema_ref": "payload=json.dumps({'schema':'apxinf.oracle-tokens.v1','output_token_ids':[7],'decoded_text':'x'}).encode()+b'\\n'\n(out / 'output-tokens.json').write_bytes(payload)\n(out / 'artifact-report.json').write_text(json.dumps({'schema':'apxinf.oracle-artifact-report.v1','artifacts':[{'file':'output-tokens.json','dtype':'json','shape':[1],'sha256':hashlib.sha256(payload).hexdigest()}]}))\n",
            "extra_directory": "payload=json.dumps({'schema':'apxinf.oracle-tokens.v1','output_token_ids':[7],'decoded_text':'x'}).encode()+b'\\n'\n(out / 'output-tokens.json').write_bytes(payload)\n(out / 'unexpected').mkdir()\n(out / 'artifact-report.json').write_text(json.dumps({'schema':'apxinf.oracle-artifact-report.v1','artifacts':[{'file':'output-tokens.json','schema_ref':'tokens','dtype':'json','shape':[1],'sha256':hashlib.sha256(payload).hexdigest()}]}))\n",
            "symlink": "target=out.parent / 'outside.json'\ntarget.write_text('{}')\n(out / 'output-tokens.json').symlink_to(target)\npayload=(out / 'output-tokens.json').read_bytes()\n(out / 'artifact-report.json').write_text(json.dumps({'schema':'apxinf.oracle-artifact-report.v1','artifacts':[{'file':'output-tokens.json','schema_ref':'tokens','dtype':'json','shape':[0],'sha256':hashlib.sha256(payload).hexdigest()}]}))\n",
            "control_tamper": "payload=json.dumps({'schema':'apxinf.oracle-tokens.v1','output_token_ids':[7],'decoded_text':'x'}).encode()+b'\\n'\n(out / 'output-tokens.json').write_bytes(payload)\n(out.parent / 'selection.json').write_text('{}')\n(out / 'artifact-report.json').write_text(json.dumps({'schema':'apxinf.oracle-artifact-report.v1','artifacts':[{'file':'output-tokens.json','schema_ref':'tokens','dtype':'json','shape':[1],'sha256':hashlib.sha256(payload).hexdigest()}]}))\n",
            "empty_tokens": "payload=json.dumps({'schema':'apxinf.oracle-tokens.v1','output_token_ids':[],'decoded_text':''}).encode()+b'\\n'\n(out / 'output-tokens.json').write_bytes(payload)\n(out / 'artifact-report.json').write_text(json.dumps({'schema':'apxinf.oracle-artifact-report.v1','generation':{'completion_tokens':0,'stop_reason':'eos'},'artifacts':[{'file':'output-tokens.json','schema_ref':'tokens','dtype':'json','shape':[0],'sha256':hashlib.sha256(payload).hexdigest()}]}))\n",
            "eos_followed": "payload=json.dumps({'schema':'apxinf.oracle-tokens.v1','output_token_ids':[248046,7],'decoded_text':'x'}).encode()+b'\\n'\n(out / 'output-tokens.json').write_bytes(payload)\n(out / 'artifact-report.json').write_text(json.dumps({'schema':'apxinf.oracle-artifact-report.v1','generation':{'completion_tokens':2,'stop_reason':'eos'},'artifacts':[{'file':'output-tokens.json','schema_ref':'tokens','dtype':'json','shape':[2],'sha256':hashlib.sha256(payload).hexdigest()}]}))\n",
            "artifact_dir_symlink": "outside=out.parent / 'outside-artifacts'\noutside.mkdir()\nout.rmdir()\nout.symlink_to(outside, target_is_directory=True)\npayload=json.dumps({'schema':'apxinf.oracle-tokens.v1','output_token_ids':[7],'decoded_text':'x'}).encode()+b'\\n'\n(out / 'output-tokens.json').write_bytes(payload)\n(out / 'artifact-report.json').write_text(json.dumps({'schema':'apxinf.oracle-artifact-report.v1','artifacts':[{'file':'output-tokens.json','schema_ref':'tokens','dtype':'json','shape':[1],'sha256':hashlib.sha256(payload).hexdigest()}]}))\n",
            "nonzero": "raise SystemExit(7)\n",
        }
        for name, body in cases.items():
            with self.subTest(name=name):
                bundle = self._make_bundle()
                runner = self._write_runner(body)
                with self.assertRaises((ValueError, RuntimeError)):
                    oracle.run_runner(bundle, [sys.executable, str(runner)], [])

    def test_runner_rejects_early_non_eos_and_model_metadata_drift(self) -> None:
        bundle = self._make_bundle(max_new_tokens=2)
        runner = self._write_runner(
            "payload=json.dumps({'schema':'apxinf.oracle-tokens.v1','output_token_ids':[7],'decoded_text':'x'}).encode()+b'\\n'\n"
            "(out / 'output-tokens.json').write_bytes(payload)\n"
            "(out / 'artifact-report.json').write_text(json.dumps({'schema':'apxinf.oracle-artifact-report.v1','generation':{'completion_tokens':1,'stop_reason':'budget'},'artifacts':[{'file':'output-tokens.json','schema_ref':'tokens','dtype':'json','shape':[1],'sha256':hashlib.sha256(payload).hexdigest()}]}))\n"
        )
        with self.assertRaisesRegex(ValueError, "budget or end with EOS"):
            oracle.run_runner(bundle, [sys.executable, str(runner)], [])

        bundle = self._make_bundle()
        config = json.loads((self.model_dir / "config.json").read_text())
        config["text_config"]["vocab_size"] = 248044
        write_json(self.model_dir / "config.json", config)
        with self.assertRaisesRegex(ValueError, "metadata SHA256"):
            oracle.run_runner(bundle, [sys.executable, str(runner)], [])


if __name__ == "__main__":
    unittest.main()
