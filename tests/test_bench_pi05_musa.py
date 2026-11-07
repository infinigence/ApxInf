"""The MUSA benchmark scores the call it times, and says so when it refuses.

What a device-free test can hold this tool to is the wiring: which engine call
the gate reads, what the load is pinned to, what a run without a bundle reports,
and that every refusal is a refusal and not a score. The numbers themselves need
the card; these do not.
"""

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
for package in ("python/apxinf", "python/apxinf_ref"):
    if str(ROOT / package) not in sys.path:
        sys.path.insert(0, str(ROOT / package))

from apxinf import _bench  # noqa: E402
from pi05.torch_gold import SCHEMA, artifact  # noqa: E402


def _load_script():
    path = ROOT / "scripts" / "bench_pi05_musa.py"
    spec = importlib.util.spec_from_file_location("bench_pi05_musa_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BENCH = _load_script()


def _bundle(tmp_path, *, steps=4, views=2, tokens=4, horizon=2, action_dim=32, action=None):
    """A miniature but structurally real bundle, plus the checkpoint it names."""

    fixture = artifact.make_fixture(seed=1, views=views, token_count=tokens, horizon=horizon,
                                    action_dim=action_dim, num_flow_steps=steps)
    expected = (np.linspace(-1.0, 1.0, horizon * action_dim, dtype=np.float32)
                .reshape(horizon, action_dim) if action is None else np.asarray(action, np.float32))
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    (checkpoint / "config.json").write_text("{}\n")
    reference = tmp_path / "bundle"
    reference.mkdir()
    np.savez(reference / "inputs.npz", **fixture)
    np.savez(reference / "debug.npz", raw_actions=np.ascontiguousarray(expected))
    manifest = {
        "schema": SCHEMA,
        "implementation": "unit-test",
        "inputs": artifact.describe_inputs(fixture),
        "model_config": {"action_horizon": horizon, "action_dim": action_dim,
                         "num_views": views},
        "checkpoint": {"sha256": artifact.sha256(checkpoint / "model.safetensors")},
        "debug": {"raw_actions": {"dtype": "torch.float32", "encoding": artifact.NATIVE}},
    }
    (reference / "manifest.json").write_text(json.dumps(manifest))
    return reference, checkpoint, expected, manifest


class _Handle:
    """A ModelRunner stand-in: the four getters the script reads and two entry points."""

    def __init__(self, expected, *, execution_mode="eager", patch_size=16):
        self.expected = expected
        self.execution_mode = execution_mode
        self.patch_size = patch_size
        self.action_horizon, self.action_dim = expected.shape
        self.num_views, self.image_size, self.patches_per_view = 2, 224, (224 // patch_size) ** 2
        self.max_token_len = 200
        self.calls = []

    def _infer_patches(self, patches, token_ids, noise=None):
        self.calls.append(("patches", patches.shape, token_ids.shape, noise.shape))
        return self.expected

    def infer_rgb(self, rgb_u8, layout, token_ids, noise=None):
        self.calls.append(("rgb", rgb_u8.shape, rgb_u8.dtype, layout))
        return self.expected


class _Runner:
    """Stands in for `apxinf_py.ModelRunner`, recording how it was asked to load."""

    def __init__(self, handle):
        self.handle = handle
        self.load_kwargs = None

    def load(self, model, path, **kwargs):
        self.load_kwargs = {"model": model, "path": path, **kwargs}
        # The real loader applies its pins to the handle, so the stand-in must
        # too -- otherwise a shape derived from a flag would be invisible here.
        self.handle.num_views = kwargs.get("num_views", self.handle.num_views)
        return self.handle


@pytest.fixture
def engine(monkeypatch, tmp_path):
    """Install a fake binding and hand back a factory for one bundled run."""

    def install(*, action=None, execution_mode="eager", **fixture_options):
        reference, checkpoint, expected, manifest = _bundle(
            tmp_path, action=action, **fixture_options
        )
        handle = _Handle(expected, execution_mode=execution_mode)
        runner = _Runner(handle)
        module = types.ModuleType("apxinf_py")
        module.ModelRunner = runner
        monkeypatch.setitem(sys.modules, "apxinf_py", module)
        return reference, checkpoint, handle, runner

    return install


def _run(monkeypatch, capsys, *argv):
    """Run the CLI once and return (exit code or None, stdout, stderr, report)."""

    monkeypatch.setattr(sys, "argv", ["bench_pi05_musa.py", *argv])
    out = Path(argv[argv.index("--out") + 1]) if "--out" in argv else None
    code = None
    try:
        code = BENCH.main()
    except SystemExit as exit_status:
        code = exit_status.code
    captured = capsys.readouterr()
    report = json.loads(out.read_text()) if out is not None and out.is_file() else None
    return code, captured.out, captured.err, report


def test_the_report_arithmetic_matches_the_upstream_benchmark():
    """The v2 schema is a by-value contract; a second implementation has to agree."""

    upstream = importlib.util.spec_from_file_location(
        "bench_pi05_under_test", ROOT / "scripts" / "bench_pi05.py"
    )
    other = importlib.util.module_from_spec(upstream)
    upstream.loader.exec_module(other)
    generator = np.random.default_rng(0)
    for count in (1, 5, 30, 7):
        samples = [float(value) for value in generator.uniform(400.0, 600.0, count)]
        assert _bench.stats(samples) == other._stats(samples)
    for spec in ("all", "l0", "l2,l0", "l0,l1,l1", "l3"):
        assert _bench.parse_layers(spec) == other._parse_layers(spec)
    with pytest.raises(SystemExit):
        _bench.parse_layers("l7")


def test_l3_times_the_split_deployment_client():
    """L3's four segments are the repository client's breakdown, not a second copy."""

    policy, timing_segments = BENCH._remote_client()
    assert policy.__module__.endswith("pi05_remote_client")
    segments = timing_segments(
        {"policy_timing": {"infer_ms": 1.5, "policy_ms": 2.5},
         "server_timing": {"infer_ms": 3.5}},
        4.5,
    )
    assert tuple(segments) == BENCH.L3_SEGMENTS
    assert segments == {
        "round_trip": 4.5,
        "model": 1.5,
        "server_processor": 1.0,
        "transport": 1.0,
    }


def test_the_gate_reads_the_call_the_benchmark_timed(engine, monkeypatch, capsys, tmp_path):
    reference, checkpoint, handle, runner = engine()
    out = tmp_path / "report.json"

    code, stdout, _, report = _run(
        monkeypatch, capsys, "--layer", "l0", "--warmup", "1", "--samples", "3",
        "--reference", str(reference), "--checkpoint", str(checkpoint), "--out", str(out),
    )

    assert code == 0, stdout
    # Pinned from the bundle, not from the checkpoint: the schedule and the view
    # count are what makes the engine's computation the one the bundle recorded.
    assert runner.load_kwargs["num_flow_steps"] == 4
    assert runner.load_kwargs["flow_start_time"] == 1.0
    assert runner.load_kwargs["num_views"] == 2
    assert runner.load_kwargs["execution"] == "prefer-graph"
    assert runner.load_kwargs["tactics"] is None and runner.load_kwargs["autotune"] is False
    assert [call[0] for call in handle.calls] == ["patches"] * 4  # 1 warm-up + 3 samples
    assert handle.calls[0][1] == (2 * 14 * 14, 3 * 16 * 16)
    assert report["accuracy"]["cosine"] == pytest.approx(1.0)
    assert report["accuracy"]["timed"] is True and report["accuracy"]["passed"] is True
    assert report["layers_ms"]["L0_model"]["samples"] == 3
    assert list(report["layers_ms"]["L0_model"]) == ["p50", "p95", "min", "max", "mean", "std",
                                                     "samples"]
    # The fake engine only ever reports eager, so the default prefer-graph
    # request is a recorded fallback here.
    assert report["execution"] == {"requested": "prefer-graph", "layers": {"l0": "eager"},
                                   "fallback": True}


def test_a_divergent_action_fails_the_gate(engine, monkeypatch, capsys, tmp_path):
    reference, checkpoint, handle, runner = engine()
    handle.expected = -np.asarray(handle.expected)  # same magnitude, opposite direction

    code, stdout, _, report = _run(
        monkeypatch, capsys, "--layer", "l0", "--warmup", "0", "--samples", "2",
        "--reference", str(reference), "--checkpoint", str(checkpoint),
        "--out", str(tmp_path / "report.json"),
    )

    assert code == 1, stdout
    assert report["accuracy"]["cosine"] == pytest.approx(-1.0)
    assert report["accuracy"]["band"] == "DIVERGENT"
    assert report["accuracy"]["passed"] is False


def test_a_bundle_the_engine_would_integrate_differently_is_refused(engine, monkeypatch, capsys,
                                                                    tmp_path):
    reference, checkpoint, _, _ = engine()
    manifest = json.loads((reference / "manifest.json").read_text())
    document = dict(np.load(reference / "inputs.npz"))
    document["dt"] = np.asarray(-0.5, dtype=np.float32)  # four steps of -1/4, not -1/2
    np.savez(reference / "inputs.npz", **document)
    manifest["inputs"] = artifact.describe_inputs(document)
    (reference / "manifest.json").write_text(json.dumps(manifest))

    code, _, stderr, _ = _run(
        monkeypatch, capsys, "--layer", "l0", "--reference", str(reference),
        "--checkpoint", str(checkpoint),
    )

    assert code == 2 and "`dt`" in stderr


def test_a_different_checkpoint_is_refused_unless_it_is_named(engine, monkeypatch, capsys, tmp_path):
    reference, checkpoint, _, _ = engine()
    (checkpoint / "model.safetensors").write_bytes(b"other weights")

    refused, _, stderr, _ = _run(
        monkeypatch, capsys, "--layer", "l0", "--warmup", "0", "--samples", "1",
        "--reference", str(reference), "--checkpoint", str(checkpoint),
    )
    assert refused == 2 and "sha256" in stderr

    allowed, _, _, report = _run(
        monkeypatch, capsys, "--layer", "l0", "--warmup", "0", "--samples", "1",
        "--reference", str(reference), "--checkpoint", str(checkpoint),
        "--allow-checkpoint-mismatch", "--out", str(tmp_path / "report.json"),
    )
    assert allowed == 0
    assert report["checkpoint"]["matches_reference"] is False


def test_an_execution_policy_the_engine_did_not_take_is_refused(engine, monkeypatch, capsys):
    reference, checkpoint, _, _ = engine(execution_mode="eager")

    code, _, stderr, _ = _run(
        monkeypatch, capsys, "--layer", "l0", "--warmup", "0", "--samples", "1",
        "--execution", "require-graph", "--reference", str(reference),
        "--checkpoint", str(checkpoint),
    )

    assert code == 2 and "ran as 'eager'" in stderr


def test_l2_wraps_the_handle_already_loaded(engine, monkeypatch, capsys, tmp_path):
    import apxinf

    reference, checkpoint, handle, runner = engine()
    seen = {}

    class _Policy:
        image_keys = ("observation/image", "observation/wrist_image")
        prompt_key = "prompt"
        action_dim = 7

        def infer(self, observation):
            seen["observation"] = observation
            return {"actions": np.zeros((2, 7), dtype=np.float32)}

        def close(self):
            seen["closed"] = True

    def build(model_dir, **options):
        seen["build"] = {"model_dir": Path(model_dir), **options}
        return _Policy()

    monkeypatch.setattr(apxinf.Pi05Policy, "from_pretrained", build)
    out = tmp_path / "report.json"

    code, stdout, _, report = _run(
        monkeypatch, capsys, "--layer", "l2", "--warmup", "0", "--samples", "1",
        "--reference", str(reference), "--checkpoint", str(checkpoint),
        "--tokenizer", "/tokenizer.model", "--norm-stats", "/norm_stats.json",
        "--action-dim", "7", "--out", str(out),
    )

    assert code == 0, stdout
    assert seen["build"]["model_runner"] is handle
    assert seen["build"]["tokenizer_path"] == Path("/tokenizer.model")
    assert seen["build"]["norm_stats"] == "/norm_stats.json"
    assert seen["build"]["action_dim"] == 7
    assert set(seen["observation"]) == {*_Policy.image_keys, "prompt"}
    assert seen["closed"] is True
    # L2 alone still states the gate, from one extra L0 call rather than a timed one.
    assert [call[0] for call in handle.calls] == ["patches"]
    assert report["accuracy"]["timed"] is False
    # No L0 run measured repeats, so the report says "not measured" rather than
    # claiming a determinism check it did not make.
    assert report["accuracy"]["deterministic"] is None
    assert report["accuracy"]["cosine"] == pytest.approx(1.0)
    assert report["workload"]["deploy_action_dim"] == 7


def test_without_a_bundle_the_run_measures_and_says_so(engine, monkeypatch, capsys, tmp_path):
    """No bundle is a latency run, not a refusal: synthetic inputs, nothing scored."""

    _, checkpoint, handle, runner = engine()
    out = tmp_path / "report.json"

    code, stdout, _, report = _run(
        monkeypatch, capsys, "--layer", "l0", "--warmup", "1", "--samples", "2",
        "--checkpoint", str(checkpoint), "--out", str(out),
    )

    assert code == 0, stdout
    assert report["accuracy"] == {"scored": False, "reason": "no --reference"}
    assert "not scored (no --reference)" in stdout
    assert "--reference <gold-bundle>" in stdout
    # The checkpoint declares neither a view count nor a token count, so both are
    # this tool's defaults, and the solver schedule is a loader default.
    assert report["workload"]["source"] == "synthetic"
    assert report["workload"]["num_views"] == 2
    assert report["workload"]["token_count"] == 10
    assert report["workload"]["num_flow_steps"] is None
    assert report["workload"]["flow_start_time"] is None
    assert runner.load_kwargs["num_views"] == 2
    assert "num_flow_steps" not in runner.load_kwargs
    assert "flow_start_time" not in runner.load_kwargs
    assert [call[0] for call in handle.calls] == ["patches"] * 3  # 1 warm-up + 2 samples
    # Which mode the timed calls ran in is part of a latency number, scored or not.
    assert "execution: prefer-graph -> l0=eager" in stdout


def test_a_synthetic_run_takes_its_shape_from_the_flags(engine, monkeypatch, capsys, tmp_path):
    _, checkpoint, handle, runner = engine()

    code, stdout, _, report = _run(
        monkeypatch, capsys, "--layer", "l0,l1", "--warmup", "0", "--samples", "1",
        "--views", "1", "--token-count", "5", "--checkpoint", str(checkpoint),
        "--out", str(tmp_path / "report.json"),
    )

    assert code == 0, stdout
    assert runner.load_kwargs["num_views"] == 1
    assert report["workload"]["token_count"] == 5
    assert [call[0] for call in handle.calls] == ["patches", "rgb"]
    assert handle.calls[0][2] == (5,)  # the prefix the layers were handed
    # One camera, one camera's patches: the flag reaches the arrays, not just the load.
    assert handle.calls[0][1] == (1 * 14 * 14, 3 * 16 * 16)


def test_synthetic_shape_flags_are_validated(engine, monkeypatch, capsys):
    _, checkpoint, _, _ = engine()

    for flag in ("--views", "--token-count"):
        code, _, stderr, _ = _run(
            monkeypatch, capsys, "--layer", "l0", flag, "0", "--checkpoint", str(checkpoint),
        )
        assert code == 2 and flag in stderr


def test_token_count_is_refused_when_l2_will_re_tokenize(engine, monkeypatch, capsys):
    """L2's prefix is the tokenizer's; one flag for it would make the delta meaningless."""

    _, checkpoint, _, _ = engine()

    code, _, stderr, _ = _run(
        monkeypatch, capsys, "--layer", "l0,l2", "--token-count", "5",
        "--checkpoint", str(checkpoint),
    )

    assert code == 2 and "L2 re-tokenizes" in stderr


def test_export_needs_a_reference(engine, monkeypatch, capsys, tmp_path):
    """Actions computed from zeros must not be written as if they were measured."""

    _, checkpoint, _, _ = engine()

    code, _, stderr, _ = _run(
        monkeypatch, capsys, "--layer", "l0", "--export", str(tmp_path / "actions.npz"),
        "--checkpoint", str(checkpoint),
    )

    assert code == 2 and "--export" in stderr


def test_synthetic_shape_flags_are_refused_with_a_bundle(engine, monkeypatch, capsys):
    reference, checkpoint, _, _ = engine()

    code, _, stderr, _ = _run(
        monkeypatch, capsys, "--layer", "l0", "--views", "2",
        "--reference", str(reference), "--checkpoint", str(checkpoint),
    )

    assert code == 2 and "--views/--token-count" in stderr


def test_stored_actions_need_a_reference(engine, monkeypatch, capsys, tmp_path):
    """`--engine-actions` is nothing but a score, so there has to be something to score."""

    _, checkpoint, _, _ = engine()
    stored = tmp_path / "actions.npz"
    np.savez(stored, actions=np.zeros((2, 32), dtype=np.float32))

    code, _, stderr, _ = _run(
        monkeypatch, capsys, "--engine-actions", str(stored), "--checkpoint", str(checkpoint),
    )

    assert code == 2 and "needs --reference" in stderr


def test_l3_alone_cannot_be_scored(engine, monkeypatch, capsys):
    reference, checkpoint, _, _ = engine()

    code, _, stderr, _ = _run(
        monkeypatch, capsys, "--layer", "l3",
        "--reference", str(reference), "--checkpoint", str(checkpoint),
    )

    assert code == 2 and "drop --reference" in stderr


def test_the_no_verify_flag_is_gone(engine, monkeypatch, capsys):
    """`--reference` is the switch now; the old escape hatch must not linger."""

    _, checkpoint, _, _ = engine()

    code, _, stderr, _ = _run(
        monkeypatch, capsys, "--layer", "l0", "--no-verify", "--checkpoint", str(checkpoint),
    )

    assert code == 2 and "unrecognized arguments" in stderr


def test_requantization_matches_the_remote_client():
    """The benchmark and the client must turn one fixture into the same bytes."""
    pytest.importorskip("websockets")
    pytest.importorskip("msgpack")
    spec = importlib.util.spec_from_file_location(
        "pi05_remote_client_under_test", ROOT / "scripts" / "pi05_remote_client.py")
    client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)

    rng = np.random.default_rng(0)
    images = rng.uniform(-1.0, 1.0, size=(2, 3, 8, 8)).astype(np.float32)
    images[0, 0, 0, 0] = -1.0
    images[0, 1, 0, 0] = 1.0
    images[1, 2, 3, 4] = 1.0
    assert np.array_equal(client._requantize_to_u8(images), BENCH._requantize_to_u8(images))
