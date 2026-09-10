"""Exercise the evaluator CLI-to-checkpoint path without CUDA or a simulator."""

import json
import sys
from types import SimpleNamespace

import pytest

from apxinf import AutoPolicy
from apxinf.checkpoints import detect_checkpoint
from scripts import eval_libero


def parse(monkeypatch, tmp_path, *extra, backend="in-process"):
    monkeypatch.setattr(sys, "argv", [
        "eval_libero.py", "--backend", backend, "--precision", "bf16",
        "--model-dir", str(tmp_path),
        "--results-jsonl", str(tmp_path / "results.jsonl"),
        "--summary-json", str(tmp_path / "summary.json"), *extra,
    ])
    return eval_libero.parse_args()


def test_explicit_norm_stats_reaches_checkpoint_loader(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"type": "pi05"}))
    stats = tmp_path / "external-norms.json"
    stats.write_text(json.dumps({"norm_stats": {
        "state": {"q01": [0.0] * 7, "q99": [1.0] * 7},
        "actions": {"q01": [2.0] * 7, "q99": [4.0] * 7},
    }}))
    loaded = []

    def load(model_dir, **options):
        loaded.append(detect_checkpoint(model_dir, norm_stats=options["norm_stats"]))
        return SimpleNamespace(metadata={})

    monkeypatch.setattr(AutoPolicy, "from_pretrained", load)
    args = parse(monkeypatch, tmp_path, "--norm-stats", str(stats))
    eval_libero.InProcessBackend(args, eval_libero.resolve_wire_keys(args))
    assert loaded[0].norm_stats == stats
    assert loaded[0].normalization.action.values["q01"] == (2.0,) * 7


def test_omitted_norm_stats_preserves_checkpoint_defaults(monkeypatch, tmp_path):
    options = {}

    def load(model_dir, **kwargs):
        options.update(kwargs)
        return SimpleNamespace(metadata={})

    monkeypatch.setattr(AutoPolicy, "from_pretrained", load)
    args = parse(monkeypatch, tmp_path)
    eval_libero.InProcessBackend(args, eval_libero.resolve_wire_keys(args))
    assert "norm_stats" not in options


def test_websocket_norm_stats_is_rejected(monkeypatch, tmp_path, capsys):
    stats = tmp_path / "norm_stats.json"
    stats.write_text("{}")
    with pytest.raises(SystemExit) as exc:
        parse(monkeypatch, tmp_path, "--norm-stats", str(stats), backend="websocket")
    assert exc.value.code == 2
    assert "pass it to pi05_openpi_websocket_server.py" in capsys.readouterr().err


def test_missing_norm_stats_is_rejected_before_rollout(monkeypatch, tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        parse(monkeypatch, tmp_path, "--norm-stats", str(tmp_path / "missing.json"))
    assert exc.value.code == 2
    assert "--norm-stats must name an existing file" in capsys.readouterr().err
