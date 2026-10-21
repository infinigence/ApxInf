"""The benchmark must not label an eager fallback as a graph measurement."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize("eager_requested", [False, True])
def test_benchmark_checks_effective_mode_without_profile_flag(monkeypatch, tmp_path, eager_requested):
    path = Path(__file__).resolve().parents[1] / "scripts/bench_walloss.py"
    spec = importlib.util.spec_from_file_location("bench_walloss_under_test", path)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    import apxinf

    tokens = np.array([1], dtype=np.uint32)
    calls = []

    class Processor:
        resolutions = (252, 252)
        image_processor = SimpleNamespace(
            merge_size=2,
            resize=lambda *args: {"image_grid_thw": [[1, 18, 18]] * 2},
        )

        def __call__(self, observation):
            return np.zeros((2, 252, 252, 3), np.uint8), tokens, None

    def profile(*args, **kwargs):
        calls.append(1)
        return np.zeros((10, 7), np.float32), {"vision_ms": 1.0}  # Eager fallback.

    policy = SimpleNamespace(
        processor=Processor(), action_horizon=10,
        model_runner=SimpleNamespace(action_dim=7, patch_size=14, infer_rgb_profiled=profile),
        infer=lambda *args, **kwargs: {
            "normalized_actions": np.zeros((10, 7), np.float32), "token_ids": tokens,
        },
    )
    monkeypatch.setattr(apxinf, "WallossPolicy", SimpleNamespace(from_pretrained=lambda *a, **kw: policy))
    output = tmp_path / "report.json"
    argv = [str(path), "--model-dir", str(tmp_path), "--iterations", "1", "--warmup", "0", "--output", str(output)]
    if eager_requested:
        argv.append("--eager")
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setenv("APXINF_WALLOSS_NO_GRAPH", "1")
    if eager_requested:
        bench.main()
        report = json.loads(output.read_text())
        assert report["mode"] == "eager"
        assert "profile_ms" not in report
    else:
        with pytest.raises(RuntimeError, match="requested graph benchmark but runtime used eager"):
            bench.main()
        assert not output.exists()
    assert calls == [1]
