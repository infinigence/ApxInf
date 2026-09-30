"""Benchmark measurements must exercise real stop points and deterministic input."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ar_fit_uses_identical_input_and_verified_prefixes(monkeypatch):
    bench = load('bench_pi0_fast')
    full = np.array([9, 1, 9, 2, 3, 4, 5, 6], dtype=np.uint32)
    clock = [0.0]
    observed = []
    payload = object()

    def call(observation, stop):
        assert observation is payload
        length = len(full) if stop is None else full.tolist().index(stop) + 1
        observed.append(length)
        clock[0] += (30 + 10 * length) / 1000
        return full[:length]

    monkeypatch.setattr(bench.time, 'perf_counter', lambda: clock[0])
    result = bench._run_ar(call, payload, repeats=3, warmup=2)
    assert len(set(observed)) >= 2
    assert result['fit']['fixed_ms'] == pytest.approx(30)
    assert result['fit']['per_step_ms'] == pytest.approx(10)
    assert all(len(samples) == 3 for samples in result['samples_ms'].values())


def test_ar_fit_rejects_a_changed_token_prefix():
    bench = load('bench_pi0_fast')

    def call(_, stop):
        return np.arange(8, dtype=np.uint32) if stop is None else np.array([99])

    with pytest.raises(RuntimeError, match='changed the generated token prefix'):
        bench._run_ar(call, None, repeats=1, warmup=0)


def test_ar_fit_rejects_unidentifiable_per_token_cost():
    bench = load('bench_pi0_fast')
    with pytest.raises(RuntimeError, match='fewer than two'):
        bench._run_ar(lambda *_: np.ones(8, dtype=np.uint32), None, 1, 0)


def test_warmup_is_not_limited_by_input_count():
    bench = load('bench_pi0_fast')
    calls = []
    bench._time_stream(lambda item: calls.append(item) or [1], ['a'], 3, 10)
    assert calls == ['a'] * 13


def test_drive_constructs_complete_deterministic_observation():
    bench = load('bench_qwen_drive')
    first, noise = bench.synthetic_observation(7)
    second, repeated = bench.synthetic_observation(7)
    assert len(first['views']) == 3
    for camera, frames in first['views'].items():
        assert len(frames) == 4
        for index, frame in enumerate(frames):
            assert frame['image'].shape == (900, 1600, 3)
            assert frame['image'].dtype == np.uint8
            assert np.array_equal(frame['image'], second['views'][camera][index]['image'])
    assert first['history'].shape == (16, 3)
    assert first['history_velocity'].shape == (16, 2)
    assert noise.shape == (1, 50, 3)
    assert np.array_equal(noise, repeated)


def test_checkpoint_local_tokenizers_do_not_need_environment(tmp_path, monkeypatch):
    from apxinf.policies.impls.pi0fast import _checkpoint_tokenizer, _resolve_asset
    local = tmp_path / 'assets/fast-tokenizer'
    local.mkdir(parents=True)
    (local / 'tokenizer.json').write_text('{}')
    monkeypatch.setenv('APXINF_FAST_TOKENIZER', '/missing/old-machine-path')
    selected = _checkpoint_tokenizer(tmp_path, None, 'remote/tokenizer', 'fast-tokenizer')
    assert _resolve_asset(selected, 'tokenizer.json', env='APXINF_FAST_TOKENIZER', what='FAST') == local / 'tokenizer.json'
    assert _checkpoint_tokenizer(tmp_path, '/explicit/tokenizer', 'remote/tokenizer', 'fast-tokenizer') == '/explicit/tokenizer'


def test_pi0fast_forwards_tactics_and_autotune_to_native(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from apxinf.policies.impls.pi0fast import Pi0FastPolicy
    (tmp_path / 'config.json').write_text('{}')
    captured = {}
    class Loaded(Exception):
        pass
    class Native:
        @staticmethod
        def load(*args, **kwargs):
            captured.update(kwargs)
            raise Loaded()
    monkeypatch.setitem(sys.modules, 'apxinf_py', SimpleNamespace(ModelRunner=Native))
    with pytest.raises(Loaded):
        Pi0FastPolicy.from_pretrained(tmp_path, tactics=tmp_path/'tactics.json', autotune=True,
                                     calibration=tmp_path/'calibration.json')
    assert captured['tactics'] == str(tmp_path/'tactics.json')
    assert captured['calibration'] == str(tmp_path/'calibration.json')
    assert captured['autotune'] is True
