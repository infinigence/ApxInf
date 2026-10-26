"""Benchmark measurements preserve their recorded-frame and synthetic paths."""
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


def test_ar_fit_uses_distinct_observed_frame_lengths(monkeypatch):
    bench = load('bench_pi0_fast')
    clock = [0.0]
    observed = []

    def call(steps):
        observed.append(steps)
        clock[0] += (30 + 10 * steps) / 1000
        return np.arange(steps, dtype=np.uint32)

    monkeypatch.setattr(bench.time, 'perf_counter', lambda: clock[0])
    result = bench._run_ar(call, [2, 4, 4], survey=3, repeats=3)
    assert observed == [2, 4, 4, 2, 2, 2, 4, 4, 4]
    assert result['fit']['fixed_ms'] == pytest.approx(30)
    assert result['fit']['per_step_ms'] == pytest.approx(10)
    assert [point['steps'] for point in result['fit']['points']] == [2, 4]


def test_ar_fit_with_one_decode_length_has_no_split():
    bench = load('bench_pi0_fast')
    result = bench._run_ar(lambda _: np.ones(8, dtype=np.uint32), [0, 1], 2, 1)
    assert result['fit'] is None


def test_recorded_frames_remain_optional(tmp_path):
    bench = load('bench_pi0_fast')
    path = tmp_path / 'frames.npz'
    images = np.zeros((2, 2, 2, 3), dtype=np.uint8)
    states = np.zeros((2, 8), dtype=np.float32)
    np.savez(path, base_raw=images, wrist_raw=images, state=states, task=['a', 'b'])
    frames = bench._load_frames(path, 'raw', 1)
    assert len(frames) == 1
    assert frames[0]['state'].shape == (8,)
    assert frames[0]['task'] == 'a'


def test_warmup_matches_original_input_count_limit():
    bench = load('bench_pi0_fast')
    calls = []
    bench._time_stream(lambda item: calls.append(item) or [1], ['a'], 3, 10)
    assert calls == ['a'] * 4


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


@pytest.mark.parametrize('variant', ['bf16', 'fp8_static', 'int8_dynamic'])
def test_gr00t_cli_passes_variant_unchanged_to_rust(tmp_path, monkeypatch, variant):
    bench = load('bench_gr00t')
    output = tmp_path / 'report.json'
    binary = tmp_path / 'gr00t_bench'
    monkeypatch.setattr(sys, 'argv', ['bench_gr00t', '--model-dir', str(tmp_path),
                        '--model-variant', variant, '--binary', str(binary), '--out', str(output)])
    def run(command, **kwargs):
        assert command[3] == variant
        output.write_text('{}')
    monkeypatch.setattr(bench.subprocess, 'run', run)
    monkeypatch.setattr(bench, 'provenance', lambda *a, **kw: {})
    bench.main()


def test_fixed_decode_times_requested_prefixes_and_interleaves(monkeypatch):
    bench = load('bench_pi0_fast')
    clock = [0.0]
    calls = []
    def call(payload, length):
        assert payload == 'same-input'
        calls.append(length)
        clock[0] += (25 + 7 * length) / 1000
        return np.arange(length, dtype=np.uint32)
    monkeypatch.setattr(bench.time, 'perf_counter', lambda: clock[0])
    result = bench._run_fixed(call, 'same-input', [12, 20, 40], samples=2, warmup=1)
    assert calls == [40, 12, 20, 40, 12, 20, 40, 40, 20, 12]
    assert result['fit']['fixed_ms'] == pytest.approx(25)
    assert result['fit']['per_step_ms'] == pytest.approx(7)
    assert result['prefix_equal'] is True
    assert all(len(values) == 2 for values in result['raw_ms'].values())


@pytest.mark.parametrize('failure', ['length', 'prefix'])
def test_fixed_decode_rejects_early_stop_or_changed_tokens(failure):
    bench = load('bench_pi0_fast')
    def call(_, length):
        if length == 40:
            return np.arange(length, dtype=np.uint32)
        return np.arange(length - (failure == 'length'), dtype=np.uint32) + (failure == 'prefix')
    with pytest.raises(RuntimeError, match='requested|changed'):
        bench._run_fixed(call, None, [12, 40], samples=1, warmup=0)


@pytest.mark.parametrize('lengths', [['0','12'], ['12','12']])
def test_fixed_decode_rejects_invalid_cli_counts(monkeypatch, lengths):
    bench = load('bench_pi0_fast')
    monkeypatch.setattr(sys, 'argv', ['bench', '--model-dir', '/unused', '--layer', 'l1',
                                   '--mode', 'fixed', '--decode-lengths', *lengths])
    with pytest.raises(SystemExit):
        bench.parse_args()
