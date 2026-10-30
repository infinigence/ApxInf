"""CPU contract checks for the documented benchmark, with only assets/GPU stubbed."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    from apxinf.policies.impls import walloss
    (tmp_path / 'config.json').write_text(json.dumps({'model_type': 'walloss'}))
    calls = []

    class Runner:
        action_horizon, action_dim, num_views, image_size, patch_size = 10, 26, 2, 252, 14
        accepts_rgb_u8 = True

        @staticmethod
        def load(model, path, *, device, model_variant, autotune, sampling_seed, tactics=None):
            calls.append(('load', model_variant, autotune, tactics))
            return Runner()

        def infer_rgb(self, rgb_u8, layout, token_ids, noise=None, action_mask=None,
                      state=None, image_grid_thw=None):
            assert layout == 'nhwc'
            assert rgb_u8.shape == (2, 336, 448, 3)
            np.testing.assert_array_equal(image_grid_thw, [[1, 24, 32]] * 2)
            assert action_mask.shape == noise.shape == (10, 26)
            calls.append(('infer',))
            return np.zeros((10, 26), np.float32)

    class Processor:
        image_keys = ('observation/image', 'observation/wrist_image')
        state_key, prompt_key, state_bins = 'observation/state', 'prompt', 256

        def __call__(self, observation):
            assert all(key in observation for key in (*self.image_keys, self.state_key, self.prompt_key))
            return (np.zeros((2, 336, 448, 3), np.uint8), np.arange(400, dtype=np.uint32),
                    np.ones((10, 26), np.float32))

    monkeypatch.setitem(sys.modules, 'apxinf_py', SimpleNamespace(ModelRunner=Runner))
    monkeypatch.setattr(walloss, '_checkpoint_state_bins', lambda _: 256)
    monkeypatch.setattr(walloss, '_load_normalizer', lambda *a: (np.zeros(26), np.ones(26)))
    monkeypatch.setattr(walloss, '_WallossProcessor', lambda *a, **kw: Processor())
    return tmp_path, calls


@pytest.mark.parametrize('variant,precision', [('bf16', 'bf16'), ('fp8_dynamic', 'fp8')])
@pytest.mark.parametrize('layer', ['l1', 'l2', 'l1,l2'])
def test_documented_benchmark_real_policy_dispatch(checkpoint, monkeypatch, variant, precision, layer):
    path, calls = checkpoint
    output = path / 'report.json'
    monkeypatch.setattr(sys, 'argv', ['bench_pi05.py', '--model-dir', str(path),
                        '--model-variant', variant, '--layer', layer, '--warmup', '0',
                        '--samples', '1', '--out', str(output), '--autotune',
                        '--tactics', str(path / 'tactics.json')])
    load_script('bench_pi05').main()
    assert calls[0] == ('load', variant, True, str(path / 'tactics.json'))
    assert sum(call[0] == 'infer' for call in calls) == len(layer.split(','))
    assert json.loads(output.read_text())['layers'] == layer.split(',')


def test_variant_rejects_legacy_precision(checkpoint):
    from apxinf import AutoPolicy
    path, calls = checkpoint
    with pytest.raises(TypeError, match='precision'):
        AutoPolicy.from_pretrained(path, precision='bf16')
    with pytest.raises(ValueError, match='model_variant'):
        AutoPolicy.from_pretrained(path, model_variant='fp8_static')
    assert not calls


def test_walloss_cli_variant(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['bench_walloss.py', '--model-dir', '/checkpoint', '--model-variant', 'bf16'])
    assert load_script('bench_walloss').parse_args().model_variant == 'bf16'
    monkeypatch.setattr(sys, 'argv', ['bench_walloss.py', '--model-dir', '/checkpoint', '--precision', 'bf16'])
    with pytest.raises(SystemExit):
        load_script('bench_walloss').parse_args()


def test_benchmark_scripts_compile():
    # Compile without importing GPU packages or writing __pycache__ artifacts.
    for path in sorted((ROOT / 'scripts').glob('bench_*.py')):
        compile(path.read_text(), str(path), 'exec')


def test_rgb_keyword_contract_matches_native_binding(checkpoint):
    import ast
    import inspect
    import re
    from apxinf import AutoPolicy
    path, _ = checkpoint
    policy = AutoPolicy.from_pretrained(path, model_variant='bf16')
    observation = {key: np.zeros((256, 256, 3), np.uint8) for key in policy.processor.image_keys}
    observation.update({'observation/state': np.zeros(26), 'prompt': 'pick up the block'})
    inputs = policy.prepare_rgb_inputs(observation, noise=np.zeros((10, 26)))
    # Read the actual PyO3 declaration so changing native keyword names breaks CI,
    # even if a test runner's Python signature accidentally remains permissive.
    binding = (ROOT / 'crates/apxinf-py/src/lib.rs').read_text()
    declaration = binding[:binding.index("    fn infer_rgb<'py>")]
    args = re.findall(r'#\[pyo3\(signature = \(([^\n]*)\)\)\]', declaration)[-1]
    namespace = {}
    exec(compile(ast.parse(f'def native({args}): pass'), '<native signature>', 'exec'), namespace)
    inspect.signature(namespace['native']).bind(**inputs)


@pytest.mark.parametrize('script,variant,extra', [
    ('bench_qwen_drive', 'bf16', []),
    ('bench_pi05', 'bf16', []), ('bench_pi05', 'fp8_static', []),
    ('bench_pi05', 'fp8_dynamic', []), ('bench_pi05', 'int8_dynamic', []),
    ('bench_walloss', 'bf16', []), ('bench_walloss', 'fp8_dynamic', []),
    ('bench_smolvla', 'bf16', []), ('bench_smolvla', 'fp16', []),
    ('bench_gr00t', 'bf16', []), ('bench_gr00t', 'fp8_static', []),
    ('bench_gr00t', 'int8_dynamic', []),
    ('bench_pi0_fast', 'bf16', ['--state-key', 'observation/state']),
    ('bench_pi0_fast', 'fp8_static', ['--state-key', 'observation/state']),
])
def test_benchmark_cli_variant_matrix(monkeypatch, script, variant, extra):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    monkeypatch.setattr(sys, 'argv', [script, '--model-dir', '/checkpoint', '--model-variant', variant, *extra])
    assert load_script(script).parse_args().model_variant == variant
