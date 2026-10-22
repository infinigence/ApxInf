"""Every public VLA loader forwards model_variant unchanged to the native seam."""
import ast
import inspect
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import pytest

CASES = [
    ('pi05', 'bf16'), ('pi05', 'fp8_static'), ('pi05', 'int8_dynamic'),
    ('walloss', 'bf16'), ('walloss', 'fp8_dynamic'),
    ('smolvla', 'bf16'), ('smolvla', 'fp16'),
    ('qwen_drive', 'bf16'),
    ('gr00t', 'bf16'), ('gr00t', 'fp8_static'), ('gr00t', 'int8_dynamic'),
    ('pi0_fast', 'bf16'), ('pi0_fast', 'fp8_static'),
]


@pytest.mark.parametrize('family,variant', CASES)
def test_public_variant_reaches_native_unchanged(tmp_path, monkeypatch, family, variant):
    from apxinf import AutoPolicy
    from apxinf.policies import get_policy
    from apxinf.policies.impls import gr00t, pi05, walloss
    (tmp_path / 'config.json').write_text(json.dumps({'model_type': family}))
    monkeypatch.setattr(pi05, 'has_layout_metadata', lambda _: False)
    monkeypatch.setattr(pi05, 'resolve_pi05_tactics', lambda *a, **kw: None)
    monkeypatch.setattr(walloss, '_checkpoint_state_bins', lambda _: 256)
    monkeypatch.setattr(gr00t, 'resolve_assets', lambda *a: tmp_path)
    monkeypatch.setattr(gr00t._NvidiaProcessorAdapter, 'load', lambda *a, **kw:
                        SimpleNamespace(action_dim=7, action_horizon=16))
    # Use the actual native signature, not a **kwargs mock that hides API drift.
    root = Path(__file__).resolve().parents[3]
    source = (root / 'crates/apxinf-py/src/lib.rs').read_text()
    args = next(args for args in re.findall(r'#\[pyo3\(signature = \(([^\n]*)\)\)\]', source)
                if args.startswith('model, path,'))
    namespace = {}
    exec(compile(ast.parse(f'def native({args.replace("false", "False")}): pass'), '<native load>', 'exec'), namespace)
    signature = inspect.signature(namespace['native'])
    captured = {}

    class ReachedNative(Exception):
        pass

    def load(*args, **kwargs):
        signature.bind(*args, **kwargs)
        captured.update(kwargs)
        raise ReachedNative

    monkeypatch.setitem(sys.modules, 'apxinf_py', SimpleNamespace(ModelRunner=SimpleNamespace(load=load)))
    options = {'calibration': tmp_path / 'calibration.json'} if family == 'gr00t' and variant == 'fp8_static' else {}
    with pytest.raises(ReachedNative):
        AutoPolicy.from_pretrained(tmp_path, model_variant=variant, **options)
    assert captured['model_variant'] == variant
    assert 'precision' not in captured
    parameters = inspect.signature(get_policy(family).from_pretrained).parameters
    assert 'model_variant' in parameters
    assert 'precision' not in parameters


@pytest.mark.parametrize('family', ['pi05', 'walloss', 'smolvla', 'qwen_drive', 'gr00t', 'pi0_fast'])
def test_public_precision_is_rejected_before_loading(tmp_path, family):
    from apxinf import AutoPolicy
    with pytest.raises(TypeError, match='precision'):
        AutoPolicy.from_pretrained(tmp_path, model_type=family, precision='bf16')
