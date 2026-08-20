"""AutoPolicy dispatch, the policy registry, and the import boundary.

All offline except two gated checks that need the ``apxinf_py`` binding.
"""

from __future__ import annotations

import sys

import pytest


def test_import_apxinf_does_not_pull_binding():
    """Importing the package (processor/offline use) must not import apxinf_py.

    Meaningful on machines where apxinf_py IS installed (e.g. Thor): it proves
    ``import apxinf`` stays CUDA-free.
    """
    import apxinf  # noqa: F401

    assert "apxinf_py" not in sys.modules


def test_registry_registers_pi05():
    from apxinf import Pi05Policy
    from apxinf.policies import available_policies, get_policy

    assert "pi05" in available_policies()
    assert get_policy("pi05") is Pi05Policy
    assert get_policy("PI05") is Pi05Policy  # case-insensitive


def test_registry_rejects_conflicting_reregister():
    from apxinf.policies import register_policy

    @register_policy("conflict_probe")
    class _A:
        pass

    # Same key, different class -> error; same class again -> no-op.
    register_policy("conflict_probe")(_A)  # idempotent
    with pytest.raises(ValueError):

        @register_policy("conflict_probe")
        class _B:
            pass


def test_autopolicy_not_instantiable():
    from apxinf import AutoPolicy

    with pytest.raises(TypeError):
        AutoPolicy()


def test_autopolicy_dispatches_by_config_type(tmp_path):
    from apxinf import AutoPolicy
    from apxinf.policies import register_policy

    sentinel = object()

    @register_policy("stub_dispatch")
    class _StubPolicy:
        @classmethod
        def from_pretrained(cls, model_dir, **kwargs):
            return sentinel, model_dir, kwargs

    (tmp_path / "config.json").write_text('{"type": "stub_dispatch"}')
    got, model_dir, kwargs = AutoPolicy.from_pretrained(tmp_path, precision="bf16")
    assert got is sentinel
    assert kwargs == {"precision": "bf16"}


def test_autopolicy_model_type_override_skips_config(tmp_path):
    from apxinf import AutoPolicy
    from apxinf.policies import register_policy

    @register_policy("stub_override")
    class _StubPolicy:
        @classmethod
        def from_pretrained(cls, model_dir, **kwargs):
            return "ok"

    # No config.json present; explicit model_type must still work.
    assert AutoPolicy.from_pretrained(tmp_path, model_type="stub_override") == "ok"


def test_autopolicy_missing_config_raises(tmp_path):
    from apxinf import AutoPolicy

    with pytest.raises(FileNotFoundError):
        AutoPolicy.from_pretrained(tmp_path)


def test_autopolicy_unknown_type_raises(tmp_path):
    from apxinf import AutoPolicy

    (tmp_path / "config.json").write_text('{"type": "definitely-not-registered"}')
    with pytest.raises(KeyError):
        AutoPolicy.from_pretrained(tmp_path)


def test_model_reexport_matches_binding():
    apxinf_py = pytest.importorskip("apxinf_py")
    import apxinf

    assert apxinf.Model is apxinf_py.Model
