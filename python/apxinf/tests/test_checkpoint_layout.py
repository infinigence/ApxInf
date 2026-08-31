"""Checkpoint layout detection: which format, and which files it implies.

Offline and dependency-free — no torch, no CUDA, no real checkpoint. The
``metadata.pt`` fixtures are built by hand because that is precisely the claim
under test: ``torch.save``'s modern format is a zip whose ``<archive>/data.pkl``
member is an ordinary pickle, so the standard library can read one. If that ever
stops being true these tests fail rather than the field.
"""

from __future__ import annotations

import json
import logging
import pickle
import zipfile
from collections import Counter, OrderedDict
from pathlib import Path

import pytest

from apxinf.checkpoints import (
    LEROBOT,
    OPENPI_PYTORCH,
    CheckpointError,
    MetadataError,
    detect_checkpoint,
    read_metadata_pt,
    require_norm_stats,
    resolve_tokenizer,
    train_config_facts,
)

STATS = {"actions": {"q01": [-1.0] * 16, "q99": [1.0] * 16}, "state": {"q01": [0.0] * 16, "q99": [1.0] * 16}}


def write_metadata_pt(path: Path, payload) -> Path:
    """Write ``payload`` the way ``torch.save`` does: a zip around one pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("metadata/data.pkl", pickle.dumps(payload, protocol=2))
        archive.writestr("metadata/version", "3\n")
    return path


def openpi_payload(
    *,
    asset_id="example-asset",
    repo_id=None,
    images=("cam_high", "cam_left_wrist", "cam_right_wrist"),
    **model_overrides,
):
    """A payload shaped like a real openpi ``TrainConfig`` dump."""
    model = {
        "action_dim": 32,
        "action_horizon": 50,
        "max_token_len": 200,
        "discrete_state_input": True,
        "pi05": True,
        "paligemma_variant": "gemma_2b",
        "action_expert_variant": "gemma_300m",
    }
    model.update(model_overrides)
    return {
        "global_step": 14002,
        "timestamp": "2025-07-27T17:01:00",
        "config": {
            "exp_name": "pi05_unitree_g1_example",
            "model": model,
            "data": {
                "repo_id": repo_id,
                "assets": {"assets_dir": "/data/train-assets/", "asset_id": asset_id},
                "adapt_to_pi": True,
                "use_delta_joint_actions": True,
                "default_prompt": "",
                "repack_transforms": {
                    "inputs": [
                        {
                            "structure": {
                                "images": {key: f"observation.images.{key}" for key in images},
                                "state": "observation.state",
                            }
                        }
                    ]
                },
            },
        },
    }


def openpi_dir(root: Path, *, stats_at=None, **kwargs) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "model.safetensors").write_bytes(b"")
    write_metadata_pt(root / "metadata.pt", openpi_payload(**kwargs))
    if stats_at is not None:
        target = root / stats_at
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(STATS))
    return root


def lerobot_dir(root: Path, *, stats=True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "model.safetensors").write_bytes(b"")
    (root / "config.json").write_text(json.dumps({"type": "pi05", "chunk_size": 50}))
    if stats:
        (root / "norm_stats.json").write_text(json.dumps(STATS))
    return root


# --- the torch-free metadata.pt reader -------------------------------------


def test_reads_a_torch_style_zip_without_torch(tmp_path):
    path = write_metadata_pt(tmp_path / "metadata.pt", openpi_payload())
    payload = read_metadata_pt(path)
    assert payload["global_step"] == 14002
    assert payload["config"]["model"]["action_horizon"] == 50


def test_ordered_dict_survives_but_other_classes_are_stubbed(tmp_path):
    """Container types on the allowlist are real; everything else is inert."""
    path = write_metadata_pt(
        tmp_path / "metadata.pt", {"ordered": OrderedDict(a=1), "counter": Counter("aab")}
    )
    payload = read_metadata_pt(path)
    assert payload["ordered"] == OrderedDict(a=1)
    assert not isinstance(payload["counter"], Counter)


def test_builtins_are_not_blanket_allowed():
    """``builtins.eval`` must not resolve: this parses untrusted checkpoints.

    The allowlist is ``(module, name)`` pairs rather than module names for
    exactly this reason — ``collections`` and ``builtins`` both contain harmless
    container types *and* a route to arbitrary execution.
    """
    import io

    from apxinf.checkpoints.metadata import _RestrictedUnpickler

    unpickler = _RestrictedUnpickler(io.BytesIO(b""))
    for module, name in (("builtins", "eval"), ("builtins", "exec"), ("os", "system")):
        resolved = unpickler.find_class(module, name)
        assert resolved is not eval and callable(resolved)
        assert resolved().__class__.__name__ == name  # an inert stub instance

    # The container types that *are* allowed still come back for real.
    assert unpickler.find_class("collections", "OrderedDict") is OrderedDict


def test_non_zip_file_is_reported_not_crashed(tmp_path):
    path = tmp_path / "metadata.pt"
    path.write_bytes(b"not a zip at all")
    with pytest.raises(MetadataError, match="not a zip-format torch archive"):
        read_metadata_pt(path)


def test_missing_config_key_is_reported(tmp_path):
    path = write_metadata_pt(tmp_path / "metadata.pt", {"global_step": 1})
    with pytest.raises(MetadataError, match="no 'config' dict"):
        train_config_facts(read_metadata_pt(path))


# --- fact extraction --------------------------------------------------------


def test_architecture_is_extracted_in_config_json_vocabulary(tmp_path):
    facts = train_config_facts(openpi_payload())
    assert facts["arch"] == {
        "action_dim": 32,
        "action_horizon": 50,
        "max_token_len": 200,
        "num_views": 3,  # counted from the repack image structure
        "discrete_state_input": True,
    }
    # Upstream openpi does not serialize num_steps, and asserting a value the
    # checkpoint never stated would present the loader's default as a fact.
    assert "num_flow_steps" not in facts["arch"]


def test_unnamed_cameras_leave_num_views_to_the_loader():
    """A TrainConfig whose repack transform names no cameras at all."""
    facts = train_config_facts(openpi_payload(images=()))
    assert facts["image_keys"] == ()
    assert "num_views" not in facts["arch"]


def test_fork_specific_num_steps_wins_over_the_default():
    facts = train_config_facts(openpi_payload(num_steps=5, action_horizon=5))
    assert facts["arch"]["num_flow_steps"] == 5
    assert facts["arch"]["action_horizon"] == 5


def test_asset_id_comes_from_assets_then_repo_id():
    explicit = train_config_facts(openpi_payload(asset_id="example-asset", repo_id="x/y"))
    assert (explicit["asset_id"], explicit["asset_id_source"]) == (
        "example-asset",
        "data.assets.asset_id",
    )

    # openpi: `asset_id = data.assets.asset_id or data.repo_id`.
    fallback = train_config_facts(openpi_payload(asset_id=None, repo_id="some-org/some-task"))
    assert (fallback["asset_id"], fallback["asset_id_source"]) == (
        "some-org/some-task",
        "data.repo_id",
    )


def test_deployment_facts_are_carried_through():
    facts = train_config_facts(openpi_payload())
    assert facts["image_keys"] == ("cam_high", "cam_left_wrist", "cam_right_wrist")
    assert facts["state_key"] == "observation.state"
    assert facts["adapt_to_pi"] is True
    assert facts["use_delta_joint_actions"] is True
    assert facts["exp_name"] == "pi05_unitree_g1_example"
    assert facts["global_step"] == 14002


def test_a_pi0_checkpoint_is_refused():
    with pytest.raises(MetadataError, match="pi05=False"):
        train_config_facts(openpi_payload(pi05=False))


def test_an_unsupported_backbone_is_refused():
    with pytest.raises(MetadataError, match="paligemma_variant"):
        train_config_facts(openpi_payload(paligemma_variant="gemma_2b_lora"))


# --- format detection -------------------------------------------------------


def test_openpi_export_is_detected_from_metadata_pt(tmp_path):
    root = openpi_dir(tmp_path / "ckpt", stats_at="assets/example-asset/norm_stats.json")
    layout = detect_checkpoint(root)
    assert layout.format == OPENPI_PYTORCH
    assert layout.asset_id == "example-asset"
    assert layout.norm_stats == root / "assets/example-asset/norm_stats.json"
    assert layout.norm_stats_is_fallback is False
    assert json.loads(layout.config_json_text())["num_views"] == 3


def test_lerobot_directory_is_detected_from_config_json(tmp_path):
    root = lerobot_dir(tmp_path / "ckpt")
    layout = detect_checkpoint(root)
    assert layout.format == LEROBOT
    assert layout.norm_stats == root / "norm_stats.json"
    # No architecture override: the Rust loader reads config.json itself, which
    # is what LeRobot checkpoints already did.
    assert layout.config_json_text() is None


def test_metadata_pt_outranks_a_stale_config_json(tmp_path):
    """A shipped directory shape: real metadata.pt, hand-added config.json."""
    root = openpi_dir(tmp_path / "ckpt", stats_at="assets/example-asset/norm_stats.json")
    (root / "config.json").write_text(json.dumps({"type": "pi05", "chunk_size": 10}))

    layout = detect_checkpoint(root)
    assert layout.format == OPENPI_PYTORCH
    assert json.loads(layout.config_json_text())["action_horizon"] == 50
    assert any("config.json is ignored" in note for note in layout.notes)


def test_neither_layout_names_both_requirements(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    (root / "model.safetensors").write_bytes(b"")
    with pytest.raises(CheckpointError) as excinfo:
        detect_checkpoint(root)
    message = str(excinfo.value)
    assert "metadata.pt" in message and "config.json" in message


def test_pinned_format_does_not_fall_back_to_sniffing(tmp_path):
    root = lerobot_dir(tmp_path / "ckpt")
    with pytest.raises(CheckpointError, match="metadata.pt does not"):
        detect_checkpoint(root, checkpoint_format=OPENPI_PYTORCH)


# --- norm_stats resolution --------------------------------------------------


def test_slashed_asset_id_becomes_nested_directories(tmp_path):
    """No asset_id, so openpi falls back to repo_id, which is 'org/name'."""
    root = openpi_dir(
        tmp_path / "ckpt",
        asset_id=None,
        repo_id="some-org/some-task",
        stats_at="assets/some-org/some-task/norm_stats.json",
    )
    layout = detect_checkpoint(root)
    assert layout.norm_stats == root / "assets/some-org/some-task/norm_stats.json"
    assert layout.norm_stats_is_fallback is False


def test_asset_path_without_the_assets_prefix_is_accepted(tmp_path):
    root = openpi_dir(
        tmp_path / "ckpt",
        asset_id=None,
        repo_id="some-org/some-task",
        stats_at="some-org/some-task/norm_stats.json",
    )
    assert detect_checkpoint(root).norm_stats == root / "some-org/some-task/norm_stats.json"


def test_root_fallback_is_used_and_logged(tmp_path, caplog):
    """The decision: keep reading the root file, but never do it silently."""
    root = openpi_dir(tmp_path / "ckpt", stats_at="norm_stats.json")
    with caplog.at_level(logging.WARNING, logger="apxinf.checkpoints"):
        layout = detect_checkpoint(root)

    assert layout.norm_stats == root / "norm_stats.json"
    assert layout.norm_stats_is_fallback is True
    message = caplog.text
    assert "assets/example-asset/norm_stats.json" in message  # the path that missed
    assert str(root / "norm_stats.json") in message  # the one actually used
    assert "example-asset" in message  # the asset_id, so the ask is actionable


def test_the_asset_path_wins_over_a_root_file(tmp_path):
    root = openpi_dir(tmp_path / "ckpt", stats_at="assets/example-asset/norm_stats.json")
    (root / "norm_stats.json").write_text(json.dumps(STATS))

    layout = detect_checkpoint(root)
    assert layout.norm_stats == root / "assets/example-asset/norm_stats.json"
    assert layout.norm_stats_is_fallback is False
    assert any("is ignored" in note for note in layout.notes)


def test_explicit_path_outranks_every_convention(tmp_path):
    root = openpi_dir(tmp_path / "ckpt", stats_at="assets/example-asset/norm_stats.json")
    elsewhere = tmp_path / "elsewhere" / "norm_stats.json"
    elsewhere.parent.mkdir()
    elsewhere.write_text(json.dumps(STATS))

    assert detect_checkpoint(root, norm_stats=elsewhere).norm_stats == elsewhere


def test_explicit_asset_id_overrides_metadata(tmp_path):
    root = openpi_dir(tmp_path / "ckpt", stats_at="assets/other/norm_stats.json")
    layout = detect_checkpoint(root, asset_id="other")
    assert layout.norm_stats == root / "assets/other/norm_stats.json"
    assert layout.asset_id_source == "explicit override"


def test_a_traversing_asset_id_is_refused(tmp_path):
    root = openpi_dir(tmp_path / "ckpt", asset_id="../../etc")
    with pytest.raises(CheckpointError, match="not a usable relative path"):
        detect_checkpoint(root)


def test_missing_stats_is_deferred_not_raised_by_detection(tmp_path):
    """preflight reports all findings at once, so detection must not crash."""
    root = openpi_dir(tmp_path / "ckpt")
    layout = detect_checkpoint(root)
    assert layout.norm_stats is None
    assert len(layout.norm_stats_tried) == 3


def test_require_norm_stats_names_every_path_and_the_asset_id(tmp_path):
    root = openpi_dir(tmp_path / "ckpt")
    with pytest.raises(CheckpointError) as excinfo:
        require_norm_stats(detect_checkpoint(root))
    message = str(excinfo.value)
    for candidate in ("assets/example-asset/norm_stats.json", "example-asset/norm_stats.json"):
        assert str(root / candidate) in message
    assert "train_pytorch.py" in message  # where the file comes from


def test_lerobot_error_explains_where_lerobot_keeps_statistics(tmp_path):
    root = lerobot_dir(tmp_path / "ckpt", stats=False)
    with pytest.raises(CheckpointError) as excinfo:
        require_norm_stats(detect_checkpoint(root))
    message = str(excinfo.value)
    assert "normalizer_processor.safetensors" in message
    assert "meta/stats.json" in message


# --- tokenizer --------------------------------------------------------------


def test_tokenizer_resolution_order(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    (root / "paligemma_tokenizer.model").write_bytes(b"sp")
    elsewhere = tmp_path / "shared.model"
    elsewhere.write_bytes(b"sp")

    assert resolve_tokenizer(root, elsewhere) == elsewhere
    assert resolve_tokenizer(root, env={"APXINF_TOKENIZER": str(elsewhere)}) == elsewhere
    assert resolve_tokenizer(root, env={}) == root / "paligemma_tokenizer.model"


def test_tokenizer_error_says_where_to_get_the_file(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    with pytest.raises(CheckpointError) as excinfo:
        resolve_tokenizer(root, env={})
    message = str(excinfo.value)
    assert "gs://big_vision/paligemma_tokenizer.model" in message
    assert "APXINF_TOKENIZER" in message
    assert str(root / "paligemma_tokenizer.model") in message


# --- the wiring into Pi05Policy.from_pretrained -----------------------------
#
# Detection is only worth anything if what it resolves reaches the loader. An
# openpi export has no config.json, so before this the Rust side fell back to
# Pi05Config::default() without a word — a checkpoint with a 50-step horizon
# would have been served at whatever the default happened to be.


class _FakeModel:
    action_horizon = 50
    action_dim = 32
    num_views = 3
    image_size = 224
    max_token_len = 200

    def reset_sampling(self, seed=None):
        pass


def _load_with_fake_binding(monkeypatch, model_dir, **kwargs):
    """Run ``from_pretrained`` against a stub binding; return the load kwargs."""
    import sys
    import types

    from apxinf.policies.impls import pi05

    captured = {}

    class FakeBindingModel:
        @staticmethod
        def load(*args, **load_kwargs):
            captured.update(load_kwargs)
            captured["positional"] = args
            return _FakeModel()

    class FakePipeline:
        names = []

        def __getitem__(self, name):
            raise KeyError(name)

    monkeypatch.setitem(sys.modules, "apxinf_py", types.SimpleNamespace(Model=FakeBindingModel))
    monkeypatch.setattr(pi05, "resolve_pi05_tactics", lambda *a, **k: None)
    monkeypatch.setattr(pi05, "PromptTokenizer", lambda *a, **k: object())
    monkeypatch.setattr(
        pi05.Pi05Policy, "default_pipelines", classmethod(
            lambda cls, *a, **k: (FakePipeline(), FakePipeline())
        )
    )
    (Path(model_dir) / "paligemma_tokenizer.model").write_bytes(b"sp")
    pi05.Pi05Policy.from_pretrained(model_dir, device="cuda:0", precision="bf16", **kwargs)
    return captured


def test_metadata_pt_architecture_reaches_the_rust_loader(tmp_path, monkeypatch):
    root = openpi_dir(
        tmp_path / "ckpt",
        stats_at="assets/example-asset/norm_stats.json",
        action_horizon=50,
    )

    captured = _load_with_fake_binding(monkeypatch, root, action_dim=16, discrete_state=False)

    config = json.loads(captured["config_json"])
    assert config["action_horizon"] == 50
    assert config["action_dim"] == 32
    assert config["num_views"] == 3


def test_a_lerobot_directory_still_lets_the_loader_read_config_json(tmp_path, monkeypatch):
    """No ``config_json=``: the Rust loader reads config.json itself, as before."""
    root = lerobot_dir(tmp_path / "ckpt")

    captured = _load_with_fake_binding(monkeypatch, root, action_dim=16, discrete_state=False)

    assert "config_json" not in captured


def test_the_asset_path_statistics_are_the_ones_loaded(tmp_path, monkeypatch):
    """A wrong root file next to correct asset statistics — the delivered shape."""
    root = openpi_dir(tmp_path / "ckpt", stats_at="assets/example-asset/norm_stats.json")
    (root / "norm_stats.json").write_text(
        json.dumps({"actions": {"q01": [-1.0] * 7, "q99": [1.0] * 7}})
    )

    from apxinf.policies.impls import pi05

    seen = {}
    original = pi05.Unnormalizer.from_norm_stats

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(pi05.Unnormalizer, "from_norm_stats", spy)
    _load_with_fake_binding(monkeypatch, root, discrete_state=False)

    assert seen["path"] == root / "assets" / "example-asset" / "norm_stats.json"


def test_a_flat_directory_loads_with_no_layout_at_all(tmp_path, monkeypatch):
    """The hand-assembled layout that predates this module must keep working."""
    root = tmp_path / "flat"
    root.mkdir()
    (root / "model.safetensors").write_bytes(b"")
    (root / "norm_stats.json").write_text(json.dumps(STATS))

    captured = _load_with_fake_binding(monkeypatch, root, discrete_state=False)

    assert "config_json" not in captured


def test_explicit_norm_stats_needs_no_layout(tmp_path, monkeypatch):
    """``--norm-stats`` names one file; it does not assert a directory shape."""
    root = tmp_path / "flat"
    root.mkdir()
    (root / "model.safetensors").write_bytes(b"")
    elsewhere = tmp_path / "stats.json"
    elsewhere.write_text(json.dumps(STATS))

    from apxinf.policies.impls import pi05

    seen = {}
    original = pi05.Unnormalizer.from_norm_stats
    monkeypatch.setattr(
        pi05.Unnormalizer,
        "from_norm_stats",
        lambda *a, **k: (seen.update(k), original(*a, **k))[1],
    )
    _load_with_fake_binding(monkeypatch, root, discrete_state=False, norm_stats=elsewhere)

    assert seen["path"] == elsewhere


def test_a_missing_explicit_norm_stats_path_is_refused(tmp_path, monkeypatch):
    root = tmp_path / "flat"
    root.mkdir()
    (root / "model.safetensors").write_bytes(b"")

    with pytest.raises(CheckpointError, match="does not exist"):
        _load_with_fake_binding(
            monkeypatch, root, discrete_state=False, norm_stats=tmp_path / "nope.json"
        )
