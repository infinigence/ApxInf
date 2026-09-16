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
import struct
import zipfile
from collections import Counter, OrderedDict
from pathlib import Path

import pytest

from apxinf.checkpoints import (
    LEROBOT,
    OPENPI_PYTORCH,
    CheckpointError,
    MetadataError,
    NormalizationPlan,
    detect_checkpoint,
    read_metadata_pt,
    resolve_tokenizer,
    train_config_facts,
)
from apxinf.checkpoints.descriptor import IDENTITY_MISSING_STATS, MEAN_STD, QUANTILE
from apxinf.checkpoints.layout import _main as inspect_checkpoint

STATS = {"actions": {"q01": [-1.0] * 16, "q99": [1.0] * 16}, "state": {"q01": [0.0] * 16, "q99": [1.0] * 16}}


def write_safetensors(path: Path, tensors) -> Path:
    """Write the small float32 subset LeRobot processor state uses."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {}
    payload = bytearray()
    for name, values in tensors.items():
        flat = tuple(float(value) for value in values)
        start = len(payload)
        payload.extend(struct.pack(f"<{len(flat)}f", *flat))
        header[name] = {
            "dtype": "F32",
            "shape": [len(flat)],
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    return path


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


def lerobot_processor_dir(root: Path, *, mode="QUANTILES", state_files=True) -> Path:
    """A current LeRobot PI0.5 policy directory, without model-weight contents."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "model.safetensors").write_bytes(b"")
    (root / "config.json").write_text(
        json.dumps(
            {
                "type": "pi05",
                "chunk_size": 50,
                "input_features": {
                    "observation.images.image": {"type": "VISUAL", "shape": [3, 256, 256]},
                    "observation.state": {"type": "STATE", "shape": [8]},
                },
                "output_features": {"action": {"type": "ACTION", "shape": [7]}},
            }
        )
    )
    norm_map = {"VISUAL": "IDENTITY", "STATE": mode, "ACTION": mode}
    features = (
        {
            "observation.state": {"type": "STATE", "shape": [8]},
            "action": {"type": "ACTION", "shape": [7]},
        }
        if state_files
        else {}
    )
    pre_step = {
        "registry_name": "normalizer_processor",
        "config": {"eps": 1e-8, "features": features, "norm_map": norm_map},
    }
    post_step = {
        "registry_name": "unnormalizer_processor",
        "config": {
            "eps": 1e-8,
            "features": ({"action": features["action"]} if state_files else {}),
            "norm_map": norm_map,
        },
    }
    if state_files:
        pre_step["state_file"] = "policy_preprocessor_step_2_normalizer_processor.safetensors"
        post_step["state_file"] = "policy_postprocessor_step_0_unnormalizer_processor.safetensors"

    (root / "policy_preprocessor.json").write_text(
        json.dumps(
            {
                "name": "policy_preprocessor",
                "steps": [
                    pre_step,
                    {
                        "registry_name": "tokenizer_processor",
                        "config": {
                            "tokenizer_name": "google/paligemma-3b-pt-224",
                            "max_length": 200,
                        },
                    },
                ],
            }
        )
    )
    (root / "policy_postprocessor.json").write_text(
        json.dumps({"name": "policy_postprocessor", "steps": [post_step]})
    )

    if state_files:
        if mode == "QUANTILES":
            state_stats = {
                "observation.state.q01": [-2.0] * 8,
                "observation.state.q99": [2.0] * 8,
                "action.q01": [-1.0] * 7,
                "action.q99": [1.0] * 7,
            }
            action_stats = {"action.q01": [-1.0] * 7, "action.q99": [1.0] * 7}
        else:
            state_stats = {
                "observation.state.mean": [0.25] * 8,
                "observation.state.std": [0.5] * 8,
                "action.mean": [0.1] * 7,
                "action.std": [0.2] * 7,
            }
            action_stats = {"action.mean": [0.1] * 7, "action.std": [0.2] * 7}
        write_safetensors(root / pre_step["state_file"], state_stats)
        write_safetensors(root / post_step["state_file"], action_stats)
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
    assert isinstance(layout.normalization, NormalizationPlan)
    assert layout.normalization.state.feature_key == "state"
    assert layout.normalization.action.feature_key == "actions"
    assert layout.normalization.action.mode == QUANTILE
    assert layout.normalization.action.eps == 1e-6
    assert layout.normalization.action.source == str(layout.norm_stats)


def test_openpi_nested_norm_stats_become_the_same_canonical_plan(tmp_path):
    root = openpi_dir(tmp_path / "ckpt")
    path = root / "assets" / "example-asset" / "norm_stats.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"norm_stats": STATS}))

    plan = detect_checkpoint(root).normalization

    assert plan.action.values["q01"] == tuple(STATS["actions"]["q01"])
    assert plan.state.values["q99"] == tuple(STATS["state"]["q99"])


def test_openpi_flat_stats_ignore_unrelated_serialized_metadata(tmp_path):
    root = openpi_dir(tmp_path / "ckpt")
    path = root / "assets" / "example-asset" / "norm_stats.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({**STATS, "generated_at_step": 14002}))

    plan = detect_checkpoint(root).normalization

    assert plan.action.width == 16


def test_openpi_normalization_keys_are_part_of_the_public_resolver(tmp_path):
    root = openpi_dir(tmp_path / "ckpt")
    path = root / "assets" / "example-asset" / "norm_stats.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "libero_all": {"q01": [-1.0] * 7, "q99": [1.0] * 7},
                "proprio": {"q01": [0.0] * 8, "q99": [1.0] * 8},
            }
        )
    )

    plan = detect_checkpoint(
        root, norm_key="libero_all", state_norm_key="proprio"
    ).normalization

    assert plan.action.feature_key == "libero_all"
    assert plan.action.width == 7
    assert plan.state.feature_key == "proprio"
    assert plan.state.width == 8


def test_openpi_plan_rejects_non_quantile_action_statistics(tmp_path):
    root = openpi_dir(tmp_path / "ckpt")
    path = root / "assets" / "example-asset" / "norm_stats.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"actions": {"mean": [0.0], "std": [1.0]}}))

    with pytest.raises(CheckpointError, match=r"actions\.q01 must be a numeric vector"):
        detect_checkpoint(root)


def test_lerobot_directory_is_detected_from_config_json(tmp_path):
    root = lerobot_dir(tmp_path / "ckpt")
    layout = detect_checkpoint(root)
    assert layout.format == LEROBOT
    assert layout.norm_stats == root / "norm_stats.json"
    # No architecture override: the Rust loader reads config.json itself, which
    # is what LeRobot checkpoints already did.
    assert layout.config_json_text() is None


def test_lerobot_base_processor_declares_identity_fallback(tmp_path):
    root = lerobot_processor_dir(tmp_path / "ckpt", state_files=False)

    layout = detect_checkpoint(root)

    assert layout.format == LEROBOT
    assert layout.normalization is not None
    assert layout.normalization.state.mode == "identity"
    assert layout.normalization.state.status == IDENTITY_MISSING_STATS
    assert layout.normalization.state.width == 8
    assert layout.normalization.action.mode == "identity"
    assert layout.normalization.action.status == IDENTITY_MISSING_STATS
    assert layout.normalization.action.width == 7
    assert layout.tokenizer is not None
    assert layout.tokenizer.name == "google/paligemma-3b-pt-224"


def test_inspector_warns_when_processor_stats_are_missing(tmp_path, capsys):
    root = lerobot_processor_dir(tmp_path / "ckpt", state_files=False)
    tokenizer = root / "tokenizer.model"
    tokenizer.write_bytes(b"sp")

    assert inspect_checkpoint([str(root), "--tokenizer", str(tokenizer)]) == 0

    output = capsys.readouterr().out
    assert "WARN [normalization]" in output
    assert "review the normalization warning" in output


@pytest.mark.parametrize(
    ("serialized_mode", "mode", "names"),
    [
        ("QUANTILES", QUANTILE, ("q01", "q99")),
        ("MEAN_STD", MEAN_STD, ("mean", "std")),
    ],
)
def test_lerobot_processor_state_becomes_canonical_transforms(
    tmp_path, serialized_mode, mode, names
):
    root = lerobot_processor_dir(tmp_path / "ckpt", mode=serialized_mode)

    layout = detect_checkpoint(root)

    plan = layout.normalization
    assert plan is not None
    assert plan.state.mode == mode
    assert plan.state.width == 8
    assert set(plan.state.values) == set(names)
    assert plan.action.mode == mode
    assert plan.action.width == 7
    assert set(plan.action.values) == set(names)
    assert "policy_preprocessor_step_2" in plan.state.source
    assert "policy_postprocessor_step_0" in plan.action.source


def test_lerobot_quantile_epsilon_is_canonicalized_in_the_adapter(tmp_path):
    root = lerobot_processor_dir(tmp_path / "ckpt")
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        path = root / name
        document = json.loads(path.read_text())
        document["steps"][0]["config"]["eps"] = 0.25
        path.write_text(json.dumps(document))
    write_safetensors(
        root / "policy_preprocessor_step_2_normalizer_processor.safetensors",
        {
            "observation.state.q01": [0.0] * 8,
            "observation.state.q99": [0.0] + [1.0] * 7,
            "action.q01": [0.0] * 7,
            "action.q99": [0.0] + [1.0] * 6,
        },
    )
    write_safetensors(
        root / "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        {"action.q01": [0.0] * 7, "action.q99": [0.0] + [1.0] * 6},
    )

    plan = detect_checkpoint(root).normalization

    assert plan.action.eps == 0.0
    assert plan.action.values["q99"] == (0.25,) + (1.0,) * 6


def test_lerobot_processor_sidecars_ignore_a_stray_root_norm_stats(tmp_path):
    root = lerobot_processor_dir(tmp_path / "ckpt")
    (root / "norm_stats.json").write_text(
        json.dumps({"actions": {"q01": [-10.0] * 7, "q99": [10.0] * 7}})
    )

    layout = detect_checkpoint(root)

    assert layout.norm_stats is None
    assert "policy_postprocessor_step_0" in layout.normalization.action.source


def test_explicit_stats_override_keeps_lerobot_tokenizer_metadata(tmp_path):
    root = lerobot_processor_dir(tmp_path / "ckpt")
    override = tmp_path / "override.json"
    override.write_text(
        json.dumps(
            {
                "action": {"q01": [-3.0] * 7, "q99": [3.0] * 7},
                "observation.state": {"q01": [-2.0] * 8, "q99": [2.0] * 8},
            }
        )
    )

    layout = detect_checkpoint(
        root,
        norm_stats=override,
        norm_key="action",
        state_norm_key="observation.state",
    )

    assert layout.tokenizer.name == "google/paligemma-3b-pt-224"
    assert layout.normalization.action.source == str(override)
    assert layout.normalization.action.values["q99"] == (3.0,) * 7


def test_lerobot_declared_state_file_missing_is_corrupt(tmp_path):
    root = lerobot_processor_dir(tmp_path / "ckpt")
    (root / "policy_preprocessor_step_2_normalizer_processor.safetensors").unlink()

    with pytest.raises(CheckpointError, match="state_file.*does not exist"):
        detect_checkpoint(root)


def test_an_unrelated_hf_config_is_not_called_lerobot(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    (root / "model.safetensors").write_bytes(b"")
    (root / "config.json").write_text(json.dumps({"model_type": "qwen3_vl"}))

    with pytest.raises(CheckpointError, match="not a supported LeRobot PI0.5"):
        detect_checkpoint(root)


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


def test_lerobot_state_file_cannot_escape_checkpoint(tmp_path):
    root = lerobot_processor_dir(tmp_path / "ckpt", state_files=False)
    preprocessor = root / "policy_preprocessor.json"
    document = json.loads(preprocessor.read_text())
    document["steps"][0]["state_file"] = "../outside.safetensors"
    preprocessor.write_text(json.dumps(document))

    with pytest.raises(CheckpointError, match="unsafe state_file"):
        detect_checkpoint(root)


def test_lerobot_uses_postprocessor_action_stats(tmp_path):
    root = lerobot_processor_dir(tmp_path / "ckpt")
    write_safetensors(
        root / "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        {"action.q01": [-0.5] * 7, "action.q99": [0.5] * 7},
    )

    plan = detect_checkpoint(root).normalization

    assert plan.action.values["q01"] == (-0.5,) * 7
    assert plan.action.values["q99"] == (0.5,) * 7


def test_lerobot_rejects_non_finite_processor_statistics(tmp_path):
    root = lerobot_processor_dir(tmp_path / "ckpt")
    write_safetensors(
        root / "policy_preprocessor_step_2_normalizer_processor.safetensors",
        {
            "observation.state.q01": [float("nan")] * 8,
            "observation.state.q99": [1.0] * 8,
            "action.q01": [-1.0] * 7,
            "action.q99": [1.0] * 7,
        },
    )

    with pytest.raises(CheckpointError, match="non-finite"):
        detect_checkpoint(root)


def test_lerobot_rejects_processor_statistics_with_wrong_width(tmp_path):
    root = lerobot_processor_dir(tmp_path / "ckpt")
    write_safetensors(
        root / "policy_preprocessor_step_2_normalizer_processor.safetensors",
        {
            "observation.state.q01": [-1.0] * 7,
            "observation.state.q99": [1.0] * 7,
            "action.q01": [-1.0] * 7,
            "action.q99": [1.0] * 7,
        },
    )

    with pytest.raises(CheckpointError, match="width 7, expected 8"):
        detect_checkpoint(root)


def test_lerobot_rejects_malformed_processor_safetensors(tmp_path):
    root = lerobot_processor_dir(tmp_path / "ckpt")
    (root / "policy_preprocessor_step_2_normalizer_processor.safetensors").write_bytes(
        b"broken"
    )

    with pytest.raises(CheckpointError, match="truncated SafeTensors header"):
        detect_checkpoint(root)


def test_lerobot_rejects_declared_but_empty_processor_state(tmp_path):
    root = lerobot_processor_dir(tmp_path / "ckpt")
    write_safetensors(
        root / "policy_preprocessor_step_2_normalizer_processor.safetensors", {}
    )

    with pytest.raises(CheckpointError, match="state_file contains no tensors"):
        detect_checkpoint(root)


def test_lerobot_stateless_postprocessor_means_identity_actions(tmp_path):
    root = lerobot_processor_dir(tmp_path / "ckpt")
    postprocessor = root / "policy_postprocessor.json"
    document = json.loads(postprocessor.read_text())
    del document["steps"][0]["state_file"]
    postprocessor.write_text(json.dumps(document))

    plan = detect_checkpoint(root).normalization

    assert plan.action.mode == "identity"
    assert plan.action.status == IDENTITY_MISSING_STATS


def test_lerobot_rejects_postprocessor_state_without_action_tensors(tmp_path):
    root = lerobot_processor_dir(tmp_path / "ckpt")
    write_safetensors(
        root / "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        {"unrelated.q01": [-1.0] * 7, "unrelated.q99": [1.0] * 7},
    )

    with pytest.raises(CheckpointError, match="no tensors for action ACTION"):
        detect_checkpoint(root)


def test_lerobot_action_only_sidecars_leave_state_as_identity(tmp_path):
    root = lerobot_processor_dir(tmp_path / "ckpt")
    write_safetensors(
        root / "policy_preprocessor_step_2_normalizer_processor.safetensors",
        {"action.q01": [-1.0] * 7, "action.q99": [1.0] * 7},
    )

    plan = detect_checkpoint(root).normalization

    assert plan.state.mode == "identity"
    assert plan.state.status == IDENTITY_MISSING_STATS
    assert plan.action.mode == QUANTILE
    assert plan.action.status == "resolved"


def test_lerobot_missing_preprocessor_state_keeps_postprocessor_action_stats(tmp_path):
    root = lerobot_processor_dir(tmp_path / "ckpt")
    preprocessor = root / "policy_preprocessor.json"
    document = json.loads(preprocessor.read_text())
    del document["steps"][0]["state_file"]
    preprocessor.write_text(json.dumps(document))

    plan = detect_checkpoint(root).normalization

    assert plan.state.mode == "identity"
    assert plan.state.status == IDENTITY_MISSING_STATS
    assert plan.action.mode == QUANTILE
    assert plan.action.status == "resolved"


def test_lerobot_rejects_processor_feature_that_disagrees_with_policy_config(tmp_path):
    root = lerobot_processor_dir(tmp_path / "ckpt")
    preprocessor = root / "policy_preprocessor.json"
    document = json.loads(preprocessor.read_text())
    document["steps"][0]["config"]["features"]["observation.state"]["shape"] = [9]
    preprocessor.write_text(json.dumps(document))
    write_safetensors(
        root / "policy_preprocessor_step_2_normalizer_processor.safetensors",
        {
            "observation.state.q01": [-1.0] * 9,
            "observation.state.q99": [1.0] * 9,
            "action.q01": [-1.0] * 7,
            "action.q99": [1.0] * 7,
        },
    )

    with pytest.raises(CheckpointError, match="disagrees with config.json"):
        detect_checkpoint(root)


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
# An openpi export has no config.json, so resolved architecture values must reach
# the Rust loader through config_json.


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

    def fake_default_pipelines(cls, *args, **pipeline_kwargs):
        captured["pipeline_kwargs"] = pipeline_kwargs
        return FakePipeline(), FakePipeline()

    monkeypatch.setitem(sys.modules, "apxinf_py", types.SimpleNamespace(Model=FakeBindingModel))
    monkeypatch.setattr(pi05, "resolve_pi05_tactics", lambda *a, **k: None)
    monkeypatch.setattr(pi05, "PromptTokenizer", lambda *a, **k: object())
    monkeypatch.setattr(
        pi05.Pi05Policy, "default_pipelines", classmethod(fake_default_pipelines)
    )
    (Path(model_dir) / "paligemma_tokenizer.model").write_bytes(b"sp")
    captured["policy"] = pi05.Pi05Policy.from_pretrained(
        model_dir, device="cuda:0", compute_variant="bf16", **kwargs
    )
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


def test_lerobot_base_policy_uses_explicit_identity_processors(tmp_path, monkeypatch):
    root = lerobot_processor_dir(tmp_path / "ckpt", state_files=False)

    captured = _load_with_fake_binding(
        monkeypatch,
        root,
        discrete_state=True,
        state_key="observation.state",
    )

    pipeline = captured["pipeline_kwargs"]
    assert pipeline["state_normalizer"] is None
    values = [[-0.75] * 7]
    assert pipeline["unnormalizer"](values).tolist() == values
    normalization = captured["policy"].metadata["normalization"]
    assert normalization == {
        "state": "identity/identity_missing_stats",
        "action": "identity/identity_missing_stats",
    }


def test_injected_unnormalizer_is_the_published_action_transform(
    tmp_path, monkeypatch, caplog
):
    from apxinf.processors import Unnormalizer

    root = lerobot_processor_dir(tmp_path / "ckpt", state_files=False)
    injected = Unnormalizer(q01=[0.0] * 7, q99=[2.0] * 7)

    with caplog.at_level(logging.WARNING, logger="apxinf.policies.pi05"):
        captured = _load_with_fake_binding(
            monkeypatch,
            root,
            discrete_state=False,
            unnormalizer=injected,
        )

    assert captured["pipeline_kwargs"]["unnormalizer"] is injected
    assert captured["policy"].metadata["normalization"] == {
        "state": "identity/identity_missing_stats",
        "action": "custom/injected",
    }
    assert "identity passthrough" not in caplog.text


def test_openpi_missing_stats_loads_with_identity_passthrough(
    tmp_path, monkeypatch, caplog
):
    root = openpi_dir(tmp_path / "ckpt")

    with caplog.at_level(logging.WARNING, logger="apxinf.policies.pi05"):
        captured = _load_with_fake_binding(
            monkeypatch,
            root,
            action_dim=16,
            discrete_state=True,
            state_key="observation.state",
        )

    pipeline = captured["pipeline_kwargs"]
    assert pipeline["state_normalizer"] is None
    values = [[-0.75] * 16]
    assert pipeline["unnormalizer"](values).tolist() == values
    assert captured["policy"].metadata["normalization"] == {
        "state": "absent",
        "action": "identity/identity_missing_stats",
    }
    assert "identity passthrough" in caplog.text


def test_action_only_stats_leave_state_unnormalized(tmp_path, monkeypatch, caplog):
    root = openpi_dir(tmp_path / "ckpt")
    stats = root / "assets" / "example-asset" / "norm_stats.json"
    stats.parent.mkdir(parents=True)
    stats.write_text(
        json.dumps({"actions": {"q01": [-1.0] * 16, "q99": [1.0] * 16}})
    )

    with caplog.at_level(logging.WARNING, logger="apxinf.policies.pi05"):
        captured = _load_with_fake_binding(
            monkeypatch,
            root,
            action_dim=16,
            discrete_state=True,
            state_key="observation.state",
        )

    assert captured["pipeline_kwargs"]["state_normalizer"] is None
    assert captured["policy"].metadata["normalization"]["state"] == "absent"
    assert "identity passthrough" in caplog.text


def test_lerobot_sidecars_build_existing_apxinf_processors(tmp_path, monkeypatch):
    root = lerobot_processor_dir(tmp_path / "ckpt", mode="MEAN_STD")

    captured = _load_with_fake_binding(
        monkeypatch,
        root,
        discrete_state=True,
        state_key="observation.state",
    )

    pipeline = captured["pipeline_kwargs"]
    assert pipeline["state_normalizer"]([0.75] * 8).tolist() == pytest.approx([1.0] * 8)
    assert pipeline["unnormalizer"]([[1.0] * 7]).tolist()[0] == pytest.approx([0.3] * 7)


def test_lerobot_quantile_epsilon_only_guards_zero_ranges(tmp_path, monkeypatch):
    root = lerobot_processor_dir(tmp_path / "ckpt")
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        path = root / name
        document = json.loads(path.read_text())
        document["steps"][0]["config"]["eps"] = 0.25
        path.write_text(json.dumps(document))
    write_safetensors(
        root / "policy_preprocessor_step_2_normalizer_processor.safetensors",
        {
            "observation.state.q01": [0.0] * 8,
            "observation.state.q99": [1.0] * 8,
            "action.q01": [0.0] * 7,
            "action.q99": [1.0] * 7,
        },
    )
    write_safetensors(
        root / "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        {"action.q01": [0.0] * 7, "action.q99": [1.0] * 7},
    )

    captured = _load_with_fake_binding(
        monkeypatch,
        root,
        discrete_state=True,
        state_key="observation.state",
    )

    pipeline = captured["pipeline_kwargs"]
    assert pipeline["state_normalizer"]([0.5] * 8).tolist() == pytest.approx([0.0] * 8)
    assert pipeline["unnormalizer"]([[0.0] * 7]).tolist()[0] == pytest.approx([0.5] * 7)


def test_lerobot_mean_std_epsilon_applies_only_when_normalizing(tmp_path, monkeypatch):
    root = lerobot_processor_dir(tmp_path / "ckpt", mode="MEAN_STD")
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        path = root / name
        document = json.loads(path.read_text())
        document["steps"][0]["config"]["eps"] = 0.25
        path.write_text(json.dumps(document))

    captured = _load_with_fake_binding(
        monkeypatch,
        root,
        discrete_state=True,
        state_key="observation.state",
    )

    pipeline = captured["pipeline_kwargs"]
    assert pipeline["state_normalizer"]([0.75] * 8).tolist() == pytest.approx(
        [2.0 / 3.0] * 8
    )
    assert pipeline["unnormalizer"]([[1.0] * 7]).tolist()[0] == pytest.approx([0.3] * 7)


def test_the_asset_path_statistics_are_the_ones_loaded(tmp_path, monkeypatch):
    """A wrong root file next to correct asset statistics — the delivered shape."""
    root = openpi_dir(tmp_path / "ckpt", stats_at="assets/example-asset/norm_stats.json")
    (root / "norm_stats.json").write_text(
        json.dumps({"actions": {"q01": [-10.0] * 7, "q99": [10.0] * 7}})
    )

    from apxinf.policies.impls import pi05

    monkeypatch.setattr(
        pi05.Unnormalizer,
        "from_norm_stats",
        lambda *args, **kwargs: pytest.fail("policy bypassed checkpoint descriptor"),
    )
    captured = _load_with_fake_binding(
        monkeypatch, root, action_dim=7, discrete_state=False
    )

    # The selected asset stats map normalized 1 to ~1; the ignored root file
    # would map it to ~10.
    result = captured["pipeline_kwargs"]["unnormalizer"]([[1.0] * 7])
    assert result.tolist()[0] == pytest.approx([1.000001] * 7)


def test_a_flat_directory_loads_with_no_layout_at_all(tmp_path, monkeypatch):
    """Native flat checkpoint directories remain supported."""
    root = tmp_path / "flat"
    root.mkdir()
    (root / "model.safetensors").write_bytes(b"")
    (root / "norm_stats.json").write_text(json.dumps(STATS))

    captured = _load_with_fake_binding(monkeypatch, root, discrete_state=False)

    assert "config_json" not in captured


def test_a_legacy_native_pi05_config_does_not_get_misclassified_as_lerobot(
    tmp_path, monkeypatch
):
    """ApxInf's pre-layout config.json is architecture input, not an HF marker."""
    root = tmp_path / "flat"
    root.mkdir()
    (root / "model.safetensors").write_bytes(b"")
    (root / "config.json").write_text(
        json.dumps(
            {
                "action_dim": 32,
                "action_horizon": 10,
                "paligemma_variant": "gemma_2b",
                "action_expert_variant": "gemma_300m",
            }
        )
    )
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

    captured = _load_with_fake_binding(
        monkeypatch,
        root,
        action_dim=7,
        discrete_state=False,
        norm_stats=elsewhere,
    )

    result = captured["pipeline_kwargs"]["unnormalizer"]([[1.0] * 7])
    assert result.tolist()[0] == pytest.approx([1.000001] * 7)


def test_a_missing_explicit_norm_stats_path_is_refused(tmp_path, monkeypatch):
    root = tmp_path / "flat"
    root.mkdir()
    (root / "model.safetensors").write_bytes(b"")

    with pytest.raises(CheckpointError, match="does not exist"):
        _load_with_fake_binding(
            monkeypatch, root, discrete_state=False, norm_stats=tmp_path / "nope.json"
        )
