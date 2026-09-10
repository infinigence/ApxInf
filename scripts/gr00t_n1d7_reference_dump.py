#!/usr/bin/env python3
"""Dump a deterministic NVIDIA GR00T N1.7 BF16 inference reference.

The synthetic fixture is intentionally defined at the already-preprocessed
model boundary: two 2x2
vision grids, two image tokens, zero state, and zero initial flow noise.  That
keeps the comparison independent of image/tokenizer preprocessing while still
executing the complete vision, language, VL-adapter, and action-head graph.

This script must be run against the pinned Isaac-GR00T source tree and local
GR00T/Cosmos checkpoints.  It never downloads model files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from typing import Any

# Loading must be hermetic. If a local model path is incomplete, fail instead
# of silently filling it from Hugging Face Hub.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
import diffusers
import transformers
from transformers.feature_extraction_utils import BatchFeature


SCHEMA = "apxinf.gr00t-n1.7.reference.v1"
EXPECTED_SOURCE_REVISION = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--backbone", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--embodiment-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--input-npz",
        type=Path,
        help="optional official processor dump; defaults to the synthetic fixture",
    )
    parser.add_argument(
        "--fixture",
        help="fixture identifier stored in metadata (derived when omitted)",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_revision(source_dir: Path) -> str:
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={source_dir}",
            "-C",
            str(source_dir),
            "rev-parse",
            "HEAD",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def as_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().contiguous().numpy()


def clone_batch(batch: BatchFeature) -> BatchFeature:
    return BatchFeature(
        data={
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
    )


def import_reference_classes(source_dir: Path) -> tuple[type[Any], type[Any]]:
    """Import model math without pulling in the dataset/preprocessing stack.

    ``gr00t.model.__init__`` eagerly imports the training pipeline, and the
    model constructor eagerly creates a data collator. Neither is involved in
    this preprocessed-boundary fixture. Stubbing only those two orchestration
    modules keeps the numerical model implementation untouched and avoids
    making the reference environment depend on torchvision/video packages.
    """

    source_dir = source_dir.resolve()
    sys.path.insert(0, str(source_dir))
    model_package = types.ModuleType("gr00t.model")
    model_package.__path__ = [str(source_dir / "gr00t" / "model")]
    model_package.__package__ = "gr00t.model"
    sys.modules["gr00t.model"] = model_package

    processing_name = "gr00t.model.gr00t_n1d7.processing_gr00t_n1d7"
    processing_module = types.ModuleType(processing_name)

    class BoundaryOnlyCollator:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def __call__(self, _batch: Any) -> Any:
            raise RuntimeError(
                "the deterministic reference starts after preprocessing; "
                "the GR00T data collator must not be called"
            )

    processing_module.Gr00tN1d7DataCollator = BoundaryOnlyCollator
    sys.modules[processing_name] = processing_module

    from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7

    return Gr00tN1d7, Gr00tN1d7Config


def load_model(args: argparse.Namespace) -> tuple[Any, Any]:
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {args.checkpoint}")
    if not args.backbone.is_dir():
        raise FileNotFoundError(f"backbone directory not found: {args.backbone}")
    revision = source_revision(args.source_dir)
    if revision != EXPECTED_SOURCE_REVISION:
        raise RuntimeError(
            f"Isaac-GR00T source revision is {revision}, expected "
            f"{EXPECTED_SOURCE_REVISION}"
        )
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("the GR00T BF16 reference requires an available CUDA device")

    Gr00tN1d7, Gr00tN1d7Config = import_reference_classes(args.source_dir)
    config = Gr00tN1d7Config.from_pretrained(
        str(args.checkpoint), local_files_only=True
    )
    backbone_path = args.backbone.resolve()
    temporary_backbone_root: tempfile.TemporaryDirectory[str] | None = None
    if "nvidia/Cosmos-Reason2" not in str(backbone_path):
        # The pinned NVIDIA implementation selects the Qwen3 backbone class by
        # matching the Hugging Face repository name in ``model_name``. A local
        # checkpoint may live at an arbitrary path (for example
        # ``/workspace/models/Cosmos-Reason2-2B``), even though its config is
        # identical. Give the loader a temporary, read-only symlink whose name
        # satisfies that provider-name check; the original checkpoint remains
        # untouched and is still the artifact hashed in the report.
        temporary_backbone_root = tempfile.TemporaryDirectory(
            prefix="apxinf-gr00t-local-backbone-"
        )
        local_backbone = (
            Path(temporary_backbone_root.name)
            / "nvidia"
            / "Cosmos-Reason2-2B"
        )
        local_backbone.parent.mkdir(parents=True)
        local_backbone.symlink_to(backbone_path, target_is_directory=True)
        backbone_path = local_backbone
    config.model_name = str(backbone_path)
    config.load_bf16 = True
    config.use_flash_attention = False
    expected_contract = {
        "model_type": "Gr00tN1d7",
        "model_dtype": "bfloat16",
        "action_horizon": 40,
        "max_action_dim": 132,
        "max_state_dim": 132,
        "num_inference_timesteps": 4,
        "num_timestep_buckets": 1000,
    }
    for name, expected in expected_contract.items():
        actual = getattr(config, name)
        if actual != expected:
            raise RuntimeError(
                f"checkpoint {name} is {actual!r}, expected {expected!r}"
            )

    try:
        model = Gr00tN1d7.from_pretrained(
            str(args.checkpoint),
            config=config,
            dtype=torch.bfloat16,
            local_files_only=True,
            transformers_loading_kwargs={
                "local_files_only": True,
                "trust_remote_code": True,
            },
        )
    finally:
        if temporary_backbone_root is not None:
            temporary_backbone_root.cleanup()
    return model.to(device=args.device, dtype=torch.bfloat16).eval(), config


def reference_inputs(
    model: Any, config: Any, args: argparse.Namespace
) -> tuple[BatchFeature, BatchFeature, torch.Tensor]:
    if args.input_npz is not None:
        with np.load(args.input_npz, allow_pickle=False) as archive:
            required = {
                "input_ids",
                "attention_mask",
                "pixel_values",
                "image_grid_thw",
                "state",
                "embodiment_id",
            }
            missing = sorted(required.difference(archive.files))
            if missing:
                raise RuntimeError(f"processor dump is missing {missing}")
            device = torch.device(args.device)
            backbone_input = BatchFeature(
                data={
                    "pixel_values": torch.from_numpy(
                        np.asarray(archive["pixel_values"], dtype=np.float32)
                    )
                    .to(device=device, dtype=torch.bfloat16)
                    .contiguous(),
                    "image_grid_thw": torch.from_numpy(
                        np.asarray(archive["image_grid_thw"], dtype=np.int64)
                    )
                    .to(device=device)
                    .contiguous(),
                    "input_ids": torch.from_numpy(
                        np.asarray(archive["input_ids"], dtype=np.int64)
                    )
                    .to(device=device)
                    .contiguous(),
                    "attention_mask": torch.from_numpy(
                        np.asarray(archive["attention_mask"], dtype=np.int64)
                    )
                    .to(device=device)
                    .contiguous(),
                }
            )
            action_input = BatchFeature(
                data={
                    "state": torch.from_numpy(
                        np.asarray(archive["state"], dtype=np.float32)
                    )
                    .to(device=device, dtype=torch.bfloat16)
                    .contiguous(),
                    "embodiment_id": torch.from_numpy(
                        np.asarray(archive["embodiment_id"], dtype=np.int64)
                    )
                    .to(device=device)
                    .contiguous(),
                }
            )
            if "initial_noise" in archive:
                noise = torch.from_numpy(
                    np.asarray(archive["initial_noise"], dtype=np.float32)
                ).to(device=device, dtype=torch.bfloat16)
            else:
                noise = torch.zeros(
                    (1, int(config.action_horizon), int(config.max_action_dim)),
                    dtype=torch.bfloat16,
                    device=device,
                )
        return backbone_input, action_input, noise.contiguous()

    backbone_config = model.backbone.model.config
    vision_config = backbone_config.vision_config
    patch_width = (
        int(vision_config.in_channels)
        * int(vision_config.temporal_patch_size)
        * int(vision_config.patch_size)
        * int(vision_config.patch_size)
    )
    image_token_id = int(backbone_config.image_token_id)
    device = torch.device(args.device)

    # Each [1, 2, 2] grid contributes four input patches and one merged token.
    backbone_input = BatchFeature(
        data={
            "pixel_values": torch.zeros(
                (8, patch_width), dtype=torch.bfloat16, device=device
            ),
            "image_grid_thw": torch.tensor(
                [[1, 2, 2], [1, 2, 2]], dtype=torch.long, device=device
            ),
            "input_ids": torch.tensor(
                [[1, image_token_id, image_token_id, 2]],
                dtype=torch.long,
                device=device,
            ),
            "attention_mask": torch.ones((1, 4), dtype=torch.long, device=device),
        }
    )
    action_input = BatchFeature(
        data={
            "state": torch.zeros(
                (1, int(config.state_history_length), int(config.max_state_dim)),
                dtype=torch.bfloat16,
                device=device,
            ),
            "embodiment_id": torch.tensor(
                [args.embodiment_id], dtype=torch.long, device=device
            ),
        }
    )
    noise = torch.zeros(
        (1, int(config.action_horizon), int(config.max_action_dim)),
        dtype=torch.bfloat16,
        device=device,
    )
    return backbone_input, action_input, noise


@torch.no_grad()
def run_reference(
    model: Any,
    config: Any,
    backbone_input: BatchFeature,
    action_input: BatchFeature,
    noise: torch.Tensor,
) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {
        "input_ids": backbone_input.input_ids.detach().cpu().numpy(),
        "attention_mask": backbone_input.attention_mask.detach().cpu().numpy(),
        "pixel_values": as_numpy(backbone_input.pixel_values),
        "image_grid_thw": backbone_input.image_grid_thw.detach().cpu().numpy(),
        "state": as_numpy(action_input.state),
        "embodiment_id": action_input.embodiment_id.detach().cpu().numpy(),
        "initial_noise": as_numpy(noise),
    }

    backbone_output = model.backbone(backbone_input)
    payload["backbone_features_raw"] = as_numpy(backbone_output.backbone_features)
    payload["backbone_attention_mask"] = (
        backbone_output.backbone_attention_mask.detach().cpu().numpy()
    )
    payload["image_mask"] = backbone_output.image_mask.detach().cpu().numpy()

    head = model.action_head
    processed = head.process_backbone_output(clone_batch(backbone_output))
    backbone_features = processed.backbone_features
    payload["backbone_features_post_adapter"] = as_numpy(backbone_features)

    state = action_input.state.view(action_input.state.shape[0], 1, -1)
    state_features = head.state_encoder(state, action_input.embodiment_id)
    payload["state_features"] = as_numpy(state_features)

    actions = noise.clone()
    delta = 1.0 / int(config.num_inference_timesteps)
    batch_size = actions.shape[0]
    for step in range(int(config.num_inference_timesteps)):
        timestep = int(
            (step / float(config.num_inference_timesteps))
            * int(config.num_timestep_buckets)
        )
        timesteps = torch.full(
            (batch_size,), timestep, dtype=torch.long, device=actions.device
        )
        payload[f"step_{step}_action_in"] = as_numpy(actions)
        payload[f"step_{step}_timestep_embedding"] = as_numpy(
            head.model.timestep_encoder(timesteps)
        )

        action_features = head.action_encoder(
            actions, timesteps, action_input.embodiment_id
        )
        if config.add_pos_embed:
            position_ids = torch.arange(
                action_features.shape[1], dtype=torch.long, device=actions.device
            )
            action_features = action_features + head.position_embedding(position_ids).unsqueeze(0)
        payload[f"step_{step}_action_features"] = as_numpy(action_features)

        state_action = torch.cat((state_features, action_features), dim=1)
        model_output, hidden_states = head.model(
            hidden_states=state_action,
            encoder_hidden_states=backbone_features,
            timestep=timesteps,
            image_mask=processed.image_mask,
            backbone_attention_mask=processed.backbone_attention_mask,
            return_all_hidden_states=True,
        )
        for layer, hidden in enumerate(hidden_states):
            payload[f"step_{step}_dit_hidden_{layer}"] = as_numpy(hidden)
        payload[f"step_{step}_dit_output"] = as_numpy(model_output)

        decoded = head.action_decoder(model_output, action_input.embodiment_id)
        velocity = decoded[:, -int(config.action_horizon) :]
        payload[f"step_{step}_velocity"] = as_numpy(velocity)
        actions = actions + delta * velocity
        payload[f"step_{step}_action_out"] = as_numpy(actions)

    payload["final_action"] = as_numpy(actions)
    return payload


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    model, config = load_model(args)
    backbone_input, action_input, noise = reference_inputs(model, config, args)
    payload = run_reference(model, config, backbone_input, action_input, noise)

    checkpoint_config = args.checkpoint / "config.json"
    backbone_config = args.backbone / "config.json"
    metadata: dict[str, Any] = {
        "schema": SCHEMA,
        "source_revision": source_revision(args.source_dir),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_config_sha256": sha256(checkpoint_config),
        "backbone": str(args.backbone.resolve()),
        "backbone_config_sha256": sha256(backbone_config),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "diffusers": diffusers.__version__,
        "device": args.device,
        "dtype": "bfloat16",
        "seed": args.seed,
        "fixture": args.fixture
        or (
            "nvidia-libero-episode0-step0-real-v1"
            if args.input_npz is not None
            else "two-view-zero-boundary-input-v1"
        ),
        "output_shape": list(payload["final_action"].shape),
    }
    if args.input_npz is not None:
        metadata["input_npz"] = str(args.input_npz.resolve())
        metadata["input_npz_sha256"] = sha256(args.input_npz)
    payload["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **payload)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"wrote {args.output} with {len(payload)} entries")


if __name__ == "__main__":
    main()
