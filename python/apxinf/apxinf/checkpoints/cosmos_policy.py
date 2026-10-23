"""Prepare the pinned official Cosmos Policy LIBERO assets for native loading.

Run ``python -m apxinf.checkpoints.cosmos_policy --assets DIR --output DIR``.
Conversion is offline; inference does not import torch or the reference model.
Original files are read in place, hash checked, and never modified. The trusted
upstream pickle is deserialized only after checking its pinned SHA256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

SOURCE_REVISION = "18a2accadf4e7a3531e56754102af5a24d2316da"
CHECKPOINT_REVISION = "cb689ec0e3347c13667d70a78a3447388f5c3bb8"
PROFILE = "libero-2b-direct-v1"
ASSETS = {
    "Cosmos-Policy-LIBERO-Predict2-2B.pt": "8818528d8c9150cda0ddf8c711b0f221b21dac8ac379bd26d5690235954d33e2",
    "tokenizer.pth": "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981",
    "libero_t5_embeddings.pkl": "8a03499676c6c196127c577144b9fd09bb02c30f9caf058adbcef09bb99ad8f5",
    "libero_dataset_statistics.json": "5b119a98ad7824507ddff3c6c7507ca244261ee416dcfa623876702160c580d3",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def convert(assets: Path, output: Path) -> dict:
    import torch
    from safetensors.torch import save_file

    for name, expected in ASSETS.items():
        if sha256(assets / name) != expected:
            raise ValueError(f"Cosmos Policy official asset SHA256 mismatch: {name}")
    # Refuse an existing destination, including partially prepared artifacts.
    # config.json is written last and is the loader's completion marker.
    output.mkdir(parents=True, exist_ok=False)
    policy = torch.load(assets / "Cosmos-Policy-LIBERO-Predict2-2B.pt",
                        map_location="cpu", weights_only=True, mmap=True)
    counters = {"net.accum_video_sample_counter", "net.accum_image_sample_counter",
                "net.accum_iteration", "net.accum_train_in_hours"}
    removed = []
    network = {}
    for name, tensor in policy.items():
        if name in counters or name.endswith("._extra_state"):
            removed.append(name)
            continue
        if not name.startswith("net.") or not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Unexpected policy entry: {name}")
        if tensor.dtype != torch.bfloat16 or not torch.isfinite(tensor).all():
            raise ValueError(f"Invalid BF16 policy tensor: {name}")
        # Keep source shapes and orientation. Native weights owns GPU packing.
        network[name.removeprefix("net.")] = tensor.contiguous()
    save_file(network, str(output / "model.safetensors"))
    del network, policy

    vae = torch.load(assets / "tokenizer.pth", map_location="cpu",
                     weights_only=True, mmap=True)
    encoder = {}
    for name, tensor in vae.items():
        if name.startswith(("encoder.", "conv1.")):
            if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
                raise ValueError(f"Invalid FP32 VAE tensor: {name}")
            encoder[name] = tensor.to(torch.bfloat16).contiguous()
        elif not name.startswith(("decoder.", "conv2.")):
            raise ValueError(f"Unexpected VAE entry: {name}")
    save_file(encoder, str(output / "vae_encoder.safetensors"))
    del encoder, vae

    with (assets / "libero_t5_embeddings.pkl").open("rb") as stream:
        cached = pickle.load(stream)
    tasks = sorted(cached)
    if len(tasks) != 40 or any(not isinstance(task, str) or not task for task in tasks):
        raise ValueError("Expected the official 40-task LIBERO text cache")
    for task in tasks:
        tensor = cached[task]
        if (not isinstance(tensor, torch.Tensor) or tensor.shape != (1, 512, 1024)
                or tensor.dtype != torch.bfloat16 or not torch.isfinite(tensor).all()):
            raise ValueError(f"Invalid cached T5 tensor for {task!r}")
    save_file({"embeddings": torch.cat([cached[task].cpu() for task in tasks]).contiguous()},
              str(output / "text.safetensors"))
    (output / "tasks.json").write_text(json.dumps({"tasks": tasks}, indent=2) + "\n")
    (output / "libero_dataset_statistics.json").write_bytes(
        (assets / "libero_dataset_statistics.json").read_bytes())
    manifest = {
        "source_revision": SOURCE_REVISION, "checkpoint_revision": CHECKPOINT_REVISION,
        "source_sha256": ASSETS, "removed_non_inference_entries": sorted(removed),
        "vae_scope": "encoder and mean projection; FP32 to BF16 as in the official loader",
        "output_sha256": {path.name: sha256(path) for path in sorted(output.iterdir())},
    }
    (output / "conversion.json").write_text(json.dumps(manifest, indent=2) + "\n")
    config = {
        "model_type": "cosmos-policy", "profile": PROFILE, "format_version": 1,
        "source_revision": SOURCE_REVISION, "checkpoint_revision": CHECKPOINT_REVISION,
        "precision": "bf16", "num_tasks": 40, "text_shape": [512, 1024],
        "initial_latent_shape": [1, 16, 9, 28, 28], "action_shape": [16, 7],
        "num_views": 2, "image_size": 224, "state_dim": 9,
        "denoiser_calls": 5, "scaling": "rectified_flow", "solver": "2ab",
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    convert(args.assets, args.output)


if __name__ == "__main__":
    main()
