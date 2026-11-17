"""Prepare task-balanced LIBERO LeRobot observations with the checkpoint processor."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "apxinf.gr00t.calibration-inputs.v1"


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def source_identity(root):
    digest = hashlib.sha256()
    paths = sorted((root / "gr00t").rglob("*.py"))
    if not paths:
        raise ValueError("GR00T processor source files are missing")
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(file_hash(path).encode())
    return "sha256:" + digest.hexdigest()


def select_samples(episodes, tasks, episodes_per_task, frames_per_episode, seed):
    if episodes_per_task < 1 or frames_per_episode < 1 or seed < 0:
        raise ValueError("sample counts must be positive and seed non-negative")
    groups = {task["task"]: [] for task in tasks}
    for episode in episodes:
        if len(episode["tasks"]) != 1 or episode["tasks"][0] not in groups:
            raise ValueError("expected one known task per LIBERO episode")
        groups[episode["tasks"][0]].append(episode)
    selected = []
    for task in sorted(tasks, key=lambda task: task["task_index"]):
        candidates = sorted(groups[task["task"]], key=lambda episode: episode["episode_index"])
        if len(candidates) < episodes_per_task:
            raise ValueError(f"not enough episodes for task {task['task_index']}")
        rng = np.random.default_rng(np.random.SeedSequence([seed, task["task_index"]]))
        indices = sorted(rng.choice(len(candidates), episodes_per_task, replace=False).tolist())
        for index in indices:
            episode = candidates[index]
            fractions = np.linspace(0.1, 0.9, frames_per_episode) if frames_per_episode > 1 else [0.5]
            frames = [int(round(fraction * (episode["length"] - 1))) for fraction in fractions]
            if len(set(frames)) != len(frames):
                raise ValueError("episode is too short for distinct temporal samples")
            selected.extend({"task_index": task["task_index"], "prompt": task["task"],
                             "episode_index": episode["episode_index"], "frame_index": frame,
                             "episode_length": episode["length"]} for frame in frames)
    return selected


def decode_frames(path, indices, fps):
    import av

    selected = {}
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in indices:
                if frame.time is None or abs(float(frame.time) - index / fps) > 0.5 / fps:
                    raise ValueError(f"video timestamp mismatch: {path}, frame {index}")
                selected[index] = frame.to_ndarray(format="rgb24")
            if index >= max(indices):
                break
    if set(selected) != set(indices):
        raise ValueError(f"missing video frames: {path}")
    return selected


def prepare(args):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    sys.path.insert(0, str(ROOT / "python/apxinf"))
    sys.path.insert(0, str(args.gr00t_source.resolve()))
    import pyarrow.parquet as parquet
    import torch
    from apxinf.policies.impls.gr00t import Gr00tPolicy, _NvidiaProcessorAdapter, _round_to_bf16

    torch.set_num_threads(4)
    dataset = args.dataset.resolve()
    checkpoint = args.checkpoint.resolve()
    backbone = checkpoint / "assets/cosmos"
    info = json.loads((dataset / "meta/info.json").read_text())
    if info["robot_type"] != "franka" or info["features"]["observation.state"]["shape"] != [8]:
        raise ValueError("only the LIBERO Franka 8-component state layout is supported")
    tasks = read_jsonl(dataset / "meta/tasks.jsonl")
    episodes = read_jsonl(dataset / "meta/episodes.jsonl")
    if len(tasks) != info["total_tasks"] or len(episodes) != info["total_episodes"]:
        raise ValueError("dataset metadata is incomplete")
    samples = select_samples(episodes, tasks, args.episodes_per_task, args.frames_per_episode, args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    adapter = _NvidiaProcessorAdapter.load(
        checkpoint, backbone=backbone, embodiment="libero_sim",
        image_keys=("observation/image", "observation/wrist_image"),
        state_key="observation/state", prompt_key="prompt", action_key=None,
    )
    modality_path = args.gr00t_source / "examples/LIBERO/modality.json"
    modality = json.loads(modality_path.read_text())
    if adapter.video_keys != ["image", "wrist_image"] or adapter.state_dim != 8:
        raise ValueError("checkpoint LIBERO modality does not match dataset mapping")
    config = json.loads((checkpoint / "config.json").read_text())
    sources = {str(path.relative_to(dataset)): file_hash(path)
               for path in (dataset / "meta/info.json", dataset / "meta/tasks.jsonl", dataset / "meta/episodes.jsonl")}
    current_episode = None
    for sample_index, sample in enumerate(samples):
        episode_index = sample["episode_index"]
        if current_episode != episode_index:
            current_episode = episode_index
            values = {"episode_chunk": episode_index // info["chunks_size"], "episode_index": episode_index}
            path = dataset / info["data_path"].format(**values)
            sources[str(path.relative_to(dataset))] = file_hash(path)
            table = parquet.read_table(path).to_pydict()
            if len(table["frame_index"]) != sample["episode_length"]:
                raise ValueError(f"episode length mismatch: {path}")
            indices = {item["frame_index"] for item in samples if item["episode_index"] == episode_index}
            videos = {}
            for camera in adapter.video_keys:
                video_key = modality["video"][camera]["original_key"]
                path = dataset / info["video_path"].format(**values, video_key=video_key)
                sources[str(path.relative_to(dataset))] = file_hash(path)
                videos[camera] = decode_frames(path, indices, info["fps"])
        frame_index = sample["frame_index"]
        if (table["frame_index"][frame_index] != frame_index
                or table["episode_index"][frame_index] != episode_index
                or table["task_index"][frame_index] != sample["task_index"]
                or abs(table["timestamp"][frame_index] - frame_index / info["fps"]) > 0.5 / info["fps"]):
            raise ValueError("Parquet row/task/timestamp does not match sampled frame")
        state = np.asarray(table["observation.state"][frame_index], dtype=np.float32)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError("invalid LIBERO state")
        states = {key: state[modality["state"][key]["start"]:modality["state"][key]["end"]]
                  for key in adapter.state_keys}
        observation = {"observation/image": videos["image"][frame_index],
                       "observation/wrist_image": videos["wrist_image"][frame_index],
                       "observation/state": states, "prompt": sample["prompt"]}
        with torch.inference_mode():
            encoded = adapter.encode(observation)
        encoded.pop("raw_states")
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, sample_index]))
        encoded["noise"] = _round_to_bf16(rng.standard_normal(
            (1, config["action_horizon"], config["max_action_dim"]), dtype=np.float32))
        for name in ("pixel_values", "state"):
            encoded[name] = _round_to_bf16(encoded[name])
        filename = f"sample-{sample_index:04d}.npz"
        np.savez(args.output / filename, **encoded)
        sample.update(file=filename, sha256=file_hash(args.output / filename),
                      token_count=int(encoded["token_ids"].size), embodiment_id=int(encoded["embodiment_id"]),
                      image_grid_thw=encoded["image_grid_thw"].tolist())
        print(f"prepared={sample_index + 1}/{len(samples)} task={sample['task_index']} M={sample['token_count']}", flush=True)
    manifest = {"schema": SCHEMA, "checkpoint": Gr00tPolicy.checkpoint_identity(checkpoint, backbone),
                "dataset": str(dataset), "kind": "representative", "seed": args.seed,
                "sampling": {"episodes_per_task": args.episodes_per_task, "frames_per_episode": args.frames_per_episode},
                "sources": sources, "samples": samples,
                "modality_mapping": {"file": str(modality_path), "sha256": file_hash(modality_path)},
                "processor_source_identity": source_identity(args.gr00t_source),
                "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "av", "pyarrow")}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gr00t-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes-per-task", type=int, default=3)
    parser.add_argument("--frames-per-episode", type=int, default=3)
    parser.add_argument("--seed", type=int, default=9000)
    prepare(parser.parse_args())
