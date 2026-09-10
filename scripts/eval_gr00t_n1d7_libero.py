#!/usr/bin/env python3
"""Run the shipped ``apxinf.Gr00tPolicy`` in NVIDIA's LIBERO harness.

The simulator adapter only translates NVIDIA's batched rollout dictionary to
the public ApxInf observation/action contract. Preprocessing, native model
execution, BF16-rounded numpy noise and action decoding all go through the
same public policy users deploy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def optional_artifact(path: Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": sha256(resolved)}


def array_signature(value: np.ndarray) -> dict[str, object]:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("utf-8"))
    digest.update(
        json.dumps(list(contiguous.shape), separators=(",", ":")).encode("utf-8")
    )
    digest.update(contiguous.tobytes(order="C"))
    return {
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "sha256": digest.hexdigest(),
    }


def git_revision(path: Path) -> dict[str, object]:
    revision = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"revision": revision, "dirty": dirty}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--backbone", required=True, type=Path)
    parser.add_argument(
        "--precision", choices=("bf16", "fp8", "int8"), required=True
    )
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--tactics", type=Path)
    parser.add_argument("--env-name", required=True)
    parser.add_argument(
        "--views",
        type=int,
        choices=(1, 2),
        default=2,
        help="Number of physical LIBERO camera streams presented to the policy.",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="parallel simulator environments; model calls remain serialized at batch 1",
    )
    parser.add_argument("--max-episode-steps", type=int, default=720)
    parser.add_argument("--n-action-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--noise-mode",
        choices=("fixed", "stream"),
        default="stream",
        help=(
            "stream draws the seeded sequence used for task evaluation; fixed "
            "reuses one BF16 noise tensor and is diagnostic only"
        ),
    )
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument(
        "--record-first-normalized-action",
        action="store_true",
        help="store the first normalized [1,H,D] action for paired precision parity",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.precision == "fp8" and args.calibration is None:
        parser.error("--precision fp8 requires --calibration")
    return args


def main() -> None:
    args = parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    sys.path.insert(0, str(args.source_dir.resolve()))
    import gr00t.model  # noqa: F401
    from gr00t.eval._horizon_contract import PolicyHorizonSpec
    from gr00t.eval.rollout_policy import (
        MultiStepConfig,
        VideoConfig,
        WrapperConfigs,
        run_rollout_gymnasium_policy,
    )
    from gr00t.policy.gr00t_policy import (
        Gr00tPolicy as NvidiaGr00tPolicy,
        Gr00tSimPolicyWrapper,
    )
    from gr00t.policy.policy import BasePolicy
    from apxinf import Gr00tPolicy as PublicGr00tPolicy
    import apxinf_py

    # Maturin editable installs expose a Python package that re-exports the
    # native submodule.  Hash the loaded shared object, not its tiny __init__.py.
    apxinf_extension = getattr(apxinf_py, "apxinf_py", apxinf_py)

    class ApxInfGr00tPolicy(NvidiaGr00tPolicy):
        """Thin simulator adapter around the shipped public ApxInf policy."""

        def __init__(self) -> None:
            BasePolicy.__init__(self, strict=True)
            image_keys = (
                ("observation/image",)
                if args.views == 1
                else ("observation/image", "observation/wrist_image")
            )
            self.deployed_policy = PublicGr00tPolicy.from_pretrained(
                args.checkpoint,
                backbone=args.backbone,
                precision=args.precision,
                calibration=args.calibration,
                tactics=args.tactics,
                embodiment="libero_sim",
                image_keys=image_keys,
                seed=args.seed,
                noise_mode=args.noise_mode,
            )
            adapter = self.deployed_policy.processor
            self.processor = adapter.processor
            self.embodiment_tag = adapter.embodiment_tag
            self.modality_configs = adapter.modality_configs
            self.video_keys = adapter.video_keys
            self.language_key = self.modality_configs["language"].modality_keys[0]
            self.model = self.deployed_policy.model
            self.inference_count = 0
            self.inference_seconds = 0.0
            self.first_normalized_action = None
            self.first_model_input_signatures = None

        def _get_action(self, observation, options=None):
            del options
            unbatched = self._unbatch_observation(observation)
            action_batches = {}
            for obs in unbatched:
                language = obs["language"][self.language_key]
                prompt = language if isinstance(language, str) else language[0]
                request = {
                    user_key: obs["video"][model_key]
                    for user_key, model_key in zip(
                        self.deployed_policy.image_keys, self.video_keys
                    )
                }
                request["observation/state"] = obs["state"]
                request["prompt"] = prompt
                result = self.deployed_policy.infer(
                    request, include_model_inputs=True
                )
                normalized = result["normalized_actions"][None]
                if self.first_normalized_action is None:
                    self.first_normalized_action = normalized.copy()
                if self.first_model_input_signatures is None:
                    inputs = result["model_inputs"]
                    self.first_model_input_signatures = {
                        key: array_signature(inputs[key])
                        for key in (
                            "pixel_values",
                            "image_grid_thw",
                            "token_ids",
                            "attention_mask",
                            "state",
                        )
                    }
                    self.first_model_input_signatures["noise"] = array_signature(
                        result["noise"]
                    )
                    self.first_model_input_signatures["embodiment_id"] = int(
                        inputs["embodiment_id"]
                    )
                self.inference_seconds += result["timing"]["model_ms"] / 1000.0
                self.inference_count += 1
                offset = 0
                for key in self.deployed_policy.processor.selected_action_keys:
                    width = self.deployed_policy.processor.action_dims[key]
                    value = result["actions"][:, offset : offset + width][None]
                    action_batches.setdefault(key, []).append(value)
                    offset += width
            return {
                key: np.concatenate(values, axis=0)
                for key, values in action_batches.items()
            }, {"apxinf_model_seconds": self.inference_seconds}

        def reset(self, options=None):
            del options
            return {}

    policy = Gr00tSimPolicyWrapper(ApxInfGr00tPolicy())
    contract = PolicyHorizonSpec.from_policy(policy, n_action_steps=args.n_action_steps)
    wrappers = WrapperConfigs(
        multistep=MultiStepConfig(
            contract=contract,
            max_episode_steps=args.max_episode_steps,
            terminate_on_success=True,
        ),
        video=VideoConfig(
            video_dir=str(args.video_dir) if args.video_dir else None,
            max_episode_steps=args.max_episode_steps,
        ),
    )
    env_name, successes, info = run_rollout_gymnasium_policy(
        env_name=args.env_name,
        policy=policy,
        wrapper_configs=wrappers,
        n_episodes=args.episodes,
        n_envs=min(args.n_envs, args.episodes),
        seed=args.seed,
    )
    # Some vector-environment implementations finish more than one environment
    # on the final step.  Keep the requested prefix so the receipt has exactly
    # the declared sample count and the campaign cannot silently over-count.
    successes = list(successes)[: args.episodes]
    info = {
        key: value[: args.episodes] if isinstance(value, list) else value
        for key, value in info.items()
    }
    tuning_lookups = policy.policy.model.tuning_lookup_stats()
    gemm_plans = policy.policy.model.gemm_plan_stats()
    result = {
        "schema": "apxinf.gr00t-n1.7.libero-rollout.v1",
        "env_name": env_name,
        "precision": args.precision,
        "views": args.views,
        "video_keys": policy.policy.video_keys,
        "seed": args.seed,
        "episodes": len(successes),
        "successes": int(sum(bool(value) for value in successes)),
        "success_rate": float(np.mean(successes)),
        "episode_successes": [bool(value) for value in successes],
        "episode_info": info,
        "model_calls": policy.policy.inference_count,
        "model_seconds": policy.policy.inference_seconds,
        "tuning": {
            "runtime_records": int(policy.policy.model.tuning_record_count),
            "lookups": {
                "exact": int(tuning_lookups[0]),
                "bucket": int(tuning_lookups[1]),
                "miss": int(tuning_lookups[2]),
                "direct_exact_applied": int(tuning_lookups[3]),
            },
            "prepared_plans": {
                "exact": int(gemm_plans[0]),
                "bucket": int(gemm_plans[1]),
                "default": int(gemm_plans[2]),
            },
        },
        "runtime_contract": {
            "action_horizon": int(policy.policy.model.action_horizon),
            "action_dim": int(policy.policy.model.action_dim),
            "n_action_steps": args.n_action_steps,
            "max_episode_steps": args.max_episode_steps,
            "n_envs": min(args.n_envs, args.episodes),
            "noise_mode": args.noise_mode,
            "hf_hub_offline": os.environ["HF_HUB_OFFLINE"],
            "mujoco_gl": os.environ["MUJOCO_GL"],
            "pyopengl_platform": os.environ["PYOPENGL_PLATFORM"],
        },
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "artifacts": {
            "apxinf_source_git": git_revision(Path(__file__).resolve().parents[1]),
            "source_dir": str(args.source_dir.resolve()),
            "source_git": git_revision(args.source_dir),
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_config": optional_artifact(args.checkpoint / "config.json"),
            "backbone": str(args.backbone.resolve()),
            "backbone_config": optional_artifact(args.backbone / "config.json"),
            "apxinf_python_extension": optional_artifact(
                Path(apxinf_extension.__file__)
            ),
            "calibration": optional_artifact(args.calibration),
            "tactics": optional_artifact(args.tactics),
        },
    }
    if args.record_first_normalized_action:
        first_action = policy.policy.first_normalized_action
        if first_action is None:
            raise RuntimeError("rollout completed without producing a normalized action")
        result["first_normalized_action"] = first_action.reshape(-1).tolist()
        result["first_normalized_action_shape"] = list(first_action.shape)
        result["first_model_input_signatures"] = policy.policy.first_model_input_signatures
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(encoded)
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    policy.policy.deployed_policy.close()


if __name__ == "__main__":
    main()
