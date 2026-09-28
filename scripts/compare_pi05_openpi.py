#!/usr/bin/env python3
"""Deterministic PI0.5 parity against the official OpenPI implementation.

Run the stages in separate environments so OpenPI and ApxInf never occupy GPU
memory together. All generated data stays in an ignored ``devlocal`` suite.

    python scripts/compare_pi05_openpi.py prepare --suite-dir devlocal/pi05-openpi-parity/libero \
        --image-keys observation/image,observation/wrist_image \
        --source-npz /path/to/real_observation.npz
    # In an OpenPI environment with the official source installed:
    python scripts/compare_pi05_openpi.py openpi --suite-dir devlocal/pi05-openpi-parity/libero \
        --checkpoint-dir /path/to/pi05_libero --config-name pi05_libero
    # In the ApxInf CUDA environment:
    python scripts/compare_pi05_openpi.py apxinf --suite-dir devlocal/pi05-openpi-parity/libero \
        --checkpoint-dir /path/to/pi05_libero --precision fp8
    python scripts/compare_pi05_openpi.py compare --suite-dir devlocal/pi05-openpi-parity/libero

The default bare comparison checks the full normalized ``[H, 32]`` model
output. ``--interface policy`` checks the deployed action width for an OpenPI
policy-compatible checkpoint. ``--source-npz`` must contain arrays named by
``--image-keys`` and ``--state-key``. Two source observations are accepted;
the suite has seven cases regardless of the model horizon. Without a source,
the suite uses synthetic images and is marked diagnostic rather than representative.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
from typing import Any

import numpy as np


SCHEMA = "apxinf.pi05.openpi-parity.v1"
DEFAULT_DIR = pathlib.Path(__file__).resolve().parents[1] / "devlocal/pi05-openpi-parity/default"
SHORT_PROMPT = "put both moka pots on the stove"
LONG_PROMPT = (
    "put the white mug on the left plate and put the yellow and white mug "
    "on the right plate"
)
SHORT_TOKENS = (2, 1065, 2145, 705, 1161, 37801, 611, 573, 37932, 108)
LONG_TOKENS = (
    2, 1065, 573, 2674, 24464, 611, 573, 2731, 8811, 578, 2507,
    573, 8123, 578, 2674, 24464, 611, 573, 1833, 8811, 108,
)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: pathlib.Path, value: dict[str, Any], *, force: bool = False) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists; use --force to replace this stage")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def git_revision(module: Any) -> str | None:
    if getattr(module, "__file__", None) is None:
        return None
    location = pathlib.Path(module.__file__).resolve()
    # A source archive may live inside another Git worktree's devlocal tree.
    # Do not mistake that enclosing repository's revision for OpenPI's.
    repository = location.parents[2]
    if not (repository / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def image_to_uint8_hwc(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"image must be HWC or CHW rank 3, got {image.shape}")
    if image.shape[-1] != 3 and image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] != 3:
        raise ValueError(f"image must have three channels, got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("floating images must be finite and in [0, 1]")
        image = (image * 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        raise TypeError(f"images must be uint8 or float in [0, 1], got {image.dtype}")
    return np.ascontiguousarray(image)


def parse_token_ids(value: str) -> tuple[int, ...]:
    ids = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not ids or any(item < 0 or item >= 2**32 for item in ids):
        raise ValueError("token IDs must be a nonempty comma-separated list of uint32 values")
    return ids


def synthetic_observation(keys: list[str], state_key: str, state_dim: int) -> dict[str, np.ndarray]:
    y, x = np.indices((224, 224), dtype=np.uint16)
    result = {}
    for index, key in enumerate(keys):
        result[key] = np.stack(
            ((x + index * 41) % 256, (y * 3 + index * 29) % 256, (x + y * 2) % 256),
            axis=-1,
        ).astype(np.uint8)
    result[state_key] = np.linspace(-0.2, 0.2, state_dim, dtype=np.float32)
    return result


def load_source(path: pathlib.Path, keys: list[str], state_key: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        required = [*keys, state_key]
        missing = [key for key in required if key not in source]
        if missing:
            raise ValueError(f"{path} missing {missing}")
        observation = {key: np.array(source[key], copy=True) for key in required}
    for key in keys:
        image_to_uint8_hwc(observation[key])
    state = observation[state_key]
    if state.ndim != 1 or not np.issubdtype(state.dtype, np.number) or not np.isfinite(state).all():
        raise ValueError(f"{path}: state must be a finite numeric vector")
    return observation


def save_case(
    root: pathlib.Path,
    name: str,
    observation: dict[str, np.ndarray],
    prompt: str,
    noise: np.ndarray,
    token_ids: tuple[int, ...],
) -> dict[str, str]:
    path = root / "cases" / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, **observation, prompt=np.asarray(prompt), noise=noise,
        token_ids=np.asarray(token_ids, dtype=np.uint32),
    )
    return {"name": name, "path": str(path.relative_to(root)), "sha256": sha256(path)}


def prepare(args: argparse.Namespace) -> None:
    root = args.suite_dir
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and not args.force:
        raise FileExistsError(f"{manifest_path} exists; use a new suite directory or --force")
    if args.horizon <= 0 or args.model_dim <= 0 or args.state_dim <= 0:
        raise ValueError("horizon, model dimension, and state dimension must be positive")
    if args.action_dim is not None and args.action_dim <= 0:
        raise ValueError("--action-dim must be positive")
    if args.num_flow_steps <= 0:
        raise ValueError("--num-flow-steps must be positive")
    if args.interface == "bare" and args.discrete_state:
        raise ValueError("bare comparison uses explicit token IDs; use --interface policy for state-token preprocessing")
    if args.interface == "bare" and (args.prompt != SHORT_PROMPT or args.long_prompt != LONG_PROMPT):
        if args.short_token_ids is None or args.long_token_ids is None:
            raise ValueError("custom bare prompts require --short-token-ids and --long-token-ids")
    short_tokens = parse_token_ids(args.short_token_ids) if args.short_token_ids else SHORT_TOKENS
    long_tokens = parse_token_ids(args.long_token_ids) if args.long_token_ids else LONG_TOKENS
    keys = [part.strip() for part in args.image_keys.split(",") if part.strip()]
    if not 1 <= len(keys) <= 3 or len(set(keys)) != len(keys) or args.state_key in keys:
        raise ValueError("--image-keys needs 1-3 distinct keys, separate from --state-key")
    if len(args.source_npz) > 2:
        raise ValueError("at most two --source-npz observations keep the suite bounded")
    if args.force:
        for old in ("openpi.json", "apxinf.json", "report.json"):
            (root / old).unlink(missing_ok=True)
    sources = [load_source(path, keys, args.state_key) for path in args.source_npz]
    if sources and any(source[args.state_key].size != args.state_dim for source in sources):
        raise ValueError("source state width differs from --state-dim")
    base = sources[0] if sources else synthetic_observation(keys, args.state_key, args.state_dim)
    if args.interface == "bare":
        for source in [base, *sources[1:]]:
            for key in keys:
                if image_to_uint8_hwc(source[key]).shape != (224, 224, 3):
                    raise ValueError("bare comparison requires already-resized 224x224 RGB source images")
    rng = np.random.default_rng(args.seed)
    shape = (args.horizon, args.model_dim)
    first_noise = rng.standard_normal(shape).astype(np.float32)
    second_noise = rng.standard_normal(shape).astype(np.float32)
    cases = [save_case(root, "typical", base, args.prompt, first_noise, short_tokens)]
    if len(sources) == 2:
        cases.append(save_case(root, "second-scene", sources[1], args.prompt, second_noise, short_tokens))
    else:
        cases.append(save_case(root, "second-noise", base, args.prompt, second_noise, short_tokens))
    cases.append(save_case(root, "long-language", base, args.long_prompt, first_noise, long_tokens))
    dark = {key: np.zeros_like(image_to_uint8_hwc(base[key])) for key in keys}
    dark[args.state_key] = np.zeros(args.state_dim, dtype=np.float32)
    cases.append(save_case(root, "dark-zero-noise", dark, args.prompt, np.zeros(shape, np.float32), short_tokens))
    high = {key: np.full_like(image_to_uint8_hwc(base[key]), 255) for key in keys}
    high[args.state_key] = np.linspace(-0.5, 0.5, args.state_dim, dtype=np.float32)
    cases.append(save_case(root, "bright-negative-noise", high, args.prompt, -first_noise, short_tokens))
    mixed = {}
    for index, key in enumerate(keys):
        image = image_to_uint8_hwc(base[key])
        # Float CHW and a non-square resolution exercise parse + resize paths.
        resized = image[::2] if index == 0 and args.interface == "policy" else image
        mixed[key] = np.moveaxis(resized, -1, 0).astype(np.float32) / 255
    mixed[args.state_key] = np.asarray(base[args.state_key], dtype=np.float64)
    cases.append(save_case(root, "float-chw-resize" if args.interface == "policy" else "float-chw", mixed, args.prompt, first_noise, short_tokens))
    contrast = {
        key: np.full((224, 224, 3), (index * 83 + 17) % 256, dtype=np.uint8)
        for index, key in enumerate(keys)
    }
    contrast[args.state_key] = np.asarray(base[args.state_key], dtype=np.float32)
    cases.append(save_case(root, "view-order-contrast", contrast, args.prompt, first_noise, short_tokens))
    write_json(
        manifest_path,
        {
            "schema": SCHEMA,
            "interface": args.interface,
            "seed": args.seed,
            "image_keys": keys,
            "state_key": args.state_key,
            "state_dim": args.state_dim,
            "horizon": args.horizon,
            "model_dim": args.model_dim,
            "num_flow_steps": args.num_flow_steps,
            "action_dim": args.action_dim if args.action_dim is not None else (32 if args.interface == "bare" else 7),
            "discrete_state": args.discrete_state,
            "source_paths": [str(path.resolve()) for path in args.source_npz],
            "representative": bool(sources),
            "cases": cases,
        },
        force=args.force,
    )
    print(f"Prepared {len(cases)} cases in {root} (representative={bool(sources)})")


def load_suite(root: pathlib.Path) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported or missing parity suite schema")
    for case in manifest["cases"]:
        if sha256(root / case["path"]) != case["sha256"]:
            raise ValueError(f"case input changed: {case['name']}")
    return manifest


def load_case(root: pathlib.Path, case: dict[str, str]) -> tuple[dict, np.ndarray]:
    with np.load(root / case["path"], allow_pickle=False) as archive:
        observation = {key: np.array(archive[key], copy=True) for key in archive.files if key not in ("noise", "prompt")}
        observation["prompt"] = str(archive["prompt"].item())
        noise = np.array(archive["noise"], dtype=np.float32, copy=True)
    return observation, noise


def checkpoint_hash(path: pathlib.Path) -> str:
    weight = path / "model.safetensors"
    if not weight.is_file():
        raise FileNotFoundError(f"expected OpenPI PyTorch weight file {weight}")
    return sha256(weight)


class OpenPiBare:
    """Official PyTorch PI0.5 with explicit images, token IDs, and latent."""

    def __init__(self, checkpoint: pathlib.Path, manifest: dict[str, Any], device: str):
        import torch
        from safetensors.torch import load_file
        from openpi.models.pi0_config import Pi0Config
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

        self.torch = torch
        self.device = torch.device(device)
        self.keys = manifest["image_keys"]
        self.steps = manifest["num_flow_steps"]
        model = PI0Pytorch(Pi0Config(
            pi05=True, action_horizon=manifest["horizon"], pytorch_compile_mode=None,
        ))
        weights = load_file(str(checkpoint / "model.safetensors"))
        embedding = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
        head = "paligemma_with_expert.paligemma.lm_head.weight"
        if embedding not in weights:
            weights[embedding] = weights[head]
        model.load_state_dict(weights, strict=True)
        del weights
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        self.model = model.eval().to(self.device)

    def infer(self, observation: dict[str, Any], *, noise: np.ndarray) -> dict[str, np.ndarray]:
        from types import SimpleNamespace

        torch = self.torch
        names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        images = {}
        masks = {}
        for index, name in enumerate(names):
            if index < len(self.keys):
                image = image_to_uint8_hwc(observation[self.keys[index]])
                tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().to(self.device)
                images[name] = tensor.float().div(255).mul(2).sub(1)[None]
            else:
                images[name] = torch.zeros_like(images[names[0]])
            masks[name] = torch.tensor([index < len(self.keys)], device=self.device)
        ids = np.asarray(observation["token_ids"], dtype=np.int64)
        model_input = SimpleNamespace(
            images=images,
            image_masks=masks,
            state=torch.zeros((1, 32), device=self.device),
            tokenized_prompt=torch.from_numpy(ids[None]).to(self.device),
            tokenized_prompt_mask=torch.ones((1, ids.size), dtype=torch.bool, device=self.device),
            token_ar_mask=None,
            token_loss_mask=None,
        )
        latent = torch.from_numpy(noise[None]).to(self.device)
        with torch.inference_mode():
            actions = self.model.sample_actions(
                self.device, model_input, noise=latent, num_steps=self.steps
            )[0].float().cpu().numpy()
        return {"actions": actions}


class ApxInfBare:
    def __init__(self, policy: Any, keys: list[str]):
        self.policy = policy
        self.keys = keys

    def infer(self, observation: dict[str, Any], *, noise: np.ndarray) -> dict[str, np.ndarray]:
        rgb = np.stack([image_to_uint8_hwc(observation[key]) for key in self.keys])
        ids = np.asarray(observation["token_ids"], dtype=np.uint32)
        return {"actions": self.policy.model_runner.infer_rgb(rgb, "nhwc", ids, noise)}


def collect(
    root: pathlib.Path,
    manifest: dict[str, Any],
    policy: Any,
    engine: str,
    metadata: dict[str, Any],
    *,
    force: bool,
) -> None:
    rows = []
    for case in manifest["cases"]:
        observation, noise = load_case(root, case)
        if manifest["interface"] == "policy":
            observation.pop("token_ids")
        result = policy.infer(observation, noise=noise)
        actions = np.asarray(result["actions"], dtype=np.float32)
        expected = (manifest["horizon"], manifest["action_dim"])
        if actions.shape != expected or not np.isfinite(actions).all():
            raise ValueError(f"{engine}/{case['name']}: output shape or values invalid: {actions.shape}, expected {expected}")
        rows.append({"name": case["name"], "input_sha256": case["sha256"], "actions": actions.tolist()})
        print(f"{engine}: {case['name']} {actions.shape}", flush=True)
    write_json(root / f"{engine}.json", {"schema": SCHEMA, "engine": engine, **metadata, "cases": rows}, force=force)


def run_openpi(args: argparse.Namespace) -> None:
    manifest = load_suite(args.suite_dir)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import openpi  # type: ignore[import-not-found]
    if manifest["interface"] == "bare":
        policy = OpenPiBare(args.checkpoint_dir, manifest, args.device)
    else:
        from openpi.policies import policy_config  # type: ignore[import-not-found]
        from openpi.training import config  # type: ignore[import-not-found]

        policy = policy_config.create_trained_policy(
            config.get_config(args.config_name),
            args.checkpoint_dir,
            pytorch_device=args.device,
            sample_kwargs={"num_steps": manifest["num_flow_steps"]},
        )
    collect(
        args.suite_dir,
        manifest,
        policy,
        "openpi",
        {"revision": args.openpi_revision or git_revision(openpi), "config_name": args.config_name if manifest["interface"] == "policy" else None,
         "checkpoint_sha256": checkpoint_hash(args.checkpoint_dir)},
        force=args.force,
    )


def run_apxinf(args: argparse.Namespace) -> None:
    manifest = load_suite(args.suite_dir)
    source_path = pathlib.Path(__file__).resolve().parents[1] / "python/apxinf"
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))
    from apxinf import Pi05Policy

    policy = Pi05Policy.from_pretrained(
        args.checkpoint_dir,
        device=args.device,
        model_variant={"bf16": "bf16", "fp8": "fp8_static", "int8": "int8_dynamic"}[args.precision],
        image_keys=manifest["image_keys"],
        state_key=manifest["state_key"] if manifest["discrete_state"] else None,
        discrete_state=manifest["discrete_state"],
        num_views=len(manifest["image_keys"]),
        action_dim=manifest["action_dim"] if manifest["interface"] == "policy" else None,
        action_horizon=manifest["horizon"],
        num_flow_steps=manifest["num_flow_steps"],
        flow_start_time=1.0,
        norm_dtype="float64",
        calibration=args.calibration,
    )
    try:
        runner = ApxInfBare(policy, manifest["image_keys"]) if manifest["interface"] == "bare" else policy
        collect(
            args.suite_dir,
            manifest,
            runner,
            "apxinf",
            {"precision": args.precision, "checkpoint_sha256": checkpoint_hash(args.checkpoint_dir),
             "calibration_sha256": sha256(args.calibration) if args.calibration else None},
            force=args.force,
        )
    finally:
        policy.close()


def metrics(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    if reference.shape != actual.shape or not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError("outputs must have equal shapes and finite values")
    a = actual.astype(np.float64).ravel()
    b = reference.astype(np.float64).ravel()
    delta = a - b
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm == 0 or b_norm == 0:
        cosine = 1.0 if a_norm == b_norm else 0.0
    else:
        cosine = float(np.dot(a, b) / (a_norm * b_norm))
    delta_norm = float(np.linalg.norm(delta))
    relative_l2 = delta_norm / b_norm if b_norm else (0.0 if delta_norm == 0 else float(np.finfo(np.float64).max))
    return {
        "cosine": max(-1.0, min(1.0, cosine)),
        "relative_l2": relative_l2,
        "max_abs": float(np.abs(delta).max(initial=0)),
    }


def compare(args: argparse.Namespace) -> int:
    if not -1 <= args.min_cosine <= 1 or args.max_relative_l2 < 0:
        raise ValueError("--min-cosine must be in [-1, 1] and --max-relative-l2 must be nonnegative")
    if args.max_abs is not None and args.max_abs < 0:
        raise ValueError("--max-abs must be nonnegative")
    manifest = load_suite(args.suite_dir)
    reference = json.loads((args.suite_dir / "openpi.json").read_text())
    actual = json.loads((args.suite_dir / "apxinf.json").read_text())
    if reference.get("schema") != SCHEMA or actual.get("schema") != SCHEMA:
        raise ValueError("result schema mismatch")
    if reference["checkpoint_sha256"] != actual["checkpoint_sha256"]:
        raise ValueError("OpenPI and ApxInf did not load identical model.safetensors weights")
    expected_names = [case["name"] for case in manifest["cases"]]
    if [case["name"] for case in reference["cases"]] != expected_names or [case["name"] for case in actual["cases"]] != expected_names:
        raise ValueError("result cases differ from the manifest")
    rows = []
    for case, left, right in zip(manifest["cases"], reference["cases"], actual["cases"], strict=True):
        if left["input_sha256"] != case["sha256"] or right["input_sha256"] != case["sha256"]:
            raise ValueError(f"case input digest mismatch: {case['name']}")
        result = metrics(np.asarray(left["actions"]), np.asarray(right["actions"]))
        passed = result["cosine"] >= args.min_cosine and result["relative_l2"] <= args.max_relative_l2
        if args.max_abs is not None:
            passed &= result["max_abs"] <= args.max_abs
        rows.append({"name": case["name"], **result, "passed": bool(passed)})
        print(f"{case['name']}: cosine={result['cosine']:.8f} relative_l2={result['relative_l2']:.6f} max_abs={result['max_abs']:.6f} {'PASS' if passed else 'FAIL'}")
    report = {
        "schema": SCHEMA,
        "passed": all(row["passed"] for row in rows),
        "representative": manifest["representative"],
        "profile": {
            "interface": manifest["interface"],
            "views": len(manifest["image_keys"]),
            "horizon": manifest["horizon"],
            "num_flow_steps": manifest["num_flow_steps"],
            "action_dim": manifest["action_dim"],
            "discrete_state": manifest["discrete_state"],
            "openpi_config": reference["config_name"],
            "apxinf_precision": actual["precision"],
        },
        "thresholds": {"min_cosine": args.min_cosine, "max_relative_l2": args.max_relative_l2, "max_abs": args.max_abs},
        "openpi_revision": reference["revision"],
        "checkpoint_sha256": reference["checkpoint_sha256"],
        "calibration_sha256": actual.get("calibration_sha256"),
        "cases": rows,
    }
    write_json(args.suite_dir / "report.json", report, force=args.force)
    return 0 if report["passed"] else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "openpi", "apxinf", "compare"):
        command = sub.add_parser(name)
        command.add_argument("--suite-dir", type=pathlib.Path, default=DEFAULT_DIR)
        command.add_argument("--force", action="store_true")
        if name in ("openpi", "apxinf"):
            command.add_argument("--checkpoint-dir", type=pathlib.Path, required=True)
            command.add_argument("--device", default="cuda:0")
        if name == "prepare":
            command.add_argument("--interface", choices=("bare", "policy"), default="bare")
            command.add_argument("--image-keys", default="observation/image,observation/wrist_image")
            command.add_argument("--state-key", default="observation/state")
            command.add_argument("--source-npz", type=pathlib.Path, action="append", default=[])
            command.add_argument("--state-dim", type=int, default=8)
            command.add_argument("--horizon", type=int, default=10)
            command.add_argument("--model-dim", type=int, default=32)
            command.add_argument("--num-flow-steps", type=int, default=10)
            command.add_argument("--action-dim", type=int)
            command.add_argument("--seed", type=int, default=7)
            command.add_argument("--prompt", default=SHORT_PROMPT)
            command.add_argument("--long-prompt", default=LONG_PROMPT)
            command.add_argument("--short-token-ids", help="comma-separated IDs for a custom bare prompt")
            command.add_argument("--long-token-ids", help="comma-separated IDs for a custom bare long prompt")
            command.add_argument("--discrete-state", action="store_true")
        elif name == "openpi":
            command.add_argument("--config-name", default="pi05_libero")
            command.add_argument("--openpi-revision", help="commit of an OpenPI source archive without .git metadata")
        elif name == "apxinf":
            command.add_argument("--precision", choices=("bf16", "fp8", "int8"), default="fp8")
            command.add_argument("--calibration", type=pathlib.Path, help="FP8 calibration JSON when it is not in the checkpoint directory")
        else:
            command.add_argument("--min-cosine", type=float, default=0.997)
            command.add_argument("--max-relative-l2", type=float, default=0.10)
            command.add_argument("--max-abs", type=float)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare":
        prepare(args)
    elif args.command == "openpi":
        run_openpi(args)
    elif args.command == "apxinf":
        run_apxinf(args)
    else:
        return compare(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
