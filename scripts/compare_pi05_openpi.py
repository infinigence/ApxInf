#!/usr/bin/env python3
"""Compare PI0.5 normalized model outputs with official OpenPI PyTorch.

Run prepare, openpi, apxinf, and compare in order. The inference stages may use
separate Python environments; they read the same saved inputs and checkpoint.
All generated data belongs under ignored devlocal/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np


SCHEMA = "apxinf.pi05.openpi-parity.v2"
DEFAULT_DIR = Path(__file__).resolve().parents[1] / "devlocal/pi05-openpi-parity/default"
SHORT_TOKENS = (2, 1065, 2145, 705, 1161, 37801, 611, 573, 37932, 108)
LONG_TOKENS = (
    2, 1065, 573, 2674, 24464, 611, 573, 2731, 8811, 578, 2507,
    573, 8123, 578, 2674, 24464, 611, 573, 1833, 8811, 108,
)
CAMERAS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any], *, force: bool = False) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists; use --force or a new suite directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def image_to_uint8_hwc(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"expected HWC or CHW image, got {image.shape}")
    if image.shape[-1] != 3 and image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] != 3:
        raise ValueError(f"expected RGB image, got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("float images must be finite and in [0, 1]")
        image = (image * 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        raise TypeError(f"expected uint8 or float image, got {image.dtype}")
    if image.shape != (224, 224, 3):
        raise ValueError(f"expected already-resized 224x224 RGB image, got {image.shape}")
    return np.ascontiguousarray(image)


def source_images(path: Path, keys: list[str]) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        missing = [key for key in keys if key not in archive]
        if missing:
            raise ValueError(f"{path} missing {missing}")
        return np.stack([image_to_uint8_hwc(archive[key]) for key in keys])


def synthetic_images(views: int) -> np.ndarray:
    y, x = np.indices((224, 224), dtype=np.uint16)
    return np.stack([
        np.stack(((x + view * 41) % 256, (y * 3 + view * 29) % 256,
                  (x + y * 2) % 256), axis=-1).astype(np.uint8)
        for view in range(views)
    ])


def save_case(root: Path, name: str, images: np.ndarray,
              tokens: tuple[int, ...], noise: np.ndarray) -> dict[str, str]:
    path = root / "cases" / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, images=images,
                        token_ids=np.asarray(tokens, dtype=np.uint32), noise=noise)
    return {"name": name, "path": str(path.relative_to(root)), "sha256": sha256(path)}


def prepare(args: argparse.Namespace) -> None:
    root = args.suite_dir
    if (root / "manifest.json").exists() and not args.force:
        raise FileExistsError(f"{root}/manifest.json exists; use --force or a new suite")
    keys = [key.strip() for key in args.image_keys.split(",") if key.strip()]
    if not 1 <= len(keys) <= 3 or len(set(keys)) != len(keys):
        raise ValueError("--image-keys requires 1-3 distinct keys")
    if len(args.source_npz) > 2:
        raise ValueError("at most two --source-npz inputs are supported")
    if args.horizon < 1 or args.num_flow_steps < 1:
        raise ValueError("horizon and flow steps must be positive")
    sources = [source_images(path, keys) for path in args.source_npz]
    base = sources[0] if sources else synthetic_images(len(keys))
    rng = np.random.default_rng(args.seed)
    first = rng.standard_normal((args.horizon, 32)).astype(np.float32)
    second = rng.standard_normal((args.horizon, 32)).astype(np.float32)
    cases = [save_case(root, "typical", base, SHORT_TOKENS, first)]
    cases.append(save_case(root, "second-scene" if len(sources) == 2 else "second-noise",
                           sources[1] if len(sources) == 2 else base, SHORT_TOKENS, second))
    cases.append(save_case(root, "long-language", base, LONG_TOKENS, first))
    cases.append(save_case(root, "dark-zero-noise", np.zeros_like(base),
                           SHORT_TOKENS, np.zeros_like(first)))
    cases.append(save_case(root, "bright-negative-noise", np.full_like(base, 255),
                           SHORT_TOKENS, -first))
    chw = np.moveaxis(base.astype(np.float32) / 255, -1, 1)
    cases.append(save_case(root, "float-chw", chw, SHORT_TOKENS, first))
    contrast = np.stack([
        np.full_like(base[0], (view * 83 + 17) % 256) for view in range(len(keys))
    ])
    cases.append(save_case(root, "view-order-contrast", contrast, SHORT_TOKENS, first))
    if args.force:
        for name in ("openpi.json", "apxinf.json", "report.json"):
            (root / name).unlink(missing_ok=True)
    write_json(root / "manifest.json", {
        "schema": SCHEMA, "image_keys": keys, "horizon": args.horizon,
        "num_flow_steps": args.num_flow_steps, "seed": args.seed,
        "source_paths": [str(path.resolve()) for path in args.source_npz],
        "representative": bool(sources), "cases": cases,
    }, force=args.force)
    print(f"Prepared {len(cases)} cases in {root} (representative={bool(sources)})")


def load_suite(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported or missing parity suite schema")
    for case in manifest["cases"]:
        if sha256(root / case["path"]) != case["sha256"]:
            raise ValueError(f"case input changed: {case['name']}")
    return manifest


def load_case(root: Path, case: dict[str, str]) -> dict[str, np.ndarray]:
    with np.load(root / case["path"], allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def checkpoint_hash(root: Path) -> str:
    weight = root / "model.safetensors"
    if not weight.is_file():
        raise FileNotFoundError(f"expected PyTorch weight file {weight}")
    return sha256(weight)


def openpi_revision(module: Any) -> str | None:
    root = Path(module.__file__).resolve().parents[2]
    if not (root / ".git").exists():
        return None
    result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


class OpenPiModel:
    def __init__(self, checkpoint: Path, manifest: dict[str, Any], device: str):
        import torch
        from safetensors.torch import load_file
        from openpi.models.pi0_config import Pi0Config
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

        self.torch = torch
        self.device = torch.device(device)
        self.views = len(manifest["image_keys"])
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

    def infer(self, case: dict[str, np.ndarray]) -> np.ndarray:
        from types import SimpleNamespace

        torch = self.torch
        images, masks = {}, {}
        for view, name in enumerate(CAMERAS):
            if view < self.views:
                image = image_to_uint8_hwc(case["images"][view])
                tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().to(self.device)
                images[name] = tensor.float().div(255).mul(2).sub(1)[None]
            else:
                images[name] = torch.zeros_like(images[CAMERAS[0]])
            masks[name] = torch.tensor([view < self.views], device=self.device)
        tokens = np.asarray(case["token_ids"], dtype=np.int64)
        observation = SimpleNamespace(
            images=images, image_masks=masks,
            state=torch.zeros((1, 32), device=self.device),
            tokenized_prompt=torch.from_numpy(tokens[None]).to(self.device),
            tokenized_prompt_mask=torch.ones((1, tokens.size), dtype=torch.bool, device=self.device),
            token_ar_mask=None, token_loss_mask=None,
        )
        noise = torch.from_numpy(case["noise"][None]).to(self.device)
        with torch.inference_mode():
            return self.model.sample_actions(self.device, observation, noise=noise,
                                             num_steps=self.steps)[0].float().cpu().numpy()


class ApxInfModel:
    def __init__(self, checkpoint: Path, manifest: dict[str, Any], device: str,
                 precision: str, calibration: Path | None):
        import apxinf_py

        self.model = apxinf_py.ModelRunner.load(
            "pi05", str(checkpoint / "model.safetensors"), device=device,
            model_variant={"bf16": "bf16", "fp8": "fp8_static", "int8": "int8_dynamic"}[precision],
            calibration=str(calibration) if calibration else None,
            config_json=(checkpoint / "config.json").read_text(),
            action_horizon=manifest["horizon"], num_views=len(manifest["image_keys"]),
            num_flow_steps=manifest["num_flow_steps"], flow_start_time=1.0,
        )

    def infer(self, case: dict[str, np.ndarray]) -> np.ndarray:
        images = np.stack([image_to_uint8_hwc(view) for view in case["images"]])
        tokens = np.asarray(case["token_ids"], dtype=np.uint32)
        noise = np.asarray(case["noise"], dtype=np.float32)
        return self.model.infer_rgb(images, "nhwc", tokens, noise)

def collect(root: Path, manifest: dict[str, Any], model: Any,
            engine: str, metadata: dict[str, Any], *, force: bool) -> None:
    rows = []
    expected = (manifest["horizon"], 32)
    for entry in manifest["cases"]:
        actions = np.asarray(model.infer(load_case(root, entry)), dtype=np.float32)
        if actions.shape != expected or not np.isfinite(actions).all():
            raise ValueError(f"{engine}/{entry['name']}: invalid output {actions.shape}, expected {expected}")
        rows.append({"name": entry["name"], "input_sha256": entry["sha256"],
                     "actions": actions.tolist()})
        print(f"{engine}: {entry['name']} {actions.shape}", flush=True)
    write_json(root / f"{engine}.json", {
        "schema": SCHEMA, "engine": engine, **metadata, "cases": rows,
    }, force=force)


def run_openpi(args: argparse.Namespace) -> None:
    manifest = load_suite(args.suite_dir)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import openpi

    model = OpenPiModel(args.checkpoint_dir, manifest, args.device)
    collect(args.suite_dir, manifest, model, "openpi", {
        "revision": args.openpi_revision or openpi_revision(openpi),
        "checkpoint_sha256": checkpoint_hash(args.checkpoint_dir),
    }, force=args.force)


def run_apxinf(args: argparse.Namespace) -> None:
    manifest = load_suite(args.suite_dir)
    model = ApxInfModel(args.checkpoint_dir, manifest, args.device,
                        args.precision, args.calibration)
    collect(args.suite_dir, manifest, model, "apxinf", {
        "precision": args.precision,
        "checkpoint_sha256": checkpoint_hash(args.checkpoint_dir),
        "calibration_sha256": sha256(args.calibration) if args.calibration else None,
    }, force=args.force)


def metrics(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    if reference.shape != actual.shape or not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError("outputs must have equal shapes and finite values")
    left, right = reference.astype(np.float64).ravel(), actual.astype(np.float64).ravel()
    delta = right - left
    left_norm, right_norm = np.linalg.norm(left), np.linalg.norm(right)
    cosine = (float(np.dot(left, right) / (left_norm * right_norm))
              if left_norm and right_norm else float(left_norm == right_norm))
    relative_l2 = (float(np.linalg.norm(delta) / left_norm) if left_norm else
                   (0.0 if not np.any(delta) else float(np.finfo(np.float64).max)))
    return {"cosine": max(-1.0, min(1.0, cosine)), "relative_l2": relative_l2,
            "max_abs": float(np.abs(delta).max(initial=0))}


def compare(args: argparse.Namespace) -> int:
    if not -1 <= args.min_cosine <= 1 or args.max_relative_l2 < 0:
        raise ValueError("invalid comparison thresholds")
    manifest = load_suite(args.suite_dir)
    reference = json.loads((args.suite_dir / "openpi.json").read_text())
    actual = json.loads((args.suite_dir / "apxinf.json").read_text())
    if reference.get("schema") != SCHEMA or actual.get("schema") != SCHEMA:
        raise ValueError("result schema mismatch")
    if reference["checkpoint_sha256"] != actual["checkpoint_sha256"]:
        raise ValueError("OpenPI and ApxInf loaded different model.safetensors weights")
    names = [entry["name"] for entry in manifest["cases"]]
    if [row["name"] for row in reference["cases"]] != names or [row["name"] for row in actual["cases"]] != names:
        raise ValueError("result cases differ from manifest")
    rows = []
    for entry, left, right in zip(manifest["cases"], reference["cases"], actual["cases"], strict=True):
        if left["input_sha256"] != entry["sha256"] or right["input_sha256"] != entry["sha256"]:
            raise ValueError(f"input digest mismatch: {entry['name']}")
        result = metrics(np.asarray(left["actions"]), np.asarray(right["actions"]))
        passed = result["cosine"] >= args.min_cosine and result["relative_l2"] <= args.max_relative_l2
        rows.append({"name": entry["name"], **result, "passed": bool(passed)})
        print(f"{entry['name']}: cosine={result['cosine']:.8f} "
              f"relative_l2={result['relative_l2']:.6f} max_abs={result['max_abs']:.6f} "
              f"{'PASS' if passed else 'FAIL'}")
    write_json(args.suite_dir / "report.json", {
        "schema": SCHEMA, "passed": all(row["passed"] for row in rows),
        "representative": manifest["representative"],
        "views": len(manifest["image_keys"]), "horizon": manifest["horizon"],
        "num_flow_steps": manifest["num_flow_steps"],
        "thresholds": {"min_cosine": args.min_cosine, "max_relative_l2": args.max_relative_l2},
        "openpi_revision": reference["revision"],
        "checkpoint_sha256": reference["checkpoint_sha256"],
        "apxinf_precision": actual["precision"],
        "calibration_sha256": actual.get("calibration_sha256"), "cases": rows,
    }, force=args.force)
    return 0 if all(row["passed"] for row in rows) else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "openpi", "apxinf", "compare"):
        command = sub.add_parser(name)
        command.add_argument("--suite-dir", type=Path, default=DEFAULT_DIR)
        command.add_argument("--force", action="store_true")
        if name in ("openpi", "apxinf"):
            command.add_argument("--checkpoint-dir", type=Path, required=True)
            command.add_argument("--device", default="cuda:0")
        if name == "prepare":
            command.add_argument("--image-keys", default="observation/image,observation/wrist_image")
            command.add_argument("--source-npz", type=Path, action="append", default=[])
            command.add_argument("--horizon", type=int, default=10)
            command.add_argument("--num-flow-steps", type=int, default=10)
            command.add_argument("--seed", type=int, default=7)
        elif name == "openpi":
            command.add_argument("--openpi-revision")
        elif name == "apxinf":
            command.add_argument("--precision", choices=("bf16", "fp8", "int8"), default="fp8")
            command.add_argument("--calibration", type=Path)
        else:
            command.add_argument("--min-cosine", type=float, default=0.997)
            command.add_argument("--max-relative-l2", type=float, default=0.10)
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
