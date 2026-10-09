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
import math
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np


SCHEMA = "apxinf.pi05.openpi-parity.v4"
DEFAULT_DIR = Path(__file__).resolve().parents[1] / "devlocal/pi05-ci-gate/default"
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


def cuda_runtime_libraries() -> dict[str, dict[str, str]]:
    """Record the loaded DSOs, rather than infer them from the installed toolkit."""
    maps = Path("/proc/self/maps")
    if not maps.exists():
        return {}
    paths = {line.split()[-1] for line in maps.read_text().splitlines()}
    libraries = {}
    for name in ("libcuda", "libcudart", "libcublas", "libcublasLt"):
        loaded = [Path(path) for path in paths if Path(path).name.startswith(name + ".so")]
        if len(loaded) > 1:
            raise ValueError(f"multiple loaded {name} libraries")
        if loaded:
            libraries[name] = {"path": str(loaded[0]), "sha256": sha256(loaded[0])}
    return libraries


def write_json(path: Path, value: dict[str, Any], *, force: bool = False) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists; use --force or a new suite directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
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
    if args.horizon < 1 or args.num_flow_steps < 1 or args.max_token_len < 1:
        raise ValueError("horizon and flow steps must be positive")
    if not (args.case_npz or args.source_npz or args.diagnostic):
        raise ValueError("supply frozen --case-npz observations or explicitly use --diagnostic")
    cases = []
    provenance = json.loads(args.provenance_json.read_text()) if args.provenance_json else {}
    for path in args.case_npz:
        with np.load(path, allow_pickle=False) as archive:
            images = np.stack([image_to_uint8_hwc(image) for image in archive["images"]])
            if len(images) < len(keys):
                raise ValueError("saved observation has too few cameras")
            tokens, noise = np.asarray(archive["token_ids"]), np.asarray(archive["noise"])
            if tokens.ndim != 1 or not tokens.size or not np.issubdtype(tokens.dtype, np.integer):
                raise ValueError("token_ids must be a nonempty integer vector")
            if tokens.min() < 0 or tokens.max() >= 257152:
                raise ValueError("token outside PI05 vocabulary")
            if noise.shape != (args.horizon, 32) or not np.isfinite(noise).all():
                raise ValueError("noise must be finite with shape (horizon, 32)")
            entry = save_case(root, path.stem, images[:len(keys)], tuple(tokens), noise.astype(np.float32))
            if args.provenance_json:
                entry["provenance"] = provenance[path.stem]
            cases.append(entry)
    sources = [source_images(path, keys) for path in args.source_npz]
    rng = np.random.default_rng(args.seed)
    first = rng.standard_normal((args.horizon, 32)).astype(np.float32)
    for index, source in enumerate(sources):
        cases.append(save_case(root, f"replay-observation-{index}", source, SHORT_TOKENS, first))
    if args.diagnostic:
        base = synthetic_images(len(keys))
        second = rng.standard_normal((args.horizon, 32)).astype(np.float32)
        zeros = np.zeros_like(first)
        black, white = np.zeros_like(base), np.full_like(base, 255)
        contrast = np.stack([np.full_like(base[0], (view * 83 + 17) % 256)
                             for view in range(len(keys))])
        boundary_tokens = tuple((SHORT_TOKENS * ((args.max_token_len + 9) // 10))[:args.max_token_len])
        diagnostic = [
            ("gradient-t10-normal-noise", base, SHORT_TOKENS, first),
            ("gradient-t10-second-noise", base, SHORT_TOKENS, second),
            ("gradient-t21-normal-noise", base, LONG_TOKENS, first),
            (f"gradient-t{args.max_token_len}-boundary", base, boundary_tokens, first),
            ("black-normal-noise", black, SHORT_TOKENS, first),
            ("gradient-zero-noise", base, SHORT_TOKENS, zeros),
            ("black-zero-noise-combined", black, SHORT_TOKENS, zeros),
            ("white-normal-noise", white, SHORT_TOKENS, first),
            ("gradient-negative-noise", base, SHORT_TOKENS, -first),
            ("white-negative-noise-combined", white, SHORT_TOKENS, -first),
            ("camera-distinct-gray", contrast, SHORT_TOKENS, first),
        ]
        cases.extend(save_case(root, *case) for case in diagnostic)
    if args.force:
        for name in ("openpi.json", "apxinf.json", "report.json"):
            (root / name).unlink(missing_ok=True)
    kind = "frozen-model-inputs" if args.case_npz else "image-replay" if args.source_npz else "diagnostic"
    if args.case_npz and (args.source_npz or args.diagnostic):
        kind = "mixed"
    write_json(root / "manifest.json", {
        "schema": SCHEMA, "image_keys": keys, "horizon": args.horizon,
        "num_flow_steps": args.num_flow_steps, "seed": args.seed,
        "source_paths": [str(path.resolve()) for path in args.case_npz + args.source_npz],
        "input_kind": kind, "representative": False, "cases": cases,
    }, force=args.force)
    load_suite(root)
    print(f"Prepared {len(cases)} cases in {root} (input_kind={kind}, representative=False)")


def load_suite(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported or missing parity suite schema")
    if not manifest["cases"] or len({case["name"] for case in manifest["cases"]}) != len(manifest["cases"]):
        raise ValueError("suite must contain distinct cases")
    if not 1 <= len(manifest["image_keys"]) <= 3 or manifest["horizon"] < 1 or manifest["num_flow_steps"] < 1:
        raise ValueError("invalid view count, horizon or flow steps")
    for case in manifest["cases"]:
        path = (root / case["path"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("case path escapes suite")
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

def checked_infer(model: Any, case: dict[str, np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    actions = np.asarray(model.infer(case), dtype=np.float32)
    if actions.shape != shape or not np.isfinite(actions).all():
        raise ValueError(f"invalid output {actions.shape}, expected finite {shape}")
    return actions.copy()


def stability(model: Any, root: Path, manifest: dict[str, Any], rows: list[dict],
              repeats: int) -> dict[str, Any] | None:
    """Separate phase: repeated inputs and A -> all other inputs -> A."""
    if repeats == 0:
        return None
    shape = (manifest["horizon"], 32)
    results = []
    for entry, row in zip(manifest["cases"], rows, strict=True):
        original = np.asarray(row["actions"], dtype=np.float32)
        case = load_case(root, entry)
        drift = max(float(np.abs(checked_infer(model, case, shape) - original).max())
                    for _ in range(repeats))
        results.append({"name": entry["name"], "repeat_max_abs": drift})
    checked_infer(model, load_case(root, manifest["cases"][0]), shape)
    for entry in manifest["cases"][1:]:
        checked_infer(model, load_case(root, entry), shape)
    revisited = checked_infer(model, load_case(root, manifest["cases"][0]), shape)
    return {"repeats": repeats, "cases": results,
            "revisit_max_abs": float(np.abs(revisited - np.asarray(rows[0]["actions"])).max())}


def collect(root: Path, manifest: dict[str, Any], model: Any,
            engine: str, metadata: dict[str, Any], *, force: bool,
            output: Path | None = None, stability_repeats: int = 0) -> None:
    rows = []
    for entry in manifest["cases"]:
        actions = checked_infer(model, load_case(root, entry), (manifest["horizon"], 32))
        rows.append({"name": entry["name"], "input_sha256": entry["sha256"],
                     "actions": actions.tolist()})
        print(f"{engine}: {entry['name']} {actions.shape}", flush=True)
    write_json(output or root / f"{engine}.json", {
        "schema": SCHEMA, "engine": engine, "suite_sha256": sha256(root / "manifest.json"),
        **metadata, "runtime_libraries": cuda_runtime_libraries(), "cases": rows,
        "stability": stability(model, root, manifest, rows, stability_repeats),
    }, force=force)


def run_openpi(args: argparse.Namespace) -> None:
    manifest = load_suite(args.suite_dir)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import openpi

    model = OpenPiModel(args.checkpoint_dir, manifest, args.device)
    revision = args.openpi_revision or openpi_revision(openpi)
    if not revision:
        raise ValueError("reference source lacks Git metadata; supply --openpi-revision")
    device = {"type": model.device.type}
    hardware = model.device.type
    if model.device.type == "cuda":
        properties = model.torch.cuda.get_device_properties(model.device)
        capability = (properties.major, properties.minor)
        hardware = {(11, 0): "thor", (8, 7): "orin"}.get(capability, f"sm{properties.major}{properties.minor}")
        device.update(name=properties.name, compute_capability=list(capability))
    collect(args.suite_dir, manifest, model, "openpi", {
        "revision": revision, "hardware": hardware, "device": device,
        "torch_version": str(model.torch.__version__), "torch_cuda_version": model.torch.version.cuda,
        "float32_matmul_precision": model.torch.get_float32_matmul_precision(),
        "matmul_allow_tf32": model.torch.backends.cuda.matmul.allow_tf32,
        "matmul_allow_bf16_reduced_precision_reduction": model.torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "matmul_allow_fp16_reduced_precision_reduction": model.torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        "cudnn_allow_tf32": model.torch.backends.cudnn.allow_tf32,
        "config_sha256": sha256(args.checkpoint_dir / "config.json"),
        "checkpoint_sha256": checkpoint_hash(args.checkpoint_dir),
    }, force=args.force, output=args.output, stability_repeats=args.stability_repeats)


def run_apxinf(args: argparse.Namespace) -> None:
    import apxinf_py

    manifest = load_suite(args.suite_dir)
    model = ApxInfModel(args.checkpoint_dir, manifest, args.device,
                        args.precision, args.calibration)
    collect(args.suite_dir, manifest, model, "apxinf", {
        "precision": args.precision,
        "config_sha256": sha256(args.checkpoint_dir / "config.json"),
        "checkpoint_sha256": checkpoint_hash(args.checkpoint_dir),
        "calibration_sha256": sha256(args.calibration) if args.calibration else None,
        "revision": args.revision, "hardware": args.hardware,
        "extension_sha256": sha256(Path(apxinf_py.__file__)),
    }, force=args.force, output=args.output, stability_repeats=args.stability_repeats)


def metrics(reference: np.ndarray, actual: np.ndarray) -> dict[str, float | None]:
    if reference.shape != actual.shape or not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError("outputs must have equal shapes and finite values")
    left, right = reference.astype(np.float64).ravel(), actual.astype(np.float64).ravel()
    delta = right - left
    left_norm, right_norm = np.linalg.norm(left), np.linalg.norm(right)
    # Equal nonzero vectors have cosine exactly one; avoid a one-ULP false failure.
    cosine = None
    if left_norm and right_norm:
        cosine = 1.0 if not np.any(delta) else float(np.dot(left, right) / (left_norm * right_norm))
    relative_l2 = (float(np.linalg.norm(delta) / left_norm) if left_norm else
                   None)
    return {"cosine": max(-1.0, min(1.0, cosine)) if cosine is not None else None,
            "relative_l2": relative_l2,
            "max_abs": float(np.abs(delta).max(initial=0))}


def meets_limits(result: dict[str, Any], limits: dict[str, float]) -> bool:
    if result["max_abs"] > limits["max_abs"]:
        return False
    if result["cosine"] is None or result["relative_l2"] is None:
        return result["max_abs"] <= limits["zero_max_abs"]
    return result["cosine"] >= limits["min_cosine"] and result["relative_l2"] <= limits["max_relative_l2"]


def compare_results(root: Path, reference: dict[str, Any], actual: dict[str, Any],
                    limits: dict[str, float]) -> list[dict[str, Any]]:
    manifest = load_suite(root)
    for result in (reference, actual):
        if result.get("schema") != SCHEMA or result.get("suite_sha256") != sha256(root / "manifest.json"):
            raise ValueError("result suite/schema mismatch")
        if [row["name"] for row in result["cases"]] != [row["name"] for row in manifest["cases"]]:
            raise ValueError("result cases differ from manifest")
    if reference["checkpoint_sha256"] != actual["checkpoint_sha256"]:
        raise ValueError("reference and candidate loaded different checkpoint weights")
    rows = []
    for entry, left, right in zip(manifest["cases"], reference["cases"], actual["cases"], strict=True):
        if left["input_sha256"] != entry["sha256"] or right["input_sha256"] != entry["sha256"]:
            raise ValueError(f"input digest mismatch: {entry['name']}")
        a, b = np.asarray(left["actions"]), np.asarray(right["actions"])
        if a.shape != (manifest["horizon"], 32) or a.shape != b.shape:
            raise ValueError("unexpected action shape")
        result = metrics(a, b)
        steps = [metrics(x, y) for x, y in zip(a, b, strict=True)]
        passed = meets_limits(result, limits) and all(meets_limits(step, limits) for step in steps)
        rows.append({"name": entry["name"], **result, "steps": steps, "passed": passed})
    return rows


def compare(args: argparse.Namespace) -> int:
    values = (args.min_cosine, args.max_relative_l2, args.max_abs, args.zero_max_abs)
    if not all(math.isfinite(v) for v in values) or not -1 <= args.min_cosine <= 1 or min(values[1:]) < 0:
        raise ValueError("invalid comparison thresholds")
    manifest = load_suite(args.suite_dir)
    reference = json.loads((args.reference or args.suite_dir / "openpi.json").read_text())
    actual = json.loads((args.actual or args.suite_dir / "apxinf.json").read_text())
    limits = {"min_cosine": args.min_cosine, "max_relative_l2": args.max_relative_l2,
              "max_abs": args.max_abs, "zero_max_abs": args.zero_max_abs}
    rows = compare_results(args.suite_dir, reference, actual, limits)
    for row in rows:
        print(f"{row['name']}: cosine={row['cosine']} relative_l2={row['relative_l2']} "
              f"max_abs={row['max_abs']:.6f} {'PASS' if row['passed'] else 'FAIL'}")
    write_json(args.output or args.suite_dir / "report.json", {
        "schema": SCHEMA, "passed": all(row["passed"] for row in rows),
        "representative": manifest["representative"],
        "views": len(manifest["image_keys"]), "horizon": manifest["horizon"],
        "num_flow_steps": manifest["num_flow_steps"],
        "thresholds": limits,
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
        command.add_argument("--output", type=Path)
        if name in ("openpi", "apxinf"):
            command.add_argument("--checkpoint-dir", type=Path, required=True)
            command.add_argument("--device", default="cuda:0")
            command.add_argument("--stability-repeats", type=int, default=0,
                                 help="separate stability phase; zero disables it, never a latency sample count")
        if name == "prepare":
            command.add_argument("--image-keys", default="observation/image,observation/wrist_image")
            command.add_argument("--source-npz", type=Path, action="append", default=[])
            command.add_argument("--case-npz", type=Path, action="append", default=[])
            command.add_argument("--diagnostic", action="store_true")
            command.add_argument("--provenance-json", type=Path,
                                 help="case stem -> task/trial/frame/camera/source metadata; frozen in manifest")
            command.add_argument("--horizon", type=int, default=50)
            command.add_argument("--num-flow-steps", type=int, default=10)
            command.add_argument("--seed", type=int, default=7)
            command.add_argument("--max-token-len", type=int, default=200,
                                 help="diagnostic boundary from checkpoint tokenizer_max_length/max_token_len")
        elif name == "openpi":
            command.add_argument("--openpi-revision")
        elif name == "apxinf":
            command.add_argument("--precision", choices=("bf16", "fp8", "int8"), default="fp8")
            command.add_argument("--calibration", type=Path)
            command.add_argument("--revision", required=True)
            command.add_argument("--hardware", required=True, choices=("thor", "orin"))
        else:
            command.add_argument("--min-cosine", type=float, default=0.997)
            command.add_argument("--max-relative-l2", type=float, default=0.10)
            command.add_argument("--max-abs", type=float, required=True)
            command.add_argument("--zero-max-abs", type=float, required=True)
            command.add_argument("--reference", type=Path)
            command.add_argument("--actual", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if getattr(args, "stability_repeats", 0) < 0:
        raise ValueError("stability repeats must be nonnegative")
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
