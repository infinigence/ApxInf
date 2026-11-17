"""Change only the language input of a verified Qwen3-VL image fixture."""

from __future__ import annotations

import hashlib
import json
import re
import shutil

import numpy as np

from formats import TENSOR_DTYPES, load_fixture


def encode_prompt(prompt, grid, backbone, checkpoint):
    from apxinf_py import HfTokenizer
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    processor_path = checkpoint / "processor_config.json"
    if not processor_path.is_file():
        processor_path = checkpoint / "processor" / "processor_config.json"
    processor = json.loads(processor_path.read_text())
    if processor.get("processor_kwargs", processor).get("formalize_language", True):
        prompt = re.sub(r"[^\w\s]", "", prompt.lower())
    image_config = json.loads((backbone / "preprocessor_config.json").read_text())
    if image_config.get("processor_class") != "Qwen3VLProcessor":
        raise ValueError("--prompt requires Qwen3VLProcessor image fixtures")
    merge_size = int(image_config["merge_size"])
    if merge_size <= 0 or np.any(grid[:, 1:] % merge_size):
        raise ValueError("image grid is incompatible with spatial merge size")
    template = json.loads((backbone / "chat_template.json").read_text())["chat_template"]
    content = [{"type": "image"} for _ in grid] + [{"type": "text", "text": prompt}]
    rendered = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True).from_string(
        template
    ).render(messages=[{"role": "user", "content": content}], add_generation_prompt=False)
    parts = rendered.split("<|image_pad|>")
    if len(parts) != len(grid) + 1:
        raise ValueError("chat template image placeholders do not match the fixture")
    expanded = parts[0] + "".join(
        "<|image_pad|>" * (int(np.prod(dimensions)) // merge_size**2) + suffix
        for dimensions, suffix in zip(grid, parts[1:])
    )
    tokenizer = HfTokenizer.from_file(backbone / "tokenizer.json")
    return np.asarray(tokenizer.encode_with_special_tokens(expanded, False), dtype="<u4")


def prepare_fixture(args):
    source = args.fixture.expanduser().resolve()
    manifest, arrays = load_fixture(source)
    if args.prompt is None:
        return source, manifest, arrays
    if not args.prompt.strip():
        raise ValueError("--prompt must not be empty")
    original_prompt = manifest.get("prompt")
    if not isinstance(original_prompt, str):
        raise ValueError("--prompt requires the original prompt in the fixture manifest")
    backbone = args.backbone.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    original = encode_prompt(original_prompt, arrays["image_grid_thw"], backbone, checkpoint)
    if not np.array_equal(original, arrays["token_ids"]) or not np.all(arrays["attention_mask"] == 1):
        raise ValueError("original prompt does not reproduce fixture tokens/mask; refusing replacement")
    tokens = encode_prompt(args.prompt, arrays["image_grid_thw"], backbone, checkpoint)
    changed = not np.array_equal(tokens, original)
    if changed and args.reference is not None:
        raise ValueError("--prompt changes tokens: omit --reference; generate a matching reference for the new fixture")
    if args.prompt == original_prompt:
        return source, manifest, arrays
    destination = args.output.expanduser().resolve().with_suffix(".fixture")
    if destination == source or source.is_relative_to(destination):
        raise ValueError("generated fixture must not overwrite the source fixture")
    destination.mkdir(parents=True, exist_ok=False)
    manifest["source_fixture"] = str(source)
    manifest["prompt"] = args.prompt
    manifest["fixture"] += "-prompt-" + hashlib.sha256(args.prompt.encode()).hexdigest()[:12]
    for name in TENSOR_DTYPES:
        entry = manifest["tensors"][name]
        filename = f"{name}.bin"
        if name in ("token_ids", "attention_mask"):
            values = tokens if name == "token_ids" else np.ones(tokens.shape, dtype="u1")
            values.tofile(destination / filename)
            entry["shape"] = list(values.shape)
        else:
            shutil.copyfile(source / entry["file"], destination / filename)
        entry["file"] = filename
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    manifest, arrays = load_fixture(destination)
    return destination, manifest, arrays
