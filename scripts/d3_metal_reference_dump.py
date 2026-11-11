"""Record the PyTorch references the `metal_d3` image device tests compare against.

One case is one still image plus one question. For each case this writes the processor's
`pixel_values`, the vision tower's merged `pooler_output`, the expanded prompt token ids and
the masked-softmax probabilities over the first `option_count` answer codes, in the layout
`crates/apxinf-model/src/metal_d3/{model,preprocess}.rs` read.

The numbers come from the runtime's public image path: `D3.prepare` renders and plans the
prompt, `D3.image_probabilities` runs it. The processor and the vision tower are both wrapped
so every array written here is a tensor that one forward pass produced or consumed. Nothing is
recomputed afterwards, so a `pixel_values` file and the `probabilities` beside it cannot drift.

Run:
    python3 scripts/d3_metal_reference_dump.py <checkpoint-dir> <images-dir> <out-dir> <cpu|mps>

Recorded with torch 2.15.0, transformers 5.17.0 and Pillow 13.0.0. `<images-dir>` comes from
`scripts/d3_metal_reference_images.py`; point `APXINF_D3_VISION_REF` at `<out-dir>/cases.json`.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import transformers
from PIL import Image


@dataclass(frozen=True)
class Case:
    """One reference request: an image, a state and a single question."""

    id: str
    image: str
    state: str
    question: dict[str, Any]


# Sizes cover the rounding and min_pixels branches of `smart_resize`; see make_images.py for
# which case covers which. Questions are neutral; the longest prompt is 733 tokens.
CASES = (
    Case(
        id="shapes-384-identity",
        image="shapes-384.png",
        state="An image attached to a support request.",
        question={
            "type": "choice",
            "instructions": "Which shape is drawn in the image?",
            "criteria": {
                "circle": "A round shape",
                "square": "A four-sided shape",
                "triangle": "A three-sided shape",
            },
        },
    ),
    Case(
        id="bars-400x320-round-down",
        image="bars-400x320.png",
        state="An image attached to a support request.",
        question={
            "type": "choice",
            "instructions": "How many coloured bars are in the image?",
            "criteria": {"two": None, "three": None, "four": None},
        },
    ),
    Case(
        id="grid-432x272-round-up",
        image="grid-432x272.png",
        state="An image attached to a support request.",
        question={
            "type": "noul",
            "instructions": "Does the image contain a red shape?",
        },
    ),
    Case(
        id="gradient-224-upscaled",
        image="gradient-224.png",
        state="An image attached to a support request.",
        question={
            "type": "choice",
            "instructions": "What does the image show?",
            "criteria": {
                "gradient": "A colour that changes across the image",
                "solid": "A single flat colour",
            },
        },
    ),
    Case(
        id="receipt-300x500-jpeg",
        image="receipt-300x500.jpg",
        state="An image attached to a support request.",
        question={
            "type": "choice",
            "instructions": "What kind of image is this?",
            "criteria": {
                "document": "A scanned or printed document",
                "photograph": "A photograph of a scene",
                "chart": "A chart or plot",
            },
        },
    ),
    Case(
        id="receipt-crop-352x288",
        image="receipt-crop-352x288.png",
        state="An image attached to a support request.",
        question={
            "type": "noul",
            "instructions": "Does the image show printed text?",
        },
    ),
    Case(
        id="receipt-520x400",
        image="receipt-520x400.png",
        state="An image attached to a support request.",
        question={
            "type": "choice",
            "instructions": "What kind of image is this?",
            "criteria": {
                "document": "A scanned or printed document",
                "photograph": "A photograph of a scene",
                "chart": "A chart or plot",
            },
        },
    ),
    Case(
        id="plot-800x800-large",
        image="plot-800x800.png",
        state="An image attached to a support request.",
        question={
            "type": "choice",
            "instructions": "What kind of image is this?",
            "criteria": {
                "chart": "A chart or plot",
                "photograph": "A photograph of a scene",
            },
        },
    ),
)


class Recorder:
    """A callable that delegates and remembers what it last returned.

    `D3.image_probabilities` owns the whole forward pass: it encodes the batch itself and runs
    the vision tower inside `D3.image_embeddings`. Reaching in afterwards for `pixel_values` or
    the tower output would mean a second encode and a second tower forward, which is a reference
    of something adjacent to the measured run rather than of the run. Recording both calls as
    they happen keeps every written array on the same forward pass as the probabilities.
    """

    def __init__(self, inner: Any):
        self.inner = inner
        self.last: Any = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.last = self.inner(*args, **kwargs)
        return self.last

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def write_f32(path: Path, tensor: torch.Tensor) -> int:
    values = tensor.detach().to("cpu", torch.float32).contiguous().numpy()
    values.tofile(path)
    return values.size


def run_case(runtime: Any, images_dir: Path, out: Path, case: Case) -> dict[str, Any]:
    image_path = images_dir / case.image
    prepared = runtime.prepare(case.state, {case.id: case.question}, images=[str(image_path)])
    if prepared.errors:
        raise RuntimeError(f"{case.id}: the runtime rejected the request: {prepared.errors}")
    question = prepared.questions[case.id]
    text = prepared.texts[case.id]
    width = prepared.lengths[case.id]
    option_count = len(question.keys)

    processor = Recorder(runtime.processor)
    tower = Recorder(runtime.backbone.get_image_features)
    runtime.processor = processor
    runtime.backbone.get_image_features = tower
    try:
        probabilities = runtime.image_probabilities(
            [text], prepared.images, [option_count], width
        )[0]
    finally:
        runtime.processor = processor.inner
        del runtime.backbone.get_image_features
    encoded = processor.last
    if encoded is None:
        raise RuntimeError(f"{case.id}: the forward pass did not call the processor")
    if tower.last is None:
        raise RuntimeError(f"{case.id}: the forward pass did not run the vision tower")

    input_ids = encoded["input_ids"]
    if tuple(input_ids.shape) != (1, width):
        raise RuntimeError(
            f"{case.id}: planned {width} tokens, the processor produced {tuple(input_ids.shape)}"
        )
    if not bool(encoded["attention_mask"].all()):
        raise RuntimeError(f"{case.id}: a single-prompt batch must not be padded")
    grid = encoded["image_grid_thw"]
    if tuple(grid.shape) != (1, 3):
        raise RuntimeError(f"{case.id}: expected one image grid, got {tuple(grid.shape)}")
    grid_thw = [int(v) for v in grid[0].tolist()]
    patches = grid_thw[1] * grid_thw[2]
    pixel_values = encoded["pixel_values"]
    if tuple(pixel_values.shape) != (patches, 1536):
        raise RuntimeError(
            f"{case.id}: pixel_values {tuple(pixel_values.shape)} for {patches} patches"
        )

    # `get_image_features` splits its output per image, so one image is a one-element tuple.
    features = torch.cat(list(tower.last.pooler_output), dim=0)
    if tuple(features.shape) != (patches // 4, 1024):
        raise RuntimeError(
            f"{case.id}: image_features {tuple(features.shape)} for {patches} patches"
        )

    pixels_file = f"{case.id}.pixels.f32"
    features_file = f"{case.id}.features.f32"
    write_f32(out / pixels_file, pixel_values)
    write_f32(out / features_file, features)

    with Image.open(image_path) as opened:
        source_size = [opened.size[0], opened.size[1]]
        source_format = opened.format

    return {
        "id": case.id,
        "grid_thw": grid_thw,
        "token_ids": [int(v) for v in input_ids[0].tolist()],
        "option_count": option_count,
        "probabilities": [float(v) for v in probabilities],
        "arrays": {
            "pixel_values": {"file": pixels_file},
            "image_features": {"file": features_file},
        },
        "image": {
            "file": case.image,
            "size": source_size,
            "format": source_format,
            "resized_hw": [grid_thw[1] * 16, grid_thw[2] * 16],
        },
        "image_tokens": patches // 4,
        "token_count": width,
        "question_type": case.question["type"],
        "option_keys": question.keys,
        "instructions": case.question["instructions"],
        "state": case.state,
    }


def main() -> int:
    checkpoint, images_dir, out, device = (
        Path(sys.argv[1]),
        Path(sys.argv[2]),
        Path(sys.argv[3]),
        sys.argv[4],
    )
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModel

    model = AutoModel.from_pretrained(str(checkpoint), trust_remote_code=True, device=device)
    runtime = model.runtime
    if runtime.image_unavailable is not None:
        raise RuntimeError(f"image inputs unavailable: {runtime.image_unavailable}")
    # Both would silently change what the reference measures: option-order averaging has no
    # equivalent in the port, and the fused ROCm path skips `image_embeddings` entirely, so the
    # vision tower the recorder watches would never run.
    if runtime.permutation_average:
        raise RuntimeError("the port scores one option order; permutation averaging must be off")
    if runtime.fast is not None and runtime.fast.fused is not None:
        raise RuntimeError("the fused kernel path bypasses the vision tower this script records")

    cases = [run_case(runtime, images_dir, out, case) for case in CASES]
    for case in cases:
        print(
            f"{case['id']:<26} grid={case['grid_thw']} tokens={case['token_count']:<4}"
            f" options={case['option_count']} p={[round(v, 4) for v in case['probabilities']]}"
        )

    summary = {
        "device": str(runtime.device),
        "dtype": "bfloat16 backbone, float32 readout",
        "checkpoint": str(checkpoint),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cases": cases,
    }
    (out / "cases.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out / 'cases.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
