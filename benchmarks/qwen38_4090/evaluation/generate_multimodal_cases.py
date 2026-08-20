#!/usr/bin/env python3
"""Generate the deterministic public image-input capability suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import struct
import zlib
from pathlib import Path
from typing import Any


PUBLIC_SEED = 380038
WIDTH = 448
HEIGHT = 448
SCHEMA = "apxinf.qwen38_27b.multimodal_cases.v1"
PALETTE = {
    "WHITE": (250, 248, 242),
    "INK": (24, 28, 36),
    "GRID": (196, 202, 210),
    "RED": (220, 54, 58),
    "GREEN": (42, 166, 92),
    "BLUE": (55, 105, 220),
    "YELLOW": (242, 190, 38),
}
SEGMENTS = {
    "0": "abcedf",
    "1": "bc",
    "2": "abdeg",
    "3": "abcdg",
    "4": "bcfg",
    "5": "acdfg",
    "6": "acdefg",
    "7": "abc",
    "8": "abcdefg",
    "9": "abcdfg",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Canvas:
    def __init__(self, width: int = WIDTH, height: int = HEIGHT) -> None:
        self.width = width
        self.height = height
        self.pixels = bytearray(PALETTE["WHITE"] * (width * height))

    def _set(self, x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            offset = (y * self.width + x) * 3
            self.pixels[offset : offset + 3] = bytes(color)

    def rect(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        color: tuple[int, int, int],
    ) -> None:
        x0, x1 = sorted((max(0, x0), min(self.width, x1)))
        y0, y1 = sorted((max(0, y0), min(self.height, y1)))
        row = bytes(color) * max(0, x1 - x0)
        for y in range(y0, y1):
            start = (y * self.width + x0) * 3
            self.pixels[start : start + len(row)] = row

    def circle(
        self, cx: int, cy: int, radius: int, color: tuple[int, int, int]
    ) -> None:
        r2 = radius * radius
        for y in range(cy - radius, cy + radius + 1):
            dy2 = (y - cy) ** 2
            span = int(max(0, r2 - dy2) ** 0.5)
            self.rect(cx - span, y, cx + span + 1, y + 1, color)

    def outline(self, x0: int, y0: int, x1: int, y1: int, width: int = 3) -> None:
        color = PALETTE["INK"]
        self.rect(x0, y0, x1, y0 + width, color)
        self.rect(x0, y1 - width, x1, y1, color)
        self.rect(x0, y0, x0 + width, y1, color)
        self.rect(x1 - width, y0, x1, y1, color)

    def write_png(self, path: Path) -> None:
        raw = bytearray()
        stride = self.width * 3
        for y in range(self.height):
            raw.append(0)
            start = y * stride
            raw.extend(self.pixels[start : start + stride])

        def chunk(kind: bytes, payload: bytes) -> bytes:
            return (
                struct.pack(">I", len(payload))
                + kind
                + payload
                + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
            )

        png = b"\x89PNG\r\n\x1a\n"
        png += chunk("IHDR".encode(), struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0))
        png += chunk("IDAT".encode(), zlib.compress(bytes(raw), 9))
        png += chunk("IEND".encode(), b"")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(png)


def draw_digit(canvas: Canvas, digit: str, x: int, y: int) -> None:
    width, height, thick = 64, 128, 12
    half = height // 2
    color = PALETTE["INK"]
    segments = {
        "a": (x + thick, y, x + width - thick, y + thick),
        "b": (x + width - thick, y + thick, x + width, y + half - 3),
        "c": (x + width - thick, y + half + 3, x + width, y + height - thick),
        "d": (x + thick, y + height - thick, x + width - thick, y + height),
        "e": (x, y + half + 3, x + thick, y + height - thick),
        "f": (x, y + thick, x + thick, y + half - 3),
        "g": (x + thick, y + half - thick // 2, x + width - thick, y + half + thick // 2),
    }
    for name in SEGMENTS[digit]:
        canvas.rect(*segments[name], color)


def build_seven_segment(rng: random.Random, case_id: str) -> tuple[Canvas, dict[str, Any]]:
    code = "".join(str(rng.randrange(10)) for _ in range(4))
    canvas = Canvas()
    canvas.rect(38, 104, 410, 338, (232, 236, 240))
    canvas.outline(38, 104, 410, 338, 5)
    for index, digit in enumerate(code):
        draw_digit(canvas, digit, 70 + index * 82, 158)
    return canvas, {
        "id": case_id,
        "category": "seven_segment_ocr",
        "prompt": "图中显示一个四位数的七段数码管代码。只输出这四位数字，不要解释。",
        "expected": code,
    }


def build_spatial_color(rng: random.Random, case_id: str) -> tuple[Canvas, dict[str, Any]]:
    colors = ["RED", "GREEN", "BLUE", "YELLOW"]
    cells = [rng.choice(colors) for _ in range(9)]
    target = rng.randrange(9)
    row, column = divmod(target, 3)
    canvas = Canvas()
    x0, y0, step = 74, 74, 100
    for i in range(4):
        canvas.rect(x0 + i * step - 2, y0, x0 + i * step + 2, y0 + 300, PALETTE["GRID"])
        canvas.rect(x0, y0 + i * step - 2, x0 + 300, y0 + i * step + 2, PALETTE["GRID"])
    for index, name in enumerate(cells):
        r, c = divmod(index, 3)
        canvas.circle(x0 + c * step + 50, y0 + r * step + 50, 31, PALETTE[name])
    return canvas, {
        "id": case_id,
        "category": "spatial_color",
        "prompt": (
            f"把图中的 3×3 圆点矩阵从上到下编号为第1至第3行，从左到右编号为第1至第3列。"
            f"第{row + 1}行第{column + 1}列是什么颜色？只输出 RED、GREEN、BLUE 或 YELLOW。"
        ),
        "expected": cells[target],
    }


def build_bar_arithmetic(rng: random.Random, case_id: str) -> tuple[Canvas, dict[str, Any]]:
    red_height, blue_height = rng.sample(range(2, 9), 2)
    canvas = Canvas()
    left, bottom, unit = 72, 382, 36
    for level in range(9):
        y = bottom - level * unit
        canvas.rect(left, y - 1, 376, y + 1, PALETTE["GRID"])
    canvas.rect(128, bottom - red_height * unit, 218, bottom, PALETTE["RED"])
    canvas.rect(266, bottom - blue_height * unit, 356, bottom, PALETTE["BLUE"])
    canvas.outline(128, bottom - red_height * unit, 218, bottom, 3)
    canvas.outline(266, bottom - blue_height * unit, 356, bottom, 3)
    return canvas, {
        "id": case_id,
        "category": "bar_arithmetic",
        "prompt": "图中每相邻两条水平网格线表示 1 个单位。红色柱和蓝色柱的高度相差多少个单位？只输出整数。",
        "expected": str(abs(red_height - blue_height)),
    }


def build_object_count(rng: random.Random, case_id: str) -> tuple[Canvas, dict[str, Any]]:
    green_count = rng.randrange(4, 10)
    distractor_count = rng.randrange(3, 8)
    positions = [(64 + c * 64, 82 + r * 72) for r in range(5) for c in range(6)]
    rng.shuffle(positions)
    canvas = Canvas()
    for index, (x, y) in enumerate(positions[: green_count + distractor_count]):
        name = "GREEN" if index < green_count else rng.choice(["RED", "BLUE", "YELLOW"])
        canvas.circle(x, y, 20, PALETTE[name])
    return canvas, {
        "id": case_id,
        "category": "object_count",
        "prompt": "图中有多少个绿色实心圆？只输出整数。",
        "expected": str(green_count),
    }


BUILDERS = (build_seven_segment, build_spatial_color, build_bar_arithmetic, build_object_count)


def generate_cases(seed: int, prefix: str, cases_per_category: int) -> list[tuple[Canvas, dict[str, Any]]]:
    rng = random.Random(seed)
    generated: list[tuple[Canvas, dict[str, Any]]] = []
    for builder in BUILDERS:
        for repeat in range(cases_per_category):
            case_id = f"{prefix}-{builder.__name__.removeprefix('build_').replace('_', '-')}-{repeat + 1:02d}"
            generated.append(builder(rng, case_id))
    return generated


def write_suite(
    generated: list[tuple[Canvas, dict[str, Any]]],
    output_dir: Path,
    contract_path: Path,
    split: str,
    seed_sha256: str | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    rows: list[dict[str, Any]] = []
    for canvas, case in generated:
        image_path = image_dir / f"{case['id']}.png"
        canvas.write_png(image_path)
        rows.append(
            {
                **case,
                "image": f"images/{image_path.name}",
                "image_sha256": sha256_file(image_path),
                "validator": "normalized_exact",
                "max_completion_tokens": 32,
            }
        )
    cases_path = output_dir / "cases.jsonl"
    cases_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = {
        "schema": SCHEMA,
        "split": split,
        "case_count": len(rows),
        "categories": {name: sum(row["category"] == name for row in rows) for name in sorted({row["category"] for row in rows})},
        "contract_sha256": sha256_file(contract_path),
        "cases_jsonl_sha256": sha256_file(cases_path),
        "images": {row["id"]: row["image_sha256"] for row in rows},
    }
    if seed_sha256 is not None:
        manifest["seed_sha256"] = seed_sha256
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(__file__).with_name("multimodal-contract-v1.json"),
    )
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    seed = contract["image_generation"]["public_seed"]
    if seed != PUBLIC_SEED:
        raise ValueError("public seed drift between contract and generator")
    generated = generate_cases(seed, "public-mm", 1)
    manifest = write_suite(generated, args.output_dir, args.contract, "public")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
