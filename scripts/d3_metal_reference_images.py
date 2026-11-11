"""Build the images that `scripts/d3_metal_reference_dump.py` records references for.

Five are drawn from scratch with PIL, three are crops or rescales of the checkpoint's own
`assets/example-receipt.png`. Sizes exercise the rounding and min_pixels branches of
`smart_resize` (factor 32, min_pixels 65536, max_pixels 1638400). The max_pixels downscale
branch is not covered: it only triggers above 1.6 MP, which costs 1600 image tokens in every
case and so cannot fit the 1000-token budget these prompts are held to.

    384x384 -> 384x384   identity, already a multiple of 32
    400x320 -> 384x320   12.5 rounds down to 12 (ties to even)
    432x272 -> 448x256   13.5 rounds up to 14, 8.5 rounds down to 8
    224x224 -> 256x256   under min_pixels, upscaled
    300x500 -> 288x512   both edges move, JPEG source
    352x288 -> 352x288   identity
    520x400 -> 512x384   both edges round down
    800x800 -> 800x800   identity, the large case

Run:
    python3 scripts/d3_metal_reference_images.py <checkpoint-dir> <out-dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw


def shapes_384(path: Path) -> None:
    image = Image.new("RGB", (384, 384), (246, 246, 242))
    draw = ImageDraw.Draw(image)
    draw.ellipse((96, 96, 288, 288), fill=(40, 90, 200), outline=(20, 40, 110), width=6)
    image.save(path)


def bars_400x320(path: Path) -> None:
    image = Image.new("RGB", (400, 320), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    for index, colour in enumerate([(200, 50, 50), (60, 160, 70), (50, 90, 200)]):
        left = 40 + index * 120
        draw.rectangle((left, 60, left + 80, 260), fill=colour)
    image.save(path)


def grid_432x272(path: Path) -> None:
    image = Image.new("RGB", (432, 272), (252, 252, 252))
    draw = ImageDraw.Draw(image)
    for x in range(0, 432, 24):
        draw.line((x, 0, x, 272), fill=(210, 210, 210), width=1)
    for y in range(0, 272, 24):
        draw.line((0, y, 432, y), fill=(210, 210, 210), width=1)
    draw.polygon([(216, 56), (324, 216), (108, 216)], fill=(205, 45, 45))
    image.save(path)


def gradient_224(path: Path) -> None:
    image = Image.new("RGB", (224, 224))
    pixels = image.load()
    for y in range(224):
        for x in range(224):
            share = x / 223.0
            pixels[x, y] = (
                int(30 + share * 215),
                int(70 + share * 180),
                int(160 + share * 95),
            )
    image.save(path)


def plot_800x800(path: Path) -> None:
    image = Image.new("RGB", (800, 800), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((100, 80, 740, 700), outline=(120, 120, 120), width=2)
    for step in range(1, 8):
        y = 80 + step * 77
        draw.line((100, y, 740, y), fill=(226, 226, 226), width=1)
    points = [(100 + index * 64, 660 - int((index**1.6) * 26)) for index in range(11)]
    draw.line(points, fill=(30, 80, 190), width=5)
    for x, y in points:
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=(30, 80, 190))
    draw.line((100, 700, 740, 700), fill=(40, 40, 40), width=3)
    draw.line((100, 80, 100, 700), fill=(40, 40, 40), width=3)
    image.save(path)


def main() -> int:
    checkpoint, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    shapes_384(out / "shapes-384.png")
    bars_400x320(out / "bars-400x320.png")
    grid_432x272(out / "grid-432x272.png")
    gradient_224(out / "gradient-224.png")
    plot_800x800(out / "plot-800x800.png")

    with Image.open(checkpoint / "assets" / "example-receipt.png") as source:
        receipt = source.convert("RGB")
    receipt.resize((300, 500), Image.LANCZOS).save(
        out / "receipt-300x500.jpg", quality=90
    )
    receipt.crop((120, 150, 472, 438)).save(out / "receipt-crop-352x288.png")
    receipt.resize((520, 400), Image.LANCZOS).save(out / "receipt-520x400.png")

    for path in sorted(out.iterdir()):
        with Image.open(path) as image:
            print(f"{path.name:<28} {image.size[0]}x{image.size[1]} {image.format}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
