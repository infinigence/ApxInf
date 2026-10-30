"""Materialize the tracked diagnostic GPU corpus without NumPy or random draws."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()
manifest = json.loads(Path(__file__).with_name("manifest.json").read_text())
rgb = bytes(
    channel % 256
    for view in range(2)
    for y in range(224)
    for x in range(224)
    for channel in (x + 41 * view, 3 * y + 29 * view, x + 2 * y)
)
for case in manifest["cases"]:
    if hashlib.sha256(rgb).hexdigest() != case["images_sha256"]:
        raise ValueError("tracked RGB digest does not match generator")
args.output.mkdir(parents=True, exist_ok=False)
(args.output / "gradient.rgb").write_bytes(rgb)
(args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(args.output / "manifest.json")
