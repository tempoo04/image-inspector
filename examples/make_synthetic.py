"""Generate a small synthetic image dataset for testing and demos.

Creates class-based folders by default, with a mix of sizes/formats and one
deliberately corrupt file so scanner error handling can be exercised.

Usage:
    python -m examples.make_synthetic --out data/sample --per-class 20
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image

CLASSES = ("cat", "dog", "bird")
FORMATS = ("JPEG", "PNG", "BMP")
_EXT = {"JPEG": ".jpg", "PNG": ".png", "BMP": ".bmp"}


def _random_image(w: int, h: int, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def generate(out: Path, per_class: int = 20, flat: bool = False, seed: int = 0) -> dict:
    """Create the dataset under *out*. Returns a small summary dict."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    made = 0

    targets = [out] if flat else [out / c for c in CLASSES]
    for t in targets:
        t.mkdir(parents=True, exist_ok=True)

    classes = ["flat"] if flat else list(CLASSES)
    for ci, cls in enumerate(classes):
        folder = out if flat else out / cls
        for i in range(per_class):
            fmt = rng.choice(FORMATS)
            w, h = rng.choice([(64, 64), (128, 96), (200, 200), (30, 30)])
            img = _random_image(w, h, seed=seed + ci * 1000 + i)
            img.save(folder / f"{cls}_{i:03d}{_EXT[fmt]}", format=fmt)
            made += 1

    # One corrupt file: valid extension, invalid bytes.
    corrupt = (out if flat else out / CLASSES[0]) / "broken.jpg"
    corrupt.write_bytes(b"\xff\xd8\xff\xe0 not a real jpeg \x00\x01\x02")

    return {"out": str(out), "images": made, "corrupt": 1, "flat": flat}


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a synthetic image dataset.")
    ap.add_argument("--out", default="data/sample", type=Path)
    ap.add_argument("--per-class", default=20, type=int)
    ap.add_argument("--flat", action="store_true", help="flat directory instead of class folders")
    ap.add_argument("--seed", default=0, type=int)
    args = ap.parse_args()
    summary = generate(args.out, args.per_class, args.flat, args.seed)
    print(summary)


if __name__ == "__main__":
    main()
