"""Image hashing and duplicate grouping (Module F).

Pure computation, no UI dependency:

* **sha256** — exact byte-for-byte duplicate detection.
* **pHash** — a 64-bit perceptual hash (DCT of a 32x32 grayscale reduction) for
  near-duplicate detection; small Hamming distance means visually similar.

Exact grouping is a trivial ``GROUP BY sha256`` and lives in the repository.
Near-duplicate grouping clusters images whose pHashes are within a Hamming
threshold and lives here (:func:`group_near`). A batch helper
(:func:`compute_hashes_for_dataset`) drives computation over images lacking
hashes, via a repository.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

from .logging_config import get_logger

log = get_logger(__name__)

ProgressCb = Callable[[int, int], None]

_PHASH_IMG = 32  # downscale edge before DCT
_PHASH_LOW = 8  # low-frequency block kept (8x8 = 64 bits)
_READ_CHUNK = 1 << 20  # 1 MiB


@dataclass
class ImageHashes:
    """Exact and perceptual hashes for one image (maps to the hashes table)."""

    sha256: str
    phash: str | None  # 16 hex chars (64 bits), or None if undecodable as image


def sha256_file(abs_path: str) -> str | None:
    """Streaming SHA-256 of a file's bytes, or ``None`` if it cannot be read."""
    h = hashlib.sha256()
    try:
        with open(abs_path, "rb") as f:
            for chunk in iter(lambda: f.read(_READ_CHUNK), b""):
                h.update(chunk)
    except OSError as exc:
        log.warning("sha256: could not read %s: %s", abs_path, exc)
        return None
    return h.hexdigest()


def phash(abs_path: str) -> str | None:
    """Perceptual hash as 16 hex chars, or ``None`` if the image cannot be decoded.

    Reduces to a 32x32 grayscale image, takes the 2-D DCT, keeps the top-left
    8x8 low-frequency block, and thresholds each coefficient against the block's
    median (excluding the DC term, which otherwise biases the threshold). The
    resulting 64 bits are packed MSB-first into a hex string.
    """
    img = cv2.imread(abs_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    small = cv2.resize(img, (_PHASH_IMG, _PHASH_IMG), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(small))
    low = dct[:_PHASH_LOW, :_PHASH_LOW]
    med = float(np.median(low.flatten()[1:]))  # drop DC term [0,0]
    bits = (low > med).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def hamming(a: str, b: str) -> int:
    """Hamming distance between two 64-bit pHash hex strings (0 = identical)."""
    return int(bin(int(a, 16) ^ int(b, 16)).count("1"))


def compute_hashes(abs_path: str) -> ImageHashes | None:
    """Compute both hashes for *abs_path*.

    Returns ``None`` only if the bytes cannot be read at all. If the file reads
    but does not decode as an image, ``sha256`` is still set and ``phash`` is
    ``None`` (it can still participate in exact-duplicate detection).
    """
    digest = sha256_file(abs_path)
    if digest is None:
        return None
    return ImageHashes(sha256=digest, phash=phash(abs_path))


def group_near(
    rows: "list",
    max_distance: int,
    *,
    phash_key: str = "phash",
    id_key: str = "id",
) -> list[list]:
    """Cluster rows whose pHashes are within *max_distance* (Hamming), by union-find.

    *rows* are mapping-like objects carrying a pHash (``phash_key``) and an id
    (``id_key``); rows with a missing pHash are ignored. Returns groups of two or
    more rows, each group's rows preserving input order, largest groups first.

    Pairwise comparison is O(n^2); fine for local datasets up to a few thousand
    hashed images. Exact duplicates (distance 0) are included, so with
    ``max_distance=0`` this reproduces exact-pHash grouping.
    """
    items = [r for r in rows if r[phash_key]]
    n = len(items)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for i in range(n):
        hi = items[i][phash_key]
        for j in range(i + 1, n):
            if hamming(hi, items[j][phash_key]) <= max_distance:
                union(i, j)

    clusters: dict[int, list] = {}
    for idx, row in enumerate(items):
        clusters.setdefault(find(idx), []).append(row)
    groups = [g for g in clusters.values() if len(g) > 1]
    groups.sort(key=lambda g: (-len(g), g[0][id_key]))
    return groups


def compute_hashes_for_dataset(
    repo,
    *,
    batch_size: int = 200,
    progress_cb: ProgressCb | None = None,
) -> int:
    """Compute and store hashes for every indexed image lacking them.

    *repo* must provide ``images_without_hashes`` and ``upsert_hashes_many``.
    Returns the number of images for which hashes were computed and stored.
    Files whose bytes cannot be read are skipped (and remain without hashes).
    """
    pending = repo.images_without_hashes()
    total = len(pending)
    done = 0
    computed = 0
    buffer: list[tuple[int, ImageHashes]] = []

    for row in pending:
        h = compute_hashes(row["abs_path"])
        done += 1
        if h is not None:
            buffer.append((row["id"], h))
            computed += 1
        if len(buffer) >= batch_size:
            repo.upsert_hashes_many(buffer)
            buffer.clear()
        if progress_cb is not None:
            progress_cb(done, total)
    if buffer:
        repo.upsert_hashes_many(buffer)
    return computed
