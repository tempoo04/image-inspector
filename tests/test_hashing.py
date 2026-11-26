"""Tests for core.hashing: sha256, pHash, Hamming, grouping, batch run."""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np

from core.hashing import (
    compute_hashes,
    compute_hashes_for_dataset,
    group_near,
    hamming,
    phash,
    sha256_file,
)
from core.scanner import scan_dataset
from database.repository import Repository


def _write(path: Path, bgr: np.ndarray) -> Path:
    cv2.imwrite(str(path), bgr)
    return path


def test_sha256_matches_and_differs(tmp_path: Path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    c = tmp_path / "c.bin"
    a.write_bytes(b"hello world")
    b.write_bytes(b"hello world")
    c.write_bytes(b"different")
    assert sha256_file(str(a)) == sha256_file(str(b))
    assert sha256_file(str(a)) != sha256_file(str(c))


def test_sha256_unreadable_returns_none(tmp_path: Path):
    assert sha256_file(str(tmp_path / "missing.bin")) is None


def test_hamming_known_values():
    assert hamming("0000000000000000", "0000000000000000") == 0
    assert hamming("0000000000000000", "0000000000000001") == 1
    assert hamming("0000000000000000", "000000000000000f") == 4


def test_phash_identical_image_distance_zero(tmp_path: Path):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (80, 80, 3), dtype=np.uint8)
    p1 = phash(str(_write(tmp_path / "a.png", img)))
    p2 = phash(str(_write(tmp_path / "b.png", img.copy())))
    assert p1 is not None and p1 == p2
    assert hamming(p1, p2) == 0


def test_phash_similar_closer_than_different(tmp_path: Path):
    rng = np.random.default_rng(1)
    base = rng.integers(0, 256, (80, 80, 3), dtype=np.uint8)
    # Slight brightness tweak -> near duplicate.
    similar = cv2.convertScaleAbs(base, alpha=1.0, beta=10)
    different = rng.integers(0, 256, (80, 80, 3), dtype=np.uint8)
    pb = phash(str(_write(tmp_path / "base.png", base)))
    ps = phash(str(_write(tmp_path / "similar.png", similar)))
    pd = phash(str(_write(tmp_path / "different.png", different)))
    assert hamming(pb, ps) < hamming(pb, pd)


def test_phash_unreadable_returns_none(tmp_path: Path):
    bad = tmp_path / "broken.png"
    bad.write_bytes(b"not an image")
    assert phash(str(bad)) is None


def test_compute_hashes_sets_sha_without_phash_on_nonimage(tmp_path: Path):
    f = tmp_path / "data.txt"
    f.write_bytes(b"plain text, readable but not an image")
    h = compute_hashes(str(f))
    assert h is not None
    assert h.sha256 and h.phash is None


def test_group_near_clusters_duplicates(tmp_path: Path):
    rng = np.random.default_rng(2)
    base = rng.integers(0, 256, (80, 80, 3), dtype=np.uint8)
    other = rng.integers(0, 256, (80, 80, 3), dtype=np.uint8)
    rows = [
        {"id": 1, "phash": phash(str(_write(tmp_path / "x1.png", base)))},
        {"id": 2, "phash": phash(str(_write(tmp_path / "x2.png", base.copy())))},
        {"id": 3, "phash": phash(str(_write(tmp_path / "y.png", other)))},
        {"id": 4, "phash": None},  # ignored
    ]
    groups = group_near(rows, max_distance=2)
    assert len(groups) == 1
    assert {r["id"] for r in groups[0]} == {1, 2}


def test_exact_duplicate_groups(class_dataset: Path, tmp_path: Path):
    # Duplicate one real image by copying its bytes to a new file in the tree.
    first = next(class_dataset.rglob("*.png"))
    shutil.copyfile(first, first.with_name("copy_of_" + first.name))
    with Repository(tmp_path / "c.db") as repo:
        scan_dataset(class_dataset, repo)
        compute_hashes_for_dataset(repo)
        groups = repo.exact_duplicate_groups()
        assert len(groups) == 1
        assert len(groups[0]) == 2
        shas = {repo.get_hashes(r["id"])["sha256"] for r in groups[0]}
        assert len(shas) == 1  # identical bytes


def test_batch_compute_and_idempotent(class_dataset: Path, tmp_path: Path):
    seen: list[tuple[int, int]] = []
    with Repository(tmp_path / "c.db") as repo:
        scan_dataset(class_dataset, repo)
        non_corrupt = repo.count_images(include_corrupt=False)
        assert repo.count_without_hashes() == non_corrupt

        computed = compute_hashes_for_dataset(repo, progress_cb=lambda d, t: seen.append((d, t)))
        assert computed == non_corrupt
        assert repo.count_with_hashes() == non_corrupt
        assert repo.count_without_hashes() == 0
        assert seen and seen[-1][0] == seen[-1][1]
        assert len(repo.hashed_images()) == non_corrupt

        # Re-running finds nothing pending.
        assert compute_hashes_for_dataset(repo) == 0
