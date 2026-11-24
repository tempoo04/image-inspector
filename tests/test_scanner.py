"""Tests for the scanner: layout detection, nested + flat scans, incremental rescan."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.scanner import detect_layout, scan_dataset
from database.repository import Repository


def test_detect_layout_class_based(class_dataset: Path):
    assert detect_layout(class_dataset) == "class-based"


def test_detect_layout_flat(flat_dataset: Path):
    assert detect_layout(flat_dataset) == "flat"


def test_detect_layout_empty(tmp_path: Path):
    (tmp_path / "empty").mkdir()
    assert detect_layout(tmp_path / "empty") == "empty"


def test_scan_class_dataset_indexes_all(class_dataset: Path, tmp_path: Path):
    with Repository(tmp_path / "c.db") as repo:
        res = scan_dataset(class_dataset, repo)
        # 3 classes * 5 images + 1 corrupt = 16 files.
        assert res.written == 16
        assert res.corrupt == 1
        assert repo.count_images() == 16
        assert repo.count_corrupt() == 1
        classes = dict(repo.list_classes())
        assert classes.get("cat") == 6  # 5 valid + broken.jpg lives in cat/


def test_scan_flat_dataset(flat_dataset: Path, tmp_path: Path):
    with Repository(tmp_path / "f.db") as repo:
        res = scan_dataset(flat_dataset, repo)
        assert res.layout == "flat"
        assert res.written == 11  # 10 + 1 corrupt
        # Flat images have no class label.
        assert dict(repo.list_classes()).get(None) == 11


def test_incremental_rescan_skips_unchanged(class_dataset: Path, tmp_path: Path):
    with Repository(tmp_path / "c.db") as repo:
        first = scan_dataset(class_dataset, repo)
        assert first.written == 16

        second = scan_dataset(class_dataset, repo)
        assert second.skipped_unchanged == 16
        assert second.written == 0
        assert repo.count_images() == 16


def test_incremental_rescan_picks_up_changes(class_dataset: Path, tmp_path: Path):
    with Repository(tmp_path / "c.db") as repo:
        scan_dataset(class_dataset, repo)

        # Add one new image under a new class folder.
        import numpy as np
        from PIL import Image

        (class_dataset / "fish").mkdir()
        Image.fromarray(np.zeros((20, 20, 3), dtype=np.uint8)).save(
            class_dataset / "fish" / "newpic.png"
        )

        res = scan_dataset(class_dataset, repo)
        assert res.written == 1
        assert res.skipped_unchanged == 16
        assert repo.count_images() == 17
        assert dict(repo.list_classes()).get("fish") == 1


def test_prune_missing_removes_deleted(flat_dataset: Path, tmp_path: Path):
    with Repository(tmp_path / "f.db") as repo:
        scan_dataset(flat_dataset, repo)
        before = repo.count_images()
        # Delete one file, rescan with pruning.
        victim = next(flat_dataset.glob("*.png"))
        victim.unlink()
        res = scan_dataset(flat_dataset, repo, prune_missing=True)
        assert res.deleted == 1
        assert repo.count_images() == before - 1


def test_progress_callback_invoked(class_dataset: Path, tmp_path: Path):
    seen: list[tuple[int, int]] = []
    with Repository(tmp_path / "c.db") as repo:
        scan_dataset(class_dataset, repo, progress_cb=lambda d, t: seen.append((d, t)))
    assert seen
    assert seen[-1][0] == 16  # all files reported done
    assert seen[-1][1] == 16  # total


def test_scan_rejects_non_directory(tmp_path: Path):
    with Repository(tmp_path / "c.db") as repo:
        with pytest.raises(NotADirectoryError):
            scan_dataset(tmp_path / "nope", repo)
