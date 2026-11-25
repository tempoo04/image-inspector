"""Tests for core.image_metrics: metric computation, flags, and batch run."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from core.config import QualityThresholds
from core.image_metrics import compute_metrics, compute_metrics_for_dataset, rgb_histograms
from core.scanner import scan_dataset
from database.repository import Repository

T = QualityThresholds()


def _write(path: Path, bgr: np.ndarray) -> Path:
    cv2.imwrite(str(path), bgr)
    return path


def test_black_image_flags(tmp_path: Path):
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    m = compute_metrics(str(_write(tmp_path / "black.png", img)), T)
    assert m is not None
    assert m.brightness == 0
    assert m.is_black and m.is_dark
    assert not m.is_white and not m.is_bright
    assert m.is_blurry  # flat image -> zero Laplacian variance
    assert m.entropy == 0.0
    assert not m.is_small  # 64 >= 32


def test_white_image_flags(tmp_path: Path):
    img = np.full((64, 64, 3), 255, dtype=np.uint8)
    m = compute_metrics(str(_write(tmp_path / "white.png", img)), T)
    assert m is not None
    assert m.is_white and m.is_bright
    assert not m.is_black and not m.is_dark


def test_small_image_flag(tmp_path: Path):
    img = np.full((16, 16, 3), 128, dtype=np.uint8)
    m = compute_metrics(str(_write(tmp_path / "small.png", img)), T, width=16, height=16)
    assert m is not None and m.is_small


def test_sharp_vs_blurry(tmp_path: Path):
    noise = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    sharp = compute_metrics(str(_write(tmp_path / "noise.png", noise)), T)
    flat = compute_metrics(
        str(_write(tmp_path / "flat.png", np.full((64, 64, 3), 100, np.uint8))), T
    )
    assert sharp is not None and flat is not None
    assert sharp.sharpness > flat.sharpness
    assert not sharp.is_blurry
    assert flat.is_blurry


def test_channel_means(tmp_path: Path):
    # Pure red in BGR order: B=0, G=0, R=255.
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[:, :, 2] = 255
    m = compute_metrics(str(_write(tmp_path / "red.png", img)), T)
    assert m is not None
    assert m.mean_r == 255 and m.mean_g == 0 and m.mean_b == 0


def test_unreadable_returns_none(tmp_path: Path):
    bad = tmp_path / "broken.png"
    bad.write_bytes(b"not an image")
    assert compute_metrics(str(bad), T) is None


def test_rgb_histograms(tmp_path: Path):
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    hists = rgb_histograms(str(_write(tmp_path / "x.png", img)))
    assert hists is not None
    assert set(hists) == {"r", "g", "b"}
    assert hists["r"][0] == 100  # all 100 pixels in bin 0 (value 0)


def test_batch_compute_over_dataset(class_dataset: Path, tmp_path: Path):
    with Repository(tmp_path / "c.db") as repo:
        scan_dataset(class_dataset, repo)
        non_corrupt = repo.count_images(include_corrupt=False)
        assert repo.count_without_metrics() == non_corrupt

        computed = compute_metrics_for_dataset(repo, T)
        assert computed == non_corrupt
        assert repo.count_with_metrics() == non_corrupt
        assert repo.count_without_metrics() == 0

        counts = repo.metric_flag_counts()
        assert set(counts) == {
            "is_black",
            "is_white",
            "is_dark",
            "is_bright",
            "is_blurry",
            "is_small",
        }
        assert len(repo.metrics_joined()) == non_corrupt


def test_batch_progress_and_idempotent(class_dataset: Path, tmp_path: Path):
    seen: list[tuple[int, int]] = []
    with Repository(tmp_path / "c.db") as repo:
        scan_dataset(class_dataset, repo)
        compute_metrics_for_dataset(repo, T, progress_cb=lambda d, t: seen.append((d, t)))
        assert seen and seen[-1][0] == seen[-1][1]
        # Re-running finds nothing pending.
        again = compute_metrics_for_dataset(repo, T)
        assert again == 0
