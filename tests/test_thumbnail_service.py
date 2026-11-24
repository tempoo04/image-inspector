"""Tests for the thumbnail service: generation, caching, invalidation, errors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from core.config import ThumbnailSettings
from services.thumbnail_service import ThumbnailService


def _make_image(path: Path, size=(300, 200)) -> None:
    arr = np.random.default_rng(0).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def _service(tmp_path: Path, max_edge=128) -> ThumbnailService:
    return ThumbnailService(ThumbnailSettings(max_edge=max_edge, cache_dir=tmp_path / "cache"))


def test_generates_and_caches(tmp_path: Path):
    src = tmp_path / "img.png"
    _make_image(src)
    svc = _service(tmp_path)
    mtime = src.stat().st_mtime

    thumb = svc.get_thumbnail(src, mtime)
    assert thumb is not None and thumb.exists()

    with Image.open(thumb) as im:
        assert max(im.size) <= 128  # scaled down to max_edge

    # Second call returns the same cached file without regenerating.
    first_mtime = thumb.stat().st_mtime
    again = svc.get_thumbnail(src, mtime)
    assert again == thumb
    assert thumb.stat().st_mtime == first_mtime


def test_mtime_change_invalidates(tmp_path: Path):
    src = tmp_path / "img.png"
    _make_image(src)
    svc = _service(tmp_path)

    t1 = svc.get_thumbnail(src, 100.0)
    t2 = svc.get_thumbnail(src, 200.0)  # different mtime -> different cache key
    assert t1 != t2
    assert t1.exists() and t2.exists()


def test_corrupt_source_returns_none(tmp_path: Path):
    bad = tmp_path / "broken.jpg"
    bad.write_bytes(b"\xff\xd8\xff not a jpeg")
    svc = _service(tmp_path)
    assert svc.get_thumbnail(bad, 1.0) is None


def test_missing_source_returns_none(tmp_path: Path):
    svc = _service(tmp_path)
    assert svc.get_thumbnail(tmp_path / "ghost.png", 1.0) is None


def test_clear_cache(tmp_path: Path):
    src = tmp_path / "img.png"
    _make_image(src)
    svc = _service(tmp_path)
    svc.get_thumbnail(src, 1.0)
    assert any(svc.cache_dir.glob("*.jpg"))
    removed = svc.clear_cache()
    assert removed >= 1
    assert not any(svc.cache_dir.glob("*.jpg"))
