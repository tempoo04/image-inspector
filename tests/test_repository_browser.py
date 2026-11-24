"""Tests for browser queries: filtering, sorting, pagination, aggregates."""

from __future__ import annotations

from pathlib import Path

from core.metadata import ImageMetadata
from database.repository import Repository


def _meta(rel, *, ext=".png", size=100, w=10, h=10, cls=None, corrupt=False):
    aspect = round(w / h, 6) if w and h else None
    return ImageMetadata(
        abs_path=f"/abs/{rel}",
        rel_path=rel,
        filename=rel.split("/")[-1],
        ext=ext,
        size_bytes=size,
        mtime=1.0,
        width=w,
        height=h,
        mode="RGB",
        aspect_ratio=aspect,
        class_label=cls,
        corrupt=corrupt,
    )


def _seed(repo: Repository):
    repo.upsert_images(
        [
            _meta("cat/a.png", ext=".png", size=100, w=64, h=64, cls="cat"),
            _meta("cat/b.jpg", ext=".jpg", size=300, w=128, h=64, cls="cat"),
            _meta("dog/c.png", ext=".png", size=200, w=200, h=200, cls="dog"),
            _meta("dog/d.jpg", ext=".jpg", size=500, w=32, h=64, cls="dog"),
            _meta("dog/bad.jpg", ext=".jpg", size=10, w=None, h=None, cls="dog", corrupt=True),
        ]
    )


def test_filter_by_class(tmp_path: Path):
    with Repository(tmp_path / "t.db") as repo:
        _seed(repo)
        rows, total = repo.list_images({"class_label": "cat"})
        assert total == 2
        assert {r["filename"] for r in rows} == {"a.png", "b.jpg"}


def test_filter_by_ext(tmp_path: Path):
    with Repository(tmp_path / "t.db") as repo:
        _seed(repo)
        _, total = repo.list_images({"exts": [".png"]})
        assert total == 2


def test_filter_filename_substring(tmp_path: Path):
    with Repository(tmp_path / "t.db") as repo:
        _seed(repo)
        rows, total = repo.list_images({"filename": "a."})
        assert total == 1
        assert rows[0]["filename"] == "a.png"


def test_exclude_corrupt_by_default_toggle(tmp_path: Path):
    with Repository(tmp_path / "t.db") as repo:
        _seed(repo)
        _, with_corrupt = repo.list_images({"include_corrupt": True})
        _, without = repo.list_images({"include_corrupt": False})
        assert with_corrupt == 5
        assert without == 4


def test_filter_min_width(tmp_path: Path):
    with Repository(tmp_path / "t.db") as repo:
        _seed(repo)
        _, total = repo.list_images({"min_width": 100, "include_corrupt": True})
        assert total == 2  # 128 and 200 wide


def test_sort_by_size_desc(tmp_path: Path):
    with Repository(tmp_path / "t.db") as repo:
        _seed(repo)
        rows, _ = repo.list_images({"include_corrupt": True}, sort="size", descending=True)
        sizes = [r["size_bytes"] for r in rows]
        assert sizes == sorted(sizes, reverse=True)
        assert sizes[0] == 500


def test_pagination(tmp_path: Path):
    with Repository(tmp_path / "t.db") as repo:
        _seed(repo)
        page1, total = repo.list_images({"include_corrupt": True}, limit=2, offset=0)
        page2, _ = repo.list_images({"include_corrupt": True}, limit=2, offset=2)
        assert total == 5
        assert len(page1) == 2 and len(page2) == 2
        assert {r["id"] for r in page1}.isdisjoint({r["id"] for r in page2})


def test_unknown_sort_falls_back_to_name(tmp_path: Path):
    with Repository(tmp_path / "t.db") as repo:
        _seed(repo)
        rows, _ = repo.list_images({"include_corrupt": True}, sort="bogus")
        paths = [r["rel_path"] for r in rows]
        assert paths == sorted(paths)


def test_value_ranges_and_aggregates(tmp_path: Path):
    with Repository(tmp_path / "t.db") as repo:
        _seed(repo)
        ranges = repo.value_ranges()  # corrupt excluded
        assert ranges["width"] == (32, 200)
        assert ranges["size"][0] == 100
        assert dict(repo.count_by_ext()) == {".png": 2, ".jpg": 3}
        assert set(repo.distinct_exts()) == {".png", ".jpg"}


def test_value_ranges_empty_catalog(tmp_path: Path):
    with Repository(tmp_path / "t.db") as repo:
        assert repo.value_ranges() == {}
