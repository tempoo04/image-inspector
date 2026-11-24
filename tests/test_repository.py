"""Tests for the SQLite repository: upsert, incremental signatures, pruning."""

from __future__ import annotations

from pathlib import Path

from core.metadata import ImageMetadata
from database.repository import Repository


def _meta(rel: str, size: int = 100, mtime: float = 1.0, cls: str | None = None) -> ImageMetadata:
    return ImageMetadata(
        abs_path=f"/abs/{rel}",
        rel_path=rel,
        filename=rel.split("/")[-1],
        ext=".png",
        size_bytes=size,
        mtime=mtime,
        width=10,
        height=5,
        mode="RGB",
        aspect_ratio=2.0,
        class_label=cls,
    )


def test_upsert_is_idempotent(tmp_path: Path):
    db = tmp_path / "t.db"
    with Repository(db) as repo:
        n = repo.upsert_images([_meta("a.png"), _meta("b.png")])
        assert n == 2
        assert repo.count_images() == 2
        # Re-upsert same rel_paths -> still 2 rows, no duplicates.
        repo.upsert_images([_meta("a.png"), _meta("b.png")])
        assert repo.count_images() == 2


def test_signatures_roundtrip(tmp_path: Path):
    with Repository(tmp_path / "t.db") as repo:
        repo.upsert_images([_meta("a.png", size=123, mtime=9.5)])
        sigs = repo.load_signatures()
        assert sigs["a.png"] == (123, 9.5)


def test_update_changes_fields_not_rowcount(tmp_path: Path):
    with Repository(tmp_path / "t.db") as repo:
        repo.upsert_images([_meta("a.png", size=100)])
        img_id = repo.iter_images()[0]["id"]
        repo.upsert_images([_meta("a.png", size=999)])
        row = repo.get_image(img_id)
        assert row is not None
        assert row["size_bytes"] == 999  # updated in place, id preserved
        assert repo.count_images() == 1


def test_delete_by_rel_paths(tmp_path: Path):
    with Repository(tmp_path / "t.db") as repo:
        repo.upsert_images([_meta("a.png"), _meta("b.png"), _meta("c.png")])
        deleted = repo.delete_by_rel_paths(["a.png", "c.png"])
        assert deleted == 2
        assert repo.count_images() == 1


def test_class_and_size_aggregates(tmp_path: Path):
    with Repository(tmp_path / "t.db") as repo:
        repo.upsert_images(
            [
                _meta("cat/a.png", size=10, cls="cat"),
                _meta("cat/b.png", size=20, cls="cat"),
                _meta("dog/c.png", size=30, cls="dog"),
            ]
        )
        assert repo.total_size_bytes() == 60
        classes = dict(repo.list_classes())
        assert classes["cat"] == 2
        assert classes["dog"] == 1
