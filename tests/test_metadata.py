"""Tests for core.metadata extraction and corruption handling."""

from __future__ import annotations

from pathlib import Path

from core.metadata import extract_metadata


def test_extracts_basic_fields(class_dataset: Path):
    img = next(p for p in class_dataset.rglob("*.jpg") if p.name != "broken.jpg")
    meta = extract_metadata(img, class_dataset)

    assert meta.corrupt is False
    assert meta.width and meta.width > 0
    assert meta.height and meta.height > 0
    assert meta.mode in {"RGB", "RGBA", "L", "P"}
    assert meta.size_bytes > 0
    assert meta.aspect_ratio == round(meta.width / meta.height, 6)
    assert meta.ext == ".jpg"


def test_class_label_from_folder(class_dataset: Path):
    img = next(p for p in (class_dataset / "cat").glob("*") if p.suffix == ".png")
    meta = extract_metadata(img, class_dataset)
    assert meta.class_label == "cat"


def test_flat_dataset_has_no_class_label(flat_dataset: Path):
    img = next(p for p in flat_dataset.glob("*.png"))
    meta = extract_metadata(img, flat_dataset)
    assert meta.class_label is None


def test_corrupt_file_flagged(class_dataset: Path):
    broken = class_dataset / "cat" / "broken.jpg"
    meta = extract_metadata(broken, class_dataset)
    assert meta.corrupt is True
    assert meta.error is not None
    assert meta.size_bytes > 0  # stat still worked


def test_missing_file_flagged(tmp_path: Path):
    meta = extract_metadata(tmp_path / "ghost.png", tmp_path)
    assert meta.corrupt is True
    assert meta.error is not None
