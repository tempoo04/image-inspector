"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from examples.make_synthetic import generate


@pytest.fixture
def class_dataset(tmp_path: Path) -> Path:
    """A small class-based dataset (3 classes, 5 each + 1 corrupt)."""
    root = tmp_path / "classes"
    generate(root, per_class=5, flat=False, seed=1)
    return root


@pytest.fixture
def flat_dataset(tmp_path: Path) -> Path:
    """A small flat dataset (10 images + 1 corrupt, no class folders)."""
    root = tmp_path / "flat"
    generate(root, per_class=10, flat=True, seed=2)
    return root
