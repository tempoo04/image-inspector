"""Tests for core.embeddings: projection, similarity, serialisation, storage.

These cover the always-available NumPy paths and the repository. The torch/timm
extractor is not exercised here (no weight download in the suite); it is smoke-
tested separately and guarded at runtime by ``backend_available``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.embeddings import (
    EMBED_MODEL,
    available_methods,
    from_blob,
    nearest,
    pca_2d,
    project,
    to_blob,
)
from database.repository import Repository


def _two_clusters(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 0.05, (10, 8)) + np.array([5, 0, 0, 0, 0, 0, 0, 0])
    b = rng.normal(0, 0.05, (10, 8)) + np.array([0, 5, 0, 0, 0, 0, 0, 0])
    return np.vstack([a, b]).astype(np.float32)


def test_blob_roundtrip():
    vec = np.array([1.5, -2.0, 3.25, 0.0], dtype=np.float32)
    back = from_blob(to_blob(vec))
    assert np.array_equal(back, vec)
    assert back.dtype == np.float32


def test_pca_2d_shape_and_determinism():
    mat = _two_clusters()
    c1 = pca_2d(mat)
    c2 = pca_2d(mat)
    assert c1.shape == (20, 2)
    assert np.allclose(c1, c2)


def test_pca_separates_clusters():
    coords = pca_2d(_two_clusters())
    # First component should split the two clusters: group means differ clearly.
    g1 = coords[:10, 0].mean()
    g2 = coords[10:, 0].mean()
    assert abs(g1 - g2) > 1.0


def test_available_methods_has_pca():
    assert "PCA" in available_methods()


def test_project_pca_matches_pca_2d():
    mat = _two_clusters()
    assert np.allclose(project(mat, "PCA"), pca_2d(mat))


def test_nearest_orders_by_similarity_and_excludes_self():
    # Row 0 points along +x; its nearest should be other +x-ish rows, not the -x one.
    mat = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )
    res = nearest(mat, query_index=0, k=3)
    idxs = [i for i, _ in res]
    assert 0 not in idxs  # self excluded
    assert idxs[0] == 1 and idxs[1] == 2  # closest first
    assert idxs[-1] == 3  # opposite vector ranks last
    # Similarities are descending.
    sims = [s for _, s in res]
    assert sims == sorted(sims, reverse=True)


def test_nearest_k_caps_results():
    mat = _two_clusters()
    assert len(nearest(mat, 0, k=5)) == 5


def test_repository_embedding_storage(class_dataset: Path, tmp_path: Path):
    from core.scanner import scan_dataset

    with Repository(tmp_path / "e.db") as repo:
        scan_dataset(class_dataset, repo)
        non_corrupt = repo.count_images(include_corrupt=False)
        assert repo.count_without_embeddings(EMBED_MODEL) == non_corrupt
        assert repo.count_with_embeddings(EMBED_MODEL) == 0

        # Store fake embeddings for the first few images.
        pending = repo.images_without_embeddings(EMBED_MODEL)
        items = []
        for k, row in enumerate(pending[:3]):
            vec = np.full(4, float(k), dtype=np.float32)
            items.append((row["id"], to_blob(vec), 4))
        assert repo.upsert_embeddings_many(EMBED_MODEL, items) == 3

        assert repo.count_with_embeddings(EMBED_MODEL) == 3
        assert repo.count_without_embeddings(EMBED_MODEL) == non_corrupt - 3

        rows = repo.embedding_rows(EMBED_MODEL)
        assert len(rows) == 3
        assert np.array_equal(from_blob(rows[0]["vector"]), np.zeros(4, dtype=np.float32))

        # Upsert is idempotent on (image_id, model_name).
        repo.upsert_embeddings_many(EMBED_MODEL, items)
        assert repo.count_with_embeddings(EMBED_MODEL) == 3

        # A different model namespaces separately.
        assert repo.count_with_embeddings("other") == 0

        assert repo.clear_embeddings(EMBED_MODEL) == 3
        assert repo.count_with_embeddings(EMBED_MODEL) == 0
