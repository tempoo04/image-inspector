"""Deep-feature embeddings and 2-D projection (Module G).

Optional, heavyweight module. The feature extractor needs ``torch`` + ``timm``
(install ``requirements-optional.txt``); if they are absent the app degrades
gracefully — :func:`backend_available` is ``False`` and the page explains how to
enable it. The pure pieces here (PCA projection, cosine nearest-neighbours,
vector (de)serialisation) depend only on NumPy and are always available.

Embeddings are 512-d ResNet-18 features, L2-comparable, stored as float32 BLOBs
in the ``embeddings`` table keyed by ``(image_id, model_name)``. A batch helper
(:func:`compute_embeddings_for_dataset`) drives extraction over images lacking
an embedding for the active model.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from importlib.util import find_spec

import numpy as np

from .logging_config import get_logger

log = get_logger(__name__)

ProgressCb = Callable[[int, int], None]

EMBED_MODEL = "resnet18"
EMBED_DIM = 512
_VEC_DTYPE = np.float32

# Lazily-built (model, transform); kept at module scope so repeated batches in a
# single process reuse the loaded network and its downloaded weights.
_MODEL_CACHE: tuple = ()


class EmbeddingBackendUnavailable(RuntimeError):
    """Raised when embedding extraction is requested but torch/timm are missing."""


def backend_available() -> bool:
    """True if the deep-feature extractor can run (torch + timm importable)."""
    return find_spec("torch") is not None and find_spec("timm") is not None


# -- vector (de)serialisation ------------------------------------------------
def to_blob(vec: np.ndarray) -> bytes:
    """Serialise a 1-D float vector to a compact float32 byte string."""
    return np.ascontiguousarray(vec, dtype=_VEC_DTYPE).tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    """Deserialise a float32 BLOB back to a 1-D NumPy array."""
    return np.frombuffer(blob, dtype=_VEC_DTYPE)


# -- feature extraction (torch) ----------------------------------------------
def _load_model():
    global _MODEL_CACHE
    if _MODEL_CACHE:
        return _MODEL_CACHE
    if not backend_available():
        raise EmbeddingBackendUnavailable(
            "Embedding extraction needs torch + timm. "
            "Install them with: pip install -r requirements-optional.txt"
        )
    import timm
    import torch
    from timm.data import create_transform, resolve_data_config

    model = timm.create_model(EMBED_MODEL, pretrained=True, num_classes=0)
    model.eval()
    transform = create_transform(**resolve_data_config({}, model=model))
    torch.set_grad_enabled(False)
    _MODEL_CACHE = (model, transform)
    return _MODEL_CACHE


def extract_embeddings(
    abs_paths: Sequence[str], *, batch_size: int = 32
) -> list[np.ndarray | None]:
    """Extract one L2-normalised feature vector per path (``None`` if unreadable).

    Raises :class:`EmbeddingBackendUnavailable` if torch/timm are not installed.
    Images that fail to open are returned as ``None`` so a single bad file never
    aborts a batch.
    """
    import torch
    from PIL import Image, UnidentifiedImageError

    model, transform = _load_model()
    out: list[np.ndarray | None] = [None] * len(abs_paths)

    # Group decodable images into tensor batches; keep a map back to positions.
    buf_idx: list[int] = []
    buf_tensors: list = []

    def flush() -> None:
        if not buf_tensors:
            return
        batch = torch.stack(buf_tensors)
        feats = model(batch).cpu().numpy().astype(_VEC_DTYPE)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        feats = feats / np.clip(norms, 1e-8, None)
        for pos, vec in zip(buf_idx, feats):
            out[pos] = vec
        buf_idx.clear()
        buf_tensors.clear()

    for i, path in enumerate(abs_paths):
        try:
            with Image.open(path) as im:
                tensor = transform(im.convert("RGB"))
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            log.warning("embedding: could not read %s: %s", path, exc)
            continue
        buf_idx.append(i)
        buf_tensors.append(tensor)
        if len(buf_tensors) >= batch_size:
            flush()
    flush()
    return out


def compute_embeddings_for_dataset(
    repo,
    *,
    model_name: str = EMBED_MODEL,
    batch_size: int = 32,
    progress_cb: ProgressCb | None = None,
) -> int:
    """Extract and store embeddings for every indexed image lacking one.

    *repo* must provide ``images_without_embeddings`` and
    ``upsert_embeddings_many``. Returns the number stored. Undecodable images are
    skipped. Raises :class:`EmbeddingBackendUnavailable` if the backend is absent.
    """
    pending = repo.images_without_embeddings(model_name)
    total = len(pending)
    done = 0
    stored = 0
    for start in range(0, total, batch_size):
        chunk = pending[start : start + batch_size]
        vecs = extract_embeddings([r["abs_path"] for r in chunk], batch_size=batch_size)
        items = [
            (row["id"], to_blob(vec), EMBED_DIM) for row, vec in zip(chunk, vecs) if vec is not None
        ]
        if items:
            repo.upsert_embeddings_many(model_name, items)
            stored += len(items)
        done += len(chunk)
        if progress_cb is not None:
            progress_cb(done, total)
    return stored


# -- projection (optional backends) ------------------------------------------
def available_methods() -> list[str]:
    """2-D projection methods usable right now (PCA always; others if installed)."""
    methods = ["PCA"]
    if find_spec("umap") is not None:
        methods.append("UMAP")
    if find_spec("sklearn") is not None:
        methods.append("t-SNE")
    return methods


def pca_2d(matrix: np.ndarray) -> np.ndarray:
    """Project rows onto their first two principal components (NumPy SVD)."""
    x = np.asarray(matrix, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    # Economy SVD; columns of Vh are principal axes.
    _, _, vh = np.linalg.svd(x, full_matrices=False)
    comps = vh[:2].T if vh.shape[0] >= 2 else np.pad(vh, ((0, 2 - vh.shape[0]), (0, 0))).T
    return x @ comps


def project(matrix: np.ndarray, method: str) -> np.ndarray:
    """Project an (n, d) matrix to (n, 2) using *method* from :func:`available_methods`."""
    if method == "PCA":
        return pca_2d(matrix)
    if method == "UMAP":
        import umap

        n = len(matrix)
        reducer = umap.UMAP(n_components=2, n_neighbors=min(15, max(2, n - 1)))
        return np.asarray(reducer.fit_transform(matrix))
    if method == "t-SNE":
        from sklearn.manifold import TSNE

        n = len(matrix)
        tsne = TSNE(n_components=2, perplexity=min(30, max(2, (n - 1) / 3)), init="pca")
        return np.asarray(tsne.fit_transform(np.asarray(matrix, dtype=np.float64)))
    raise ValueError(f"Unknown projection method: {method}")


# -- similarity --------------------------------------------------------------
def nearest(matrix: np.ndarray, query_index: int, k: int = 8) -> list[tuple[int, float]]:
    """Return up to *k* ``(row_index, cosine_similarity)`` nearest to *query_index*.

    Excludes the query row itself; results are ordered by descending similarity.
    """
    x = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    unit = x / np.clip(norms, 1e-8, None)
    sims = unit @ unit[query_index]
    order = np.argsort(-sims)
    out: list[tuple[int, float]] = []
    for idx in order:
        if idx == query_index:
            continue
        out.append((int(idx), float(sims[idx])))
        if len(out) >= k:
            break
    return out
