"""Image quality metrics (Module D).

Pure computation over a single image: brightness, contrast, sharpness
(variance of Laplacian), entropy, per-channel means, and derived quality flags.
No database or UI dependency. A batch helper (:func:`compute_metrics_for_dataset`)
drives computation over images that do not yet have metrics, via a repository.

Thresholds come from :class:`core.config.QualityThresholds` so flagging is
configurable.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

from .config import QualityThresholds
from .logging_config import get_logger

log = get_logger(__name__)

ProgressCb = Callable[[int, int], None]


@dataclass
class ImageQualityMetrics:
    """Computed quality metrics and flags for one image (maps to the metrics table)."""

    brightness: float
    contrast: float
    sharpness: float
    entropy: float
    mean_r: float
    mean_g: float
    mean_b: float
    is_black: bool
    is_white: bool
    is_dark: bool
    is_bright: bool
    is_blurry: bool
    is_small: bool


def _shannon_entropy(gray: np.ndarray) -> float:
    """Shannon entropy (bits) of an 8-bit grayscale image's intensity histogram."""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    total = hist.sum()
    if total <= 0:
        return 0.0
    p = hist[hist > 0] / total
    return float(-(p * np.log2(p)).sum())


def compute_metrics(
    abs_path: str,
    thresholds: QualityThresholds,
    width: int | None = None,
    height: int | None = None,
) -> ImageQualityMetrics | None:
    """Compute metrics for the image at *abs_path*.

    Returns ``None`` if the image cannot be decoded (corrupt/unsupported), so a
    single bad file never aborts a batch. *width*/*height* (from the catalog)
    are used for the "too small" flag; if omitted they are read from the image.
    """
    img = cv2.imread(abs_path, cv2.IMREAD_COLOR)
    if img is None:
        log.warning("quality metrics: could not read %s", abs_path)
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    entropy = _shannon_entropy(gray)

    # cv2 loads BGR; split to per-channel means.
    mean_b = float(img[:, :, 0].mean())
    mean_g = float(img[:, :, 1].mean())
    mean_r = float(img[:, :, 2].mean())

    h, w = gray.shape[:2]
    min_edge = min(width or w, height or h)

    t = thresholds
    return ImageQualityMetrics(
        brightness=round(brightness, 3),
        contrast=round(contrast, 3),
        sharpness=round(sharpness, 3),
        entropy=round(entropy, 4),
        mean_r=round(mean_r, 2),
        mean_g=round(mean_g, 2),
        mean_b=round(mean_b, 2),
        is_black=brightness <= t.black_mean_max,
        is_white=brightness >= t.white_mean_min,
        is_dark=brightness <= t.dark_mean_max,
        is_bright=brightness >= t.bright_mean_min,
        is_blurry=sharpness < t.blur_laplacian_max,
        is_small=min_edge < t.min_dimension_px,
    )


def rgb_histograms(abs_path: str, bins: int = 256) -> dict[str, np.ndarray] | None:
    """Return per-channel intensity histograms for display, or ``None`` if unreadable."""
    img = cv2.imread(abs_path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    out: dict[str, np.ndarray] = {}
    for idx, name in ((2, "r"), (1, "g"), (0, "b")):
        out[name] = cv2.calcHist([img], [idx], None, [bins], [0, 256]).ravel()
    return out


def compute_metrics_for_dataset(
    repo,
    thresholds: QualityThresholds,
    *,
    batch_size: int = 200,
    progress_cb: ProgressCb | None = None,
) -> int:
    """Compute and store metrics for every indexed image lacking them.

    *repo* must provide ``images_without_metrics`` and ``upsert_metrics_many``.
    Returns the number of images for which metrics were computed and stored.
    Images that fail to decode are skipped (and remain without metrics).
    """
    pending = repo.images_without_metrics()
    total = len(pending)
    done = 0
    computed = 0
    buffer: list[tuple[int, ImageQualityMetrics]] = []

    for row in pending:
        m = compute_metrics(row["abs_path"], thresholds, row["width"], row["height"])
        done += 1
        if m is not None:
            buffer.append((row["id"], m))
            computed += 1
        if len(buffer) >= batch_size:
            repo.upsert_metrics_many(buffer)
            buffer.clear()
        if progress_cb is not None:
            progress_cb(done, total)
    if buffer:
        repo.upsert_metrics_many(buffer)
    return computed
