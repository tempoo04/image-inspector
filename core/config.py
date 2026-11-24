"""Central configuration: supported formats, quality thresholds, and paths.

Pure data + helpers. Imports nothing from the app or database layers so it can
be used safely from any module (core, services, UI, tests).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

# --- Supported image formats -------------------------------------------------
# Lower-case extensions including the leading dot. Kept as a frozenset for O(1)
# membership tests during directory scanning.
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
)


def is_supported(path: Path | str) -> bool:
    """Return True if *path* has a supported image extension (case-insensitive)."""
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


# --- Default filesystem layout ----------------------------------------------
# Resolved relative to the repository root (two levels up from this file:
# core/config.py -> repo root).
REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR: Path = REPO_ROOT / "data"
DEFAULT_DB_PATH: Path = DEFAULT_DATA_DIR / "catalog.db"
DEFAULT_THUMBNAIL_DIR: Path = DEFAULT_DATA_DIR / "thumbnails"


@dataclass(frozen=True)
class QualityThresholds:
    """Configurable thresholds for image-quality flagging (Module D).

    All values are inclusive bounds used by ``core.image_metrics``. Brightness
    and channel means are on a 0-255 scale; sharpness is variance-of-Laplacian.
    """

    dark_mean_max: float = 25.0  # <= flagged "too dark"
    bright_mean_min: float = 230.0  # >= flagged "too bright"
    black_mean_max: float = 2.0  # <= flagged "effectively black"
    white_mean_min: float = 253.0  # >= flagged "effectively white"
    blur_laplacian_max: float = 100.0  # < flagged "blurry"
    low_entropy_max: float = 1.0  # < flagged "low information"
    min_dimension_px: int = 32  # min(width,height) < flagged "too small"


@dataclass
class ThumbnailSettings:
    """Thumbnail generation + caching (Module B)."""

    max_edge: int = 256  # longest edge in pixels
    jpeg_quality: int = 80  # cache format quality
    cache_dir: Path = field(default_factory=lambda: DEFAULT_THUMBNAIL_DIR)


@dataclass
class AppConfig:
    """Top-level runtime configuration passed through the app."""

    db_path: Path = field(default_factory=lambda: DEFAULT_DB_PATH)
    data_dir: Path = field(default_factory=lambda: DEFAULT_DATA_DIR)
    thumbnails: ThumbnailSettings = field(default_factory=ThumbnailSettings)
    thresholds: QualityThresholds = field(default_factory=QualityThresholds)
    scan_batch_size: int = 500  # rows per DB commit during scanning
    browser_page_size: int = 60  # thumbnails per browser page

    def ensure_dirs(self) -> None:
        """Create data + thumbnail directories if missing. Safe to call repeatedly."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.thumbnails.cache_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        """Serializable view (Paths rendered as strings) for logging/UI display."""
        d = asdict(self)
        d["db_path"] = str(self.db_path)
        d["data_dir"] = str(self.data_dir)
        d["thumbnails"]["cache_dir"] = str(self.thumbnails.cache_dir)
        return d
