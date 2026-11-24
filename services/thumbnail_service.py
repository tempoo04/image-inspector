"""Thumbnail generation and on-disk caching (Module B).

Thumbnails are generated lazily on first request and cached as JPEGs on disk.
The cache key includes the source path, its mtime, and the target edge size, so
a modified or replaced source image transparently invalidates its thumbnail.

Only one image is decoded at a time; full-resolution pixels are never retained
beyond generating the (small) thumbnail, honoring the "never load the whole
dataset into RAM" invariant.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from core.config import ThumbnailSettings
from core.logging_config import get_logger

log = get_logger(__name__)


class ThumbnailService:
    """Generate and cache thumbnails under a configured cache directory."""

    def __init__(self, settings: ThumbnailSettings | None = None) -> None:
        self.settings = settings or ThumbnailSettings()
        self.cache_dir = Path(self.settings.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, abs_path: str, mtime: float) -> Path:
        """Deterministic cache filename for a source path + mtime + size."""
        key = f"{abs_path}|{mtime}|{self.settings.max_edge}"
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]
        return self.cache_dir / f"{digest}.jpg"

    def get_thumbnail(self, abs_path: str | Path, mtime: float) -> Path | None:
        """Return a cached thumbnail path, generating it if needed.

        Returns ``None`` if the source image cannot be opened (corrupt, missing,
        or unsupported); callers should render a placeholder in that case. Never
        raises for a bad image.
        """
        abs_path = str(abs_path)
        cache_path = self._cache_path(abs_path, mtime)
        if cache_path.exists():
            return cache_path
        return self._generate(abs_path, cache_path)

    def _generate(self, abs_path: str, cache_path: Path) -> Path | None:
        try:
            with Image.open(abs_path) as img:
                # Respect EXIF orientation; convert to RGB for JPEG cache.
                img = ImageOps.exif_transpose(img)
                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")
                img.thumbnail(
                    (self.settings.max_edge, self.settings.max_edge),
                    Image.Resampling.LANCZOS,
                )
                img.save(cache_path, format="JPEG", quality=self.settings.jpeg_quality)
            return cache_path
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            log.warning("thumbnail failed for %s: %s", abs_path, exc)
            # Remove any partial file so a retry regenerates cleanly.
            cache_path.unlink(missing_ok=True)
            return None

    def clear_cache(self) -> int:
        """Delete all cached thumbnails. Returns count removed."""
        removed = 0
        for f in self.cache_dir.glob("*.jpg"):
            try:
                f.unlink()
                removed += 1
            except OSError:
                pass
        return removed
