"""Image metadata extraction (Module A).

Reads lightweight metadata from image headers without decoding full pixel data
into RAM. Corruption is detected and reported via a flag rather than raised, so
a single bad file never aborts a dataset scan.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .logging_config import get_logger

log = get_logger(__name__)

# Pillow refuses very large images by default (decompression-bomb guard). We
# keep that guard but allow it to flag rather than crash the scan.
Image.MAX_IMAGE_PIXELS = None  # disable hard error; we size-check ourselves


@dataclass
class ImageMetadata:
    """Header-level metadata for one image file.

    When ``corrupt`` is True the pixel-dependent fields (width/height/mode) may
    be ``None`` and ``error`` carries a short reason.
    """

    abs_path: str
    rel_path: str
    filename: str
    ext: str
    size_bytes: int
    mtime: float
    width: int | None = None
    height: int | None = None
    mode: str | None = None
    aspect_ratio: float | None = None
    class_label: str | None = None
    corrupt: bool = False
    error: str | None = None


def _relative_path(path: Path, root: Path) -> str:
    """Portable POSIX-style relative path of *path* under *root*.

    Falls back to the filename if *path* is not under *root* (e.g. symlink
    escaping the tree), keeping the DB key stable and cross-platform.
    """
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _class_label(path: Path, root: Path) -> str | None:
    """Infer a class label from class-based folder layout.

    For ``root/<class>/image.jpg`` returns ``<class>``. For a flat directory
    (image directly under root) returns ``None``.
    """
    rel = _relative_path(path, root)
    parts = rel.split("/")
    return parts[0] if len(parts) > 1 else None


def extract_metadata(path: Path | str, root: Path | str) -> ImageMetadata:
    """Extract metadata for a single image, flagging corruption on failure.

    *root* is the dataset root used to compute the relative path and class
    label. Filesystem (``stat``) errors and decode errors are both captured on
    the returned object; this function does not raise for bad images.
    """
    path = Path(path)
    root = Path(root)
    rel = _relative_path(path, root)

    # stat() first: a file we cannot stat cannot be indexed at all.
    try:
        st = path.stat()
        size_bytes, mtime = st.st_size, st.st_mtime
    except OSError as exc:
        log.warning("stat failed for %s: %s", path, exc)
        return ImageMetadata(
            abs_path=str(path),
            rel_path=rel,
            filename=path.name,
            ext=path.suffix.lower(),
            size_bytes=0,
            mtime=0.0,
            class_label=_class_label(path, root),
            corrupt=True,
            error=f"stat: {exc}",
        )

    meta = ImageMetadata(
        abs_path=str(path),
        rel_path=rel,
        filename=path.name,
        ext=path.suffix.lower(),
        size_bytes=size_bytes,
        mtime=mtime,
        class_label=_class_label(path, root),
    )

    # Read dimensions/mode from the header (does not load pixels), then verify
    # integrity on a second handle because verify() leaves the image unusable.
    try:
        with Image.open(path) as img:
            meta.width, meta.height = img.width, img.height
            meta.mode = img.mode
        with Image.open(path) as img:
            img.verify()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        log.warning("corrupt/unreadable image %s: %s", path, exc)
        meta.corrupt = True
        meta.error = f"decode: {exc}"
        return meta

    if meta.height and meta.height > 0 and meta.width:
        meta.aspect_ratio = round(meta.width / meta.height, 6)
    return meta
