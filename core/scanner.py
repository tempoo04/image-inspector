"""Dataset scanner and indexer (Module A).

Recursively walks a dataset directory, extracts metadata for supported image
files, and streams rows into the repository in batches. Supports both
class-based folder layouts and flat image directories, and rescans
incrementally by skipping files whose size and mtime are unchanged.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from .config import is_supported
from .logging_config import get_logger
from .metadata import ImageMetadata, extract_metadata

log = get_logger(__name__)

ProgressCb = Callable[[int, int], None]


@dataclass
class ScanResult:
    """Summary of one scan pass."""

    root: str
    layout: str                 # "class-based" | "flat" | "mixed" | "empty"
    total_files: int = 0        # supported image files found on disk
    processed: int = 0          # files for which metadata was (re)extracted
    written: int = 0            # rows inserted or updated
    skipped_unchanged: int = 0  # files skipped by incremental check
    corrupt: int = 0            # files flagged unreadable during this pass
    deleted: int = 0            # stale rows pruned (if prune_missing)
    duration_s: float = 0.0


def _iter_image_files(root: Path) -> Iterator[Path]:
    """Yield supported image files under *root*, recursively and deterministically."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()  # stable traversal order
        for name in sorted(filenames):
            if is_supported(name):
                yield Path(dirpath) / name


def detect_layout(root: Path) -> str:
    """Classify dataset layout by where images sit relative to *root*.

    Returns ``"flat"`` (images directly under root only), ``"class-based"``
    (images only inside subfolders), ``"mixed"`` (both), or ``"empty"``.
    """
    has_root_level = False
    has_nested = False
    for f in _iter_image_files(root):
        if f.parent == root:
            has_root_level = True
        else:
            has_nested = True
        if has_root_level and has_nested:
            return "mixed"
    if has_root_level:
        return "flat"
    if has_nested:
        return "class-based"
    return "empty"


def scan_dataset(
    root: Path | str,
    repo,
    *,
    batch_size: int = 500,
    prune_missing: bool = False,
    progress_cb: ProgressCb | None = None,
) -> ScanResult:
    """Scan *root* and index images into *repo*.

    Parameters
    ----------
    root:
        Dataset directory. Must exist and be a directory.
    repo:
        A :class:`database.repository.Repository` (or compatible) providing
        ``load_signatures``, ``upsert_images`` and ``delete_by_rel_paths``.
    batch_size:
        Rows per database commit.
    prune_missing:
        If True, delete indexed rows whose files are no longer on disk.
    progress_cb:
        Optional ``callback(processed, total)`` invoked as files are handled.

    Raises
    ------
    NotADirectoryError
        If *root* does not exist or is not a directory.
    """
    import time

    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {root}")

    start = time.time()
    result = ScanResult(root=str(root), layout=detect_layout(root))

    # Pre-count only when progress reporting is requested (one extra name-only
    # walk; no stat/decode cost).
    total = sum(1 for _ in _iter_image_files(root)) if progress_cb else 0
    result.total_files = total

    signatures = repo.load_signatures()
    seen_rel_paths: set[str] = set()

    def changed_metas() -> Iterator[ImageMetadata]:
        """Yield metadata for new/changed files; update counters as a side effect."""
        for path in _iter_image_files(root):
            rel = path.relative_to(root).as_posix() if path.is_relative_to(root) else path.name
            seen_rel_paths.add(rel)

            # Cheap incremental check: stat only, skip if size+mtime unchanged.
            prev = signatures.get(rel)
            if prev is not None:
                try:
                    st = path.stat()
                    if st.st_size == prev[0] and st.st_mtime == prev[1]:
                        result.skipped_unchanged += 1
                        _emit(progress_cb, result, total)
                        continue
                except OSError:
                    pass  # fall through to full extraction, which records the error

            meta = extract_metadata(path, root)
            result.processed += 1
            if meta.corrupt:
                result.corrupt += 1
            _emit(progress_cb, result, total)
            yield meta

    result.written = repo.upsert_images(changed_metas(), batch_size=batch_size)

    if prune_missing:
        stale = [rp for rp in signatures if rp not in seen_rel_paths]
        result.deleted = repo.delete_by_rel_paths(stale)

    result.duration_s = round(time.time() - start, 3)
    log.info(
        "Scan complete: %s layout=%s written=%d skipped=%d corrupt=%d deleted=%d in %.2fs",
        root, result.layout, result.written, result.skipped_unchanged,
        result.corrupt, result.deleted, result.duration_s,
    )
    return result


def _emit(cb: ProgressCb | None, result: ScanResult, total: int) -> None:
    if cb is not None:
        done = result.processed + result.skipped_unchanged
        cb(done, total)
