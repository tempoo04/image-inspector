"""Data-access layer over SQLite.

All SQL lives here; callers work with :class:`~core.metadata.ImageMetadata` and
plain dicts/rows. The repository owns one connection, enables WAL (concurrent
reads during a scan) and foreign-key cascades, and writes images in batches.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

from core.logging_config import get_logger
from core.metadata import ImageMetadata

from .models import DDL_STATEMENTS, SCHEMA_VERSION

log = get_logger(__name__)

# Columns written per image row, in bind order.
_IMAGE_COLUMNS = (
    "rel_path", "abs_path", "filename", "ext", "size_bytes", "width", "height",
    "mode", "aspect_ratio", "mtime", "class_label", "corrupt", "error", "scanned_at",
)

_UPSERT_SQL = f"""
INSERT INTO images ({", ".join(_IMAGE_COLUMNS)})
VALUES ({", ".join("?" for _ in _IMAGE_COLUMNS)})
ON CONFLICT(rel_path) DO UPDATE SET
    abs_path=excluded.abs_path,
    filename=excluded.filename,
    ext=excluded.ext,
    size_bytes=excluded.size_bytes,
    width=excluded.width,
    height=excluded.height,
    mode=excluded.mode,
    aspect_ratio=excluded.aspect_ratio,
    mtime=excluded.mtime,
    class_label=excluded.class_label,
    corrupt=excluded.corrupt,
    error=excluded.error,
    scanned_at=excluded.scanned_at;
"""


class Repository:
    """Thin SQLite gateway for the image catalog.

    Use as a context manager (``with Repository(path) as repo:``) or call
    :meth:`close` explicitly. Not safe to share one instance across threads.
    """

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self.init_schema()

    # -- lifecycle -----------------------------------------------------------
    def __enter__(self) -> "Repository":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None  # type: ignore[assignment]

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        """Transaction scope: commit on success, roll back on error."""
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # -- schema --------------------------------------------------------------
    def init_schema(self) -> None:
        with self._tx() as conn:
            for stmt in DDL_STATEMENTS:
                conn.execute(stmt)
            conn.execute(
                "INSERT INTO schema_meta(key, value) VALUES('version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value;",
                (str(SCHEMA_VERSION),),
            )

    # -- incremental scan support -------------------------------------------
    def load_signatures(self) -> dict[str, tuple[int, float]]:
        """Return ``{rel_path: (size_bytes, mtime)}`` for every indexed image.

        Loaded once per scan so the scanner can skip unchanged files in memory
        without a per-file database round-trip.
        """
        rows = self._conn.execute(
            "SELECT rel_path, size_bytes, mtime FROM images"
        ).fetchall()
        return {r["rel_path"]: (r["size_bytes"], r["mtime"]) for r in rows}

    def upsert_images(self, metas: Iterable[ImageMetadata], batch_size: int = 500) -> int:
        """Insert or update image rows in batches. Returns rows written."""
        now = time.time()
        batch: list[tuple] = []
        written = 0

        def flush() -> None:
            nonlocal written
            if not batch:
                return
            with self._tx() as conn:
                conn.executemany(_UPSERT_SQL, batch)
            written += len(batch)
            batch.clear()

        for m in metas:
            batch.append(
                (
                    m.rel_path, m.abs_path, m.filename, m.ext, m.size_bytes,
                    m.width, m.height, m.mode, m.aspect_ratio, m.mtime,
                    m.class_label, int(m.corrupt), m.error, now,
                )
            )
            if len(batch) >= batch_size:
                flush()
        flush()
        return written

    def delete_by_rel_paths(self, rel_paths: Iterable[str]) -> int:
        """Remove rows whose files no longer exist. Returns rows deleted."""
        rel_paths = list(rel_paths)
        if not rel_paths:
            return 0
        deleted = 0
        with self._tx() as conn:
            for i in range(0, len(rel_paths), 500):
                chunk = rel_paths[i : i + 500]
                placeholders = ",".join("?" for _ in chunk)
                cur = conn.execute(
                    f"DELETE FROM images WHERE rel_path IN ({placeholders})", chunk
                )
                deleted += cur.rowcount
        return deleted

    # -- queries -------------------------------------------------------------
    def count_images(self, include_corrupt: bool = True) -> int:
        sql = "SELECT COUNT(*) AS n FROM images"
        if not include_corrupt:
            sql += " WHERE corrupt = 0"
        return int(self._conn.execute(sql).fetchone()["n"])

    def count_corrupt(self) -> int:
        return int(
            self._conn.execute(
                "SELECT COUNT(*) AS n FROM images WHERE corrupt = 1"
            ).fetchone()["n"]
        )

    def list_classes(self) -> list[tuple[str, int]]:
        """Return ``(class_label, count)`` pairs, newest query each call.

        Flat-dataset images (no class) are grouped under the label ``None``.
        """
        rows = self._conn.execute(
            "SELECT class_label, COUNT(*) AS n FROM images "
            "GROUP BY class_label ORDER BY n DESC"
        ).fetchall()
        return [(r["class_label"], r["n"]) for r in rows]

    def total_size_bytes(self) -> int:
        row = self._conn.execute(
            "SELECT COALESCE(SUM(size_bytes), 0) AS s FROM images"
        ).fetchone()
        return int(row["s"])

    def get_image(self, image_id: int) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM images WHERE id = ?", (image_id,)
        ).fetchone()

    def iter_images(self, limit: int | None = None, offset: int = 0) -> list[sqlite3.Row]:
        sql = "SELECT * FROM images ORDER BY rel_path"
        params: list = []
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params = [limit, offset]
        return self._conn.execute(sql, params).fetchall()
