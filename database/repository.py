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
    "rel_path",
    "abs_path",
    "filename",
    "ext",
    "size_bytes",
    "width",
    "height",
    "mode",
    "aspect_ratio",
    "mtime",
    "class_label",
    "corrupt",
    "error",
    "scanned_at",
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
        rows = self._conn.execute("SELECT rel_path, size_bytes, mtime FROM images").fetchall()
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
                    m.rel_path,
                    m.abs_path,
                    m.filename,
                    m.ext,
                    m.size_bytes,
                    m.width,
                    m.height,
                    m.mode,
                    m.aspect_ratio,
                    m.mtime,
                    m.class_label,
                    int(m.corrupt),
                    m.error,
                    now,
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
                cur = conn.execute(f"DELETE FROM images WHERE rel_path IN ({placeholders})", chunk)
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
            self._conn.execute("SELECT COUNT(*) AS n FROM images WHERE corrupt = 1").fetchone()["n"]
        )

    def list_classes(self) -> list[tuple[str, int]]:
        """Return ``(class_label, count)`` pairs, newest query each call.

        Flat-dataset images (no class) are grouped under the label ``None``.
        """
        rows = self._conn.execute(
            "SELECT class_label, COUNT(*) AS n FROM images GROUP BY class_label ORDER BY n DESC"
        ).fetchall()
        return [(r["class_label"], r["n"]) for r in rows]

    def total_size_bytes(self) -> int:
        row = self._conn.execute("SELECT COALESCE(SUM(size_bytes), 0) AS s FROM images").fetchone()
        return int(row["s"])

    def get_image(self, image_id: int) -> sqlite3.Row | None:
        return self._conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()

    def iter_images(self, limit: int | None = None, offset: int = 0) -> list[sqlite3.Row]:
        sql = "SELECT * FROM images ORDER BY rel_path"
        params: list = []
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params = [limit, offset]
        return self._conn.execute(sql, params).fetchall()

    # -- browser: filtering, sorting, pagination (Module B) ------------------
    # Whitelist of sortable expressions; keys are the only values accepted from
    # the UI, so untrusted input can never reach the SQL string.
    _SORT_EXPR = {
        "name": "rel_path",
        "date": "mtime",
        "size": "size_bytes",
        "resolution": "(COALESCE(width, 0) * COALESCE(height, 0))",
    }

    def _build_filters(self, f: dict | None) -> tuple[str, list]:
        """Translate a filter dict into a WHERE clause and bind parameters.

        Recognized keys (all optional): ``class_label`` (str, or the sentinel
        ``"__flat__"`` for NULL class), ``exts`` (list[str]), ``filename``
        (substring), ``min_width``/``max_width``, ``min_height``/``max_height``,
        ``min_size``/``max_size``, ``min_aspect``/``max_aspect``,
        ``include_corrupt`` (bool, default True).
        """
        f = f or {}
        clauses: list[str] = []
        params: list = []

        if not f.get("include_corrupt", True):
            clauses.append("corrupt = 0")

        cls = f.get("class_label")
        if cls == "__flat__":
            clauses.append("class_label IS NULL")
        elif cls:
            clauses.append("class_label = ?")
            params.append(cls)

        exts = f.get("exts")
        if exts:
            placeholders = ",".join("?" for _ in exts)
            clauses.append(f"ext IN ({placeholders})")
            params.extend(exts)

        if name := f.get("filename"):
            clauses.append("filename LIKE ?")
            params.append(f"%{name}%")

        for key, col, op in (
            ("min_width", "width", ">="),
            ("max_width", "width", "<="),
            ("min_height", "height", ">="),
            ("max_height", "height", "<="),
            ("min_size", "size_bytes", ">="),
            ("max_size", "size_bytes", "<="),
            ("min_aspect", "aspect_ratio", ">="),
            ("max_aspect", "aspect_ratio", "<="),
        ):
            val = f.get(key)
            if val is not None:
                clauses.append(f"{col} {op} ?")
                params.append(val)

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        return where, params

    def list_images(
        self,
        filters: dict | None = None,
        sort: str = "name",
        descending: bool = False,
        limit: int = 60,
        offset: int = 0,
    ) -> tuple[list[sqlite3.Row], int]:
        """Return a page of images plus the total matching count.

        *sort* must be one of ``_SORT_EXPR``; unknown values fall back to name.
        """
        where, params = self._build_filters(filters)
        total = int(
            self._conn.execute(f"SELECT COUNT(*) AS n FROM images{where}", params).fetchone()["n"]
        )
        order_expr = self._SORT_EXPR.get(sort, self._SORT_EXPR["name"])
        direction = "DESC" if descending else "ASC"
        rows = self._conn.execute(
            f"SELECT * FROM images{where} ORDER BY {order_expr} {direction}, id LIMIT ? OFFSET ?",
            [*params, limit, offset],
        ).fetchall()
        return rows, total

    def distinct_exts(self) -> list[str]:
        rows = self._conn.execute("SELECT DISTINCT ext FROM images ORDER BY ext").fetchall()
        return [r["ext"] for r in rows]

    def count_by_ext(self) -> list[tuple[str, int]]:
        """Return ``(ext, count)`` pairs for the format distribution chart."""
        rows = self._conn.execute(
            "SELECT ext, COUNT(*) AS n FROM images GROUP BY ext ORDER BY n DESC"
        ).fetchall()
        return [(r["ext"], r["n"]) for r in rows]

    def value_ranges(self) -> dict[str, tuple]:
        """Min/max of numeric fields for building filter widgets.

        Returns ``{}`` when the catalog is empty so callers can show an empty
        state instead of rendering sliders with null bounds.
        """
        row = self._conn.execute(
            "SELECT MIN(width) AS wmin, MAX(width) AS wmax, "
            "MIN(height) AS hmin, MAX(height) AS hmax, "
            "MIN(size_bytes) AS smin, MAX(size_bytes) AS smax, "
            "MIN(aspect_ratio) AS amin, MAX(aspect_ratio) AS amax "
            "FROM images WHERE corrupt = 0"
        ).fetchone()
        if row is None or row["wmin"] is None:
            return {}
        return {
            "width": (row["wmin"], row["wmax"]),
            "height": (row["hmin"], row["hmax"]),
            "size": (row["smin"], row["smax"]),
            "aspect": (row["amin"], row["amax"]),
        }
