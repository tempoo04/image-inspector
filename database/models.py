"""SQLite schema definition.

One source of truth for table DDL and indexes. The ``images`` table is keyed by
``rel_path`` (unique) so incremental rescans can match files by stable identity
across runs and platforms. Metric/hash/embedding tables reference it by id and
cascade on delete.

``schema_version`` guards future migrations.
"""

from __future__ import annotations

SCHEMA_VERSION = 1

DDL_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS schema_meta (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS images (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        rel_path     TEXT    NOT NULL UNIQUE,
        abs_path     TEXT    NOT NULL,
        filename     TEXT    NOT NULL,
        ext          TEXT    NOT NULL,
        size_bytes   INTEGER NOT NULL,
        width        INTEGER,
        height       INTEGER,
        mode         TEXT,
        aspect_ratio REAL,
        mtime        REAL    NOT NULL,
        class_label  TEXT,
        corrupt      INTEGER NOT NULL DEFAULT 0,
        error        TEXT,
        scanned_at   REAL    NOT NULL
    );
    """,
    # Metrics computed lazily (Module D); 1:1 with images.
    """
    CREATE TABLE IF NOT EXISTS metrics (
        image_id    INTEGER PRIMARY KEY REFERENCES images(id) ON DELETE CASCADE,
        brightness  REAL,
        contrast    REAL,
        sharpness   REAL,
        entropy     REAL,
        mean_r      REAL,
        mean_g      REAL,
        mean_b      REAL,
        is_black    INTEGER,
        is_white    INTEGER,
        is_dark     INTEGER,
        is_bright   INTEGER,
        is_blurry   INTEGER,
        is_small    INTEGER,
        computed_at REAL
    );
    """,
    # Exact + perceptual hashes (Module F).
    """
    CREATE TABLE IF NOT EXISTS hashes (
        image_id    INTEGER PRIMARY KEY REFERENCES images(id) ON DELETE CASCADE,
        sha256      TEXT,
        phash       TEXT,
        computed_at REAL
    );
    """,
    # Cached embedding vectors (Module G); many models per image possible.
    """
    CREATE TABLE IF NOT EXISTS embeddings (
        image_id    INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
        model_name  TEXT    NOT NULL,
        dim         INTEGER NOT NULL,
        vector      BLOB    NOT NULL,
        computed_at REAL,
        PRIMARY KEY (image_id, model_name)
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_images_class    ON images(class_label);",
    "CREATE INDEX IF NOT EXISTS idx_images_ext      ON images(ext);",
    "CREATE INDEX IF NOT EXISTS idx_images_corrupt  ON images(corrupt);",
    "CREATE INDEX IF NOT EXISTS idx_images_size     ON images(size_bytes);",
    "CREATE INDEX IF NOT EXISTS idx_hashes_sha256   ON hashes(sha256);",
    "CREATE INDEX IF NOT EXISTS idx_hashes_phash    ON hashes(phash);",
)
