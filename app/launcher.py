"""Command-line entry point.

Two uses:

* ``--scan <dir>``  Index a dataset into SQLite from the terminal (works today,
  no UI required). Prints a live progress line and a summary.
* ``--ui``          Launch the Streamlit interface (available from Milestone 2).

Examples::

    python -m app.launcher --scan data/sample
    python -m app.launcher --scan data/sample --db data/catalog.db --prune
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core.config import AppConfig
from core.scanner import scan_dataset
from database.repository import Repository


def _run_scan(root: Path, db_path: Path, prune: bool) -> int:
    cfg = AppConfig(db_path=db_path)
    cfg.ensure_dirs()

    last = {"pct": -1}

    def progress(done: int, total: int) -> None:
        if total <= 0:
            return
        pct = int(done * 100 / total)
        if pct != last["pct"]:
            last["pct"] = pct
            print(f"\rScanning... {pct:3d}%  ({done}/{total})", end="", flush=True)

    with Repository(db_path) as repo:
        res = scan_dataset(root, repo, batch_size=cfg.scan_batch_size,
                           prune_missing=prune, progress_cb=progress)
    print()  # newline after progress line
    print(
        f"Layout: {res.layout}\n"
        f"Indexed: {res.written}  Skipped (unchanged): {res.skipped_unchanged}\n"
        f"Corrupt: {res.corrupt}  Deleted: {res.deleted}\n"
        f"Duration: {res.duration_s}s\n"
        f"Database: {db_path}"
    )
    return 0


def _run_ui() -> int:
    print(
        "The Streamlit UI arrives in Milestone 2.\n"
        "For now, index a dataset with:  python -m app.launcher --scan <dir>",
        file=sys.stderr,
    )
    return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="vision-explorer", description=__doc__)
    ap.add_argument("--scan", metavar="DIR", type=Path, help="dataset directory to index")
    ap.add_argument("--db", type=Path, default=AppConfig().db_path, help="SQLite catalog path")
    ap.add_argument("--prune", action="store_true", help="remove index rows for deleted files")
    ap.add_argument("--ui", action="store_true", help="launch the Streamlit UI (M2+)")
    args = ap.parse_args(argv)

    if args.scan:
        return _run_scan(args.scan, args.db, args.prune)
    if args.ui:
        return _run_ui()
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
