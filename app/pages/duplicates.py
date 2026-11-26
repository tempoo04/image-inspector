"""Duplicate Detection page (Module F): exact + near-duplicate review.

Computes sha256 and pHash over the dataset, then surfaces:

* **Exact duplicates** — identical bytes (``GROUP BY sha256``).
* **Near duplicates** — visually similar images within a Hamming threshold on
  their pHash.

Review only — the app never deletes originals. A CSV of the duplicate list can
be exported for use by an external cleanup step.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.context import get_thumbnail_service
from app.ui_utils import human_bytes
from core.config import AppConfig
from core.hashing import compute_hashes_for_dataset, group_near, hamming
from database.repository import Repository

_THUMB_COLS = 5
_MAX_THUMBS = 24  # per group, to keep the page responsive


def _run_compute(repo: Repository) -> None:
    progress = st.progress(0.0, text="Hashing images…")

    def cb(done: int, total: int) -> None:
        if total:
            progress.progress(min(done / total, 1.0), text=f"Hashing… {done}/{total}")

    computed = compute_hashes_for_dataset(repo, progress_cb=cb)
    progress.empty()
    st.success(f"Hashed {computed} image(s).")
    st.rerun()


def _thumb_grid(rows: list, cfg: AppConfig, *, captions: dict | None = None) -> None:
    thumbs = get_thumbnail_service(str(cfg.thumbnails.cache_dir), cfg.thumbnails.max_edge)
    grid = st.columns(_THUMB_COLS)
    for i, row in enumerate(rows[:_MAX_THUMBS]):
        with grid[i % _THUMB_COLS]:
            thumb = thumbs.get_thumbnail(row["abs_path"], row["mtime"])
            if thumb is not None:
                st.image(str(thumb), width="stretch")
            extra = f"\n\n{captions[row['id']]}" if captions and row["id"] in captions else ""
            st.caption(f"{row['rel_path']}\n\n{human_bytes(row['size_bytes'])}{extra}")
    if len(rows) > _MAX_THUMBS:
        st.caption(f"…and {len(rows) - _MAX_THUMBS} more in this group.")


def _exact_section(repo: Repository, cfg: AppConfig) -> list[dict]:
    st.subheader("Exact duplicates")
    groups = repo.exact_duplicate_groups()
    if not groups:
        st.success("No exact duplicates found. 🎉")
        return []

    wasted = sum(g[0]["size_bytes"] * (len(g) - 1) for g in groups)
    st.caption(
        f"**{len(groups)}** group(s), "
        f"**{sum(len(g) for g in groups)}** files · reclaimable: **{human_bytes(wasted)}** "
        f"(keeping one copy per group)"
    )

    export: list[dict] = []
    for gi, g in enumerate(groups, 1):
        with st.expander(
            f"Group {gi} — {len(g)} identical files · {human_bytes(g[0]['size_bytes'])} each"
        ):
            st.caption("First file is a suggested keep; the rest are redundant copies.")
            _thumb_grid(g, cfg)
        for rank, row in enumerate(g):
            export.append(
                {
                    "kind": "exact",
                    "group": gi,
                    "role": "keep" if rank == 0 else "duplicate",
                    "rel_path": row["rel_path"],
                    "size_bytes": row["size_bytes"],
                    "sha256": row["sha256"],
                }
            )
    return export


def _near_section(repo: Repository, cfg: AppConfig) -> list[dict]:
    st.subheader("Near duplicates")
    st.caption(
        "Visually similar images (small perceptual-hash distance), not necessarily identical bytes."
    )
    max_dist = st.slider(
        "Max pHash distance (lower = stricter)",
        min_value=0,
        max_value=16,
        value=6,
        help="Hamming distance between 64-bit perceptual hashes. 0 ≈ visually identical; 10+ gets loose.",
    )
    rows = repo.hashed_images()
    groups = group_near(rows, max_dist)
    if not groups:
        st.info("No near-duplicate groups at this threshold.")
        return []

    st.caption(f"**{len(groups)}** group(s), **{sum(len(g) for g in groups)}** images.")
    export: list[dict] = []
    for gi, g in enumerate(groups, 1):
        ref = g[0]["phash"]
        caps = {r["id"]: f"d={hamming(ref, r['phash'])}" for r in g}
        with st.expander(f"Group {gi} — {len(g)} images"):
            st.caption("`d` = pHash distance from the first image in the group.")
            _thumb_grid(g, cfg, captions=caps)
        for row in g:
            export.append(
                {
                    "kind": "near",
                    "group": gi,
                    "role": "reference" if row["id"] == g[0]["id"] else "near",
                    "rel_path": row["rel_path"],
                    "size_bytes": row["size_bytes"],
                    "phash": row["phash"],
                    "distance": hamming(ref, row["phash"]),
                }
            )
    return export


def render(repo: Repository, cfg: AppConfig) -> None:
    st.header("Duplicate Detection")

    if repo.count_images() == 0:
        st.info("No dataset indexed yet. Scan a directory from the sidebar first.")
        return

    pending = repo.count_without_hashes()
    done = repo.count_with_hashes()

    top = st.columns([2, 2, 3])
    top[0].metric("Hashed", f"{done:,}")
    top[1].metric("Pending", f"{pending:,}")
    with top[2]:
        st.write("")
        if st.button(
            f"Hash {pending} image(s)" if pending else "Rehash all",
            type="primary",
            disabled=pending == 0 and done == 0,
            width="stretch",
        ):
            if pending == 0 and done > 0:
                repo.clear_hashes()
            _run_compute(repo)

    if done == 0:
        st.info("No hashes computed yet. Click the button above to scan for duplicates.")
        return

    st.divider()
    exact_rows = _exact_section(repo, cfg)
    st.divider()
    near_rows = _near_section(repo, cfg)

    # Combined CSV export of everything flagged.
    all_rows = exact_rows + near_rows
    if all_rows:
        st.divider()
        st.download_button(
            "Duplicate report (CSV)",
            pd.DataFrame(all_rows).to_csv(index=False).encode("utf-8"),
            file_name="duplicates.csv",
            mime="text/csv",
        )
