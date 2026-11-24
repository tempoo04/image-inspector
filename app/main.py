"""Vision Dataset Explorer — Streamlit entry point.

Run with::

    streamlit run app/main.py

Provides the sidebar navigation, dataset scan controls, and dispatches to each
page module. Only Overview and Dataset Browser are implemented (Milestone 2);
the remaining pages show a tracked "planned" notice.
"""

from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run app/main.py` puts this file's own directory (app/) on sys.path,
# not the project root, so the `app`/`core`/`database` packages are not
# importable by default. Add the repository root (parent of app/) up front.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st  # noqa: E402  (import after sys.path bootstrap)

from app.context import get_config, open_repo  # noqa: E402
from app.pages import browser, overview, placeholder  # noqa: E402
from app.ui_utils import human_bytes  # noqa: E402
from core.scanner import scan_dataset  # noqa: E402

PAGE_ORDER = [
    "Overview",
    "Dataset Browser",
    "Statistics",
    "Image Quality",
    "Image Comparison",
    "Duplicate Detection",
    "Embedding Visualization",
    "Export and Cleaning",
]

# title -> (issue number, one-line summary, planned bullet points)
_PLACEHOLDERS = {
    "Statistics": (3, "Dataset-wide statistics and distributions.",
                   ["Resolution / aspect / format / file-size distributions",
                    "Min / max / mean / median dimensions",
                    "Class-imbalance indicators", "CSV export"]),
    "Image Quality": (4, "Per-image quality metrics with configurable thresholds.",
                      ["Brightness, contrast, sharpness, entropy",
                       "RGB channel stats + histograms",
                       "Flags: black/white, dark/bright, blurry, small"]),
    "Image Comparison": (5, "Side-by-side comparison of two or more images.",
                         ["Synchronized zoom", "Overlaid histograms",
                          "Absolute pixel-difference images"]),
    "Duplicate Detection": (6, "Find exact and near-duplicate images.",
                            ["sha256 exact match", "Perceptual hash (pHash)",
                             "Grouped manual review — never auto-delete"]),
    "Embedding Visualization": (7, "Deep-feature embeddings and 2D projection.",
                                ["ResNet18 / DINOv2 embeddings",
                                 "PCA / UMAP / t-SNE", "Similarity search"]),
    "Export and Cleaning": (8, "Build cleaned datasets and reports.",
                            ["Filtered dataset copies", "CSV metadata export",
                             "PDF analysis report", "Source kept read-only"]),
}


def _sidebar(cfg) -> None:
    st.sidebar.title("🔍 Vision Dataset Explorer")

    if "page" not in st.session_state:
        st.session_state.page = "Overview"
    st.sidebar.radio("Navigate", PAGE_ORDER, key="page")

    st.sidebar.divider()
    st.sidebar.subheader("Dataset")
    st.sidebar.text_input("Directory", key="dataset_root", value="data/sample")
    prune = st.sidebar.checkbox("Prune deleted files", value=False)

    if st.sidebar.button("Scan / rescan", type="primary", width='stretch'):
        _run_scan(cfg, st.session_state.dataset_root, prune)

    # Catalog status line.
    with open_repo(cfg) as repo:
        n = repo.count_images()
        size = repo.total_size_bytes()
    st.sidebar.caption(f"Indexed: **{n:,}** images · {human_bytes(size)}")
    st.sidebar.caption(f"Catalog: `{cfg.db_path}`")


def _run_scan(cfg, root: str, prune: bool) -> None:
    progress = st.sidebar.progress(0.0, text="Scanning…")

    def cb(done: int, total: int) -> None:
        if total:
            progress.progress(min(done / total, 1.0), text=f"Scanning… {done}/{total}")

    try:
        with open_repo(cfg) as repo:
            res = scan_dataset(root, repo, batch_size=cfg.scan_batch_size,
                               prune_missing=prune, progress_cb=cb)
    except NotADirectoryError as exc:
        progress.empty()
        st.sidebar.error(str(exc))
        return
    progress.empty()
    st.sidebar.success(
        f"{res.layout} · indexed {res.written} · skipped {res.skipped_unchanged} "
        f"· corrupt {res.corrupt}" + (f" · deleted {res.deleted}" if prune else "")
    )
    # New data: drop stale pagination/selection state and refresh.
    for k in ("browser_page", "browser_sig", "selected_ids", "detail_id"):
        st.session_state.pop(k, None)
    st.rerun()


def _dispatch(page: str, repo, cfg) -> None:
    if page == "Overview":
        overview.render(repo, cfg)
    elif page == "Dataset Browser":
        browser.render(repo, cfg)
    else:
        issue, summary, points = _PLACEHOLDERS[page]
        placeholder.render(page, issue, summary, points)


def main() -> None:
    st.set_page_config(page_title="Vision Dataset Explorer", page_icon="🔍", layout="wide")
    cfg = get_config()
    _sidebar(cfg)
    with open_repo(cfg) as repo:
        _dispatch(st.session_state.page, repo, cfg)


main()
