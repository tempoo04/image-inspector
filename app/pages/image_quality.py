"""Image Quality page (Module D): batch metric computation, flags, review."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from app.context import get_thumbnail_service
from app.ui_utils import human_bytes
from core.config import AppConfig
from core.image_metrics import compute_metrics_for_dataset
from database.repository import Repository

_FLAG_LABELS = {
    "is_black": "Black",
    "is_white": "White",
    "is_dark": "Too dark",
    "is_bright": "Too bright",
    "is_blurry": "Blurry",
    "is_small": "Too small",
}
_LAYOUT = dict(height=280, margin=dict(l=0, r=0, t=10, b=0))


def _run_compute(repo: Repository, cfg: AppConfig) -> None:
    progress = st.progress(0.0, text="Computing quality metrics…")

    def cb(done: int, total: int) -> None:
        if total:
            progress.progress(min(done / total, 1.0), text=f"Computing… {done}/{total}")

    computed = compute_metrics_for_dataset(repo, cfg.thresholds, progress_cb=cb)
    progress.empty()
    st.success(f"Computed metrics for {computed} image(s).")
    st.rerun()


def render(repo: Repository, cfg: AppConfig) -> None:
    st.header("Image Quality")

    if repo.count_images() == 0:
        st.info("No dataset indexed yet. Scan a directory from the sidebar first.")
        return

    pending = repo.count_without_metrics()
    done = repo.count_with_metrics()

    top = st.columns([2, 2, 3])
    top[0].metric("With metrics", f"{done:,}")
    top[1].metric("Pending", f"{pending:,}")
    with top[2]:
        st.write("")
        if st.button(
            f"Compute metrics for {pending} image(s)" if pending else "Recompute all",
            type="primary",
            disabled=pending == 0 and done == 0,
            width="stretch",
        ):
            if pending == 0 and done > 0:
                repo.clear_metrics()  # force recompute over the whole dataset
            _run_compute(repo, cfg)

    with st.expander("Active thresholds"):
        t = cfg.thresholds
        st.write(
            f"- Dark ≤ {t.dark_mean_max} · Bright ≥ {t.bright_mean_min} (mean brightness)\n"
            f"- Black ≤ {t.black_mean_max} · White ≥ {t.white_mean_min}\n"
            f"- Blurry < {t.blur_laplacian_max} (variance of Laplacian)\n"
            f"- Too small < {t.min_dimension_px}px (shorter edge)"
        )

    if done == 0:
        st.info("No metrics computed yet. Click the button above to analyze the dataset.")
        return

    # Flag summary.
    st.divider()
    st.subheader("Quality flags")
    counts = repo.metric_flag_counts()
    cols = st.columns(len(_FLAG_LABELS))
    for col, (flag, label) in zip(cols, _FLAG_LABELS.items()):
        col.metric(label, counts[flag])

    # Metric distributions.
    rows = repo.metrics_joined()
    mdf = pd.DataFrame([dict(r) for r in rows])
    st.subheader("Metric distributions")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.plotly_chart(
            px.histogram(mdf, x="brightness", nbins=30).update_layout(**_LAYOUT), width="stretch"
        )
        st.caption("Brightness")
    with d2:
        st.plotly_chart(
            px.histogram(mdf, x="sharpness", nbins=30).update_layout(**_LAYOUT), width="stretch"
        )
        st.caption("Sharpness (Laplacian var)")
    with d3:
        st.plotly_chart(
            px.histogram(mdf, x="entropy", nbins=30).update_layout(**_LAYOUT), width="stretch"
        )
        st.caption("Entropy")

    # Flagged-image review.
    st.divider()
    st.subheader("Review flagged images")
    flagged = [f for f, lbl in _FLAG_LABELS.items() if counts[f] > 0]
    if not flagged:
        st.success("No images were flagged. 🎉")
        return
    choice = st.selectbox("Flag", flagged, format_func=lambda f: f"{_FLAG_LABELS[f]} ({counts[f]})")
    images = repo.flagged_images(choice)
    thumbs = get_thumbnail_service(str(cfg.thumbnails.cache_dir), cfg.thumbnails.max_edge)
    grid = st.columns(5)
    for i, img in enumerate(images[:40]):
        with grid[i % 5]:
            thumb = thumbs.get_thumbnail(img["abs_path"], img["mtime"])
            if thumb is not None:
                st.image(str(thumb), width="stretch")
            m = repo.get_metrics(img["id"])
            st.caption(
                f"{img['filename']}\n\n"
                f"b={m['brightness']:.0f} · s={m['sharpness']:.0f} · {human_bytes(img['size_bytes'])}"
            )
    if len(images) > 40:
        st.caption(f"Showing first 40 of {len(images)} flagged images.")
