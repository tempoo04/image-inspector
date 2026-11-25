"""Statistics page (Module C): distributions, dimension stats, CSV export."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from app.ui_utils import class_display, human_bytes
from core.config import AppConfig
from database.repository import Repository

_CHART_H = 300
_LAYOUT = dict(height=_CHART_H, margin=dict(l=0, r=0, t=10, b=0))


def _images_frame(repo: Repository) -> pd.DataFrame:
    df = pd.DataFrame([dict(r) for r in repo.iter_images()])
    if not df.empty:
        df["class"] = df["class_label"].map(class_display)
        df["megapixels"] = (df["width"] * df["height"]) / 1e6
        df["size_kb"] = df["size_bytes"] / 1024
    return df


def render(repo: Repository, cfg: AppConfig) -> None:
    st.header("Statistics")

    if repo.count_images() == 0:
        st.info("No dataset indexed yet. Scan a directory from the sidebar first.")
        return

    df = _images_frame(repo)
    valid = df[df["corrupt"] == 0] if "corrupt" in df else df

    # Summary row.
    classes = repo.list_classes()
    n_classes = sum(1 for c, _ in classes if c)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Images", f"{len(df):,}")
    c2.metric("Valid", f"{len(valid):,}")
    c3.metric("Classes", n_classes or "—")
    c4.metric("Total size", human_bytes(repo.total_size_bytes()))

    st.divider()

    # Dimension statistics table.
    dims = repo.dimension_stats()
    if dims:
        st.subheader("Dimension statistics")
        tbl = pd.DataFrame(
            {
                "": ["Min", "Max", "Mean", "Median"],
                "Width (px)": [
                    dims["width_min"],
                    dims["width_max"],
                    dims["width_mean"],
                    dims["width_median"],
                ],
                "Height (px)": [
                    dims["height_min"],
                    dims["height_max"],
                    dims["height_mean"],
                    dims["height_median"],
                ],
            }
        )
        st.dataframe(tbl, hide_index=True, width="stretch")

    # Distributions.
    left, right = st.columns(2)
    with left:
        st.subheader("Resolution (megapixels)")
        st.plotly_chart(
            px.histogram(valid, x="megapixels", nbins=30).update_layout(**_LAYOUT),
            width="stretch",
        )
        st.subheader("Aspect ratio")
        st.plotly_chart(
            px.histogram(valid, x="aspect_ratio", nbins=30).update_layout(**_LAYOUT),
            width="stretch",
        )
    with right:
        st.subheader("File size (KB)")
        st.plotly_chart(
            px.histogram(valid, x="size_kb", nbins=30).update_layout(**_LAYOUT),
            width="stretch",
        )
        st.subheader("Format")
        fmt = pd.DataFrame(repo.count_by_ext(), columns=["format", "count"])
        st.plotly_chart(
            px.bar(fmt, x="format", y="count").update_layout(**_LAYOUT), width="stretch"
        )

    # Class distribution + imbalance.
    st.subheader("Images per class")
    cls_df = pd.DataFrame([(class_display(c), n) for c, n in classes], columns=["class", "count"])
    st.plotly_chart(px.bar(cls_df, x="class", y="count").update_layout(**_LAYOUT), width="stretch")
    counts = [n for c, n in classes if c]
    if len(counts) >= 2:
        st.caption(
            f"Class imbalance ratio (largest / smallest): "
            f"**{max(counts) / max(min(counts), 1):.1f}×**  ·  "
            f"largest {max(counts)}, smallest {min(counts)}"
        )

    # CSV export.
    st.divider()
    st.subheader("Export")
    e1, e2 = st.columns(2)
    e1.download_button(
        "Per-class summary (CSV)",
        cls_df.to_csv(index=False).encode("utf-8"),
        file_name="class_summary.csv",
        mime="text/csv",
        width="stretch",
    )
    full_cols = [
        "rel_path",
        "class_label",
        "ext",
        "width",
        "height",
        "aspect_ratio",
        "size_bytes",
        "corrupt",
    ]
    e2.download_button(
        "Full metadata (CSV)",
        df[[c for c in full_cols if c in df]].to_csv(index=False).encode("utf-8"),
        file_name="image_metadata.csv",
        mime="text/csv",
        width="stretch",
    )
