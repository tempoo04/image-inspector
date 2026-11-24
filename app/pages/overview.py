"""Overview page: dataset summary metrics, class distribution, shortcuts."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from app.ui_utils import class_display, human_bytes
from core.config import AppConfig
from database.repository import Repository


def render(repo: Repository, cfg: AppConfig) -> None:
    st.header("Overview")

    total = repo.count_images()
    if total == 0:
        st.info(
            "No dataset indexed yet. Enter a dataset directory in the sidebar "
            "and click **Scan / rescan** to begin."
        )
        return

    corrupt = repo.count_corrupt()
    classes = repo.list_classes()
    n_classes = sum(1 for c, _ in classes if c)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Images", f"{total:,}")
    c2.metric("Classes", n_classes if n_classes else "—")
    c3.metric("Total size", human_bytes(repo.total_size_bytes()))
    c4.metric("Corrupt", corrupt, delta=None if corrupt == 0 else "needs review",
              delta_color="inverse")

    st.divider()
    left, right = st.columns(2)

    with left:
        st.subheader("Images per class")
        df = pd.DataFrame(
            [(class_display(c), n) for c, n in classes], columns=["class", "count"]
        )
        fig = px.bar(df, x="class", y="count")
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
                          xaxis_title=None, yaxis_title=None)
        st.plotly_chart(fig, width='stretch')

        if n_classes >= 2:
            counts = [n for c, n in classes if c]
            imbalance = max(counts) / max(min(counts), 1)
            st.caption(f"Class imbalance ratio (largest / smallest): **{imbalance:.1f}×**")

    with right:
        st.subheader("Format distribution")
        fdf = pd.DataFrame(repo.count_by_ext(), columns=["format", "count"])
        fig2 = px.pie(fdf, names="format", values="count", hole=0.4)
        fig2.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig2, width='stretch')

    st.divider()
    st.subheader("Jump to")
    s1, s2, _ = st.columns(3)
    if s1.button("🖼️  Browse images", width='stretch'):
        st.session_state.page = "Dataset Browser"
        st.rerun()
    if s2.button("📊  Statistics", width='stretch'):
        st.session_state.page = "Statistics"
        st.rerun()
