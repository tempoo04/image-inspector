"""Embedding Visualization page (Module G): projection + similarity search.

Extracts ResNet-18 features (needs the optional torch/timm backend), projects
them to 2-D (PCA always; UMAP / t-SNE if installed), and offers cosine
nearest-neighbour search. Projection and search are pure NumPy, so an existing
embedding set stays explorable even where the extractor backend is absent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from app.context import get_thumbnail_service
from app.ui_utils import class_display
from core.config import AppConfig
from core.embeddings import (
    EMBED_MODEL,
    EmbeddingBackendUnavailable,
    available_methods,
    backend_available,
    compute_embeddings_for_dataset,
    from_blob,
    nearest,
    project,
)
from database.repository import Repository

_INSTALL_HINT = (
    "Deep-feature extraction needs the optional backend. Install it with "
    "`pip install -r requirements-optional.txt`, then restart the app."
)


def _run_compute(repo: Repository) -> None:
    progress = st.progress(0.0, text="Extracting embeddings…")

    def cb(done: int, total: int) -> None:
        if total:
            progress.progress(min(done / total, 1.0), text=f"Extracting… {done}/{total}")

    try:
        stored = compute_embeddings_for_dataset(repo, progress_cb=cb)
    except EmbeddingBackendUnavailable as exc:
        progress.empty()
        st.error(str(exc))
        return
    progress.empty()
    st.success(f"Stored embeddings for {stored} image(s).")
    st.rerun()


def _scatter(rows: list, coords: np.ndarray) -> None:
    df = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "class": [class_display(r["class_label"]) for r in rows],
            "file": [r["rel_path"] for r in rows],
        }
    )
    fig = px.scatter(df, x="x", y="y", color="class", hover_name="file", opacity=0.8)
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    st.plotly_chart(fig, width="stretch")


def _similarity(repo: Repository, cfg: AppConfig, rows: list, matrix: np.ndarray) -> None:
    st.subheader("Similarity search")
    ids = [r["id"] for r in rows]
    labels = {r["id"]: r["rel_path"] for r in rows}
    pos = {r["id"]: i for i, r in enumerate(rows)}
    chosen = st.selectbox("Query image", ids, format_func=lambda i: labels[i], key="emb_query")

    thumbs = get_thumbnail_service(str(cfg.thumbnails.cache_dir), cfg.thumbnails.max_edge)
    q = rows[pos[chosen]]
    st.caption("Query:")
    qthumb = thumbs.get_thumbnail(q["abs_path"], q["mtime"])
    if qthumb is not None:
        st.image(str(qthumb), width=160)

    neighbours = nearest(matrix, pos[chosen], k=8)
    st.caption("Nearest neighbours (cosine similarity):")
    grid = st.columns(4)
    for i, (row_idx, score) in enumerate(neighbours):
        r = rows[row_idx]
        with grid[i % 4]:
            thumb = thumbs.get_thumbnail(r["abs_path"], r["mtime"])
            if thumb is not None:
                st.image(str(thumb), width="stretch")
            st.caption(f"{r['rel_path']}\n\nsim={score:.3f}")


def render(repo: Repository, cfg: AppConfig) -> None:
    st.header("Embedding Visualization")

    if repo.count_images() == 0:
        st.info("No dataset indexed yet. Scan a directory from the sidebar first.")
        return

    done = repo.count_with_embeddings(EMBED_MODEL)
    pending = repo.count_without_embeddings(EMBED_MODEL)
    has_backend = backend_available()

    top = st.columns([2, 2, 3])
    top[0].metric("Embedded", f"{done:,}")
    top[1].metric("Pending", f"{pending:,}")
    with top[2]:
        st.write("")
        if not has_backend:
            st.info(_INSTALL_HINT)
        elif st.button(
            f"Embed {pending} image(s)" if pending else "Re-embed all",
            type="primary",
            disabled=pending == 0 and done == 0,
            width="stretch",
        ):
            if pending == 0 and done > 0:
                repo.clear_embeddings(EMBED_MODEL)
            _run_compute(repo)

    st.caption(f"Model: **{EMBED_MODEL}** · 512-d features · cosine similarity")

    if done == 0:
        msg = "No embeddings yet."
        st.info(
            msg
            + (" Click the button above to extract them." if has_backend else f" {_INSTALL_HINT}")
        )
        return

    # Build the matrix once for projection + search.
    rows = repo.embedding_rows(EMBED_MODEL)
    matrix = np.vstack([from_blob(r["vector"]) for r in rows])

    st.divider()
    st.subheader("2-D projection")
    methods = available_methods()
    method = st.selectbox(
        "Projection",
        methods,
        help="PCA is fast; UMAP / t-SNE are slower but reveal nonlinear structure.",
    )
    if len(rows) < 3:
        st.caption("Need at least 3 embedded images to project.")
    else:
        with st.spinner(f"Projecting {len(rows)} embeddings with {method}…"):
            coords = project(matrix, method)
        _scatter(rows, coords)

    st.divider()
    _similarity(repo, cfg, rows, matrix)
