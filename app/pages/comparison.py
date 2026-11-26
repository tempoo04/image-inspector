"""Image Comparison page (Module E): side-by-side inspection of two images.

Pick any two indexed images and compare them: previews side by side, a metadata
diff, overlaid per-channel RGB histograms, an absolute pixel-difference heatmap
(the smaller image is resized to match), and a perceptual-hash similarity score.
"""

from __future__ import annotations

import cv2
import pandas as pd
import plotly.express as px
import streamlit as st

from app.ui_utils import class_display, human_bytes
from core.config import AppConfig
from core.hashing import hamming, phash
from core.image_metrics import rgb_histograms
from database.repository import Repository

_CHANNELS = (("r", "#d62728"), ("g", "#2ca02c"), ("b", "#1f77b4"))


def _picker(label: str, ids: list[int], labels: dict[int, str], index: int, key: str) -> int:
    # Options are plain ids (picklable); sqlite3.Row objects are not, and
    # Streamlit deep-copies widget options into session_state.
    return st.selectbox(
        label,
        ids,
        index=min(index, len(ids) - 1),
        key=key,
        format_func=lambda i: labels[i],
    )


def _hist_figure(abs_path: str):
    hists = rgb_histograms(abs_path)
    if hists is None:
        return None
    frames = []
    for name, _ in _CHANNELS:
        frames.append(
            pd.DataFrame({"intensity": range(256), "count": hists[name], "channel": name})
        )
    df = pd.concat(frames, ignore_index=True)
    fig = px.line(
        df,
        x="intensity",
        y="count",
        color="channel",
        color_discrete_map={n: c for n, c in _CHANNELS},
    )
    fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
    return fig


def _meta_diff(a, b) -> pd.DataFrame:
    def fmt(row):
        return {
            "Dimensions": f"{row['width']}×{row['height']}" if row["width"] else "—",
            "Megapixels": f"{(row['width'] * row['height']) / 1e6:.2f}" if row["width"] else "—",
            "Aspect ratio": f"{row['aspect_ratio']:.3f}" if row["aspect_ratio"] else "—",
            "File size": human_bytes(row["size_bytes"]),
            "Format": row["ext"],
            "Mode": row["mode"] or "—",
            "Class": class_display(row["class_label"]),
        }

    fa, fb = fmt(a), fmt(b)
    return pd.DataFrame(
        {
            "Property": list(fa.keys()),
            "Image A": list(fa.values()),
            "Image B": list(fb.values()),
            "": ["" if fa[k] == fb[k] else "≠" for k in fa],
        }
    )


def _pixel_diff(path_a: str, path_b: str):
    """Absolute per-pixel difference heatmap + stats; resizes B to A if needed."""
    a = cv2.imread(path_a, cv2.IMREAD_COLOR)
    b = cv2.imread(path_b, cv2.IMREAD_COLOR)
    if a is None or b is None:
        return None, None, False
    resized = a.shape != b.shape
    if resized:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    diff = cv2.absdiff(a, b)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    heat = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    heat_rgb = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    stats = {
        "mean": float(gray.mean()),
        "max": int(gray.max()),
        "pct": float((gray > 15).mean() * 100.0),  # share of visibly different pixels
    }
    return heat_rgb, stats, resized


def render(repo: Repository, cfg: AppConfig) -> None:
    st.header("Image Comparison")

    rows = repo.iter_images()
    if len(rows) < 2:
        st.info("Need at least two indexed images. Scan a directory from the sidebar first.")
        return

    by_id = {r["id"]: r for r in rows}
    ids = list(by_id)
    labels = {i: by_id[i]["rel_path"] for i in ids}

    left, right = st.columns(2)
    with left:
        a = by_id[_picker("Image A", ids, labels, 0, "cmp_a")]
    with right:
        b = by_id[_picker("Image B", ids, labels, 1, "cmp_b")]
    if a["id"] == b["id"]:
        st.warning("Pick two different images to compare.")
        return

    # Previews.
    pa, pb = st.columns(2)
    pa.image(a["abs_path"], caption=a["rel_path"], width="stretch")
    pb.image(b["abs_path"], caption=b["rel_path"], width="stretch")

    # Perceptual similarity.
    ha, hb = phash(a["abs_path"]), phash(b["abs_path"])
    if ha and hb:
        dist = hamming(ha, hb)
        st.metric(
            "Perceptual similarity",
            f"{(1 - dist / 64) * 100:.0f}%",
            help=f"pHash Hamming distance {dist}/64 (0 = visually identical).",
        )

    # Metadata diff.
    st.subheader("Metadata")
    st.dataframe(_meta_diff(a, b), hide_index=True, width="stretch")

    # RGB histograms.
    st.subheader("RGB histograms")
    hl, hr = st.columns(2)
    for col, img in ((hl, a), (hr, b)):
        with col:
            fig = _hist_figure(img["abs_path"])
            if fig is not None:
                st.plotly_chart(fig, width="stretch")
            else:
                st.caption("Could not read image for histogram.")

    # Pixel difference.
    st.subheader("Pixel difference")
    heat, stats, resized = _pixel_diff(a["abs_path"], b["abs_path"])
    if heat is None:
        st.caption("Could not decode one of the images for a pixel diff.")
        return
    if resized:
        st.caption("Images differ in size — B was resized to A before differencing.")
    dc1, dc2 = st.columns([2, 3])
    with dc1:
        st.metric("Mean difference", f"{stats['mean']:.1f}", help="0–255 per pixel; 0 = identical.")
        st.metric("Max difference", stats["max"])
        st.metric("Changed pixels", f"{stats['pct']:.1f}%")
    with dc2:
        st.image(heat, caption="Absolute difference (brighter = larger)", width="stretch")
