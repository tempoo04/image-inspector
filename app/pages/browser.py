"""Dataset Browser page (Module B).

Thumbnail grid with filtering, sorting, pagination, multi-selection, and a
detail view with zoom. All data access goes through the repository's paginated
``list_images``; only one page of thumbnails is ever decoded at a time.
"""

from __future__ import annotations

import math

import streamlit as st

from app.context import get_thumbnail_service
from app.ui_utils import class_display, human_bytes, human_time
from core.config import AppConfig
from database.repository import Repository

_GRID_COLS = 5
_SORT_LABELS = {
    "name": "Name",
    "date": "Date modified",
    "size": "File size",
    "resolution": "Resolution",
}


def _int_range(label: str, lo: int, hi: int, *, step: int = 1, fmt=str):
    """Render a range slider, or nothing when the data has a single value.

    Returns ``(low, high)`` or ``None`` (no constraint / not applicable).
    """
    if lo is None or hi is None or lo >= hi:
        return None
    val = st.slider(
        label, min_value=int(lo), max_value=int(hi), value=(int(lo), int(hi)), step=step
    )
    if val == (int(lo), int(hi)):
        return None  # full range selected -> no filter
    return val


def _filter_controls(repo: Repository) -> tuple[dict, str, bool]:
    """Render filter + sort widgets and return (filters, sort_key, descending)."""
    filters: dict = {}

    with st.expander("Filters & sorting", expanded=True):
        row1 = st.columns([2, 2, 3])

        # Class
        classes = [c for c, _ in repo.list_classes()]
        has_flat = any(c is None for c in classes)
        named = [c for c in classes if c]
        options = ["(all)"] + named + (["(no class)"] if has_flat else [])
        chosen = row1[0].selectbox("Class", options, index=0)
        if chosen == "(no class)":
            filters["class_label"] = "__flat__"
        elif chosen != "(all)":
            filters["class_label"] = chosen

        # Format
        exts = repo.distinct_exts()
        picked = row1[1].multiselect("Format", exts)
        if picked:
            filters["exts"] = picked

        # Filename substring
        name = row1[2].text_input("Filename contains")
        if name.strip():
            filters["filename"] = name.strip()

        # Numeric ranges
        ranges = repo.value_ranges()
        if ranges:
            rc = st.columns(4)
            with rc[0]:
                w = _int_range("Width (px)", *ranges["width"])
            with rc[1]:
                h = _int_range("Height (px)", *ranges["height"])
            with rc[2]:
                kb = _int_range(
                    "Size (KB)", ranges["size"][0] // 1024, max(ranges["size"][1] // 1024, 1)
                )
            with rc[3]:
                amin, amax = ranges["aspect"]
                a = None
                if amin is not None and amax is not None and amin < amax:
                    av = st.slider(
                        "Aspect ratio",
                        float(round(amin, 2)),
                        float(round(amax, 2)),
                        (float(round(amin, 2)), float(round(amax, 2))),
                    )
                    if av != (float(round(amin, 2)), float(round(amax, 2))):
                        a = av
            if w:
                filters["min_width"], filters["max_width"] = w
            if h:
                filters["min_height"], filters["max_height"] = h
            if kb:
                filters["min_size"], filters["max_size"] = kb[0] * 1024, kb[1] * 1024
            if a:
                filters["min_aspect"], filters["max_aspect"] = a

        # Sort + corrupt toggle
        sc = st.columns([2, 1, 2])
        sort_key = sc[0].selectbox(
            "Sort by", list(_SORT_LABELS), format_func=lambda k: _SORT_LABELS[k]
        )
        descending = sc[1].checkbox("Desc")
        show_corrupt = sc[2].checkbox("Show corrupt images", value=False)
        filters["include_corrupt"] = show_corrupt

    return filters, sort_key, descending


def _render_detail(repo: Repository, cfg: AppConfig, image_id: int) -> None:
    row = repo.get_image(image_id)
    if row is None:
        st.session_state.detail_id = None
        st.rerun()
        return

    top = st.columns([1, 5])
    if top[0].button("← Back"):
        st.session_state.detail_id = None
        st.rerun()
    top[1].subheader(row["filename"])

    img_col, meta_col = st.columns([3, 2])
    with img_col:
        if row["corrupt"]:
            st.error(f"Image is unreadable: {row['error'] or 'corrupt'}")
        else:
            zoom = st.slider("Zoom (display width, px)", 150, 1400, 600, step=50)
            try:
                st.image(row["abs_path"], width=zoom)
            except Exception as exc:  # noqa: BLE001 - display any decode failure
                st.error(f"Could not display image: {exc}")
    with meta_col:
        st.markdown("**Metadata**")
        st.table(
            {
                "Class": class_display(row["class_label"]),
                "Path": row["rel_path"],
                "Format": row["ext"],
                "Dimensions": f"{row['width']}×{row['height']}" if row["width"] else "—",
                "Mode": row["mode"] or "—",
                "Aspect ratio": f"{row['aspect_ratio']:.3f}" if row["aspect_ratio"] else "—",
                "File size": human_bytes(row["size_bytes"]),
                "Modified": human_time(row["mtime"]),
                "Corrupt": "yes" if row["corrupt"] else "no",
            }
        )


def render(repo: Repository, cfg: AppConfig) -> None:
    st.header("Dataset Browser")

    if repo.count_images() == 0:
        st.info("No dataset indexed yet. Scan a directory from the sidebar first.")
        return

    # Detail view takes over the page when an image is open.
    if st.session_state.get("detail_id") is not None:
        _render_detail(repo, cfg, st.session_state.detail_id)
        return

    filters, sort_key, descending = _filter_controls(repo)

    # Reset to first page whenever the query changes.
    sig = (tuple(sorted(filters.items())), sort_key, descending)
    if st.session_state.get("browser_sig") != sig:
        st.session_state.browser_sig = sig
        st.session_state.browser_page = 0

    page_size = cfg.browser_page_size
    page = st.session_state.get("browser_page", 0)
    rows, total = repo.list_images(
        filters, sort=sort_key, descending=descending, limit=page_size, offset=page * page_size
    )
    pages = max(1, math.ceil(total / page_size))

    selected: set = st.session_state.setdefault("selected_ids", set())

    # Header bar: counts + selection + pagination.
    bar = st.columns([3, 2, 1, 2, 1])
    bar[0].caption(f"{total:,} image(s) match — page {page + 1} of {pages}")
    bar[1].caption(f"**{len(selected)}** selected")
    if bar[2].button("Clear", disabled=not selected):
        selected.clear()
        st.rerun()
    if bar[3].button("← Prev", disabled=page <= 0):
        st.session_state.browser_page = page - 1
        st.rerun()
    if bar[4].button("Next →", disabled=page >= pages - 1):
        st.session_state.browser_page = page + 1
        st.rerun()

    if not rows:
        st.warning("No images match the current filters.")
        return

    thumbs = get_thumbnail_service(str(cfg.thumbnails.cache_dir), cfg.thumbnails.max_edge)

    cols = st.columns(_GRID_COLS)
    for i, row in enumerate(rows):
        with cols[i % _GRID_COLS]:
            iid = row["id"]
            if row["corrupt"]:
                st.markdown(f"⚠️ **{row['filename']}**")
                st.caption("corrupt / unreadable")
            else:
                thumb = thumbs.get_thumbnail(row["abs_path"], row["mtime"])
                if thumb is not None:
                    st.image(str(thumb), width="stretch")
                else:
                    st.caption("⚠️ thumbnail failed")
                st.caption(f"{row['filename']}  ·  {row['width']}×{row['height']}")

            cc = st.columns([1, 1])
            checked = cc[0].checkbox("Select", key=f"sel_{iid}", value=iid in selected)
            if checked:
                selected.add(iid)
            else:
                selected.discard(iid)
            if cc[1].button("Details", key=f"det_{iid}"):
                st.session_state.detail_id = iid
                st.rerun()
