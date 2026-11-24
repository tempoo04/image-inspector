"""Generic 'planned module' page.

Renders an honest not-yet-implemented notice linking to the tracking issue,
rather than fake controls. Used for modules C–H until each ships.
"""

from __future__ import annotations

import streamlit as st

_REPO_ISSUES = "https://github.com/tempoo04/image-inspector/issues"


def render(title: str, issue: int, summary: str, points: list[str]) -> None:
    st.header(title)
    st.info(
        f"Planned module — not yet implemented. Tracked in "
        f"[issue #{issue}]({_REPO_ISSUES}/{issue})."
    )
    st.write(summary)
    if points:
        st.markdown("**Planned capabilities:**")
        for p in points:
            st.markdown(f"- {p}")
