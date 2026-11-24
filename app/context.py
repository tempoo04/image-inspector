"""Streamlit-scoped resource helpers.

Keeps expensive singletons (config, thumbnail service) alive across reruns via
``st.cache_resource``. The SQLite repository is intentionally *not* cached: a
fresh short-lived connection is opened per run (see ``open_repo``) to avoid
cross-thread connection reuse, which SQLite forbids.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from core.config import AppConfig, ThumbnailSettings
from database.repository import Repository
from services.thumbnail_service import ThumbnailService


@st.cache_resource
def get_config() -> AppConfig:
    cfg = AppConfig()
    cfg.ensure_dirs()
    return cfg


@st.cache_resource
def get_thumbnail_service(cache_dir: str, max_edge: int) -> ThumbnailService:
    return ThumbnailService(ThumbnailSettings(max_edge=max_edge, cache_dir=Path(cache_dir)))


def open_repo(cfg: AppConfig) -> Repository:
    """Open a fresh repository connection for the current run."""
    return Repository(cfg.db_path)
