# Vision Dataset Explorer

[![CI](https://github.com/tempoo04/image-inspector/actions/workflows/ci.yml/badge.svg)](https://github.com/tempoo04/image-inspector/actions/workflows/ci.yml)
[![CodeQL](https://github.com/tempoo04/image-inspector/actions/workflows/codeql.yml/badge.svg)](https://github.com/tempoo04/image-inspector/actions/workflows/codeql.yml)
[![codecov](https://codecov.io/gh/tempoo04/image-inspector/branch/main/graph/badge.svg)](https://codecov.io/gh/tempoo04/image-inspector)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://image-inspector.streamlit.app/)

A local-first, interactive application for exploring, analyzing, comparing, and
cleaning large image datasets. Built for computer-vision researchers, ML
engineers, and data scientists. Everything runs on your machine — no cloud, no
uploads.

**▶️ Live demo: https://image-inspector.streamlit.app/** — open it and click
*Generate demo dataset* in the sidebar to explore a sample.

> **Status:** Runnable. Scanner + SQLite indexing, the interactive browser, a
> statistics dashboard, and image-quality analysis are done. Remaining analysis
> modules are tracked in the
> [issue tracker](https://github.com/tempoo04/image-inspector/issues).

## Architecture

Strict layering — `core/` and `database/` never import Streamlit, so all logic
is unit-testable in isolation.

```
core/        pure logic: scanner, metadata, metrics, duplicates, embeddings
database/    SQLite schema + repository (all SQL lives here)
services/    thumbnail cache, export (CSV/PDF/filtered copies)
app/         Streamlit UI (sidebar nav + one module per page)
examples/    synthetic dataset generator
tests/       pytest suite
```

### Design invariants

- **Never loads a full dataset into RAM.** Metadata is read from image headers;
  the browser pages through SQLite; thumbnails are generated lazily and cached.
- **Originals are read-only.** Writes only ever go to a user-selected output
  directory, after explicit confirmation.
- **Optional dependencies are lazy-imported.** The app starts without PyTorch,
  FAISS, UMAP, or ReportLab; those features degrade gracefully when missing.
- **Incremental rescans** skip files whose size and modification time are
  unchanged.

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt            # core app
pip install -r requirements-optional.txt   # embeddings, FAISS, UMAP, PDF
```

Requires Python 3.11+.

## Quick start

Generate a synthetic sample dataset and index it:

```bash
# 1. Generate a sample dataset (or point the app at your own)
python -m examples.make_synthetic --out data/sample --per-class 20

# 2a. Launch the interactive app, then enter a directory and click "Scan"
streamlit run app/main.py

# 2b. …or index from the command line without the UI
python -m app.launcher --scan data/sample
```

## Run with Docker

A container image is published to GHCR on each release:

```bash
docker run --rm -p 8501:8501 \
  -v /path/to/your/images:/data \
  ghcr.io/tempoo04/image-inspector:latest
```

Then open <http://localhost:8501> and point the app at `/data`.

## Running tests

```bash
pip install pytest
pytest
```

## Roadmap

The project is built in incremental, independently testable stages:

1. **Foundation** — project structure, dataset scanner, SQLite indexing, tests. ✅
2. **Browser** — Streamlit UI, thumbnail cache, pagination, filtering, sorting. ✅
3. **Statistics & quality** — image metrics, interactive charts, batch processing. ✅
4. **Comparison & duplicates** — side-by-side view, exact + perceptual hashing.
5. **Embeddings** — optional deep features, PCA/UMAP, similarity search.
6. **Export** — filtered dataset copies, CSV/PDF reports, cross-platform checks.

Contributions are welcome at any stage — see the
[open issues](https://github.com/tempoo04/image-inspector/issues) for scope.

## License

MIT
