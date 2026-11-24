# Vision Dataset Explorer

A local-first, interactive application for exploring, analyzing, comparing, and
cleaning large image datasets. Built for computer-vision researchers, ML
engineers, and data scientists. Everything runs on your machine — no cloud, no
uploads.

> **Status:** Milestone 1 complete (scanner + SQLite indexing + tests). UI and
> later modules are in progress. See [Roadmap](#roadmap).

## Features (planned)

| Module | Capability | Status |
|--------|------------|--------|
| A | Dataset scanner & SQLite indexer (class-based + flat, incremental) | ✅ M1 |
| B | Thumbnail browser: grid, pagination, filter, sort, detail view | ⏳ M2 |
| C | Statistics dashboard with interactive Plotly charts + CSV export | ⏳ M3 |
| D | Image quality metrics (brightness, sharpness, entropy, flags) | ⏳ M3 |
| E | Side-by-side comparison, histograms, pixel diff | ⏳ M4 |
| F | Duplicate detection (sha256 + perceptual hash + embeddings) | ⏳ M4 |
| G | Embedding visualization (ResNet/DINOv2, PCA/UMAP, similarity) | ⏳ M5 |
| H | Dataset cleaning & export (filtered copies, CSV, PDF reports) | ⏳ M6 |

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
python -m examples.make_synthetic --out data/sample --per-class 20
python -m app.launcher --scan data/sample        # (CLI index; UI arrives in M2)
```

## Running tests

```bash
pip install pytest
pytest
```

## Roadmap

The project is built in incremental, independently testable stages:

1. **Foundation** — project structure, dataset scanner, SQLite indexing, tests. ✅
2. **Browser** — Streamlit UI, thumbnail cache, pagination, filtering, sorting.
3. **Statistics & quality** — image metrics, interactive charts, batch processing.
4. **Comparison & duplicates** — side-by-side view, exact + perceptual hashing.
5. **Embeddings** — optional deep features, PCA/UMAP, similarity search.
6. **Export** — filtered dataset copies, CSV/PDF reports, cross-platform checks.

Contributions are welcome at any stage — see the module table above for scope.

## License

MIT
