# Contributing to Vision Dataset Explorer

Thanks for your interest in improving the project. This guide covers the local
setup and the conventions CI enforces.

## Development setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -e ".[dev]"      # core app + test/lint tooling
pre-commit install            # run ruff + hygiene hooks on every commit
```

Python 3.11+ is required.

## Project layout

- `core/`, `database/` — pure logic; **must not import Streamlit** (keeps them
  unit-testable in isolation).
- `services/` — thumbnail cache, export (no UI dependency).
- `app/` — Streamlit UI only.
- `tests/` — pytest suite. `examples/` — synthetic dataset generator.

## Before opening a pull request

```bash
ruff check .            # lint
ruff format --check .   # formatting
pytest                  # tests
```

All three must pass; CI runs them on every PR across Ubuntu and Windows for
Python 3.11 and 3.12.

## Guidelines

- Keep PRs focused on a single change; reference the issue (`Closes #N`).
- New behavior needs tests. Don't skip or silence failing tests.
- Preserve the design invariants: never load a full dataset into RAM, keep
  original files read-only, and lazy-import optional dependencies (torch, faiss,
  umap, reportlab) so the app still starts without them.
- Match existing style; `ruff format` is the source of truth.

## Reporting bugs / requesting features

Use the issue templates. Planned modules are tracked as issues (A–H).
